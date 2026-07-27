/**
 * NOISEFIELD — Procedural soldier: rig, geometry and animator.
 *
 * ZERO external assets. Everything here is built from primitives at boot and
 * cached at module scope, so spawning the Nth soldier costs only Object3D
 * allocation — never geometry generation.
 *
 * ── Rig model ────────────────────────────────────────────────────────────────
 * A bone-parented rigid hierarchy rather
 * than a `SkinnedMesh`. Reasons:
 *   1. Hard-surface military kit (plate carrier, helmet, boots, rifle) does not
 *      deform; skinning would only add a per-frame CPU/GPU skinning cost and a
 *      large skinIndex/skinWeight payload for zero visual gain.
 *   2. Rigid segments map 1:1 onto the verlet ragdoll in Ragdoll.ts — the same
 *      bone transforms are driven by either the animator or the ragdoll.
 *
 * ── Draw-call budget ─────────────────────────────────────────────────────────
 * Geometry for every bone is merged down to a single BufferGeometry per bone,
 * and colour variation is baked into a `color` vertex attribute so an entire
 * soldier needs only two materials (soft kit + hard kit). ~16 draws/soldier.
 *
 * ── Raycast contract ─────────────────────────────────────────────────────────
 * Visual meshes have their `raycast` method replaced with a no-op so they are
 * invisible to every Raycaster-driven query in the engine (this also shadows
 * `three-mesh-bvh`'s prototype patch, because an own-property always wins over
 * the prototype). Only the dedicated hitbox meshes are hit-testable, so a
 * bullet resolves to exactly one tagged region. Verified against three r185:
 * `Raycaster.intersect()` tests `object.layers` only — it does NOT skip
 * `visible === false` objects, so invisible hitboxes are still raycastable.
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { EntityId, IMaterialLibrary } from '../core/Types';

// ─────────────────────────────────────────────────────────────────────────────
// Skeleton metrics — 1.8 m soldier, roughly 8-head canon (head unit = 0.225 m)
// ─────────────────────────────────────────────────────────────────────────────

/** Total standing height, metres. Matches the player capsule so cover reads honestly. */
export const CHAR_HEIGHT = 1.8;
/** Eye height used for perception rays and aim origin. */
export const CHAR_EYE_HEIGHT = 1.63;

const Y_PELVIS = 0.97;
const D_SPINE = 0.13; // pelvis → spine
const D_CHEST = 0.16; // spine  → chest      (world 1.26)
const D_NECK = 0.245; // chest  → neck       (world 1.505)
const D_HEAD = 0.075; // neck   → head base  (world 1.58)

const X_SHOULDER = 0.185;
const D_SHOULDER_UP = 0.175; // chest → shoulder joint (world 1.435)
const L_UPPERARM = 0.29;
const L_LOWERARM = 0.26;

const X_HIP = 0.1;
const D_HIP_DOWN = -0.04; // pelvis → hip joint (world 0.93)
const L_UPPERLEG = 0.44; // knee at world 0.49
const L_LOWERLEG = 0.42; // ankle at world 0.07
const ANKLE_HEIGHT = 0.07;

/** Exported so IK and the ragdoll agree on segment lengths without re-deriving them. */
export const RIG_METRICS = {
  height: CHAR_HEIGHT,
  eyeHeight: CHAR_EYE_HEIGHT,
  pelvisY: Y_PELVIS,
  upperArm: L_UPPERARM,
  lowerArm: L_LOWERARM,
  upperLeg: L_UPPERLEG,
  lowerLeg: L_LOWERLEG,
  ankleHeight: ANKLE_HEIGHT,
  shoulderX: X_SHOULDER,
  hipX: X_HIP,
} as const;

export type BoneName =
  | 'pelvis' | 'spine' | 'chest' | 'neck' | 'head'
  | 'upperArmL' | 'lowerArmL' | 'handL'
  | 'upperArmR' | 'lowerArmR' | 'handR'
  | 'upperLegL' | 'lowerLegL' | 'footL'
  | 'upperLegR' | 'lowerLegR' | 'footR';

export type HitboxRegion = 'head' | 'chest' | 'limb';

// ─────────────────────────────────────────────────────────────────────────────
// Palette (sRGB hex; three converts to the linear working space on assignment)
// ─────────────────────────────────────────────────────────────────────────────

const C_UNIFORM = 0x8e8259; // desert-tan camouflage base
const C_UNIFORM_DK = 0x6d6244; // shadowed panels / knees / elbows
const C_CARRIER = 0x4d5138; // olive plate carrier
const C_POUCH = 0x5c5f42;
const C_WEBBING = 0x3a3c2c;
const C_BOOT = 0x2e2822;
const C_SKIN = 0xb08560;
const C_GLOVE = 0x33302a;
const C_HELMET = 0x555a44;
const C_NVG = 0x23252a;
const C_GUN = 0x2b2d31;
const C_GUN_HI = 0x4a4d52;

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers — used only at cache-build time, never per frame
// ─────────────────────────────────────────────────────────────────────────────

const _col = new THREE.Color();

/** Bakes a flat colour into a `color` attribute so one material can serve many parts. */
function paint(geo: THREE.BufferGeometry, hex: number): THREE.BufferGeometry {
  _col.setHex(hex, THREE.SRGBColorSpace);
  const count = geo.attributes.position.count;
  const arr = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    arr[i * 3] = _col.r;
    arr[i * 3 + 1] = _col.g;
    arr[i * 3 + 2] = _col.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(arr, 3));
  // UVs are kept: if the material library hands us a textured 'fabric' or
  // 'metal' material, stripping them would collapse every texel to (0,0).
  return geo;
}

interface Placement {
  x?: number; y?: number; z?: number;
  rx?: number; ry?: number; rz?: number;
  sx?: number; sy?: number; sz?: number;
}

function place(geo: THREE.BufferGeometry, p: Placement): THREE.BufferGeometry {
  if (p.sx !== undefined || p.sy !== undefined || p.sz !== undefined) {
    geo.scale(p.sx ?? 1, p.sy ?? 1, p.sz ?? 1);
  }
  if (p.rx) geo.rotateX(p.rx);
  if (p.ry) geo.rotateY(p.ry);
  if (p.rz) geo.rotateZ(p.rz);
  geo.translate(p.x ?? 0, p.y ?? 0, p.z ?? 0);
  return geo;
}

function box(w: number, h: number, d: number, hex: number, p: Placement = {}): THREE.BufferGeometry {
  return paint(place(new THREE.BoxGeometry(w, h, d), p), hex);
}

function sphere(r: number, hex: number, p: Placement = {}, wSeg = 12, hSeg = 8): THREE.BufferGeometry {
  return paint(place(new THREE.SphereGeometry(r, wSeg, hSeg), p), hex);
}

function dome(r: number, hex: number, p: Placement = {}): THREE.BufferGeometry {
  // Upper hemisphere only — the helmet shell.
  return paint(place(new THREE.SphereGeometry(r, 18, 10, 0, Math.PI * 2, 0, Math.PI * 0.58), p), hex);
}

/** Tapered limb segment authored along -Y from the joint origin. */
function limb(rTop: number, rBot: number, len: number, hex: number, p: Placement = {}): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(rTop, rBot, len, 10, 1, false);
  g.translate(0, -len * 0.5, 0);
  return paint(place(g, p), hex);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cached geometry — built once for the whole process
// ─────────────────────────────────────────────────────────────────────────────

/** A merged, ready-to-instance geometry plus which material slot(s) it needs. */
interface PartGeo {
  geo: THREE.BufferGeometry;
  /** true when the merge produced two groups: [soft, hard]. */
  twoSlot: boolean;
}

interface GeoCache {
  parts: Partial<Record<BoneName, PartGeo>>;
  rifle: THREE.BufferGeometry;
  hbHead: THREE.BufferGeometry;
  hbChest: THREE.BufferGeometry;
  hbPelvis: THREE.BufferGeometry;
  hbUpperArm: THREE.BufferGeometry;
  hbLowerArm: THREE.BufferGeometry;
  hbUpperLeg: THREE.BufferGeometry;
  hbLowerLeg: THREE.BufferGeometry;
}

let CACHE: GeoCache | null = null;

/** Merge soft-kit and hard-kit lists into one geometry with up to two groups. */
function mergeParts(soft: THREE.BufferGeometry[], hard: THREE.BufferGeometry[]): PartGeo {
  if (hard.length === 0) {
    const g = mergeGeometries(soft, false);
    return { geo: g, twoSlot: false };
  }
  const softMerged = mergeGeometries(soft, false);
  const hardMerged = mergeGeometries(hard, false);
  const g = mergeGeometries([softMerged, hardMerged], true);
  softMerged.dispose();
  hardMerged.dispose();
  return { geo: g, twoSlot: true };
}

function buildRifleGeometry(): THREE.BufferGeometry {
  // Authored in right-hand space: grip at the origin, muzzle toward +Z.
  const g: THREE.BufferGeometry[] = [
    box(0.045, 0.115, 0.05, C_GUN, { y: -0.02, z: 0.0, rx: 0.12 }), // pistol grip
    box(0.052, 0.075, 0.30, C_GUN, { y: 0.075, z: 0.10 }), // receiver
    box(0.046, 0.06, 0.16, C_GUN_HI, { y: 0.08, z: -0.14 }), // stock
    box(0.03, 0.045, 0.10, C_GUN, { y: 0.075, z: -0.05 }), // buffer tube
    box(0.038, 0.13, 0.05, C_GUN, { y: -0.005, z: 0.135, rx: -0.18 }), // magazine
    box(0.036, 0.036, 0.30, C_GUN, { y: 0.085, z: 0.40 }), // handguard
    paint(place(new THREE.CylinderGeometry(0.011, 0.011, 0.34, 8), { rx: Math.PI * 0.5, y: 0.085, z: 0.46 }), C_GUN_HI), // barrel
    box(0.03, 0.03, 0.055, C_GUN_HI, { y: 0.085, z: 0.63 }), // muzzle brake
    box(0.032, 0.05, 0.09, C_NVG, { y: 0.135, z: 0.14 }), // optic body
    box(0.05, 0.012, 0.05, C_GUN, { y: 0.118, z: 0.14 }), // optic mount
    box(0.02, 0.055, 0.03, C_GUN, { y: 0.05, z: 0.34, rx: 0.25 }), // fore-grip stub
  ];
  const merged = mergeGeometries(g, false);
  for (const x of g) x.dispose();
  return merged;
}

function buildCache(): GeoCache {
  const parts: Partial<Record<BoneName, PartGeo>> = {};

  // ── Head: skull, helmet, NVG mount, comms. Head bone sits at the skull base,
  //    so everything is authored +Y from there.
  parts.head = mergeParts(
    [
      sphere(0.098, C_SKIN, { y: 0.085, sy: 1.1, sz: 1.05 }, 14, 10), // cranium
      box(0.125, 0.075, 0.11, C_SKIN, { y: 0.02, z: 0.03 }), // jaw
      box(0.118, 0.062, 0.03, C_WEBBING, { y: 0.012, z: 0.082 }), // face shemagh
      box(0.20, 0.03, 0.022, C_WEBBING, { y: 0.128, z: 0.0 }), // helmet strap band
    ],
    [
      dome(0.117, C_HELMET, { y: 0.082, sy: 0.98, sz: 1.06 }), // helmet shell
      box(0.185, 0.02, 0.045, C_HELMET, { y: 0.088, z: 0.098, rx: -0.18 }), // brow brim
      sphere(0.046, C_HELMET, { x: 0.107, y: 0.078, sx: 0.5 }, 10, 8), // ear cup L
      sphere(0.046, C_HELMET, { x: -0.107, y: 0.078, sx: 0.5 }, 10, 8), // ear cup R
      box(0.056, 0.05, 0.036, C_NVG, { y: 0.142, z: 0.098 }), // NVG shroud
      box(0.028, 0.062, 0.02, C_NVG, { y: 0.182, z: 0.106, rx: 0.3 }), // NVG arm
      box(0.045, 0.02, 0.05, C_NVG, { x: 0.055, y: 0.16, z: -0.02 }), // counterweight rail
    ],
  );

  // ── Neck.
  parts.neck = mergeParts([limb(0.052, 0.062, 0.09, C_SKIN, { y: 0.02 })], []);

  // ── Chest: torso, plate carrier, pouches, shoulder pads, pack + antenna.
  parts.chest = mergeParts(
    [
      box(0.34, 0.40, 0.215, C_UNIFORM, { y: 0.055 }), // ribcage
      box(0.30, 0.06, 0.20, C_UNIFORM_DK, { y: -0.10 }), // waist join
      box(0.365, 0.30, 0.245, C_CARRIER, { y: 0.055 }), // plate carrier shell
      box(0.155, 0.11, 0.06, C_POUCH, { x: 0.085, y: 0.02, z: 0.145 }), // mag pouch L
      box(0.155, 0.11, 0.06, C_POUCH, { x: -0.085, y: 0.02, z: 0.145 }), // mag pouch R
      box(0.10, 0.085, 0.05, C_POUCH, { x: 0.135, y: 0.13, z: 0.128, rz: -0.2 }), // utility pouch
      box(0.075, 0.10, 0.055, C_WEBBING, { x: -0.14, y: 0.135, z: 0.12 }), // radio
      box(0.30, 0.05, 0.045, C_WEBBING, { y: 0.175, z: 0.115 }), // upper webbing
      box(0.26, 0.20, 0.10, C_CARRIER, { y: 0.05, z: -0.155 }), // assault pack
      box(0.10, 0.07, 0.045, C_POUCH, { y: 0.155, z: -0.19 }), // pack lid
      sphere(0.075, C_UNIFORM_DK, { x: X_SHOULDER, y: D_SHOULDER_UP, sy: 0.8 }, 10, 8), // shoulder L
      sphere(0.075, C_UNIFORM_DK, { x: -X_SHOULDER, y: D_SHOULDER_UP, sy: 0.8 }, 10, 8), // shoulder R
      box(0.11, 0.055, 0.135, C_CARRIER, { x: X_SHOULDER, y: D_SHOULDER_UP + 0.03, rz: 0.22 }), // pauldron L
      box(0.11, 0.055, 0.135, C_CARRIER, { x: -X_SHOULDER, y: D_SHOULDER_UP + 0.03, rz: -0.22 }), // pauldron R
    ],
    [
      paint(place(new THREE.CylinderGeometry(0.006, 0.004, 0.34, 5), { x: -0.11, y: 0.30, z: -0.16, rx: -0.22 }), C_NVG), // antenna
      box(0.05, 0.03, 0.09, C_GUN_HI, { x: 0.14, y: -0.02, z: -0.10 }), // sidearm holster hardware
    ],
  );

  // ── Pelvis: hips, belt, dump pouch, holster.
  parts.pelvis = mergeParts(
    [
      box(0.30, 0.20, 0.20, C_UNIFORM, { y: -0.02 }),
      box(0.315, 0.055, 0.215, C_WEBBING, { y: -0.075 }), // belt
      box(0.11, 0.13, 0.075, C_POUCH, { x: 0.155, y: -0.10, z: 0.0, rz: 0.12 }), // dump pouch
      box(0.09, 0.14, 0.06, C_WEBBING, { x: -0.16, y: -0.12, z: 0.02 }), // holster
      box(0.12, 0.06, 0.10, C_UNIFORM_DK, { y: -0.115, z: -0.04 }), // seat
    ],
    [],
  );

  // ── Arms. Elbow/wrist detail merged into the segment that owns it.
  for (const side of [1, -1] as const) {
    const ua: BoneName = side > 0 ? 'upperArmL' : 'upperArmR';
    const la: BoneName = side > 0 ? 'lowerArmL' : 'lowerArmR';
    parts[ua] = mergeParts(
      [
        limb(0.058, 0.048, L_UPPERARM, C_UNIFORM),
        box(0.105, 0.09, 0.105, C_UNIFORM_DK, { y: -0.235 }), // elbow pad
      ],
      [],
    );
    parts[la] = mergeParts(
      [
        limb(0.048, 0.040, L_LOWERARM, C_UNIFORM),
        box(0.078, 0.045, 0.085, C_WEBBING, { y: -0.215 }), // cuff / watch
        // Hand geometry lives on the forearm: the hand bone stays an empty
        // attach point for the rifle, saving two draw calls per soldier.
        box(0.062, 0.10, 0.085, C_GLOVE, { y: -0.305, rx: 0.15 }),
        box(0.03, 0.055, 0.055, C_GLOVE, { x: side * 0.032, y: -0.345, rz: -side * 0.35 }), // thumb
      ],
      [],
    );
  }

  // ── Legs + boots.
  for (const side of [1, -1] as const) {
    const ul: BoneName = side > 0 ? 'upperLegL' : 'upperLegR';
    const ll: BoneName = side > 0 ? 'lowerLegL' : 'lowerLegR';
    const ft: BoneName = side > 0 ? 'footL' : 'footR';
    parts[ul] = mergeParts(
      [
        limb(0.085, 0.068, L_UPPERLEG, C_UNIFORM),
        box(0.135, 0.115, 0.135, C_UNIFORM_DK, { y: -0.395 }), // knee pad
        box(0.10, 0.10, 0.05, C_POUCH, { x: side * 0.065, y: -0.17, z: 0.055 }), // thigh pouch
      ],
      [],
    );
    parts[ll] = mergeParts(
      [
        limb(0.066, 0.05, L_LOWERLEG, C_UNIFORM),
        box(0.105, 0.09, 0.11, C_BOOT, { y: -0.355 }), // boot cuff
      ],
      [],
    );
    parts[ft] = mergeParts(
      [
        box(0.10, 0.075, 0.235, C_BOOT, { y: -0.03, z: 0.048 }),
        box(0.108, 0.028, 0.25, C_WEBBING, { y: -0.058, z: 0.05 }), // sole
        box(0.095, 0.055, 0.06, C_BOOT, { y: 0.0, z: -0.055 }), // heel
      ],
      [],
    );
  }

  // ── Hitboxes. Head deliberately tight (0.105 m) so headshots are earned.
  //
  // The chest box is sized so its top (chest bone 1.26 + 0.05 + 0.25 = 1.560 m)
  // exactly meets the bottom of the head sphere (head bone 1.58 + 0.085 − 0.105
  // = 1.560 m). Leaving a gap there puts an unhittable hole across the neck,
  // which reads to the player as random shots that do nothing.
  const hbHead = new THREE.SphereGeometry(0.105, 12, 8);
  hbHead.translate(0, 0.085, 0.005);
  const hbChest = new THREE.BoxGeometry(0.42, 0.53, 0.28);
  hbChest.translate(0, 0.065, 0);
  const hbPelvis = new THREE.BoxGeometry(0.34, 0.27, 0.24);
  hbPelvis.translate(0, -0.02, 0);
  const hbUpperArm = new THREE.BoxGeometry(0.13, L_UPPERARM, 0.13);
  hbUpperArm.translate(0, -L_UPPERARM * 0.5, 0);
  const hbLowerArm = new THREE.BoxGeometry(0.115, L_LOWERARM + 0.12, 0.115);
  hbLowerArm.translate(0, -(L_LOWERARM + 0.12) * 0.5, 0);
  const hbUpperLeg = new THREE.BoxGeometry(0.17, L_UPPERLEG, 0.17);
  hbUpperLeg.translate(0, -L_UPPERLEG * 0.5, 0);
  const hbLowerLeg = new THREE.BoxGeometry(0.15, L_LOWERLEG, 0.15);
  hbLowerLeg.translate(0, -L_LOWERLEG * 0.5, 0);

  return {
    parts,
    rifle: buildRifleGeometry(),
    hbHead, hbChest, hbPelvis, hbUpperArm, hbLowerArm, hbUpperLeg, hbLowerLeg,
  };
}

function cache(): GeoCache {
  if (!CACHE) CACHE = buildCache();
  return CACHE;
}

/** Shared, never-rendered material for hitboxes. `Mesh.raycast` requires one to exist. */
const HITBOX_MATERIAL = new THREE.MeshBasicMaterial({ visible: false });

/** Replaces `Mesh.raycast` on visual meshes so only hitboxes are hit-testable. */
function noRaycast(): void {
  /* intentionally empty */
}

// ─────────────────────────────────────────────────────────────────────────────
// CharacterModel
// ─────────────────────────────────────────────────────────────────────────────

export interface CharacterModelOptions {
  /** Optional material library; falls back to standalone MeshStandardMaterials. */
  materials?: IMaterialLibrary;
  castShadow?: boolean;
}

/**
 * One soldier instance: bone hierarchy, merged visual meshes, hitboxes, rifle.
 * Pooled — call `reset()` when recycling rather than rebuilding.
 */
export class CharacterModel {
  /** Scene-graph root. Position = feet, +Z = facing. */
  readonly root = new THREE.Group();
  readonly bones: Record<BoneName, THREE.Object3D>;
  /** Invisible, tagged, raycastable colliders. */
  readonly hitboxes: THREE.Mesh[] = [];
  /** Rifle attached to the right hand. */
  readonly rifle: THREE.Object3D;
  /** World-space fore-grip anchor, used as the left-hand IK goal. */
  readonly rifleForegrip = new THREE.Object3D();
  /** Muzzle anchor for tracer / flash origin. */
  readonly rifleMuzzle = new THREE.Object3D();

  private readonly softMat: THREE.MeshStandardMaterial;
  private readonly hardMat: THREE.MeshStandardMaterial;
  private readonly visualMeshes: THREE.Mesh[] = [];
  private entityId: EntityId = -1;

  constructor(opts: CharacterModelOptions = {}) {
    const c = cache();

    // Materials: one soft (uniform/webbing/skin) and one hard (helmet/weapon).
    // Cloned per instance because death fades opacity independently.
    this.softMat = this.deriveMaterial(opts.materials, 'fabric', 0.86, 0.02);
    this.hardMat = this.deriveMaterial(opts.materials, 'metal', 0.44, 0.62);

    const mk = (name: BoneName, parent: THREE.Object3D, x: number, y: number, z: number): THREE.Object3D => {
      const b = new THREE.Object3D();
      b.name = name;
      b.position.set(x, y, z);
      parent.add(b);
      return b;
    };

    const pelvis = mk('pelvis', this.root, 0, Y_PELVIS, 0);
    const spine = mk('spine', pelvis, 0, D_SPINE, 0);
    const chest = mk('chest', spine, 0, D_CHEST, 0);
    const neck = mk('neck', chest, 0, D_NECK, 0);
    const head = mk('head', neck, 0, D_HEAD, 0);

    const upperArmL = mk('upperArmL', chest, X_SHOULDER, D_SHOULDER_UP, 0);
    const lowerArmL = mk('lowerArmL', upperArmL, 0, -L_UPPERARM, 0);
    const handL = mk('handL', lowerArmL, 0, -L_LOWERARM, 0);
    const upperArmR = mk('upperArmR', chest, -X_SHOULDER, D_SHOULDER_UP, 0);
    const lowerArmR = mk('lowerArmR', upperArmR, 0, -L_UPPERARM, 0);
    const handR = mk('handR', lowerArmR, 0, -L_LOWERARM, 0);

    const upperLegL = mk('upperLegL', pelvis, X_HIP, D_HIP_DOWN, 0);
    const lowerLegL = mk('lowerLegL', upperLegL, 0, -L_UPPERLEG, 0);
    const footL = mk('footL', lowerLegL, 0, -L_LOWERLEG, 0);
    const upperLegR = mk('upperLegR', pelvis, -X_HIP, D_HIP_DOWN, 0);
    const lowerLegR = mk('lowerLegR', upperLegR, 0, -L_UPPERLEG, 0);
    const footR = mk('footR', lowerLegR, 0, -L_LOWERLEG, 0);

    this.bones = {
      pelvis, spine, chest, neck, head,
      upperArmL, lowerArmL, handL,
      upperArmR, lowerArmR, handR,
      upperLegL, lowerLegL, footL,
      upperLegR, lowerLegR, footR,
    };

    // Tag the ROOT, not just the hitboxes. Everything that filters characters
    // out of world queries — level-bounds derivation, the raycast candidate
    // list, `isWorldCollider` — tests the object it was handed or its
    // ancestors, and the object added to the scene is this root.
    this.root.userData.character = true;

    const shadow = opts.castShadow !== false;
    for (const key of Object.keys(c.parts) as BoneName[]) {
      const part = c.parts[key];
      if (!part) continue;
      const mat: THREE.Material | THREE.Material[] = part.twoSlot
        ? [this.softMat, this.hardMat]
        : this.softMat;
      const mesh = new THREE.Mesh(part.geo, mat);
      mesh.castShadow = shadow;
      mesh.receiveShadow = false; // limbs self-shadowing is noisy and costly
      mesh.raycast = noRaycast;
      this.bones[key].add(mesh);
      this.visualMeshes.push(mesh);
    }

    // Rifle in the right hand. Held rather than slung so the aim pose reads;
    // the assault pack + antenna carry the "slung kit" silhouette instead.
    const rifle = new THREE.Mesh(c.rifle, this.hardMat);
    rifle.castShadow = shadow;
    rifle.raycast = noRaycast;
    rifle.position.set(0, -0.055, 0.035);
    handR.add(rifle);
    this.rifle = rifle;
    this.rifleForegrip.position.set(0, 0.075, 0.34);
    rifle.add(this.rifleForegrip);
    this.rifleMuzzle.position.set(0, 0.085, 0.66);
    rifle.add(this.rifleMuzzle);

    this.addHitbox(head, c.hbHead, 'head');
    this.addHitbox(chest, c.hbChest, 'chest');
    this.addHitbox(pelvis, c.hbPelvis, 'chest');
    this.addHitbox(upperArmL, c.hbUpperArm, 'limb');
    this.addHitbox(upperArmR, c.hbUpperArm, 'limb');
    this.addHitbox(lowerArmL, c.hbLowerArm, 'limb');
    this.addHitbox(lowerArmR, c.hbLowerArm, 'limb');
    this.addHitbox(upperLegL, c.hbUpperLeg, 'limb');
    this.addHitbox(upperLegR, c.hbUpperLeg, 'limb');
    this.addHitbox(lowerLegL, c.hbLowerLeg, 'limb');
    this.addHitbox(lowerLegR, c.hbLowerLeg, 'limb');

    // A freshly constructed model is not in play yet: it sits at the origin
    // until `spawn()` places it. Leaving its hitboxes live would put a full set
    // of raycastable 'flesh' colliders with entityId −1 at (0,0,0).
    this.setHitboxesEnabled(false);
  }

  private deriveMaterial(
    lib: IMaterialLibrary | undefined,
    id: string,
    roughness: number,
    metalness: number,
  ): THREE.MeshStandardMaterial {
    if (lib) {
      try {
        const src = lib.get(id);
        if (src instanceof THREE.MeshStandardMaterial) {
          const m = src.clone();
          m.vertexColors = true;
          return m;
        }
      } catch {
        // Library present but does not know this id — fall through to the default.
      }
    }
    return new THREE.MeshStandardMaterial({
      color: 0xffffff, // vertex colours carry the palette
      vertexColors: true,
      roughness,
      metalness,
    });
  }

  private addHitbox(parent: THREE.Object3D, geo: THREE.BufferGeometry, region: HitboxRegion): void {
    const m = new THREE.Mesh(geo, HITBOX_MATERIAL);
    m.visible = false; // never drawn; still raycastable in three r185
    m.userData.hitbox = region;
    m.userData.surface = 'flesh';
    m.userData.entityId = this.entityId;
    m.userData.character = true;
    parent.add(m);
    this.hitboxes.push(m);
  }

  /** Stamp the owning entity onto every collider so raycasts resolve the target. */
  setEntityId(id: EntityId): void {
    this.entityId = id;
    for (const h of this.hitboxes) h.userData.entityId = id;
    this.root.userData.entityId = id;
  }

  /** Toggle hit-testability without touching visibility (used while dying/pooled). */
  setHitboxesEnabled(enabled: boolean): void {
    for (const h of this.hitboxes) h.layers.set(enabled ? 0 : 31);
  }

  setVisible(v: boolean): void {
    this.root.visible = v;
  }

  /** Death fade. Enabling transparency only when needed keeps the opaque pass fast. */
  setOpacity(o: number): void {
    const transparent = o < 0.999;
    for (const m of [this.softMat, this.hardMat]) {
      // `transparent` feeds the OPAQUE shader define but is NOT in three's
      // needsProgramChange ladder, so flipping it without needsUpdate leaves
      // the old program compiled with `#define OPAQUE`, which hard-sets alpha
      // to 1.0 — the corpse stays solid and then pops out of existence.
      if (m.transparent !== transparent) m.needsUpdate = true;
      m.transparent = transparent;
      m.opacity = o;
      m.depthWrite = !transparent;
    }
  }

  /** Restore the bind pose. Called when a pooled instance is respawned. */
  reset(): void {
    for (const key of Object.keys(this.bones) as BoneName[]) {
      const b = this.bones[key];
      b.quaternion.identity();
      b.scale.set(1, 1, 1);
    }
    this.bones.pelvis.position.set(0, Y_PELVIS, 0);
    this.setOpacity(1);
    this.setVisible(true);
    this.setHitboxesEnabled(true);
    this.rifle.visible = true;
  }

  dispose(): void {
    this.softMat.dispose();
    this.hardMat.dispose();
    // Geometry is module-cached and shared; never disposed per instance.
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Procedural animator
// ─────────────────────────────────────────────────────────────────────────────

/** Ground query the animator needs for foot planting. Injected to stay decoupled. */
export type GroundSampler = (x: number, z: number, fromY: number, out: THREE.Vector3) => boolean;

export interface AnimatorInput {
  /** Horizontal speed, m/s. Drives cadence and stride length. */
  speed: number;
  grounded: boolean;
  /** 0 = standing, 1 = fully crouched. */
  crouch: number;
  /** World-space aim goal; null = no aiming (idle / patrol carry). */
  aimAt: THREE.Vector3 | null;
  /** 0..1 blend into the shouldered firing stance. */
  aimWeight: number;
  /** Set true on the frame a shot is fired; adds weapon kick to the arms. */
  firing: boolean;
}

const TAU = Math.PI * 2;

/**
 * Progress 0..1 through a foot's swing half-cycle.
 * Stance runs π/2 → 3π/2, so swing runs 3π/2 → π/2 (wrapping through 0).
 */
function swingPhase(phase: number): number {
  let psi = (phase - Math.PI * 1.5) % TAU;
  if (psi < 0) psi += TAU;
  return Math.min(1, psi / Math.PI);
}
const DEG = Math.PI / 180;

// Scratch — the animator runs for every soldier every frame, so it allocates nothing.
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _q0 = new THREE.Quaternion();
const _q1 = new THREE.Quaternion();
const _m0 = new THREE.Matrix4();
const _DOWN = new THREE.Vector3(0, -1, 0);
const _XAXIS = new THREE.Vector3(1, 0, 0);

/**
 * Two-bone analytic IK.
 *
 * Solves for `upper`/`lower` so the end effector reaches `targetWorld`, with the
 * mid-joint bending in a fixed plane (knees forward, elbows back). Cheap, exact
 * and stable — no iteration, no jitter.
 *
 * @param bendSign +1 bends the mid joint toward -Z (elbows), -1 toward +Z (knees).
 */
function solveTwoBoneIK(
  upper: THREE.Object3D,
  lower: THREE.Object3D,
  a: number,
  b: number,
  targetWorld: THREE.Vector3,
  bendSign: number,
): void {
  const parent = upper.parent;
  if (!parent) return;

  // Express the goal in the upper bone's parent space; the bone origin is the joint.
  _m0.copy(parent.matrixWorld).invert();
  _v0.copy(targetWorld).applyMatrix4(_m0).sub(upper.position);

  let d = _v0.length();
  if (d < 1e-5) return;
  const min = Math.abs(a - b) + 1e-3;
  const max = a + b - 1e-3;
  d = Math.min(max, Math.max(min, d));
  _v0.normalize();

  // Law of cosines: A = hip/shoulder deviation, K = interior knee/elbow angle.
  const cosA = (a * a + d * d - b * b) / (2 * a * d);
  const cosK = (a * a + b * b - d * d) / (2 * a * b);
  const A = Math.acos(Math.min(1, Math.max(-1, cosA)));
  const K = Math.acos(Math.min(1, Math.max(-1, cosK)));

  _q0.setFromUnitVectors(_DOWN, _v0); // aim the whole chain at the goal
  _q1.setFromAxisAngle(_XAXIS, bendSign * A); // then swing the upper out of line
  upper.quaternion.copy(_q0).multiply(_q1);
  lower.quaternion.setFromAxisAngle(_XAXIS, -bendSign * (Math.PI - K));
}

/**
 * Drives a CharacterModel with hand-authored procedural cycles.
 *
 * Locomotion is a phase accumulator scaled by ground speed, so cadence always
 * matches translation and the feet cannot skate. On top of that, a two-bone IK
 * pass clamps each foot to the ground it is standing on, which is what stops
 * limbs floating on slopes, stairs and uneven cover.
 */
export class ProceduralAnimator {
  private readonly m: CharacterModel;
  private phase = 0;
  private idleT = Math.random() * 100; // desync soldiers so a squad does not breathe in unison
  private speedSm = 0;
  private crouchSm = 0;
  private aimSm = 0;
  private flinch = 0;
  private flinchDirX = 0;
  private flinchDirZ = 0;
  private fireKick = 0;
  private readonly aimGoal = new THREE.Vector3();
  private hasAim = false;
  /** Injected ground query for foot planting; null disables the IK pass. */
  groundSampler: GroundSampler | null = null;
  /**
   * Foot-lock state, indexed [left, right]: whether the foot is currently
   * pinned, and the world XZ it was pinned at.
   */
  private readonly plantActive: [boolean, boolean] = [false, false];
  private readonly plantX = new Float64Array(2);
  private readonly plantZ = new Float64Array(2);
  /** Written by `measureGait` each frame; consumed by the pose and the IK. */
  private strideCache = 1;
  private pelvisCache = Y_PELVIS;
  /** Skipped for distant soldiers (LOD 2) — bones simply hold their last pose. */
  footIK = true;

  constructor(model: CharacterModel) {
    this.m = model;
  }

  /** Brief stagger when hit. `dirLocal` is the impact direction in model space. */
  hit(dirX: number, dirZ: number, strength: number): void {
    this.flinch = Math.min(1, this.flinch + strength);
    this.flinchDirX = dirX;
    this.flinchDirZ = dirZ;
  }

  reset(): void {
    this.phase = 0;
    this.speedSm = 0;
    this.crouchSm = 0;
    this.aimSm = 0;
    this.flinch = 0;
    this.fireKick = 0;
    this.plantActive[0] = false;
    this.plantActive[1] = false;
  }

  update(dt: number, input: AnimatorInput): void {
    const b = this.m.bones;

    // Smooth the drivers so state changes never pop the pose.
    this.speedSm = THREE.MathUtils.damp(this.speedSm, input.speed, 10, dt);
    this.crouchSm = THREE.MathUtils.damp(this.crouchSm, input.crouch, 9, dt);
    this.aimSm = THREE.MathUtils.damp(this.aimSm, input.aimWeight, 8, dt);
    this.flinch = Math.max(0, this.flinch - dt * 4.5);
    if (input.firing) this.fireKick = 1;
    this.fireKick = Math.max(0, this.fireKick - dt * 9);
    this.idleT += dt;

    const speed = this.speedSm;
    const moving = speed > 0.12;

    const run = THREE.MathUtils.clamp((speed - 1.6) / 3.4, 0, 1); // walk→run blend
    const gait = THREE.MathUtils.clamp(speed / 1.6, 0, 1); // idle→walk blend
    const crouch = this.crouchSm;

    // Leg cycle parameters. These are authored for *looks*; the stride that
    // matches them is then derived, never guessed. Tuned so the resulting
    // cadence lands where a real gait does — ~2 steps/s at a walk rising to
    // ~4.7 at a sprint — rather than the mincing 4+ steps/s a short swing
    // would force once the stride is honest.
    const hipSwing = (0.55 + run * 0.45) * gait;
    const kneeBase = 0.07 + crouch * 1.05;
    const kneeSwing = (0.35 + run * 0.7) * gait;

    // Stride is measured from the pose the legs will actually adopt, by
    // evaluating the ankle's forward offset at both stance extremes with the
    // same forward kinematics `poseLeg` uses.
    //
    // This is the whole anti-skate mechanism. If stride were an independent
    // constant it would agree with the leg pose only by luck, and every
    // mismatch shows up directly as feet sliding across the ground — the one
    // artefact that instantly reads as "cheap procedural animation". Deriving
    // it means the contact point is stationary in world space for ANY hip
    // swing, knee bend, crouch depth or run blend.
    // The swing may be reduced to stay within the legs' reach; everything
    // downstream must use the value actually adopted.
    const hipSwingUsed = this.measureGait(hipSwing, kneeBase, kneeSwing, crouch, run, gait);
    const stride = Math.max(0.45, this.strideCache);

    if (moving && input.grounded) {
      this.phase += (speed / stride) * TAU * dt;
      if (this.phase > TAU) this.phase -= TAU;
    } else {
      // Ease the phase back to the neutral stance instead of snapping.
      const target = this.phase < Math.PI ? 0 : TAU;
      this.phase = THREE.MathUtils.damp(this.phase, target, 7, dt);
      if (this.phase >= TAU - 1e-3) this.phase = 0;
    }

    const sinP = Math.sin(this.phase);
    const cosP = Math.cos(this.phase);

    // ── Root / pelvis ────────────────────────────────────────────────────────
    const bobAmp = (0.018 + run * 0.030) * gait;
    const swayAmp = (0.012 + run * 0.016) * gait;
    // Hip height is the reach-limited value once moving, easing back to the
    // full standing height at rest.
    b.pelvis.position.set(
      sinP * swayAmp,
      this.pelvisCache - (1 - cosP * cosP) * bobAmp,
      -crouch * 0.05,
    );
    // Hips counter-rotate against the shoulders: the single strongest cue that
    // a biped is walking rather than sliding.
    const hipTwist = (0.09 + run * 0.13) * gait;
    b.pelvis.rotation.set(
      crouch * 0.16 + run * 0.10,
      -sinP * hipTwist,
      sinP * (0.035 + run * 0.05) * gait,
    );

    // ── Spine / chest ────────────────────────────────────────────────────────
    const lean = run * 0.16 + crouch * 0.10;
    b.spine.rotation.set(lean * 0.4, sinP * hipTwist * 0.35, -sinP * 0.02 * gait);
    // `flinchDir*` is the bullet's direction of travel in model space, so the
    // body must be pushed ALONG it — a round from the front drives the torso
    // backwards. Both terms are negated relative to the raw axis because a
    // positive chest rotation.z tilts toward the model's left (−X) and a
    // positive rotation.x pitches the chest forward (+Z).
    b.chest.rotation.set(
      lean * 0.6 - this.fireKick * 0.045,
      sinP * hipTwist * 0.75,
      -sinP * (0.03 + run * 0.04) * gait - this.flinch * this.flinchDirX * 0.18,
    );

    // Idle breathing — small, slow, and only when nearly still.
    const idleAmt = 1 - gait;
    if (idleAmt > 0.01) {
      const breath = Math.sin(this.idleT * 1.35);
      b.chest.rotation.x += breath * 0.022 * idleAmt;
      b.pelvis.position.y += breath * 0.006 * idleAmt;
      b.chest.rotation.y += Math.sin(this.idleT * 0.42) * 0.05 * idleAmt;
    }

    // ── Legs (FK pass) ───────────────────────────────────────────────────────
    // Character faces +Z, so a *negative* rotation.x swings the limb forward.
    this.poseLeg(b.upperLegL, b.lowerLegL, b.footL, this.phase, hipSwingUsed, kneeBase, kneeSwing, crouch, gait, run);
    this.poseLeg(b.upperLegR, b.lowerLegR, b.footR, this.phase + Math.PI, hipSwingUsed, kneeBase, kneeSwing, crouch, gait, run);

    // ── Arms ─────────────────────────────────────────────────────────────────
    this.poseArms(input, run, gait, crouch);

    // ── Head look ────────────────────────────────────────────────────────────
    this.hasAim = input.aimAt !== null;
    if (input.aimAt) this.aimGoal.copy(input.aimAt);
    this.poseHead(sinP, gait, run);

    // Matrices must be current before the IK pass reads world transforms.
    this.m.root.updateMatrixWorld(true);

    if (this.footIK && this.groundSampler && input.grounded) {
      // Contact is scheduled by PHASE, not discovered from the FK foot height.
      // cos(phase) ≤ 0 is exactly the half-cycle between the forward extreme
      // (π/2) and the rearward extreme (3π/2) that `measureGait` integrates the
      // stride over, so schedule and stride cannot drift apart.
      // Swing feet need a lift arc. Left to pure FK the swing foot tracks below
      // the floor once the hips drop to reach-height, and clamping it to the
      // surface makes it *drag* — the exact artefact the pin removes from the
      // stance leg, reintroduced on the other half of the cycle.
      const lift = 0.085 + run * 0.16;
      this.plantFoot(b.upperLegL, b.lowerLegL, b.footL, 0, cosP <= 0, swingPhase(this.phase), lift);
      this.plantFoot(b.upperLegR, b.lowerLegR, b.footR, 1, cosP >= 0, swingPhase(this.phase + Math.PI), lift);
    }
    if (this.aimSm > 0.05) this.placeSupportHand();
  }

  /**
   * Forward-kinematic ankle offset along +Z for one leg, in pelvis space.
   * Mirrors `poseLeg` exactly — if one changes, so must the other.
   *
   * Thigh points down −Y and rotates about X by `h`; the shin adds `k` on the
   * same axis, so the ankle sits at −L₁·sin(h) − L₂·sin(h+k).
   */
  private ankleZ(phase: number, hipSwing: number, kneeBase: number, kneeSwing: number, crouch: number, run: number): number {
    const s = Math.sin(phase);
    const swing = Math.max(0, Math.sin(phase - 0.55));
    const h = -s * hipSwing - crouch * 0.95 - run * 0.10;
    const k = kneeBase + swing * swing * kneeSwing;
    return -L_UPPERLEG * Math.sin(h) - L_LOWERLEG * Math.sin(h + k);
  }

  /**
   * Derives the two numbers that keep the gait self-consistent.
   *
   * `strideCache` — ground distance one full cycle covers. The foot is planted
   * from its forward extreme (phase π/2) to its rearward extreme (3π/2); across
   * that half-cycle the body must travel exactly the distance the ankle sweeps,
   * or the foot slides. Two steps per cycle, hence the doubling.
   *
   * `pelvisCache` — how high the hips may sit. This is the constraint that is
   * easy to miss: with the pelvis at full standing height the leg is *already*
   * at maximum extension reaching straight down, so at any stride offset the
   * foot physically cannot reach the floor and the IK silently clamps — leaving
   * the foot hanging and skating. Solving √(reach² − offset²) for the hip
   * height is what lets a planted foot actually touch the ground, and it
   * produces the crouched running posture a real sprinter adopts for exactly
   * the same reason.
   */
  private measureGait(hipSwing: number, kneeBase: number, kneeSwing: number, crouch: number, run: number, gait: number): number {
    // Hip height comes FIRST, from a natural gait crouch: ~9 cm walking, ~17 cm
    // at a sprint. Solving the other way round — lowering the hips until they
    // can reach whatever swing was requested — bottoms out at a 38 cm drop for
    // a full sprint, i.e. a 1.42 m soldier duck-walking, because this rig's
    // bind pose already has the legs at ~100% extension (hip 0.93, leg 0.86).
    const drop = (0.09 + run * 0.08) * gait;
    this.pelvisCache = Y_PELVIS - drop - crouch * 0.34;

    // Whatever reach is left over caps how far the ankle may travel; anything
    // beyond it is unreachable and the foot would hang in the air.
    const vertical = this.pelvisCache + D_HIP_DOWN - ANKLE_HEIGHT;
    const reach = L_UPPERLEG + L_LOWERLEG - 0.02;
    const maxOffset = Math.sqrt(Math.max(0.0025, reach * reach - vertical * vertical));

    let swing = hipSwing;
    let front = this.ankleZ(Math.PI * 0.5, swing, kneeBase, kneeSwing, crouch, run);
    let back = this.ankleZ(Math.PI * 1.5, swing, kneeBase, kneeSwing, crouch, run);
    let offset = Math.max(Math.abs(front), Math.abs(back));
    if (offset > maxOffset) {
      // ankleZ is very nearly linear in the swing angle over this range, so one
      // proportional correction lands within a millimetre.
      swing *= maxOffset / offset;
      front = this.ankleZ(Math.PI * 0.5, swing, kneeBase, kneeSwing, crouch, run);
      back = this.ankleZ(Math.PI * 1.5, swing, kneeBase, kneeSwing, crouch, run);
    }

    this.strideCache = Math.abs(front - back) * 2;
    return swing;
  }

  private poseLeg(
    upper: THREE.Object3D,
    lower: THREE.Object3D,
    foot: THREE.Object3D,
    phase: number,
    hipSwing: number,
    kneeBase: number,
    kneeSwing: number,
    crouch: number,
    gait: number,
    run: number,
  ): void {
    const s = Math.sin(phase);
    // Swing phase is the half-cycle where the foot is off the ground; the knee
    // must fold there and stay near-straight through stance or the leg clips
    // the floor.
    const swing = Math.max(0, Math.sin(phase - 0.55));
    upper.rotation.set(-s * hipSwing - crouch * 0.95 - run * 0.10, 0, 0);
    lower.rotation.set(kneeBase + swing * swing * kneeSwing, 0, 0);
    // Ankle keeps the sole roughly level, with toe-off at the end of stance.
    const toeOff = Math.max(0, -Math.sin(phase + 0.9)) * 0.45 * gait;
    foot.rotation.set(s * hipSwing * 0.45 - lower.rotation.x * 0.55 + toeOff - crouch * 0.18, 0, 0);
  }

  private poseArms(input: AnimatorInput, run: number, gait: number, crouch: number): void {
    const b = this.m.bones;
    const aim = this.aimSm;
    const armSwing = (0.34 + run * 0.42) * gait * (1 - aim * 0.85);
    const s = Math.sin(this.phase);

    // Free-carry pose: arms swing opposed to the same-side leg.
    // Shouldered pose: right arm forms the firing triangle, left is IK'd later.
    const carryUpX = s * armSwing;
    const carryLoX = 0.35 + Math.max(0, s) * 0.3;

    // Right (trigger) arm.
    const fireUpX = -0.62 + this.fireKick * 0.10;
    const fireUpZ = -0.30;
    const fireLoX = 1.22 - this.fireKick * 0.16;
    b.upperArmR.rotation.set(
      THREE.MathUtils.lerp(-carryUpX - crouch * 0.12, fireUpX, aim),
      0,
      THREE.MathUtils.lerp(-0.10, fireUpZ, aim),
    );
    b.lowerArmR.rotation.set(THREE.MathUtils.lerp(carryLoX, fireLoX, aim), 0, 0);
    b.handR.rotation.set(THREE.MathUtils.lerp(0.1, -0.28, aim), 0, THREE.MathUtils.lerp(0, 0.18, aim));

    // Left (support) arm — a plausible default; overwritten by IK when aiming.
    b.upperArmL.rotation.set(
      THREE.MathUtils.lerp(carryUpX - crouch * 0.12, -0.75, aim),
      0,
      THREE.MathUtils.lerp(0.10, 0.42, aim),
    );
    b.lowerArmL.rotation.set(THREE.MathUtils.lerp(carryLoX, 1.55, aim), 0, 0);
    b.handL.rotation.set(0, 0, 0);

    // Low-ready: when not aiming the muzzle points down and inboard.
    this.m.rifle.rotation.set(
      THREE.MathUtils.lerp(-0.55, 0, aim),
      THREE.MathUtils.lerp(0.35, 0, aim),
      0,
    );

    if (this.flinch > 0.01) {
      const f = this.flinch * this.flinch;
      b.upperArmL.rotation.x += f * 0.35;
      b.upperArmR.rotation.x += f * 0.30;
      // Positive chest rotation.x pitches forward (+Z), and flinchDirZ is the
      // bullet's forward component, so a frontal hit (flinchDirZ < 0) must
      // produce a negative pitch: the soldier rocks away from the shooter.
      b.chest.rotation.x += f * 0.22 * this.flinchDirZ;
    }
    void input;
  }

  private poseHead(sinP: number, gait: number, run: number): void {
    const b = this.m.bones;
    let yaw = 0;
    let pitch = 0;

    if (this.hasAim) {
      // Aim in chest space so the neck twist is relative to the torso, and clamp
      // it to a human range — an unclamped look-at produces owl necks.
      _v0.copy(this.aimGoal);
      // updateMatrixWorld only walks DOWN; the chest's own world matrix depends
      // on pelvis and spine, so ancestors must be refreshed or the head aims at
      // where the torso was last frame.
      b.chest.updateWorldMatrix(true, false);
      _m0.copy(b.chest.matrixWorld).invert();
      _v0.applyMatrix4(_m0);
      yaw = THREE.MathUtils.clamp(Math.atan2(_v0.x, _v0.z), -70 * DEG, 70 * DEG);
      const flat = Math.hypot(_v0.x, _v0.z);
      pitch = THREE.MathUtils.clamp(-Math.atan2(_v0.y, flat), -50 * DEG, 45 * DEG);
    }

    // Distribute the look: mostly neck+head, a little chest counter-rotation.
    const w = this.aimSm;
    b.chest.rotation.y += yaw * 0.35 * w;
    b.neck.rotation.set(pitch * 0.35, yaw * 0.30, 0);
    b.head.rotation.set(
      pitch * 0.65 + (1 - gait) * Math.sin(this.idleT * 0.6) * 0.03,
      yaw * 0.35,
      -sinP * 0.03 * gait - run * 0.02,
    );
  }

  /**
   * Foot locking — the mechanism that actually stops skating.
   *
   * Matching the stride to the leg pose (see `measureStride`) makes the *average*
   * foot velocity zero across a stance phase, but the ankle traces a sinusoid,
   * so instantaneously it still slides forward and back. Measured on a straight
   * walk that leaves the contact point moving at ~100% of body speed — visually
   * indistinguishable from having no foot IK at all.
   *
   * So the contact point is latched. The frame a foot first touches down its
   * world XZ is recorded, and for the rest of the stance the leg is IK'd to
   * that fixed point while the body walks past it. Height still tracks the
   * ground every frame, so slopes and stairs stay correct.
   *
   * The pin cannot over-extend the leg: total FK travel across a stance equals
   * the body's travel by construction, and `solveTwoBoneIK` clamps reach anyway.
   */
  private plantFoot(
    upper: THREE.Object3D,
    lower: THREE.Object3D,
    foot: THREE.Object3D,
    idx: number,
    inStance: boolean,
    swingT: number,
    liftHeight: number,
  ): void {
    const sampler = this.groundSampler;
    if (!sampler) return;

    if (!inStance) {
      this.plantActive[idx] = false;
      // Swing leg: keep the FK sweep in XZ (that sweep is what makes the stride
      // arithmetic work) but drive height along a lift arc that returns to zero
      // exactly at touchdown, so it meets the pin continuously.
      foot.getWorldPosition(_v1);
      upper.getWorldPosition(_v2);
      if (!sampler(_v1.x, _v1.z, _v2.y + 0.4, _v3)) return;
      const floorY = _v3.y + ANKLE_HEIGHT;
      _v1.y = floorY + Math.sin(Math.PI * swingT) * liftHeight;
      solveTwoBoneIK(upper, lower, L_UPPERLEG, L_LOWERLEG, _v1, -1);
      // Refresh from the UPPER bone: the IK rewrote *its* quaternion, and
      // `updateMatrixWorld` only recomputes the subtree it is called on. Doing
      // this on `lower` leaves the thigh's world matrix stale, so the foot
      // never actually lands where it was told to.
      upper.updateMatrixWorld(true);
      return;
    }

    foot.getWorldPosition(_v1);
    upper.getWorldPosition(_v2);

    if (!this.plantActive[idx]) {
      // Touchdown: latch the world XZ the foot will stay welded to.
      this.plantActive[idx] = true;
      this.plantX[idx] = _v1.x;
      this.plantZ[idx] = _v1.z;
    }

    // Height tracks the ground under the *pinned* point, so a foot planted on
    // a stair tread or a slope stays on it while the body walks past.
    if (!sampler(this.plantX[idx], this.plantZ[idx], _v2.y + 0.6, _v3)) {
      this.plantActive[idx] = false;
      return;
    }
    _v1.set(this.plantX[idx], _v3.y + ANKLE_HEIGHT, this.plantZ[idx]);

    // If the body has walked beyond the leg's reach the pin is stale (stepped
    // off a ledge, teleported, shoved): drop it and re-plant next frame.
    if (_v1.distanceToSquared(_v2) > (L_UPPERLEG + L_LOWERLEG) * (L_UPPERLEG + L_LOWERLEG)) {
      this.plantActive[idx] = false;
      return;
    }

    solveTwoBoneIK(upper, lower, L_UPPERLEG, L_LOWERLEG, _v1, -1);
    upper.updateMatrixWorld(true);

    // Level the sole. The IK rotates the shin to reach the pin, which would
    // otherwise drag the foot to whatever pitch the shin ended at.
    lower.getWorldQuaternion(_q0);
    foot.quaternion.copy(_q0).invert().multiply(this.m.root.quaternion);
    foot.rotation.x += 0.06; // a touch of toe-down so the heel is not proud
  }

  /** Two-bone IK that welds the left hand to the rifle's fore-grip while aiming. */
  private placeSupportHand(): void {
    const b = this.m.bones;
    this.m.rifleForegrip.getWorldPosition(_v1);
    b.upperArmL.getWorldPosition(_v2);
    const reach = _v1.distanceTo(_v2);
    if (reach > (L_UPPERARM + L_LOWERARM) * 0.99) return; // out of reach: keep FK
    solveTwoBoneIK(b.upperArmL, b.lowerArmL, L_UPPERARM, L_LOWERARM, _v1, 1);
  }
}
