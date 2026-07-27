/**
 * NOISEFIELD — "COMPOUND ZULU"
 * A sun-baked North-African urban compound, roughly 90 m x 90 m playable.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * LAYOUT INTENT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Classic three-lane arena, lanes running north-south, with lateral connectors
 * at the north, middle and south so the lanes are a *net* rather than three
 * parallel corridors. Every lane has a distinct character, a distinct engagement
 * range, and a distinct dominant weapon — which is what makes loadout choice
 * mean something and gives a losing team somewhere else to go.
 *
 *              -Z  (north)
 *      ┌───────────────────────────────────────────┐
 *      │  NORTH WALL     [ blocked gate ]          │
 *      │ ┌───┐  RADIO HOUSE      GARAGE     ┌────┐ │
 *      │ │ W │   (2 storey)     (1 storey)  │ E  │ │
 *      │ │ E │ ═══════ NORTH YARD ════════  │ A  │ │
 *  -X  │ │ S │      ┌── gate wall ──┐       │ S  │ │  +X
 * (west)│ │ T │      │               │       │ T  │ │ (east)
 *      │ │   │ ALLEY │   COURTYARD   │ MARKET│    │ │
 *      │ │ T │  6 m  │  ⌂ COMMS TOWER│ STREET│ T  │ │
 *      │ │ E │       │   + fountain  │  9 m  │ E  │ │
 *      │ │ R │       └── low wall ───┘       │ R  │ │
 *      │ │ R │  SW RUIN        MARKET HALL   │ R  │ │
 *      │ └───┘  (collapsed)     (2 storey)   └────┘ │
 *      │        SOUTH BLOCK   SOUTH CONNECTOR       │
 *      └───────────────────────────────────────────┘
 *              +Z  (south)
 *
 * WEST LANE — THE ALLEY (x -39..-33, 6 m wide).
 *   Tight, shaded, cluttered. SMG/shotgun territory. Roofed visually by laundry
 *   lines and pipe runs so it feels enclosed without actually being covered.
 *   Three breaches in its east wall stop it being a pure kill-tunnel: you can
 *   always cut into the courtyard rather than committing to the full run. A
 *   fire-escape ladder at mid-length lifts you onto the terrace roof, which is
 *   the alley's vertical counterpart.
 *
 * CENTRE LANE — THE COURTYARD (x -21..14, z -26..12).
 *   Open, long sightlines, but never a bare field: the comms tower and its
 *   fountain basin sit slightly off-centre (deliberately not at the origin —
 *   perfect symmetry is boring and unmemorable), and containers, barriers,
 *   wrecks and a chicane of jersey barriers break the space into a rhythm of
 *   ~6 m cover beats. This is where ARs win. The gate wall on the north edge
 *   creates a genuine choke that the north yard feeds into.
 *
 * EAST LANE — THE MARKET STREET (x 30..39, 9 m wide).
 *   Medium range, heavily cluttered at waist height by stalls and crates, with
 *   awnings breaking up vertical sightlines. Two gangways cross it overhead and
 *   a scaffold stands mid-street, so the lane is contested in three dimensions.
 *
 * ROOFTOP ROUTE (the "fourth lane").
 *   West terrace roof → gangway → Radio House roof; Garage roof → gangway →
 *   east terrace low roof; Market Hall roof → gangway → east terrace roof.
 *   Every roof has a 1.05 m parapet, so a roof player has crouch cover but is
 *   skylined the moment they stand — high reward, real risk, and counterable
 *   from the ground. Every roof has at least two ways down.
 *
 * COVER RHYTHM.
 *   Heights are chosen, not scattered. 0.81 m jersey barriers and 0.95 m
 *   fountain rim / low walls are crouch cover. 1.05 m market counters and
 *   parapets are the "lean over" band. 2.59 m containers and 2.2 m stacked
 *   gabion are full cover you must go around. Nothing lands in the useless
 *   1.6-1.9 m band where a player can neither see over nor be safe behind.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PERFORMANCE STRATEGY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Every piece of static geometry goes into one of ~14 material buckets and is
 * merged into a single BufferGeometry per bucket. Every repeated prop goes into
 * an InstancedMesh. The result is roughly 14 merged draws + ~18 instanced draws
 * + 12 decal draws ≈ 45 draw calls for the entire map, against a 350 budget —
 * which leaves headroom for three shadow cascades and the entities/VFX systems.
 *
 * The cost of merging is that static geometry cannot be culled per-object. That
 * is the right trade here: the map is 90 m across and the player can see a
 * large fraction of it at once, so per-object culling would reject very little
 * while costing us 40x the draw calls.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  FrameContext,
  IMaterialLibrary,
  SurfaceType,
  System,
} from '../core/Types';
import {
  GeometryBatcher,
  InstanceBuilder,
  Rng,
  type WallOpening,
  chamferedBox,
  chamferedCylinder,
  rampGeometry,
  rockGeometry,
  sandDrift,
  staircase,
  trs,
  triCount,
  wallWithOpenings,
} from './Geometry';
import { Props, type GeoParts } from './Props';
import type { Decals, GrungeKind } from './Decals';

// ─────────────────────────────────────────────────────────────────────────────
// Material buckets
// ─────────────────────────────────────────────────────────────────────────────

interface BucketDef {
  /** Physical classification, for ballistics / VFX / footstep audio. */
  surface: SurfaceType;
  /** World-space metres covered by one texture tile. */
  tile: number;
  /** Take the shared material straight from the library rather than authoring one. */
  fromLibrary: boolean;
  /**
   * Library recipe to tint when this bucket needs the library's procedural PBR
   * maps but a different base colour (e.g. "plaster, but painted turquoise").
   * `MaterialLibrary.variant` rebuilds the recipe with overridden params, so the
   * bucket keeps every generated map instead of falling back to a flat colour.
   */
  libId?: string;
  color: number;
  roughness: number;
  metalness: number;
  transparent?: boolean;
  opacity?: number;
  doubleSided?: boolean;
}

/**
 * Fourteen buckets. The palette is the art direction made literal: sand
 * #C2A878, weathered concrete #8A8578, rust #7A4A32, plus two saturated painted
 * accents. The accents are the important part — a compound rendered entirely in
 * browns is unreadable and depressing, and three or four doors in faded
 * turquoise are enough to make the whole thing feel lived in.
 */
const BUCKETS: Readonly<Record<string, BucketDef>> = {
  concrete:    { surface: 'concrete', tile: 2.0, fromLibrary: true,  color: 0x8a8578, roughness: 0.94, metalness: 0.0 },
  plaster:     { surface: 'concrete', tile: 2.6, fromLibrary: false, libId: 'plaster', color: 0xc2a878, roughness: 0.90, metalness: 0.0 },
  plasterAlt:  { surface: 'concrete', tile: 2.6, fromLibrary: false, libId: 'plaster', color: 0xa89272, roughness: 0.92, metalness: 0.0 },
  plasterPale: { surface: 'concrete', tile: 2.6, fromLibrary: false, libId: 'plaster', color: 0xd6c3a0, roughness: 0.88, metalness: 0.0 },
  sand:        { surface: 'sand',     tile: 4.0, fromLibrary: true,  color: 0xc2a878, roughness: 1.00, metalness: 0.0 },
  dirt:        { surface: 'dirt',     tile: 2.0, fromLibrary: true,  color: 0x6b5a44, roughness: 1.00, metalness: 0.0 },
  rock:        { surface: 'concrete', tile: 3.2, fromLibrary: false, libId: 'concrete', color: 0x9a8b72, roughness: 1.00, metalness: 0.0 },
  metal:       { surface: 'metal',    tile: 1.6, fromLibrary: true,  color: 0x9a9186, roughness: 0.52, metalness: 0.88 },
  rust:        { surface: 'metal',    tile: 1.6, fromLibrary: false, libId: 'metal', color: 0x7a4a32, roughness: 0.84, metalness: 0.52 },
  wood:        { surface: 'wood',     tile: 1.2, fromLibrary: true,  color: 0x8a6a46, roughness: 0.92, metalness: 0.0 },
  glass:       { surface: 'glass',    tile: 2.0, fromLibrary: false, libId: 'glass', color: 0x8fb4b8, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.24, doubleSided: true },
  fabric:      { surface: 'fabric',   tile: 1.6, fromLibrary: false, libId: 'fabric', color: 0xd8c9a8, roughness: 0.96, metalness: 0.0, doubleSided: true },
  rubber:      { surface: 'rubber',   tile: 1.0, fromLibrary: false, libId: 'rubber', color: 0x2c2926, roughness: 0.94, metalness: 0.0 },
  accentTeal:  { surface: 'wood',     tile: 1.6, fromLibrary: false, libId: 'steel_painted', color: 0x2f8a86, roughness: 0.68, metalness: 0.06 },
  accentOchre: { surface: 'wood',     tile: 1.6, fromLibrary: false, libId: 'steel_painted', color: 0xbe7420, roughness: 0.72, metalness: 0.06 },
  // Vehicle bodywork: sun-bleached, oxidised, nowhere near shutter-saturation.
  paintFaded:  { surface: 'metal',    tile: 1.8, fromLibrary: false, libId: 'steel_painted', color: 0x5f7d76, roughness: 0.74, metalness: 0.24 },

  // ── Detailing-pass additions ──────────────────────────────────────────────
  /**
   * The unlit box behind a window on a building you cannot enter. Nearly black
   * and fully rough so it eats every bounce: the whole job of this material is
   * to make an opening read as depth rather than as a pale rectangle painted on
   * a wall. Classified as concrete because that is what a bullet finds there.
   */
  interiorDark: { surface: 'concrete', tile: 1.6, fromLibrary: false, color: 0x14100c, roughness: 1.0, metalness: 0.0 },
  /**
   * Painted base course. Whitewashed walls over a dark painted plinth is the
   * single most characteristic colour move in the region, and it costs one
   * material: it puts a horizontal band of *non-beige* along the foot of every
   * building, which is exactly where the eye rests at 1.7 m.
   */
  plinthPaint: { surface: 'concrete', tile: 2.2, fromLibrary: false, libId: 'concrete', color: 0x2d474f, roughness: 0.78, metalness: 0.0 },
  /** Faded polytarp — the only saturated blue in the compound, used sparingly. */
  tarpBlue:    { surface: 'fabric',   tile: 1.6, fromLibrary: false, libId: 'fabric', color: 0x2f6f96, roughness: 0.88, metalness: 0.0, doubleSided: true },
  /** Oxidised red-lead paint: one shipping container, signage, a few steel doors. */
  paintRust:   { surface: 'metal',    tile: 1.8, fromLibrary: false, color: 0x8c3f2a, roughness: 0.78, metalness: 0.22 },
  /** Sun-bleached cardboard and paper litter. */
  paper:       { surface: 'fabric',   tile: 0.8, fromLibrary: false, color: 0xa8967a, roughness: 0.98, metalness: 0.0, doubleSided: true },
};

const DEFAULT_BUCKET: BucketDef = BUCKETS.concrete;

// ─────────────────────────────────────────────────────────────────────────────
// Layout constants — every number here is load-bearing for gameplay
// ─────────────────────────────────────────────────────────────────────────────

/** Perimeter wall centre lines. */
const EDGE = 45;
/** Storey height. 3.4 m gives a 2.1 m door plus a generous soffit. */
const STOREY = 3.4;
/** Exterior wall thickness. Thick enough for window reveals to read. */
const WALL_T = 0.36;
/** Parapet above a roof deck: crouch cover, skylines you if you stand. */
const PARAPET = 1.05;
const DOOR_W = 1.1;
const DOOR_H = 2.1;
const WIN_W = 1.1;
const WIN_H = 1.3;
/** Window sill height. Below chest so a window is a usable firing position. */
const WIN_SILL = 1.05;

interface Rect {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
}

const WEST_TERRACE: Rect = { x0: -EDGE, x1: -39, z0: -42, z1: 42 };
const EAST_TERRACE_N: Rect = { x0: 39, x1: EDGE, z0: -42, z1: -26 };
const EAST_TERRACE_M: Rect = { x0: 39, x1: EDGE, z0: -26, z1: 12 };
const EAST_TERRACE_S: Rect = { x0: 39, x1: EDGE, z0: 12, z1: 42 };
const RADIO_HOUSE: Rect = { x0: -34, x1: -21, z0: -38, z1: -26 };
const GARAGE: Rect = { x0: 14, x1: 30, z0: -38, z1: -27 };
const MARKET_HALL: Rect = { x0: 7, x1: 21, z0: 12, z1: 26 };
const SW_RUIN: Rect = { x0: -34, x1: -21, z0: 17, z1: 30 };
const SOUTH_BLOCK: Rect = { x0: -8, x1: 4, z0: 30, z1: 40 };

/** Roof deck heights. Matched pairs so gangways run level. */
const ROOF_2 = STOREY * 2;      // 6.80 — two-storey roofs
const ROOF_GARAGE = 4.4;        // garage + east terrace north section
const ROOF_SOUTH = 3.9;

/** The hero landmark's position — deliberately off the exact map centre. */
const TOWER = new THREE.Vector3(-2, 0, -4);

// ─────────────────────────────────────────────────────────────────────────────

interface FacadeWindow {
  /** Offset along the wall from its centre, metres. */
  x: number;
  y: number;
  w?: number;
  h?: number;
  glass?: boolean;
  /** Paint a shutter across it in one of the accent colours. */
  shutter?: 'teal' | 'ochre';
  /** Skip the moulded frame (for openings that are just holes in a ruin). */
  bare?: boolean;
  /** Force the opening's condition instead of rolling for it. */
  state?: WindowState;
}

/**
 * What has happened to a given opening.
 *
 * A terrace where every window is identically glazed reads as wallpaper. Rolling
 * a condition per opening — most intact, a few boarded, one broken, one shut up
 * behind closed shutters — costs nothing and is the difference between a row of
 * windows and a row of *buildings*.
 */
type WindowState = 'glazed' | 'boarded' | 'broken' | 'shuttered' | 'open';

interface FacadeDoor {
  x: number;
  w?: number;
  h?: number;
  /** Radians the leaf is swung open; omit for an empty doorway. */
  leaf?: number;
  accent?: 'teal' | 'ochre';
}

interface FacadeOpts {
  key: string;
  ax: number;
  az: number;
  bx: number;
  bz: number;
  baseY?: number;
  height: number;
  thickness?: number;
  windows?: FacadeWindow[];
  doors?: FacadeDoor[];
  /** Spacing of proud pilasters along the wall; 0 disables. */
  pilasters?: number;
  /**
   * Mirror the pilasters onto the inward face as well.
   *
   * For a freestanding wall — the gate wall, the alley's east wall, the market
   * boundary — BOTH faces are seen, often from opposite lanes within seconds of
   * each other. Detailing only the nominal outward side leaves a 40 m blank
   * plane facing the other way, which is the largest untextured surface in the
   * map and reads exactly as what it is.
   */
  pilastersBothSides?: boolean;
  /** Cap the wall with a proud coping course. */
  coping?: boolean;
  /** Pile wind-blown sand against the outward face. */
  drift?: number;
  /** Extend the wall by half a thickness at each end so corners close. */
  extend?: boolean;
  bevel?: number;

  // ── Architectural relief ──────────────────────────────────────────────────
  /**
   * Height of a projecting base course, metres. 0.45-0.7 is right for this
   * vernacular. Automatically punched around doorways.
   */
  plinth?: number;
  /** Bucket for the plinth; use a painted accent to break the beige. */
  plinthKey?: string;
  /** Y heights of projecting string courses (floor lines). */
  bands?: number[];
  /** Projecting cornice / eaves band at the head of the wall. */
  cornice?: boolean;
  /**
   * Put an unlit recess behind every glazed opening. Only for shells the player
   * cannot enter — on an enterable building it would blind a firing position.
   */
  blind?: boolean;
  /**
   * Probability that any given opening is boarded, broken or shuttered shut
   * rather than plainly glazed.
   */
  decay?: number;
  /**
   * Density of service dressing on the outward face: downpipes, conduit,
   * junction boxes, AC units, lamps, vents. 0 disables, 1 is a busy street
   * elevation. Only ever set on faces the player can actually see.
   */
  dress?: number;
  /** Emit painted shop signage over doorways. */
  signage?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────

export class Level implements System {
  readonly name = 'level';

  /**
   * Everything the physics system should turn into static collision.
   *
   * Contains merged `THREE.Mesh` (use the geometry directly as a trimesh) and
   * `THREE.InstancedMesh` (iterate `count` and `getMatrixAt` to instance the
   * source geometry). Every entry has been passed through
   * `IMaterialLibrary.tag`, so `surfaceOf` resolves the right SurfaceType for
   * ballistics, impact VFX and footstep audio.
   */
  readonly collidables: THREE.Object3D[] = [];

  /** Ten spawns, spread around the perimeter, all on flat standable ground. */
  readonly spawnPoints: THREE.Vector3[] = [];

  /** Playable volume, including roof space. */
  readonly bounds = new THREE.Box3(
    new THREE.Vector3(-EDGE - 1, -1, -EDGE - 1),
    new THREE.Vector3(EDGE + 1, 22, EDGE + 1),
  );

  readonly root = new THREE.Group();

  private readonly rng = new Rng(0x2b1e5a7d);
  private readonly materials = new Map<string, THREE.Material>();
  /**
   * Metres-per-texture-tile actually in force for each bucket.
   *
   * The bucket table declares a tile size, but for buckets that come from the
   * shared library the *library* is authoritative — it bakes its textures at a
   * specific world scale and publishes it as `material.userData.tileMeters`.
   * Trusting the local constant instead produced a visible texel-density seam
   * where a library concrete surface met a locally-authored plaster one
   * (2.0 declared here versus 2.5 in the library, a 25% mismatch). Resolved at
   * material time, consumed by the batcher.
   */
  private readonly tiles = new Map<string, number>();
  private readonly instances = new Map<string, { builder: InstanceBuilder; matKey: string; shadow: boolean; collide: boolean }>();
  private readonly ownedMaterials: THREE.Material[] = [];
  private batcher!: GeometryBatcher;
  private lib: IMaterialLibrary | undefined;

  /**
   * Grunge is deferred to the first frame: the decal system may not have been
   * registered (let alone initialised) when the level builds, and system
   * ordering in main.ts is not ours to depend on.
   */
  private grungePlaced = false;
  private readonly grungeQueue: Array<{
    p: THREE.Vector3;
    n: THREE.Vector3;
    kind: GrungeKind;
    size: number;
    roll: number;
    opacity: number;
  }> = [];

  private stats = { drawCalls: 0, triangles: 0, merged: 0, instanced: 0 };

  // ═══════════════════════════════════════════════════════════════════════════
  // Lifecycle
  // ═══════════════════════════════════════════════════════════════════════════

  init(ctx: EngineContext): void {
    this.root.name = 'level';
    this.resolveMaterials(ctx);
    this.batcher = new GeometryBatcher((key) => this.tileOf(key));

    this.buildGround();
    this.buildPerimeter();
    this.buildTerraces();
    this.buildRadioHouse();
    this.buildGarage();
    this.buildMarketHall();
    this.buildRuin();
    this.buildSouthBlock();
    this.buildInternalWalls();
    this.buildHeroTower();
    this.buildVerticality();
    this.buildMarket();
    this.scatterCover();
    this.scatterDebris();
    // Detailing passes run last so they can sit on top of everything above.
    this.buildSkyline();
    this.buildStreetFurniture();
    this.scatterGroundDetail();

    this.flush(ctx);
    this.defineSpawnPoints();

    ctx.scene.add(this.root);
    this.report();
  }

  update(ctx: EngineContext, _frame: FrameContext): void {
    if (this.grungePlaced) return;
    this.grungePlaced = true;
    const decals = ctx.get<Decals>('decals');
    if (!decals || typeof decals.addGrunge !== 'function') return;
    for (const g of this.grungeQueue) decals.addGrunge(g.p, g.n, g.kind, g.size, g.roll, g.opacity);
    this.grungeQueue.length = 0;
  }

  dispose(): void {
    this.root.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
    });
    for (const m of this.ownedMaterials) m.dispose();
    this.root.clear();
    this.root.removeFromParent();
    this.collidables.length = 0;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Materials
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Resolve every bucket to a concrete material.
   *
   * The four primary surfaces come from the shared library so they pick up its
   * procedural PBR maps. The colour variants (plaster tones, rust, the painted
   * accents) are authored locally because the library is keyed on SurfaceType
   * and has no vocabulary for "the same concrete but painted turquoise" — but
   * every mesh is still tagged with its true SurfaceType, so ballistics and
   * audio behave correctly regardless of which material it ended up with.
   *
   * Written defensively: the library is a peer system that may not exist yet.
   */
  private resolveMaterials(ctx: EngineContext): void {
    this.lib = ctx.get<IMaterialLibrary>('materials');
    for (const [key, def] of Object.entries(BUCKETS)) {
      let mat: THREE.Material | undefined;
      if (def.fromLibrary && this.lib) {
        try {
          mat = this.lib.forSurface(def.surface);
        } catch {
          mat = undefined;
        }
      }
      // A tinted library variant keeps every procedural map (albedo, normal,
      // ORM, detail blending) while overriding only the base colour, which is
      // what these buckets actually want. Falling through to a flat colour is
      // the last resort, not the default: it is what made facades read as
      // untextured boxes.
      if (!mat && def.libId && this.lib) {
        try {
          const lib = this.lib as unknown as {
            variant?: (id: string, o: THREE.MeshPhysicalMaterialParameters) => THREE.Material;
          };
          if (typeof lib.variant === 'function') {
            const v = lib.variant(def.libId, {
              color: new THREE.Color(def.color),
              roughness: def.roughness,
              metalness: def.metalness,
              transparent: def.transparent ?? false,
              opacity: def.opacity ?? 1,
              side: def.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
            });
            v.name = `level_${key}_variant`;
            this.ownedMaterials.push(v);
            mat = v;
          }
        } catch (err) {
          console.warn(`[Level] variant("${def.libId}") failed for bucket "${key}":`, err);
          mat = undefined;
        }
      }
      if (!mat) {
        const local = new THREE.MeshStandardMaterial({
          color: def.color,
          roughness: def.roughness,
          metalness: def.metalness,
          transparent: def.transparent ?? false,
          opacity: def.opacity ?? 1,
          side: def.doubleSided ? THREE.DoubleSide : THREE.FrontSide,
        });
        local.name = `level_${key}`;
        this.ownedMaterials.push(local);
        mat = local;
      }
      this.materials.set(key, mat);
      // Prefer the material's own declared world tile size over our constant.
      const declared: unknown = mat.userData['tileMeters'];
      this.tiles.set(key, typeof declared === 'number' && declared > 0.05 ? declared : def.tile);
    }
  }

  /** World metres covered by one texture tile for a bucket. */
  private tileOf(key: string): number {
    return this.tiles.get(key) ?? (BUCKETS[key] ?? DEFAULT_BUCKET).tile;
  }

  private material(key: string): THREE.Material {
    return this.materials.get(key) ?? this.materials.get('concrete') ?? new THREE.MeshStandardMaterial();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Batching conveniences
  // ═══════════════════════════════════════════════════════════════════════════

  private add(key: string, geo: THREE.BufferGeometry, m?: THREE.Matrix4): void {
    this.batcher.add(key, geo, m);
    geo.dispose();
  }

  private addParts(parts: GeoParts, m?: THREE.Matrix4): void {
    for (const p of parts) {
      this.batcher.add(p.key, p.geo, m, { keepUV: p.keepUV });
      p.geo.dispose();
    }
  }

  /** Add a chamfered box by its centre, in world space. */
  private boxAt(
    key: string,
    w: number, h: number, d: number,
    x: number, y: number, z: number,
    bevel = 0.03, ry = 0,
  ): void {
    this.add(key, chamferedBox(w, h, d, bevel), trs(x, y, z, ry));
  }

  /**
   * Instance batches whose members are too small, too high or too soft to be
   * worth a physics collider.
   *
   * Every instanced mesh used to be registered as static collision, which meant
   * the broadphase carried 1500 gravel pebbles and 300 scraps of paper. None of
   * them can stop a bullet or a player in any way the player would notice, and
   * a 7 cm collider under a capsule controller is strictly a source of jitter.
   * They still receive light and shadow; they simply do not exist to physics.
   */
  private static readonly NON_COLLIDING = new Set([
    'gravel', 'litter', 'cardboard', 'sack', 'brick', 'chunk', 'stone', 'produce',
    'scupper', 'conduitBox', 'vent', 'wallLamp', 'grate', 'tvAerial', 'dish',
  ]);

  /** Lazily create (and remember) an instance builder for a repeated prop. */
  private inst(name: string, matKey: string, factory: () => THREE.BufferGeometry, shadow = true): InstanceBuilder {
    const found = this.instances.get(name);
    if (found) return found.builder;
    const builder = new InstanceBuilder(name, factory());
    this.instances.set(name, { builder, matKey, shadow, collide: !Level.NON_COLLIDING.has(name) });
    return builder;
  }

  private grunge(
    p: THREE.Vector3,
    n: THREE.Vector3,
    kind: GrungeKind,
    size: number,
    roll = 0,
    opacity = 1,
  ): void {
    this.grungeQueue.push({ p, n, kind, size, roll, opacity });
  }

  /**
   * Pile wind-blown sand along a line, banked against whatever is on its left.
   * The line runs a→b and the drift climbs toward the a→b left-hand side, which
   * matches `facade`'s outward-normal convention.
   */
  private driftAlong(ax: number, az: number, bx: number, bz: number, height: number, depth = 1.2): void {
    const dx = bx - ax;
    const dz = bz - az;
    const length = Math.hypot(dx, dz);
    if (length < 0.5) return;
    const ry = Math.atan2(-dz / length, dx / length);
    const g = sandDrift(length * 0.94, height, depth, this.rng, 16);
    // The drift's crest is on its +Z side, so face it back into the wall.
    this.add('sand', g, trs((ax + bx) * 0.5, 0.005, (az + bz) * 0.5, ry + Math.PI));
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Ground
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The desert floor, plus the paved surfaces that give the compound its
   * legible shape from above.
   *
   * The plane is flat across the entire playable area and only starts to
   * undulate outside it. Undulating ground inside the arena would fight the
   * character controller, make prop placement guesswork and — worst of all —
   * produce sightlines that change with sub-metre position, which players
   * cannot learn.
   */
  /**
   * Terrain height at a world position.
   *
   * Flat across the entire playable area, dunes beyond it. Anything placed
   * OUTSIDE the compound — backdrop buildings, off-map poles, the minaret —
   * must consult this or it floats, and a floating building is the single most
   * conspicuous error a skyline can contain.
   */
  private groundHeight(x: number, z: number): number {
    const r = Math.max(Math.abs(x), Math.abs(z));
    const t = Math.min(1, Math.max(0, (r - EDGE) / 26));
    const blend = t * t * (3 - 2 * t);
    const dune =
      Math.sin(x * 0.055) * 1.5 +
      Math.cos(z * 0.041) * 1.8 +
      Math.sin((x + z) * 0.026) * 2.4;
    return dune * blend;
  }

  private buildGround(): void {
    const SIZE = 200;
    const SEG = 80;
    const geo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.getAttribute('position');
    for (let i = 0; i < pos.count; i++) {
      pos.setY(i, this.groundHeight(pos.getX(i), pos.getZ(i)));
    }
    geo.computeVertexNormals();
    this.add('sand', geo);

    // Paved courtyard apron. A flush 2 cm quad rather than a slab: it changes
    // the material and footstep sound under the centre of the map without
    // introducing a lip for the character controller to trip on.
    const apron = new THREE.PlaneGeometry(38, 41);
    apron.rotateX(-Math.PI / 2);
    this.add('concrete', apron, trs(-3.5, 0.02, -7));
    // Market street paving.
    const street = new THREE.PlaneGeometry(9, 84);
    street.rotateX(-Math.PI / 2);
    this.add('concrete', street, trs(34.5, 0.02, 0));
    // Alley paving, a shade darker.
    const alley = new THREE.PlaneGeometry(6, 84);
    alley.rotateX(-Math.PI / 2);
    this.add('plasterAlt', alley, trs(-36, 0.02, 0));
    // North yard hardstanding, where the vehicles are.
    const yard = new THREE.PlaneGeometry(34, 11);
    yard.rotateX(-Math.PI / 2);
    this.add('concrete', yard, trs(-3, 0.02, -32));

    // Kerbs framing the market street: they read the lane's edge from a
    // distance and give the eye something to follow along its length.
    for (const sx of [-1, 1]) {
      this.boxAt('concrete', 0.22, 0.14, 84, 34.5 + sx * 4.5, 0.06, 0, 0.03);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Facade builder
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Build one wall between two points, with punched openings, moulded frames,
   * glazing, doors, pilasters, coping and a sand drift at its foot.
   *
   * The wall's local +X runs a→b and its local +Z is the *outward* normal, so
   * the caller controls which way a facade faces purely by the order of the two
   * points. Traversing a rectangular footprint as
   * (x0,z0)→(x0,z1)→(x1,z1)→(x1,z0)→(x0,z0) puts all four normals outward.
   */
  private facade(o: FacadeOpts): void {
    const dx = o.bx - o.ax;
    const dz = o.bz - o.az;
    let length = Math.hypot(dx, dz);
    if (length < 1e-4) return;
    const ux = dx / length;
    const uz = dz / length;
    const t = o.thickness ?? WALL_T;
    const baseY = o.baseY ?? 0;
    const bevel = o.bevel ?? 0.035;
    // Outward normal is the direction rotated -90 degrees about +Y.
    const nx = -uz;
    const nz = ux;
    const ry = Math.atan2(nx, nz);
    if (o.extend !== false) length += t;
    const cx = (o.ax + o.bx) * 0.5;
    const cz = (o.az + o.bz) * 0.5;
    const wallM = trs(cx, baseY, cz, ry);

    const openings: WallOpening[] = [];
    for (const w of o.windows ?? []) {
      openings.push({ x: w.x, y: w.y, width: w.w ?? WIN_W, height: w.h ?? WIN_H });
    }
    for (const d of o.doors ?? []) {
      openings.push({ x: d.x, y: 0, width: d.w ?? DOOR_W, height: d.h ?? DOOR_H });
    }
    this.add(o.key, wallWithOpenings(length, o.height, t, openings, bevel), wallM);

    // Local→world for a point on the wall's face.
    const at = (lx: number, ly: number, lz = 0): THREE.Vector3 =>
      new THREE.Vector3(cx + ux * lx + nx * lz, baseY + ly, cz + uz * lx + nz * lz);

    // ── Window furniture ────────────────────────────────────────────────────
    const rng = this.rng;
    const decay = o.decay ?? 0;
    for (const w of o.windows ?? []) {
      const ww = w.w ?? WIN_W;
      const wh = w.h ?? WIN_H;
      const p = at(w.x, w.y);
      const m = trs(p.x, p.y, p.z, ry);
      if (!w.bare) this.add(o.key, Props.windowFrame(ww, wh, t), m);

      // Roll the opening's condition once, up front, so every branch below
      // agrees about what this window is.
      let state: WindowState = w.state ?? 'glazed';
      if (!w.state && !w.bare) {
        const roll = rng.next();
        if (roll < decay * 0.4) state = 'boarded';
        else if (roll < decay * 0.72) state = 'broken';
        else if (roll < decay) state = 'shuttered';
      }
      if (w.bare) state = 'open';
      if (w.glass === false && state === 'glazed') state = 'open';

      // An unlit recess behind the opening. Do this before the glazing so the
      // pane reads as a reflection *over* darkness, which is what a real window
      // looks like from outside on a bright day.
      if (o.blind && state !== 'boarded') {
        const rp = at(w.x, w.y + wh * 0.5, -0.04);
        this.add('interiorDark', Props.windowRecess(ww * 0.99, wh * 0.99, 0.62), trs(rp.x, rp.y, rp.z, ry));
      }

      if (state === 'boarded') {
        const bp = at(w.x, w.y + wh * 0.5, t * 0.5 - 0.03);
        this.add('wood', Props.windowBoards(ww, wh, rng), trs(bp.x, bp.y, bp.z, ry));
      } else if (state === 'broken') {
        const gp = at(w.x, w.y + wh * 0.5, 0.01);
        this.add('glass', Props.brokenGlass(ww * 0.96, wh * 0.96, rng), trs(gp.x, gp.y, gp.z, ry));
      } else if (state === 'glazed' || state === 'shuttered') {
        this.add('glass', Props.windowGlass(ww * 0.98, wh * 0.98), m);
      }

      if (w.shutter) {
        // Shutters live where the saturated accent lives. Their *angles* are
        // the point: two leaves latched flat at every window on a terrace reads
        // as a decal sheet, so each leaf gets its own swing, and a shuttered
        // window closes both leaves right across the opening.
        const key = w.shutter === 'teal' ? 'accentTeal' : 'accentOchre';
        const leafW = ww * 0.54;
        for (const sx of [-1, 1] as const) {
          // Closed: leaf swung flat across the opening. Open: folded back
          // against the facade at a per-leaf angle, so no two are identical.
          // One leaf in eight is left hanging half-open, which is the detail
          // that actually sells the row.
          const swing = state === 'shuttered'
            ? 0
            : sx * (rng.chance(0.12) ? rng.range(1.0, 1.9) : rng.range(2.45, 2.95));
          const hinge = at(w.x + sx * (ww * 0.5 + 0.05), w.y + wh * 0.5, t * 0.5 + 0.04);
          this.add(key, Props.shutterLeaf(leafW, wh * 0.97, swing, sx), trs(hinge.x, hinge.y, hinge.z, ry));
        }
      }
    }

    // ── Doors ───────────────────────────────────────────────────────────────
    for (const d of o.doors ?? []) {
      const dw = d.w ?? DOOR_W;
      const dh = d.h ?? DOOR_H;
      const p = at(d.x, 0);
      this.add(o.key, Props.doorFrame(dw, dh, t), trs(p.x, p.y, p.z, ry));
      if (d.leaf !== undefined) {
        const key = d.accent === 'ochre' ? 'accentOchre' : 'accentTeal';
        const lp = at(d.x, 0, t * 0.35);
        this.add(key, Props.doorLeaf(dw * 0.96, dh * 0.98, d.leaf), trs(lp.x, lp.y, lp.z, ry));
      }
      // Doorways are where people funnel, so doorways are where the wall is
      // shot up. Bullet scars either side of every entrance.
      const sp = at(d.x + (this.rng.chance(0.5) ? 1 : -1) * (dw * 0.5 + 0.55), 1.35, t * 0.5);
      this.grunge(
        sp,
        new THREE.Vector3(nx, 0, nz),
        'bulletscar',
        this.rng.range(1.1, 1.9),
        this.rng.range(0, Math.PI * 2),
        this.rng.range(0.55, 0.95),
      );
    }

    // ── Pilasters, coping, drift ────────────────────────────────────────────
    if (o.pilasters && o.pilasters > 0) {
      const n = Math.max(2, Math.round(length / o.pilasters));
      const sides = o.pilastersBothSides ? [1, -1] : [1];
      for (let i = 0; i <= n; i++) {
        const lx = -length * 0.5 + (length * i) / n;
        for (const s of sides) {
          const p = at(lx, 0, s * (t * 0.5 + 0.08));
          this.add(o.key, chamferedBox(0.42, o.height * 1.01, 0.18, 0.03), trs(p.x, p.y + o.height * 0.5, p.z, ry));
          // A splayed foot at the base of each pier: a buttress reads as
          // structure, a plain rectangle reads as a stripe.
          const f = at(lx, 0, s * (t * 0.5 + 0.13));
          this.add(o.key, chamferedBox(0.56, 0.34, 0.28, 0.03), trs(f.x, f.y + 0.17, f.z, ry));
        }
      }
    }
    if (o.coping) {
      this.add(o.key, Props.coping(length + 0.16, t), trs(cx, baseY + o.height + 0.04, cz, ry));
    }

    // ── Horizontal relief: plinth, string courses, cornice ──────────────────
    //
    // Three projecting bands is all it takes. They cost ~130 triangles per wall,
    // merge into the wall's own bucket so they are free in draw calls, and they
    // are the difference between an extruded rectangle and a building: each one
    // throws a hard horizontal shadow that gives the eye a scale reference and
    // splits the elevation into storeys.
    //
    // Every band is punched around any opening it would otherwise cross —
    // running a string course through a doorway head is the classic tell that
    // the detail was applied by a script rather than drawn by a person.
    const openingSpan = (ox: number, ow: number, pad: number): [number, number] => [ox - ow * 0.5 - pad, ox + ow * 0.5 + pad];

    if (o.plinth && o.plinth > 0) {
      const ph = o.plinth;
      const gaps: Array<[number, number]> = (o.doors ?? []).map((d) => openingSpan(d.x, d.w ?? DOOR_W, 0.16));
      for (const w of o.windows ?? []) {
        if (w.y < ph + 0.05) gaps.push(openingSpan(w.x, (w.w ?? WIN_W) + 0.5, 0.05));
      }
      const key = o.plinthKey ?? o.key;
      for (const [a, b] of Level.gapSpans(length, gaps)) {
        const seg = b - a;
        if (seg < 0.25) continue;
        const p = at((a + b) * 0.5, ph * 0.5);
        this.add(key, chamferedBox(seg, ph, t + 0.15, 0.03), trs(p.x, p.y, p.z, ry));
      }
    }

    for (const by of o.bands ?? []) {
      if (by <= 0.1 || by >= o.height - 0.1) continue;
      const gaps: Array<[number, number]> = [];
      for (const d of o.doors ?? []) {
        if ((d.h ?? DOOR_H) + 0.34 > by - 0.09) gaps.push(openingSpan(d.x, (d.w ?? DOOR_W) + 0.6, 0.06));
      }
      for (const w of o.windows ?? []) {
        const wh = w.h ?? WIN_H;
        if (by + 0.09 > w.y - 0.24 && by - 0.09 < w.y + wh + 0.34) {
          gaps.push(openingSpan(w.x, (w.w ?? WIN_W) + 0.62, 0.05));
        }
      }
      for (const [a, b] of Level.gapSpans(length, gaps)) {
        const seg = b - a;
        if (seg < 0.3) continue;
        const p = at((a + b) * 0.5, by);
        this.add(o.key, chamferedBox(seg, 0.15, t + 0.13, 0.025), trs(p.x, p.y, p.z, ry));
      }
    }

    if (o.cornice) {
      const c0 = at(0, o.height - 0.15);
      this.add(o.key, chamferedBox(length + 0.1, 0.19, t + 0.28, 0.025), trs(c0.x, c0.y, c0.z, ry));
      const c1 = at(0, o.height - 0.32);
      this.add(o.key, chamferedBox(length + 0.06, 0.11, t + 0.15, 0.02), trs(c1.x, c1.y, c1.z, ry));
    }

    if (o.dress && o.dress > 0) this.dressFacade(o, length, t, baseY, ry, at);
    if (o.signage) this.signFacade(o, t, ry, at);

    if (o.drift && o.drift > 0) {
      const p = at(0, 0, t * 0.5 + 0.45);
      const g = sandDrift(length * 0.92, o.drift, 1.1, this.rng, 14);
      // The drift's wall side is +Z locally, so face it back at the wall.
      this.add('sand', g, trs(p.x, 0.005, p.z, ry + Math.PI));
    }
  }

  /**
   * Spans along a wall of `length` that avoid a set of gaps, in wall-local X.
   *
   * Used to punch plinths and string courses around doors and windows. Static
   * because it is pure arithmetic and gets called a few hundred times at build.
   */
  private static gapSpans(length: number, gaps: Array<[number, number]>): Array<[number, number]> {
    const half = length * 0.5;
    const sorted = gaps
      .map(([a, b]) => [Math.max(-half, a), Math.min(half, b)] as [number, number])
      .filter(([a, b]) => b > a)
      .sort((p, q) => p[0] - q[0]);
    const out: Array<[number, number]> = [];
    let cursor = -half;
    for (const [a, b] of sorted) {
      if (a > cursor + 0.05) out.push([cursor, a]);
      cursor = Math.max(cursor, b);
    }
    if (half > cursor + 0.05) out.push([cursor, half]);
    return out;
  }

  /** Local→world helper signature shared by the facade dressing passes. */
  private dressFacade(
    o: FacadeOpts,
    length: number,
    t: number,
    baseY: number,
    ry: number,
    at: (lx: number, ly: number, lz?: number) => THREE.Vector3,
  ): void {
    const rng = this.rng;
    const d = o.dress ?? 1;
    const face = t * 0.5;

    // ── Rainwater goods ─────────────────────────────────────────────────────
    // Downpipes are merged rather than instanced: they need per-run heights and
    // the whole run is only ~240 triangles, so a merge into the existing rust
    // bucket costs nothing at all in draw calls.
    const pipes = Math.max(1, Math.round((length / 13) * d));
    for (let i = 0; i < pipes; i++) {
      const lx = -length * 0.5 + (length * (i + 0.5)) / pipes + rng.range(-1.2, 1.2);
      const p = at(lx, 0, face + 0.055);
      this.add('rust', Props.downpipe(o.height + 0.24), trs(p.x, baseY, p.z, ry));
      // Every downpipe ends somewhere, and the splash stain under it is the
      // cheapest bit of storytelling in the map.
      this.grunge(
        at(lx, 0.02, face + 0.5),
        new THREE.Vector3(0, 1, 0),
        'stain',
        rng.range(0.9, 1.7),
        rng.range(0, Math.PI),
        rng.range(0.35, 0.6),
      );
    }

    // ── Surface-run conduit: a spine down the wall with horizontal spurs ────
    const runs = Math.max(1, Math.round((length / 20) * d));
    for (let i = 0; i < runs; i++) {
      const lx = -length * 0.5 + (length * (i + 0.7)) / runs;
      const top = Math.min(o.height - 0.3, 2.4 + rng.range(0, 1.6));
      const a = at(lx, 0.5, face + 0.03);
      const b = at(lx, top, face + 0.03);
      this.add('rust', Props.cable(a, b, 0.0, 0.017));
      const spur = rng.range(1.4, 3.4) * rng.sign();
      this.add('rust', Props.cable(b, at(lx + spur, top + rng.range(-0.2, 0.2), face + 0.03), 0.02, 0.015));
      this.inst('conduitBox', 'metal', () => Props.conduitBox(), false)
        .pushTRS(at(lx, rng.range(1.5, 2.1), face + 0.005), ry);
    }

    // ── Plant: AC condensers, dishes, vents ─────────────────────────────────
    const acs = Math.round((length / 16) * d * 1.6);
    for (let i = 0; i < acs; i++) {
      const lx = rng.range(-length * 0.45, length * 0.45);
      const y = rng.chance(0.55) ? rng.range(2.5, 3.1) : rng.range(STOREY + 2.4, Math.max(STOREY + 2.5, o.height - 0.8));
      if (y > o.height - 0.55) continue;
      this.inst('acUnit', 'metal', () => Props.acUnit()[0].geo)
        .pushTRS(at(lx, y, face + 0.21), ry);
      // Bracket shelf under it, so it is carried rather than glued on.
      this.add('rust', chamferedBox(0.9, 0.05, 0.44, 0.012), trs(...at(lx, y - 0.03, face + 0.2).toArray(), ry));
      this.grunge(
        at(lx, y - 0.45, face + 0.01),
        new THREE.Vector3(Math.sin(ry), 0, Math.cos(ry)),
        'stain',
        rng.range(0.7, 1.3),
        0,
        0.5,
      );
    }
    if (d > 0.5 && length > 9) {
      for (let i = 0; i < Math.max(1, Math.round(length / 26)); i++) {
        const lx = rng.range(-length * 0.4, length * 0.4);
        this.inst('dish', 'metal', () => Props.satelliteDish(0.55))
          .pushTRS(at(lx, rng.range(3.2, Math.max(3.4, o.height - 1.2)), face + 0.42), ry + rng.range(-0.6, 0.6));
      }
    }
    for (let i = 0; i < Math.round((length / 22) * d); i++) {
      this.inst('vent', 'metal', () => Props.ventGrille(0.62, 0.44), false)
        .pushTRS(at(rng.range(-length * 0.42, length * 0.42), rng.range(2.2, 2.8), face + 0.03), ry);
    }

    // ── Lighting over doorways ──────────────────────────────────────────────
    for (const door of o.doors ?? []) {
      if (!rng.chance(0.7 * d)) continue;
      const dh = door.h ?? DOOR_H;
      this.inst('wallLamp', 'metal', () => Props.wallLamp(), false)
        .pushTRS(at(door.x + (door.w ?? DOOR_W) * 0.5 + 0.35, dh + 0.62, face + 0.01), ry);
    }
  }

  /**
   * Painted shop signage over doorways.
   *
   * Two things fight the compound's monotone: painted joinery, and painted
   * *boards*. A board is a rectangle of flat saturated colour a metre and a half
   * wide, hung at eye level where the player is already looking, and it is the
   * cheapest colour in the map by a wide margin.
   */
  private signFacade(
    o: FacadeOpts,
    t: number,
    ry: number,
    at: (lx: number, ly: number, lz?: number) => THREE.Vector3,
  ): void {
    const rng = this.rng;
    for (const door of o.doors ?? []) {
      if (!rng.chance(0.75)) continue;
      const dh = door.h ?? DOOR_H;
      const w = rng.range(1.9, 3.0);
      const key = rng.pick(['accentTeal', 'accentOchre', 'paintRust'] as const);
      this.add(key, Props.signBoard(w, rng.range(0.55, 0.8)), trs(...at(door.x, dh + 0.5, t * 0.5).toArray(), ry));
      // A fabric awning over the same opening. Cloth is the only thing in the
      // compound allowed to curve, it throws a soft diagonal shadow across the
      // shopfront, and it is what separates a market street from a corridor.
      if (rng.chance(0.55)) {
        const ap = at(door.x, 0, t * 0.5 - 0.05);
        // `ry` already maps the prop's local +Z onto the wall's outward
        // normal, and the awning projects along +Z, so no extra half-turn.
        // Hung clear above the sign board so the two never intersect.
        this.addParts(
          Props.awning(w * rng.range(0.9, 1.15), rng.range(1.25, 1.75), dh + rng.range(1.05, 1.4), rng),
          trs(ap.x, 0, ap.z, ry),
        );
      }
      // A second, smaller board hung perpendicular so it reads down the street
      // rather than only from straight on.
      if (rng.chance(0.4)) {
        const p = at(door.x + rng.range(-0.9, 0.9), dh + 0.5, t * 0.5 + 0.62);
        this.add(key, Props.signBoard(rng.range(0.7, 1.1), 0.4), trs(p.x, p.y, p.z, ry + Math.PI / 2));
        this.add('rust', chamferedBox(0.05, 0.05, 0.72, 0.012), trs(p.x, p.y + 0.28, p.z, ry, 1, 0, 0));
      }
    }
  }

  /** Regularly spaced window definitions along a wall of a given length. */
  private windowRow(
    length: number,
    count: number,
    y: number,
    opts: { w?: number; h?: number; shutterEvery?: number; shutter?: 'teal' | 'ochre'; skip?: number[] } = {},
  ): FacadeWindow[] {
    const out: FacadeWindow[] = [];
    const skip = new Set(opts.skip ?? []);
    for (let i = 0; i < count; i++) {
      if (skip.has(i)) continue;
      const x = -length * 0.5 + (length * (i + 1)) / (count + 1);
      const shutter =
        opts.shutter && opts.shutterEvery && i % opts.shutterEvery === 0 ? opts.shutter : undefined;
      out.push({ x, y, w: opts.w, h: opts.h, shutter });
    }
    return out;
  }

  /** Four outward-facing walls around a rectangular footprint. */
  private shell(
    r: Rect,
    key: string,
    baseY: number,
    height: number,
    faces: {
      west?: Partial<FacadeOpts>;
      south?: Partial<FacadeOpts>;
      east?: Partial<FacadeOpts>;
      north?: Partial<FacadeOpts>;
    } = {},
    common: Partial<FacadeOpts> = {},
  ): void {
    const base = { key, baseY, height, ...common };
    this.facade({ ...base, ax: r.x0, az: r.z0, bx: r.x0, bz: r.z1, ...faces.west } as FacadeOpts);
    this.facade({ ...base, ax: r.x0, az: r.z1, bx: r.x1, bz: r.z1, ...faces.south } as FacadeOpts);
    this.facade({ ...base, ax: r.x1, az: r.z1, bx: r.x1, bz: r.z0, ...faces.east } as FacadeOpts);
    this.facade({ ...base, ax: r.x1, az: r.z0, bx: r.x0, bz: r.z0, ...faces.north } as FacadeOpts);
  }

  /**
   * Quoin piers at the four corners of a footprint, running full height.
   *
   * Free in draw calls (they merge into the building's own bucket) and they fix
   * the single worst thing about a box: the naked 90-degree arris where two
   * elevations meet. With a pier on it the corner becomes two shadowed reveals
   * and the building gains a vertical order to hang its bands off.
   */
  private cornerQuoins(r: Rect, baseY: number, height: number, key: string): void {
    for (const [x, z] of [[r.x0, r.z0], [r.x0, r.z1], [r.x1, r.z1], [r.x1, r.z0]] as const) {
      this.add(key, Props.cornerQuoin(height, 0.58, 0.1), trs(x, baseY, z));
    }
  }

  /**
   * Juliet balconies hung under a scatter of upper-storey windows.
   *
   * `xs` are wall-local offsets along the a→b direction, matching `facade`.
   */
  private balconies(
    ax: number, az: number, bx: number, bz: number,
    y: number, xs: number[], t = WALL_T,
  ): void {
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz);
    if (len < 1e-4) return;
    const ux = dx / len;
    const uz = dz / len;
    const nx = -uz;
    const nz = ux;
    const ry = Math.atan2(nx, nz);
    const cx = (ax + bx) * 0.5;
    const cz = (az + bz) * 0.5;
    for (const lx of xs) {
      const px = cx + ux * lx + nx * (t * 0.5);
      const pz = cz + uz * lx + nz * (t * 0.5);
      this.addParts(Props.julietBalcony(1.75, 0.58), trs(px, y, pz, ry));
    }
  }

  /** A floor or roof slab covering a footprint, optionally with a void in it. */
  private slab(r: Rect, y: number, thickness = 0.28, hole?: Rect, key = 'concrete'): void {
    const put = (a: Rect): void => {
      const w = a.x1 - a.x0;
      const d = a.z1 - a.z0;
      if (w <= 0.05 || d <= 0.05) return;
      this.boxAt(key, w, thickness, d, (a.x0 + a.x1) * 0.5, y - thickness * 0.5, (a.z0 + a.z1) * 0.5, 0.03);
    };
    if (!hole) {
      put(r);
      return;
    }
    // Partition into up to four rectangles around the void.
    put({ x0: r.x0, x1: r.x1, z0: r.z0, z1: hole.z0 });
    put({ x0: r.x0, x1: r.x1, z0: hole.z1, z1: r.z1 });
    put({ x0: r.x0, x1: hole.x0, z0: hole.z0, z1: hole.z1 });
    put({ x0: hole.x1, x1: r.x1, z0: hole.z0, z1: hole.z1 });
  }

  /**
   * A parapet wall around a roof, with a coping cap, raised corner piers and
   * drainage scuppers.
   *
   * The piers matter more than they look. A parapet is a single unbroken
   * horizontal line 7 m up, and an unbroken horizontal line 7 m up is exactly
   * what makes a building read as an extruded rectangle. Punctuating it every
   * few metres with a 40 cm raised pier turns the roof edge into a rhythm, and
   * it is visible from every position on the map.
   */
  private parapetRing(r: Rect, y: number, height = PARAPET, key = 'plaster'): void {
    const t = 0.22;
    const opts = { key, baseY: y, height, thickness: t, coping: true, bevel: 0.025 };
    this.facade({ ...opts, ax: r.x0, az: r.z0, bx: r.x0, bz: r.z1 });
    this.facade({ ...opts, ax: r.x0, az: r.z1, bx: r.x1, bz: r.z1 });
    this.facade({ ...opts, ax: r.x1, az: r.z1, bx: r.x1, bz: r.z0 });
    this.facade({ ...opts, ax: r.x1, az: r.z0, bx: r.x0, bz: r.z0 });

    const rng = this.rng;
    const top = y + height + 0.08;
    const piers = this.inst('parapetPier', key, () => Props.parapetPier(0.52, 0.44, 0.3));
    // Corner piers always; intermediate piers on a ~5 m rhythm with a little
    // jitter so the row is not a metronome.
    for (const [px, pz] of [
      [r.x0, r.z0], [r.x0, r.z1], [r.x1, r.z1], [r.x1, r.z0],
    ] as const) {
      piers.pushTRS(new THREE.Vector3(px, top, pz), 0, new THREE.Vector3(1, rng.range(1.0, 1.5), 1));
    }
    for (const [ax, az, bx, bz] of [
      [r.x0, r.z0, r.x0, r.z1], [r.x1, r.z0, r.x1, r.z1],
      [r.x0, r.z0, r.x1, r.z0], [r.x0, r.z1, r.x1, r.z1],
    ] as const) {
      const len = Math.hypot(bx - ax, bz - az);
      const n = Math.floor(len / 5.2);
      const along = Math.abs(bx - ax) > Math.abs(bz - az);
      for (let i = 1; i <= n; i++) {
        const f = i / (n + 1) + rng.range(-0.03, 0.03);
        piers.pushTRS(
          new THREE.Vector3(ax + (bx - ax) * f, top, az + (bz - az) * f),
          along ? 0 : Math.PI / 2,
          new THREE.Vector3(1, rng.range(0.8, 1.25), 1),
        );
      }
      // A scupper or two per run, projecting outward through the parapet foot.
      const outward = new THREE.Vector3(-(bz - az) / len, 0, (bx - ax) / len);
      for (let i = 0; i < Math.max(1, Math.round(len / 14)); i++) {
        const f = rng.range(0.2, 0.8);
        const p = new THREE.Vector3(ax + (bx - ax) * f, y + 0.14, az + (bz - az) * f);
        this.inst('scupper', 'rust', () => Props.wallScupper(), false)
          .pushTRS(p.clone().addScaledVector(outward, 0.1), Math.atan2(outward.x, outward.z));
        // Every scupper has streaked the wall below it for twenty years.
        this.grunge(
          p.clone().addScaledVector(outward, 0.13).setY(y - 1.4),
          outward.clone(),
          'stain',
          rng.range(1.6, 2.8),
          0,
          rng.range(0.45, 0.7),
        );
      }
    }
  }

  /**
   * Roof dressing: water tanks, AC condensers, satellite dishes, vent stacks
   * and a stray pile of blocks.
   *
   * Roofs are a lane in this map, so they must be as readable and as full of
   * cover as the ground. They also form the compound's skyline, and a skyline
   * of flat empty rectangles is the fastest way to make a city look unfinished.
   */
  private roofClutter(r: Rect, y: number, density = 1): void {
    const rng = this.rng;
    const cx = (r.x0 + r.x1) * 0.5;
    const cz = (r.z0 + r.z1) * 0.5;
    const w = r.x1 - r.x0 - 2.6;
    const d = r.z1 - r.z0 - 2.6;
    if (w <= 1 || d <= 1) return;
    /** A random point on the deck, kept clear of the parapet. */
    const spot = (inset = 0.42): THREE.Vector3 =>
      new THREE.Vector3(cx + rng.range(-w * inset, w * inset), y, cz + rng.range(-d * inset, d * inset));

    // ── Stair bulkhead ──────────────────────────────────────────────────────
    // The biggest thing on any roof and the strongest silhouette break in the
    // whole skyline: a 2.4 m hut standing above a 1.05 m parapet is the only
    // rooftop object that shows from ground level across the map. It is also
    // real hard cover on an otherwise open deck.
    if (Math.min(w, d) > 4.5 && density > 0.65) {
      const p = spot(0.3);
      this.addParts(
        Props.stairBulkhead(rng.range(2.0, 2.6), rng.range(1.9, 2.4), rng.range(2.2, 2.6),
          rng.chance(0.5) ? 'accentTeal' : 'paintRust'),
        trs(p.x, y, p.z, rng.pick([0, Math.PI * 0.5, Math.PI, -Math.PI * 0.5])),
      );
    }

    // ── Water tanks ─────────────────────────────────────────────────────────
    const tanks = Math.max(1, Math.round(density * 2));
    for (let i = 0; i < tanks; i++) {
      const p = spot(0.4);
      this.addParts(Props.waterTank(rng.range(0.48, 0.66), rng.range(0.95, 1.3)), trs(p.x, y, p.z, rng.range(0, 3.1)));
    }

    // ── AC condensers ───────────────────────────────────────────────────────
    for (let i = 0; i < Math.round(density * 3); i++) {
      const p = spot(0.45);
      this.inst('acUnit', 'metal', () => Props.acUnit()[0].geo).pushTRS(p, rng.range(0, Math.PI * 2));
      // Condensate stain under every unit — free storytelling.
      this.grunge(
        new THREE.Vector3(p.x, y + 0.012, p.z + 0.5),
        new THREE.Vector3(0, 1, 0),
        'stain',
        rng.range(0.8, 1.5),
        rng.range(0, Math.PI * 2),
        0.65,
      );
    }

    // ── Aerials, dishes and masts: the fine end of the silhouette ───────────
    // These are almost free (a hundred triangles each, instanced) and they do
    // more for the skyline than anything else up here, because they are the
    // only things that break the roof line with something *thin*.
    const aerials = this.inst('tvAerial', 'metal', () => Props.tvAerial(2.4), false);
    for (let i = 0; i < Math.max(1, Math.round(density * 2.4)); i++) {
      const p = spot(0.46);
      aerials.pushTRS(p, rng.range(0, Math.PI * 2), new THREE.Vector3(1, rng.range(0.75, 1.5), 1));
    }
    if (density > 0.6) {
      const dishes = this.inst('dish', 'metal', () => Props.satelliteDish(0.55));
      for (let i = 0; i < Math.max(1, Math.round(density * 1.6)); i++) {
        dishes.pushTRS(new THREE.Vector3(spot(0.44).x, y + rng.range(0.9, 1.5), r.z0 + rng.range(0.5, 1.4)), rng.range(-1.1, 1.1));
      }
      this.add('metal', Props.antenna(rng.range(2.2, 3.6)), trs(spot(0.42).x, y, cz + d * 0.42));
    }

    // ── Chimney / flue stack ────────────────────────────────────────────────
    if (rng.chance(0.55 * density)) {
      const p = spot(0.38);
      this.addParts(Props.chimneyStack(rng.range(1.1, 2.0), rng.range(0.42, 0.6)), trs(p.x, y, p.z, rng.range(0, 1.5)));
    }

    // ── Vent stacks and a roof hatch ────────────────────────────────────────
    for (let i = 0; i < Math.round(density * 3); i++) {
      const p = spot(0.45);
      const h = rng.range(0.5, 1.0);
      this.add('rust', chamferedCylinder(0.09, h, 0.02, 10), trs(p.x, y + h * 0.5, p.z));
      this.add('rust', chamferedCylinder(0.13, 0.1, 0.02, 10), trs(p.x, y + h + 0.06, p.z));
    }

    // ── Building materials abandoned mid-job ────────────────────────────────
    if (rng.chance(0.8)) {
      const p = spot(0.35);
      const blocks = this.inst('cinder', 'concrete', () => Props.cinderBlock());
      const rows = rng.int(5, 12);
      for (let i = 0; i < rows; i++) {
        blocks.pushTRS(
          new THREE.Vector3(p.x + rng.range(-0.25, 0.25), 0.215 * Math.floor(i / 2) + y, p.z + rng.range(-0.25, 0.25)),
          rng.chance(0.5) ? 0 : Math.PI / 2,
        );
      }
      this.inst('pallet', 'wood', () => Props.pallet(), false)
        .pushTRS(new THREE.Vector3(p.x + rng.range(0.9, 1.6), y, p.z + rng.range(-0.8, 0.8)), rng.range(0, 3.1));
      // A tarp thrown over the stack: the compound's only blue.
      if (rng.chance(0.4)) {
        this.add('tarpBlue', Props.tarpSheet(1.3, 1.0, 0.62, rng), trs(p.x + rng.range(-0.2, 0.2), y, p.z));
      }
    }
    if (rng.chance(0.5)) {
      const p = spot(0.3);
      this.addParts(Props.rubblePile(rng.range(0.8, 1.4), rng.int(10, 20), rng), trs(p.x, y, p.z));
    }
    for (let i = 0; i < Math.round(density * 2); i++) {
      const p = spot(0.44);
      this.inst('drumRust', 'rust', () => Props.oilDrum()).pushTRS(p, rng.range(0, Math.PI * 2));
    }

    // ── Laundry lines strung between the parapets ───────────────────────────
    if (Math.min(w, d) > 4 && rng.chance(0.75)) {
      const z = cz + rng.range(-d * 0.3, d * 0.3);
      this.addParts(
        Props.laundryLine(
          new THREE.Vector3(r.x0 + 1.0, y + rng.range(1.5, 2.0), z),
          new THREE.Vector3(r.x1 - 1.0, y + rng.range(1.5, 2.0), z + rng.range(-1.5, 1.5)),
          rng,
        ),
      );
      // The posts the line is tied to.
      for (const px of [r.x0 + 1.0, r.x1 - 1.0]) {
        this.add('rust', chamferedBox(0.07, 2.0, 0.07, 0.015), trs(px, y + 1.0, z));
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Perimeter
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The compound wall on the north and south edges.
   *
   * The map's boundary is *always* a real object: a 5.2 m compound wall here, a
   * continuous terrace of buildings east and west. There is no point anywhere
   * on the map where the player walks into nothing. Invisible walls in open
   * ground are the fastest way to destroy the illusion that a level is a place.
   */
  private buildPerimeter(): void {
    const H = 5.2;
    const T = 0.6;

    for (const [z, outward] of [[-EDGE, -1], [EDGE, 1]] as const) {
      // Facing outward: a→b chosen so the normal points away from the compound.
      const a = outward < 0 ? new THREE.Vector2(EDGE, z) : new THREE.Vector2(-EDGE, z);
      const b = outward < 0 ? new THREE.Vector2(-EDGE, z) : new THREE.Vector2(EDGE, z);
      this.facade({
        key: 'plasterAlt',
        ax: a.x, az: a.y, bx: b.x, bz: b.y,
        height: H, thickness: T,
        pilasters: 6,
        pilastersBothSides: true,
        coping: true,
        // A painted base course and a cornice under the coping. Both wrap the
        // full wall thickness, so the compound's most-seen surface picks up two
        // hard horizontal shadow lines for about 90 triangles.
        plinth: 0.62,
        plinthKey: 'plinthPaint',
        cornice: true,
      });
      // Buttresses and a drift on the inward side, which is the side the player
      // actually sees for 99% of the match.
      this.driftAlong(b.x, b.y - outward * 0.45, a.x, a.y - outward * 0.45, 0.9, 1.5);
      for (let i = -6; i <= 6; i++) {
        const x = i * 7;
        this.boxAt('plasterAlt', 0.5, H * 0.86, 0.5, x, H * 0.43, z - outward * 0.5, 0.04);
      }
    }

    // The blocked north gateway: an arched opening bricked shut and reinforced
    // with gabion. It reads as "the way out, closed", which is exactly what a
    // map boundary wants to say.
    const gate = { x: 0, z: -EDGE };
    for (const sx of [-1, 1]) {
      this.boxAt('concrete', 0.9, 6.4, 1.3, gate.x + sx * 3.0, 3.2, gate.z, 0.05);
      this.boxAt('concrete', 0.7, 0.7, 1.5, gate.x + sx * 3.0, 6.6, gate.z, 0.06);
    }
    this.boxAt('concrete', 6.9, 0.85, 1.3, gate.x, 6.0, gate.z, 0.05);
    // Arch head, faked with a stepped corbel — cheaper than a real arch and
    // reads correctly in silhouette.
    for (let i = 0; i < 4; i++) {
      const w = 5.1 - i * 1.0;
      this.boxAt('concrete', w, 0.3, 1.2, gate.x, 4.4 + i * 0.3, gate.z, 0.04);
    }
    const gabion = this.inst('gabion', 'sand', () => Props.gabion());
    for (let r = 0; r < 2; r++) {
      for (let i = -2; i <= 2; i++) {
        gabion.pushTRS(new THREE.Vector3(gate.x + i * 1.09, r * 1.07, gate.z + 0.9), 0);
      }
    }
    this.addParts(Props.rubblePile(2.4, 26, this.rng), trs(gate.x + 1.2, 0, gate.z + 2.2));

    // A rocky berm outside the south wall, visible over the coping. Depth cues
    // beyond the play space cost almost nothing and stop the map feeling like a
    // diorama sitting on a table.
    const rocks = this.inst('bergRock', 'rock', () => rockGeometry(1, this.rng, 0), true);
    for (let i = 0; i < 46; i++) {
      const x = this.rng.range(-EDGE - 8, EDGE + 8);
      const zs = this.rng.chance(0.5) ? 1 : -1;
      const z = zs * this.rng.range(EDGE + 2.5, EDGE + 16);
      const s = this.rng.range(1.1, 3.6);
      rocks.pushTRS(
        new THREE.Vector3(x, this.groundHeight(x, z) + this.rng.range(-0.5, 0.6), z),
        this.rng.range(0, Math.PI * 2),
        new THREE.Vector3(s * this.rng.range(0.8, 1.4), s * this.rng.range(0.5, 1.0), s * this.rng.range(0.8, 1.4)),
      );
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Terraces — the east and west boundary buildings
  // ═══════════════════════════════════════════════════════════════════════════

  private buildTerraces(): void {
    // ── West terrace: the alley's west wall ─────────────────────────────────
    // One long block, but broken into three visual bays with different plaster
    // tones and different window rhythms so it does not read as one 84 m slab.
    const wt = WEST_TERRACE;
    const bays: Array<{ z0: number; z1: number; key: string }> = [
      { z0: wt.z0, z1: -16, key: 'plaster' },
      { z0: -16, z1: 12, key: 'plasterPale' },
      { z0: 12, z1: wt.z1, key: 'plasterAlt' },
    ];
    for (const bay of bays) {
      const r: Rect = { x0: wt.x0, x1: wt.x1, z0: bay.z0, z1: bay.z1 };
      const len = bay.z1 - bay.z0;
      // Only the alley-facing (east) side is detailed: nothing can see the
      // other three, and detail you cannot see is triangles you have wasted.
      this.facade({
        key: bay.key,
        ax: wt.x1, az: bay.z1, bx: wt.x1, bz: bay.z0,
        height: ROOF_2, thickness: WALL_T,
        windows: [
          ...this.windowRow(len, Math.max(2, Math.round(len / 5)), WIN_SILL, { shutter: 'teal', shutterEvery: 3 }),
          ...this.windowRow(len, Math.max(2, Math.round(len / 5)), STOREY + WIN_SILL, { shutter: 'ochre', shutterEvery: 4 }),
        ],
        doors: [
          { x: -len * 0.28, leaf: 0.35, accent: 'teal' },
          { x: len * 0.3, leaf: -1.1, accent: 'ochre' },
        ],
        drift: 0.42,
        // The alley's only elevation, seen from 3 m away for the whole match,
        // so it gets the full treatment: painted plinth, floor band, cornice,
        // recessed interiors behind the glass, service dressing and signage.
        plinth: 0.55,
        plinthKey: 'plinthPaint',
        bands: [STOREY - 0.15],
        cornice: true,
        blind: true,
        decay: 0.3,
        dress: 1.0,
        signage: true,
      });
      // The other three faces are plain shells that close the silhouette. The
      // back (x0) face is outside the map entirely; the two end walls are only
      // ever glimpsed, so they get the cheap relief (plinth + cornice) and none
      // of the expensive dressing.
      this.facade({ key: bay.key, ax: wt.x0, az: bay.z0, bx: wt.x0, bz: bay.z1, height: ROOF_2, thickness: WALL_T, cornice: true });
      this.facade({ key: bay.key, ax: wt.x0, az: bay.z1, bx: wt.x1, bz: bay.z1, height: ROOF_2, thickness: WALL_T, plinth: 0.55, plinthKey: 'plinthPaint', cornice: true });
      this.facade({ key: bay.key, ax: wt.x1, az: bay.z0, bx: wt.x0, bz: bay.z0, height: ROOF_2, thickness: WALL_T, plinth: 0.55, plinthKey: 'plinthPaint', cornice: true });
      this.slab(r, ROOF_2 + 0.28, 0.28);
      // Stagger the parapet height bay to bay. A terrace built over decades is
      // never one continuous line, and the step between bays is what stops 84 m
      // of building reading as a single extrusion.
      this.parapetRing(r, ROOF_2 + 0.28, PARAPET + this.rng.range(-0.12, 0.42), bay.key);
      this.roofClutter(r, ROOF_2 + 0.28, 0.85);
      // Quoins only on the two corners the alley can see. The back of the
      // terrace is outside the map and nothing there is ever rendered from a
      // position a player can occupy.
      for (const z of [bay.z0, bay.z1]) {
        this.add(bay.key, Props.cornerQuoin(ROOF_2, 0.58, 0.1), trs(wt.x1, 0, z));
      }
      // Balconies on a scatter of the upper windows: the alley is 6 m wide and
      // read almost entirely in silhouette against the sky, so anything that
      // projects from these walls is worth several times its triangle cost.
      this.balconies(
        wt.x1, bay.z1, wt.x1, bay.z0,
        STOREY + WIN_SILL - 0.06,
        [-len * 0.5 + len * 0.28, len * 0.5 - len * 0.24],
      );
    }

    // Alley greeble: pipe runs, gutters, downpipes, AC units and laundry.
    // This is what makes 6 m of corridor feel like a place rather than a hall.
    const alleyX = wt.x1 + 0.02;
    for (let i = 0; i < 6; i++) {
      const z = -38 + i * 15;
      this.addParts(
        Props.pipeRun(
          [
            new THREE.Vector3(alleyX + 0.14, 4.6, z - 6),
            new THREE.Vector3(alleyX + 0.14, 4.75, z),
            new THREE.Vector3(alleyX + 0.14, 4.6, z + 6),
          ],
          0.055,
          3,
          0.16,
        ),
      );
      this.add('rust', Props.downpipe(6.6), trs(alleyX + 0.1, 0, z + 3.2));
      this.add('rust', Props.gutter(11), trs(alleyX + 0.16, ROOF_2 + 0.18, z, Math.PI / 2));
      this.inst('acUnit', 'metal', () => Props.acUnit()[0].geo).pushTRS(
        new THREE.Vector3(alleyX + 0.22, this.rng.range(2.6, 4.6), z + this.rng.range(-4, 4)),
        Math.PI / 2,
      );
    }
    // Laundry strung across the alley at head height.
    for (let i = 0; i < 5; i++) {
      const z = -32 + i * 16 + this.rng.range(-2, 2);
      this.addParts(
        Props.laundryLine(
          new THREE.Vector3(wt.x1 + 0.1, this.rng.range(4.0, 5.2), z),
          new THREE.Vector3(-33.1, this.rng.range(3.6, 4.8), z + this.rng.range(-1.5, 1.5)),
          this.rng,
        ),
      );
    }
    // Fire escape at mid-alley: the alley's route onto the rooftop lane.
    this.add('metal', Props.ladder(ROOF_2 + 0.5, 0.48), trs(wt.x1 + 0.2, 0, 2, Math.PI / 2));
    this.boxAt('metal', 1.5, 0.08, 1.2, wt.x1 - 0.4, ROOF_2 + 0.3, 2, 0.02);

    // ── East terrace: the market street's east wall ─────────────────────────
    // Three sections; the northern one is deliberately one storey lower so its
    // roof lines up with the garage for a gangway.
    const sections: Array<{ r: Rect; roof: number; key: string; shops: boolean }> = [
      { r: EAST_TERRACE_N, roof: ROOF_GARAGE, key: 'plasterAlt', shops: false },
      { r: EAST_TERRACE_M, roof: ROOF_2, key: 'plaster', shops: true },
      { r: EAST_TERRACE_S, roof: ROOF_2, key: 'plasterPale', shops: true },
    ];
    for (const s of sections) {
      const len = s.r.z1 - s.r.z0;
      const upper = this.windowRow(len, Math.max(2, Math.round(len / 5)), STOREY + WIN_SILL, {
        shutter: 'teal',
        shutterEvery: 3,
      });
      this.facade({
        key: s.key,
        ax: s.r.x0, az: s.r.z0, bx: s.r.x0, bz: s.r.z1,
        height: s.roof, thickness: WALL_T,
        windows: s.roof > 5 ? upper : this.windowRow(len, Math.max(2, Math.round(len / 6)), WIN_SILL, { shutter: 'ochre', shutterEvery: 2 }),
        doors: s.shops ? [{ x: -len * 0.3, leaf: -0.5, accent: 'ochre' }, { x: len * 0.34, leaf: 0.9, accent: 'teal' }] : [{ x: 0, leaf: 0.2 }],
        drift: 0.38,
        // Market street shopfront elevation: the busiest wall in the map.
        plinth: 0.5,
        plinthKey: 'plinthPaint',
        bands: s.roof > 5 ? [STOREY - 0.15] : [],
        cornice: true,
        blind: true,
        decay: 0.26,
        dress: 1.0,
        signage: s.shops,
      });
      this.facade({ key: s.key, ax: s.r.x1, az: s.r.z1, bx: s.r.x1, bz: s.r.z0, height: s.roof, thickness: WALL_T, cornice: true });
      this.facade({ key: s.key, ax: s.r.x0, az: s.r.z1, bx: s.r.x1, bz: s.r.z1, height: s.roof, thickness: WALL_T, plinth: 0.5, plinthKey: 'plinthPaint', cornice: true });
      this.facade({ key: s.key, ax: s.r.x1, az: s.r.z0, bx: s.r.x0, bz: s.r.z0, height: s.roof, thickness: WALL_T, plinth: 0.5, plinthKey: 'plinthPaint', cornice: true });
      this.slab(s.r, s.roof + 0.28, 0.28);
      this.parapetRing(s.r, s.roof + 0.28, PARAPET + this.rng.range(-0.1, 0.4), s.key);
      this.roofClutter(s.r, s.roof + 0.28, 0.9);
      for (const z of [s.r.z0, s.r.z1]) {
        this.add(s.key, Props.cornerQuoin(s.roof, 0.58, 0.1), trs(s.r.x0, 0, z));
      }
      if (s.roof > 5) {
        this.balconies(
          s.r.x0, s.r.z0, s.r.x0, s.r.z1,
          STOREY + WIN_SILL - 0.06,
          [-len * 0.5 + len * 0.3, len * 0.5 - len * 0.28],
        );
      }

      if (s.shops) {
        // Shopfront awnings, projecting west over the market street.
        const n = Math.max(2, Math.round(len / 8));
        for (let i = 0; i < n; i++) {
          const z = s.r.z0 + (len * (i + 0.5)) / n;
          const parts = Props.awning(this.rng.range(2.4, 3.4), this.rng.range(1.4, 2.0), this.rng.range(2.5, 2.9), this.rng);
          // Awning frame projects along +Z locally; rotate so it points -X.
          this.addParts(parts, trs(s.r.x0 - 0.05, 0, z, -Math.PI / 2));
        }
      }
    }
    // Ladder onto the low north section, from the market street.
    this.add('metal', Props.ladder(ROOF_GARAGE + 0.5, 0.48), trs(EAST_TERRACE_N.x0 - 0.2, 0, -34, -Math.PI / 2));
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Buildings
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * RADIO HOUSE — the north-west power position.
   *
   * Two storeys, fully enterable, with an internal stair to the first floor and
   * a second flight to the roof. Its south balcony overlooks the courtyard from
   * 3.6 m, which is the map's strongest single angle — so it is deliberately
   * reachable three ways (interior stair, alley gangway from the west terrace
   * roof, and a drop from the roof) and exposed to the market-hall roof
   * 45 m away, so holding it is a commitment rather than a free win.
   */
  private buildRadioHouse(): void {
    const r = RADIO_HOUSE;
    const w = r.x1 - r.x0;
    const d = r.z1 - r.z0;

    // South facade: the courtyard-facing hero elevation.
    this.facade({
      key: 'plaster',
      ax: r.x0, az: r.z1, bx: r.x1, bz: r.z1,
      height: ROOF_2, thickness: WALL_T,
      windows: [
        { x: -3.6, y: WIN_SILL, shutter: 'teal' },
        { x: 3.6, y: WIN_SILL, shutter: 'teal' },
        { x: -3.6, y: STOREY + WIN_SILL, w: 1.3, shutter: 'ochre' },
        { x: 0, y: STOREY + WIN_SILL, w: 2.4, h: 1.5, glass: false },
        { x: 3.6, y: STOREY + WIN_SILL, w: 1.3, shutter: 'ochre' },
      ],
      doors: [{ x: 0, w: 1.4, leaf: 1.45, accent: 'teal' }],
      drift: 0.5,
      // The hero elevation of the map's strongest position. No `blind` — the
      // building is enterable and every one of these windows is a firing
      // position, so putting an unlit box behind the glass would take a real
      // gameplay affordance away for a visual one.
      plinth: 0.6,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.1],
      cornice: true,
      decay: 0.14,
      dress: 0.9,
      signage: true,
    });
    // West facade onto the alley.
    this.facade({
      key: 'plaster',
      ax: r.x0, az: r.z0, bx: r.x0, bz: r.z1,
      height: ROOF_2, thickness: WALL_T,
      windows: [
        ...this.windowRow(d, 2, WIN_SILL, { shutter: 'ochre', shutterEvery: 1 }),
        ...this.windowRow(d, 3, STOREY + WIN_SILL),
      ],
      drift: 0.55,
      plinth: 0.6,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.1],
      cornice: true,
      decay: 0.2,
      dress: 0.9,
    });
    // East facade toward the courtyard gate.
    this.facade({
      key: 'plaster',
      ax: r.x1, az: r.z1, bx: r.x1, bz: r.z0,
      height: ROOF_2, thickness: WALL_T,
      windows: [...this.windowRow(d, 2, WIN_SILL), ...this.windowRow(d, 3, STOREY + WIN_SILL, { shutter: 'teal', shutterEvery: 2 })],
      doors: [{ x: d * 0.28, leaf: 0 }],
      drift: 0.35,
      plinth: 0.6,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.1],
      cornice: true,
      decay: 0.2,
      dress: 0.8,
    });
    // North facade onto the service strip.
    this.facade({
      key: 'plaster',
      ax: r.x1, az: r.z0, bx: r.x0, bz: r.z0,
      height: ROOF_2, thickness: WALL_T,
      windows: this.windowRow(w, 3, STOREY + WIN_SILL, { w: 0.8, h: 0.7 }),
      drift: 0.7,
      plinth: 0.6,
      plinthKey: 'plinthPaint',
      cornice: true,
      blind: true,
      decay: 0.35,
      dress: 0.7,
    });

    // ── Interior ────────────────────────────────────────────────────────────
    const inner: Rect = { x0: r.x0 + WALL_T, x1: r.x1 - WALL_T, z0: r.z0 + WALL_T, z1: r.z1 - WALL_T };
    // Stair void in the first-floor slab, sized so a 1.8 m player clears it.
    const stairVoid: Rect = { x0: r.x1 - 6.6, x1: r.x1 - 1.2, z0: r.z0 + 0.8, z1: r.z0 + 3.0 };
    this.slab(inner, STOREY, 0.3, stairVoid);
    this.slab(inner, ROOF_2, 0.3, { x0: r.x1 - 6.6, x1: r.x1 - 1.2, z0: r.z1 - 3.2, z1: r.z1 - 1.0 });
    this.parapetRing(r, ROOF_2, PARAPET, 'plaster');
    this.roofClutter(r, ROOF_2, 1.2);
    this.cornerQuoins(r, 0, ROOF_2, 'plaster');
    // Two small balconies flanking the big first-floor opening, so the hero
    // elevation has something projecting at the height the eye reads it.
    this.balconies(r.x0, r.z1, r.x1, r.z1, STOREY + WIN_SILL - 0.06, [-3.6, 3.6]);

    // Ground-to-first stair, running north under the void.
    this.add(
      'concrete',
      staircase(17, STOREY, 4.6, 1.3),
      trs(r.x1 - 2.4, 0, r.z0 + 5.6, Math.PI),
    );
    // First-to-roof stair at the opposite end, so the two flights do not stack
    // into one straight shaft the whole building can be cleared through.
    this.add(
      'concrete',
      staircase(17, STOREY, 4.6, 1.3),
      trs(r.x1 - 2.4, STOREY, r.z1 - 5.8),
    );
    for (const y of [STOREY, ROOF_2]) {
      this.add('metal', Props.railing(5.4, 1.0), trs(r.x1 - 3.9, y, y === STOREY ? r.z0 + 3.1 : r.z1 - 3.3, 0));
    }

    // Two internal pillars and a dividing wall with a doorway: an open shell
    // is boring to fight in, and a single internal wall doubles the number of
    // ways a room can be cleared.
    for (const sx of [-1, 1]) this.boxAt('concrete', 0.42, STOREY, 0.42, r.x0 + w * 0.5 + sx * 3.2, STOREY * 0.5, r.z0 + d * 0.55, 0.03);
    this.facade({
      key: 'plasterPale',
      ax: r.x0 + WALL_T, az: r.z0 + 5.0, bx: r.x1 - 5.0, bz: r.z0 + 5.0,
      height: STOREY, thickness: 0.22,
      doors: [{ x: 1.4 }],
      extend: false,
    });
    this.facade({
      key: 'plasterPale',
      ax: r.x1 - 5.0, az: r.z0 + 5.4, bx: r.x0 + WALL_T, bz: r.z0 + 5.4,
      baseY: STOREY, height: STOREY, thickness: 0.22,
      doors: [{ x: -2.0 }],
      extend: false,
    });

    // Balcony over the courtyard.
    const by = STOREY + 0.2;
    this.boxAt('concrete', 5.6, 0.22, 1.8, r.x0 + w * 0.5, by, r.z1 + 0.75, 0.035);
    this.add('metal', Props.railing(5.6, 1.0), trs(r.x0 + w * 0.5, by + 0.11, r.z1 + 1.6));
    for (const sx of [-1, 1]) {
      this.add('metal', Props.railing(1.8, 1.0), trs(r.x0 + w * 0.5 + sx * 2.8, by + 0.11, r.z1 + 0.75, Math.PI / 2));
      // Corbel brackets, so the balcony is carried rather than floating.
      this.add('concrete', chamferedBox(0.2, 0.5, 1.5, 0.03), trs(r.x0 + w * 0.5 + sx * 2.2, by - 0.35, r.z1 + 0.7, 0, 1, -0.25));
    }
  }

  /**
   * GARAGE / WORKSHOP — the north-east anchor.
   *
   * Single storey with a wide roller-shutter opening facing the north yard, so
   * it is an obvious and inviting flank route that happens to be a killbox if
   * you enter it carelessly. Its roof, at 4.4 m, is one end of the northern
   * gangway.
   */
  private buildGarage(): void {
    const r = GARAGE;
    const w = r.x1 - r.x0;
    const d = r.z1 - r.z0;
    const H = ROOF_GARAGE;

    this.facade({
      key: 'plasterAlt',
      ax: r.x0, az: r.z1, bx: r.x1, bz: r.z1,
      height: H, thickness: WALL_T,
      windows: [{ x: -5.6, y: 2.3, w: 1.0, h: 0.9 }],
      doors: [{ x: 4.6, w: 4.4, h: 3.4 }, { x: -1.6, leaf: -0.9, accent: 'ochre' }],
      drift: 0.4,
      plinth: 0.52,
      plinthKey: 'plinthPaint',
      cornice: true,
      decay: 0.3,
      dress: 1.0,
      signage: true,
    });
    this.facade({
      key: 'plasterAlt',
      ax: r.x0, az: r.z0, bx: r.x0, bz: r.z1,
      height: H, thickness: WALL_T,
      windows: this.windowRow(d, 2, 2.4, { w: 1.0, h: 0.9 }),
      doors: [{ x: -d * 0.3, leaf: 0.4 }],
      drift: 0.4,
      plinth: 0.52,
      plinthKey: 'plinthPaint',
      cornice: true,
      decay: 0.3,
      dress: 0.8,
    });
    this.facade({
      key: 'plasterAlt',
      ax: r.x1, az: r.z1, bx: r.x1, bz: r.z0,
      height: H, thickness: WALL_T,
      windows: this.windowRow(d, 2, 2.4, { w: 1.0, h: 0.9 }),
      drift: 0.4,
      plinth: 0.52,
      plinthKey: 'plinthPaint',
      cornice: true,
      decay: 0.3,
      dress: 0.6,
    });
    this.facade({
      key: 'plasterAlt',
      ax: r.x1, az: r.z0, bx: r.x0, bz: r.z0,
      height: H, thickness: WALL_T,
      windows: this.windowRow(w, 4, 2.6, { w: 1.2, h: 1.0 }),
      drift: 0.85,
      plinth: 0.52,
      plinthKey: 'plinthPaint',
      cornice: true,
      blind: true,
      decay: 0.4,
      dress: 0.7,
    });

    // Half-raised roller shutter across the vehicle opening. Half-raised is the
    // right answer: it tells you the door works, it frames the interior, and it
    // forces a crouch to enter, which is a real tactical cost.
    this.addParts(Props.rollShutter(4.4, 3.4, 0.55), trs(r.x0 + 4.6 + w * 0.5, 0, r.z1 + 0.04));

    const inner: Rect = { x0: r.x0 + WALL_T, x1: r.x1 - WALL_T, z0: r.z0 + WALL_T, z1: r.z1 - WALL_T };
    this.slab(inner, H, 0.3);
    this.parapetRing(r, H, 0.95, 'plasterAlt');
    this.roofClutter(r, H, 0.8);
    this.cornerQuoins(r, 0, H, 'plasterAlt');
    // Roof ladder from the north service strip.
    this.add('metal', Props.ladder(H + 0.6, 0.48), trs(r.x1 - 2.2, 0, r.z0 - 0.22, Math.PI));

    // ── Workshop interior ───────────────────────────────────────────────────
    this.addParts(Props.burntTruck(), trs(r.x0 + 5.0, 0, r.z0 + 5.4, 0.22));
    // Benches down the back wall.
    for (let i = 0; i < 3; i++) {
      const x = r.x0 + 9.5 + i * 2.3;
      this.boxAt('wood', 2.1, 0.07, 0.75, x, 0.92, r.z0 + 0.95, 0.015);
      for (const sx of [-1, 1]) this.boxAt('metal', 0.07, 0.9, 0.07, x + sx * 0.9, 0.45, r.z0 + 0.95, 0.012);
      this.boxAt('wood', 2.1, 0.9, 0.06, x, 1.5, r.z0 + 0.62, 0.012); // pegboard
    }
    const drums = this.inst('drumRust', 'rust', () => Props.oilDrum());
    for (let i = 0; i < 7; i++) {
      drums.pushTRS(
        new THREE.Vector3(r.x0 + this.rng.range(1.2, 3.4), 0, r.z0 + this.rng.range(7.4, 9.6)),
        this.rng.range(0, Math.PI * 2),
      );
    }
    const tyres = this.inst('tyre', 'rubber', () => Props.tyre(), false);
    for (let s = 0; s < 3; s++) {
      const bx = r.x0 + 12.5 + s * 1.0;
      const bz = r.z0 + 8.6;
      for (let i = 0; i < this.rng.int(3, 6); i++) {
        tyres.pushTRS(new THREE.Vector3(bx, i * 0.19, bz), this.rng.range(0, Math.PI));
      }
    }
    // Inspection pit lip and a chain hoist gantry.
    this.boxAt('concrete', 4.6, 0.12, 1.5, r.x0 + 5.0, 0.06, r.z0 + 5.4, 0.02);
    for (const sx of [-1, 1]) this.boxAt('metal', 0.14, H - 0.4, 0.14, r.x0 + 5.0 + sx * 2.6, (H - 0.4) * 0.5, r.z0 + 5.4, 0.02);
    this.boxAt('metal', 5.6, 0.24, 0.16, r.x0 + 5.0, H - 0.5, r.z0 + 5.4, 0.03);
  }

  /**
   * MARKET HALL — the south-east anchor and the market street's west wall.
   *
   * Its first-floor west window is a 3 m opening that looks straight down the
   * courtyard's long axis: a genuine overwatch position, balanced by an
   * external stair on the far side, so anyone holding it can be flanked from
   * the market street without entering the building at all.
   */
  private buildMarketHall(): void {
    const r = MARKET_HALL;
    const w = r.x1 - r.x0;
    const d = r.z1 - r.z0;

    // West: courtyard elevation with the big overwatch opening.
    this.facade({
      key: 'plasterPale',
      ax: r.x0, az: r.z0, bx: r.x0, bz: r.z1,
      height: ROOF_2, thickness: WALL_T,
      windows: [
        { x: -4.2, y: WIN_SILL, shutter: 'ochre' },
        { x: 4.2, y: WIN_SILL, shutter: 'ochre' },
        { x: 0, y: STOREY + 0.85, w: 3.0, h: 1.7, glass: false },
        { x: -4.6, y: STOREY + WIN_SILL, shutter: 'teal' },
        { x: 4.6, y: STOREY + WIN_SILL, shutter: 'teal' },
      ],
      doors: [{ x: 0, w: 1.5, leaf: -1.4, accent: 'ochre' }],
      drift: 0.45,
      plinth: 0.58,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.12],
      cornice: true,
      decay: 0.16,
      dress: 0.9,
      signage: true,
    });
    // North: courtyard/market junction.
    this.facade({
      key: 'plasterPale',
      ax: r.x1, az: r.z0, bx: r.x0, bz: r.z0,
      height: ROOF_2, thickness: WALL_T,
      windows: [...this.windowRow(w, 3, WIN_SILL, { shutter: 'teal', shutterEvery: 2 }), ...this.windowRow(w, 4, STOREY + WIN_SILL)],
      doors: [{ x: -w * 0.26, w: 1.3, leaf: 0.9 }],
      drift: 0.4,
      plinth: 0.58,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.12],
      cornice: true,
      decay: 0.2,
      dress: 0.9,
      signage: true,
    });
    this.facade({
      key: 'plasterPale',
      ax: r.x1, az: r.z1, bx: r.x1, bz: r.z0,
      height: ROOF_2, thickness: WALL_T,
      windows: this.windowRow(d, 3, STOREY + WIN_SILL, { shutter: 'ochre', shutterEvery: 2 }),
      drift: 0.4,
      plinth: 0.58,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.12],
      cornice: true,
      decay: 0.24,
      dress: 0.9,
    });
    this.facade({
      key: 'plasterPale',
      ax: r.x0, az: r.z1, bx: r.x1, bz: r.z1,
      height: ROOF_2, thickness: WALL_T,
      windows: [...this.windowRow(w, 3, WIN_SILL), ...this.windowRow(w, 3, STOREY + WIN_SILL)],
      drift: 0.5,
      plinth: 0.58,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.12],
      cornice: true,
      decay: 0.24,
      dress: 0.8,
    });

    const inner: Rect = { x0: r.x0 + WALL_T, x1: r.x1 - WALL_T, z0: r.z0 + WALL_T, z1: r.z1 - WALL_T };
    // A light well through the first floor: sightlines *between* floors make an
    // interior far more dangerous and far more interesting than stacked boxes.
    const well: Rect = { x0: r.x0 + 4.6, x1: r.x0 + 9.4, z0: r.z0 + 4.6, z1: r.z0 + 9.4 };
    this.slab(inner, STOREY, 0.3, well);
    this.slab(inner, ROOF_2, 0.3, { x0: r.x1 - 4.4, x1: r.x1 - 1.0, z0: r.z0 + 1.0, z1: r.z0 + 3.6 });
    this.parapetRing(r, ROOF_2, PARAPET, 'plasterPale');
    this.roofClutter(r, ROOF_2, 1.1);
    this.cornerQuoins(r, 0, ROOF_2, 'plasterPale');
    this.balconies(r.x0, r.z0, r.x0, r.z1, STOREY + WIN_SILL - 0.06, [-4.6, 4.6]);
    this.balconies(r.x1, r.z0, r.x0, r.z0, STOREY + WIN_SILL - 0.06, [0]);

    // Railing round the light well, and four hall pillars.
    this.add('metal', Props.railing(4.8, 1.0), trs((well.x0 + well.x1) * 0.5, STOREY, well.z0, 0));
    this.add('metal', Props.railing(4.8, 1.0), trs((well.x0 + well.x1) * 0.5, STOREY, well.z1, 0));
    this.add('metal', Props.railing(4.8, 1.0), trs(well.x0, STOREY, (well.z0 + well.z1) * 0.5, Math.PI / 2));
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        this.boxAt('concrete', 0.46, STOREY, 0.46, r.x0 + w * 0.5 + sx * 3.9, STOREY * 0.5, r.z0 + d * 0.5 + sz * 3.9, 0.035);
        this.boxAt('concrete', 0.42, STOREY, 0.42, r.x0 + w * 0.5 + sx * 3.9, STOREY * 1.5, r.z0 + d * 0.5 + sz * 3.9, 0.035);
      }
    }
    // Interior stair from ground to first, tucked against the east wall.
    this.add('concrete', staircase(17, STOREY, 4.6, 1.3), trs(r.x1 - 2.6, 0, r.z0 + 8.6, Math.PI));
    // First to roof.
    this.add('concrete', staircase(17, STOREY, 4.6, 1.3), trs(r.x1 - 2.6, STOREY, r.z0 + 3.4, Math.PI));

    // ── External stair on the east face, up from the market street ──────────
    const sx0 = r.x1 + 0.1;
    this.add('concrete', staircase(20, ROOF_2 * 0.55, 5.0, 1.5), trs(sx0 + 1.0, 0, r.z1 - 6.4, Math.PI));
    this.boxAt('concrete', 2.6, 0.3, 2.6, sx0 + 1.0, ROOF_2 * 0.55, r.z1 - 8.2, 0.04); // half landing
    this.add('concrete', staircase(20, ROOF_2 * 0.45 + 0.3, 4.6, 1.5), trs(sx0 + 1.0, ROOF_2 * 0.55, r.z1 - 7.0));
    this.add('metal', Props.railing(11.4, 1.0), trs(sx0 + 1.85, 0.9, r.z1 - 6.6, 0));
    // Underside piers so the flight is carried, not floating.
    for (let i = 0; i < 4; i++) {
      const z = r.z1 - 9.6 + i * 1.7;
      this.boxAt('concrete', 2.4, 1.4 + i * 0.8, 0.34, sx0 + 1.0, (1.4 + i * 0.8) * 0.5, z, 0.03);
    }

    // ── Ground-floor market hall interior ───────────────────────────────────
    for (let i = 0; i < 4; i++) {
      const parts = Props.marketStall(this.rng, 2.2, 1.1, false);
      this.addParts(parts, trs(r.x0 + 3.0 + (i % 2) * 6.4, 0, r.z0 + 3.4 + Math.floor(i / 2) * 6.0, (i % 2) * Math.PI));
    }
    const crates = this.inst('crate', 'wood', () => Props.crate(0.86, this.rng));
    for (let i = 0; i < 12; i++) {
      crates.pushTRS(
        new THREE.Vector3(r.x0 + this.rng.range(1.2, 12.6), 0.86 * this.rng.int(0, 1), r.z0 + this.rng.range(1.2, 12.6)),
        this.rng.range(0, Math.PI * 2),
      );
    }
  }

  /**
   * SW RUIN — the map's damage set-piece and its riskiest verticality.
   *
   * Two storeys, but the north and west walls are collapsed and the first floor
   * survives over only half the footprint. A rubble ramp climbs to the surviving
   * slab, which gives a 3.4 m position with no parapet and no cover: strong
   * angle onto the south connector, lethal to be caught on.
   */
  private buildRuin(): void {
    const r = SW_RUIN;
    const w = r.x1 - r.x0;
    const d = r.z1 - r.z0;

    // Intact south and east walls.
    this.facade({
      key: 'plasterAlt',
      ax: r.x0, az: r.z1, bx: r.x1, bz: r.z1,
      height: ROOF_2 * 0.92, thickness: WALL_T,
      windows: [
        { x: -3.4, y: WIN_SILL, bare: true, w: 1.3, h: 1.5 },
        { x: 3.4, y: WIN_SILL, shutter: 'teal' },
        { x: 0, y: STOREY + WIN_SILL, w: 1.6, h: 1.5, bare: true },
        { x: -4.2, y: STOREY + WIN_SILL, bare: true },
      ],
      doors: [{ x: 0.4, w: 1.3 }],
      drift: 0.75,
      // A ruin still has the relief it was built with; the damage sits on top
      // of it. Skipping the plinth here would make the wreck read as a
      // different kind of building rather than as a damaged one.
      plinth: 0.56,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.15],
      decay: 0.75,
      dress: 0.5,
    });
    this.facade({
      key: 'plasterAlt',
      ax: r.x1, az: r.z1, bx: r.x1, bz: r.z0,
      height: ROOF_2 * 0.86, thickness: WALL_T,
      windows: [...this.windowRow(d, 2, WIN_SILL, { bare: true } as { w?: number }), ...this.windowRow(d, 2, STOREY + WIN_SILL)],
      doors: [{ x: -d * 0.25, w: 1.4 }],
      drift: 0.55,
      plinth: 0.56,
      plinthKey: 'plinthPaint',
      bands: [STOREY - 0.15],
      decay: 0.8,
      dress: 0.5,
    });
    // Collapsed north and west walls: a ragged decaying crown instead of a top.
    this.addParts(Props.collapsedWall(w, 4.6, WALL_T, this.rng), trs(r.x0 + w * 0.5, 0, r.z0, 0));
    this.addParts(Props.collapsedWall(d, 5.4, WALL_T, this.rng), trs(r.x0, 0, r.z0 + d * 0.5, Math.PI / 2));
    // A surviving pier at the broken corner keeps the silhouette from sagging.
    this.boxAt('plasterAlt', 0.9, ROOF_2 * 0.78, 0.9, r.x0 + 0.45, ROOF_2 * 0.39, r.z0 + 0.45, 0.05);

    // Surviving slab over the east half.
    const survivor: Rect = { x0: r.x0 + w * 0.42, x1: r.x1 - WALL_T, z0: r.z0 + WALL_T, z1: r.z1 - WALL_T };
    this.slab(survivor, STOREY, 0.32);
    // Torn slab edge: broken teeth of concrete with rebar hanging out.
    for (let i = 0; i < 9; i++) {
      const z = survivor.z0 + ((survivor.z1 - survivor.z0) * (i + 0.5)) / 9;
      const over = this.rng.range(-0.5, 0.7);
      this.boxAt('concrete', 0.9 + over, 0.3, (survivor.z1 - survivor.z0) / 9, survivor.x0 - 0.45 + over * 0.5, STOREY - 0.16, z, 0.04);
    }
    for (let i = 0; i < 7; i++) {
      const z = survivor.z0 + this.rng.range(0, survivor.z1 - survivor.z0);
      this.add(
        'rust',
        Props.cable(
          new THREE.Vector3(survivor.x0 - 0.2, STOREY - 0.2, z),
          new THREE.Vector3(survivor.x0 - this.rng.range(0.6, 1.6), STOREY - this.rng.range(0.4, 1.5), z + this.rng.range(-0.4, 0.4)),
          0.15,
          0.014,
        ),
      );
    }
    // Ceiling joists over the collapsed half — the floor that used to be there.
    for (let i = 0; i < 5; i++) {
      const z = r.z0 + 1.4 + i * 2.4;
      this.boxAt('wood', w * 0.44, 0.18, 0.16, r.x0 + w * 0.21, STOREY - 0.2, z, 0.02);
    }

    // Rubble ramp up to the surviving slab: the whole point of the building.
    this.add('concrete', rampGeometry(3.2, STOREY - 0.15, 5.2, 0.06), trs(r.x0 + w * 0.55, 0, r.z1 - 2.0, Math.PI));
    this.addParts(Props.rubblePile(3.4, 60, this.rng), trs(r.x0 + w * 0.55, 0, r.z1 - 3.4));
    this.addParts(Props.rubblePile(2.6, 40, this.rng), trs(r.x0 + 1.6, 0, r.z0 + 2.2));
    this.addParts(Props.rubblePile(2.2, 30, this.rng), trs(r.x0 - 1.4, 0, r.z0 + d * 0.6));

    // Fire damage: soot up the surviving wall, and a burnt-out barrel.
    for (let i = 0; i < 4; i++) {
      this.grunge(
        new THREE.Vector3(r.x1 - WALL_T * 0.5 - 0.02, this.rng.range(1.2, 4.0), r.z0 + this.rng.range(2, d - 2)),
        new THREE.Vector3(-1, 0, 0),
        'stain',
        this.rng.range(1.8, 3.4),
        this.rng.range(-0.3, 0.3),
        0.8,
      );
    }
    this.addParts(Props.fireBarrel(), trs(r.x0 + 3.2, 0, r.z1 - 1.6, 0.4));
  }

  /** SOUTH BLOCK — a low block anchoring the south connector, roof reachable. */
  private buildSouthBlock(): void {
    const r = SOUTH_BLOCK;
    const w = r.x1 - r.x0;
    const d = r.z1 - r.z0;
    this.shell(
      r,
      'plaster',
      0,
      ROOF_SOUTH,
      {
        north: {
          windows: this.windowRow(w, 3, WIN_SILL, { shutter: 'ochre', shutterEvery: 2 }),
          doors: [{ x: -w * 0.28, leaf: 0.7, accent: 'teal' }],
          dress: 0.9,
          signage: true,
        },
        west: { windows: this.windowRow(d, 2, WIN_SILL), dress: 0.7 },
        east: { windows: this.windowRow(d, 2, WIN_SILL, { shutter: 'teal', shutterEvery: 1 }), dress: 0.7 },
        south: { windows: this.windowRow(w, 3, 2.2, { w: 0.9, h: 0.8 }), blind: true, dress: 0.6 },
      },
      {
        thickness: WALL_T, drift: 0.45,
        plinth: 0.5, plinthKey: 'plinthPaint', cornice: true, decay: 0.2,
      },
    );
    const inner: Rect = { x0: r.x0 + WALL_T, x1: r.x1 - WALL_T, z0: r.z0 + WALL_T, z1: r.z1 - WALL_T };
    this.slab(inner, ROOF_SOUTH, 0.3);
    this.parapetRing(r, ROOF_SOUTH, 0.95, 'plaster');
    this.roofClutter(r, ROOF_SOUTH, 0.7);
    this.cornerQuoins(r, 0, ROOF_SOUTH, 'plaster');
    this.add('metal', Props.ladder(ROOF_SOUTH + 0.6, 0.48), trs(r.x1 - 1.8, 0, r.z1 + 0.2));
    // Awning over the north door.
    this.addParts(Props.awning(3.0, 1.6, 2.7, this.rng), trs(r.x0 + w * 0.28, 0, r.z0 - 0.05, Math.PI));
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Internal walls — the pieces that actually shape the lanes
  // ═══════════════════════════════════════════════════════════════════════════

  private buildInternalWalls(): void {
    // ── Courtyard north gate wall ───────────────────────────────────────────
    // A 3 m wall across the courtyard's north edge with one vehicle gate and
    // one personnel gap. This is the map's primary choke: the north yard has to
    // funnel through it, which gives the courtyard defenders a readable
    // engagement and gives attackers a reason to use the alley or market.
    this.facade({
      key: 'concrete',
      ax: 14, az: -26, bx: -21, bz: -26,
      height: 3.0, thickness: 0.4,
      doors: [{ x: 0.0, w: 4.6, h: 3.0 }, { x: -14.0, w: 1.5, h: 2.4 }],
      pilasters: 5,
      pilastersBothSides: true,
      coping: true,
      drift: 0.5,
      plinth: 0.48,
      plinthKey: 'plinthPaint',
    });
    // Gate piers and the swung-open steel leaves.
    for (const sx of [-1, 1]) {
      this.boxAt('concrete', 0.85, 3.9, 0.85, -3.5 + sx * 2.6, 1.95, -26, 0.05);
      this.boxAt('concrete', 1.05, 0.35, 1.05, -3.5 + sx * 2.6, 4.05, -26, 0.05);
      const leaf = chamferedBox(2.1, 2.7, 0.09, 0.02);
      this.add('accentTeal', leaf, trs(-3.5 + sx * 3.6, 1.4, -25.0 + sx * 0.9, sx * 0.9));
      for (let i = 0; i < 6; i++) {
        this.add('accentTeal', chamferedBox(0.1, 2.5, 0.05, 0.012), trs(-3.5 + sx * 3.6 + (i - 2.5) * 0.32, 1.4, -24.95 + sx * 0.9, sx * 0.9));
      }
    }

    // ── Alley east wall ─────────────────────────────────────────────────────
    // Runs between the Radio House and the Ruin, with three breaches so the
    // alley always has an exit east. A corridor with one exit is a trap, not a
    // lane, and players correctly refuse to use it.
    this.facade({
      key: 'plasterAlt',
      ax: -33, az: -26, bx: -33, bz: 17,
      height: 3.1, thickness: 0.4,
      doors: [
        { x: -13.5, w: 1.6, h: 2.4 },
        { x: 2.0, w: 3.2, h: 2.6 },
        { x: 14.5, w: 1.6, h: 2.4 },
      ],
      pilasters: 5,
      pilastersBothSides: true,
      coping: true,
      drift: 0.55,
      plinth: 0.48,
      plinthKey: 'plinthPaint',
      dress: 0.5,
    });
    // The middle breach is a blown hole, not a doorway: rubble on both sides.
    this.addParts(Props.rubblePile(1.9, 24, this.rng), trs(-33, 0, -3.5));
    this.addParts(Props.rubblePile(1.4, 16, this.rng), trs(-31.6, 0, -3.5));

    // ── Courtyard south low wall ────────────────────────────────────────────
    // 1.05 m: pure crouch cover. It defines the courtyard's south edge for the
    // eye without closing the sightline, which is what stops the middle of the
    // map from becoming a single 40 m corridor.
    this.facade({
      key: 'concrete',
      ax: -21, az: 12, bx: 6, bz: 12,
      height: 1.05, thickness: 0.34,
      pilasters: 4.5,
      pilastersBothSides: true,
      coping: true,
      drift: 0.3,
    });
    // Two gaps in it, so movement south is never fully blocked.
    // (Modelled by simply omitting wall between x = -8 and -5.)
    this.facade({
      key: 'concrete',
      ax: 6, az: 12, bx: 6, bz: 6.5,
      height: 1.05, thickness: 0.34, coping: true,
    });

    // ── Market street west wall ─────────────────────────────────────────────
    // Separates the courtyard from the market for part of its length, so the
    // two lanes are distinct spaces rather than one wide one.
    this.facade({
      key: 'plasterAlt',
      ax: 30, az: -6, bx: 30, bz: 10,
      height: 2.7, thickness: 0.4,
      doors: [{ x: 0, w: 2.4, h: 2.4 }],
      pilasters: 5,
      pilastersBothSides: true,
      coping: true,
      drift: 0.4,
      plinth: 0.44,
      plinthKey: 'plinthPaint',
    });
    this.facade({
      key: 'plasterAlt',
      ax: 30, az: 20, bx: 30, bz: 40,
      height: 2.7, thickness: 0.4,
      doors: [{ x: 3, w: 1.8, h: 2.3 }],
      pilasters: 5,
      pilastersBothSides: true,
      coping: true,
      drift: 0.4,
      plinth: 0.44,
      plinthKey: 'plinthPaint',
    });
    // North service strip wall behind the buildings.
    this.facade({
      key: 'plasterAlt',
      ax: -21, az: -38, bx: -21, bz: -30,
      height: 2.9, thickness: 0.36, coping: true, drift: 0.4,
    });
    this.facade({
      key: 'plasterAlt',
      ax: 30, az: -30, bx: 30, bz: -38,
      height: 2.9, thickness: 0.36, coping: true, drift: 0.4,
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Hero landmark
  // ═══════════════════════════════════════════════════════════════════════════

  private buildHeroTower(): void {
    this.addParts(Props.commsTower(this.rng), trs(TOWER.x, TOWER.y, TOWER.z, 0.24));

    // Sandbag emplacements at the base, facing outward along the two main
    // approaches. Facing is the whole point — an emplacement pointing the wrong
    // way is set dressing; one pointing at the gate is a story.
    const emplacements: Array<[number, number]> = [
      [-Math.PI * 0.5, 5.4],
      [Math.PI * 0.42, 5.6],
    ];
    for (const [az, rad] of emplacements) {
      const px = TOWER.x + Math.sin(az) * rad;
      const pz = TOWER.z + Math.cos(az) * rad;
      this.add('sand', Props.sandbagWall(4.4, 6, this.rng), trs(px, 0, pz, az + Math.PI));
      // Wing walls, angled back — a real fighting position is not a straight line.
      for (const s of [-1, 1]) {
        const wx = px + Math.cos(az) * s * 2.2 + Math.sin(az) * -0.7;
        const wz = pz - Math.sin(az) * s * 2.2 + Math.cos(az) * -0.7;
        this.add('sand', Props.sandbagWall(1.9, 5, this.rng), trs(wx, 0, wz, az + Math.PI + s * 0.85));
      }
      // Spent brass and an ammo crate behind it.
      this.inst('crate', 'wood', () => Props.crate(0.86, this.rng)).pushTRS(
        new THREE.Vector3(px - Math.sin(az) * 1.5, 0, pz - Math.cos(az) * 1.5),
        az + this.rng.range(-0.3, 0.3),
      );
      this.grunge(
        new THREE.Vector3(px - Math.sin(az) * 1.0, 0.03, pz - Math.cos(az) * 1.0),
        new THREE.Vector3(0, 1, 0),
        'stain',
        2.2,
        this.rng.range(0, Math.PI),
        0.5,
      );
    }

    // The wrecked car that took the tower's collapse: the two events read as
    // one incident, which is more convincing than two unrelated bits of damage.
    this.addParts(Props.derelictCar(this.rng), trs(TOWER.x + 6.2, 0, TOWER.z + 5.6, 2.1));
    this.addParts(Props.rubblePile(2.8, 34, this.rng), trs(TOWER.x + 4.4, 0, TOWER.z + 4.0));
    for (let i = 0; i < 5; i++) {
      const a = this.rng.range(0, Math.PI * 2);
      const rr = this.rng.range(5, 11);
      this.grunge(
        new THREE.Vector3(TOWER.x + Math.cos(a) * rr, 0.03, TOWER.z + Math.sin(a) * rr),
        new THREE.Vector3(0, 1, 0),
        'stain',
        this.rng.range(1.4, 3.2),
        this.rng.range(0, Math.PI),
        this.rng.range(0.3, 0.6),
      );
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Verticality — gangways, scaffolding, the rooftop lane
  // ═══════════════════════════════════════════════════════════════════════════

  private buildVerticality(): void {
    const deck = (y: number): number => y + 0.28;

    // West terrace roof ↔ Radio House roof, over the alley.
    this.addParts(Props.gangway(6.6, 1.3), trs(-36.2, deck(ROOF_2) + 0.02, -32, Math.PI / 2));
    // Garage roof ↔ east terrace north section, over the market street.
    this.addParts(Props.gangway(9.4, 1.2), trs(34.5, deck(ROOF_GARAGE) + 0.02, -33, Math.PI / 2));
    // Market Hall roof ↔ east terrace south, over the market street. Sited at
    // z = 22 so its mid-span lands directly over the x = 30 boundary wall.
    const GW_Z = 22;
    this.addParts(Props.gangway(18.4, 1.2), trs(30.0, deck(ROOF_2) + 0.02, GW_Z, Math.PI / 2));
    // An 18 m unsupported plank would be the least believable object in the
    // map, so it is carried at mid-span — by a short pier rising off the wall
    // that is already there, rather than by a column planted in the lane. A
    // free-standing post in a 9 m street both looks arbitrary and snags
    // movement along the market's main run.
    const WALL_TOP = 2.7;
    for (const sz of [-1, 1]) {
      this.boxAt('metal', 0.18, ROOF_2 - WALL_TOP, 0.18, 30.0, (ROOF_2 + WALL_TOP) * 0.5, GW_Z + sz * 0.55, 0.03);
      // Knee brace back down onto the wall head.
      this.add(
        'metal',
        chamferedBox(0.12, 1.6, 0.12, 0.025),
        // Tilt about X so the brace leans out along Z, back onto the wall head.
        trs(30.0, WALL_TOP + 0.62, GW_Z + sz * 1.05, 0, 1, -sz * 0.68, 0),
      );
    }
    this.boxAt('metal', 0.55, 0.16, 1.6, 30.0, ROOF_2 - 0.08, GW_Z, 0.03);
    // Thickened wall head acting as the pad stone under the pier.
    this.boxAt('concrete', 1.3, 0.5, 2.4, 30.0, WALL_TOP - 0.2, GW_Z, 0.04);

    // Mid-market scaffold: two decks at 2.6 and 5.2 with a ladder run.
    this.addParts(Props.scaffold(7.5, 2, 2.6, 1.35), trs(33.4, 0, 2.0, Math.PI / 2));
    // Second scaffold against the ruin, giving the south-west its own perch.
    this.addParts(Props.scaffold(6.0, 2, 2.6, 1.3), trs(-19.6, 0, 23.5, Math.PI / 2));

    // Stacked containers in the courtyard, climbable via crates. Two-high is
    // 5.18 m, which is roof-adjacent — deliberately just below the parapets so
    // it feeds the rooftop lane rather than dominating it.
    const cont = this.inst('container', 'rust', () => Props.shippingContainer()[0].geo);
    // Two of the six carry faded red-lead paint instead of bare rust. Same
    // geometry, same transforms, same gameplay — but six identical brown boxes
    // is a texture and four-plus-two is a *place*, and it gives the courtyard a
    // second landmark colour that is not the tower.
    const contRed = this.inst('containerRed', 'paintRust', () => Props.shippingContainer()[0].geo);
    cont.pushTRS(new THREE.Vector3(-14.5, 0, 4.0), Math.PI / 2);
    contRed.pushTRS(new THREE.Vector3(-14.5, 2.591, 4.0), Math.PI / 2 + 0.03);
    contRed.pushTRS(new THREE.Vector3(8.5, 0, -14.5), 0.18);
    cont.pushTRS(new THREE.Vector3(2.0, 0, 8.6), 1.92);
    cont.pushTRS(new THREE.Vector3(24.0, 0, -20.0), Math.PI / 2 - 0.1);
    cont.pushTRS(new THREE.Vector3(-27.0, 0, -18.0), 0.06);
    // The step-up crates that make the stack climbable.
    const crates = this.inst('crate', 'wood', () => Props.crate(0.86, this.rng));
    for (const [bx, bz, ry] of [[-13.0, 7.4, 0.3], [-12.2, 6.9, -0.2]] as const) {
      crates.pushTRS(new THREE.Vector3(bx, 0, bz), ry);
      crates.pushTRS(new THREE.Vector3(bx + 0.1, 0.86, bz - 0.05), ry + 0.4);
    }
    this.add('metal', Props.ladder(2.7, 0.44), trs(-14.5, 2.62, 6.4, 0));

    // Ruin slab ↔ nothing: deliberately a dead end you must drop from. Not
    // every high position should chain into another one; some should cost you.
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // The market street
  // ═══════════════════════════════════════════════════════════════════════════

  private buildMarket(): void {
    const rng = this.rng;
    // Stalls down the west side of the street, staggered so the lane weaves.
    const stallZ = [-24, -14, -5, 6, 15, 27, 34];
    for (let i = 0; i < stallZ.length; i++) {
      const toppled = i === 3; // one knocked over — evidence, not decoration
      const x = 31.6 + (i % 2) * 0.7;
      this.addParts(
        Props.marketStall(rng, rng.range(2.1, 2.8), 1.15, toppled),
        trs(x, 0, stallZ[i], Math.PI / 2 + rng.range(-0.12, 0.12)),
      );
      // Produce crates and a barrel beside each stall.
      const crates = this.inst('crate', 'wood', () => Props.crate(0.86, rng));
      for (let k = 0; k < rng.int(1, 4); k++) {
        crates.pushTRS(
          new THREE.Vector3(x + rng.range(-0.9, 0.9), 0.86 * (k > 1 ? 1 : 0), stallZ[i] + rng.range(1.6, 2.8)),
          rng.range(0, Math.PI * 2),
        );
      }
      if (rng.chance(0.5)) {
        this.inst('drumRust', 'rust', () => Props.oilDrum()).pushTRS(
          new THREE.Vector3(x + rng.range(-1.2, 1.2), 0, stallZ[i] - rng.range(1.6, 2.6)),
          rng.range(0, Math.PI * 2),
        );
      }
      // Scattered produce from the toppled stall.
      if (toppled) {
        const goods = this.inst('produce', 'accentOchre', () => new THREE.SphereGeometry(0.07, 6, 4), false);
        for (let k = 0; k < 26; k++) {
          const a = rng.range(0, Math.PI * 2);
          const rr = rng.range(0.4, 3.4);
          goods.pushTRS(new THREE.Vector3(x + Math.cos(a) * rr, 0.07, stallZ[i] + Math.sin(a) * rr), 0, rng.range(0.7, 1.3));
        }
      }
    }

    // Bunting and cable spans across the street: the strongest single cue that
    // this lane is "market" and not "alley".
    for (let i = 0; i < 6; i++) {
      const z = -30 + i * 13 + rng.range(-2, 2);
      this.add(
        'metal',
        Props.cable(
          new THREE.Vector3(30.2, rng.range(4.4, 5.8), z),
          new THREE.Vector3(38.8, rng.range(4.6, 6.0), z + rng.range(-2, 2)),
          rng.range(0.5, 1.1),
          0.02,
        ),
      );
    }

    // Bollards protecting the shopfronts, and tyre marks on the paving.
    const bollards = this.inst('bollard', 'concrete', () => Props.bollard());
    for (let i = 0; i < 14; i++) {
      bollards.pushTRS(new THREE.Vector3(38.2, 0, -38 + i * 5.6), 0);
    }
    for (let i = 0; i < 9; i++) {
      this.grunge(
        new THREE.Vector3(rng.range(31, 38), 0.03, rng.range(-38, 38)),
        new THREE.Vector3(0, 1, 0),
        'tyremark',
        rng.range(3.0, 6.5),
        rng.chance(0.5) ? 0 : Math.PI / 2 + rng.range(-0.4, 0.4),
        rng.range(0.4, 0.8),
      );
    }
    // Posters and graffiti on the shopfront wall.
    for (let i = 0; i < 10; i++) {
      const z = rng.range(-40, 40);
      this.grunge(
        new THREE.Vector3(39 - WALL_T * 0.5 - 0.02, rng.range(1.1, 2.4), z),
        new THREE.Vector3(-1, 0, 0),
        rng.chance(0.6) ? 'poster' : 'graffiti',
        rng.range(0.8, 1.7),
        rng.range(-0.06, 0.06),
        rng.range(0.75, 1),
      );
    }
    for (let i = 0; i < 8; i++) {
      const z = rng.range(-40, 40);
      this.grunge(
        new THREE.Vector3(-39 + WALL_T * 0.5 + 0.02, rng.range(1.0, 2.6), z),
        new THREE.Vector3(1, 0, 0),
        rng.chance(0.45) ? 'poster' : 'graffiti',
        rng.range(0.9, 2.0),
        rng.range(-0.06, 0.06),
        rng.range(0.7, 1),
      );
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Cover scatter
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The cover pass.
   *
   * Placement is authored, not random: barriers form chicanes across the two
   * longest sightlines, drum and crate clusters sit at the ends of lanes where
   * a player needs something to break contact behind, and every cover island is
   * within about 6 m of another so a fighting retreat is always possible.
   * Random scatter produces cover that is everywhere and useful nowhere.
   */
  private scatterCover(): void {
    const rng = this.rng;

    // ── Jersey barrier chicane across the courtyard's long axis ─────────────
    const jersey = this.inst('jersey', 'concrete', () => Props.jerseyBarrier());
    const chicane: Array<[number, number, number]> = [
      [-16.5, -19.5, 0.06], [-11.5, -17.0, 0.35], [-6.0, -19.8, -0.18],
      [-0.5, -17.6, 0.28], [5.0, -20.2, -0.1], [10.0, -18.0, 0.4],
      [-17.0, 5.0, 1.62], [-12.0, 7.6, 1.35], [4.5, 3.0, 1.7], [9.0, 5.4, 1.42],
    ];
    for (const [x, z, ry] of chicane) jersey.pushTRS(new THREE.Vector3(x, 0, z), ry);

    // ── gabion L-wall on the courtyard's east flank ──────────────────────────
    const gabion = this.inst('gabion', 'sand', () => Props.gabion());
    for (let r = 0; r < 2; r++) {
      for (let i = 0; i < 6; i++) gabion.pushTRS(new THREE.Vector3(11.5 + i * 1.09, r * 1.07, 0.5), 0);
      for (let i = 0; i < 3; i++) gabion.pushTRS(new THREE.Vector3(11.5, r * 1.07, 1.6 + i * 1.09), 0);
    }
    // A second stack guarding the gate's inner side.
    for (let r = 0; r < 2; r++) {
      for (let i = 0; i < 4; i++) gabion.pushTRS(new THREE.Vector3(-9.0 - i * 1.09, r * 1.07, -23.5), 0);
    }

    // ── Sandbag positions covering the alley breaches ───────────────────────
    this.add('sand', Props.sandbagWall(3.4, 5, rng), trs(-30.6, 0, -3.5, -Math.PI / 2));
    this.add('sand', Props.sandbagWall(2.6, 4, rng), trs(-30.4, 0, -19.0, -Math.PI / 2 + 0.3));
    this.add('sand', Props.sandbagWall(3.0, 6, rng), trs(-24.0, 0, -24.0, 0.5));

    // ── Concrete pipe sections ──────────────────────────────────────────────
    for (const [x, z, ry] of [[-6.0, 9.5, 0.4], [-6.4, 11.0, 0.1], [17.0, -8.0, 1.3]] as const) {
      this.add('concrete', Props.concretePipe(2.6, 0.62), trs(x, 0, z, ry));
    }

    // ── Planters softening the courtyard's paved edge ───────────────────────
    for (const [x, z, ry] of [[-19.0, -12.0, 0], [-19.0, -8.0, 0], [12.5, -8.5, 0], [12.5, -4.5, 0]] as const) {
      this.addParts(Props.planter(1.9, 0.95, 0.95), trs(x, 0, z, ry));
    }

    // ── Drum clusters at lane mouths ────────────────────────────────────────
    const drums = this.inst('drumRust', 'rust', () => Props.oilDrum());
    const drumClusters: Array<[number, number, number]> = [
      [-36.2, -25.0, 4], [-35.6, 12.0, 5], [-36.4, 30.0, 3],
      [-18.5, -30.5, 4], [6.0, -30.0, 5], [26.0, -12.0, 4],
      [24.5, 12.0, 3], [-2.0, 33.0, 5], [-16.0, 36.0, 4], [17.0, 33.5, 3],
    ];
    for (const [cx, cz, n] of drumClusters) {
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 + rng.range(-0.4, 0.4);
        const rr = rng.range(0.25, 0.85);
        drums.pushTRS(
          new THREE.Vector3(cx + Math.cos(a) * rr, 0, cz + Math.sin(a) * rr),
          rng.range(0, Math.PI * 2),
          1,
          rng.chance(0.12) ? 1.57 : 0,
        );
      }
    }

    // ── Crate and pallet stacks ─────────────────────────────────────────────
    const crates = this.inst('crate', 'wood', () => Props.crate(0.86, rng));
    const cratesSmall = this.inst('crateSmall', 'wood', () => Props.crate(0.55, rng), false);
    const pallets = this.inst('pallet', 'wood', () => Props.pallet(), false);
    const stacks: Array<[number, number]> = [
      [-36.5, -12], [-36.0, 22], [-35.0, -35], [-25.0, -20.5], [-16.0, 0.5],
      [0.5, -10.0], [7.5, -22.0], [19.0, -16.0], [23.0, 6.0], [26.5, 22.0],
      [-6.0, 27.0], [8.0, 33.0], [-20.0, 33.0], [31.0, -30.0],
    ];
    for (const [cx, cz] of stacks) {
      pallets.pushTRS(new THREE.Vector3(cx, 0, cz), rng.range(0, Math.PI * 2));
      const n = rng.int(2, 5);
      for (let i = 0; i < n; i++) {
        const level = Math.floor(i / 2);
        crates.pushTRS(
          new THREE.Vector3(cx + rng.range(-0.5, 0.5), 0.144 + level * 0.86, cz + rng.range(-0.5, 0.5)),
          rng.range(0, Math.PI * 2),
        );
      }
      for (let i = 0; i < rng.int(0, 3); i++) {
        cratesSmall.pushTRS(
          new THREE.Vector3(cx + rng.range(-1.4, 1.4), 0, cz + rng.range(-1.4, 1.4)),
          rng.range(0, Math.PI * 2),
        );
      }
    }

    // ── Tyre piles ──────────────────────────────────────────────────────────
    for (const [x, z] of [[-34.5, 8.0], [22.0, -25.0], [31.5, -35.0], [-11.0, 30.0]] as const) {
      this.add('rubber', Props.tyrePile(rng.int(4, 7), rng), trs(x, 0, z, rng.range(0, Math.PI)));
    }

    // ── A second wrecked vehicle in the north yard ──────────────────────────
    this.addParts(Props.derelictCar(rng), trs(-12.0, 0, -33.0, 1.15));
    this.addParts(Props.derelictCar(rng), trs(20.5, 0, 26.5, -0.55));
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Debris and small storytelling detail
  // ═══════════════════════════════════════════════════════════════════════════

  private scatterDebris(): void {
    const rng = this.rng;
    // Loose bricks and small rubble drifting against walls and in corners.
    // Small debris does not cast shadows: at this size the shadow is a couple
    // of pixels and costs a full shadow-map draw for nothing.
    const bricks = this.inst('brick', 'concrete', () => Props.brick(), false);
    const chunks = this.inst('chunk', 'concrete', () => Props.rubbleChunk(0.22, rng), false);
    const stones = this.inst('stone', 'rock', () => rockGeometry(0.16, rng, 0), false);

    // Hug debris to vertical surfaces: that is where wind and traffic put it,
    // and scattering it uniformly across open ground looks like confetti.
    const walls: Array<[number, number, number, number]> = [
      [-33, -26, -33, 17], [-39, -42, -39, 42], [39, -42, 39, 42],
      [-21, -38, -21, -26], [-45, -45, 45, -45], [-45, 45, 45, 45],
      [14, -26, -21, -26], [7, 12, 7, 26], [-34, 17, -21, 17],
    ];
    for (const [ax, az, bx, bz] of walls) {
      const len = Math.hypot(bx - ax, bz - az);
      const n = Math.round(len * 0.55);
      for (let i = 0; i < n; i++) {
        const t = rng.next();
        const off = rng.range(0.3, 1.9) * (rng.chance(0.5) ? 1 : -1);
        const dx = (bx - ax) / len;
        const dz = (bz - az) / len;
        const px = ax + (bx - ax) * t - dz * off;
        const pz = az + (bz - az) * t + dx * off;
        const pick = rng.next();
        if (pick < 0.4) bricks.pushTRS(new THREE.Vector3(px, 0.033, pz), rng.range(0, Math.PI * 2), 1, rng.range(-0.1, 0.1), rng.range(-0.1, 0.1));
        else if (pick < 0.75) chunks.pushTRS(new THREE.Vector3(px, 0.06, pz), rng.range(0, Math.PI * 2), rng.range(0.6, 1.5));
        else stones.pushTRS(new THREE.Vector3(px, 0.05, pz), rng.range(0, Math.PI * 2), rng.range(0.5, 1.6));
      }
    }

    // Half-built wall: bricks stacked on a pallet next to a course laid in mud.
    for (let row = 0; row < 4; row++) {
      const n = 9 - row;
      for (let i = 0; i < n; i++) {
        bricks.pushTRS(
          new THREE.Vector3(-24.6 + i * 0.225 + (row % 2) * 0.11, 0.033 + row * 0.072, 8.4),
          rng.range(-0.03, 0.03),
        );
      }
    }
    this.inst('pallet', 'wood', () => Props.pallet(), false)
      .pushTRS(new THREE.Vector3(-23.0, 0, 9.8), 0.3);

    // Blood and scorch storytelling near the ruin and the gate.
    for (let i = 0; i < 6; i++) {
      this.grunge(
        new THREE.Vector3(rng.range(-30, -22), 0.03, rng.range(18, 30)),
        new THREE.Vector3(0, 1, 0),
        'stain',
        rng.range(1.2, 2.6),
        rng.range(0, Math.PI),
        rng.range(0.4, 0.75),
      );
    }
    // Bullet-scarred walls around the gate and the ruin's doorways.
    const scarSpots: Array<[number, number, number, number, number]> = [
      [-6.6, 1.6, -25.7, 0, 1],
      [-0.4, 2.1, -25.7, 0, 1],
      [-1.2, 1.2, -26.3, 0, -1],
      [-21.3, 1.7, 22.0, -1, 0],
      [-21.3, 2.6, 25.0, -1, 0],
      [6.7, 1.5, 18.0, -1, 0],
      [30.3, 1.8, 2.0, 1, 0],
      [-32.7, 1.6, -14.0, 1, 0],
      [-32.7, 2.2, 6.0, 1, 0],
    ];
    for (const [x, y, z, nx, nz] of scarSpots) {
      this.grunge(
        new THREE.Vector3(x, y, z),
        new THREE.Vector3(nx, 0, nz),
        'bulletscar',
        rng.range(1.3, 2.4),
        rng.range(0, Math.PI * 2),
        rng.range(0.6, 1),
      );
    }

    // Sand drifts in the courtyard's lee corners, independent of walls.
    for (const [x, z, ry] of [[-20.0, -24.0, 0.6], [12.5, -24.0, -0.5], [-20.0, 10.0, 2.6], [28.0, 8.0, 1.6]] as const) {
      this.add('sand', sandDrift(rng.range(4, 8), rng.range(0.25, 0.5), 1.6, rng, 12), trs(x, 0.01, z, ry));
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Skyline — everything that breaks the horizon
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The town beyond the wall, plus the tall forms that punctuate it.
   *
   * The problem this solves: from anywhere inside the compound the horizon was
   * a single flat band of two-storey parapets, and above it, nothing. A player
   * cannot navigate by that, and it makes the map feel like a diorama on a
   * table rather than a piece cut out of a city.
   *
   * The fix is entirely outside the play space and therefore entirely free of
   * gameplay consequences: backdrop massing at varied heights, one minaret and
   * two water towers as unmistakable bearings, and a line of utility poles
   * whose wires cross the sky. All of it merges into buckets that already
   * exist, so the whole pass costs about six draw calls.
   */
  private buildSkyline(): void {
    const rng = this.rng;

    // ── Backdrop massing ────────────────────────────────────────────────────
    // Rings of blocks outside each wall, at increasing distance and decreasing
    // detail. Kept far enough back that the perimeter wall's coping still reads
    // against sky at eye level — the town appears *above* the wall, not behind
    // a fence, which is what gives the compound its sense of being enclosed by
    // something larger.
    const placed: Array<{ x: number; z: number; r: number }> = [];
    const tryPlace = (x: number, z: number, r: number): boolean => {
      for (const p of placed) if (Math.hypot(p.x - x, p.z - z) < p.r + r) return false;
      placed.push({ x, z, r });
      return true;
    };
    for (let ring = 0; ring < 3; ring++) {
      const count = 11 + ring * 4;
      for (let i = 0; i < count; i++) {
        const a = (i / count) * Math.PI * 2 + rng.range(-0.14, 0.14);
        const bw = rng.range(8, 19);
        const bd = rng.range(8, 17);
        const clear = Math.max(bw, bd) * 0.75;
        // Keep every block's near face clear of the 45 m perimeter wall by a
        // full building width. A backdrop block that overlaps the compound is
        // not a backdrop, it is a bug you can walk into.
        const rr = 64 + clear + ring * 24 + rng.range(0, 18);
        const x = Math.cos(a) * rr;
        const z = Math.sin(a) * rr;
        if (!tryPlace(x, z, clear)) continue;
        // Heights step down with distance: aerial perspective plus a genuine
        // silhouette gradient, so the eye reads depth rather than a wall of boxes.
        const h = rng.range(4.2, ring === 0 ? 13.0 : 9.5) * (1 - ring * 0.1);
        // Sunk 1.4 m into the dune. The terrain out here rolls through ±5 m, so
        // sitting a block on y = 0 leaves it hanging in mid-air on the far side
        // of a crest — which is exactly what it did before this line existed.
        const y = this.groundHeight(x, z) - 1.4;
        this.addParts(Props.backdropBlock(bw, h, bd, rng), trs(x, y, z, rng.range(0, Math.PI * 2)));
        // A tank or an aerial on one backdrop roof in three, so even the
        // distant skyline has texture rather than a clean parapet line.
        if (rng.chance(0.4)) {
          this.inst('tvAerial', 'metal', () => Props.tvAerial(2.4), false)
            .pushTRS(new THREE.Vector3(x + rng.range(-2, 2), y + h + 0.6, z + rng.range(-2, 2)), 0, new THREE.Vector3(1, rng.range(0.9, 1.8), 1));
        }
        if (rng.chance(0.3)) {
          this.addParts(Props.waterTank(0.6, 1.2), trs(x + rng.range(-2, 2), y + h + 0.6, z + rng.range(-2, 2)));
        }
      }
    }

    // ── The minaret: the map's compass ──────────────────────────────────────
    // North-west, outside the wall, 24 m — tall enough to clear the west
    // terrace roof from anywhere in the courtyard, close enough to read.
    this.addParts(Props.minaret(24), trs(-58, this.groundHeight(-58, -49) - 0.4, -49, 0.5));

    // ── Water towers ────────────────────────────────────────────────────────
    this.addParts(Props.waterTower(7.6, 1.5), trs(54, this.groundHeight(54, -33) - 0.3, -33, 0.4));
    this.addParts(Props.waterTower(6.4, 1.25), trs(-51, this.groundHeight(-51, 36) - 0.3, 36, -0.8));

    // ── Utility poles and the wires between them ────────────────────────────
    // Wires crossing the sky are the single most effective way to stop a
    // skyline reading as a cut-out, because they are the only lines in the
    // scene that are neither vertical nor horizontal.
    const poleRuns: Array<{ x: number; z: number }[]> = [
      // Down the market street, inside the play space (the poles are slim
      // enough not to interfere with the lane and are set against the kerb).
      [{ x: 39.4, z: -40 }, { x: 39.4, z: -18 }, { x: 39.4, z: 4 }, { x: 39.4, z: 26 }, { x: 39.4, z: 43 }],
      // Along the north yard and out over the wall.
      [{ x: -30, z: -41.5 }, { x: -10, z: -41.5 }, { x: 10, z: -41.5 }, { x: 28, z: -41.5 }, { x: 46, z: -48 }],
      // Off-map to the south-west, purely for the horizon.
      [{ x: -52, z: 18 }, { x: -55, z: 34 }, { x: -60, z: 52 }, { x: -68, z: 70 }],
    ];
    for (const run of poleRuns) {
      let prevTop = 0;
      for (let i = 0; i < run.length; i++) {
        const base = this.groundHeight(run[i].x, run[i].z) - 0.1;
        const h = 8.2 + rng.range(-0.6, 0.9);
        this.addParts(Props.utilityPole(h), trs(run[i].x, base, run[i].z, rng.range(0, Math.PI)));
        const top = base + h - 0.35;
        if (i > 0) {
          const a = run[i - 1];
          const b = run[i];
          const span = Math.hypot(b.x - a.x, b.z - a.z);
          // Four conductors on the crossarm, each with its own sag. Anchored to
          // the *actual* crossarm heights so the wires meet the poles.
          for (let k = 0; k < 4; k++) {
            const off = (k - 1.5) * 0.44;
            const dirX = (b.z - a.z) / span;
            const dirZ = -(b.x - a.x) / span;
            this.add(
              'metal',
              Props.cable(
                new THREE.Vector3(a.x + dirX * off, prevTop, a.z + dirZ * off),
                new THREE.Vector3(b.x + dirX * off, top, b.z + dirZ * off),
                span * rng.range(0.035, 0.055),
                0.014,
              ),
            );
          }
        }
        prevTop = top;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Street furniture
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The things that live in a street rather than on a building: standpipes,
   * gully grates, cable spools, signage, a delivery abandoned mid-unload.
   *
   * Placement is against edges. Nothing here sits in the middle of a lane,
   * because anything in the middle of a lane is either cover (and therefore
   * belongs in the cover pass, where it can be balanced) or an obstruction the
   * player will resent.
   */
  private buildStreetFurniture(): void {
    const rng = this.rng;

    // ── Gully grates along the paved lanes ──────────────────────────────────
    const grates = this.inst('grate', 'rust', () => Props.drainGrate(), false);
    for (const [x, z0, z1] of [[30.6, -36, 36], [38.1, -36, 36], [-33.6, -22, 14]] as const) {
      for (let z = z0; z < z1; z += rng.range(11, 17)) {
        grates.pushTRS(new THREE.Vector3(x, 0.02, z), rng.range(-0.05, 0.05));
        this.grunge(
          new THREE.Vector3(x, 0.025, z),
          new THREE.Vector3(0, 1, 0),
          'stain',
          rng.range(1.1, 1.9),
          rng.range(0, Math.PI),
          0.4,
        );
      }
    }

    // ── Cable drums and jerrycans against walls ─────────────────────────────
    for (const [x, z, ry] of [
      [-31.4, -21.0, 0.4], [32.2, -31.0, 1.2], [12.0, -29.5, 2.4], [-19.0, 15.5, 0.9],
    ] as const) {
      this.addParts(Props.cableSpool(), trs(x, 0, z, ry));
    }
    const cans = this.inst('jerrycan', 'paintFaded', () => Props.jerrycan());
    // Deliberately kept 3 m clear of every spawn point — a 0.47 m steel can
    // interpenetrating a freshly-spawned capsule is a bug report, not dressing.
    for (const [cx, cz] of [
      [-19.4, -29.5], [17.5, -30.5], [-30.9, -6.0], [32.6, 8.5], [4.0, 30.0], [-22.6, 21.0], [20.8, -19.0],
    ] as const) {
      for (let i = 0; i < rng.int(2, 4); i++) {
        cans.pushTRS(
          new THREE.Vector3(cx + rng.range(-0.5, 0.5), 0, cz + rng.range(-0.5, 0.5)),
          rng.range(0, Math.PI * 2),
        );
      }
    }

    // ── Wheelbarrows, parked where the work stopped ─────────────────────────
    this.addParts(Props.wheelbarrow(rng), trs(-23.6, 0, 9.4, 1.1));
    this.addParts(Props.wheelbarrow(rng), trs(-19.2, 0, 24.6, -0.5));
    this.addParts(Props.wheelbarrow(rng), trs(9.5, 0, -30.2, 2.6));

    // ── Tarped stacks: the compound's blue ──────────────────────────────────
    for (const [x, z, w, d, h] of [
      [-24.0, -21.5, 2.0, 1.5, 1.1], [16.0, 6.0, 2.2, 1.6, 1.0], [-8.0, 29.5, 1.8, 1.4, 0.95],
    ] as const) {
      // Something under the tarp, or the drape has nothing to sit on.
      const crates = this.inst('crate', 'wood', () => Props.crate(0.86, rng));
      for (let i = 0; i < 3; i++) {
        crates.pushTRS(new THREE.Vector3(x + rng.range(-0.4, 0.4), 0, z + rng.range(-0.3, 0.3)), rng.range(0, 3.1));
      }
      this.add('tarpBlue', Props.tarpSheet(w, d, h, rng), trs(x, 0, z, rng.range(0, Math.PI)));
    }

    // ── Standpipe and hose bib on the yard wall ─────────────────────────────
    for (const [x, z, ry] of [[-20.6, -26.9, 0], [30.6, -4.0, -Math.PI / 2]] as const) {
      this.add('rust', chamferedCylinder(0.04, 1.15, 0.01, 8), trs(x, 0.58, z));
      this.add('rust', chamferedBox(0.16, 0.1, 0.22, 0.02), trs(x, 1.12, z, ry));
      this.add('concrete', chamferedBox(0.7, 0.12, 0.7, 0.02), trs(x, 0.06, z, ry));
      this.grunge(new THREE.Vector3(x, 0.03, z), new THREE.Vector3(0, 1, 0), 'stain', 1.4, 0, 0.5);
    }

    // ── Debris skirts around the courtyard's cover islands ──────────────────
    //
    // The courtyard reads as a bare plane because every prop in it is an
    // isolated object standing on clean ground. Real cover accumulates a skirt:
    // the sand that piles in its lee, the rubble that was shovelled against it,
    // the pallet somebody leaned there. Skirting each island also *widens* it
    // visually, which makes the cover rhythm easier to read at a glance without
    // changing a single collision volume.
    const islands: Array<[number, number, number]> = [
      [-14.5, 4.0, 3.4],   // stacked containers
      [8.5, -14.5, 3.2],   // container
      [2.0, 8.6, 3.0],     // container
      [24.0, -20.0, 3.0],  // container
      [-27.0, -18.0, 3.0], // container
      [11.5, 0.5, 2.6],    // gabion L-wall
      [-2.0, -4.0, 5.0],   // the tower basin
      [-11.5, -17.0, 2.2], // chicane
      [5.0, -20.2, 2.2],
      [-17.0, 5.0, 2.2],
      [9.0, 5.4, 2.2],
      [-9.0, -23.5, 2.4],  // gabion by the gate
      [-6.2, 10.2, 2.2],   // concrete pipes
    ];
    const gravel = this.inst('gravel', 'rock', () => rockGeometry(0.075, rng, 0), false);
    const chunks = this.inst('chunk', 'concrete', () => Props.rubbleChunk(0.22, rng), false);
    const litter = this.inst('litter', 'paper', () => Props.litterScrap(rng), false);
    for (const [cx, cz, rad] of islands) {
      // Windward side gets swept clean, leeward side collects: bias the whole
      // skirt toward one quadrant instead of ringing the object evenly.
      const lee = rng.range(0, Math.PI * 2);
      for (let i = 0; i < rng.int(20, 34); i++) {
        const a = lee + rng.gauss(0, 1.1);
        const rr = rad * rng.range(0.75, 1.5);
        const x = cx + Math.cos(a) * rr;
        const z = cz + Math.sin(a) * rr;
        const pick = rng.next();
        if (pick < 0.62) gravel.pushTRS(new THREE.Vector3(x, rng.range(0.01, 0.06), z), rng.range(0, 6.28), rng.range(0.5, 1.7));
        else if (pick < 0.86) chunks.pushTRS(new THREE.Vector3(x, 0.055, z), rng.range(0, 6.28), rng.range(0.5, 1.3));
        else litter.pushTRS(new THREE.Vector3(x, 0.012, z), rng.range(0, 6.28), rng.range(0.8, 1.4));
      }
      // A sand fillet banked against the lee face.
      this.add(
        'sand',
        sandDrift(rad * 2.0, rng.range(0.13, 0.24), 1.1, rng, 10),
        trs(cx + Math.cos(lee) * rad * 0.85, 0.008, cz + Math.sin(lee) * rad * 0.85, -lee + Math.PI * 0.5),
      );
      // And, half the time, something somebody left leaning on it.
      if (rng.chance(0.5)) {
        const a = lee + rng.range(-1.0, 1.0);
        this.inst('pallet', 'wood', () => Props.pallet(), false).pushTRS(
          new THREE.Vector3(cx + Math.cos(a) * rad * 1.05, 0.35, cz + Math.sin(a) * rad * 1.05),
          a + Math.PI * 0.5,
          1,
          1.35,
        );
      }
      if (rng.chance(0.4)) {
        const a = lee + rng.range(-1.4, 1.4);
        this.inst('tyre', 'rubber', () => Props.tyre(), false).pushTRS(
          new THREE.Vector3(cx + Math.cos(a) * rad * 1.15, 0, cz + Math.sin(a) * rad * 1.15),
          rng.range(0, 3.1),
        );
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Ground detail
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The scatter pass: gravel, litter, loose cable, sacks, boxes, sand fillets.
   *
   * The governing rule is that debris ACCUMULATES AT OBSTRUCTIONS. Wind and
   * foot traffic sweep an open yard clean and pile everything against the first
   * vertical thing they meet, so the scatter here is driven by a list of edges
   * and corners, with density falling off sharply with distance from them.
   * Uniform scatter across the courtyard is the single most recognisable
   * signature of a level that was dressed by a script, and it also happens to
   * look like confetti.
   *
   * All of it is instanced and none of it casts a shadow: at 20 cm the shadow
   * is two pixels and costs a full shadow-map draw across three cascades.
   */
  private scatterGroundDetail(): void {
    const rng = this.rng;
    const gravel = this.inst('gravel', 'rock', () => rockGeometry(0.075, rng, 0), false);
    const litter = this.inst('litter', 'paper', () => Props.litterScrap(rng), false);
    const boxes = this.inst('cardboard', 'paper', () => Props.cardboardBox(0.46), false);
    const sacks = this.inst('sack', 'fabric', () => Props.sack(rng), false);
    const bricks = this.inst('brick', 'concrete', () => Props.brick(), false);
    const chunks = this.inst('chunk', 'concrete', () => Props.rubbleChunk(0.22, rng), false);

    /**
     * Edges that debris collects against: [ax, az, bx, bz, weight]. These are
     * the *inward* faces of everything the player walks past — building feet,
     * lane walls, kerbs. Weight scales the amount.
     */
    const edges: Array<[number, number, number, number, number]> = [
      [-38.7, -42, -38.7, 42, 1.3],   // west terrace foot, alley side
      [-33.2, -26, -33.2, 17, 1.1],   // alley east wall
      [38.8, -42, 38.8, 42, 1.2],     // east terrace foot, market side
      [30.2, -6, 30.2, 10, 0.9],      // market west wall
      [30.2, 20, 30.2, 40, 0.9],
      [-20.8, -38, -20.8, -26, 0.8],  // radio house east
      [-34, -25.8, -21, -25.8, 1.0],  // radio house south
      [14, -26.5, -21, -26.5, 1.1],   // courtyard gate wall, north side
      [-21, -25.5, 14, -25.5, 1.0],   // ... and south side
      [6.8, 12, 6.8, 26, 0.9],        // market hall west
      [7, 11.6, 21, 11.6, 0.9],       // market hall north
      [-34, 16.6, -21, 16.6, 1.2],    // ruin north
      [-20.8, 17, -20.8, 30, 1.0],    // ruin east
      [-8.2, 29.6, 4, 29.6, 0.8],     // south block north
      [-44.4, -44.4, 44.4, -44.4, 0.9], // perimeter, north inner face
      [-44.4, 44.4, 44.4, 44.4, 0.9],   // perimeter, south inner face
      [14.2, -38, 14.2, -27, 0.7],    // garage west
      [-21, 11.7, 6, 11.7, 0.8],      // courtyard south low wall
    ];

    for (const [ax, az, bx, bz, weight] of edges) {
      const len = Math.hypot(bx - ax, bz - az);
      const dx = (bx - ax) / len;
      const dz = (bz - az) / len;
      // Perpendicular, pointing to the a→b left side (matching `facade`).
      const px = -dz;
      const pz = dx;
      const n = Math.round(len * 2.4 * weight);
      for (let i = 0; i < n; i++) {
        const t = rng.next();
        // Squared falloff: most of the debris is within 40 cm of the wall.
        const off = 0.12 + 2.1 * rng.next() * rng.next() * (rng.chance(0.82) ? 1 : -1);
        const x = ax + (bx - ax) * t + px * off;
        const z = az + (bz - az) * t + pz * off;
        const pick = rng.next();
        if (pick < 0.5) {
          gravel.pushTRS(new THREE.Vector3(x, rng.range(0.01, 0.05), z), rng.range(0, 6.28), rng.range(0.5, 1.6));
        } else if (pick < 0.72) {
          chunks.pushTRS(new THREE.Vector3(x, 0.055, z), rng.range(0, 6.28), rng.range(0.5, 1.3));
        } else if (pick < 0.88) {
          bricks.pushTRS(new THREE.Vector3(x, 0.033, z), rng.range(0, 6.28), 1, rng.range(-0.12, 0.12), rng.range(-0.12, 0.12));
        } else {
          litter.pushTRS(new THREE.Vector3(x, 0.012, z), rng.range(0, 6.28), rng.range(0.8, 1.5));
        }
      }
      // Sand fillet along the same edge: a soft wind-blown slope where the
      // ground meets the wall. Two triangles of this per metre is the cheapest
      // realism in the entire map — a perfect 90-degree ground/wall junction is
      // the thing that most reliably says "render".
      if (weight > 0.85) this.driftAlong(bx, bz, ax, az, rng.range(0.14, 0.26), rng.range(0.9, 1.5));
    }

    // ── Corner middens ──────────────────────────────────────────────────────
    // Interior corners are where wind-blown rubbish actually ends up, so each
    // one gets a genuine little heap rather than the ambient dusting above.
    const corners: Array<[number, number]> = [
      [-38.4, -25.6], [-38.4, 16.4], [-33.5, -25.6], [-33.5, 16.6],
      [38.5, -26.4], [38.5, 11.6], [30.5, -5.6], [30.5, 20.4],
      [-20.6, -25.6], [7.2, 11.9], [-20.6, 17.3], [14.4, -26.3],
      [-44.1, -20], [-44.1, 26], [44.1, -14], [44.1, 22],
      [-8.0, 29.9], [4.0, 29.9], [-20.6, 29.6],
    ];
    for (const [cx, cz] of corners) {
      const n = rng.int(14, 26);
      for (let i = 0; i < n; i++) {
        const a = rng.range(0, Math.PI * 2);
        const rr = rng.range(0.1, 1.5) * rng.next();
        const x = cx + Math.cos(a) * rr;
        const z = cz + Math.sin(a) * rr;
        const pick = rng.next();
        if (pick < 0.55) gravel.pushTRS(new THREE.Vector3(x, rng.range(0.02, 0.09), z), rng.range(0, 6.28), rng.range(0.6, 1.9));
        else if (pick < 0.8) chunks.pushTRS(new THREE.Vector3(x, 0.06, z), rng.range(0, 6.28), rng.range(0.5, 1.4));
        else litter.pushTRS(new THREE.Vector3(x, 0.014, z), rng.range(0, 6.28), rng.range(0.8, 1.6));
      }
      if (rng.chance(0.55)) boxes.pushTRS(new THREE.Vector3(cx + rng.range(-0.6, 0.6), 0, cz + rng.range(-0.6, 0.6)), rng.range(0, 6.28), rng.range(0.75, 1.2));
      if (rng.chance(0.45)) {
        for (let i = 0; i < rng.int(1, 3); i++) {
          sacks.pushTRS(new THREE.Vector3(cx + rng.range(-0.7, 0.7), 0, cz + rng.range(-0.7, 0.7)), rng.range(0, 6.28), rng.range(0.8, 1.15));
        }
      }
    }

    // ── Gravel patches where the paving has failed ──────────────────────────
    // Real hardstanding breaks up at its edges and around anything heavy that
    // has been parked on it. Dense circular patches read as potholes.
    for (const [cx, cz, rad] of [
      [-9.0, -22.0, 2.6], [4.5, -12.0, 2.2], [-17.0, -6.0, 2.4], [11.0, 2.5, 2.0],
      [-4.0, 6.0, 2.8], [34.0, -20.0, 2.2], [34.5, 14.0, 2.4], [-36.0, 3.0, 1.9],
      [-2.0, -33.5, 2.6], [22.0, -33.0, 2.2], [-28.0, 25.0, 2.3], [8.0, 22.0, 2.0],
    ] as const) {
      const n = Math.round(rad * rad * 9);
      for (let i = 0; i < n; i++) {
        const a = rng.range(0, Math.PI * 2);
        const rr = rad * Math.sqrt(rng.next());
        gravel.pushTRS(
          new THREE.Vector3(cx + Math.cos(a) * rr, rng.range(0.005, 0.045), cz + Math.sin(a) * rr),
          rng.range(0, 6.28),
          rng.range(0.35, 1.15),
        );
      }
      this.grunge(
        new THREE.Vector3(cx, 0.02, cz),
        new THREE.Vector3(0, 1, 0),
        'stain',
        rad * 2.1,
        rng.range(0, Math.PI),
        rng.range(0.3, 0.55),
      );
    }

    // ── Loose cable snaking out of the buildings ────────────────────────────
    for (const [x, z, ry, len] of [
      [-16.0, -28.5, 0.3, 5.0], [26.0, -18.0, 1.4, 4.2], [-30.0, 4.0, 1.6, 3.6],
      [10.0, 14.0, 0.2, 4.6], [-6.0, 31.0, 1.1, 3.8],
    ] as const) {
      this.add('rubber', Props.groundCable(len, rng), trs(x, 0, z, ry));
    }

    // ── Litter drifting across the open ground, sparse and directional ──────
    // A little scatter away from walls stops the middle of the courtyard being
    // conspicuously *clean*, but it stays sparse: the wind has been through.
    for (let i = 0; i < 90; i++) {
      const x = rng.range(-32, 30);
      const z = rng.range(-34, 34);
      if (Math.abs(x + 2) < 5 && Math.abs(z + 4) < 5) continue; // keep the tower base clear
      litter.pushTRS(new THREE.Vector3(x, 0.012, z), rng.range(0, 6.28), rng.range(0.7, 1.3));
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Flush: merge, instance, tag, register
  // ═══════════════════════════════════════════════════════════════════════════

  private flush(ctx: EngineContext): void {
    const lib = this.lib;
    let tris = 0;

    // ── Merged static geometry ──────────────────────────────────────────────
    for (const { key, mesh } of this.batcher.build((k) => this.material(k))) {
      const def = BUCKETS[key] ?? DEFAULT_BUCKET;
      mesh.castShadow = key !== 'sand' && key !== 'glass';
      mesh.receiveShadow = true;
      // Glass must not write depth or the transparent pass sorts wrongly, and
      // it must not block the shadow of the frame it sits in.
      if (key === 'glass') {
        mesh.castShadow = false;
        mesh.renderOrder = 1;
      }
      mesh.frustumCulled = true;
      if (lib) lib.tag(mesh, def.surface);
      this.root.add(mesh);
      this.collidables.push(mesh);
      tris += triCount(mesh.geometry);
      this.stats.merged++;
    }

    // ── Instanced props ─────────────────────────────────────────────────────
    for (const [, entry] of this.instances) {
      const def = BUCKETS[entry.matKey] ?? DEFAULT_BUCKET;
      const mesh = entry.builder.build(this.material(entry.matKey), entry.shadow, true);
      if (!mesh) continue;
      if (lib) lib.tag(mesh, def.surface);
      this.root.add(mesh);
      if (entry.collide) this.collidables.push(mesh);
      tris += triCount(mesh.geometry) * mesh.count;
      this.stats.instanced++;
    }

    this.stats.drawCalls = this.stats.merged + this.stats.instanced;
    this.stats.triangles = Math.round(tris);
    this.batcher.dispose();
    void ctx;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Spawns
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Ten spawn points, every one verified clear of building footprints and prop
   * clusters, and all on the flat part of the ground plane.
   *
   * They form two loose team clusters at opposite corners (north-west / north
   * and south-east / south) plus neutral positions at the lane mouths, which is
   * the arrangement a respawn director needs: it can always find a point that
   * is far from the enemy but still close to the action, so the map never
   * degenerates into a single spawn-trapped corner.
   */
  private defineSpawnPoints(): void {
    const pts: Array<[number, number]> = [
      [-36.5, -34.0],  // alley, north end
      [-36.5, 34.0],   // alley, south end
      [-16.0, -32.0],  // north yard, west — behind the gate wall
      [6.0, -33.5],    // north yard, east — by the garage apron
      [34.5, -34.0],   // market street, north
      [34.5, 32.0],    // market street, south
      [10.5, 38.0],    // south connector, east
      [-14.0, 38.5],   // south connector, west
      [-27.0, 6.0],    // west courtyard edge, by the middle alley breach
      [24.0, 4.5],     // east courtyard edge, by the market gate
    ];
    for (const [x, z] of pts) {
      // 0.05 m above the paving so a capsule controller resolves downward onto
      // the floor rather than starting interpenetrated with it.
      this.spawnPoints.push(new THREE.Vector3(x, 0.05, z));
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════

  private report(): void {
    console.info(
      `[Level] Compound Zulu built — ${this.stats.drawCalls} draw calls ` +
        `(${this.stats.merged} merged, ${this.stats.instanced} instanced), ` +
        `~${(this.stats.triangles / 1000).toFixed(0)}k triangles, ` +
        `${this.collidables.length} colliders, ${this.spawnPoints.length} spawns.`,
    );
  }

  /** Draw calls and triangles contributed by the level, for the perf overlay. */
  get renderStats(): Readonly<{ drawCalls: number; triangles: number; merged: number; instanced: number }> {
    return this.stats;
  }
}
