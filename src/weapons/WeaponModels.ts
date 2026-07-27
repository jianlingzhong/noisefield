/**
 * NOISEFIELD — Procedural weapon geometry.
 *
 * Zero external assets: every weapon is constructed from Three.js primitives at
 * load time.
 *
 * DESIGN RULES THAT KEEP THIS FROM LOOKING LIKE PROGRAMMER ART
 * ------------------------------------------------------------
 * 1. NOTHING IS A RAW BoxGeometry. Every rectangular part goes through
 *    `beveledBox`, which is an `ExtrudeGeometry` of a rounded-rect profile with
 *    a bevel on both caps. Chamfers here run from 0.1 mm on a reticle bar to
 *    5.5 mm on a gloved palm, sized to the part; even a sub-millimetre chamfer
 *    catches a specular highlight along every edge, and that highlight is 90%
 *    of what makes a shape read as machined metal rather than as a cube.
 * 2. ROUND THINGS ARE LATHED. Barrels, scope tubes, turrets and muzzle devices
 *    are `LatheGeometry` from an authored radius profile, so they have real
 *    steps, tapers and crowns instead of being uniform cylinders.
 * 3. SMALL REPEATED DETAIL IS MERGED. The rifle alone carries 23 rail ridges,
 *    8 hand-guard slats, 6 accessory cut-outs, 5 grip grooves and 5 flash-hider
 *    slots; the pistol adds 11 slide serrations. Each cluster is modelled part
 *    by part and then merged into ONE BufferGeometry, so all of that detail
 *    costs a handful of draw calls instead of dozens.
 * 4. REAL PROPORTIONS. Sight height over bore on the rifle is 60 mm (bore at
 *    y = 0.052, red-dot axis at y = 0.112), within a couple of millimetres of a
 *    real AR-pattern rifle with a reflex on a flat-top upper. Getting numbers
 *    like this right is why the silhouette reads as a gun at a glance.
 *
 * COORDINATE CONVENTION (weapon-local):
 *    +X right, +Y up, −Z forward (matches the camera).
 *    The ORIGIN is the firing hand / grip, because that is the point a
 *    viewmodel should rotate about and the natural anchor for hip/ADS posing.
 */

import * as THREE from 'three';
import { mergeGeometries, toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import type { EngineContext, IMaterialLibrary, WeaponId } from '../core/Types';

// ─────────────────────────────────────────────────────────────────────────────
// Rig description
// ─────────────────────────────────────────────────────────────────────────────

export interface WeaponRig {
  /** Everything, parented so the ViewModel can transform one object. */
  readonly root: THREE.Group;
  /** Empty at the muzzle crown, −Z forward. Tracer / flash origin. */
  readonly muzzle: THREE.Object3D;
  /** Empty at the ejection port, +X out. Shell spawn point. */
  readonly ejectPort: THREE.Object3D;
  /**
   * Empty sitting exactly on the sight's optical axis. The ADS pose is solved
   * from this, never hand-tuned — see the "Solve the ADS pose from the OPTIC"
   * block in `ViewModel.setWeapon`.
   */
  readonly opticAim: THREE.Object3D;
  /** Detaches and falls during a reload. */
  readonly magazine: THREE.Object3D;
  /** Pulled rearward on an empty reload / bolt cycle. */
  readonly chargingHandle: THREE.Object3D | null;
  /** How far, in metres, the charging handle travels. */
  readonly chargingTravel: number;
  /** Rotates on the trigger pull. */
  readonly trigger: THREE.Object3D | null;
  /** Reciprocates on each shot (slide / bolt carrier / pump). */
  readonly bolt: THREE.Object3D | null;
  /** Local axis the bolt travels along, magnitude = travel distance. */
  readonly boltTravel: THREE.Vector3;
  /** The illuminated aiming dot, so the ViewModel can flicker it. */
  readonly dot: THREE.Mesh | null;
  /**
   * Dot + glow. Drawn with `depthTest: false` so the sight body can never
   * occlude the reticle while aiming — which also means it would draw straight
   * through the weapon during a reload or an inspect, so the ViewModel hides
   * this group whenever the sight is not being looked through.
   */
  readonly reticle: THREE.Object3D | null;
  /** Support-hand group; ridden along by the handguard during animation. */
  readonly leftArm: THREE.Object3D;
  /** Firing-hand group. */
  readonly rightArm: THREE.Object3D;
  /**
   * Rest transforms of every animated part, captured at BUILD time.
   *
   * Snapshotting these from the live scene graph instead — e.g. when a weapon
   * is re-selected — risks baking a mid-animation pose in as the new "rest",
   * after which every subsequent restore compounds the error. Capturing once,
   * from pristine geometry, makes that class of bug impossible.
   */
  readonly rest: RigRest;
}

export interface RigRest {
  readonly magazine: THREE.Vector3;
  readonly leftArm: THREE.Vector3;
  readonly chargingHandle: THREE.Vector3;
  readonly bolt: THREE.Vector3;
  readonly triggerX: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Materials
// ─────────────────────────────────────────────────────────────────────────────

interface GunMaterials {
  metal: THREE.MeshStandardMaterial;
  metalDark: THREE.MeshStandardMaterial;
  metalWorn: THREE.MeshStandardMaterial;
  polymer: THREE.MeshStandardMaterial;
  polymerFDE: THREE.MeshStandardMaterial;
  rubber: THREE.MeshStandardMaterial;
  cavity: THREE.MeshStandardMaterial;
  glass: THREE.MeshPhysicalMaterial;
  dot: THREE.MeshBasicMaterial;
  glow: THREE.SpriteMaterial;
  brass: THREE.MeshStandardMaterial;
  /** Back-of-hand / palm fabric of the tactical glove. */
  gloveTan: THREE.MeshStandardMaterial;
  /** Reinforced knuckle and finger armour — dark, slightly rubbery. */
  gloveBlack: THREE.MeshStandardMaterial;
  /** Grippy palm-print silicone. Darker than the fabric, much rougher. */
  glovePalm: THREE.MeshStandardMaterial;
  /** Rolled combat-shirt sleeve where the forearm leaves the frame. */
  sleeve: THREE.MeshStandardMaterial;
}

let MATERIALS: GunMaterials | null = null;
let GLOW_TEX: THREE.Texture | null = null;
let FLASH_TEX: THREE.Texture | null = null;
/** A texture cloned off a borrowed library material; ours to release. */
let CLONED_MAP: THREE.Texture | null = null;
/** Procedural cloth detail maps, shared by every glove/sleeve material. */
let FABRIC_NORMAL: THREE.Texture | null = null;
let FABRIC_ALBEDO: THREE.Texture | null = null;
let FABRIC_ROUGH: THREE.Texture | null = null;
/** Same cloth image, retiled for revolved (normalised-UV) geometry. */
let SLEEVE_NORMAL: THREE.Texture | null = null;
let SLEEVE_ROUGH: THREE.Texture | null = null;
let SLEEVE_ALBEDO: THREE.Texture | null = null;
/** Micro-scratch detail for gun metal — breaks up the "wet plastic" read. */
let WEAR_ROUGH: THREE.Texture | null = null;

/** Deterministic 2D value hash in [0,1). Seeded so builds are reproducible. */
function hash2(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/** Smoothed value noise, so the fabric grain has structure rather than static. */
function valueNoise(x: number, y: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi);
  const b = hash2(xi + 1, yi);
  const c = hash2(xi, yi + 1);
  const d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/**
 * The same value noise, but with its integer lattice wrapped at (px, py) so the
 * field is SEAMLESS when the texture repeats.
 *
 * `valueNoise` is not. Sampling it over u ∈ [0,1) at 26 cells reads lattice
 * points 0…26, and hash2(26, y) has nothing to do with hash2(0, y) — so every
 * tile boundary is a hard discontinuity in the height field, which becomes a
 * hard crease in the normal map. At the 30 tiles/metre these maps are used at
 * that is a visible grid every 33 mm running the length of the forearm, which
 * is a large part of why the sleeve read as a woven hose rather than as cloth.
 * Wrapping the lattice costs two modulos per axis and removes the grid.
 */
function tileNoise(x: number, y: number, px: number, py: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const x0 = ((xi % px) + px) % px;
  const y0 = ((yi % py) + py) % py;
  const x1 = (x0 + 1) % px;
  const y1 = (y0 + 1) % py;
  const a = hash2(x0, y0);
  const b = hash2(x1, y0);
  const c = hash2(x0, y1);
  const d = hash2(x1, y1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

/**
 * Turn a periodic height field into a tangent-space normal map and a matching
 * roughness map.
 *
 * Wrapped differencing (`% S`) rather than clamped, because every one of these
 * fields is tiled — a clamped edge derivative puts a bright seam at the tile
 * boundary that mipmapping then smears into a visible band.
 */
function heightToMaps(
  height: Float32Array,
  S: number,
  strength: number,
  roughBase: number,
  roughGain: number,
): { n: Uint8Array; r: Uint8Array } {
  const n = new Uint8Array(S * S * 4);
  const r = new Uint8Array(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const l = height[y * S + ((x + S - 1) % S)];
      const rt = height[y * S + ((x + 1) % S)];
      const d = height[((y + S - 1) % S) * S + x];
      const t = height[((y + 1) % S) * S + x];
      let nx = (l - rt) * strength;
      let ny = (d - t) * strength;
      let nz = 1;
      const inv = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);
      nx *= inv;
      ny *= inv;
      nz *= inv;
      n[i * 4] = Math.round((nx * 0.5 + 0.5) * 255);
      n[i * 4 + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      n[i * 4 + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      n[i * 4 + 3] = 255;
      const rough = Math.round(THREE.MathUtils.clamp(roughBase - height[i] * roughGain, 110, 255));
      r[i * 4] = rough;
      r[i * 4 + 1] = rough;
      r[i * 4 + 2] = rough;
      r[i * 4 + 3] = 255;
    }
  }
  return { n, r };
}

/**
 * Finish a detail texture: wrap, tile, and — the part that actually mattered —
 * FILTER IT.
 *
 * `DataTexture` defaults to `NearestFilter` on both mag and min and to
 * `generateMipmaps = false`. A nearest-sampled 128² image tiled thirty times
 * across a forearm is one texel per two or three pixels at viewmodel distance,
 * which is textbook aliasing: the weave beats against the pixel grid and
 * resolves as a coarse regular moiré running around the limb. That moiré — not
 * the weave itself — is what read as corduroy or as a knitted sock. Trilinear
 * minification plus anisotropy makes the same weave settle into the fine matte
 * sheen it is supposed to be, and costs nothing at this texture size.
 */
function finishDetailTexture(t: THREE.DataTexture, repeatU: number, repeatV: number): void {
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeatU, repeatV);
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  // Clamped to the device maximum by three on upload, so 8 is safe everywhere.
  t.anisotropy = 8;
  t.needsUpdate = true;
}

/**
 * Build the shared cloth detail maps from one height field.
 *
 * WHY THIS EXISTS. A glove lit by a MeshStandardMaterial with a flat albedo and
 * a constant roughness has exactly one shading cue — the Lambert term — so at
 * viewmodel distance it reads as moulded vinyl no matter what colour it is.
 * That is precisely the "pool noodle" failure. A normal map with a plain-weave
 * over/under pattern plus fibre noise gives the surface a specular break-up
 * that survives at 30 cm from the lens, and the matching roughness map makes
 * the weave crests catch a slightly sharper highlight than the troughs. Two
 * 128² textures buy more perceived quality here than any amount of extra
 * polygons on the arm silhouette.
 *
 * Returns null in a headless environment (no canvas); every call site treats
 * the maps as optional, so the material simply falls back to flat shading.
 */
function buildFabricMaps(): void {
  if (FABRIC_NORMAL || typeof document === 'undefined') return;
  const S = 128;
  const TAU = Math.PI * 2;
  const height = new Float32Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const v = y / S;
      // Plain weave: two out-of-phase ripples crossing at right angles. 18
      // repeats over the tile puts a thread pitch of roughly 1.9 mm on a
      // glove, which is about right for a nylon-backed tactical glove.
      const weave = Math.sin(u * TAU * 18) * 0.5 + Math.sin(v * TAU * 18) * 0.5;
      // Fibre noise at two octaves so the weave is not mechanically perfect.
      // TILEABLE octaves — see `tileNoise`; the un-wrapped version put a hard
      // crease at every tile boundary, i.e. a 33 mm grid over the whole glove.
      const fibre = tileNoise(u * 26, v * 26, 26, 26) * 0.6 + tileNoise(u * 61, v * 61, 61, 61) * 0.4;
      height[y * S + x] = weave * 0.34 + (fibre - 0.5) * 0.66;
    }
  }

  // Strength is in height-units per texel; 2.4 gives a visible weave without
  // the plastic-shrink-wrap look an over-driven normal map produces.
  const maps = heightToMaps(height, S, 2.4, 196, 46);

  // ── TILING ─────────────────────────────────────────────────────────────────
  // `ExtrudeGeometry` — which every `beveledBox` is — generates its UVs from
  // the shape's coordinates, so on these parts one UV unit is ONE METRE. A
  // finger is 20 mm long, so at repeat 6 the whole finger got 0.12 of a tile:
  // a single 9 mm-pitch "thread" smeared across it. 30 puts the 18-thread tile
  // in a 33 mm square and the weave lands at ~1.9 mm.
  const METRE_TILE = 30;
  const nTex = new THREE.DataTexture(maps.n, S, S, THREE.RGBAFormat);
  finishDetailTexture(nTex, METRE_TILE, METRE_TILE);
  FABRIC_NORMAL = nTex;

  const rTex = new THREE.DataTexture(maps.r, S, S, THREE.RGBAFormat);
  finishDetailTexture(rTex, METRE_TILE, METRE_TILE);
  FABRIC_ROUGH = rTex;

  /*
   * Glove albedo.
   *
   * Same omission the sleeve had: a normal map and a roughness map over a
   * single flat `color`, which gives a surface that is correctly textured and
   * still reads as moulded plastic, because nothing varies its VALUE. On the
   * hands this was the most visible remaining weakness — the fingers came out
   * a uniform pale tan, closer to marzipan than to a worn nomex glove.
   *
   * The amplitude here is deliberately lower than the sleeve's. This map shares
   * the extruded parts' UVs, where one unit is a metre and the tile lands at
   * 33 mm, so any large soft blotch would repeat about every finger segment and
   * read as tiling. What it carries instead is thread-scale variation — the
   * weave's own light and shade, plus fibre noise — which is exactly the sort
   * of detail that SHOULD repeat at that pitch, with only a gentle broad term
   * on top.
   */
  const BASE_R = 0x57;
  const BASE_G = 0x4d;
  const BASE_B = 0x38;
  const albedo = new Uint8Array(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const u = x / S;
      const v = y / S;
      // Broad term kept subtle for the tiling reason above.
      const broad = tileNoise(u * 3, v * 3, 3, 3);
      const h = THREE.MathUtils.clamp(height[i] * 0.85 + 0.5, 0, 1);
      // Raised threads catch light and abrade pale; the valleys between them
      // hold dirt. That single relationship is most of what separates woven
      // fabric from a painted surface.
      const shade = THREE.MathUtils.clamp(0.84 + (h - 0.5) * 0.34 + (broad - 0.5) * 0.13, 0.5, 1.22);
      const o = i * 4;
      albedo[o] = THREE.MathUtils.clamp(BASE_R * shade, 0, 255);
      albedo[o + 1] = THREE.MathUtils.clamp(BASE_G * shade, 0, 255);
      albedo[o + 2] = THREE.MathUtils.clamp(BASE_B * shade * 0.98, 0, 255);
      albedo[o + 3] = 255;
    }
  }
  const aTex = new THREE.DataTexture(albedo, S, S, THREE.RGBAFormat);
  finishDetailTexture(aTex, METRE_TILE, METRE_TILE);
  aTex.colorSpace = THREE.SRGBColorSpace;
  aTex.needsUpdate = true;
  FABRIC_ALBEDO = aTex;
}

/**
 * Sleeve cloth. A SEPARATE map from the glove, at a genuinely different scale.
 *
 * WHY IT CANNOT BE THE GLOVE MAP RETILED. The old code cloned the glove image
 * and set `repeat = (11, 14)`. `CylinderGeometry` normalises its UVs to 0..1
 * over the whole surface, so that repeat means "eleven tiles around and
 * fourteen along, WHATEVER LENGTH the cylinder is" — the 10 cm lower forearm
 * and the 30 cm shirt therefore wore threads at 0.4 mm and 1.2 mm
 * respectively, and neither matched the glove they ran into at the cuff. There
 * was no scale in metres anywhere in the chain, which is exactly how a weave
 * ends up reading as corduroy on one part and as smooth vinyl on the next.
 *
 * So this map is authored for a fixed PHYSICAL tile — `SLEEVE_TILE_M` — and
 * every sleeve mesh gets metre-space UVs from `metreTubeUV`. What it contains
 * is deliberately not more weave: a very fine, very low-amplitude twill that
 * dissolves into a matte sheen by 40 cm, plus soft low-frequency CREASING that
 * carries the visual interest instead. Cloth at arm's length reads as folds,
 * not as threads.
 */
const SLEEVE_TILE_M = 0.075;

function buildSleeveMaps(): void {
  if (SLEEVE_NORMAL || typeof document === 'undefined') return;
  const S = 256;
  const TAU = Math.PI * 2;
  const height = new Float32Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const v = y / S;
      // 40 threads over a 75 mm tile → a 1.9 mm pitch, matched to the glove so
      // the two garments look woven at the same gauge across the cuff line.
      // Amplitude is a fifth of the glove's: at 40 cm this must not resolve.
      const weave = Math.sin(u * TAU * 40) * 0.5 + Math.sin(v * TAU * 40) * 0.5;
      // Twill diagonal — real ripstop is not a square grid, and a diagonal
      // beats less badly against a horizontal pixel grid than a square one.
      const twill = Math.sin((u + v) * TAU * 20) * 0.42;
      /*
       * Ripstop reinforcement grid.
       *
       * This one feature was single-handedly making the forearm read as a
       * ribbed sock. `pow(|sin|, 14)` is a very narrow spike, and six of them
       * per 75 mm tile puts a sharp ridge every 12.5 mm; over a 250 mm forearm
       * that is twenty hard bands wrapping the limb, amplified further by a
       * 1.3 normal scale. Real ripstop at arm's length is a faint sheen
       * difference you have to look for, not corduroy.
       *
       * Exponent 14 → 5 widens each line from a spike into a soft thread
       * bundle, and the amplitude drops by two thirds. It is still there — it
       * still catches the light as a grid when the arm turns — it just no
       * longer dominates the garment.
       */
      const rip =
        Math.pow(Math.abs(Math.sin(u * TAU * 6)), 5) + Math.pow(Math.abs(Math.sin(v * TAU * 6)), 5);
      /*
       * Creasing, now carrying the silhouette instead of the ripstop.
       *
       * Three octaves rather than two, and the lowest is stretched hard along
       * the limb (3 cells around, 4 along) so it produces a few big soft folds
       * of the kind a rolled sleeve actually forms over a forearm, rather than
       * an even sprinkle of small wrinkles.
       */
      const crease =
        tileNoise(u * 3, v * 4, 3, 4) * 0.52 +
        tileNoise(u * 3, v * 8, 3, 8) * 0.30 +
        tileNoise(u * 7, v * 17, 7, 17) * 0.18;
      const micro = tileNoise(u * 96, v * 96, 96, 96);
      height[y * S + x] =
        (weave + twill) * 0.085 + rip * 0.035 + (crease - 0.5) * 1.75 + (micro - 0.5) * 0.10;
    }
  }

  const maps = heightToMaps(height, S, 3.4, 202, 34);
  const tiles = 1 / SLEEVE_TILE_M;

  const nTex = new THREE.DataTexture(maps.n, S, S, THREE.RGBAFormat);
  finishDetailTexture(nTex, tiles, tiles);
  SLEEVE_NORMAL = nTex;

  const rTex = new THREE.DataTexture(maps.r, S, S, THREE.RGBAFormat);
  finishDetailTexture(rTex, tiles, tiles);
  SLEEVE_ROUGH = rTex;

  /*
   * Albedo.
   *
   * The sleeve previously had no colour map at all — one flat `color` plus a
   * per-vertex shade gradient. A normal map alone cannot rescue that, because
   * the thing that makes worn cloth look like cloth is VALUE VARIATION: dust
   * ground into the weave, sun-bleaching on the outer face, dirt sitting in the
   * bottom of every crease. Without it the limb reads as moulded rubber with a
   * texture stamped on, which is precisely how it was reading.
   *
   * Authored directly in sRGB bytes around the base garment colour, so the
   * material's own `color` goes to white and this map carries the tint. The
   * dominant term is deliberately the same crease field that drives the normal
   * map, so the shading and the staining agree with each other — dirt in a
   * fold that the geometry also shades is what sells the fold as real.
   */
  const BASE_R = 0x4a;
  const BASE_G = 0x4b;
  const BASE_B = 0x3c;
  const albedo = new Uint8Array(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const u = x / S;
      const v = y / S;
      // Dust and bleaching, at a far larger scale than the weave so it reads as
      // history rather than as another texture.
      const blotch = tileNoise(u * 2, v * 5, 2, 5) * 0.66 + tileNoise(u * 5, v * 11, 5, 11) * 0.34;
      const h = THREE.MathUtils.clamp(height[i] * 0.9 + 0.5, 0, 1);
      const shade = THREE.MathUtils.clamp(0.82 + (blotch - 0.5) * 0.34 + (h - 0.5) * 0.28, 0.42, 1.28);
      // Bleaching lifts the raised, sun-facing threads slightly yellow-grey,
      // while shaded creases stay cooler — a small hue shift, but it is the
      // difference between "dirty cloth" and "flat colour with noise on it".
      const warm = (h - 0.5) * 14;
      const o = i * 4;
      albedo[o] = THREE.MathUtils.clamp(BASE_R * shade + warm, 0, 255);
      albedo[o + 1] = THREE.MathUtils.clamp(BASE_G * shade + warm * 0.85, 0, 255);
      albedo[o + 2] = THREE.MathUtils.clamp(BASE_B * shade + warm * 0.45, 0, 255);
      albedo[o + 3] = 255;
    }
  }
  const aTex = new THREE.DataTexture(albedo, S, S, THREE.RGBAFormat);
  finishDetailTexture(aTex, tiles, tiles);
  // Colour data, unlike the normal and roughness maps this shares a helper with.
  aTex.colorSpace = THREE.SRGBColorSpace;
  aTex.needsUpdate = true;
  SLEEVE_ALBEDO = aTex;
}

/**
 * Micro-wear roughness for gun metal.
 *
 * The receiver previously rendered as uniform wet-looking plastic with hard
 * bright streaks, which is what a single roughness constant plus a borrowed
 * world-scale albedo tiled 14× does to a 4 cm part. Replacing that with a
 * per-texel roughness field breaks the specular into scuffs and polish marks —
 * the "this thing has been carried" cue — without touching albedo at all.
 */
function buildWearMap(): void {
  if (WEAR_ROUGH) return;
  const S = 128;
  const data = new Uint8Array(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const u = x / S;
      const v = y / S;
      // Long, shallow, mostly-horizontal polish streaks (anisotropic noise),
      // plus fine speckle for blasted phosphate.
      const streak = valueNoise(u * 5, v * 44);
      const speckle = valueNoise(u * 90, v * 90);
      const g = THREE.MathUtils.clamp(0.62 + (streak - 0.5) * 0.55 + (speckle - 0.5) * 0.28, 0.22, 1);
      const b = Math.round(g * 255);
      const i = (y * S + x) * 4;
      data[i] = b;
      data[i + 1] = b;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  // Extruded weapon parts also carry metre-scale UVs, so at repeat 3 the wear
  // pattern was a 33 cm blotch — on a 9 cm cheek pad, one enormous smear that
  // read as camouflage. 18 puts the scuff scale at ~5 cm, the size of a real
  // rub mark, and it stays coherent across adjacent parts because the tiling is
  // in world units rather than per-object.
  tex.repeat.set(18, 18);
  tex.needsUpdate = true;
  WEAR_ROUGH = tex;
}

/**
 * Build a small radial-gradient texture on a 2D canvas. Procedural, so there
 * are still zero external assets, but it gives the flash and the reticle glow a
 * soft falloff that a flat quad cannot.
 */
function radialTexture(size: number, spikes: number, spikeGain: number): THREE.Texture {
  // `spikes` draws radial streaks; the reticle glow passes 0 for a plain bloom.
  if (typeof document === 'undefined') {
    // Headless safety net — a 1×1 white texture still renders something.
    const t = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    t.needsUpdate = true;
    return t;
  }
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const g = canvas.getContext('2d');
  if (!g) {
    const t = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    t.needsUpdate = true;
    return t;
  }
  const c = size / 2;
  g.clearRect(0, 0, size, size);

  // Core glow.
  const grad = g.createRadialGradient(c, c, 0, c, c, c);
  grad.addColorStop(0.0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.18, 'rgba(255,244,205,0.95)');
  grad.addColorStop(0.42, 'rgba(255,176,72,0.45)');
  grad.addColorStop(1.0, 'rgba(255,120,30,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);

  // Star spikes — a muzzle flash photographed at 1/250 s has hard radial
  // streaks, and they are what sells the "one frame of light" read.
  if (spikes > 0) {
    g.globalCompositeOperation = 'lighter';
    for (let i = 0; i < spikes; i++) {
      const a = (i / spikes) * Math.PI * 2;
      g.save();
      g.translate(c, c);
      g.rotate(a);
      const lg = g.createLinearGradient(0, 0, c * spikeGain, 0);
      lg.addColorStop(0, 'rgba(255,240,200,0.9)');
      lg.addColorStop(1, 'rgba(255,150,40,0)');
      g.fillStyle = lg;
      g.beginPath();
      g.moveTo(0, -size * 0.028);
      g.lineTo(c * spikeGain, 0);
      g.lineTo(0, size * 0.028);
      g.closePath();
      g.fill();
      g.restore();
    }
    g.globalCompositeOperation = 'source-over';
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Shared material set. Cached at module scope because five weapons and a dozen
 * spawned props all draw from it, and swapping weapons must not churn the GPU.
 *
 * If a material library system is registered we borrow its 'metal' entry as a
 * base — it will have the project's authored roughness/normal maps — but we
 * clone it and retune both the PBR constants and the texture tiling, because a
 * world-scale material repeats once per metre and would land a single texel on
 * a 4 cm receiver.
 */
function getMaterials(ctx: EngineContext): GunMaterials {
  if (MATERIALS) return MATERIALS;

  buildFabricMaps();
  buildSleeveMaps();
  buildWearMap();

  // The material library is consulted only to confirm it exists; nothing is
  // borrowed from it any more. A world material's ALBEDO is authored for square
  // metres of wall, and tiling it onto a 4 cm receiver was the direct cause of
  // the bright diagonal streaks that made the gun read as wet plastic. Its
  // NORMAL map is no better — a brick or plaster relief at 8× on a receiver
  // reads as a checkerboard. Colour comes from a constant and break-up comes
  // from `WEAR_ROUGH`, which is authored at gun scale.
  const lib = ctx.get<IMaterialLibrary>('materials');
  void lib;

  // WHY THIS IS NOT A MIRROR. A metalness-1 surface has NO diffuse term at all:
  // its albedo only tints what it reflects. Under this scene's golden-hour
  // environment — a tan desert under a warm sky — a dark grey mirror renders
  // TAN, which is exactly how the first pass came out. Real gun finishes are
  // dark because phosphate and ceramic gun coatings are genuinely rough and only partly
  // metallic. Metalness 0.72 restores enough diffuse for the albedo to read,
  // roughness 0.56 blurs the environment into a sheen instead of a picture of
  // the sky, and envMapIntensity 0.45 keeps the whole thing anchored dark.
  const metal = new THREE.MeshStandardMaterial({
    color: 0x26282c,
    metalness: 0.72,
    roughness: 0.56,
    roughnessMap: WEAR_ROUGH,
    envMapIntensity: 0.45,
  });
  metal.name = 'vm_metal';

  MATERIALS = {
    metal,
    // Parkerised / phosphate finish: darker and rougher than bare steel. The
    // wear map is shared with `metal` so the two read as the same weapon that
    // has been finished twice, not as two unrelated surfaces.
    // Roughness 0.80: this material covers the top rail, whose 23 cross-ridges
    // present 23 near-perpendicular facets to a very strong key. At 0.66 they
    // returned a continuous blown-out streak down the whole receiver — the
    // single brightest thing in frame and pure noise.
    metalDark: new THREE.MeshStandardMaterial({
      color: 0x191b1e,
      metalness: 0.34,
      roughness: 0.88,
      roughnessMap: WEAR_ROUGH,
      envMapIntensity: 0.22,
    }),
    // Edge wear — where the finish has rubbed through to bright steel. Used on
    // charging handles, bolt faces and the leading edges of controls. Kept
    // genuinely polished (0.26) because this material's entire job is to be the
    // one thing on the weapon that throws a hard highlight.
    metalWorn: new THREE.MeshStandardMaterial({
      color: 0x4a4e55,
      metalness: 0.85,
      roughness: 0.38,
      roughnessMap: WEAR_ROUGH,
      envMapIntensity: 0.55,
    }),
    polymer: new THREE.MeshStandardMaterial({
      color: 0x1a1b1f,
      metalness: 0.0,
      roughness: 0.66,
      roughnessMap: WEAR_ROUGH,
      envMapIntensity: 0.5,
    }),
    // FDE: deliberately desaturated. Saturated tan reads as plastic toy.
    // The engine's own viewmodel key light runs at 4.68, so a mid-tone albedo
    // renders roughly 2.3× brighter in sRGB than it is authored. FDE is picked
    // against THAT: 0x3d3729 comes back on screen as the muted grey-brown that
    // "flat dark earth" actually is, where 0x8a7a5c came back as gold.
    polymerFDE: new THREE.MeshStandardMaterial({
      color: 0x3d3729,
      metalness: 0.0,
      roughness: 0.78,
      roughnessMap: WEAR_ROUGH,
      envMapIntensity: 0.20,
    }),
    rubber: new THREE.MeshStandardMaterial({
      color: 0x101114,
      metalness: 0.0,
      roughness: 0.92,
      envMapIntensity: 0.25,
    }),
    // Interior of ports, slots and vents. Near-black and matte so recesses read
    // as depth rather than as painted-on decals.
    cavity: new THREE.MeshStandardMaterial({
      color: 0x08090b,
      metalness: 0.3,
      roughness: 0.85,
      envMapIntensity: 0.2,
    }),
    // Anti-reflective coated glass. Deliberately NOT `transmission`, which
    // would force a whole extra render target for a 2 cm lens.
    glass: new THREE.MeshPhysicalMaterial({
      color: 0x21323d,
      metalness: 0.0,
      roughness: 0.04,
      transparent: true,
      opacity: 0.26,
      clearcoat: 1.0,
      clearcoatRoughness: 0.03,
      side: THREE.DoubleSide,
      depthWrite: false,
      envMapIntensity: 2.0,
    }),
    // `depthTest: false` is deliberate and load-bearing. An aiming reticle is a
    // virtual image projected at infinity — it is never occluded by the sight
    // body in reality, and in a renderer any hood lip, ring or lens that ends
    // up nearer the eye would silently swallow it. Combined with a high
    // `renderOrder` on the dot meshes, the reticle is always drawn.
    dot: new THREE.MeshBasicMaterial({
      color: 0xff2a12,
      toneMapped: false, // must survive ACES or it goes orange and dull
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
    }),
    glow: new THREE.SpriteMaterial({
      map: (GLOW_TEX ??= radialTexture(64, 0, 0)),
      color: 0xff3b1a,
      toneMapped: false,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
      opacity: 0.55,
    }),
    brass: new THREE.MeshStandardMaterial({
      color: 0x9c7a34,
      metalness: 1.0,
      roughness: 0.28,
      envMapIntensity: 1.3,
    }),
    // ── Glove and sleeve ─────────────────────────────────────────────────────
    // Deliberately DESATURATED and fairly dark. The old value read as a bright
    // uniform tan, which is the single strongest "untextured primitive" signal
    // there is: real coyote-brown nylon under a warm key light sits far closer
    // to a mid olive-brown than to the saturated sand people expect.
    gloveTan: new THREE.MeshStandardMaterial({
      // White: the garment colour now lives in FABRIC_ALBEDO, together with the
      // thread-level value variation that stops the glove reading as plastic.
      color: 0xffffff,
      map: FABRIC_ALBEDO,
      metalness: 0.0,
      roughness: 0.80,
      normalMap: FABRIC_NORMAL,
      normalScale: new THREE.Vector2(0.85, 0.85),
      roughnessMap: FABRIC_ROUGH,
      envMapIntensity: 0.40,
    }),
    // Knuckle / finger armour. Not pure black — a black that dark loses every
    // form cue at viewmodel distance and the fingers merge into one mass.
    gloveBlack: new THREE.MeshStandardMaterial({
      color: 0x26272b,
      metalness: 0.0,
      roughness: 0.68,
      normalMap: FABRIC_NORMAL,
      normalScale: new THREE.Vector2(0.55, 0.55),
      envMapIntensity: 0.38,
    }),
    // Silicone palm print: the grippiest, least reflective thing on the hand.
    glovePalm: new THREE.MeshStandardMaterial({
      color: 0x1b1c1f,
      metalness: 0.0,
      roughness: 0.94,
      normalMap: FABRIC_NORMAL,
      normalScale: new THREE.Vector2(1.1, 1.1),
      envMapIntensity: 0.22,
    }),
    // Combat shirt. Still cooler and greyer than the glove so the cuff line
    // where the two meet is legible as a *garment boundary* rather than as a
    // seam in one continuous tube — but pulled well back from the saturated
    // olive it was, which at viewmodel size read as a green sock rather than
    // as a shirt sleeve in the shooter's own shadow.
    //
    // `vertexColors` is the important one. The sleeve is the ONLY material that
    // uses it, and every mesh in `buildArm` writes a per-vertex shade that
    // darkens with distance from the wrist. That is not decoration: the far end
    // of a forearm is tucked into the shooter's own body shadow, and baking
    // that gradient is what stops the limb reading as an evenly-lit tube
    // extending forever off the bottom of the frame.
    sleeve: new THREE.MeshStandardMaterial({
      // White, because the garment colour now lives in SLEEVE_ALBEDO along with
      // the dust and crease staining that stops it reading as flat rubber.
      color: 0xffffff,
      map: SLEEVE_ALBEDO,
      metalness: 0.0,
      roughness: 0.95,
      normalMap: SLEEVE_NORMAL,
      // Was 1.30. With the ripstop spikes softened, the folds carry the surface,
      // and 1.30 was over-driving them into a quilted look.
      normalScale: new THREE.Vector2(1.05, 1.05),
      roughnessMap: SLEEVE_ROUGH,
      envMapIntensity: 0.16,
      vertexColors: true,
    }),
  };
  return MATERIALS;
}

/** Release the shared material/texture cache. */
export function disposeSharedMaterials(): void {
  if (MATERIALS) {
    for (const key of Object.keys(MATERIALS) as Array<keyof GunMaterials>) {
      MATERIALS[key].dispose();
    }
    MATERIALS = null;
  }
  GLOW_TEX?.dispose();
  GLOW_TEX = null;
  FABRIC_NORMAL?.dispose();
  FABRIC_NORMAL = null;
  FABRIC_ALBEDO?.dispose();
  FABRIC_ALBEDO = null;
  FABRIC_ROUGH?.dispose();
  FABRIC_ROUGH = null;
  SLEEVE_NORMAL?.dispose();
  SLEEVE_NORMAL = null;
  SLEEVE_ROUGH?.dispose();
  SLEEVE_ROUGH = null;
  SLEEVE_ALBEDO?.dispose();
  SLEEVE_ALBEDO = null;
  WEAR_ROUGH?.dispose();
  WEAR_ROUGH = null;
  FLASH_TEX?.dispose();
  FLASH_TEX = null;
  CLONED_MAP?.dispose();
  CLONED_MAP = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Star-burst texture for the muzzle flash: a hot core plus six hard radial
 * streaks. Shared and cached, and disposed with the rest of the weapon assets.
 */
export function muzzleFlashTexture(): THREE.Texture {
  return (FLASH_TEX ??= radialTexture(128, 6, 0.98));
}

/** Rounded rectangle in XY, centred on the origin. */
function roundedRect(w: number, h: number, r: number): THREE.Shape {
  const x = w * 0.5;
  const y = h * 0.5;
  const rr = Math.max(1e-5, Math.min(r, Math.min(x, y) * 0.95));
  const s = new THREE.Shape();
  s.moveTo(-x + rr, -y);
  s.lineTo(x - rr, -y);
  s.absarc(x - rr, -y + rr, rr, -Math.PI / 2, 0, false);
  s.lineTo(x, y - rr);
  s.absarc(x - rr, y - rr, rr, 0, Math.PI / 2, false);
  s.lineTo(-x + rr, y);
  s.absarc(-x + rr, y - rr, rr, Math.PI / 2, Math.PI, false);
  s.lineTo(-x, -y + rr);
  s.absarc(-x + rr, -y + rr, rr, Math.PI, Math.PI * 1.5, false);
  return s;
}

/**
 * Crease threshold for `toCreasedNormals`, in radians.
 *
 * 50° sits above the ~30° facet-to-facet step inside a fillet arc and well
 * below the 90° of a true box corner, which is the whole window we need.
 *
 * Declared ahead of its first use because module-level geometry construction
 * would otherwise hit it in the temporal dead zone.
 */
const CREASE_ANGLE = THREE.MathUtils.degToRad(50);

/**
 * THE workhorse. A box with a chamfer on every edge, centred on the origin.
 *
 * Implemented as an extruded rounded rect with a bevel on both caps, so the
 * bevel wraps the profile silhouette AND the two end faces — i.e. all twelve
 * edges. `w/h/d` are the FINAL outer dimensions including the bevel, so call
 * sites can reason in real millimetres.
 */
export function beveledBox(w: number, h: number, d: number, bevel = 0.0022, corner?: number): THREE.BufferGeometry {
  const b = Math.max(1e-4, Math.min(bevel, Math.min(w, h, d) * 0.22));
  const cr = corner ?? Math.min(w, h) * 0.18;
  const shape = roundedRect(Math.max(2e-4, w - 2 * b), Math.max(2e-4, h - 2 * b), cr);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(1e-4, d - 2 * b),
    bevelEnabled: true,
    bevelThickness: b,
    bevelSize: b,
    // 3 bevel segments and 5 curve segments, up from 2 and 3. The silhouette of
    // a rounded corner is only as round as its segment count, and at viewmodel
    // distance a 3-segment quarter circle is visibly a chamfered triangle.
    bevelSegments: 3,
    curveSegments: 5,
    steps: 1,
  });
  geo.center();

  /*
   * Re-derive the normals with a crease threshold.
   *
   * This is the single most important line in the file, and its absence was
   * quietly cancelling design rule #1 at the top of this module.
   *
   * `ExtrudeGeometry` emits NON-INDEXED geometry and finishes by calling
   * `computeVertexNormals()`. On non-indexed geometry that function has no
   * shared vertices to average across, so it writes the face normal to all
   * three corners of every triangle — i.e. it produces flat shading, on every
   * part, no matter that `flatShading` is left false everywhere. The fillet we
   * pay for on all twelve edges therefore rendered as two or three more flat
   * facets instead of as the soft rolling highlight that makes a shape read as
   * machined metal. Everything built through here — the receiver, the rail, the
   * handguard, the gloved fingers — came out looking like faceted cardboard.
   *
   * `toCreasedNormals` welds coincident vertices and averages their normals,
   * but only where the angle between the adjoining faces is under the
   * threshold. At 50° the ~30° steps within a fillet arc and the fillet-to-face
   * transition are smoothed, while a genuine 90° box corner stays hard — so
   * parts keep their crisp machined edges and gain a real chamfer highlight,
   * which is exactly the distinction the design rule was asking for.
   */
  const creased = toCreasedNormals(geo, CREASE_ANGLE);
  geo.dispose();
  return creased;
}

interface BoxSpec {
  w: number;
  h: number;
  d: number;
  bevel?: number;
  corner?: number;
  x?: number;
  y?: number;
  z?: number;
  rx?: number;
  ry?: number;
  rz?: number;
}

const _mat4 = new THREE.Matrix4();
const _euler = new THREE.Euler();
const _quat = new THREE.Quaternion();
const _vec = new THREE.Vector3();
const _one = new THREE.Vector3(1, 1, 1);

/**
 * Merge a cluster of chamfered boxes into one BufferGeometry.
 *
 * This is what makes fine detail affordable. The rifle's top rail alone is 23
 * cross-ridges; its hand-guard is 8 slats and 6 accessory cut-outs; the pistol
 * slide is 11 serrations. Modelled individually that is dozens of extra draw
 * calls on the object the player stares at for an entire match; merged, each
 * cluster is one. All inputs are ExtrudeGeometry (non-indexed, identical
 * attribute set), so the merge is always valid.
 */
function mergedBoxes(specs: readonly BoxSpec[]): THREE.BufferGeometry {
  const geos: THREE.BufferGeometry[] = [];
  for (const s of specs) {
    const g = beveledBox(s.w, s.h, s.d, s.bevel, s.corner);
    _euler.set(s.rx ?? 0, s.ry ?? 0, s.rz ?? 0);
    _quat.setFromEuler(_euler);
    _vec.set(s.x ?? 0, s.y ?? 0, s.z ?? 0);
    _mat4.compose(_vec, _quat, _one);
    g.applyMatrix4(_mat4);
    geos.push(g);
  }
  if (geos.length === 1) return geos[0];
  const merged = mergeGeometries(geos, false);
  for (const g of geos) g.dispose();
  return merged;
}

/**
 * Lathe a radius profile about the −Z axis.
 *
 * `LatheGeometry` spins about +Y, so we author `[radius, distance]` pairs with
 * distance increasing forward and then rotate −90° about X, which maps +Y → −Z.
 */
function latheForward(profile: ReadonlyArray<readonly [number, number]>, segments = 20): THREE.BufferGeometry {
  const pts: THREE.Vector2[] = [];
  for (const [r, d] of profile) pts.push(new THREE.Vector2(Math.max(1e-4, r), d));
  const g = new THREE.LatheGeometry(pts, segments);
  g.rotateX(-Math.PI / 2);
  g.computeVertexNormals();
  return g;
}

interface Placement {
  pos?: readonly [number, number, number];
  rot?: readonly [number, number, number];
  name?: string;
}

function addMesh(
  parent: THREE.Object3D,
  geo: THREE.BufferGeometry,
  mat: THREE.Material,
  place?: Placement,
): THREE.Mesh {
  const m = new THREE.Mesh(geo, mat);
  if (place?.pos) m.position.set(place.pos[0], place.pos[1], place.pos[2]);
  if (place?.rot) m.rotation.set(place.rot[0], place.rot[1], place.rot[2]);
  if (place?.name) m.name = place.name;
  m.castShadow = false;
  m.receiveShadow = false;
  // Culling is disabled rather than relied upon. Viewmodel parts are always a
  // few centimetres from `viewCamera` and always on screen, so per-object
  // frustum tests are pure overhead — and parts that animate outside their
  // build-time bounds (the dropped magazine, the muzzle flash) would otherwise
  // pop out when their stale bounding sphere leaves the frustum.
  m.frustumCulled = false;
  parent.add(m);
  return m;
}

function node(parent: THREE.Object3D, name: string, x: number, y: number, z: number): THREE.Object3D {
  const o = new THREE.Object3D();
  o.name = name;
  o.position.set(x, y, z);
  parent.add(o);
  return o;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reusable sub-assemblies
// ─────────────────────────────────────────────────────────────────────────────

/** MIL-STD-1913 rail: a base bar with regularly spaced cross ridges. */
function railGeometry(length: number, width = 0.021): THREE.BufferGeometry {
  const specs: BoxSpec[] = [{ w: width, h: 0.0055, d: length, bevel: 0.0008, corner: 0.0012 }];
  const pitch = 0.0198; // real Picatinny slot pitch is 10 mm; 2× reads better at this scale
  const count = Math.max(1, Math.floor(length / pitch));
  const start = -length * 0.5 + pitch * 0.5;
  for (let i = 0; i < count; i++) {
    specs.push({
      w: width * 0.98,
      h: 0.0042,
      d: 0.0072,
      bevel: 0.0009,
      corner: 0.0009,
      y: 0.0048,
      z: start + i * pitch,
    });
  }
  return mergedBoxes(specs);
}

/** A ring of slats around a bore axis — the modern free-float hand-guard look. */
function slattedShroud(
  radius: number,
  length: number,
  count: number,
  slatW: number,
  slatH: number,
  startAngle = 0,
  sweep = Math.PI * 2,
): THREE.BufferGeometry {
  const specs: BoxSpec[] = [];
  for (let i = 0; i < count; i++) {
    const a = startAngle + (i / count) * sweep;
    specs.push({
      w: slatW,
      h: slatH,
      d: length,
      bevel: 0.0009,
      corner: 0.0012,
      x: Math.cos(a) * radius,
      y: Math.sin(a) * radius,
      rz: a - Math.PI / 2,
    });
  }
  return mergedBoxes(specs);
}

/**
 * slotted accessory mounts, drawn as near-black inlays sitting a fraction of a
 * millimetre proud of the surface they sit on.
 *
 * They cannot be true recesses: there is no CSG here, so a hole cannot be cut
 * out of a solid prism, and an insert placed INSIDE the surface is simply
 * occluded by it. Sitting a hair proud is the only way a dark opening reads at
 * all — the same trick the ejection ports use.
 */
function slotStrip(count: number, pitch: number, w: number, h: number, d: number): THREE.BufferGeometry {
  const specs: BoxSpec[] = [];
  const start = -((count - 1) * pitch) * 0.5;
  for (let i = 0; i < count; i++) {
    specs.push({ w, h, d, bevel: 0.0006, corner: 0.0008, z: start + i * pitch });
  }
  return mergedBoxes(specs);
}

/** Grip stippling / slide serrations: a run of thin ridges, raked by `tilt`
 *  (radians about X) for the swept look of modern cocking serrations. */
function serrationStrip(
  count: number,
  pitch: number,
  w: number,
  h: number,
  d: number,
  tilt: number,
  axis: 'y' | 'z' = 'z',
): THREE.BufferGeometry {
  const specs: BoxSpec[] = [];
  const start = -((count - 1) * pitch) * 0.5;
  for (let i = 0; i < count; i++) {
    const o = start + i * pitch;
    specs.push({ w, h, d, bevel: 0.0005, corner: 0.0006, y: axis === 'y' ? o : 0, z: axis === 'z' ? o : 0, rx: tilt });
  }
  return mergedBoxes(specs);
}

/**
 * A pistol grip: the body plus its finger grooves, as one rotated group.
 *
 * Built as a group so the grooves are authored in GRIP-LOCAL space. Placing
 * them as siblings of the body meant every one had to be hand-rotated into the
 * grip's cant, and every weapon got it wrong in the same way — the grooves were
 * laid out along Z (the grip's 42 mm depth) when a finger groove obviously
 * stacks along Y (its 86 mm height), so a 68 mm strip ended up floating in
 * front of the grip with nothing behind it.
 */
function addGrip(
  parent: THREE.Object3D,
  bodyMat: THREE.Material,
  grooveMat: THREE.Material,
  o: {
    w: number;
    h: number;
    d: number;
    x?: number;
    y: number;
    z: number;
    rx: number;
    grooves: number;
    pitch: number;
  },
): THREE.Group {
  const g = new THREE.Group();
  g.position.set(o.x ?? 0, o.y, o.z);
  g.rotation.x = o.rx;
  parent.add(g);
  addMesh(g, beveledBox(o.w, o.h, o.d, 0.0040, 0.009), bodyMat);
  // Grooves ride the front face (−Z, where the fingers wrap), standing 1.5 mm
  // proud so they read as ridges rather than being swallowed by the body.
  addMesh(g, serrationStrip(o.grooves, o.pitch, o.w * 0.96, 0.0042, 0.0050, 0, 'y'), grooveMat, {
    pos: [0, 0, -o.d * 0.5 - 0.0005],
  });
  return g;
}

interface OpticParts {
  group: THREE.Group;
  aim: THREE.Object3D;
  dot: THREE.Mesh;
  /** Dot + glow. Hidden when the sight is not being looked through. */
  reticle: THREE.Group;
}

/**
 * Reflex red-dot sight.
 *
 * Built around the OPTICAL AXIS, a horizontal line at `axisY` (30 mm above the
 * mount base, scaled by `scale`). The `aim` node is placed exactly on that axis
 * at the emitter's virtual image, and the ADS pose is solved from it, so the
 * dot lands on the mathematical centre of the screen rather than "wherever a
 * magic offset happened to put it".
 */
function buildRedDot(m: GunMaterials, scale = 1): OpticParts {
  const g = new THREE.Group();
  g.name = 'optic';
  const s = scale;
  const axisY = 0.0300 * s;

  // Mount / recoil lug.
  addMesh(g, beveledBox(0.026 * s, 0.0075 * s, 0.048 * s, 0.0012 * s), m.metalDark, {
    pos: [0, 0.0038 * s, 0],
  });
  // Body.
  addMesh(g, beveledBox(0.030 * s, 0.014 * s, 0.044 * s, 0.0016 * s), m.metal, {
    pos: [0, 0.0135 * s, 0],
  });
  // Window uprights (left/right) and the top bridge form the hood.
  addMesh(g, beveledBox(0.0048 * s, 0.030 * s, 0.034 * s, 0.0011 * s), m.metal, {
    pos: [-0.0128 * s, axisY + 0.0025 * s, -0.001 * s],
  });
  addMesh(g, beveledBox(0.0048 * s, 0.030 * s, 0.034 * s, 0.0011 * s), m.metal, {
    pos: [0.0128 * s, axisY + 0.0025 * s, -0.001 * s],
  });
  addMesh(g, beveledBox(0.0304 * s, 0.0055 * s, 0.034 * s, 0.0011 * s), m.metal, {
    pos: [0, axisY + 0.0195 * s, -0.001 * s],
  });
  // Front and rear hood lips — the proud rings that protect the glass.
  //
  // These MUST be open frames, not slabs. The rear lip sits directly between
  // the shooter's eye and the emitter dot, so a solid plate there hides the
  // entire reticle while still looking perfectly correct from every angle
  // except the one that matters. The bars are kept outside the clear window
  // (|x| > 0.0104, y outside [axisY − 0.0065, axisY + 0.0168]).
  const lip = (): THREE.BufferGeometry =>
    mergedBoxes([
      { w: 0.0304 * s, h: 0.0035 * s, d: 0.0035 * s, bevel: 0.0008 * s, corner: 0.0009 * s, y: 0.0185 * s },
      { w: 0.0304 * s, h: 0.0035 * s, d: 0.0035 * s, bevel: 0.0008 * s, corner: 0.0009 * s, y: -0.0083 * s },
      { w: 0.0044 * s, h: 0.0303 * s, d: 0.0035 * s, bevel: 0.0008 * s, corner: 0.0009 * s, x: -0.0130 * s, y: 0.0051 * s },
      { w: 0.0044 * s, h: 0.0303 * s, d: 0.0035 * s, bevel: 0.0008 * s, corner: 0.0009 * s, x: 0.0130 * s, y: 0.0051 * s },
    ]);
  addMesh(g, lip(), m.metalDark, { pos: [0, axisY, -0.0175 * s] });
  addMesh(g, lip(), m.metalDark, { pos: [0, axisY, 0.0155 * s] });
  // Elevation and windage turrets.
  // +π/2 about X: `latheForward` extends along −Z, and +π/2 maps −Z to +Y so
  // the turret grows UP out of the body. −π/2 would bury it inside.
  addMesh(g, latheForward([[0.0042 * s, 0], [0.0042 * s, 0.005 * s], [0.0030 * s, 0.0062 * s]], 12), m.metalWorn, {
    pos: [0, axisY + 0.0165 * s, 0.019 * s],
    rot: [Math.PI / 2, 0, 0],
  });
  addMesh(g, latheForward([[0.0042 * s, 0], [0.0042 * s, 0.005 * s], [0.0030 * s, 0.0062 * s]], 12), m.metalWorn, {
    pos: [0.014 * s, 0.0135 * s, 0.019 * s],
    rot: [0, -Math.PI / 2, 0],
  });
  // Emitter housing under the glass.
  addMesh(g, beveledBox(0.008 * s, 0.006 * s, 0.008 * s, 0.0008 * s), m.cavity, {
    pos: [0, axisY - 0.0125 * s, 0.008 * s],
  });

  // Glass, canted 8° so the emitter's reflection is thrown to the eye.
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(0.0235 * s, 0.0272 * s), m.glass);
  glass.position.set(0, axisY + 0.0025 * s, -0.0125 * s);
  glass.rotation.x = -0.14;
  glass.frustumCulled = false;
  g.add(glass);

  // The dot itself. A tiny disc plus an additive glow sprite: the disc gives
  // the crisp centre, the sprite gives the bloom halo that makes it read as
  // emissive rather than as a red sticker.
  const reticle = new THREE.Group();
  reticle.name = 'reticle';
  g.add(reticle);

  const dot = new THREE.Mesh(new THREE.CircleGeometry(0.00085 * s, 12), m.dot);
  dot.position.set(0, axisY, -0.0118 * s);
  dot.frustumCulled = false;
  dot.renderOrder = 30;
  reticle.add(dot);

  const glow = new THREE.Sprite(m.glow);
  glow.position.copy(dot.position);
  glow.scale.setScalar(0.0085 * s);
  glow.frustumCulled = false;
  glow.renderOrder = 29;
  reticle.add(glow);

  const aim = node(g, 'opticAim', 0, axisY, -0.0118 * s);
  return { group: g, aim, dot, reticle };
}

/**
 * Magnified optic for the bolt gun. Same contract as the red dot: `aim` sits on
 * the optical axis, this time at the eyepiece so the ADS depth reads as a
 * plausible eye relief.
 */
function buildScope(m: GunMaterials): OpticParts {
  const g = new THREE.Group();
  g.name = 'optic';
  const axisY = 0.040;

  // Main tube, objective bell, eyepiece. One lathe, one authored profile.
  const tube = latheForward(
    [
      [0.0250, 0.000], // eyepiece rear face
      [0.0262, 0.008],
      [0.0258, 0.030],
      [0.0212, 0.042], // step down to the 31 mm main tube
      [0.0155, 0.050],
      [0.0155, 0.150],
      [0.0170, 0.158],
      [0.0170, 0.176],
      [0.0300, 0.196], // objective bell flare
      [0.0308, 0.232],
      [0.0296, 0.240],
    ],
    24,
  );
  addMesh(g, tube, m.metalDark, { pos: [0, axisY, 0.052], rot: [0, 0, 0] });

  // Turret housings. Elevation is +π/2 about X, NOT −π/2: `latheForward`
  // extends along −Z, so −π/2 maps it to −Y and grows the turret DOWN into the
  // tube — a solid metal plug across the entire sight picture that is invisible
  // from outside the scope. The windage and parallax knobs below use ∓π/2 about
  // Y, which is correct for their axes.
  addMesh(g, latheForward([[0.0135, 0], [0.0135, 0.014], [0.0112, 0.017], [0.0112, 0.021]], 16), m.metal, {
    pos: [0, axisY + 0.0140, -0.030],
    rot: [Math.PI / 2, 0, 0],
  });
  addMesh(g, latheForward([[0.0125, 0], [0.0125, 0.012], [0.0104, 0.015], [0.0104, 0.019]], 16), m.metal, {
    pos: [0.014, axisY, -0.030],
    rot: [0, -Math.PI / 2, 0],
  });
  // Parallax knob on the left.
  addMesh(g, latheForward([[0.0118, 0], [0.0118, 0.010], [0.0100, 0.013]], 14), m.metal, {
    pos: [-0.014, axisY, -0.030],
    rot: [0, Math.PI / 2, 0],
  });

  // Rings clamping the tube to the rail.
  //
  // A torus, NOT a box: the ring encircles a 31 mm tube and the shooter looks
  // straight down that bore, so a solid block centred on the axis would put a
  // metal wall across the middle of the sight picture. Inner radius 0.0155
  // matches the tube exactly; everything else stays below or above the axis.
  // The REAR ring (z = +0.006) needs the larger hole: the tube is still
  // tapering out of the eyepiece there (r 0.01835), so a 0.0155 hole would bury
  // 2.9 mm of it. The front ring sits on the parallel 0.0155 section.
  // Front hole is 0.0158, a hair over the 0.0155 tube: exactly equal would put
  // two surfaces coplanar and z-fight along the whole ring.
  for (const [z, hole] of [[0.006, 0.0190], [-0.078, 0.0158]] as const) {
    addMesh(g, new THREE.TorusGeometry(hole + 0.0050, 0.0050, 8, 20), m.metal, { pos: [0, axisY, z] });
    // Clamp screw strap across the TOP of the ring, clear of the bore.
    addMesh(g, beveledBox(0.030, 0.0055, 0.021, 0.0009, 0.0012), m.metalWorn, {
      pos: [0, axisY + 0.0250, z],
    });
    // Base block below the ring, tying it down to the rail.
    addMesh(g, beveledBox(0.026, 0.016, 0.020, 0.0012, 0.004), m.metalDark, {
      pos: [0, axisY - 0.0270, z],
    });
  }

  // Objective lens. It spans the full bore, so it MUST be the transparent
  // glass: with the opaque mirror material it only avoided blanking the entire
  // sight picture because its normal happened to face away from the eye and
  // three.js back-face culled it — one `side` change from a total blackout.
  addMesh(g, new THREE.CircleGeometry(0.0284, 28), m.glass, { pos: [0, axisY, -0.186] });
  // Ocular lens. This one sits between the shooter's eye and the reticle, so it
  // MUST be the transparent, non-depth-writing glass material — the mirror-like
  // `m.lens` renders in the opaque pass and would hide the reticle, the dot and
  // its glow completely, turning the entire sight into a dark disc.
  addMesh(g, new THREE.CircleGeometry(0.0234, 24), m.glass, { pos: [0, axisY, 0.0505] });

  // Reticle: a fine duplex cross drawn as four thin bars plus a centre dot,
  // sitting just inside the eyepiece so it parallaxes correctly with the tube.
  const reticleZ = 0.046;
  const bar = 0.0009;
  const duplex = mergedBoxes([
    { w: 0.0185, h: bar, d: 0.0004, bevel: 0.0001, corner: 0.0001, x: -0.0125 },
    { w: 0.0185, h: bar, d: 0.0004, bevel: 0.0001, corner: 0.0001, x: 0.0125 },
    { w: bar, h: 0.0185, d: 0.0004, bevel: 0.0001, corner: 0.0001, y: 0.0125 },
    { w: bar, h: 0.0185, d: 0.0004, bevel: 0.0001, corner: 0.0001, y: -0.0125 },
  ]);
  addMesh(g, duplex, m.cavity, { pos: [0, axisY, reticleZ] });

  // The illuminated centre. The duplex bars above are etched glass and stay
  // visible; only the emitter is switched off with the rest of the reticle.
  const reticle = new THREE.Group();
  reticle.name = 'reticle';
  g.add(reticle);

  const dot = new THREE.Mesh(new THREE.CircleGeometry(0.00062, 10), m.dot);
  dot.position.set(0, axisY, reticleZ + 0.0006);
  dot.frustumCulled = false;
  dot.renderOrder = 30;
  reticle.add(dot);

  const glow = new THREE.Sprite(m.glow);
  glow.position.copy(dot.position);
  glow.scale.setScalar(0.006);
  glow.frustumCulled = false;
  glow.renderOrder = 29;
  reticle.add(glow);

  // The aim node sits on the optical axis at the eyepiece. Eye relief itself
  // is expressed by `Handling.adsDepth` (0.115 m for this rifle), which is what
  // positions the whole weapon relative to the camera when aiming.
  const aim = node(g, 'opticAim', 0, axisY, 0.052);
  return { group: g, aim, dot, reticle };
}

/** Prong-style flash hider: a lathed body with radial gas slots. */
function buildFlashHider(
  parent: THREE.Object3D,
  m: GunMaterials,
  r: number,
  len: number,
  y: number,
  z: number,
): void {
  addMesh(
    parent,
    latheForward(
      [
        [r * 0.72, 0],
        [r, 0.004],
        [r, len * 0.28],
        [r * 0.94, len * 0.32],
        [r * 0.94, len * 0.9],
        [r * 1.02, len * 0.94],
        [r * 1.02, len],
        [r * 0.7, len],
      ],
      16,
    ),
    m.metalDark,
    { pos: [0, y, z] },
  );
  // Five gas slots across the top ~115° of the circumference, where a real
  // flash hider vents to counter muzzle rise. Modelled as dark recessed
  // inserts — cheaper than CSG and reads identically at this scale.
  const slots: BoxSpec[] = [];
  for (let i = 0; i < 5; i++) {
    const a = Math.PI * 0.18 + (i / 4) * Math.PI * 0.64;
    slots.push({
      w: r * 0.44,
      h: r * 0.5,
      d: len * 0.5,
      bevel: 0.0004,
      corner: 0.0006,
      x: Math.cos(a) * r * 0.86,
      y: Math.sin(a) * r * 0.86,
      rz: a - Math.PI / 2,
      z: -len * 0.42,
    });
  }
  addMesh(parent, mergedBoxes(slots), m.cavity, { pos: [0, y, z] });
}

/** Trigger guard loop plus the trigger blade itself (returned for animation). */
function buildTriggerGroup(parent: THREE.Object3D, m: GunMaterials, x: number, y: number, z: number): THREE.Object3D {
  const g = new THREE.Group();
  g.position.set(x, y, z);
  parent.add(g);

  // Guard: a torus arc rotated into the YZ plane so it loops fore-and-aft.
  const guard = new THREE.Mesh(
    new THREE.TorusGeometry(0.0195, 0.0034, 6, 18, Math.PI * 1.35),
    m.polymer,
  );
  guard.rotation.set(0, Math.PI / 2, Math.PI * 0.82);
  guard.position.set(0, -0.019, -0.004);
  guard.frustumCulled = false;
  g.add(guard);
  // Straight rear tie-in so the loop meets the receiver cleanly.
  addMesh(g, beveledBox(0.0062, 0.0062, 0.020, 0.0009, 0.0014), m.polymer, { pos: [0, -0.004, 0.012] });

  // Trigger blade — pivots about its top edge, so the pivot group sits there.
  const pivot = new THREE.Object3D();
  pivot.position.set(0, -0.004, -0.006);
  g.add(pivot);
  addMesh(pivot, beveledBox(0.0052, 0.017, 0.0055, 0.0008, 0.0012), m.metalWorn, {
    pos: [0, -0.0085, 0.0006],
    rot: [0.16, 0, 0],
  });
  return pivot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hands and arms
// ─────────────────────────────────────────────────────────────────────────────

/**
 * HUMAN ANATOMY, IN METRES.
 *
 * These are not stylistic choices, they are measurements, and they matter more
 * here than anywhere else in the file. A player has spent their whole life
 * looking at their own hands: a forearm that is 30% too thick or a finger that
 * is 40% too long is legible as WRONG instantly, even to someone who could not
 * say what is wrong. Every other part of the weapon gets the benefit of the
 * doubt because nobody has a rifle receiver memorised.
 *
 * Sources are ordinary adult-male anthropometry:
 *   • wrist 17.5 cm circumference → 7.4 cm across, and it is an OVAL: about
 *     5.1 cm thick front-to-back. A round wrist is the first half of the
 *     "length of hose" read;
 *   • forearm max circumference ~29 cm → 11.4 cm across, 8.8 cm thick, and it
 *     reaches that width at the ELBOW, not at the wrist. Running the elbow
 *     width down the whole limb — which is what a near-constant cylinder does
 *     — is the second half of the hose;
 *   • hand length (wrist crease → middle fingertip) 19 cm, of which the palm
 *     is 10.0 cm and the middle finger 9.0 cm;
 *   • hand breadth at the knuckles 8.5 cm, i.e. ~2.1 cm per finger row.
 */
const ANATOMY = {
  /** Half-width across the wrist, inside the glove cuff. 72 mm across. */
  wristHalf: 0.0360,
  /** Half-THICKNESS at the wrist. The wrist is markedly flat. 49 mm deep. */
  wristThin: 0.0245,
  /** Half-width at the widest part of the forearm. 108 mm across. */
  elbowHalf: 0.0540,
  /** Half-thickness at the same station. 84 mm deep. */
  elbowThin: 0.0420,
  /** Vertical pitch between finger rows: hand breadth 85 mm over four rows. */
  fingerPitch: 0.0212,
  /** Palm slab: 100 mm wrist-to-knuckle, 26 mm through a gloved hand. */
  palmLength: 0.1000,
  palmThick: 0.0262,
} as const;

/**
 * Phalanx lengths, proximal → distal, in metres, before the per-finger scale
 * below. Sums to 89 mm, a middle finger.
 */
const PHALANX: readonly number[] = [0.0420, 0.0260, 0.0210];
/** index, middle, ring, little — relative length. */
const FINGER_SCALE: readonly number[] = [0.94, 1.0, 0.93, 0.79];
/**
 * Finger widths. DELIBERATELY 2–5 mm NARROWER than the 21.2 mm row pitch.
 *
 * This is the single most important number on the hand and the previous two
 * passes both got it wrong in the same way — by reasoning that real fingers in
 * a closed fist touch, therefore the model's should too, therefore the gaps
 * should be closed with filler. That is true of the object and false of the
 * IMAGE. At 40 cm from a 60° lens a finger is about 30 px across; if adjacent
 * fingers share a face there is no silhouette break, no contact shadow and no
 * normal discontinuity between them, so four fingers composite into one tan
 * mitten. A 2 mm gap costs nothing anatomically — real fingers separate that
 * far the moment a grip is anything but maximally clenched — and it is the
 * whole difference between "hand" and "oven glove".
 */
const FINGER_WIDTH: readonly number[] = [0.0186, 0.0190, 0.0180, 0.0158];

/** Radial thickness of each phalanx, proximal → distal. Fingers taper. */
const PHALANX_THICK: readonly number[] = [0.0180, 0.0164, 0.0146];

/**
 * Lay one finger — three phalanges — around a column of semi-axes (a, b) whose
 * axis is the local +Y.
 *
 * θ is measured from the PALM SIDE (`handed`·+X) and sweeps toward −Z, the
 * front of the weapon. The surface point is (a·cosθ, b·sinθ) and each segment
 * is pushed out along the true ellipse normal by its own half-thickness, so a
 * phalanx rests ON the grip whatever section the call site supplies — 15 × 21
 * mm for an AR pistol grip, 29 × 29 for a handguard, 34 × 26 for a shotgun
 * pump. Segments that float off the grip, or sink into it, are the classic
 * tell of a procedural viewmodel.
 *
 * The total angular sweep is passed in rather than derived per segment, which
 * is what makes the curl controllable: a tight fist on a slim grip wraps 190°
 * and the fingertips come back at the palm, an open hand on a fat handguard
 * wraps 120° and the fingers lie along the top.
 */
/**
 * One finger, as a SINGLE swept tube rather than a chain of boxes.
 *
 * The previous approach placed three `beveledBox` phalanges along the grip
 * ellipse and overlapped them by 32% so the joints would not gap. The comments
 * around it record a long fight against the result reading as "three beads on a
 * string" — tuning the overlap, the chamfer radius and the knuckle caps to hide
 * it. None of that could work, because the problem is not shading, it is
 * SILHOUETTE: three convex boxes on a curve have three separate outlines, and
 * the eye reads the two waists between them as gaps no matter how the surface
 * is lit. Against a bright background the support hand looked like a robot claw
 * for exactly this reason.
 *
 * A real finger has one continuous outline that swells slightly at each knuckle
 * and rounds off at the tip, so that is what this builds: a tube swept along
 * the same elliptical arc the boxes used, with an elliptical cross-section whose
 * radial and lateral radii vary along the length. One surface, one silhouette,
 * smooth normals through the joints.
 *
 * Built indexed so `computeVertexNormals` averages across shared vertices and
 * the tube shades smoothly, then converted to non-indexed to match the
 * attribute layout of the extruded boxes it gets merged with.
 *
 * @param handed      +1 right hand, −1 left; mirrors the arc across X.
 * @param a,b         half-extents of the gripped column's ellipse.
 * @param y           the finger's row height.
 * @param theta0      angle at the knuckle.
 * @param sweep       total angle subtended from knuckle to tip.
 * @param width       lateral width of the finger at the knuckle.
 * @param lengthScale per-finger length multiplier. NOTE: this does NOT set the
 *                    tube's physical extent — that follows the arc, and the
 *                    caller has already folded the finger's length into
 *                    `sweep`. It is used only to put the UVs in metre space
 *                    so the glove weave runs at the same gauge as the
 *                    extruded parts it meets at the knuckle.
 */
export function sweptFinger(
  handed: number,
  a: number,
  b: number,
  y: number,
  theta0: number,
  sweep: number,
  width: number,
  lengthScale: number,
): THREE.BufferGeometry {
  // 26 rings is enough that the arc is smooth at viewmodel distance without the
  // tube costing more than the three boxes it replaces; 12 around is plenty for
  // a shape this small once the normals are smooth.
  const ALONG = 30;
  const AROUND = 12;

  const total = PHALANX[0] + PHALANX[1] + PHALANX[2];
  const totalLen = total * lengthScale;
  // Fractional positions of the two knuckle joints along the finger.
  const j1 = PHALANX[0] / total;
  const j2 = (PHALANX[0] + PHALANX[1]) / total;

  const tA = PHALANX_THICK[0] * 0.5;
  const tB = PHALANX_THICK[1] * 0.5;
  const tC = PHALANX_THICK[2] * 0.5;

  /** Radial (toward/away from the grip) half-thickness at length fraction s. */
  const radialAt = (s: number): number => {
    let r: number;
    if (s < j1) r = THREE.MathUtils.lerp(tA, tB, s / j1);
    else if (s < j2) r = THREE.MathUtils.lerp(tB, tC, (s - j1) / (j2 - j1));
    // Only a slight narrowing into the distal phalanx. Tapering hard here plus
    // the lateral taper plus the tip cap compounded into a cone, and the hand
    // read as though it had carrots on it.
    else r = THREE.MathUtils.lerp(tC, tC * 0.94, (s - j2) / (1 - j2));
    // Knuckles are the widest part of a finger. Two small gaussian swells at
    // the joints put the bumps back that the box chain got from its overlaps,
    // but as part of one continuous surface.
    const bump = (c: number, amt: number): number => amt * Math.exp(-(((s - c) / 0.055) ** 2));
    return r * (1 + bump(j1, 0.085) + bump(j2, 0.065));
  };

  // A real finger barely narrows between knuckle and tip; almost all of the
  // apparent taper comes from the fingertip rounding off.
  const wHalf = width * 0.5;
  /** Lateral (across the finger) half-width at length fraction s. */
  const lateralAt = (s: number): number => wHalf * (1 - 0.07 * s);

  /** Hemispherical rounding over the last 10% so the tip is a fingertip. */
  const tipAt = (s: number): number => {
    const TIP = 0.90;
    if (s <= TIP) return 1;
    const k = (s - TIP) / (1 - TIP);
    return Math.sqrt(Math.max(0, 1 - k * k));
  };

  const pos: number[] = [];
  const uv: number[] = [];
  const centres: THREE.Vector3[] = [];

  for (let i = 0; i < ALONG; i++) {
    const s = i / (ALONG - 1);
    const theta = theta0 + sweep * s;
    const c = Math.cos(theta);
    const sn = Math.sin(theta);
    // Outward normal of the ellipse — same construction the boxes used, so the
    // finger still sits on the surface it is gripping rather than on a circle.
    let nx = c / a;
    let nz = sn / b;
    const nl = Math.hypot(nx, nz) || 1;
    nx /= nl;
    nz /= nl;
    const rRad = radialAt(s) * tipAt(s);
    const rLat = lateralAt(s) * tipAt(s);
    // 1.2 mm of squash into the grip, matching the old placement.
    const off = radialAt(s) - 0.0012;
    const px = handed * (a * c + nx * off);
    const pz = -(b * sn + nz * off);
    centres.push(new THREE.Vector3(px, y, pz));

    // World-space frame: radial out, and Y across the finger.
    const ox = handed * nx;
    const oz = -nz;
    for (let k = 0; k <= AROUND; k++) {
      const phi = (k / AROUND) * Math.PI * 2;
      const cr = Math.cos(phi) * rRad;
      const cl = Math.sin(phi) * rLat;
      pos.push(px + ox * cr, y + cl, pz + oz * cr);
      // Metre-space UVs, consistent with the extruded parts, so the glove weave
      // runs at the same gauge across the whole hand.
      uv.push((k / AROUND) * (Math.PI * (rRad + rLat)), s * totalLen);
    }
  }

  const idx: number[] = [];
  const ring = AROUND + 1;
  for (let i = 0; i < ALONG - 1; i++) {
    for (let k = 0; k < AROUND; k++) {
      const p0 = i * ring + k;
      const p1 = p0 + 1;
      const p2 = p0 + ring;
      const p3 = p2 + 1;
      idx.push(p0, p2, p1, p1, p2, p3);
    }
  }

  // Cap the knuckle end. The base ring is at full radius and sits against the
  // metacarpal, but an open tube would show its own inside through the gap
  // between finger rows when the hand is seen from the side.
  const capCentre = pos.length / 3;
  pos.push(centres[0].x, centres[0].y, centres[0].z);
  uv.push(0, 0);
  for (let k = 0; k < AROUND; k++) idx.push(capCentre, k, k + 1);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const out = geo.toNonIndexed();
  geo.dispose();
  return out;
}

function wrapFinger(
  tubes: THREE.BufferGeometry[],
  armour: BoxSpec[],
  handed: number,
  a: number,
  b: number,
  y: number,
  theta0: number,
  sweep: number,
  width: number,
  lengthScale: number,
): void {
  const total = PHALANX[0] + PHALANX[1] + PHALANX[2];

  // The whole finger, as one surface.
  tubes.push(sweptFinger(handed, a, b, y, theta0, sweep, width, lengthScale));

  let theta = theta0;
  for (let i = 0; i < 3; i++) {
    const len = PHALANX[i] * lengthScale;
    const thick = PHALANX_THICK[i];
    const dTheta = sweep * (PHALANX[i] / total);
    const tm = theta + dTheta * 0.5;
    const c = Math.cos(tm);
    const sn = Math.sin(tm);
    // Outward normal of the ellipse, so the cap sits on the SURFACE rather
    // than on a circle that only touches it at four points.
    let nx = c / a;
    let nz = sn / b;
    const nl = Math.hypot(nx, nz) || 1;
    nx /= nl;
    nz /= nl;
    const off = thick * 0.5 - 0.0012; // 1.2 mm of squash into the grip
    const px = a * c + nx * off;
    const pz = b * sn + nz * off;
    // Align the cap's local +Z with the ellipse tangent at tm.
    const ry = Math.atan2(-handed * a * sn, -b * c);
    const w = width - i * 0.0011;
    // Dark dorsal cap on the proximal and middle phalanges only. Real hard-
    // knuckle gloves leave the fingertip in palm material, and — more to the
    // point — capping all three segments produces the alternating light/dark
    // banding that reads as a bicycle chain. Two caps and a tan tip reads as a
    // finger with a knuckle on it.
    if (i < 2) {
      armour.push({
        // Narrower than the phalanx it sits on, so it is a strip down the
        // middle of the knuckle rather than a lid over the whole segment —
        // and LONGER than it, so consecutive caps meet across the joint. A
        // continuous dark line from knuckle to second joint is what fuses
        // three boxes into one finger; caps that stop short of the joint leave
        // a light band at every hinge and the finger reads as a bead chain.
        w: 0.0028,
        h: w * 0.62,
        d: len * (i === 0 ? 1.24 : 1.20),
        bevel: 0.0008,
        corner: 0.0014,
        x: handed * (px + nx * (thick * 0.5 - 0.0006)),
        y,
        z: -(pz + nz * (thick * 0.5 - 0.0006)),
        ry,
      });
    }
    theta += dTheta;
  }
}

/** Parameters that differ between the firing hand and the support hand. */
interface HandOptions {
  /** +1 = palm on the +X side of the grip (right hand), −1 on −X. */
  handed: 1 | -1;
  /**
   * Half-width (X) and half-depth (Z) of the column the fingers close around.
   * Grips are NEVER round: an AR pistol grip is 30 mm across and 42 mm deep, a
   * handguard is near-circular, a shotgun pump is wider than it is tall.
   */
  gripHalfX: number;
  gripHalfZ: number;
  /** Index finger leaves the wrap and lies on a trigger instead. */
  trigger?: boolean;
  /** Trigger position in hand-local metres — where the fingertip must land. */
  triggerAt?: readonly [number, number, number];
  /** Scales how far round the column the fingers reach. 1 = a firm fist. */
  wrap?: number;
  /**
   * Thumb heading, radians, measured from the palm side and rolling toward the
   * front of the column. 0 lays the thumb straight across the front face,
   * larger values swing it round toward the far side.
   */
  thumbRoll?: number;
  /**
   * How far the thumb travels ALONG the column per unit of its own length.
   * A firing thumb climbs the backstrap and lies over the safety (+0.55); a
   * support thumb on a handguard points down the bore, which is the column
   * axis itself, so it wants a large negative value (−1.1) and almost no roll.
   * Without this the thumb can only move in the plane the fingers wrap in,
   * which on a barrel means it sticks out sideways like a hitch-hiker.
   */
  thumbAxial?: number;
  /**
   * Where the wrist leaves the hand, in hand-local metres. Defaults to the
   * heel of the palm, straight down the column — right for a PISTOL GRIP,
   * where the column axis and the wrist-to-knuckle axis are parallel.
   *
   * They are NOT parallel on a handguard. There the column is the barrel, the
   * four finger rows are stacked ALONG it, and the wrist leaves the hand
   * sideways, around the section. Left on the default, the support arm sprouts
   * eight centimetres down the barrel from the hand that is supposed to own it
   * — which is exactly what the second attempt did, and is most of why the
   * support hand looked detached from its own forearm.
   */
  wristAt?: readonly [number, number, number];
  /** Scales the palm slab along the column. Support hands want a shorter one. */
  palmScale?: number;
  /**
   * How far BELOW the middle-knuckle row the palm slab is centred, metres. On
   * a pistol grip the wrist is straight down the column, so the default 24.5 mm
   * is right. On a handguard the four rows are stacked along the barrel and the
   * palm has to sit centred on them, or the index finger hangs off the end of
   * its own hand.
   */
  palmDrop?: number;
}

/** How far a finger can curl before the tip is inside the palm. 195°. */
const SWEEP_MAX = 3.40;
/** An open hand laid over a wide foregrip never wraps less than this. 105°. */
const SWEEP_MIN = 1.83;

/**
 * A gloved hand closed on a column.
 *
 * The origin is ON THE COLUMN AXIS, level with the middle-knuckle row, +Y up
 * the column. That choice is what lets a caller mount the same hand on a
 * pistol grip, a handguard and a pump by supplying nothing but a section and a
 * transform.
 *
 * Everything here exists to serve the SILHOUETTE, because that is all that
 * survives at 45 cm from the lens: four SEPARATED finger ridges crossing the
 * grip, a thumb breaking the outline on the far side, dark knuckle caps
 * catching the key light, and a cuff that terminates the hand instead of
 * letting it blend into the arm.
 *
 * WHAT CHANGED, AND WHY. Two earlier passes filled the space between the
 * fingers with "fist core", "knuckle bridge" and "underside" slabs 82 mm tall
 * — a continuous wall across all four rows standing as proud as the phalanges
 * themselves. The intent was to stop the fingers reading as a floating chain;
 * the effect was to delete them, because a filler that reaches the finger
 * surface IS the finger surface as far as the silhouette is concerned. The
 * mass is still needed, so it is still here — but it lives INSIDE the wrap,
 * recessed 7 mm below the phalanx crowns and, critically, SPLIT PER FINGER
 * ROW so the gaps run all the way through to the palm.
 */
function buildHand(m: GunMaterials, o: HandOptions): THREE.Group {
  const g = new THREE.Group();
  const handed = o.handed;
  const a = o.gripHalfX;
  const b = o.gripHalfZ;
  const wrap = o.wrap ?? 1;

  const fabric: BoxSpec[] = [];
  const armour: BoxSpec[] = [];
  const palmMat: BoxSpec[] = [];
  // Fingers are swept tubes rather than boxes, so they arrive as finished
  // geometry and get merged into the glove mesh alongside the extruded parts.
  const fingerTubes: THREE.BufferGeometry[] = [];

  // Four finger rows, index highest.
  const rows = [1.5, 0.5, -0.5, -1.5].map((k) => k * ANATOMY.fingerPitch);

  // HOW FAR THE FINGERS ACTUALLY GO ROUND. Arc length over mean radius gives
  // the sweep a finger of a given length subtends on a column of a given size;
  // clamping it top and bottom keeps a slim pistol grip from wrapping the tips
  // through the back of the palm and a fat handguard from leaving them
  // pointing at the sky.
  const meanR = (a + b) * 0.5 + 0.0090;
  const fingerSum = PHALANX[0] + PHALANX[1] + PHALANX[2];

  for (let f = 0; f < 4; f++) {
    if (f === 0 && o.trigger) continue; // the index is doing something else
    // Little and ring start marginally further round and curl marginally
    // tighter, which is what stops four identical parallel bars reading as a
    // comb, and matches a real hand's staggered knuckle line.
    const theta0 = 0.24 + f * 0.055;
    const want = (fingerSum * FINGER_SCALE[f] * wrap) / meanR;
    const sweep = Math.min(SWEEP_MAX - theta0, Math.max(SWEEP_MIN, want));
    // If the column is so slim that the finger would need more sweep than it
    // is allowed, the phalanges CHORD across the section instead of following
    // it; shortening them is how that is expressed without a joint solver.
    const fit = Math.min(1, (sweep * meanR) / (fingerSum * FINGER_SCALE[f] * wrap));
    wrapFinger(
      fingerTubes,
      armour,
      handed,
      a,
      b,
      rows[f],
      theta0,
      sweep,
      FINGER_WIDTH[f],
      FINGER_SCALE[f] * wrap * fit,
    );

    // Metacarpal ridge: the run of bone from the knuckle back across the hand.
    // ONE PER ROW, so the 2 mm inter-finger gap continues onto the back of the
    // hand instead of dying at the knuckle — four ridges on a slab is what
    // makes a covered hand still read as having fingers in it.
    //
    // It has to be placed on the SAME ARC as the phalanges, sweeping backwards
    // from θ0. Aligned with the local Z axis instead — which is what the first
    // pass did — it is a straight 33 mm bar butted onto the start of a curve,
    // and index-to-fingertip becomes a 75 mm ruler with a kink in it. That is
    // why the support hand read as a robot claw.
    const mSweep = 0.62;
    const mt = theta0 - mSweep * 0.5;
    const mc = Math.cos(mt);
    const ms = Math.sin(mt);
    let mnx = mc / a;
    let mnz = ms / b;
    const mnl = Math.hypot(mnx, mnz) || 1;
    mnx /= mnl;
    mnz /= mnl;
    const mOff = 0.0074;
    palmMat.push({
      w: 0.0148,
      h: FINGER_WIDTH[f] + 0.0008,
      d: mSweep * ((a + b) * 0.5 + mOff) * 1.30,
      bevel: 0.0026,
      corner: 0.0052,
      x: handed * (a * mc + mnx * mOff),
      y: rows[f],
      z: -(b * ms + mnz * mOff),
      ry: Math.atan2(-handed * a * ms, -b * mc),
    });
  }

  // ── Index finger on the trigger ────────────────────────────────────────────
  // Straightened at the proximal joint and hooked at the distal, the pad on the
  // face of the blade. A firing hand whose index is curled into the fist with
  // everything else is the second-most-obvious viewmodel error after scale.
  if (o.trigger) {
    const t = o.triggerAt ?? [handed * 0.004, rows[0], -b - 0.030];
    const baseX = handed * (a + 0.0075);
    const baseY = rows[0] + 0.001;
    const baseZ = -b * 0.62;
    const dx = t[0] - baseX;
    const dz = t[2] - baseZ;
    const yaw = -Math.atan2(dx, -dz);
    // Proximal: from the knuckle forward and inboard toward the trigger guard.
    fabric.push({
      w: 0.0176, h: 0.0186, d: 0.0440, bevel: 0.0018, corner: 0.0040,
      x: baseX + dx * 0.30, y: baseY, z: baseZ + dz * 0.30,
      ry: yaw, rx: -0.10,
    });
    armour.push({
      w: 0.0032, h: 0.0166, d: 0.0380, bevel: 0.0008, corner: 0.0016,
      x: baseX + dx * 0.30 + handed * 0.0092, y: baseY + 0.0004, z: baseZ + dz * 0.30,
      ry: yaw, rx: -0.10,
    });
    // Middle: continues forward, beginning the hook.
    fabric.push({
      w: 0.0162, h: 0.0178, d: 0.0280, bevel: 0.0017, corner: 0.0038,
      x: baseX + dx * 0.70, y: baseY - 0.0032, z: baseZ + dz * 0.70,
      ry: yaw, rx: 0.44,
    });
    armour.push({
      w: 0.0030, h: 0.0158, d: 0.0230, bevel: 0.0008, corner: 0.0016,
      x: baseX + dx * 0.70 + handed * 0.0086, y: baseY - 0.0030, z: baseZ + dz * 0.70,
      ry: yaw, rx: 0.44,
    });
    // Distal: pad rolled onto the trigger face, pointing back at the shooter.
    fabric.push({
      w: 0.0148, h: 0.0166, d: 0.0235, bevel: 0.0016, corner: 0.0036,
      x: t[0], y: t[1] - 0.0080, z: t[2] + 0.0050,
      ry: yaw, rx: 1.04,
    });
  }

  // ── Thumb ──────────────────────────────────────────────────────────────────
  // Three segments from the thenar over the FAR side of the column. This is
  // both what a real grip does and the only place a thumb reads from the
  // camera: on the near side it is swallowed by the back of the hand.
  //
  // Built as a little forward-kinematic chain in the XZ plane so the joints
  // actually line up, rather than three boxes placed by eye.
  const roll = o.thumbRoll ?? 0.55;
  // Elevation of the thumb out of the finger-wrap plane, radians. A firing
  // thumb climbs the backstrap (+0.42); a support thumb on a handguard points
  // down the bore, which IS the column axis, so it needs −1.15. Without an
  // out-of-plane term the thumb can only move where the fingers move, and on a
  // barrel that means it sticks out sideways like a hitch-hiker.
  const pitch = o.thumbAxial ?? 0.24;
  const THUMB: ReadonlyArray<readonly [number, number, number, number]> = [
    // length, half-thickness, width, extra heading turn at this joint
    [0.0400, 0.0125, 0.0250, 0.0],
    [0.0300, 0.0108, 0.0215, 0.42],
    [0.0230, 0.0096, 0.0190, 0.50],
  ];
  {
    // Start on the palm-side shoulder, one row above the index.
    let px = handed * (a + 0.0090);
    let pz = b * 0.42;
    let py = ANATOMY.fingerPitch * 1.20;
    // Heading measured in the XZ plane: 0 = straight forward (−Z).
    let head = roll;
    let el = pitch;
    for (let i = 0; i < 3; i++) {
      const [len, thick, w, turn] = THUMB[i];
      head += turn * Math.cos(el);
      // The elevation grows with the curl, but only gently. At 0.34 the firing
      // thumb climbed 47 mm over its 93 mm length and its tip emerged through
      // the bottom of the receiver as a tan post standing in front of the
      // magwell — a thumb tip belongs level with the top of the grip, beside
      // the safety, not above the ejection port.
      el += turn * 0.16 * Math.sign(el || 1);
      // Unit heading: forward (−Z) rotated by `head` toward the far side
      // (−handed·X), then elevated out of that plane by `el`.
      const ce = Math.cos(el);
      const dx = -handed * Math.sin(head) * ce;
      const dz = -Math.cos(head) * ce;
      const dy = Math.sin(el);
      // Euler that puts a box's local +Z on (dx, dy, dz) under THREE's default
      // XYZ order, where local +Z maps to (sin ry, −sin rx·cos ry, cos rx·cos ry).
      const ry = Math.asin(THREE.MathUtils.clamp(dx, -1, 1));
      const rx = Math.atan2(-dy, dz);
      const cx = px + dx * len * 0.5;
      const cy = py + dy * len * 0.5;
      const cz = pz + dz * len * 0.5;
      fabric.push({
        w: thick * 2,
        h: w,
        d: len * 1.18,
        bevel: 0.0032,
        corner: 0.0058,
        x: cx,
        y: cy,
        z: cz,
        rx,
        ry,
      });
      if (i === 1) {
        armour.push({
          w: 0.0028, h: w * 0.60, d: len * 0.84,
          bevel: 0.0008, corner: 0.0014,
          x: cx + handed * thick * 0.86, y: cy, z: cz,
          rx,
          ry,
        });
      }
      px += dx * len;
      py += dy * len;
      pz += dz * len;
    }
  }

  // ── Palm and back of the hand ──────────────────────────────────────────────
  // One slab spanning knuckles to wrist. Its FRONT face is deliberately pulled
  // back to b + 6 mm, well inside the phalanx crowns at b + 8 mm plus their own
  // thickness, so the fingers stand 10–12 mm proud of it in silhouette.
  const palmX = handed * (a + ANATOMY.palmThick * 0.5 - 0.0035);
  const palmD = b * 2 + 0.0120;
  const ps = o.palmScale ?? 1;
  const palmL = ANATOMY.palmLength * 0.86 * ps;
  const palmY = -(o.palmDrop ?? 0.0245);
  addMesh(
    g,
    beveledBox(ANATOMY.palmThick, palmL, palmD, 0.0050, 0.0105),
    m.gloveTan,
    { pos: [palmX, palmY, 0.0030] },
  );
  // Heel of the hand — the mass below the little finger that makes the outline
  // asymmetric. Without it the palm is a symmetric block and reads as a mitten.
  addMesh(
    g,
    beveledBox(ANATOMY.palmThick * 0.96, 0.0300 * ps, palmD - 0.0050, 0.0050, 0.0100),
    m.gloveTan,
    { pos: [palmX + handed * 0.0012, palmY - palmL * 0.46, 0.0050], rot: [0, 0, handed * 0.11] },
  );
  // Thenar — the muscular pad at the base of the thumb. Reads as the bulge that
  // separates a hand from a paddle.
  addMesh(
    g,
    beveledBox(ANATOMY.palmThick * 0.82, 0.0420, 0.0230, 0.0045, 0.0090),
    m.gloveTan,
    { pos: [palmX - handed * 0.0030, 0.0060, b * 0.52], rot: [0, handed * 0.22, 0] },
  );
  // Silicone palm print on the gripping face.
  addMesh(
    g,
    beveledBox(0.0050, 0.0640 * ps, palmD - 0.0150, 0.0012, 0.0070),
    m.glovePalm,
    { pos: [handed * (a + 0.0020), palmY, 0.0030] },
  );
  // Dorsal knuckle shell — the hard plate a tactical glove carries across the
  // back of the hand, and the piece that tells the eye "glove", not "skin".
  //
  // TWO plates with a 3 mm channel between them and a tilt on each, not one
  // rectangle. A single flat card 70 mm square catches the key light as one
  // unbroken value and reads as a wallet gaffer-taped to a fist; split and
  // canted, the same area reads as armour following the back of a hand.
  for (const k of [-1, 1]) {
    addMesh(
      g,
      beveledBox(0.0062, 0.0290 * ps, palmD * 0.66, 0.0013, 0.0058),
      m.gloveBlack,
      {
        pos: [handed * (a + ANATOMY.palmThick - 0.0040), palmY * 0.50 + k * 0.0165 * ps, 0.0105],
        rot: [0, 0, handed * k * 0.10],
      },
    );
  }
  // Two tendon ribs down the back of the hand — cheap, but they break the
  // dorsal plane so it does not read as a flat card.
  addMesh(
    g,
    mergedBoxes([
      { w: 0.0038, h: 0.0480 * ps, d: 0.0062, bevel: 0.0008, corner: 0.0015, x: 0, y: 0, z: 0.0090 },
      { w: 0.0038, h: 0.0480 * ps, d: 0.0062, bevel: 0.0008, corner: 0.0015, x: 0, y: 0, z: -0.0030 },
    ]),
    m.gloveTan,
    { pos: [handed * (a + ANATOMY.palmThick - 0.0066), palmY * 0.70, 0.0030] },
  );

  // ── Inner mass ─────────────────────────────────────────────────────────────
  // Closes the underside of the fist, BELOW the little finger, where nothing
  // else covers the grip. Sized so it never reaches the phalanx crowns.
  addMesh(
    g,
    beveledBox(a * 2 + 0.0170, 0.0135, b * 2 + 0.0120, 0.0030, 0.0060),
    m.gloveTan,
    { pos: [handed * 0.0025, -ANATOMY.fingerPitch * 1.66, -0.0020] },
  );

  if (palmMat.length > 0) addMesh(g, mergedBoxes(palmMat), m.gloveTan);
  // The glove mesh is the extruded palm/thumb parts plus the swept fingers.
  // Both are non-indexed with position/normal/uv, so they merge cleanly into
  // the single draw call the hand had before.
  if (fabric.length > 0 || fingerTubes.length > 0) {
    const parts: THREE.BufferGeometry[] = [];
    if (fabric.length > 0) parts.push(mergedBoxes(fabric));
    parts.push(...fingerTubes);
    const merged = parts.length === 1 ? parts[0] : mergeGeometries(parts, false);
    if (parts.length > 1) for (const p of parts) p.dispose();
    addMesh(g, merged, m.gloveTan);
  }
  if (armour.length > 0) addMesh(g, mergedBoxes(armour), m.gloveBlack);

  return g;
}

/** Clamp to 0..1. */
function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Rewrite a revolved geometry's UVs from three's normalised 0..1 into METRES.
 *
 * This is the fix for the single worst thing about the old sleeve.
 * `CylinderGeometry` and `TorusGeometry` normalise UVs across the whole
 * surface, so a texture repeat is "N tiles per PART" and the physical thread
 * size therefore depends on how long the part happens to be. The old arm had
 * a 10 cm barrel and a 30 cm shirt sharing one `repeat.set(11, 14)`, which put
 * a 0.4 mm weave on the first and a 1.2 mm weave on the second, and neither
 * matched the 1.9 mm weave on the glove they ran into at the cuff. Nothing you
 * do to the image can fix a scale error in the mapping.
 *
 * With UVs in metres, one repeat value on the shared texture means one
 * physical tile size across every part of every weapon, for ever.
 *
 * @param uSpan metres the u axis covers over the full 0..1 sweep.
 * @param vSpan metres the v axis covers over the full 0..1 sweep.
 */
function metreUV(geo: THREE.BufferGeometry, uSpan: number, vSpan: number): THREE.BufferGeometry {
  const uv = geo.getAttribute('uv');
  if (!uv) return geo;
  for (let i = 0; i < uv.count; i++) {
    uv.setXY(i, uv.getX(i) * uSpan, uv.getY(i) * vSpan);
  }
  uv.needsUpdate = true;
  return geo;
}

/** Metre-space UVs for a tube of mean radius `r` and length `len`. */
function metreTube(geo: THREE.BufferGeometry, r: number, len: number): THREE.BufferGeometry {
  return metreUV(geo, Math.PI * 2 * r, len);
}

/**
 * Metre-space UVs for a torus, ALREADY ROTATED onto the limb axis.
 *
 * `TorusGeometry` is authored in its own XY plane with the axis on local Z.
 * Baking the 90° rotation into the geometry rather than leaving it on the mesh
 * is what lets the caller treat a fold ring exactly like every other limb
 * section: local +Y runs along the limb and the oval squash goes on `scale.z`,
 * the same as the barrels. Leaving it on the mesh means the squash axis
 * depends on the rotation order, which is how the folds got flattened out of
 * existence the first time they were tried.
 */
function metreTorus(geo: THREE.BufferGeometry, major: number, minor: number): THREE.BufferGeometry {
  metreUV(geo, Math.PI * 2 * major, Math.PI * 2 * minor);
  geo.rotateX(Math.PI / 2);
  return geo;
}

/**
 * A sampled centre-line for the forearm, with a stable ring frame at every
 * station.
 *
 * WHY A SWEEP AND NOT A STACK OF CYLINDERS. Three tapered `CylinderGeometry`
 * barrels with a rotated group between each is the obvious way to build a bent
 * limb and it does not work: the barrels meet at a plane, so an 8° bend leaves
 * a wedge-shaped gap on one flank and an overlap on the other, and the eye
 * reads the resulting hard step as a JOINT. A forearm with joints in it is a
 * length of ducting. Sweeping one continuous shell along a curved path gives a
 * limb with no seams anywhere, and — much more useful — it lets the radius and
 * the fold displacement be functions of position, which is how the wrinkles
 * below become part of the silhouette rather than rings stuck onto it.
 */
interface LimbPath {
  /** Arc length from the wrist plane, metres, per station. */
  readonly s: readonly number[];
  readonly p: readonly THREE.Vector3[];
  /** Ring basis. `ex` is the limb's own +X, `ez` faces the camera. */
  readonly ex: readonly THREE.Vector3[];
  readonly ez: readonly THREE.Vector3[];
  readonly tangent: readonly THREE.Vector3[];
}

/**
 * Integrate a gently curving centre-line starting at (0, y0, 0) heading +Y.
 *
 * The bend accumulates with the square of the distance, so the limb leaves the
 * wrist straight — a forearm does not hinge at the wrist — and curves most
 * toward the elbow. `bendZ` is the one the camera sees: the arm basis is built
 * so local +X is perpendicular to the view axis, which puts a rotation about
 * local Z squarely in the image plane.
 */
function buildLimbPath(
  y0: number,
  length: number,
  stations: number,
  bendZ: number,
  bendX: number,
): LimbPath {
  const s: number[] = [];
  const p: THREE.Vector3[] = [];
  const ex: THREE.Vector3[] = [];
  const ez: THREE.Vector3[] = [];
  const tangent: THREE.Vector3[] = [];
  const step = length / stations;
  const cur = new THREE.Vector3(0, y0, 0);
  for (let i = 0; i <= stations; i++) {
    const u = i / stations;
    const az = bendZ * u * u;
    const ax = bendX * u * u;
    const t = new THREE.Vector3(
      -Math.sin(az),
      Math.cos(az) * Math.cos(ax),
      Math.cos(az) * Math.sin(ax),
    ).normalize();
    s.push(y0 + u * length);
    p.push(cur.clone());
    tangent.push(t);
    // `cross(t, +Z)` keeps the ring's +X pinned to the arm's +X wherever the
    // tangent is anywhere near +Y, which it always is here. A Frenet frame
    // would spin about the tangent wherever the curvature vanishes.
    const x = new THREE.Vector3().crossVectors(t, new THREE.Vector3(0, 0, 1));
    if (x.lengthSq() < 1e-8) x.set(1, 0, 0);
    x.normalize();
    ex.push(x);
    ez.push(new THREE.Vector3().crossVectors(x, t).normalize());
    if (i < stations) cur.addScaledVector(t, step);
  }
  return { s, p, ex, ez, tangent };
}

/** One soft fold in the sleeve. Displaces the swept surface, not a stuck-on ring. */
interface LimbFold {
  /** Centre of the fold, metres from the wrist. */
  s: number;
  /** Peak radial displacement as a fraction of the local radius. */
  amp: number;
  /** Half-width of the fold along the limb, metres. */
  sigma: number;
  /** Which side of the limb it bunches on, radians (0 = +X, π/2 = camera). */
  phase: number;
  /** 0 = an even ring, 1 = entirely on `phase`'s side. */
  lobe: number;
  /** Radians the crest is skewed per metre — a fold is rarely square-on. */
  skew: number;
}

/**
 * Sweep a shell along `path` between two arc lengths.
 *
 * Used three times: once for the whole sleeve, and twice more with a limited
 * `thetaLen` for the hard forearm pad and its stitched edges, which then track
 * the taper and the bend exactly instead of floating off it.
 *
 * @param grow    radial multiplier — >1 lifts the pad clear of the sleeve.
 * @param folds   applied only when `grow` is 1, i.e. to the sleeve itself; a
 *                pad is a stiff laminate and does not wrinkle with the cloth.
 */
function sweepLimbShell(
  path: LimbPath,
  sA: number,
  sB: number,
  theta0: number,
  thetaLen: number,
  radial: number,
  radiusAt: (s: number) => number,
  flatAt: (s: number) => number,
  grow: number,
  roll: number,
  folds: readonly LimbFold[],
): THREE.BufferGeometry {
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const closed = thetaLen >= Math.PI * 2 - 1e-6;
  const cols = closed ? radial : radial + 1;
  const rows: number[] = [];
  for (let i = 0; i < path.s.length; i++) {
    if (path.s[i] >= sA - 1e-6 && path.s[i] <= sB + 1e-6) rows.push(i);
  }
  if (rows.length < 2) rows.length = 0;
  const nRows = rows.length;
  const pos = new Float32Array(nRows * cols * 3);
  const uv = new Float32Array(nRows * cols * 2);
  const col = new Float32Array(nRows * cols * 3);
  const idx: number[] = [];

  const sLo = path.s[rows[0]];
  const sHi = path.s[rows[nRows - 1]];
  for (let r = 0; r < nRows; r++) {
    const i = rows[r];
    const s = path.s[i];
    // A raised shell (`grow` > 1) eases its lift back to zero over the last
    // fifth at each end, so it SINKS INTO the sleeve rather than terminating in
    // a hard rim. A pad that stops at a step reads as a decal stuck on a tube;
    // one that fades in reads as a laminate sewn under the cloth.
    let lift = grow - 1;
    if (lift > 1e-6) {
      const a = clamp01((s - sLo) / Math.max(1e-6, (sHi - sLo) * 0.20));
      const b = clamp01((sHi - s) / Math.max(1e-6, (sHi - sLo) * 0.20));
      const w = Math.min(a, b);
      lift *= w * w * (3 - 2 * w);
    }
    const base = radiusAt(s) * (1 + lift);
    const flat = flatAt(s);
    const bodyShade = sleeveShadeAt(s);
    for (let c = 0; c < cols; c++) {
      const th = theta0 + (c / radial) * thetaLen;
      let disp = 0;
      for (const f of folds) {
        const d = (s - f.s) / f.sigma;
        if (d < -3.2 || d > 3.2) continue;
        const along = Math.exp(-d * d);
        const side = 1 - f.lobe + f.lobe * (0.5 + 0.5 * Math.cos(th - f.phase - f.skew * (s - f.s)));
        disp += f.amp * along * side;
      }
      // A shallow lengthwise crease on the far flank. Real cloth over a
      // forearm always has one, and it is what breaks the outline on the side
      // the fold rings do not reach.
      const creaseTh = Math.cos(th - 3.72);
      disp -= 0.024 * Math.exp(-Math.pow((creaseTh - 1) * 1.5, 2)) * clamp01((s - 0.05) / 0.09);
      // Contact occlusion in the folds. A crease is a place light does not
      // reach, and shading it is most of what makes a fold read as fabric
      // rather than as a lumpy tube; the crest gets a token lift so the pair
      // reads as one gather rather than as a dent.
      const shade =
        bodyShade *
        THREE.MathUtils.clamp(1 - 1.9 * Math.max(0, -disp) + 0.12 * Math.max(0, disp), 0.5, 1.12);
      const r0 = base * (1 + disp);
      // The section is an OVAL and `roll` decides how much of its broad face
      // the camera gets. Presenting the full width — which is what an unrolled
      // section does, because `ex` lies exactly in the image plane — projects
      // a 72 mm wrist and an 88 mm forearm at their full size, and that is the
      // "too thick" read. Turning the oval about 55° out of the image plane is
      // both what a pronated forearm actually does under a handguard and worth
      // 15–20% off the on-screen width, without lying about the anatomy.
      const ox = Math.cos(th);
      const oz = flat * Math.sin(th);
      const ux = ox * cr - oz * sr;
      const uz = ox * sr + oz * cr;
      const px = path.p[i].x + path.ex[i].x * (r0 * ux) + path.ez[i].x * (r0 * uz);
      const py = path.p[i].y + path.ex[i].y * (r0 * ux) + path.ez[i].y * (r0 * uz);
      const pz = path.p[i].z + path.ex[i].z * (r0 * ux) + path.ez[i].z * (r0 * uz);
      const k = r * cols + c;
      pos[k * 3] = px;
      pos[k * 3 + 1] = py;
      pos[k * 3 + 2] = pz;
      // UVs in METRES on both axes — see `metreUV`.
      uv[k * 2] = (c / radial) * thetaLen * base * (1 + flat) * 0.5;
      uv[k * 2 + 1] = s;
      col[k * 3] = shade;
      col[k * 3 + 1] = shade;
      col[k * 3 + 2] = shade;
    }
  }
  for (let r = 0; r + 1 < nRows; r++) {
    for (let c = 0; c < (closed ? cols : cols - 1); c++) {
      const c1 = (c + 1) % cols;
      const a = r * cols + c;
      const b = r * cols + c1;
      const d = (r + 1) * cols + c;
      const e = (r + 1) * cols + c1;
      idx.push(a, d, b, b, d, e);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

/**
 * Bake the sleeve's ambient occlusion into vertex colours.
 *
 * The far end of a forearm is buried in the shooter's own body shadow, and the
 * near end — at the cuff, out on the weapon — is in open light. Rendering the
 * whole limb at one brightness is the last thing keeping it reading as an
 * evenly-lit tube extending off the bottom of the frame; with the gradient the
 * eye stops following it long before the geometry actually ends, which is
 * exactly what happens when you look down at your own arm.
 *
 * Every limb geometry here has its local +Y along the limb and its origin at
 * its own midpoint, so `sMid + localY` is the true distance from the wrist for
 * barrels, fold rings and the shirt alike.
 *
 * @param sMid distance from the wrist, in metres, of the geometry's origin.
 */
function sleeveShadeAt(s: number): number {
  const t = clamp01((s - 0.030) / 0.235);
  const e = t * t * (3 - 2 * t);
  return 1 - 0.62 * e;
}

/**
 * Apply `sleeveShadeAt` to a geometry whose local +Y runs along the limb and
 * whose origin sits `sMid` metres from the wrist. Used for the parts that are
 * still ordinary three primitives — the rolled sleeve ring — rather than swept
 * shells, which colour themselves as they are generated.
 */
function shadeSleeve(geo: THREE.BufferGeometry, sMid: number): void {
  const pos = geo.getAttribute('position');
  const col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const v = sleeveShadeAt(sMid + pos.getY(i));
    col[i * 3] = v;
    col[i * 3 + 1] = v;
    col[i * 3 + 2] = v;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
}

/**
 * A gloved hand plus the forearm it hangs off.
 *
 * WHY THE LIMB KEPT READING AS A HOSE. Three things, and none of them was the
 * radius in the anatomy table.
 *
 *  1. SCALE OF THE CLOTH. The sleeve wore a clone of the glove's weave map
 *     retiled by eye onto normalised cylinder UVs — so the physical thread
 *     pitch varied per part and matched nothing — and the texture was a
 *     `DataTexture`, which defaults to NEAREST filtering with no mipmaps. A
 *     nearest-sampled weave at roughly one texel per two pixels does not read
 *     as cloth, it beats against the pixel grid and resolves as a regular
 *     coarse moiré: corduroy, or rope. See `metreUV` and `finishDetailTexture`.
 *  2. SILHOUETTE. One straight taper from cuff to elbow, with the fattest part
 *     of the limb landing on screen right where perspective magnifies it most.
 *     A straight, smooth, widening outline is a hose whatever is painted on it.
 *  3. LENGTH. Thirty centimetres of constant-diameter shirt trailing behind a
 *     run that had already left the frame, and a bend that swung the lower
 *     forearm sideways ACROSS the screen instead of dropping it away.
 *
 * So: the profile is a single eased curve that stays near wrist width for the
 * whole of the visible run; the outline is broken by two bends, five
 * irregular off-axis fold rings, a short hard pad and a rolled sleeve; the
 * cloth is authored in metres and properly filtered; and the whole limb is
 * darkened toward the elbow so it sinks into the shooter's own shadow instead
 * of marching evenly-lit off the bottom of the frame.
 *
 * Every call site uses a direction dominated by −Y so the forearm exits
 * through the BOTTOM of the frame rather than walking into the near plane,
 * where perspective magnifies it without limit.
 *
 * @param grip     where the hand closes, in weapon-local metres.
 * @param dir      direction the forearm runs, AWAY from the hand (toward the
 *                 off-screen elbow). Normalised internally.
 * @param reach    length of the tapered glove-to-rolled-sleeve run, metres. A
 *                 fixed 150 mm of shirt is added past it.
 * @param handRot  Euler pose of the hand about the grip point.
 */
function buildArm(
  m: GunMaterials,
  grip: readonly [number, number, number],
  dir: THREE.Vector3,
  reach: number,
  handRot: readonly [number, number, number],
  opts: HandOptions,
): THREE.Group {
  const g = new THREE.Group();
  g.position.set(grip[0], grip[1], grip[2]);

  const hand = buildHand(m, opts);
  hand.rotation.set(handRot[0], handRot[1], handRot[2]);
  g.add(hand);

  // Build an explicit basis rather than `setFromUnitVectors`, which leaves the
  // roll about `dir` arbitrary. Here the arm's local +X is forced perpendicular
  // to the view axis, so the elbow bend applied below always tilts in the plane
  // the camera can see instead of edge-on at random.
  const yAxis = dir.clone().normalize();
  const ref = new THREE.Vector3(0, 0, 1);
  let xAxis = new THREE.Vector3().crossVectors(yAxis, ref);
  if (xAxis.lengthSq() < 1e-6) xAxis = new THREE.Vector3(1, 0, 0);
  xAxis.normalize();
  const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
  const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
  const arm = new THREE.Group();
  arm.quaternion.setFromRotationMatrix(basis);
  // The forearm does not sprout from the grip AXIS, it sprouts from the wrist,
  // which sits under the heel of the palm and therefore outboard of the column
  // by roughly half a hand's thickness.
  const w0 = opts.wristAt ?? [opts.handed * (opts.gripHalfX + 0.0130), -0.0810, 0.0060];
  const wrist = new THREE.Vector3(w0[0], w0[1], w0[2]);
  wrist.applyEuler(new THREE.Euler(handRot[0], handRot[1], handRot[2]));
  arm.position.copy(wrist);
  g.add(arm);

  // ── Limb profile ───────────────────────────────────────────────────────────
  // `s` throughout this block is the distance from the WRIST PLANE along the
  // limb, in metres, and every radius is read from this one curve. Having a
  // single profile function rather than a per-segment pair of radii is what
  // guarantees the silhouette is continuous where the segments meet.
  //
  // WHY IT IS SHAPED LIKE THIS. A real forearm reaches its 108 mm maximum at
  // the elbow, roughly 26 cm from the wrist. In the frame the support hand sits
  // about 29 cm of arm above the bottom edge, so a literal anatomical taper
  // puts the fattest part of the limb ON SCREEN, at the point where perspective
  // is also magnifying it most — which is precisely the ballooning that read as
  // a hose. This curve is deliberately slower: 72 mm at the wrist, 94 mm at the
  // 30 cm mark where the sleeve leaves the frame, and it only approaches the
  // true elbow figure past that, off screen where nobody has to believe it.
  const rw = ANATOMY.wristHalf;
  const rwT = ANATOMY.wristThin;
  const re = ANATOMY.elbowHalf;

  const flatW = rwT / rw;
  /** Half-width across the limb at `s` metres from the wrist. */
  const radiusAt = (s: number): number => {
    const t = clamp01(s / 0.30);
    // Cubic ease-out: most of the growth happens in the first half of the run,
    // so the WRIST is legible as a waist rather than as the thin end of a cone.
    const e = 1 - Math.pow(1 - t, 2.15);
    return rw * (1.0 + 0.225 * e) + Math.max(0, s - 0.30) * 0.048;
  };
  /** Thickness/width ratio at `s`. The wrist is markedly flatter than the arm. */
  const flatAt = (s: number): number => flatW + (0.775 - flatW) * clamp01(s / 0.26);
  /** Where the sleeve starts: just inside the glove cuff, so there is no seam. */
  const y0 = 0.0345;

  // ── Glove cuff ─────────────────────────────────────────────────────────────
  // Overlaps the heel of the hand so there is no seam. A hard material change
  // here is what visually TERMINATES the hand; without it the glove and the
  // sleeve are one continuous object and the eye has nothing to call a wrist.
  //
  // Read as a BAND, not as a colour change: a 26 mm dark barrel, a proud
  // elastic ring at its top edge and the hook-and-loop strap and tab below it,
  // so the boundary survives at 30 px across.
  // Everything at the wrist lives in one group so it can take the SAME section
  // roll as the swept sleeve; a round cuff on a rolled oval steps at the seam.
  const wristParts = new THREE.Group();
  wristParts.rotation.y = -opts.handed * 0.96;
  arm.add(wristParts);

  const cuff = new THREE.Mesh(
    metreTube(new THREE.CylinderGeometry(rw * 1.09, rw * 1.20, 0.0460, 22, 1, true), rw * 1.15, 0.046),
    m.gloveBlack,
  );
  cuff.position.y = 0.0125;
  cuff.scale.z = flatW;
  cuff.frustumCulled = false;
  wristParts.add(cuff);
  // Piping at the BOTTOM edge, where the cuff meets the shirt. This is the
  // line the eye actually uses to separate glove from sleeve: without it the
  // dark band just fades into a tan limb and the wrist stops existing.
  const piping = new THREE.Mesh(
    metreTorus(new THREE.TorusGeometry(rw * 1.10, rw * 0.062, 5, 22), rw * 1.10, rw * 0.062),
    m.glovePalm,
  );
  piping.position.y = 0.0335;
  piping.scale.z = flatW;
  piping.frustumCulled = false;
  wristParts.add(piping);
  // Elastic gathering ring at the mouth of the cuff — the crisp top edge.
  const mouth = new THREE.Mesh(
    metreTorus(new THREE.TorusGeometry(rw * 1.055, rw * 0.085, 6, 20), rw * 1.055, rw * 0.085),
    m.gloveBlack,
  );
  mouth.position.y = -0.0055;
  mouth.scale.z = flatW;
  mouth.frustumCulled = false;
  wristParts.add(mouth);
  // Hook-and-loop wrist strap, proud of the cuff, with a tab on the outboard
  // face so the cuff is not a plain ring.
  const strap = new THREE.Mesh(
    metreTube(new THREE.CylinderGeometry(rw * 1.185, rw * 1.185, 0.0092, 20, 1, true), rw * 1.185, 0.0092),
    m.glovePalm,
  );
  strap.position.y = 0.0135;
  strap.scale.z = flatW;
  strap.frustumCulled = false;
  wristParts.add(strap);
  const tab = new THREE.Mesh(new THREE.BoxGeometry(0.0165, 0.0108, 0.0040), m.gloveBlack);
  tab.position.set(-opts.handed * rw * 1.00, 0.0135, rwT * 0.44);
  tab.rotation.y = -opts.handed * 0.5;
  tab.frustumCulled = false;
  wristParts.add(tab);

  // ── Forearm ────────────────────────────────────────────────────────────────
  // ONE swept shell from the cuff to well past the bottom of the frame. See
  // `buildLimbPath` for why this is not three stacked cylinders.
  //
  // `reach` is the tapered sleeve run; `TAIL` is the shirt that carries on
  // past the rolled cuff. The total is sized so the limb always leaves the
  // frame — an arm that STOPS on screen is worse than one that is too long —
  // without dragging the 30 cm of dead constant-diameter cylinder the previous
  // version trailed behind it.
  const TAIL = 0.135;
  const total = reach + TAIL;
  const path = buildLimbPath(y0, total, 64, opts.handed * 0.150, 0.150);
  /** Roll of the oval section about the limb axis. See `sweepLimbShell`. */
  const sectionRoll = opts.handed * 0.96;
  /**
   * The ellipse parameter that actually points at the camera once the section
   * is rolled. `θ = π/2` only faces the camera on an UNROLLED section; on a
   * rolled one it is off by most of the roll, which would slide the hard pad
   * round onto the far flank where nobody can see it.
   */
  const camTheta = Math.atan2(
    Math.cos(sectionRoll),
    flatAt(y0 + reach * 0.48) * Math.sin(sectionRoll),
  );

  /**
   * Where the cloth bunches.
   *
   * Positions are ABSOLUTE metres from the wrist, not fractions of `reach`, so
   * every weapon's sleeve gathers at the same physical intervals however long
   * its arm happens to be — a fold every three or four centimetres, which is
   * what a shirt sleeve over a braced forearm actually does. Four things stop
   * them reading as corrugation: the spacing is IRREGULAR; each one is LOBED,
   * so it gathers on one flank the way cloth does rather than ringing the limb
   * evenly; each crest is SKEWED so it crosses the limb at an angle; and the
   * lobes face different directions, so no two consecutive folds bulge on the
   * same side. They displace the swept surface itself, so they are part of the
   * silhouette rather than rings stuck onto it.
   */
  const ALL_FOLDS: readonly LimbFold[] = [
    // Gathered where the shirt is pushed up against the glove cuff: three
    // close crests on alternating flanks, then a trough.
    { s: 0.034, amp: 0.132, sigma: 0.012, phase: 0.30, lobe: 0.95, skew: 7.0 },
    { s: 0.049, amp: 0.150, sigma: 0.010, phase: 3.30, lobe: 0.96, skew: -8.5 },
    { s: 0.065, amp: 0.126, sigma: 0.011, phase: 1.05, lobe: 0.92, skew: 5.5 },
    { s: 0.081, amp: -0.088, sigma: 0.008, phase: 2.20, lobe: 1.00, skew: -6.0 },
    // Then a genuinely SMOOTH stretch — this is the part that stops the run
    // reading as corrugation. Cloth over the belly of a forearm is drum-tight;
    // only one big diagonal fold breaks it, with its own trough above.
    { s: 0.150, amp: 0.205, sigma: 0.021, phase: 2.55, lobe: 0.90, skew: 3.2 },
    { s: 0.177, amp: -0.082, sigma: 0.009, phase: 2.50, lobe: 1.00, skew: 3.2 },
    // A pair low down, where the sleeve starts to bunch toward the elbow.
    { s: 0.244, amp: 0.168, sigma: 0.016, phase: 0.55, lobe: 0.93, skew: -5.0 },
    { s: 0.269, amp: 0.104, sigma: 0.010, phase: -0.35, lobe: 0.97, skew: 6.5 },
    // Off the bottom of the frame in the resting pose; insurance for the
    // frames where landing recoil drops the whole viewmodel.
    { s: 0.332, amp: 0.150, sigma: 0.018, phase: 2.10, lobe: 0.88, skew: 2.6 },
  ];
  const folds = ALL_FOLDS.filter((f) => f.s > y0 + 0.010 && f.s < y0 + total - 0.010);

  const sleeveGeo = sweepLimbShell(
    path, y0, y0 + total, 0, Math.PI * 2, 28, radiusAt, flatAt, 1, sectionRoll, folds,
  );
  const sleeve = new THREE.Mesh(sleeveGeo, m.sleeve);
  sleeve.frustumCulled = false;
  arm.add(sleeve);

  // ── Hard forearm pad ───────────────────────────────────────────────────────
  // WHICH SIDE. The basis above forces local +X perpendicular to the view axis,
  // which puts local +Z — θ = π/2 on the ring — toward the camera at every call
  // site in this file. A pad centred anywhere else is detail rendered for the
  // inside of the player's monitor.
  //
  // It is deliberately SHORT and sits in the middle of the visible run. The
  // previous version ran two thirds of the length of the arm, which turned the
  // limb into a tube with a racing stripe rather than breaking it up.
  const padA = y0 + reach * 0.30;
  const padB = y0 + reach * 0.66;
  const wrapAng = 1.24; // 71° of the circumference, centred on the camera side
  const pad = new THREE.Mesh(
    sweepLimbShell(
      path, padA, padB, camTheta - wrapAng * 0.5, wrapAng, 14, radiusAt, flatAt, 1.055, sectionRoll, [],
    ),
    m.gloveBlack,
  );
  pad.frustumCulled = false;
  arm.add(pad);
  // Stitched seam down each edge of the pad, so it is a sewn-on piece rather
  // than a floating shell.
  // The seams must share the pad's EXACT s-range: `sweepLimbShell` eases a
  // raised shell's lift over a fraction of its own span, so a seam that spans
  // even four millimetres more than the pad eases on a different curve and
  // peels away from the edge it is supposed to be stitching down.
  for (const k of [-1, 1]) {
    const seam = new THREE.Mesh(
      sweepLimbShell(
        path, padA, padB,
        camTheta + k * wrapAng * 0.5 - 0.055, 0.11, 3, radiusAt, flatAt, 1.032, sectionRoll, [],
      ),
      m.glovePalm,
    );
    seam.frustumCulled = false;
    arm.add(seam);
  }

  // ── Rolled sleeve cuff ─────────────────────────────────────────────────────
  // Sits at the end of the tapered run — normally right at or just past the
  // bottom edge of the frame — and is what stops the limb reading as a tube
  // that simply carries on for ever.
  const sRoll = y0 + reach;
  const rRoll = radiusAt(sRoll);
  const rollGeo = metreTorus(
    new THREE.TorusGeometry(rRoll * 1.005, rRoll * 0.150, 8, 22),
    rRoll,
    rRoll * 0.150,
  );
  shadeSleeve(rollGeo, sRoll);
  const roll = new THREE.Mesh(rollGeo, m.sleeve);
  const iRoll = Math.min(path.s.length - 1, Math.round(((sRoll - y0) / total) * 64));
  roll.position.copy(path.p[iRoll]);
  roll.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), path.tangent[iRoll]);
  roll.rotateY(-sectionRoll);
  roll.scale.z = flatAt(sRoll);
  roll.frustumCulled = false;
  arm.add(roll);


  return g;
}

/** Curved rifle magazine, extruded from its true side silhouette. */
function curvedMagGeometry(width: number, length: number, bodyDepth: number): THREE.BufferGeometry {
  // Author the silhouette in (u, v) where u is world +Z (rearward) and v is
  // world +Y. A STANAG magazine curves FORWARD as it descends, so u decreases
  // with v — this quadratic is what gives the mag its banana profile instead of
  // the straight brick that betrays a procedural gun instantly.
  const halfD = bodyDepth * 0.5;
  const drop = -length;
  const bend = length * 0.24;

  const s = new THREE.Shape();
  s.moveTo(-halfD, 0);
  s.quadraticCurveTo(-halfD - bend * 0.55, drop * 0.55, -halfD - bend, drop * 0.96);
  s.lineTo(-halfD - bend + 0.006, drop);
  s.lineTo(halfD - bend * 0.72, drop);
  s.quadraticCurveTo(halfD - bend * 0.35, drop * 0.52, halfD, 0);
  s.closePath();

  const geo = new THREE.ExtrudeGeometry(s, {
    depth: width,
    bevelEnabled: true,
    bevelThickness: 0.0016,
    bevelSize: 0.0016,
    bevelSegments: 2,
    curveSegments: 6,
    steps: 1,
  });
  // Extrusion runs along +Z; rotate so it becomes the magazine's WIDTH (X) and
  // the authored u axis becomes world Z.
  geo.rotateY(-Math.PI / 2);
  geo.translate(width * 0.5, 0, 0);
  return geo;
}

// ─────────────────────────────────────────────────────────────────────────────
// Weapon builders
// ─────────────────────────────────────────────────────────────────────────────

interface BuildOut {
  root: THREE.Group;
  muzzle: THREE.Object3D;
  ejectPort: THREE.Object3D;
  optic: OpticParts;
  magazine: THREE.Object3D;
  chargingHandle: THREE.Object3D | null;
  chargingTravel: number;
  trigger: THREE.Object3D | null;
  bolt: THREE.Object3D | null;
  boltTravel: THREE.Vector3;
  leftArm: THREE.Object3D;
  rightArm: THREE.Object3D;
}

/** VECTOR-5 — 5.56 assault rifle. AR-pattern with a free-float hand-guard. */
function buildAR(m: GunMaterials): BuildOut {
  const root = new THREE.Group();
  const BORE = 0.052; // bore axis height above the grip origin

  // ── Upper receiver: a genuinely non-rectangular extruded profile. Flat
  //    bottom, rounded flanks, chamfered shoulders rising to a narrow rail
  //    plinth. This silhouette is what says "AR" before any other detail.
  const upper = new THREE.Shape();
  upper.moveTo(-0.0200, -0.0180);
  upper.quadraticCurveTo(-0.0200, -0.0262, -0.0122, -0.0262);
  upper.lineTo(0.0122, -0.0262);
  upper.quadraticCurveTo(0.0200, -0.0262, 0.0200, -0.0180);
  upper.lineTo(0.0200, 0.0118);
  upper.lineTo(0.0132, 0.0240);
  upper.lineTo(-0.0132, 0.0240);
  upper.lineTo(-0.0200, 0.0118);
  upper.closePath();
  const upperGeo = new THREE.ExtrudeGeometry(upper, {
    depth: 0.296,
    bevelEnabled: true,
    bevelThickness: 0.0018,
    bevelSize: 0.0018,
    bevelSegments: 2,
    curveSegments: 4,
    steps: 1,
  });
  upperGeo.center();
  addMesh(root, upperGeo, m.metal, { pos: [0, BORE, -0.114], name: 'upper' });

  // Top rail spans the receiver and continues over the hand-guard.
  addMesh(root, railGeometry(0.470), m.metalDark, { pos: [0, BORE + 0.027, -0.192] });

  // Forward assist and brass deflector on the right flank. Both sit REARWARD
  // of z = −0.015 so they stay clear of the bolt carrier's rearmost position
  // (z = −0.0255 at full stroke); a control the bolt drives through on every
  // shot is the kind of intersection that only shows up in motion.
  addMesh(root, latheForward([[0.0062, 0], [0.0062, 0.010], [0.0048, 0.013]], 10), m.metal, {
    pos: [0.0205, BORE + 0.006, -0.008],
    rot: [0, -Math.PI / 2, 0],
  });
  addMesh(root, beveledBox(0.011, 0.020, 0.024, 0.0022, 0.005), m.metal, {
    pos: [0.0215, BORE + 0.016, 0.002],
    rot: [0, 0, -0.35],
  });

  // Ejection port assembly, with a hinged dust cover hanging open below it.
  const eject = node(root, 'eject', 0.026, BORE + 0.005, -0.048);
  // ── Ejection port ───────────────────────────────────────────────────────
  // CAREFUL WITH THE FLANK POSITION. The upper receiver is an ExtrudeGeometry
  // whose profile is drawn to x = ±0.0200, but `bevelSize: 0.0018` expands the
  // silhouette outward, so the real outer surface is at x = 0.0218. Anything
  // that must be seen has to clear THAT, not the number in the profile.
  //
  // There is no CSG here, so the port cannot be a hole cut into a solid prism.
  // It is instead a port surround standing slightly proud of the flank — which
  // is how a real AR upper looks anyway, with its raised port and brass
  // deflector.
  //   flank  0.0218 | liner [0.0195, 0.0225] | bolt [0.0203, 0.0229]
  // so the liner clears the flank by 0.7 mm and the bolt clears the liner by
  // 0.4 mm and is genuinely visible.
  addMesh(root, beveledBox(0.0030, 0.0205, 0.049, 0.0009, 0.002), m.cavity, {
    pos: [0.0210, BORE + 0.005, -0.048],
  });
  // Bolt carrier, sitting in the port and reciprocating on every shot. A static
  // port reads as a decal; a moving bolt is the clearest signal that the weapon
  // is a mechanism rather than a prop.
  const bolt = new THREE.Group();
  bolt.name = 'bolt';
  bolt.position.set(0, BORE + 0.005, -0.052);
  root.add(bolt);
  addMesh(bolt, beveledBox(0.0026, 0.0150, 0.0210, 0.0006, 0.0015), m.metalWorn, {
    pos: [0.0216, 0, 0],
  });
  // Dust cover, hanging open below the port. Kept clear of the bolt's y-range
  // ([BORE-0.0025, BORE+0.0125]) so the two never interpenetrate.
  addMesh(root, beveledBox(0.0035, 0.0225, 0.052, 0.0008, 0.002), m.metalDark, {
    pos: [0.0224, BORE - 0.018, -0.048],
    rot: [0, 0, -0.42],
  });

  // ── Lower receiver, magwell and magazine.
  addMesh(root, beveledBox(0.0355, 0.050, 0.140, 0.0026, 0.006), m.polymer, {
    pos: [0, 0.008, -0.052],
  });
  addMesh(root, beveledBox(0.0340, 0.048, 0.052, 0.0024, 0.005), m.polymer, {
    pos: [0, -0.016, -0.086],
  });
  // Magazine release button.
  addMesh(root, latheForward([[0.0052, 0], [0.0052, 0.005], [0.0040, 0.0065]], 10), m.metalWorn, {
    pos: [0.019, -0.001, -0.070],
    rot: [0, -Math.PI / 2, 0],
  });

  const magazine = new THREE.Group();
  magazine.name = 'magazine';
  magazine.position.set(0, -0.036, -0.086);
  root.add(magazine);
  // `curvedMagGeometry` already centres the extrusion on X — do NOT offset it
  // again here, or the body slides off its own baseplate and magwell.
  addMesh(magazine, curvedMagGeometry(0.0268, 0.128, 0.0300), m.polymerFDE);
  // Baseplate and floorplate catch.
  addMesh(magazine, beveledBox(0.0300, 0.0080, 0.0330, 0.0014, 0.003), m.polymer, {
    pos: [0, -0.1275, -0.0300],
  });
  // Witness holes, straddling the right-hand wall so they read as recesses in
  // the surface rather than as buried or floating cubes.
  //   The wall is at x = ±0.0150, NOT ±0.0134: `curvedMagGeometry`'s 0.0268
  //   extrusion is re-expanded by 2 × 0.0016 of bevel. And at this height
  //   (y = −0.062) the curved body only spans z ∈ [−0.032, +0.006], so three
  //   holes at a 13 mm pitch fit where four at 23 mm did not.
  addMesh(magazine, slotStrip(3, 0.0130, 0.0044, 0.0044, 0.0044), m.cavity, {
    pos: [0.0150, -0.062, -0.0128],
  });

  // ── Grip: canted back, with moulded finger grooves.
  addGrip(root, m.polymerFDE, m.polymer, {
    w: 0.0300, h: 0.086, d: 0.0420, y: -0.052, z: 0.007, rx: 0.26, grooves: 5, pitch: 0.0140,
  });
  const trigger = buildTriggerGroup(root, m, 0, -0.006, -0.030);

  // ── Hand-guard: eight slats around the bore with accessory cut-outs and a
  //    continuous top rail. Gaps between slats let the barrel show through,
  //    which reads as far more expensive than a closed tube.
  addMesh(root, slattedShroud(0.0245, 0.196, 8, 0.0165, 0.0060), m.polymer, {
    pos: [0, BORE, -0.362],
  });
  // Front and rear collars tie the slats together.
  addMesh(root, latheForward([[0.0290, 0], [0.0300, 0.004], [0.0300, 0.014], [0.0288, 0.018]], 18), m.metalDark, {
    pos: [0, BORE, -0.268],
  });
  addMesh(root, latheForward([[0.0282, 0], [0.0292, 0.004], [0.0292, 0.012], [0.0278, 0.015]], 18), m.metalDark, {
    pos: [0, BORE, -0.452],
  });
  // Slats reach x = 0.0275, so the inlays sit at 0.0265 (outer face 0.0280) to
  // clear them by 0.5 mm. At 0.0248 they were 1.2 mm inside and drew nothing.
  addMesh(root, slotStrip(3, 0.0420, 0.0030, 0.0110, 0.0330), m.cavity, {
    pos: [-0.0265, BORE, -0.362],
  });
  addMesh(root, slotStrip(3, 0.0420, 0.0030, 0.0110, 0.0330), m.cavity, {
    pos: [0.0265, BORE, -0.362],
  });

  // ── Barrel: stepped profile with a gas block and a pinned flash hider.
  addMesh(
    root,
    latheForward(
      [
        [0.0140, 0.000],
        [0.0140, 0.020],
        [0.0104, 0.026],
        [0.0104, 0.150],
        [0.0092, 0.156],
        [0.0092, 0.286],
        [0.0106, 0.290],
        [0.0106, 0.300],
      ],
      18,
    ),
    m.metalDark,
    { pos: [0, BORE, -0.262] },
  );
  // Gas block + tube.
  addMesh(root, beveledBox(0.0210, 0.0235, 0.0270, 0.0016, 0.003), m.metalDark, {
    pos: [0, BORE + 0.002, -0.432],
  });
  // Rotated rather than mirrored. A negative scale is handled inconsistently
  // between face winding and shaded normals, and reasoning about which wins is
  // not worth it when a 180 degree turn about Y produces the same silhouette
  // for an axially symmetric lathe with the handedness intact.
  addMesh(root, latheForward([[0.0032, 0], [0.0032, 0.168]], 8), m.metalWorn, {
    pos: [0, BORE + 0.0135, -0.430],
    rot: [0, Math.PI, 0],
  });
  buildFlashHider(root, m, 0.0142, 0.044, BORE, -0.562);
  const muzzle = node(root, 'muzzle', 0, BORE, -0.608);

  // ── Charging handle: T-latch at the rear of the upper.
  const charging = new THREE.Group();
  charging.name = 'chargingHandle';
  charging.position.set(0, BORE + 0.017, 0.030);
  root.add(charging);
  addMesh(charging, beveledBox(0.0300, 0.0070, 0.0180, 0.0011, 0.002), m.metal);
  addMesh(charging, beveledBox(0.0125, 0.0100, 0.0300, 0.0014, 0.003), m.metalWorn, {
    pos: [-0.0180, -0.0010, 0.0035],
  });
  addMesh(charging, beveledBox(0.0125, 0.0100, 0.0300, 0.0014, 0.003), m.metalWorn, {
    pos: [0.0180, -0.0010, 0.0035],
  });
  // Static channel the handle rides in, spanning z ∈ [0.027, 0.086]. Without
  // it the handle is unsupported past the receiver's rear face at z = 0.0358
  // and hangs in open air over the last 18 mm of its stroke. Stops short of
  // the cheek riser at z = 0.090.
  addMesh(root, beveledBox(0.0210, 0.0080, 0.0590, 0.0012, 0.002), m.metalDark, {
    pos: [0, BORE + 0.017, 0.0565],
  });

  // ── Stock: buffer tube, collapsible body, cheek weld, rubber pad.
  addMesh(root, latheForward([[0.0170, 0], [0.0170, 0.150], [0.0180, 0.154]], 16), m.metalDark, {
    pos: [0, BORE - 0.010, 0.028],
    rot: [0, Math.PI, 0],
  });
  // Stock body sits BELOW the charging handle's path. The T-latch wings occupy
  // y ∈ [0.063, 0.073] and sweep back to z = 0.0835, so the stock top has to
  // stay under 0.063 — measure against the WINGS, not the narrower centre bar.
  addMesh(root, beveledBox(0.0350, 0.0520, 0.1150, 0.0032, 0.008), m.polymer, {
    pos: [0, BORE - 0.022, 0.118],
  });
  // Cheek riser starts at z = 0.090, behind the T-latch wings' rearmost point
  // (0.0835), so the handle passes in front of it rather than through it.
  addMesh(root, beveledBox(0.0300, 0.0180, 0.0900, 0.0026, 0.006), m.rubber, {
    pos: [0, BORE + 0.016, 0.135],
  });
  // Moulded grip ribs along the comb. This face is 27 cm from the eye — the
  // nearest surface on the weapon — so a bare plane there reads as a slab of
  // untextured plastic no matter what colour it is.
  addMesh(root, serrationStrip(7, 0.0110, 0.0250, 0.0022, 0.0028, 0, 'z'), m.polymer, {
    pos: [0, BORE + 0.0252, 0.136],
  });
  addMesh(root, beveledBox(0.0380, 0.0680, 0.0130, 0.0022, 0.004), m.rubber, {
    pos: [0, BORE - 0.020, 0.176],
    rot: [-0.09, 0, 0],
  });
  // Sling loop.
  addMesh(root, new THREE.TorusGeometry(0.0075, 0.0018, 5, 12), m.metalWorn, {
    pos: [0.0165, BORE - 0.020, 0.150],
    rot: [0, Math.PI / 2, 0],
  });

  // ── Optic, sat on the rail.
  const optic = buildRedDot(m, 1.0);
  optic.group.position.set(0, BORE + 0.0300, -0.060);
  root.add(optic.group);

  // ── Hands.
  // ── Hands. Directions are dominated by −Y so the forearms exit through the
  //    bottom of the frame rather than walking into the near plane; see the
  //    note on `buildArm`. `handRot.x` matches each grip's own rake so the
  //    fingers close on the column instead of shearing through it.
  const rightArm = buildArm(
    m,
    [0.001, -0.0400, 0.0045],
    new THREE.Vector3(0.47, -0.86, 0.20),
    0.250,
    [0.26, -0.13, 0.05],
    {
      handed: 1,
      gripHalfX: 0.0150,
      gripHalfZ: 0.0210,
      trigger: true,
      // Trigger blade face at weapon-local (0, −0.019, −0.036), expressed in
      // hand-local metres: the offset from the grip origin, de-rotated by the
      // 0.26 rad grip rake.
      triggerAt: [0.0015, 0.0165, -0.0505],
    },
  );
  const leftArm = buildArm(
    m,
    [0, BORE, -0.3060],
    new THREE.Vector3(-0.55, -0.83, 0.07),
    0.278,
    // +π/2 lays the hand's finger axis along the barrel; the 0.42 rad roll
    // about that axis puts the palm on the handguard's LEFT face so the
    // fingers come over the TOP — which is the only part of a support hand the
    // camera ever sees — and carry on down the far side.
    [Math.PI / 2, 0.42, 0],
    {
      handed: -1, gripHalfX: 0.0290, gripHalfZ: 0.0290, wrap: 1.10,
      thumbRoll: -0.16, thumbAxial: -1.15, palmScale: 0.98, palmDrop: 0.0055,
      wristAt: [-0.0350, -0.0100, 0.0420],
    },
  );
  root.add(rightArm, leftArm);

  return {
    root,
    muzzle,
    ejectPort: eject,
    optic,
    magazine,
    chargingHandle: charging,
    // 35 mm of charging-handle travel: the T-latch wings reach z = 0.0835,
    // clearing the cheek riser (z >= 0.090) and passing above the stock body
    // (top y = 0.056 vs wing bottom 0.063). 16 mm of bolt stroke keeps the
    // carrier inside the 49 mm port (18 mm of rear headroom from rest).
    chargingTravel: 0.035,
    trigger,
    bolt,
    boltTravel: new THREE.Vector3(0, 0, 0.016),
    leftArm,
    rightArm,
  };
}

/** WASP-9 — compact PDW-style SMG, magazine through the grip. */
function buildSMG(m: GunMaterials): BuildOut {
  const root = new THREE.Group();
  const BORE = 0.050;

  // Monolithic polymer receiver.
  addMesh(root, beveledBox(0.0420, 0.0620, 0.2050, 0.0032, 0.008), m.polymer, {
    pos: [0, BORE - 0.002, -0.070],
  });
  // The plate has to run the full length of the rail it carries. At 140 mm
  // against a 238 mm rail, 97 mm of rail hung over a 4.25-8.75 mm gap. At
  // 230 mm the plate is itself supported for its whole length: the receiver
  // (top y 0.079) aft, and the barrel collar (0.0745) and hand-guard slats
  // (0.07225) forward, all above its 0.065 underside.
  addMesh(root, beveledBox(0.0300, 0.0180, 0.2300, 0.0022, 0.004), m.metalDark, {
    pos: [0, BORE + 0.024, -0.086],
  });
  addMesh(root, railGeometry(0.226, 0.0195), m.metalDark, { pos: [0, BORE + 0.0355, -0.086] });

  // Ejection port and side charging handle.
  const eject = node(root, 'eject', 0.026, BORE + 0.010, -0.028);
  // Port surround and bolt, proud of the flank for the same reason as the
  // rifle. The receiver here is a `beveledBox`, whose declared width IS the
  // outer dimension, so the flank is exactly 0.0210.
  //   flank 0.0210 | liner [0.0187, 0.0217] | bolt [0.0195, 0.0221]
  addMesh(root, beveledBox(0.0030, 0.0180, 0.0400, 0.0009, 0.002), m.cavity, {
    pos: [0.0202, BORE + 0.010, -0.028],
  });
  const bolt = new THREE.Group();
  bolt.name = 'bolt';
  bolt.position.set(0, BORE + 0.010, -0.030);
  root.add(bolt);
  addMesh(bolt, beveledBox(0.0026, 0.0130, 0.0180, 0.0006, 0.0015), m.metalWorn, {
    pos: [0.0208, 0, 0],
  });
  const charging = new THREE.Group();
  charging.name = 'chargingHandle';
  charging.position.set(-0.0250, BORE + 0.016, -0.012);
  root.add(charging);
  addMesh(charging, beveledBox(0.0180, 0.0090, 0.0130, 0.0016, 0.003), m.metalWorn, {
    rot: [0, 0, 0.12],
  });
  // Slot the handle rides in, so it is supported over its whole 35 mm stroke.
  addMesh(root, beveledBox(0.0060, 0.0060, 0.0520, 0.0010, 0.002), m.cavity, {
    pos: [-0.0212, BORE + 0.016, 0.004],
  });

  // Grip / magwell — the magazine feeds through the pistol grip, which is what
  // makes a PDW silhouette so compact.
  // Grip top must reach the receiver underside at y = 0.017. The rotated AABB
  // top is y + (h/2)·cos(rx) + (d/2)·sin(rx) = −0.034 + 0.0514 + 0.0036 =
  // 0.0210 — a 4 mm bite. At the old y = −0.046 / h = 0.096 it topped out at
  // 0.005 and the grip, trigger and magazine hung in mid-air below the gun.
  addGrip(root, m.polymer, m.polymerFDE, {
    w: 0.0330, h: 0.1040, d: 0.0480, y: -0.034, z: 0.004, rx: 0.15, grooves: 6, pitch: 0.0125,
  });

  const magazine = new THREE.Group();
  magazine.name = 'magazine';
  magazine.position.set(0, -0.078, 0.010);
  root.add(magazine);
  addMesh(magazine, beveledBox(0.0260, 0.0900, 0.0400, 0.0022, 0.004), m.polymerFDE, {
    pos: [0, -0.041, 0.006],
    rot: [0.15, 0, 0],
  });
  addMesh(magazine, beveledBox(0.0310, 0.0080, 0.0450, 0.0014, 0.003), m.polymer, {
    pos: [0, -0.087, 0.012],
    rot: [0.15, 0, 0],
  });

  const trigger = buildTriggerGroup(root, m, 0, -0.004, -0.028);

  // Vertical foregrip and vented barrel shroud.
  addGrip(root, m.polymer, m.polymerFDE, {
    w: 0.0250, h: 0.0560, d: 0.0300, y: 0.006, z: -0.166, rx: 0.10, grooves: 4, pitch: 0.0120,
  });
  addMesh(root, slattedShroud(0.0195, 0.104, 6, 0.0140, 0.0055), m.metalDark, {
    pos: [0, BORE, -0.228],
  });
  addMesh(root, latheForward([[0.0235, 0], [0.0245, 0.003], [0.0245, 0.011], [0.0232, 0.014]], 16), m.metalDark, {
    pos: [0, BORE, -0.180],
  });

  // Barrel and a compact three-port brake.
  addMesh(
    root,
    latheForward(
      [
        [0.0120, 0.0],
        [0.0120, 0.022],
        [0.0084, 0.027],
        [0.0084, 0.158],
        [0.0096, 0.162],
        [0.0096, 0.170],
      ],
      16,
    ),
    m.metalDark,
    // Rear face at −0.168, i.e. 4.5 mm INSIDE the receiver (front −0.1725).
    // It used to start at −0.176 and float in a 3.5 mm gap.
    { pos: [0, BORE, -0.168] },
  );
  buildFlashHider(root, m, 0.0120, 0.032, BORE, -0.336);
  const muzzle = node(root, 'muzzle', 0, BORE, -0.376);

  // Folding skeleton stock.
  addMesh(root, beveledBox(0.0070, 0.0070, 0.1250, 0.0012, 0.002), m.metalDark, {
    pos: [-0.0155, BORE + 0.006, 0.092],
  });
  addMesh(root, beveledBox(0.0070, 0.0070, 0.1250, 0.0012, 0.002), m.metalDark, {
    pos: [0.0155, BORE + 0.006, 0.092],
  });
  addMesh(root, beveledBox(0.0400, 0.0500, 0.0110, 0.0020, 0.004), m.rubber, {
    pos: [0, BORE - 0.002, 0.160],
    rot: [-0.06, 0, 0],
  });

  const optic = buildRedDot(m, 0.9);
  optic.group.position.set(0, BORE + 0.0395, -0.048);
  root.add(optic.group);

  const rightArm = buildArm(
    m,
    [0.001, -0.0260, 0.0030],
    new THREE.Vector3(0.47, -0.86, 0.20),
    0.250,
    [0.15, -0.13, 0.05],
    { handed: 1, gripHalfX: 0.0165, gripHalfZ: 0.0240, trigger: true, triggerAt: [0.0015, 0.0130, -0.0470] },
  );
  // Support hand on the VERTICAL FOREGRIP, so the column axis is +Y again and
  // the hand needs no re-orientation — only the foregrip's own 0.10 rake.
  const leftArm = buildArm(
    m,
    [-0.001, 0.0060, -0.1660],
    new THREE.Vector3(-0.55, -0.83, 0.07),
    0.278,
    [0.10, 0.16, -0.10],
    { handed: -1, gripHalfX: 0.0125, gripHalfZ: 0.0150, wrap: 0.92 },
  );
  root.add(rightArm, leftArm);

  return {
    root,
    muzzle,
    ejectPort: eject,
    optic,
    magazine,
    chargingHandle: charging,
    // 35 mm keeps the latch over the receiver; at 45 it emerged into open air.
    chargingTravel: 0.035,
    trigger,
    bolt,
    // 13 mm of rear headroom inside the 40 mm port; 11 mm of stroke.
    boltTravel: new THREE.Vector3(0, 0, 0.011),
    leftArm,
    rightArm,
  };
}

/** LONGBOW .338 — bolt-action precision rifle in a chassis stock. */
function buildSniper(m: GunMaterials): BuildOut {
  const root = new THREE.Group();
  const BORE = 0.058;

  // Machined aluminium receiver with an integral 20 MOA rail. The 20 MOA is
  // real: 20 minutes of angle = 0.333° = 0.0058 rad of forward cant, which buys
  // back elevation adjustment at extreme range.
  addMesh(root, beveledBox(0.0440, 0.0640, 0.2500, 0.0030, 0.007), m.metal, {
    pos: [0, BORE, -0.096],
  });
  addMesh(root, railGeometry(0.236, 0.0215), m.metalDark, {
    pos: [0, BORE + 0.035, -0.100],
    rot: [-0.0058, 0, 0],
  });

  // Chassis: fore-end, thumbhole grip module, adjustable buttstock.
  addMesh(root, beveledBox(0.0480, 0.0300, 0.2600, 0.0030, 0.007), m.polymerFDE, {
    pos: [0, BORE - 0.044, -0.230],
  });
  // Fore-end half-width is 0.0240, so x = ±0.0226 puts the inlay's outer face
  // at 0.0243 — 0.3 mm proud, enough to be seen and not enough to look bolted
  // on. (It used to sit at ±0.0250, standing 2.7 mm off the side.)
  addMesh(root, slotStrip(5, 0.0400, 0.0034, 0.0130, 0.0300), m.cavity, {
    pos: [-0.0226, BORE - 0.044, -0.240],
  });
  addMesh(root, slotStrip(5, 0.0400, 0.0034, 0.0130, 0.0300), m.cavity, {
    pos: [0.0226, BORE - 0.044, -0.240],
  });

  // Thumbhole: three members leaving a real hole rather than a painted-on one.
  addMesh(root, beveledBox(0.0320, 0.0260, 0.1100, 0.0028, 0.006), m.polymerFDE, {
    pos: [0, BORE - 0.038, 0.062],
  });
  // Thumbhole underside is y = 0.007; rotated AABB top = −0.040 + 0.0482 +
  // 0.0036 = 0.0118. The grip closes the hole instead of hanging 7 mm below it.
  addGrip(root, m.polymerFDE, m.polymer, {
    w: 0.0300, h: 0.0980, d: 0.0400, y: -0.040, z: 0.012, rx: 0.18, grooves: 5, pitch: 0.0130,
  });
  addMesh(root, beveledBox(0.0300, 0.0300, 0.0380, 0.0028, 0.007), m.polymerFDE, {
    pos: [0, -0.006, 0.108],
  });

  // Cheek riser and buttpad.
  // Comb starts at z = 0.077, clear of the bolt's rearmost travel (0.071), and
  // sits at y ∈ [0.031, 0.049] so it lands ON the thumbhole's top bar
  // (y ≤ 0.033) rather than floating 20 mm above it, while still staying clear
  // of the bolt body at y ≥ 0.054.
  addMesh(root, beveledBox(0.0340, 0.0180, 0.0980, 0.0026, 0.006), m.polymer, {
    pos: [0, BORE - 0.018, 0.126],
  });
  addMesh(root, beveledBox(0.0400, 0.0820, 0.0140, 0.0022, 0.004), m.rubber, {
    pos: [0, BORE - 0.032, 0.184],
    rot: [-0.07, 0, 0],
  });

  const trigger = buildTriggerGroup(root, m, 0, -0.004, -0.036);

  // Bolt: body inside the receiver plus the handle the shooter cycles.
  //   Receiver spans z ∈ [−0.221, +0.029] and the comb starts at z = 0.077, so
  //   the body has to live inside that and the 42 mm stroke has to stop short
  //   of the comb: at rest z ∈ [−0.026, +0.029], at full stroke [0.016, 0.071].
  const bolt = new THREE.Group();
  bolt.name = 'bolt';
  bolt.position.set(0, BORE + 0.008, 0.004);
  root.add(bolt);
  addMesh(bolt, latheForward([[0.0090, 0], [0.0090, 0.045], [0.0120, 0.050], [0.0120, 0.055]], 14), m.metalWorn, {
    pos: [0, 0, -0.030],
    rot: [0, Math.PI, 0],
  });
  // Stem root must reach the 12 mm bolt body: rotated by −0.5 its x half-extent
  // is 4.07 mm, so at x = 0.0130 it spans [0.0089, 0.0171] and bites into the
  // body. At the old 0.0195 it started 4.1 mm clear of it, and the knob a
  // further 1.9 mm clear of the stem — a handle assembly attached to nothing.
  addMesh(bolt, beveledBox(0.0060, 0.0060, 0.0300, 0.0010, 0.002), m.metalWorn, {
    pos: [0.0130, -0.0035, 0.020],
    rot: [0, 0, -0.5],
  });
  addMesh(bolt, new THREE.SphereGeometry(0.0105, 12, 8), m.metalWorn, {
    pos: [0.0245, -0.0095, 0.020],
  });

  const eject = node(root, 'eject', 0.028, BORE + 0.010, -0.028);
  addMesh(root, beveledBox(0.0045, 0.0210, 0.0640, 0.0009, 0.002), m.cavity, {
    pos: [0.0208, BORE + 0.008, -0.030],
  });
  // Ports narrowed to match the brake's reduced 0.0195 radius.

  // Detachable box magazine.
  const magazine = new THREE.Group();
  magazine.name = 'magazine';
  magazine.position.set(0, BORE - 0.048, -0.108);
  root.add(magazine);
  addMesh(magazine, beveledBox(0.0300, 0.0620, 0.0620, 0.0026, 0.005), m.metalDark, {
    pos: [0, -0.026, 0],
  });
  addMesh(magazine, beveledBox(0.0360, 0.0080, 0.0700, 0.0016, 0.003), m.polymer, {
    pos: [0, -0.060, 0],
  });

  // Heavy fluted barrel: the flutes are the detail that reads "precision".
  addMesh(
    root,
    latheForward(
      [
        [0.0210, 0.000],
        [0.0210, 0.026],
        [0.0175, 0.032],
        [0.0175, 0.330],
        [0.0150, 0.340],
        [0.0150, 0.470],
        [0.0162, 0.476],
        [0.0162, 0.488],
      ],
      20,
    ),
    m.metalDark,
    { pos: [0, BORE, -0.220] },
  );
  // Flutes. The barrel is 0.0175 in radius here, so the groove centres sit at
  // 0.0148: outer face 0.0178, a bare 0.3 mm proud of the barrel skin, which
  // reads as a channel cut into it rather than as six rails bolted onto it.
  const flutes: BoxSpec[] = [];
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    flutes.push({
      w: 0.0060,
      h: 0.0060,
      d: 0.270,
      bevel: 0.0010,
      corner: 0.0018,
      x: Math.cos(a) * 0.0148,
      y: Math.sin(a) * 0.0148,
      rz: a,
    });
  }
  addMesh(root, mergedBoxes(flutes), m.cavity, { pos: [0, BORE, -0.400] });

  // Muzzle brake: a lathed can with side ports.
  addMesh(
    root,
    latheForward([[0.0130, 0], [0.0178, 0.006], [0.0178, 0.056], [0.0166, 0.062], [0.0120, 0.062]], 18),
    m.metalDark,
    { pos: [0, BORE, -0.708] },
  );
  const ports: BoxSpec[] = [];
  for (let i = 0; i < 3; i++) {
    ports.push({ w: 0.0060, h: 0.0160, d: 0.0090, bevel: 0.0008, corner: 0.001, x: -0.0163, z: -0.014 - i * 0.014 });
    ports.push({ w: 0.0060, h: 0.0160, d: 0.0090, bevel: 0.0008, corner: 0.001, x: 0.0163, z: -0.014 - i * 0.014 });
  }
  addMesh(root, mergedBoxes(ports), m.cavity, { pos: [0, BORE, -0.708] });
  const muzzle = node(root, 'muzzle', 0, BORE, -0.776);

  // Folded bipod under the fore-end.
  addMesh(root, beveledBox(0.0090, 0.0090, 0.1100, 0.0014, 0.002), m.metalDark, {
    pos: [-0.0130, BORE - 0.062, -0.300],
    rot: [0.16, 0, -0.12],
  });
  addMesh(root, beveledBox(0.0090, 0.0090, 0.1100, 0.0014, 0.002), m.metalDark, {
    pos: [0.0130, BORE - 0.062, -0.300],
    rot: [0.16, 0, 0.12],
  });

  const optic = buildScope(m);
  // 77 mm sight-over-bore, and the ring bases seated on the canted rail. The
  // height is not cosmetic: at BORE + 0.029 the muzzle brake's crown intruded
  // on the bottom of the scope's field of view, putting a metal crescent under
  // the reticle at every range.
  optic.group.position.set(0, BORE + 0.0368, -0.070);
  // Inherit the rail's 20 MOA cant, or the mounts sit on a wedge. Aim is
  // unaffected either way — the ADS pose is solved from `opticAim`, which moves
  // with the group — but a scope sitting square on a canted rail is visible.
  optic.group.rotation.x = -0.0058;
  root.add(optic.group);

  const rightArm = buildArm(
    m,
    [0.001, -0.0320, 0.0090],
    new THREE.Vector3(0.46, -0.87, 0.20),
    0.250,
    [0.18, -0.12, 0.05],
    { handed: 1, gripHalfX: 0.0150, gripHalfZ: 0.0200, trigger: true, triggerAt: [0.0015, 0.0180, -0.0530] },
  );
  // Support hand on the forend, well forward — a bolt gun is held long.
  const leftArm = buildArm(
    m,
    [0, BORE - 0.0180, -0.2480],
    new THREE.Vector3(-0.56, -0.82, 0.07),
    0.278,
    [Math.PI / 2, 0.46, 0],
    {
      handed: -1, gripHalfX: 0.0250, gripHalfZ: 0.0250, wrap: 1.06,
      thumbRoll: -0.16, thumbAxial: -1.15, palmScale: 0.98, palmDrop: 0.0055,
      wristAt: [-0.0310, -0.0100, 0.0400],
    },
  );
  root.add(rightArm, leftArm);

  return {
    root,
    muzzle,
    ejectPort: eject,
    optic,
    magazine,
    // 32 mm, not 42: the HANDLE STEM sits 10 mm rearward of the bolt body
    // (rest z ∈ [0.009, 0.039]), so it — not the body — is what has to clear
    // the comb at z = 0.077. At 32 mm the stem stops at 0.071, 6 mm short; the
    // body itself only reaches 0.061.
    chargingHandle: bolt,
    chargingTravel: 0.032,
    trigger,
    bolt,
    boltTravel: new THREE.Vector3(0, 0, 0.032),
    leftArm,
    rightArm,
  };
}

/** BREACH-12 — pump-action 12 gauge with a heat shield and breaching choke. */
function buildShotgun(m: GunMaterials): BuildOut {
  const root = new THREE.Group();
  const BORE = 0.056;

  addMesh(root, beveledBox(0.0480, 0.0620, 0.1950, 0.0032, 0.007), m.metal, {
    pos: [0, BORE - 0.006, -0.070],
  });
  addMesh(root, railGeometry(0.150, 0.0205), m.metalDark, { pos: [0, BORE + 0.028, -0.070] });

  // Loading gate underneath, ejection port on the right.
  const eject = node(root, 'eject', 0.028, BORE - 0.004, -0.036);
  addMesh(root, beveledBox(0.0050, 0.0260, 0.0640, 0.0010, 0.002), m.cavity, {
    pos: [0.0226, BORE - 0.004, -0.036],
  });
  addMesh(root, beveledBox(0.0300, 0.0050, 0.0660, 0.0010, 0.002), m.cavity, {
    pos: [0, BORE - 0.038, -0.040],
  });

  // Grip + stock.
  // Receiver underside is y = 0.019; rotated AABB top = −0.030 + 0.0505 +
  // 0.0053 = 0.0258, a 6.8 mm bite. It used to stop 17 mm short.
  addGrip(root, m.polymer, m.polymerFDE, {
    w: 0.0320, h: 0.1040, d: 0.0450, y: -0.030, z: 0.008, rx: 0.24, grooves: 5, pitch: 0.0140,
  });
  // Stock pulled forward to z = 0.090 so its front face meets the receiver's
  // rear at z = 0.0275; at 0.108 there was a 16 mm hole straight through the
  // weapon between the two.
  addMesh(root, beveledBox(0.0380, 0.0640, 0.1250, 0.0032, 0.008), m.polymer, {
    pos: [0, BORE - 0.024, 0.090],
    rot: [-0.05, 0, 0],
  });
  addMesh(root, beveledBox(0.0420, 0.0800, 0.0150, 0.0024, 0.004), m.rubber, {
    pos: [0, BORE - 0.030, 0.158],
    rot: [-0.10, 0, 0],
  });
  const trigger = buildTriggerGroup(root, m, 0, -0.008, -0.032);

  // Barrel + tube magazine.
  addMesh(
    root,
    latheForward([[0.0195, 0.0], [0.0195, 0.020], [0.0168, 0.026], [0.0168, 0.330], [0.0180, 0.336], [0.0180, 0.346]], 18),
    m.metalDark,
    { pos: [0, BORE, -0.164] },
  );
  addMesh(
    root,
    latheForward([[0.0128, 0.0], [0.0128, 0.300], [0.0140, 0.306], [0.0140, 0.320], [0.0100, 0.326]], 16),
    m.metalDark,
    { pos: [0, BORE - 0.030, -0.166] },
  );

  // Heat shield: slats plus two clamp rings.
  //
  // Both the span and the arc are constrained by the pump, which strokes 60 mm
  // through this region on every shot. The pump occupies z ∈ [−0.3575, −0.2425]
  // at rest and reaches −0.1805 at full travel, so the shield lives entirely
  // FORWARD of −0.370. The arc is capped at 0.76π (21.6° → 158.4°) so no slat
  // drops to the height of the pump body or the tube magazine below it.
  // `slattedShroud` steps by sweep/count, so 7 slats over 0.76π actually span
  // 21.6°→138.9°. Widened to 0.88π to sit symmetrically over the barrel.
  addMesh(root, slattedShroud(0.0225, 0.132, 7, 0.0130, 0.0050, Math.PI * 0.12, Math.PI * 0.88), m.metalDark, {
    pos: [0, BORE, -0.440],
  });
  addMesh(root, latheForward([[0.0230, 0], [0.0242, 0.003], [0.0242, 0.010], [0.0228, 0.013]], 16), m.metal, {
    pos: [0, BORE, -0.374],
  });
  addMesh(root, latheForward([[0.0230, 0], [0.0242, 0.003], [0.0242, 0.010], [0.0228, 0.013]], 16), m.metal, {
    pos: [0, BORE, -0.478],
  });

  // Pump: this is the "bolt" for animation purposes — it cycles on every shot.
  const pump = new THREE.Group();
  pump.name = 'pump';
  pump.position.set(0, BORE - 0.030, -0.300);
  root.add(pump);
  addMesh(pump, beveledBox(0.0440, 0.0400, 0.1150, 0.0034, 0.008), m.polymer);
  addMesh(pump, serrationStrip(9, 0.0118, 0.0450, 0.0040, 0.0070, 0), m.polymerFDE, { pos: [0, 0.0, 0] });
  // Action bar linking the pump to the receiver.
  addMesh(pump, beveledBox(0.0080, 0.0080, 0.1300, 0.0012, 0.002), m.metalDark, {
    pos: [0.0180, 0.0060, 0.090],
  });

  // Breaching standoff: three prongs on the muzzle.
  addMesh(root, latheForward([[0.0182, 0], [0.0215, 0.005], [0.0215, 0.026], [0.0180, 0.030]], 16), m.metalWorn, {
    pos: [0, BORE, -0.500],
  });
  const prongs: BoxSpec[] = [];
  for (let i = 0; i < 3; i++) {
    const a = Math.PI * 0.5 + (i / 3) * Math.PI * 2;
    prongs.push({
      w: 0.0075,
      h: 0.0075,
      d: 0.0240,
      bevel: 0.0012,
      corner: 0.0016,
      x: Math.cos(a) * 0.0165,
      y: Math.sin(a) * 0.0165,
      rz: a,
      rx: 0.10,
    });
  }
  addMesh(root, mergedBoxes(prongs), m.metalWorn, { pos: [0, BORE, -0.542] });
  const muzzle = node(root, 'muzzle', 0, BORE, -0.560);

  // Shell carrier stands in for the magazine during reloads.
  const magazine = new THREE.Group();
  magazine.name = 'magazine';
  // At the loading gate, not 40 cm down the tube magazine — the reload
  // animation carries this out and back, so it has to start somewhere a shell
  // could plausibly be and somewhere the arc does not drive it through the
  // barrel.
  magazine.position.set(0, BORE - 0.042, -0.046);
  root.add(magazine);
  addMesh(magazine, latheForward([[0.0092, 0], [0.0092, 0.052], [0.0088, 0.055]], 12), m.brass, { pos: [0, 0, 0] });
  addMesh(magazine, latheForward([[0.0098, 0], [0.0098, 0.014]], 12), m.brass, { pos: [0, 0, 0.001] });

  const optic = buildRedDot(m, 0.92);
  optic.group.position.set(0, BORE + 0.0325, -0.052);
  root.add(optic.group);

  const rightArm = buildArm(
    m,
    [0.001, -0.0280, 0.0055],
    new THREE.Vector3(0.47, -0.86, 0.20),
    0.250,
    [0.24, -0.13, 0.05],
    { handed: 1, gripHalfX: 0.0160, gripHalfZ: 0.0225, trigger: true, triggerAt: [0.0015, 0.0150, -0.0480] },
  );
  const leftArm = buildArm(
    m,
    [0, BORE - 0.0300, -0.3000],
    new THREE.Vector3(-0.55, -0.83, 0.07),
    0.278,
    [Math.PI / 2, 0.44, 0],
    {
      handed: -1, gripHalfX: 0.0225, gripHalfZ: 0.0205, wrap: 1.02,
      thumbRoll: -0.14, thumbAxial: -1.10, palmScale: 0.98, palmDrop: 0.0055,
      wristAt: [-0.0290, -0.0100, 0.0380],
    },
  );
  root.add(rightArm, leftArm);
  // The support hand rides the pump, so parent it there.
  pump.add(leftArm);
  leftArm.position.set(-0.004, -0.026, 0.000);

  return {
    root,
    muzzle,
    ejectPort: eject,
    optic,
    magazine,
    // 60 mm, not 75: the pump's rear face travels −0.2425 → −0.1825, stopping
    // 15 mm short of the receiver front at −0.1675. At 75 the two faces landed
    // exactly coplanar and z-fought.
    chargingHandle: pump,
    chargingTravel: 0.060,
    trigger,
    bolt: pump,
    // Same physical part as `chargingTravel`; keep the two in step.
    boltTravel: new THREE.Vector3(0, 0, 0.060),
    leftArm,
    rightArm,
  };
}

/** SIDEARM-9 — striker-fired 9mm with a slide-mounted mini reflex. */
function buildPistol(m: GunMaterials): BuildOut {
  const root = new THREE.Group();
  const BORE = 0.032;

  // Slide (reciprocates on every shot) with front and rear cocking serrations.
  const slide = new THREE.Group();
  slide.name = 'slide';
  root.add(slide);
  addMesh(slide, beveledBox(0.0280, 0.0300, 0.1740, 0.0026, 0.005), m.metal, {
    pos: [0, BORE, -0.052],
  });
  // Raked forward, the way cocking serrations are cut on a modern slide.
  addMesh(slide, serrationStrip(7, 0.0092, 0.0290, 0.0032, 0.0055, 0.30), m.metalDark, {
    pos: [0, BORE + 0.001, 0.006],
  });
  addMesh(slide, serrationStrip(4, 0.0092, 0.0290, 0.0032, 0.0055, 0.30), m.metalDark, {
    pos: [0, BORE + 0.001, -0.106],
  });
  // Ejection port + extractor.
  addMesh(slide, beveledBox(0.0045, 0.0150, 0.0330, 0.0008, 0.002), m.cavity, {
    pos: [0.0132, BORE + 0.005, -0.046],
  });
  addMesh(slide, beveledBox(0.0035, 0.0075, 0.0180, 0.0007, 0.001), m.metalWorn, {
    pos: [0.0145, BORE + 0.002, -0.030],
  });
  // Backup iron sights. They sit below the reflex's optical axis rather than
  // co-witnessing through its window — a lower-third mount, which is what most
  // slide-mounted optics actually give you.
  addMesh(slide, beveledBox(0.0180, 0.0075, 0.0060, 0.0009, 0.0014), m.metalDark, {
    pos: [0, BORE + 0.019, 0.026],
  });
  addMesh(slide, beveledBox(0.0045, 0.0080, 0.0050, 0.0008, 0.0012), m.metalDark, {
    pos: [0, BORE + 0.019, -0.128],
  });
  // Barrel hood + crown poking out of the slide.
  addMesh(slide, latheForward([[0.0092, 0], [0.0092, 0.012], [0.0080, 0.014]], 14), m.metalWorn, {
    pos: [0, BORE - 0.003, -0.138],
  });

  const eject = node(slide, 'eject', 0.020, BORE + 0.006, -0.044);

  // Polymer frame, grip module and accessory rail.
  addMesh(root, beveledBox(0.0262, 0.0210, 0.1560, 0.0024, 0.004), m.polymer, {
    pos: [0, BORE - 0.024, -0.050],
  });
  addMesh(root, railGeometry(0.048, 0.0150), m.polymer, {
    pos: [0, BORE - 0.037, -0.104],
    rot: [Math.PI, 0, 0],
  });
  addGrip(root, m.polymer, m.polymerFDE, {
    w: 0.0290, h: 0.0940, d: 0.0430, y: -0.040, z: 0.010, rx: 0.21, grooves: 6, pitch: 0.0118,
  });
  // Beavertail + slide-stop lever.
  addMesh(root, beveledBox(0.0270, 0.0150, 0.0280, 0.0026, 0.005), m.polymer, {
    pos: [0, 0.004, 0.026],
    rot: [-0.35, 0, 0],
  });
  addMesh(root, beveledBox(0.0040, 0.0070, 0.0330, 0.0008, 0.0012), m.metalWorn, {
    pos: [-0.0148, BORE - 0.020, -0.028],
  });

  const trigger = buildTriggerGroup(root, m, 0, BORE - 0.030, -0.026);

  const magazine = new THREE.Group();
  magazine.name = 'magazine';
  magazine.position.set(0, -0.088, 0.016);
  root.add(magazine);
  // 15 rounds of 9 mm, not 30: at 90 mm the body hung 86 mm below a grip whose
  // AABB bottom is −0.0904, reading as an extended stick magazine.
  addMesh(magazine, beveledBox(0.0230, 0.0560, 0.0380, 0.0020, 0.004), m.metalDark, {
    pos: [0, -0.024, 0.005],
    rot: [0.21, 0, 0],
  });
  addMesh(magazine, beveledBox(0.0290, 0.0080, 0.0430, 0.0014, 0.003), m.polymer, {
    pos: [0, -0.054, 0.011],
    rot: [0.21, 0, 0],
  });

  // Parented to the SLIDE, not the frame: the barrel crown these sit on is
  // part of the slide, so on the frame they tore an 18 mm gap open on every
  // shot as the slide cycled out from under them.
  buildFlashHider(slide, m, 0.0090, 0.014, BORE - 0.003, -0.146);
  const muzzle = node(slide, 'muzzle', 0, BORE - 0.003, -0.164);

  // Mini reflex, mounted directly to the slide so it rides the reciprocation.
  const optic = buildRedDot(m, 0.62);
  optic.group.position.set(0, BORE + 0.0150, 0.006);
  slide.add(optic.group);

  const rightArm = buildArm(
    m,
    [0.001, -0.0340, 0.0075],
    new THREE.Vector3(0.45, -0.87, 0.21),
    0.250,
    [0.21, -0.10, 0.04],
    { handed: 1, gripHalfX: 0.0145, gripHalfZ: 0.0215, trigger: true, triggerAt: [0.0015, 0.0175, -0.0430] },
  );
  // Support hand cups the firing hand — the modern two-handed pistol grip. Its
  // "column" is the firing fist itself, hence the larger radius and the loose
  // 0.72 wrap: these fingers lie over the firing hand's knuckles, they do not
  // close on anything.
  const leftArm = buildArm(
    m,
    [-0.0170, -0.0360, 0.0080],
    new THREE.Vector3(-0.57, -0.81, 0.08),
    0.268,
    [0.21, 0.30, -0.20],
    { handed: -1, gripHalfX: 0.0260, gripHalfZ: 0.0300, wrap: 0.70 },
  );
  root.add(rightArm, leftArm);

  return {
    root,
    muzzle,
    ejectPort: eject,
    optic,
    magazine,
    chargingHandle: slide,
    chargingTravel: 0.028,
    trigger,
    bolt: slide,
    boltTravel: new THREE.Vector3(0, 0, 0.026),
    leftArm,
    rightArm,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Public factory
// ─────────────────────────────────────────────────────────────────────────────

const BUILDERS: Record<WeaponId, (m: GunMaterials) => BuildOut> = {
  ar_vector: buildAR,
  smg_wasp: buildSMG,
  sniper_longbow: buildSniper,
  shotgun_breach: buildShotgun,
  pistol_sidearm: buildPistol,
};

export function buildWeaponRig(id: WeaponId, ctx: EngineContext): WeaponRig {
  const m = getMaterials(ctx);
  const out = BUILDERS[id](m);
  out.root.name = `weapon:${id}`;

  // Capture rest poses NOW, while nothing has animated. These are LOCAL
  // transforms, so no world-matrix composition is needed here — `setWeapon`
  // does its own when it solves the optic alignment.
  const rest: RigRest = {
    magazine: out.magazine.position.clone(),
    leftArm: out.leftArm.position.clone(),
    chargingHandle: out.chargingHandle ? out.chargingHandle.position.clone() : new THREE.Vector3(),
    bolt: out.bolt ? out.bolt.position.clone() : new THREE.Vector3(),
    triggerX: out.trigger ? out.trigger.rotation.x : 0,
  };

  return {
    rest,
    root: out.root,
    muzzle: out.muzzle,
    ejectPort: out.ejectPort,
    opticAim: out.optic.aim,
    magazine: out.magazine,
    chargingHandle: out.chargingHandle,
    chargingTravel: out.chargingTravel,
    trigger: out.trigger,
    bolt: out.bolt,
    boltTravel: out.boltTravel,
    dot: out.optic.dot,
    reticle: out.optic.reticle,
    leftArm: out.leftArm,
    rightArm: out.rightArm,
  };
}

/** Dispose every geometry owned by a rig. Materials are shared, so they stay. */
export function disposeRig(rig: WeaponRig): void {
  rig.root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.isMesh && mesh.geometry) mesh.geometry.dispose();
  });
  rig.root.removeFromParent();
}

/**
 * Clone the magazine as a free-standing prop for the reload drop animation.
 * Geometry is shared with the source rig; only the transforms are new.
 */
export function cloneMagazineProp(rig: WeaponRig): THREE.Object3D {
  const clone = rig.magazine.clone(true);
  clone.name = 'magProp';
  clone.traverse((o) => {
    o.frustumCulled = false;
    // `Object3D.copy` copies `visible`, and the caller hides the source
    // magazine as it detaches — so without this the prop is born invisible and
    // the entire drop animation plays on something that never renders.
    o.visible = true;
  });
  return clone;
}

// ─────────────────────────────────────────────────────────────────────────────
// Viewmodel lighting
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A 64×32 equirectangular HDR-ish gradient standing in for the North-African
 * golden-hour sky.
 *
 * This is NOT decorative. With `metalness = 1.0` a MeshStandardMaterial gets
 * essentially all of its colour from the environment: with no env map the
 * receiver renders as a black silhouette no matter how many lights are added.
 * A cheap procedural gradient plus a sun disc is enough to give the gunmetal
 * a warm sky-to-sand falloff and a hot specular streak along every chamfer.
 */
function goldenHourEnvironment(): THREE.DataTexture {
  const w = 64;
  const h = 32;
  const data = new Float32Array(w * h * 4);

  // v = 0 is −Y (ground) and v = 1 is +Y (zenith): with `flipY` false on a
  // DataTexture, row 0 of the array is v = 0.
  const sunU = 0.30;
  const sunV = 0.615; // ≈ 21° above the horizon
  for (let y = 0; y < h; y++) {
    const v = (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const u = (x + 0.5) / w;
      let r: number;
      let g: number;
      let b: number;
      if (v < 0.5) {
        // Ground bounce: hot dust, brightening toward the horizon.
        const t = v / 0.5;
        r = 0.16 + 0.34 * t;
        g = 0.11 + 0.24 * t;
        b = 0.07 + 0.14 * t;
      } else {
        // Sky: incandescent at the horizon, cooling toward the zenith.
        const t = (v - 0.5) / 0.5;
        r = 1.55 - 1.10 * t;
        g = 0.92 - 0.52 * t;
        b = 0.55 + 0.12 * t;
      }
      // Sun disc, with a soft bloom skirt. Wrapping in u keeps it continuous.
      let du = Math.abs(u - sunU);
      if (du > 0.5) du = 1 - du;
      const dv = v - sunV;
      const d = Math.sqrt(du * du * 4 + dv * dv * 4);
      const sun = Math.exp(-d * d * 160) * 26 + Math.exp(-d * d * 12) * 1.6;
      r += sun * 1.00;
      g += sun * 0.80;
      b += sun * 0.52;

      const i = (y * w + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 1;
    }
  }

  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Is the viewmodel scene already lit by somebody else?
 *
 * WHY THIS IS NOT JUST A CONSTRUCTOR CHECK. The engine's own lighting system
 * also installs a viewmodel rig, and it does so AFTER the ViewModel is
 * constructed. A one-shot test at construction time therefore always saw an
 * empty scene, both rigs got installed, and the viewmodel was lit by a 4.68
 * key PLUS a 1.8 key — roughly 3.5× the intended exposure. That is what made
 * the FDE furniture clip to cream, put a hard specular streak down the top
 * rail, and gave every mid-tone the blown-out "wet plastic" look. The fix is
 * to keep the test callable and re-run it once the frame loop has started, at
 * which point every system has had its turn; see `ViewModel.update`.
 *
 * @param owned lights this module installed, which do not count as foreign.
 */
export function hasForeignViewLight(scene: THREE.Object3D, owned: readonly THREE.Object3D[]): boolean {
  for (const child of scene.children) {
    if (!(child as THREE.Light).isLight) continue;
    // The muzzle-flash point light is parented to the weapon, not the scene, so
    // it never appears here; anything at scene level that we did not add is a
    // competing ambient/key rig.
    if (owned.indexOf(child) === -1) return true;
  }
  return false;
}

/**
 * Light the viewmodel scene.
 *
 * `ctx.viewScene` is a separate scene that world lighting does not reach, so
 * without this the weapon is unlit. Classic three-point rig tuned for golden
 * hour: warm key from the upper left, cool sky fill from the right (so the
 * shadow side is not dead), and a hot rim from the sun ahead that separates the
 * silhouette from the background.
 *
 * No-ops if another system already lit the scene, so this stays cooperative.
 */
export function installViewLighting(ctx: EngineContext): THREE.Object3D[] {
  const vs = ctx.viewScene;
  if (hasForeignViewLight(vs, [])) return [];

  const added: THREE.Object3D[] = [];

  // EXPOSURE. The previous key was 2.6, which drove every mid-tone on the
  // weapon past 1.0 before tone mapping: the FDE stock and the gloves both
  // clipped to near-white cream and the rail ridges threw hard specular
  // streaks. A viewmodel wants to sit DARKER than the world behind it — it is
  // in the shooter's own shadow, close to the body, and it is the one object
  // the player looks at for an entire match, so it must not glare. 1.8 keeps
  // the FDE furniture reading as tan rather than as paper.
  const key = new THREE.DirectionalLight(0xffd6a2, 1.80);
  key.position.set(-0.62, 0.90, 0.38);
  key.name = 'vm:key';
  const fill = new THREE.DirectionalLight(0x8fb6d8, 0.60);
  fill.position.set(0.85, 0.18, 0.55);
  fill.name = 'vm:fill';
  // The rim is what separates the weapon's silhouette from the background, so
  // it survives the cut better than the key does — but 1.9 was putting a hot
  // line along the whole top rail.
  const rim = new THREE.DirectionalLight(0xffb46a, 1.15);
  rim.position.set(0.30, 0.42, -1.00);
  rim.name = 'vm:rim';
  const amb = new THREE.HemisphereLight(0xbcd4ec, 0x6d5433, 0.42);
  amb.name = 'vm:amb';

  vs.add(key, fill, rim, amb);
  added.push(key, fill, rim, amb);

  if (!vs.environment) {
    vs.environment = goldenHourEnvironment();
    // 0.62, not 1.0: the environment is a bright desert, and at full strength
    // it alone lifts the metal off black before any light touches it.
    vs.environmentIntensity = 0.62;
  }
  return added;
}
