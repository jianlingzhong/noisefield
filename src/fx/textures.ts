/**
 * NOISEFIELD — procedural VFX sprite atlas.
 *
 * ZERO external assets: every sprite in the game's particle atlas is synthesised
 * on a 2D canvas at boot from multi-octave value noise.
 *
 * WHY AN ATLAS AND NOT N TEXTURES
 * -------------------------------
 * The whole particle budget (up to 20k quads) is drawn in two draw calls. That
 * is only possible if every particle samples the SAME texture, so all sprite
 * variants live in one 4x4 grid and each particle carries its atlas rectangle
 * as an instanced attribute. One texture bind, one draw.
 *
 * WHY NOISE-DRIVEN ALPHA
 * ----------------------
 * The single biggest "amateur particle" tell is a plain radial gaussian blob:
 * a perfectly circular, perfectly smooth silhouette. Real smoke has a lumpy,
 * fractal edge. Every soft sprite here therefore builds its alpha as
 *
 *     alpha = smoothstep(e0, e1, radialFalloff + (fbm - 0.5) * edgeAmplitude)
 *
 * with `edgeAmplitude` ramped up toward the rim, so the interior stays dense
 * while the silhouette dissolves into fractal tendrils. The RGB channel carries
 * a second, lower-frequency noise field used as an internal density/self-shadow
 * term, which the particle shader multiplies with its wrap-lighting result —
 * that is what stops smoke reading as a flat grey card.
 *
 * ORIENTATION
 * -----------
 * The texture is uploaded with `flipY = false`, so the painter's `v` coordinate
 * IS the sampled texture `v`. Sprites with a distinguished end (spark streaks,
 * splinters, blood teardrops) are authored with the "head" toward v = 1, which
 * is the +Y end of the particle quad, which is the direction the vertex shader
 * aligns to velocity.
 *
 */

import * as THREE from 'three';

// ─────────────────────────────────────────────────────────────────────────────
// Atlas layout
// ─────────────────────────────────────────────────────────────────────────────

export const ATLAS_COLS = 4;
export const ATLAS_ROWS = 4;

/**
 * Cell indices into the 4x4 atlas. Plain const object rather than a TS `enum`
 * because `isolatedModules` is on and const-enums are unavailable; this also
 * erases to plain numbers with no runtime enum object indirection.
 */
export const AtlasCell = {
  /** Fat fractal smoke puff — the workhorse for smoke columns and muzzle smoke. */
  Smoke: 0,
  /** Thinner, stringier smoke used to break up repetition in large columns. */
  SmokeWisp: 1,
  /** Fine, grainy dust — concrete/sand impacts and footfalls. */
  Dust: 2,
  /** Vertical spark streak with a hot round head at v=1. */
  Spark: 3,
  /** Irregular opaque chunk — concrete/rock debris. */
  Debris: 4,
  /** Long tapered sliver with grain — wood splinters. */
  Splinter: 5,
  /** Teardrop with a specular dot — blood, pointing head-up. */
  Blood: 6,
  /** Six-point flare star — muzzle flash / explosion core. */
  Flash: 7,
  /** Sharp triangular sliver with a bright refracting edge — glass. */
  Glass: 8,
  /** Hot radial gradient with a noisy rim — embers and fireballs. */
  Ember: 9,
  /** Clean soft circular gradient — generic glow, light bloom seeds. */
  Circle: 10,
  /** Thin noisy annulus — shockwave and spall rings. */
  Ring: 11,
  /** Small leaf blade with a midrib — foliage hits. */
  Leaf: 12,
  /** Water droplet / splash blob with a bright rim. */
  Water: 13,
  /** Four-point anisotropic glint — glass glitter, brass catchlights. */
  Glint: 14,
  /** Tiny irregular near-opaque grain — sand, grit, fine debris. */
  Grain: 15,
} as const;

export type AtlasCellId = (typeof AtlasCell)[keyof typeof AtlasCell];

/** Handle to a generated atlas plus the UV maths for addressing its cells. */
export interface SpriteAtlas {
  readonly texture: THREE.Texture;
  /** Square edge length of the atlas in pixels. */
  readonly size: number;
  readonly cols: number;
  readonly rows: number;
  /**
   * Writes (u0, v0, du, dv) for `cell` into `out`, inset by half a texel so
   * bilinear filtering can never bleed a neighbouring sprite in.
   */
  rect(cell: number, out: THREE.Vector4): THREE.Vector4;
  dispose(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar helpers (GLSL-equivalent semantics so shader and CPU art agree)
// ─────────────────────────────────────────────────────────────────────────────

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** GLSL smoothstep. Reversed edges (e0 > e1) intentionally yield a falling ramp. */
function smoothstep(e0: number, e1: number, x: number): number {
  const d = e1 - e0;
  if (d === 0) return x < e0 ? 0 : 1;
  const t = clamp01((x - e0) / d);
  return t * t * (3 - 2 * t);
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Integer hash → [0,1). Uses Math.imul throughout so every intermediate stays
 * in int32 and the result is bit-identical across engines (important: the atlas
 * must look the same on every machine, and this also feeds shader seeds).
 */
function ihash(x: number, y: number, seed: number): number {
  let h = Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(seed | 0, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}

/** Bilinear value noise with a smoothstep-interpolated lattice. */
function valueNoise(x: number, y: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = ihash(xi, yi, seed);
  const b = ihash(xi + 1, yi, seed);
  const c = ihash(xi, yi + 1, seed);
  const d = ihash(xi + 1, yi + 1, seed);
  return mix(mix(a, b, u), mix(c, d, u), v);
}

/** Fractal Brownian motion, gain 0.5 / lacunarity 2, normalised to [0,1]. */
function fbm(x: number, y: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let fx = x;
  let fy = y;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise(fx, fy, seed + i * 1013) * amp;
    norm += amp;
    amp *= 0.5;
    fx *= 2.0;
    fy *= 2.0;
  }
  return sum / norm;
}

/**
 * Ridged / billowy variant. `1 - |2n-1|` folds the noise so its peaks become
 * creases — this is what gives smoke its cauliflower lobes rather than the
 * soft pillowy look of raw fbm.
 */
function billow(x: number, y: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let fx = x;
  let fy = y;
  for (let i = 0; i < octaves; i++) {
    const n = valueNoise(fx, fy, seed + i * 733);
    sum += (1 - Math.abs(2 * n - 1)) * amp;
    norm += amp;
    amp *= 0.5;
    fx *= 2.0;
    fy *= 2.0;
  }
  return sum / norm;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell painters — each writes linear-ish RGBA in 0..1 into `out`
// ─────────────────────────────────────────────────────────────────────────────

type CellPainter = (u: number, v: number, out: Float32Array) => void;

const TAU = Math.PI * 2;

/** Fat fractal smoke puff. Dense core, dissolving fractal silhouette. */
const paintSmoke: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const r = Math.sqrt(cx * cx + cy * cy) * 2.0; // 0 centre → 1 at cell edge midpoint

  // Domain warp: displacing the sample point by another noise field is what
  // turns "blobby" noise into swirling, advected-looking smoke.
  const wx = fbm(u * 2.4, v * 2.4, 3, 11) - 0.5;
  const wy = fbm(u * 2.4 + 5.7, v * 2.4 - 3.1, 3, 23) - 0.5;
  const n = billow(u * 4.3 + wx * 1.8, v * 4.3 + wy * 1.8, 5, 7);

  // Angular lobes give the puff a cauliflower outline instead of a circle.
  const ang = Math.atan2(cy, cx);
  const lobes = 0.5 + 0.5 * Math.sin(ang * 5.0 + n * 6.0);

  const radial = 1.0 - r;
  const edgeAmp = 0.28 + 0.78 * Math.min(r, 1.0);
  let a = smoothstep(0.05, 0.46, radial + (n - 0.46) * edgeAmp + (lobes - 0.5) * 0.12 * r);
  a *= smoothstep(1.02, 0.68, r);

  // Internal density: low-frequency field the shader uses as a fake
  // self-shadow term so the puff has volume rather than reading as a decal.
  const dens = fbm(u * 6.5 + 13.0, v * 6.5 - 4.0, 4, 31);
  const lum = 0.62 + 0.38 * dens;
  out[0] = lum;
  out[1] = lum;
  out[2] = lum;
  out[3] = a;
};

/** Stringier, lower-density smoke used to break up repetition. */
const paintSmokeWisp: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const r = Math.sqrt(cx * cx + cy * cy) * 2.0;
  // Anisotropic domain: stretched vertically so the wisp reads as rising.
  const n = billow(u * 5.5, v * 2.6, 5, 401);
  const n2 = fbm(u * 11.0, v * 5.0, 3, 409);
  const radial = 1.0 - r;
  let a = smoothstep(0.06, 0.52, radial + (n - 0.52) * (0.4 + 0.9 * r));
  a *= 0.62 + 0.5 * n2;
  a *= smoothstep(1.02, 0.6, r);
  const lum = 0.66 + 0.34 * n2;
  out[0] = lum;
  out[1] = lum;
  out[2] = lum;
  out[3] = clamp01(a);
};

/** Fine grainy dust — high-frequency, low contrast, soft. */
const paintDust: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const r = Math.sqrt(cx * cx + cy * cy) * 2.0;
  const n = fbm(u * 6.5, v * 6.5, 4, 101);
  const grain = fbm(u * 27.0, v * 27.0, 2, 55);
  const radial = 1.0 - r;
  let a = smoothstep(0.02, 0.44, radial + (n - 0.5) * (0.34 + 0.6 * r));
  a *= 0.68 + 0.42 * grain;
  a *= smoothstep(1.02, 0.66, r) * 0.92;
  const lum = 0.78 + 0.22 * n;
  out[0] = lum;
  out[1] = lum;
  out[2] = lum;
  out[3] = clamp01(a);
};

/**
 * Spark streak. Authored vertically with the hot head at v≈0.87; the vertex
 * shader rotates the quad so +Y follows the velocity vector, which turns this
 * into a proper motion-elongated spark rather than a round dot.
 */
const paintSpark: CellPainter = (u, v, out) => {
  const x = u - 0.5;
  const headV = 0.86;
  const t = clamp01((v - 0.04) / (headV - 0.04));

  const tail = smoothstep(0.04, 0.5, v);
  const cap = 1.0 - smoothstep(headV + 0.02, 0.99, v);
  const width = 0.014 + 0.05 * Math.pow(t, 1.7);
  let a = Math.exp(-(x * x) / (2 * width * width)) * tail * cap;

  // Round hot head welded onto the end of the streak.
  const dhx = x;
  const dhy = v - headV;
  const dh2 = dhx * dhx + dhy * dhy;
  a = Math.max(a, Math.exp(-dh2 / (2 * 0.042 * 0.042)));

  // Slight flicker so a shower of sparks does not look like identical clones.
  a *= 0.74 + 0.3 * fbm(u * 9.0, v * 24.0, 2, 77);
  a = clamp01(a);

  // White-hot head cooling to orange down the tail. The shader can additionally
  // apply a blackbody ramp over the particle's lifetime.
  const heat = clamp01(t * 1.15);
  out[0] = 1.0;
  out[1] = mix(0.55, 0.97, heat);
  out[2] = mix(0.14, 0.82, heat * heat);
  out[3] = a;
};

/** Irregular opaque rock/concrete chunk with cheap baked directional shading. */
const paintDebris: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const rr = Math.sqrt(cx * cx + cy * cy);
  const ang = Math.atan2(cy, cx);
  const wobble = fbm(Math.cos(ang) * 2.0 + 4.0, Math.sin(ang) * 2.0 + 4.0, 3, 601) - 0.5;
  // Modest lobe amplitudes: too much angular wobble turns a rock chip into a
  // starfish, which reads as a decorative sprite rather than debris.
  const rad = 0.315 + 0.038 * Math.sin(ang * 3.0 + 1.1) + 0.022 * Math.sin(ang * 7.0 - 2.4) + 0.035 * wobble;
  const a = 1.0 - smoothstep(rad - 0.012, rad + 0.008, rr);

  // Fake key light from the upper-left so tumbling chips catch the eye. Kept
  // below 1.0 so a chip never reads brighter than the sunlit wall behind it.
  const facing = clamp01(0.5 + (0.5 - v) * 0.9 + (0.5 - u) * 0.4);
  const grit = fbm(u * 14.0, v * 14.0, 3, 611);
  const lum = (0.22 + 0.55 * facing) * (0.80 + 0.30 * grit);
  out[0] = lum;
  out[1] = lum * 0.985;
  out[2] = lum * 0.955;
  out[3] = a;
};

/** Long tapered wood sliver with longitudinal grain. */
const paintSplinter: CellPainter = (u, v, out) => {
  const x = u - 0.5;
  const taper = Math.pow(1.0 - Math.abs(v * 2.0 - 1.0), 0.65);
  const jitter = 0.6 + 0.55 * fbm(v * 9.0, 3.0, 2, 909);
  const halfW = 0.062 * taper * jitter;
  let a = 1.0 - smoothstep(halfW - 0.006, halfW + 0.004, Math.abs(x));
  a *= smoothstep(0.02, 0.1, v) * (1.0 - smoothstep(0.9, 0.995, v));

  const grain = fbm(v * 26.0, u * 4.0, 2, 917);
  const lum = 0.55 + 0.45 * grain;
  // Warm tan; the shader tints further per surface.
  out[0] = lum;
  out[1] = lum * 0.80;
  out[2] = lum * 0.55;
  out[3] = clamp01(a);
};

/** Stylised blood droplet: teardrop with a tail toward v=1 plus a specular dot. */
const paintBlood: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.44;
  // Asymmetric metric stretches the shape into a tail on the +v side.
  const sy = cy > 0 ? cy / 1.75 : cy;
  const d = Math.sqrt(cx * cx + sy * sy);
  const wob = (fbm(u * 8.0, v * 8.0, 3, 1201) - 0.5) * 0.03;
  let a = 1.0 - smoothstep(0.19 + wob, 0.245 + wob, d);
  a = clamp01(a);

  // Darker at the rim, slight specular near the top-left → wet look.
  const rim = clamp01(d / 0.24);
  const spec = Math.exp(-(((u - 0.41) * (u - 0.41) + (v - 0.36) * (v - 0.36)) / (2 * 0.035 * 0.035)));
  const base = mix(1.0, 0.45, rim * rim);
  out[0] = clamp01(0.62 * base + spec * 0.55);
  out[1] = clamp01(0.055 * base + spec * 0.35);
  out[2] = clamp01(0.045 * base + spec * 0.3);
  out[3] = a;
};

/** Six-point muzzle-flash star with a blown-out core. */
const paintFlash: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const rr = Math.sqrt(cx * cx + cy * cy) * 2.0;
  const ang = Math.atan2(cy, cx);
  const spikes = Math.pow(Math.abs(Math.cos(ang * 3.0)), 7.0);
  // Closed circular path (see paintRing) so the star has no branch-cut seam;
  // `rr` still shifts the sample radially for depth variation.
  const noise = fbm(Math.cos(ang) * 1.9 + 7.0 + rr * 1.4, Math.sin(ang) * 1.9 + 7.0, 3, 1701);

  let a = Math.pow(clamp01(1.0 - rr), 3.0) * 0.8;
  a += spikes * Math.pow(clamp01(1.0 - rr * 0.85), 1.5) * 0.85;
  a += Math.exp(-(rr * rr) / (2 * 0.11 * 0.11));
  a *= 0.82 + 0.36 * noise;
  a = clamp01(a) * smoothstep(1.05, 0.85, rr);

  const core = clamp01(1.0 - rr * 1.7);
  out[0] = 1.0;
  out[1] = mix(0.58, 0.95, core);
  out[2] = mix(0.16, 0.78, core * core);
  out[3] = a;
};

/**
 * Sharp glass sliver: hard silhouette, bright refracting edge.
 *
 * The interior test is winding-agnostic on purpose. Hard-coding
 * `d <= 0` for all three edges silently produces a completely EMPTY sprite
 * whenever the triangle happens to be wound the other way, which is exactly
 * the bug this comment exists to prevent recurring.
 */
const paintGlass: CellPainter = (u, v, out) => {
  const ax = 0.5;
  const ay = 0.94;
  const bx = 0.2;
  const by = 0.12;
  const cx2 = 0.76;
  const cy2 = 0.34;
  const side = (px: number, py: number, x0: number, y0: number, x1: number, y1: number): number =>
    (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0);
  const d0 = side(u, v, ax, ay, bx, by);
  const d1 = side(u, v, bx, by, cx2, cy2);
  const d2 = side(u, v, cx2, cy2, ax, ay);
  const allPos = d0 >= 0 && d1 >= 0 && d2 >= 0;
  const allNeg = d0 <= 0 && d1 <= 0 && d2 <= 0;
  if (!allPos && !allNeg) {
    out[0] = out[1] = out[2] = out[3] = 0;
    return;
  }

  // Distance to the nearest edge, in the same (unnormalised) units as `side`.
  const edge = Math.min(Math.abs(d0), Math.abs(d1), Math.abs(d2));
  const a = smoothstep(0.0, 0.004, edge) * 0.55 + 0.45;
  const rim = Math.exp(-edge / 0.010);

  // Cool glass body with a hot specular rim; the shader adds a sun glint.
  const body = 0.26 + 0.22 * fbm(u * 12.0, v * 12.0, 2, 2101);
  out[0] = clamp01(body * 0.86 + rim * 0.95);
  out[1] = clamp01(body * 0.95 + rim * 1.0);
  out[2] = clamp01(body * 1.0 + rim * 1.0);
  out[3] = clamp01(a);
};

/** Hot ember / fireball gradient with a turbulent rim. */
const paintEmber: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const rr = Math.sqrt(cx * cx + cy * cy) * 2.0;
  const n = billow(u * 5.0, v * 5.0, 4, 201);
  const radial = 1.0 - rr * 1.25;
  let a = smoothstep(0.0, 0.4, radial + (n - 0.5) * (0.35 + 0.7 * rr));
  a *= smoothstep(1.05, 0.7, rr);

  const t = clamp01(1.0 - rr * 1.55) * (0.7 + 0.5 * n);
  // Hand-tuned blackbody-ish ramp: deep red → orange → yellow → white.
  out[0] = clamp01(0.35 + 2.2 * t);
  out[1] = clamp01(-0.05 + 1.55 * t * t);
  out[2] = clamp01(-0.25 + 1.4 * t * t * t);
  out[3] = clamp01(a);
};

/** Clean soft circular gradient — glows, light seeds, generic soft mass. */
const paintCircle: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const rr = Math.sqrt(cx * cx + cy * cy) * 2.0;
  const a = Math.pow(clamp01(1.0 - rr), 2.3);
  out[0] = 1;
  out[1] = 1;
  out[2] = 1;
  out[3] = a;
};

/** Thin noisy annulus — shockwaves, spall rings, water ripples. */
const paintRing: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const rr = Math.sqrt(cx * cx + cy * cy) * 2.0;
  const ang = Math.atan2(cy, cx);
  const wob = fbm(Math.cos(ang) * 3.0 + 9.0, Math.sin(ang) * 3.0 + 9.0, 3, 303);
  const radius = 0.80 + (wob - 0.5) * 0.07;
  const width = 0.05 + 0.05 * wob;
  const k = (rr - radius) / width;
  let a = Math.exp(-k * k);
  // Angular detail must be sampled along a closed circular path. Feeding the
  // raw angle in makes the two sides of the atan2 branch cut land on
  // uncorrelated parts of the noise lattice, which cuts a visible notch
  // through the ring. Same trap the god-ray shader documents.
  a *= 0.5 + 0.6 * fbm(Math.cos(ang) * 2.2 + 6.0, Math.sin(ang) * 2.2 + 6.0, 3, 91);
  a *= smoothstep(1.08, 0.9, rr);
  const lum = 0.85 + 0.15 * wob;
  out[0] = lum;
  out[1] = lum;
  out[2] = lum;
  out[3] = clamp01(a);
};

/** Small leaf blade with a midrib — foliage hits. */
const paintLeaf: CellPainter = (u, v, out) => {
  const x = u - 0.5;
  const t = clamp01(v);
  const halfW = 0.24 * Math.sin(Math.PI * t) * (0.75 + 0.35 * fbm(v * 6.0, 1.0, 2, 1301));
  let a = 1.0 - smoothstep(halfW - 0.01, halfW + 0.006, Math.abs(x));
  a *= smoothstep(0.01, 0.06, v) * (1.0 - smoothstep(0.94, 1.0, v));
  const rib = Math.exp(-(x * x) / (2 * 0.012 * 0.012));
  const shade = 0.55 + 0.45 * fbm(u * 10.0, v * 10.0, 2, 1307) - rib * 0.22;
  out[0] = clamp01(shade * 0.62);
  out[1] = clamp01(shade * 0.90);
  out[2] = clamp01(shade * 0.42);
  out[3] = clamp01(a);
};

/** Water droplet: translucent body, bright rim, tiny specular. */
const paintWater: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.46;
  const sy = cy > 0 ? cy / 1.5 : cy;
  const d = Math.sqrt(cx * cx + sy * sy);
  const a = clamp01(1.0 - smoothstep(0.2, 0.25, d));
  const rim = Math.exp(-Math.pow((d - 0.205) / 0.028, 2.0));
  const spec = Math.exp(-(((u - 0.42) * (u - 0.42) + (v - 0.38) * (v - 0.38)) / (2 * 0.03 * 0.03)));
  const body = 0.42;
  out[0] = clamp01(body * 0.82 + rim * 0.7 + spec);
  out[1] = clamp01(body * 0.92 + rim * 0.85 + spec);
  out[2] = clamp01(body * 1.0 + rim * 0.95 + spec);
  out[3] = clamp01(a * 0.8 + rim * 0.35);
};

/** Anisotropic four-point glint — glass glitter and brass catchlights. */
const paintGlint: CellPainter = (u, v, out) => {
  const x = (u - 0.5) * 2.0;
  const y = (v - 0.5) * 2.0;
  const rr = Math.sqrt(x * x + y * y);
  const core = Math.exp(-(rr * rr) / (2 * 0.055 * 0.055));
  const hx = Math.exp(-(y * y) / (2 * 0.012 * 0.012)) * Math.pow(clamp01(1.0 - Math.abs(x)), 1.6);
  const hy = Math.exp(-(x * x) / (2 * 0.012 * 0.012)) * Math.pow(clamp01(1.0 - Math.abs(y)), 1.6);
  const a = clamp01(core + (hx + hy) * 0.7);
  out[0] = 1;
  out[1] = 1;
  out[2] = 1;
  out[3] = a;
};

/** Tiny irregular grain — sand, grit, glass dust. Near-opaque, hard edged. */
const paintGrain: CellPainter = (u, v, out) => {
  const cx = u - 0.5;
  const cy = v - 0.5;
  const rr = Math.sqrt(cx * cx + cy * cy);
  const ang = Math.atan2(cy, cx);
  const rad = 0.2 + 0.022 * Math.sin(ang * 4.0 + 2.0) + 0.016 * Math.sin(ang * 9.0);
  const a = 1.0 - smoothstep(rad - 0.012, rad + 0.01, rr);
  const lum = 0.68 + 0.32 * fbm(u * 20.0, v * 20.0, 2, 1901);
  out[0] = lum;
  out[1] = lum * 0.97;
  out[2] = lum * 0.9;
  out[3] = clamp01(a);
};

/** Painter table, indexed by AtlasCell. */
const PAINTERS: readonly CellPainter[] = [
  paintSmoke,
  paintSmokeWisp,
  paintDust,
  paintSpark,
  paintDebris,
  paintSplinter,
  paintBlood,
  paintFlash,
  paintGlass,
  paintEmber,
  paintCircle,
  paintRing,
  paintLeaf,
  paintWater,
  paintGlint,
  paintGrain,
];

// ─────────────────────────────────────────────────────────────────────────────
// Atlas construction
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Bakes the full sprite atlas.
 *
 * Cost is one-shot at boot and dominated by the multi-octave noise: measured
 * at ~70 ms for cellPx=128 (512² atlas) and ~270 ms for cellPx=256 (1024²) on
 * a desktop workstation. That is a real hitch, so only the top two quality
 * tiers pay for 256; everything else uses 128. Callers should prefer
 * {@link getSharedAtlas} so the cost is paid exactly once regardless of
 * subsystem init order.
 */
export function buildSpriteAtlas(cellPx = 256): SpriteAtlas {
  const cell = Math.max(32, Math.min(512, Math.floor(cellPx)));
  const size = cell * ATLAS_COLS;

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const c2d = canvas.getContext('2d', { willReadFrequently: false });
  if (!c2d) throw new Error('[fx/textures] 2D canvas context unavailable');

  const data = new Uint8ClampedArray(size * size * 4);
  const px = new Float32Array(4);
  const invCell = 1 / cell;

  for (let cellIndex = 0; cellIndex < ATLAS_COLS * ATLAS_ROWS; cellIndex++) {
    const painter = PAINTERS[cellIndex] ?? paintCircle;
    const col = cellIndex % ATLAS_COLS;
    const row = Math.floor(cellIndex / ATLAS_COLS);
    const ox = col * cell;
    const oy = row * cell;

    for (let y = 0; y < cell; y++) {
      // +0.5 samples the texel centre, which keeps the sprite symmetric.
      const v = (y + 0.5) * invCell;
      const rowBase = (oy + y) * size;
      for (let x = 0; x < cell; x++) {
        const u = (x + 0.5) * invCell;
        px[0] = 0;
        px[1] = 0;
        px[2] = 0;
        px[3] = 0;
        painter(u, v, px);
        const o = (rowBase + ox + x) * 4;
        data[o] = px[0] * 255;
        data[o + 1] = px[1] * 255;
        data[o + 2] = px[2] * 255;
        data[o + 3] = px[3] * 255;
      }
    }
  }

  c2d.putImageData(new ImageData(data, size, size), 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  // flipY=false so painter-v equals sampled-v; see the file header.
  texture.flipY = false;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  // Mipmaps matter here: thousands of tiny distant particles alias horribly
  // without them, and the shimmer reads as noise rather than dust.
  texture.generateMipmaps = true;
  texture.anisotropy = 4;
  texture.needsUpdate = true;
  texture.name = 'noisefield.fx.spriteAtlas';

  // Half-texel inset stops bilinear taps from reaching into the next cell.
  // Mip level 2+ can still bleed slightly. The soft sprites defend against
  // that with an explicit radial cutoff before the cell edge; the hard-edged
  // ones (debris, splinter, glass, grain, leaf) instead keep their silhouette
  // well inside the cell by construction — check any new painter against
  // this before adding it.
  const inset = 1.5 / size;

  return {
    texture,
    size,
    cols: ATLAS_COLS,
    rows: ATLAS_ROWS,
    rect(cellId: number, out: THREE.Vector4): THREE.Vector4 {
      const idx = Math.max(0, Math.min(ATLAS_COLS * ATLAS_ROWS - 1, cellId | 0));
      const col = idx % ATLAS_COLS;
      const row = Math.floor(idx / ATLAS_COLS);
      const du = 1 / ATLAS_COLS;
      const dv = 1 / ATLAS_ROWS;
      return out.set(col * du + inset, row * dv + inset, du - inset * 2, dv - inset * 2);
    },
    dispose(): void {
      texture.dispose();
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared instance
// ─────────────────────────────────────────────────────────────────────────────

let shared: SpriteAtlas | null = null;
let sharedRefs = 0;

/**
 * Returns the process-wide atlas, generating it on first call.
 *
 * WHY A MODULE SINGLETON: Particles, Tracers and Volumetrics all need the same
 * texture, but the engine may init them in any order and events can arrive
 * before every system's `init` has run. A memoised module-level getter removes
 * the ordering dependency entirely and guarantees one texture upload.
 */
export function getSharedAtlas(cellPx = 256): SpriteAtlas {
  if (!shared) shared = buildSpriteAtlas(cellPx);
  sharedRefs++;
  return shared;
}

/**
 * Drops one reference taken by {@link getSharedAtlas} and disposes the texture
 * once the last holder lets go. Refcounted rather than a plain dispose because
 * three separate systems share this one texture and none of them can know
 * whether it is the last one standing.
 */
export function releaseSharedAtlas(): void {
  sharedRefs = Math.max(0, sharedRefs - 1);
  if (sharedRefs === 0) {
    shared?.dispose();
    shared = null;
  }
}

/** Force-disposes the shared atlas regardless of refcount. Teardown only. */
export function disposeSharedAtlas(): void {
  shared?.dispose();
  shared = null;
  sharedRefs = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Small data textures used by the volumetric system
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Periodic value noise: the integer lattice wraps modulo `period`, so the
 * field is exactly continuous across a tile boundary.
 *
 * WHY THIS MATTERS: the god-ray shader samples the noise around a CIRCLE in
 * texture space to avoid the atan() seam. That circle crosses the tile border,
 * so a non-tiling texture puts the seam right back — as a hard radial cut
 * through the light shafts.
 */
function valueNoisePeriodic(x: number, y: number, period: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const wrap = (n: number): number => ((n % period) + period) % period;
  const x0 = wrap(xi);
  const y0 = wrap(yi);
  const x1 = wrap(xi + 1);
  const y1 = wrap(yi + 1);
  const a = ihash(x0, y0, seed);
  const b = ihash(x1, y0, seed);
  const c = ihash(x0, y1, seed);
  const d = ihash(x1, y1, seed);
  return mix(mix(a, b, u), mix(c, d, u), v);
}

/** Periodic fbm; each octave doubles both the frequency and the wrap period. */
function fbmPeriodic(x: number, y: number, period: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoisePeriodic(x * f, y * f, period * f, seed + i * 1013) * amp;
    norm += amp;
    amp *= 0.5;
    f *= 2;
  }
  return sum / norm;
}

/**
 * A seamlessly tiling RGBA noise texture used by the god-ray, ground-fog and
 * heat-shimmer shaders. R/G = two decorrelated single-octave bands, B/A = two
 * fbm bands. Small and cheap: 128² by default, ~65 kB.
 */
export function buildNoiseTexture(res = 128): THREE.DataTexture {
  const n = Math.max(16, Math.min(512, res | 0));
  const data = new Uint8Array(n * n * 4);
  const s = 8; // lattice cells across the texture
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const u = (x / n) * s;
      const v = (y / n) * s;
      const a = valueNoisePeriodic(u, v, s, 5);
      const b = valueNoisePeriodic(u * 2, v * 2, s * 2, 17);
      const c = fbmPeriodic(u, v, s, 4, 29);
      const d = fbmPeriodic(u * 3, v * 3, s * 3, 3, 47);
      const o = (y * n + x) * 4;
      data[o] = a * 255;
      data[o + 1] = b * 255;
      data[o + 2] = c * 255;
      data[o + 3] = d * 255;
    }
  }
  const tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace; // raw data, never colour-converted
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  tex.name = 'noisefield.fx.noise';
  return tex;
}

let sharedNoise: THREE.DataTexture | null = null;
let sharedNoiseRefs = 0;

/** Process-wide tiling noise texture; see {@link buildNoiseTexture}. */
export function getSharedNoise(): THREE.DataTexture {
  if (!sharedNoise) sharedNoise = buildNoiseTexture(128);
  sharedNoiseRefs++;
  return sharedNoise;
}

/** Refcounted counterpart to {@link getSharedNoise}. */
export function releaseSharedNoise(): void {
  sharedNoiseRefs = Math.max(0, sharedNoiseRefs - 1);
  if (sharedNoiseRefs === 0) {
    sharedNoise?.dispose();
    sharedNoise = null;
  }
}

/** Force-disposes the shared noise texture regardless of refcount. */
export function disposeSharedNoise(): void {
  sharedNoise?.dispose();
  sharedNoise = null;
  sharedNoiseRefs = 0;
}

/** Exported for shaders/tests that want the exact same lattice as the atlas. */
export const NoiseMath = { ihash, valueNoise, fbm, billow, smoothstep, clamp01, TAU } as const;
