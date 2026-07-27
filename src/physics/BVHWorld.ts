/**
 * NOISEFIELD — Static world collision.
 *
 * The level is baked into a small number of world-space, position-only
 * `MeshBVH` trees — one per `SurfaceType`. Bucketing by surface is the whole
 * design: a hit against a BVH *is* the surface classification, so impact VFX,
 * decals, footstep/impact audio and ballistic penetration all get their
 * material for free with zero lookup, zero `userData` chasing and zero
 * per-triangle side table.
 *
 * Everything in here is allocation-free on the query paths. Queries run through
 * `BVH.prototype.shapecast` with *persistent* callback objects and read triangle
 * vertices straight out of the underlying `Float32Array`. That is deliberate:
 *
 *  - `MeshBVH.raycastFirst()` allocates ~4 objects per hit (`point.clone()`, a
 *    `face` record, `face.normal`, `barycoord`). At ~100 casts/frame that is
 *    close to 1 MB/s of nursery garbage, which shows up as periodic minor-GC
 *    hitches in a 120 Hz sim.
 *  - `MeshBVH.prototype.shapecast()` spreads its argument (`{ ...callbacks }`)
 *    to inject the pooled `ExtendedTriangle`, allocating one object per *call*.
 *    We do not need that pooled triangle because we read vertices ourselves, so
 *    we call the base `BVH.prototype.shapecast` and skip the spread entirely.
 *
 * Verified against three-mesh-bvh 0.9.13 (`src/core/BVH.js`,
 * `src/core/cast/shapecast.js`, `src/core/build/geometryUtils.js`).
 *
 * NOT RE-ENTRANT: a query must not be started from inside another query's
 * callbacks. Nothing in the engine does that.
 */

import * as THREE from 'three';
import {
  BVH,
  MeshBVH,
  CENTER,
  INTERSECTED,
  NOT_INTERSECTED,
  type ShapecastCallbacks,
} from 'three-mesh-bvh';
import type { SurfaceType } from '../core/Types';

// ─────────────────────────────────────────────────────────────────────────────
// Tunables
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hits retained by a single `raycastAllRaw`.
 *
 * When more surfaces than this lie along the ray the FARTHEST is evicted, never
 * the nearest — an early `return true` on overflow would leave whichever hits
 * the BVH happened to visit first, which is not distance-ordered and produced a
 * 24 m disagreement between `raycast` and `raycastAll` on the same ray.
 */
export const MAX_PENETRATION_HITS = 64;

/**
 * Two surfaces closer together than this along a penetration ray are treated as
 * the same interface. Shared triangle edges and coplanar decals would otherwise
 * report a phantom "wall" of zero thickness and eat a bullet's penetration
 * budget.
 */
const PENETRATION_MERGE_EPS = 2e-4;

/** Degenerate-triangle rejection (area-ish, in the cross-product length). */
const DEGENERATE_EPS = 1e-12;

/**
 * A swept-sphere hit at or below this distance means the sphere was already
 * touching at t=0. Such a hit only counts if the motion drives INTO the
 * surface — see `sweptSphereTriangle`.
 */
const SWEEP_TOUCHING_EPS = 1e-6;

/** Contacts considered by one `depenetrateSphere` pass. */
const DEPENETRATION_CONTACTS = 32;

/** Floats per triangle in the merged soup: 3 verts * 3 components. */
const TRI_STRIDE = 9;

/**
 * Receives every triangle overlapping a query box. Pushing the narrow phase
 * into the traversal removes the need for a candidate-triangle budget, and a
 * budget here is not survivable: capping collection at N triangles drops them
 * in BVH-tree order rather than by distance, so a body over a finely
 * tessellated floor collects N far-away triangles, finds no contact, and falls
 * through the level. There is no safe value of N — only no N.
 */
export interface TriangleVisitor {
  visitTriangle(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
    surface: SurfaceType,
  ): void;
}

/** Floats per hit in the penetration buffer. */
const HIT_STRIDE = 8; // t, px,py,pz, nx,ny,nz  (+ front flag in `frontFlags`)

// ─────────────────────────────────────────────────────────────────────────────
// Scalar geometry kernels
//
// All of these are written on raw floats rather than Vector3 so the hot loops
// stay in registers and never touch the heap. Outputs go to module-scope
// scalars, which is safe because the whole subsystem is single-threaded and
// non-re-entrant.
// ─────────────────────────────────────────────────────────────────────────────

/** Unit geometric normal of the last successful `rayTriangle` / sweep test. */
let _nx = 0;
let _ny = 0;
let _nz = 0;
/** Contact point of the last successful swept-sphere test. */
let _cx = 0;
let _cy = 0;
let _cz = 0;
/** True when the last `rayTriangle` hit the front face (normal opposes the ray). */
let _front = false;

/**
 * Möller–Trumbore, explicitly double sided.
 *
 * Back faces matter: wall penetration needs the *exit* triangle to measure
 * material thickness, and the exit face is by definition back-facing to the
 * bullet. Any culling here silently breaks penetration.
 *
 * @returns hit distance along a unit `d`, or -1.
 */
function rayTriangle(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  tMax: number,
): number {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;

  // p = d × e2
  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;

  const det = e1x * px + e1y * py + e1z * pz;
  if (det > -1e-12 && det < 1e-12) return -1; // ray parallel to the triangle

  const invDet = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;

  const u = (tx * px + ty * py + tz * pz) * invDet;
  if (u < 0 || u > 1) return -1;

  // q = t × e1
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;

  const v = (dx * qx + dy * qy + dz * qz) * invDet;
  if (v < 0 || u + v > 1) return -1;

  const t = (e2x * qx + e2y * qy + e2z * qz) * invDet;
  if (t < 0 || t > tMax) return -1;

  // Geometric normal, raw (not flipped). Callers decide orientation.
  let gx = e1y * e2z - e1z * e2y;
  let gy = e1z * e2x - e1x * e2z;
  let gz = e1x * e2y - e1y * e2x;
  const gl = Math.sqrt(gx * gx + gy * gy + gz * gz);
  if (gl < DEGENERATE_EPS) return -1;
  const inv = 1 / gl;
  gx *= inv; gy *= inv; gz *= inv;

  _nx = gx; _ny = gy; _nz = gz;
  // det > 0 ⇔ the ray hits the side the normal points away from ⇒ front face.
  _front = det > 0;
  return t;
}

/** Ray/AABB slab entry distance, inflated by `pad`. Returns Infinity on miss. */
function rayBoxEntry(
  ox: number, oy: number, oz: number,
  idx: number, idy: number, idz: number,
  minx: number, miny: number, minz: number,
  maxx: number, maxy: number, maxz: number,
  pad: number, tMax: number,
): number {
  let tmin = 0;
  let tmax = tMax;

  let t1 = (minx - pad - ox) * idx;
  let t2 = (maxx + pad - ox) * idx;
  if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
  if (t1 > tmin) tmin = t1;
  if (t2 < tmax) tmax = t2;
  if (tmax < tmin) return Infinity;

  t1 = (miny - pad - oy) * idy;
  t2 = (maxy + pad - oy) * idy;
  if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
  if (t1 > tmin) tmin = t1;
  if (t2 < tmax) tmax = t2;
  if (tmax < tmin) return Infinity;

  t1 = (minz - pad - oz) * idz;
  t2 = (maxz + pad - oz) * idz;
  if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
  if (t1 > tmin) tmin = t1;
  if (t2 < tmax) tmax = t2;
  if (tmax < tmin) return Infinity;

  return tmin;
}

/** Is the coplanar point `p` inside triangle `abc` (normal `n`)? */
function pointInTriangle(
  px: number, py: number, pz: number,
  nx: number, ny: number, nz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): boolean {
  let ex = bx - ax, ey = by - ay, ez = bz - az;
  let vx = px - ax, vy = py - ay, vz = pz - az;
  if (nx * (ey * vz - ez * vy) + ny * (ez * vx - ex * vz) + nz * (ex * vy - ey * vx) < 0) return false;

  ex = cx - bx; ey = cy - by; ez = cz - bz;
  vx = px - bx; vy = py - by; vz = pz - bz;
  if (nx * (ey * vz - ez * vy) + ny * (ez * vx - ex * vz) + nz * (ex * vy - ey * vx) < 0) return false;

  ex = ax - cx; ey = ay - cy; ez = az - cz;
  vx = px - cx; vy = py - cy; vz = pz - cz;
  if (nx * (ey * vz - ez * vy) + ny * (ez * vx - ex * vz) + nz * (ex * vy - ey * vx) < 0) return false;

  return true;
}

/** First intersection of a unit ray with a sphere. Returns t ≥ 0 or -1. */
export function raySphere(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  sx: number, sy: number, sz: number, r: number,
): number {
  const mx = ox - sx, my = oy - sy, mz = oz - sz;
  const b = mx * dx + my * dy + mz * dz;
  const c = mx * mx + my * my + mz * mz - r * r;
  if (c > 0 && b > 0) return -1; // outside and receding
  const disc = b * b - c;
  if (disc < 0) return -1;
  const t = -b - Math.sqrt(disc);
  return t < 0 ? 0 : t;
}

/**
 * Unit ray against the finite cylinder around segment A→B with radius `r`.
 * Writes the closest point on the segment into `_cx/_cy/_cz`. Caps are not
 * handled here — the swept-sphere test covers them with vertex spheres.
 */
function rayCylinderSegment(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  axv: number, ayv: number, azv: number,
  bxv: number, byv: number, bzv: number,
  r: number,
): number {
  const abx = bxv - axv, aby = byv - ayv, abz = bzv - azv;
  const aox = ox - axv, aoy = oy - ayv, aoz = oz - azv;

  const abab = abx * abx + aby * aby + abz * abz;
  if (abab < DEGENERATE_EPS) return -1;

  const dab = dx * abx + dy * aby + dz * abz;
  const uab = aox * abx + aoy * aby + aoz * abz;
  const ud = aox * dx + aoy * dy + aoz * dz;
  const uu = aox * aox + aoy * aoy + aoz * aoz;

  const A = abab - dab * dab;
  if (A > -1e-12 && A < 1e-12) return -1; // ray parallel to the axis

  const B = 2 * (ud * abab - uab * dab);
  const C = uu * abab - uab * uab - r * r * abab;

  const disc = B * B - 4 * A * C;
  if (disc < 0) return -1;

  const sq = Math.sqrt(disc);
  const inv2A = 1 / (2 * A);
  let t = (-B - sq) * inv2A;
  const t2 = (-B + sq) * inv2A;
  if (t > t2) t = t2; // A can be negative in pathological cases; take the near root
  if (t < 0) {
    const far = Math.max((-B - sq) * inv2A, t2);
    if (far < 0) return -1;
    t = 0; // already inside the cylinder
  }

  const s = (uab + t * dab) / abab;
  if (s < 0 || s > 1) return -1; // beyond the segment; caps handle it

  _cx = axv + abx * s;
  _cy = ayv + aby * s;
  _cz = azv + abz * s;
  return t;
}

/**
 * Exact swept-sphere against a triangle (Fauerby's plane→edge→vertex cascade).
 *
 * Plane case first because it is what a character controller hits 99% of the
 * time (a flat floor or wall); the edge/vertex fallback only runs when the
 * projected contact falls outside the triangle.
 *
 * Writes the outward contact normal to `_nx/_ny/_nz` and the surface contact
 * point to `_cx/_cy/_cz`. Returns t ∈ [0, tMax] or -1.
 */
function sweptSphereTriangle(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  radius: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cxv: number, cyv: number, czv: number,
  tMax: number,
): number {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cxv - ax, e2y = cyv - ay, e2z = czv - az;

  let nx = e1y * e2z - e1z * e2y;
  let ny = e1z * e2x - e1x * e2z;
  let nz = e1x * e2y - e1y * e2x;
  const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (nl < DEGENERATE_EPS) return -1;
  const invNl = 1 / nl;
  nx *= invNl; ny *= invNl; nz *= invNl;

  const nd = nx * dx + ny * dy + nz * dz;
  const distA = nx * (ox - ax) + ny * (oy - ay) + nz * (oz - az);
  const sgn = distA >= 0 ? 1 : -1;

  // Already inside this triangle's plane slab at t=0?
  const overlapping = distA < radius && distA > -radius;

  let t0: number;
  if (overlapping) {
    // Only motion that drives FURTHER into the surface is blocking. Without
    // this a character capsule standing on the floor is blocked by the floor
    // for horizontal sweeps (move-and-slide advances 0 m) and for upward ones
    // (jumping clamps to the penetration depth), with no way to recover
    // because a sweep out of an overlap has nowhere to go.
    if (sgn * nd >= 0) return -1;
    t0 = 0;
  } else if (nd > -1e-8 && nd < 1e-8) {
    return -1; // parallel and not overlapping: can never touch
  } else {
    t0 = (sgn * radius - distA) / nd;
    if (t0 < 0) return -1;  // moving away from the plane
    if (t0 > tMax) return -1;
  }

  // Where the sphere touches the plane at t0.
  const ppx = ox + dx * t0 - nx * sgn * radius;
  const ppy = oy + dy * t0 - ny * sgn * radius;
  const ppz = oz + dz * t0 - nz * sgn * radius;

  if (pointInTriangle(ppx, ppy, ppz, nx, ny, nz, ax, ay, az, bx, by, bz, cxv, cyv, czv)) {
    _nx = nx * sgn; _ny = ny * sgn; _nz = nz * sgn;
    _cx = ppx; _cy = ppy; _cz = ppz;
    // A sphere that ALREADY overlaps this triangle only blocks motion that
    // drives further into it. Without this, a character capsule resting on the
    // floor gets t=0 for every horizontal and upward sweep, so move-and-slide
    // advances 0 m and jumping is impossible.
    if (t0 <= SWEEP_TOUCHING_EPS && _nx * dx + _ny * dy + _nz * dz >= 0) return -1;
    return t0;
  }

  // Fall back to the silhouette: three vertex spheres and three edge cylinders.
  let best = tMax;
  let found = false;
  let bpx = 0, bpy = 0, bpz = 0;

  let t = raySphere(ox, oy, oz, dx, dy, dz, ax, ay, az, radius);
  if (t >= 0 && t <= best) { best = t; found = true; bpx = ax; bpy = ay; bpz = az; }
  t = raySphere(ox, oy, oz, dx, dy, dz, bx, by, bz, radius);
  if (t >= 0 && t <= best) { best = t; found = true; bpx = bx; bpy = by; bpz = bz; }
  t = raySphere(ox, oy, oz, dx, dy, dz, cxv, cyv, czv, radius);
  if (t >= 0 && t <= best) { best = t; found = true; bpx = cxv; bpy = cyv; bpz = czv; }

  t = rayCylinderSegment(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz, radius);
  if (t >= 0 && t <= best) { best = t; found = true; bpx = _cx; bpy = _cy; bpz = _cz; }
  t = rayCylinderSegment(ox, oy, oz, dx, dy, dz, bx, by, bz, cxv, cyv, czv, radius);
  if (t >= 0 && t <= best) { best = t; found = true; bpx = _cx; bpy = _cy; bpz = _cz; }
  t = rayCylinderSegment(ox, oy, oz, dx, dy, dz, cxv, cyv, czv, ax, ay, az, radius);
  if (t >= 0 && t <= best) { best = t; found = true; bpx = _cx; bpy = _cy; bpz = _cz; }

  if (!found) return -1;

  let hnx = ox + dx * best - bpx;
  let hny = oy + dy * best - bpy;
  let hnz = oz + dz * best - bpz;
  const hl = Math.sqrt(hnx * hnx + hny * hny + hnz * hnz);
  if (hl < 1e-9) {
    // Degenerate: sphere centre sits exactly on the feature. Use the face normal.
    hnx = nx * sgn; hny = ny * sgn; hnz = nz * sgn;
  } else {
    const invHl = 1 / hl;
    hnx *= invHl; hny *= invHl; hnz *= invHl;
  }
  _nx = hnx; _ny = hny; _nz = hnz;
  _cx = bpx; _cy = bpy; _cz = bpz;
  if (best <= SWEEP_TOUCHING_EPS && _nx * dx + _ny * dy + _nz * dz >= 0) return -1;
  return best;
}

/** Closest point on triangle `abc` to `p`, written to `_cx/_cy/_cz`. */
export function closestPointOnTriangle(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): void {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;

  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { _cx = ax; _cy = ay; _cz = az; return; }

  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { _cx = bx; _cy = by; _cz = bz; return; }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    _cx = ax + abx * v; _cy = ay + aby * v; _cz = az + abz * v;
    return;
  }

  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { _cx = cx; _cy = cy; _cz = cz; return; }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    _cx = ax + acx * w; _cy = ay + acy * w; _cz = az + acz * w;
    return;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    _cx = bx + (cx - bx) * w; _cy = by + (cy - by) * w; _cz = bz + (cz - bz) * w;
    return;
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  _cx = ax + abx * v + acx * w;
  _cy = ay + aby * v + acy * w;
  _cz = az + abz * v + acz * w;
}

/**
 * Unit ray against a capsule (segment A→B, radius r), caps included.
 * Writes the surface normal to `_nx/_ny/_nz`. Returns t ≥ 0 or -1.
 *
 * Shared by the swept-sphere fallback and by entity capsule hitboxes, which is
 * the shape almost every humanoid collider ends up being.
 */
export function rayCapsule(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  r: number,
): number {
  let best = -1;
  let cpx = 0, cpy = 0, cpz = 0;

  const t = rayCylinderSegment(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz, r);
  if (t >= 0) { best = t; cpx = _cx; cpy = _cy; cpz = _cz; }

  let ts = raySphere(ox, oy, oz, dx, dy, dz, ax, ay, az, r);
  if (ts >= 0 && (best < 0 || ts < best)) { best = ts; cpx = ax; cpy = ay; cpz = az; }
  ts = raySphere(ox, oy, oz, dx, dy, dz, bx, by, bz, r);
  if (ts >= 0 && (best < 0 || ts < best)) { best = ts; cpx = bx; cpy = by; cpz = bz; }

  if (best < 0) return -1;

  let nx = ox + dx * best - cpx;
  let ny = oy + dy * best - cpy;
  let nz = oz + dz * best - cpz;
  const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (l < 1e-9) {
    _nx = -dx; _ny = -dy; _nz = -dz;
  } else {
    const inv = 1 / l;
    _nx = nx * inv; _ny = ny * inv; _nz = nz * inv;
  }
  return best;
}

/** Geometric normal produced by the last kernel call. */
export function lastNormalX(): number { return _nx; }
export function lastNormalY(): number { return _ny; }
export function lastNormalZ(): number { return _nz; }

/** Read-back accessors for the scalar kernels above. */
export function lastContactX(): number { return _cx; }
export function lastContactY(): number { return _cy; }
export function lastContactZ(): number { return _cz; }

// ─────────────────────────────────────────────────────────────────────────────
// Ray result record
// ─────────────────────────────────────────────────────────────────────────────

/** Flat, mutable ray result. Deliberately not a `RaycastHit` — no Vector3s. */
export interface RayResult {
  hit: boolean;
  t: number;
  px: number; py: number; pz: number;
  /** Raw geometric normal, NOT flipped toward the ray. */
  nx: number; ny: number; nz: number;
  surface: SurfaceType;
  /** True when the ray struck the front face of the triangle. */
  frontFace: boolean;
}

export function createRayResult(): RayResult {
  return {
    hit: false, t: 0,
    px: 0, py: 0, pz: 0,
    nx: 0, ny: 1, nz: 0,
    surface: 'concrete',
    frontFace: true,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Surface bucket
// ─────────────────────────────────────────────────────────────────────────────

interface Bucket {
  surface: SurfaceType;
  /** World-space, de-indexed triangle soup. Grows by doubling. */
  pos: Float32Array;
  /** Floats used in `pos` (= triangleCount * 9). */
  count: number;
  geometry: THREE.BufferGeometry | null;
  bvh: MeshBVH | null;
  /** MeshBVH reorders this in place; we read it directly during traversal. */
  index: Uint16Array | Uint32Array | null;
  dirty: boolean;
  /**
   * Cached world bounds of this bucket's BVH root. Slab-testing this before
   * calling `shapecast` skips whole buckets a query cannot touch, which matters
   * beyond the traversal itself: three-mesh-bvh's `BufferStack.setBuffer`
   * allocates three TypedArray views per shapecast call, so every skipped
   * bucket is three fewer allocations per query.
   */
  bMinX: number; bMinY: number; bMinZ: number;
  bMaxX: number; bMaxY: number; bMaxZ: number;
}

const VALID_SURFACES: ReadonlySet<string> = new Set<SurfaceType>([
  'concrete', 'metal', 'wood', 'sand', 'dirt', 'glass',
  'water', 'flesh', 'foliage', 'fabric', 'rubber',
]);

// Scratch used only during `addStatic` (level load, not a hot path).
const _m4 = new THREE.Matrix4();
const _v3 = new THREE.Vector3();
const _instanceMat = new THREE.Matrix4();
const _box = new THREE.Box3();

// ─────────────────────────────────────────────────────────────────────────────
// BVHWorld
// ─────────────────────────────────────────────────────────────────────────────

export class BVHWorld {
  private readonly buckets = new Map<SurfaceType, Bucket>();
  /** Iteration order is stable and avoids Map-iterator allocation per query. */
  private readonly bucketList: Bucket[] = [];
  /** Guards against the same root being baked twice. Reset by `clear()`. */
  private registered = new WeakSet<THREE.Object3D>();

  private builtTriangles = 0;
  private dirtyCount = 0;

  readonly bounds = new THREE.Box3();

  // ── query state (persistent, mutated per query) ───────────────────────────
  private qOx = 0; private qOy = 0; private qOz = 0;
  private qDx = 0; private qDy = 0; private qDz = 1;
  private qIdx = 0; private qIdy = 0; private qIdz = 0;
  private qMax = 0;
  private qRadius = 0;
  private qBest = Infinity;
  private qPos: Float32Array = new Float32Array(0);
  private qIndex: Uint16Array | Uint32Array = new Uint16Array(0);
  private qSurface: SurfaceType = 'concrete';
  private qResult: RayResult = createRayResult();

  // AABB query state (rigid-body triangle collection).
  private qBMinX = 0; private qBMinY = 0; private qBMinZ = 0;
  private qBMaxX = 0; private qBMaxY = 0; private qBMaxZ = 0;
  private qTriCount = 0;
  private qVisitor: TriangleVisitor | null = null;

  // ── output buffers (reused) ───────────────────────────────────────────────
  /** Penetration hits: stride 8 → t, px, py, pz, nx, ny, nz, (unused pad). */
  private readonly hitBuf = new Float32Array(MAX_PENETRATION_HITS * HIT_STRIDE);
  private readonly hitSurface: SurfaceType[] = new Array<SurfaceType>(MAX_PENETRATION_HITS).fill('concrete');
  private readonly hitFront = new Uint8Array(MAX_PENETRATION_HITS);
  private hitCount = 0;
  /** True when more surfaces than `MAX_PENETRATION_HITS` lay along the ray. */
  penetrationTruncated = false;

  // ── shapecast callback objects (allocated once) ───────────────────────────
  private readonly rayCbs: ShapecastCallbacks;
  private readonly rayAllCbs: ShapecastCallbacks;
  private readonly sweepCbs: ShapecastCallbacks;
  private readonly boxCbs: ShapecastCallbacks;

  /**
   * The base-class `shapecast`. `MeshBVH` overrides it purely to inject a
   * pooled `ExtendedTriangle`, which costs one object spread per call and which
   * we do not need — we decode triangles from the raw arrays ourselves.
   */
  private static readonly baseShapecast: (this: BVH, cbs: ShapecastCallbacks) => boolean =
    BVH.prototype.shapecast;

  constructor() {
    // Persistent callback records. Arrow properties close over `this` once, so
    // no closure is created per query.
    this.rayCbs = {
      boundsTraverseOrder: this.rayBoundsOrder,
      intersectsBounds: this.rayBoundsClosest,
      intersectsRange: this.rayRangeClosest,
    };
    this.rayAllCbs = {
      intersectsBounds: this.rayBoundsAll,
      intersectsRange: this.rayRangeAll,
    };
    this.sweepCbs = {
      boundsTraverseOrder: this.sweepBoundsOrder,
      intersectsBounds: this.sweepBounds,
      intersectsRange: this.sweepRange,
    };
    this.boxCbs = {
      intersectsBounds: this.boxBounds,
      intersectsRange: this.boxRange,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Registration
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Walk `object`, bake every mesh triangle into world space and file it under
   * its surface bucket. Per-mesh `userData.surface` overrides the default;
   * `userData.noCollision === true` skips a subtree.
   *
   * Safe to call in stages during level construction — buckets are marked dirty
   * and rebuilt lazily on the next query, never per frame.
   */
  addStatic(object: THREE.Object3D, surface: SurfaceType): void {
    if (this.registered.has(object)) return;
    this.registered.add(object);

    object.updateWorldMatrix(true, true);
    this.bakeNode(object, surface);
  }

  /**
   * Manual recursion rather than `Object3D.traverse`, because `traverse` has no
   * way to prune: returning early from its callback still descends into the
   * children, so `userData.noCollision` on a group would bake the whole
   * subtree it was meant to exclude.
   */
  private bakeNode(node: THREE.Object3D, surface: SurfaceType): void {
    if (node.userData['noCollision'] === true) return;

    const mesh = node as THREE.Mesh;
    if (mesh.isMesh && mesh.geometry) {
      const override = node.userData['surface'];
      const surf: SurfaceType =
        typeof override === 'string' && VALID_SURFACES.has(override)
          ? (override as SurfaceType)
          : surface;

      const instanced = node as THREE.InstancedMesh;
      if (instanced.isInstancedMesh) {
        for (let i = 0; i < instanced.count; i++) {
          instanced.getMatrixAt(i, _instanceMat);
          _m4.multiplyMatrices(instanced.matrixWorld, _instanceMat);
          this.appendGeometry(mesh.geometry, _m4, surf);
        }
      } else {
        this.appendGeometry(mesh.geometry, mesh.matrixWorld, surf);
      }
    }

    for (let i = 0; i < node.children.length; i++) {
      this.bakeNode(node.children[i], surface);
    }
  }

  /** Drop everything. Used when unloading a level. */
  clear(): void {
    for (const b of this.bucketList) {
      b.bvh = null;
      b.geometry?.dispose();
      b.geometry = null;
      b.index = null;
      b.count = 0;
      b.dirty = false;
    }
    this.buckets.clear();
    this.bucketList.length = 0;
    this.builtTriangles = 0;
    this.dirtyCount = 0;
    this.bounds.makeEmpty();
    // MUST be reset. Retaining the de-dupe set across a clear means re-adding
    // the same level objects after an unload silently registers nothing, and
    // the reloaded level has no collision at all.
    this.registered = new WeakSet<THREE.Object3D>();
  }

  get triangleCount(): number { return this.builtTriangles; }
  get bucketCount(): number { return this.bucketList.length; }
  get surfaces(): readonly SurfaceType[] { return this.bucketList.map((b) => b.surface); }

  private getBucket(surface: SurfaceType): Bucket {
    let b = this.buckets.get(surface);
    if (!b) {
      b = {
        surface,
        pos: new Float32Array(TRI_STRIDE * 256),
        count: 0,
        geometry: null,
        bvh: null,
        index: null,
        dirty: false,
        bMinX: 0, bMinY: 0, bMinZ: 0,
        bMaxX: -1, bMaxY: -1, bMaxZ: -1,
      };
      this.buckets.set(surface, b);
      this.bucketList.push(b);
    }
    return b;
  }

  private appendGeometry(geometry: THREE.BufferGeometry, matrix: THREE.Matrix4, surface: SurfaceType): void {
    const position = geometry.getAttribute('position');
    if (!position) return;

    const index = geometry.getIndex();
    // Floor: a vertex count that is not a multiple of 3 would otherwise read
    // past the attribute and bake NaN vertices into the BVH.
    const triCount = Math.floor((index ? index.count : position.count) / 3);
    if (triCount < 1) return;

    const bucket = this.getBucket(surface);
    this.ensureCapacity(bucket, triCount * TRI_STRIDE);

    const out = bucket.pos;
    let w = bucket.count;

    let written = 0;
    for (let t = 0; t < triCount; t++) {
      const base = t * 3;
      const start = w;
      let finite = true;
      for (let k = 0; k < 3; k++) {
        const vi = index ? index.getX(base + k) : base + k;
        _v3.set(position.getX(vi), position.getY(vi), position.getZ(vi)).applyMatrix4(matrix);
        if (!Number.isFinite(_v3.x) || !Number.isFinite(_v3.y) || !Number.isFinite(_v3.z)) {
          finite = false;
          break;
        }
        out[w++] = _v3.x;
        out[w++] = _v3.y;
        out[w++] = _v3.z;
      }
      // A single non-finite vertex (degenerate source mesh, zero-scaled parent)
      // poisons the BVH node bounds, and from then on every raycast, sweep and
      // depenetration through that bucket returns NaN. Drop the triangle.
      if (finite) written++;
      else w = start;
    }

    bucket.count = w;
    bucket.dirty = true;
    this.dirtyCount++;
    // Counted at append time, not at build time, so `triangleCount` is honest
    // immediately after registration rather than reading 0 until a first query
    // happens to trigger the lazy rebuild.
    this.builtTriangles += written;
  }

  private ensureCapacity(bucket: Bucket, extra: number): void {
    const need = bucket.count + extra;
    if (need <= bucket.pos.length) return;
    let cap = Math.max(bucket.pos.length * 2, TRI_STRIDE * 256);
    while (cap < need) cap *= 2;
    const next = new Float32Array(cap);
    next.set(bucket.pos.subarray(0, bucket.count));
    bucket.pos = next;
  }

  /**
   * Rebuild any dirty bucket. Called at the top of every query; it is a single
   * integer compare in the steady state.
   */
  build(): void {
    if (this.dirtyCount === 0) return;
    this.dirtyCount = 0;
    this.bounds.makeEmpty();

    for (const b of this.bucketList) {
      if (b.dirty) {
        b.dirty = false;
        b.geometry?.dispose();

        if (b.count === 0) {
          b.geometry = null;
          b.bvh = null;
          b.index = null;
        } else {
          // A fresh view over exactly the used range; MeshBVH mutates the index
          // (not the positions), so sharing the underlying buffer is safe.
          const geom = new THREE.BufferGeometry();
          geom.setAttribute('position', new THREE.BufferAttribute(b.pos.subarray(0, b.count), 3));

          // CENTER split: fastest build, and the resulting tree is within a few
          // percent of SAH for architectural geometry. Level loads are already
          // long enough without paying 32-bin SAH on every staged rebuild.
          b.bvh = new MeshBVH(geom, {
            strategy: CENTER,
            maxDepth: 40,
            targetLeafSize: 8,
            setBoundingBox: true,
            verbose: false,
          });
          b.geometry = geom;

          const idx = geom.getIndex();
          // MeshBVH always materialises an index (`ensureIndex`), so this holds.
          b.index = idx ? (idx.array as Uint16Array | Uint32Array) : null;
        }
      }

      if (b.bvh) {
        b.bvh.getBoundingBox(_box);
        b.bMinX = _box.min.x; b.bMinY = _box.min.y; b.bMinZ = _box.min.z;
        b.bMaxX = _box.max.x; b.bMaxY = _box.max.y; b.bMaxZ = _box.max.z;
        this.bounds.union(_box);
      } else {
        b.bMinX = 0; b.bMinY = 0; b.bMinZ = 0;
        b.bMaxX = -1; b.bMaxY = -1; b.bMaxZ = -1;
      }
    }
  }

  /** Slab test a query ray against a bucket's cached root bounds. */
  private bucketMissesRay(b: Bucket, maxDist: number): boolean {
    return rayBoxEntry(
      this.qOx, this.qOy, this.qOz, this.qIdx, this.qIdy, this.qIdz,
      b.bMinX, b.bMinY, b.bMinZ, b.bMaxX, b.bMaxY, b.bMaxZ,
      this.qRadius, maxDist,
    ) === Infinity;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Closest-hit ray
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Nearest static hit along a unit direction. Result is written into `out`.
   * `out.normal` is the RAW geometric normal; callers orient it.
   */
  raycastFirst(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
    out: RayResult,
  ): boolean {
    this.build();
    out.hit = false;
    if (this.bucketList.length === 0) return false;

    this.setRay(ox, oy, oz, dx, dy, dz, maxDist);
    this.qBest = maxDist;
    this.qResult = out;

    for (let i = 0; i < this.bucketList.length; i++) {
      const b = this.bucketList[i];
      if (!b.bvh || !b.index) continue;
      if (this.bucketMissesRay(b, this.qBest)) continue;
      this.qPos = b.pos;
      this.qIndex = b.index;
      this.qSurface = b.surface;
      BVHWorld.baseShapecast.call(b.bvh, this.rayCbs);
    }
    return out.hit;
  }

  private setRay(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
  ): void {
    this.qOx = ox; this.qOy = oy; this.qOz = oz;
    this.qDx = dx; this.qDy = dy; this.qDz = dz;
    this.qMax = maxDist;
    this.qRadius = 0;
    // Guard against 1/0: a huge finite reciprocal keeps the slab test NaN-free
    // when the ray origin sits exactly on a slab plane (0 * Infinity).
    this.qIdx = 1 / (dx !== 0 ? dx : 1e-20);
    this.qIdy = 1 / (dy !== 0 ? dy : 1e-20);
    this.qIdz = 1 / (dz !== 0 ? dz : 1e-20);
  }

  private readonly rayBoundsOrder = (box: THREE.Box3): number => {
    const min = box.min, max = box.max;
    return rayBoxEntry(
      this.qOx, this.qOy, this.qOz, this.qIdx, this.qIdy, this.qIdz,
      min.x, min.y, min.z, max.x, max.y, max.z, 0, this.qBest,
    );
  };

  private readonly rayBoundsClosest = (
    box: THREE.Box3, _isLeaf: boolean, score: number | undefined,
  ): typeof INTERSECTED | typeof NOT_INTERSECTED => {
    // `score` is our own `boundsTraverseOrder` output, already the slab entry
    // distance — reuse it rather than recomputing the slab test.
    const t = score !== undefined
      ? score
      : rayBoxEntry(
        this.qOx, this.qOy, this.qOz, this.qIdx, this.qIdy, this.qIdz,
        box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z, 0, this.qBest,
      );
    return t < this.qBest ? INTERSECTED : NOT_INTERSECTED;
  };

  private readonly rayRangeClosest = (offset: number, count: number): boolean => {
    const pos = this.qPos;
    const idx = this.qIndex;
    const end = offset + count;
    for (let i = offset; i < end; i++) {
      const i3 = i * 3;
      const a = idx[i3] * 3, b = idx[i3 + 1] * 3, c = idx[i3 + 2] * 3;
      const t = rayTriangle(
        this.qOx, this.qOy, this.qOz, this.qDx, this.qDy, this.qDz,
        pos[a], pos[a + 1], pos[a + 2],
        pos[b], pos[b + 1], pos[b + 2],
        pos[c], pos[c + 1], pos[c + 2],
        this.qBest,
      );
      if (t >= 0) {
        this.qBest = t;
        const r = this.qResult;
        r.hit = true;
        r.t = t;
        r.px = this.qOx + this.qDx * t;
        r.py = this.qOy + this.qDy * t;
        r.pz = this.qOz + this.qDz * t;
        r.nx = _nx; r.ny = _ny; r.nz = _nz;
        r.frontFace = _front;
        r.surface = this.qSurface;
      }
    }
    return false; // never stop early; bounds culling on qBest does the work
  };

  // ───────────────────────────────────────────────────────────────────────────
  // All-hits ray (penetration)
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Every static hit along the ray, sorted near→far, INCLUDING back faces.
   *
   * Back faces are the entire point: the weapon system pairs an entry hit with
   * the following exit hit to derive material thickness, and the exit face is
   * always back-facing. Nothing here culls by winding.
   *
   * @returns number of hits written; read them with `hitAt*`.
   */
  raycastAllRaw(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
  ): number {
    this.build();
    this.hitCount = 0;
    this.penetrationTruncated = false;
    if (this.bucketList.length === 0) return 0;

    this.setRay(ox, oy, oz, dx, dy, dz, maxDist);
    this.qBest = maxDist;

    for (let i = 0; i < this.bucketList.length; i++) {
      const b = this.bucketList[i];
      if (!b.bvh || !b.index) continue;
      if (this.bucketMissesRay(b, maxDist)) continue;
      this.qPos = b.pos;
      this.qIndex = b.index;
      this.qSurface = b.surface;
      BVHWorld.baseShapecast.call(b.bvh, this.rayAllCbs);
    }

    this.sortAndDedupeHits();
    return this.hitCount;
  }

  private readonly rayBoundsAll = (
    box: THREE.Box3,
  ): typeof INTERSECTED | typeof NOT_INTERSECTED => {
    const min = box.min, max = box.max;
    const t = rayBoxEntry(
      this.qOx, this.qOy, this.qOz, this.qIdx, this.qIdy, this.qIdz,
      min.x, min.y, min.z, max.x, max.y, max.z, 0, this.qMax,
    );
    return t <= this.qMax ? INTERSECTED : NOT_INTERSECTED;
  };

  private readonly rayRangeAll = (offset: number, count: number): boolean => {
    const pos = this.qPos;
    const idx = this.qIndex;
    const buf = this.hitBuf;
    const end = offset + count;
    for (let i = offset; i < end; i++) {
      const i3 = i * 3;
      const a = idx[i3] * 3, b = idx[i3 + 1] * 3, c = idx[i3 + 2] * 3;
      const t = rayTriangle(
        this.qOx, this.qOy, this.qOz, this.qDx, this.qDy, this.qDz,
        pos[a], pos[a + 1], pos[a + 2],
        pos[b], pos[b + 1], pos[b + 2],
        pos[c], pos[c + 1], pos[c + 2],
        this.qMax,
      );
      if (t < 0) continue;

      let slot = this.hitCount;
      if (slot >= MAX_PENETRATION_HITS) {
        // Full: evict the FARTHEST hit, never stop traversing. BVH visit order
        // is not distance order, so bailing out here would keep an arbitrary
        // subset and could discard the very first surface the bullet strikes.
        this.penetrationTruncated = true;
        let farthest = 0;
        for (let k = 1; k < MAX_PENETRATION_HITS; k++) {
          if (buf[k * HIT_STRIDE] > buf[farthest * HIT_STRIDE]) farthest = k;
        }
        if (buf[farthest * HIT_STRIDE] <= t) continue; // new hit is farther still
        slot = farthest;
      } else {
        this.hitCount++;
      }

      const o = slot * HIT_STRIDE;
      buf[o] = t;
      buf[o + 1] = this.qOx + this.qDx * t;
      buf[o + 2] = this.qOy + this.qDy * t;
      buf[o + 3] = this.qOz + this.qDz * t;
      buf[o + 4] = _nx;
      buf[o + 5] = _ny;
      buf[o + 6] = _nz;
      this.hitFront[slot] = _front ? 1 : 0;
      this.hitSurface[slot] = this.qSurface;
    }
    return false;
  };

  /**
   * Insertion sort over the flat hit buffer, then collapse coincident hits.
   * Insertion sort because n is at most `MAX_PENETRATION_HITS` and it needs no
   * comparator closure, no index array and no allocation.
   */
  private sortAndDedupeHits(): void {
    const buf = this.hitBuf;
    const n = this.hitCount;

    for (let i = 1; i < n; i++) {
      const oi = i * HIT_STRIDE;
      const t = buf[oi];
      const p1 = buf[oi + 1], p2 = buf[oi + 2], p3 = buf[oi + 3];
      const n1 = buf[oi + 4], n2 = buf[oi + 5], n3 = buf[oi + 6];
      const sf = this.hitSurface[i];
      const fr = this.hitFront[i];

      let j = i - 1;
      while (j >= 0 && buf[j * HIT_STRIDE] > t) {
        const from = j * HIT_STRIDE;
        const to = (j + 1) * HIT_STRIDE;
        buf[to] = buf[from];
        buf[to + 1] = buf[from + 1]; buf[to + 2] = buf[from + 2]; buf[to + 3] = buf[from + 3];
        buf[to + 4] = buf[from + 4]; buf[to + 5] = buf[from + 5]; buf[to + 6] = buf[from + 6];
        this.hitSurface[j + 1] = this.hitSurface[j];
        this.hitFront[j + 1] = this.hitFront[j];
        j--;
      }
      const dst = (j + 1) * HIT_STRIDE;
      buf[dst] = t;
      buf[dst + 1] = p1; buf[dst + 2] = p2; buf[dst + 3] = p3;
      buf[dst + 4] = n1; buf[dst + 5] = n2; buf[dst + 6] = n3;
      this.hitSurface[j + 1] = sf;
      this.hitFront[j + 1] = fr;
    }

    // Collapse hits that share a distance AND a facing. Two front faces at the
    // same t is a T-junction/coplanar duplicate, not two walls.
    let w = 0;
    for (let i = 0; i < n; i++) {
      if (w > 0) {
        const prev = (w - 1) * HIT_STRIDE;
        if (buf[i * HIT_STRIDE] - buf[prev] < PENETRATION_MERGE_EPS
          && this.hitFront[w - 1] === this.hitFront[i]) {
          continue;
        }
      }
      if (w !== i) {
        const from = i * HIT_STRIDE;
        const to = w * HIT_STRIDE;
        for (let k = 0; k < HIT_STRIDE; k++) buf[to + k] = buf[from + k];
        this.hitSurface[w] = this.hitSurface[i];
        this.hitFront[w] = this.hitFront[i];
      }
      w++;
    }
    this.hitCount = w;
  }

  hitT(i: number): number { return this.hitBuf[i * HIT_STRIDE]; }
  hitPX(i: number): number { return this.hitBuf[i * HIT_STRIDE + 1]; }
  hitPY(i: number): number { return this.hitBuf[i * HIT_STRIDE + 2]; }
  hitPZ(i: number): number { return this.hitBuf[i * HIT_STRIDE + 3]; }
  hitNX(i: number): number { return this.hitBuf[i * HIT_STRIDE + 4]; }
  hitNY(i: number): number { return this.hitBuf[i * HIT_STRIDE + 5]; }
  hitNZ(i: number): number { return this.hitBuf[i * HIT_STRIDE + 6]; }
  hitSurfaceAt(i: number): SurfaceType { return this.hitSurface[i]; }
  hitFrontAt(i: number): boolean { return this.hitFront[i] === 1; }

  // ───────────────────────────────────────────────────────────────────────────
  // Swept sphere
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Exact swept-sphere against the static world. Not a multi-ray approximation:
   * every candidate triangle gets a plane/edge/vertex continuous test, so the
   * character controller cannot squeeze a capsule through a wall corner.
   */
  sphereCast(
    ox: number, oy: number, oz: number, radius: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
    out: RayResult,
  ): boolean {
    this.build();
    out.hit = false;
    if (this.bucketList.length === 0) return false;

    this.setRay(ox, oy, oz, dx, dy, dz, maxDist);
    this.qRadius = radius;
    this.qBest = maxDist;
    this.qResult = out;

    for (let i = 0; i < this.bucketList.length; i++) {
      const b = this.bucketList[i];
      if (!b.bvh || !b.index) continue;
      if (this.bucketMissesRay(b, this.qBest)) continue;
      this.qPos = b.pos;
      this.qIndex = b.index;
      this.qSurface = b.surface;
      BVHWorld.baseShapecast.call(b.bvh, this.sweepCbs);
    }
    return out.hit;
  }

  private readonly sweepBoundsOrder = (box: THREE.Box3): number => {
    const min = box.min, max = box.max;
    return rayBoxEntry(
      this.qOx, this.qOy, this.qOz, this.qIdx, this.qIdy, this.qIdz,
      min.x, min.y, min.z, max.x, max.y, max.z, this.qRadius, this.qBest,
    );
  };

  private readonly sweepBounds = (
    box: THREE.Box3, _isLeaf: boolean, score: number | undefined,
  ): typeof INTERSECTED | typeof NOT_INTERSECTED => {
    const t = score !== undefined
      ? score
      : rayBoxEntry(
        this.qOx, this.qOy, this.qOz, this.qIdx, this.qIdy, this.qIdz,
        box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z,
        this.qRadius, this.qBest,
      );
    return t < this.qBest ? INTERSECTED : NOT_INTERSECTED;
  };

  private readonly sweepRange = (offset: number, count: number): boolean => {
    const pos = this.qPos;
    const idx = this.qIndex;
    const end = offset + count;
    for (let i = offset; i < end; i++) {
      const i3 = i * 3;
      const a = idx[i3] * 3, b = idx[i3 + 1] * 3, c = idx[i3 + 2] * 3;
      const t = sweptSphereTriangle(
        this.qOx, this.qOy, this.qOz, this.qDx, this.qDy, this.qDz, this.qRadius,
        pos[a], pos[a + 1], pos[a + 2],
        pos[b], pos[b + 1], pos[b + 2],
        pos[c], pos[c + 1], pos[c + 2],
        this.qBest,
      );
      if (t >= 0) {
        this.qBest = t;
        const r = this.qResult;
        r.hit = true;
        r.t = t;
        r.px = _cx; r.py = _cy; r.pz = _cz;
        r.nx = _nx; r.ny = _ny; r.nz = _nz;
        r.frontFace = true; // sweep normals already point back at the sphere
        r.surface = this.qSurface;
      }
    }
    return false;
  };

  // ───────────────────────────────────────────────────────────────────────────
  // AABB triangle collection (rigid-body narrow phase)
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Visit EVERY triangle overlapping an AABB, across all surface buckets.
   *
   * Unbudgeted by design (see `TriangleVisitor`). It also never breaks out of
   * the bucket loop: an earlier, densely tessellated bucket would otherwise
   * exhaust the budget and starve every later bucket, so a ball rolling across
   * a detailed concrete floor would pass straight through a metal wall purely
   * because concrete was registered first.
   *
   * @returns the number of triangles visited.
   */
  traverseBox(
    minx: number, miny: number, minz: number,
    maxx: number, maxy: number, maxz: number,
    visitor: TriangleVisitor,
  ): number {
    this.build();
    this.qTriCount = 0;
    if (this.bucketList.length === 0) return 0;

    this.qBMinX = minx; this.qBMinY = miny; this.qBMinZ = minz;
    this.qBMaxX = maxx; this.qBMaxY = maxy; this.qBMaxZ = maxz;
    this.qVisitor = visitor;

    for (let i = 0; i < this.bucketList.length; i++) {
      const b = this.bucketList[i];
      if (!b.bvh || !b.index) continue;
      if (b.bMaxX < minx || b.bMinX > maxx) continue;
      if (b.bMaxY < miny || b.bMinY > maxy) continue;
      if (b.bMaxZ < minz || b.bMinZ > maxz) continue;
      this.qPos = b.pos;
      this.qIndex = b.index;
      this.qSurface = b.surface;
      BVHWorld.baseShapecast.call(b.bvh, this.boxCbs);
    }
    this.qVisitor = null;
    return this.qTriCount;
  }

  private readonly boxBounds = (
    box: THREE.Box3,
  ): typeof INTERSECTED | typeof NOT_INTERSECTED => {
    const min = box.min, max = box.max;
    if (max.x < this.qBMinX || min.x > this.qBMaxX) return NOT_INTERSECTED;
    if (max.y < this.qBMinY || min.y > this.qBMaxY) return NOT_INTERSECTED;
    if (max.z < this.qBMinZ || min.z > this.qBMaxZ) return NOT_INTERSECTED;
    // Never report CONTAINED: that would flush every triangle in the subtree.
    return INTERSECTED;
  };

  private readonly boxRange = (offset: number, count: number): boolean => {
    const pos = this.qPos;
    const idx = this.qIndex;
    const visitor = this.qVisitor;
    if (!visitor) return true;
    const end = offset + count;

    for (let i = offset; i < end; i++) {
      const i3 = i * 3;
      const a = idx[i3] * 3, b = idx[i3 + 1] * 3, c = idx[i3 + 2] * 3;

      const ax = pos[a], ay = pos[a + 1], az = pos[a + 2];
      const bx = pos[b], by = pos[b + 1], bz = pos[b + 2];
      const cx = pos[c], cy = pos[c + 1], cz = pos[c + 2];

      // Cheap triangle-AABB reject on the triangle's own bounds.
      if (Math.min(ax, bx, cx) > this.qBMaxX || Math.max(ax, bx, cx) < this.qBMinX) continue;
      if (Math.min(ay, by, cy) > this.qBMaxY || Math.max(ay, by, cy) < this.qBMinY) continue;
      if (Math.min(az, bz, cz) > this.qBMaxZ || Math.max(az, bz, cz) < this.qBMinZ) continue;

      this.qTriCount++;
      visitor.visitTriangle(ax, ay, az, bx, by, bz, cx, cy, cz, this.qSurface);
    }
    return false;
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Depenetration
  // ───────────────────────────────────────────────────────────────────────────

  private dpX = 0; private dpY = 0; private dpZ = 0;
  private dpR = 0;
  private dpHits = 0;
  /** Contacts gathered this pass: unit normals and required push distances. */
  private readonly dpNormals = new Float32Array(DEPENETRATION_CONTACTS * 3);
  private readonly dpDepths = new Float32Array(DEPENETRATION_CONTACTS);

  /**
   * Push a sphere out of any static geometry it overlaps.
   *
   * A sweep cannot recover a shape that is ALREADY inside a surface (it reports
   * distance 0 with nowhere to go), so a character controller that ends a frame
   * slightly embedded — which happens constantly on slopes and in corners — has
   * no way back out without this. Results are accumulated per contact so
   * wedging into a corner resolves along both walls rather than oscillating.
   *
   * @returns true when a correction was applied; the corrected centre is
   *   written to `out`.
   */
  depenetrateSphere(cx: number, cy: number, cz: number, radius: number, out: THREE.Vector3, iterations = 4): boolean {
    this.build();
    this.dpX = cx; this.dpY = cy; this.dpZ = cz; this.dpR = radius;
    let moved = false;

    for (let it = 0; it < iterations; it++) {
      this.dpHits = 0;
      this.traverseBox(
        this.dpX - radius, this.dpY - radius, this.dpZ - radius,
        this.dpX + radius, this.dpY + radius, this.dpZ + radius,
        this.depenetrator,
      );
      if (this.dpHits === 0) break;

      // Minimum-translation accumulation.
      //
      // Naively summing every triangle's push double-counts a shared edge and
      // lets the two faces of a thin wall cancel; taking only the smallest push
      // stalls on grazing contacts and never escapes a corner. Instead each
      // contact contributes only the part of its push not already covered by
      // the accumulated correction, so orthogonal walls compose in one pass and
      // coplanar duplicates contribute nothing.
      let axc = 0, ayc = 0, azc = 0;
      for (let i = 0; i < this.dpHits; i++) {
        const nx = this.dpNormals[i * 3];
        const ny = this.dpNormals[i * 3 + 1];
        const nz = this.dpNormals[i * 3 + 2];
        const need = this.dpDepths[i] - (axc * nx + ayc * ny + azc * nz);
        if (need <= 0) continue;
        axc += nx * need; ayc += ny * need; azc += nz * need;
      }
      if (axc === 0 && ayc === 0 && azc === 0) break;
      this.dpX += axc; this.dpY += ayc; this.dpZ += azc;
      moved = true;
    }

    out.set(this.dpX, this.dpY, this.dpZ);
    return moved;
  }

  /** Persistent visitor so `depenetrateSphere` allocates nothing. */
  private readonly depenetrator: TriangleVisitor = {
    visitTriangle: (
      ax: number, ay: number, az: number,
      bx: number, by: number, bz: number,
      cx: number, cy: number, cz: number,
    ): void => {
      closestPointOnTriangle(this.dpX, this.dpY, this.dpZ, ax, ay, az, bx, by, bz, cx, cy, cz);
      let dx = this.dpX - _cx, dy = this.dpY - _cy, dz = this.dpZ - _cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      const r = this.dpR;
      if (d2 >= r * r) return;
      const d = Math.sqrt(d2);
      if (d > 1e-7) {
        const inv = 1 / d;
        dx *= inv; dy *= inv; dz *= inv;
      } else {
        const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        dx = e1y * e2z - e1z * e2y;
        dy = e1z * e2x - e1x * e2z;
        dz = e1x * e2y - e1y * e2x;
        const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (l < 1e-12) return;
        const inv = 1 / l;
        dx *= inv; dy *= inv; dz *= inv;
      }
      if (this.dpHits >= DEPENETRATION_CONTACTS) return;
      const o = this.dpHits * 3;
      this.dpNormals[o] = dx; this.dpNormals[o + 1] = dy; this.dpNormals[o + 2] = dz;
      this.dpDepths[this.dpHits] = r - d;
      this.dpHits++;
    },
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Debug
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Emit BVH node bounds up to `maxDepth` as wireframe boxes.
   * `emit(minx,miny,minz,maxx,maxy,maxz)` is called once per node.
   *
   * Uses `BVH.traverse`, which allocates a small Float32Array view per node —
   * acceptable because this only runs when the debug view is (re)built.
   */
  debugBounds(maxDepth: number, emit: (
    minx: number, miny: number, minz: number,
    maxx: number, maxy: number, maxz: number,
  ) => void): void {
    this.build();
    for (const b of this.bucketList) {
      const bvh = b.bvh;
      if (!bvh) continue;
      bvh.traverse((depth, isLeaf, bounds) => {
        if (depth > maxDepth) return true;
        if (depth === maxDepth || isLeaf) {
          emit(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);
          return true;
        }
        return false;
      });
    }
  }

  dispose(): void {
    this.clear();
  }
}
