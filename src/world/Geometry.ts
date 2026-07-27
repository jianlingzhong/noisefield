/**
 * NOISEFIELD — Procedural geometry toolkit.
 *
 * Everything in the level is code-authored: zero external assets. This module
 * is the low-level vocabulary that `Props.ts` and `Level.ts` speak.
 *
 * Three ideas drive the design:
 *
 *  1. NOTHING IS A RAW BOX. `chamferedBox` is the workhorse primitive. A 2-4 cm
 *     bevel on every edge is the single cheapest, strongest cue that geometry
 *     was *modelled* rather than *primitived*: the bevel band catches a
 *     specular highlight from the low golden-hour sun and reads as a real
 *     material edge instead of an infinitely-sharp mathematical one. 44 tris
 *     versus 12 for a BoxGeometry is a trade we take every single time.
 *
 *  2. WORLD-SCALE TEXEL DENSITY. `applyBoxUV` re-projects UVs from world space
 *     at a fixed metres-per-tile, so one square metre of wall gets exactly the
 *     same number of texels as one square metre of floor. Mismatched density is
 *     the classic amateur tell and it is *free* to avoid if you bake UVs after
 *     the world transform, which `GeometryBatcher` does automatically.
 *
 *  3. MERGE OR INSTANCE, NEVER BOTH-AND-NEITHER. `GeometryBatcher` accumulates
 *     every static surface that shares a material into one buffer, so the
 *     entire compound's concrete is a single draw call. Repeated props go
 *     through InstancedMesh instead. There is no third option.
 */

import * as THREE from 'three';
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic randomness
// ─────────────────────────────────────────────────────────────────────────────

/**
 * mulberry32. Small, fast, good enough for scatter and jitter.
 *
 * A *seeded* generator matters more than it looks: the level must be
 * bit-identical every load so that spawn points, cover placement and sightlines
 * are stable between sessions. A map that reshuffles itself is unlearnable, and
 * learnability is the whole point of an arena shooter.
 */
export class Rng {
  private s: number;

  constructor(seed: number) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }

  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [a, b). */
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  /** Uniform integer in [a, b] inclusive. */
  int(a: number, b: number): number {
    return Math.floor(a + (b - a + 1) * this.next());
  }

  sign(): number {
    return this.next() < 0.5 ? -1 : 1;
  }

  chance(p: number): boolean {
    return this.next() < p;
  }

  pick<T>(arr: readonly T[]): T {
    return arr[Math.min(arr.length - 1, Math.floor(this.next() * arr.length))];
  }

  /** Central-limit approximation of a normal distribution. */
  gauss(mean = 0, sd = 1): number {
    const u = this.next() + this.next() + this.next() + this.next();
    return mean + (u - 2) * 0.8660254 * sd;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attribute hygiene
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reduce a geometry to exactly {position, normal, uv}, indexed.
 *
 * `mergeGeometries` refuses to merge buffers whose attribute *sets* differ, and
 * three's primitives are inconsistent (ExtrudeGeometry is non-indexed,
 * everything else is indexed; some carry tangents). Normalising up front turns
 * a whole class of runtime merge failures into a non-issue.
 */
export function normalizeAttributes(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  const pos = geo.getAttribute('position');
  // Degenerate inputs are legitimate: a prop factory can be asked for a wall
  // with no solid spans left after its openings, or a rubble pile with zero
  // chunks. Hand the empty buffer straight back rather than exploding, and let
  // the batcher drop it.
  if (!pos || pos.count === 0) return geo;
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  if (!geo.getAttribute('uv')) {
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(pos.count * 2), 2));
  }
  for (const name of Object.keys(geo.attributes)) {
    if (name !== 'position' && name !== 'normal' && name !== 'uv') {
      geo.deleteAttribute(name);
    }
  }
  geo.morphAttributes = {};
  geo.clearGroups();
  return geo.index ? geo : mergeVertices(geo, 1e-5);
}

/**
 * World-space box (triplanar-select) UV projection at a fixed metres-per-tile.
 *
 * For each vertex we pick the dominant axis of its normal and project the
 * position onto the other two. Because the batcher runs this *after* the world
 * matrix is baked in, every surface in the map ends up on the same virtual
 * texture grid — walls, floors and props all line up and share texel density.
 */
export function applyBoxUV(geo: THREE.BufferGeometry, metersPerTile = 2, offsetU = 0, offsetV = 0): void {
  const pos = geo.getAttribute('position');
  const nor = geo.getAttribute('normal');
  if (!pos || !nor) return;
  const inv = 1 / Math.max(1e-4, metersPerTile);
  const uv = new Float32Array(pos.count * 2);

  for (let i = 0; i < pos.count; i++) {
    const px = pos.getX(i);
    const py = pos.getY(i);
    const pz = pos.getZ(i);
    const ax = Math.abs(nor.getX(i));
    const ay = Math.abs(nor.getY(i));
    const az = Math.abs(nor.getZ(i));

    let u: number;
    let v: number;
    if (ax >= ay && ax >= az) {
      u = pz;
      v = py;
    } else if (ay >= az) {
      u = px;
      v = pz;
    } else {
      u = px;
      v = py;
    }
    uv[i * 2] = u * inv + offsetU;
    uv[i * 2 + 1] = v * inv + offsetV;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}

/** Duplicate `uv` into `uv1`, the channel three uses for aoMap/lightMap. */
export function ensureUV1(geo: THREE.BufferGeometry): void {
  const uv = geo.getAttribute('uv');
  if (!uv || geo.getAttribute('uv1')) return;
  geo.setAttribute('uv1', new THREE.BufferAttribute(new Float32Array(uv.array as Float32Array), 2));
}

// ─────────────────────────────────────────────────────────────────────────────
// chamferedBox — the workhorse
// ─────────────────────────────────────────────────────────────────────────────

type V3 = [number, number, number];

function norm3(v: V3): V3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/** Scratch builder shared by the chamfer routine; build-time only, never hot. */
class TriBuilder {
  readonly pos: number[] = [];
  readonly nor: number[] = [];
  readonly uv: number[] = [];
  readonly idx: number[] = [];

  vertex(p: V3, n: V3): number {
    const i = this.pos.length / 3;
    this.pos.push(p[0], p[1], p[2]);
    this.nor.push(n[0], n[1], n[2]);
    // Provisional planar UV at 1 unit = 1 metre. `applyBoxUV` will normally
    // overwrite this, but standalone users of the helper still get sane UVs.
    const ax = Math.abs(n[0]);
    const ay = Math.abs(n[1]);
    const az = Math.abs(n[2]);
    if (ax >= ay && ax >= az) this.uv.push(p[2], p[1]);
    else if (ay >= az) this.uv.push(p[0], p[2]);
    else this.uv.push(p[0], p[1]);
    return i;
  }

  /**
   * Emit a triangle with winding chosen so its geometric normal agrees with
   * `outward`. Deriving winding instead of hand-authoring it removes an entire
   * category of "why is this face invisible" bugs from the 26 distinct
   * face/edge/corner cases below.
   */
  tri(a: number, b: number, c: number, outward: V3): void {
    const p = this.pos;
    const ax = p[a * 3], ay = p[a * 3 + 1], az = p[a * 3 + 2];
    const e1x = p[b * 3] - ax, e1y = p[b * 3 + 1] - ay, e1z = p[b * 3 + 2] - az;
    const e2x = p[c * 3] - ax, e2y = p[c * 3 + 1] - ay, e2z = p[c * 3 + 2] - az;
    const cx = e1y * e2z - e1z * e2y;
    const cy = e1z * e2x - e1x * e2z;
    const cz = e1x * e2y - e1y * e2x;
    if (cx * outward[0] + cy * outward[1] + cz * outward[2] >= 0) this.idx.push(a, b, c);
    else this.idx.push(a, c, b);
  }

  toGeometry(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setIndex(this.idx);
    return g;
  }
}

/**
 * A box with all twelve edges chamfered (segments = 1) or filleted
 * (segments > 1), centred on the origin.
 *
 * Construction is a Minkowski offset: take a "core" box shrunk by `bevel` on
 * every axis and sweep a sphere of radius `bevel` over it. Faces stay flat,
 * edges become cylindrical bands, corners become spherical patches. At
 * segments = 1 the bands collapse to single 45-degree facets — a true chamfer —
 * at a cost of 44 triangles.
 *
 * Because edge and corner vertices carry the *sphere* normal, even the 1-segment
 * chamfer shades as a smooth rolled edge, which is what actually sells it.
 *
 * @param width  Full X extent, metres.
 * @param height Full Y extent, metres.
 * @param depth  Full Z extent, metres.
 * @param bevel  Chamfer size, metres. Clamped to just under half the smallest axis.
 * @param segments Subdivisions across the chamfer. 1 = flat facet, 2-3 = rolled.
 */
export function chamferedBox(
  width: number,
  height: number,
  depth: number,
  bevel = 0.025,
  segments = 1,
): THREE.BufferGeometry {
  const half: V3 = [width * 0.5, height * 0.5, depth * 0.5];
  const b = Math.max(1e-4, Math.min(bevel, Math.min(half[0], half[1], half[2]) * 0.98));
  const s = Math.max(1, Math.floor(segments));
  const core: V3 = [half[0] - b, half[1] - b, half[2] - b];

  const tb = new TriBuilder();

  // ── 6 flat faces, inset by the bevel on both in-plane axes ────────────────
  for (let a = 0; a < 3; a++) {
    const u = (a + 1) % 3;
    const w = (a + 2) % 3;
    for (const sgn of [-1, 1]) {
      const n: V3 = [0, 0, 0];
      n[a] = sgn;
      const ring: number[] = [];
      for (const [su, sw] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
        const p: V3 = [0, 0, 0];
        p[a] = sgn * half[a];
        p[u] = su * core[u];
        p[w] = sw * core[w];
        ring.push(tb.vertex(p, n));
      }
      tb.tri(ring[0], ring[1], ring[2], n);
      tb.tri(ring[0], ring[2], ring[3], n);
    }
  }

  // ── 12 edge bands ─────────────────────────────────────────────────────────
  for (let a = 0; a < 3; a++) {
    for (let c = a + 1; c < 3; c++) {
      const t = 3 - a - c; // the axis the edge runs along
      for (const sa of [-1, 1]) {
        for (const sc of [-1, 1]) {
          const nA: V3 = [0, 0, 0];
          nA[a] = sa;
          const nC: V3 = [0, 0, 0];
          nC[c] = sc;

          const dirs: V3[] = [];
          for (let i = 0; i <= s; i++) {
            const k = i / s;
            dirs.push(norm3([
              nA[0] * (1 - k) + nC[0] * k,
              nA[1] * (1 - k) + nC[1] * k,
              nA[2] * (1 - k) + nC[2] * k,
            ]));
          }

          const c0: V3 = [0, 0, 0];
          c0[a] = sa * core[a];
          c0[c] = sc * core[c];
          c0[t] = -core[t];
          const c1: V3 = [c0[0], c0[1], c0[2]];
          c1[t] = core[t];

          let prev0 = -1;
          let prev1 = -1;
          for (let i = 0; i <= s; i++) {
            const d = dirs[i];
            const v0 = tb.vertex([c0[0] + b * d[0], c0[1] + b * d[1], c0[2] + b * d[2]], d);
            const v1 = tb.vertex([c1[0] + b * d[0], c1[1] + b * d[1], c1[2] + b * d[2]], d);
            if (i > 0) {
              const mid = norm3([
                dirs[i - 1][0] + d[0],
                dirs[i - 1][1] + d[1],
                dirs[i - 1][2] + d[2],
              ]);
              tb.tri(prev0, v0, v1, mid);
              tb.tri(prev0, v1, prev1, mid);
            }
            prev0 = v0;
            prev1 = v1;
          }
        }
      }
    }
  }

  // ── 8 corner patches (barycentric subdivision of a spherical triangle) ────
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const C: V3 = [sx * core[0], sy * core[1], sz * core[2]];
        const nX: V3 = [sx, 0, 0];
        const nY: V3 = [0, sy, 0];
        const nZ: V3 = [0, 0, sz];

        // rows[i][j], i = 0..s, j = 0..i
        const rows: number[][] = [];
        const dirRows: V3[][] = [];
        for (let i = 0; i <= s; i++) {
          const row: number[] = [];
          const drow: V3[] = [];
          for (let j = 0; j <= i; j++) {
            const wx = (i - j) / s;
            const wy = j / s;
            const wz = (s - i) / s;
            const d = norm3([
              nX[0] * wx + nY[0] * wy + nZ[0] * wz,
              nX[1] * wx + nY[1] * wy + nZ[1] * wz,
              nX[2] * wx + nY[2] * wy + nZ[2] * wz,
            ]);
            drow.push(d);
            row.push(tb.vertex([C[0] + b * d[0], C[1] + b * d[1], C[2] + b * d[2]], d));
          }
          rows.push(row);
          dirRows.push(drow);
        }

        const outward = norm3([sx, sy, sz]);
        for (let i = 0; i < s; i++) {
          for (let j = 0; j <= i; j++) {
            tb.tri(rows[i][j], rows[i + 1][j], rows[i + 1][j + 1], outward);
            if (j < i) tb.tri(rows[i][j], rows[i + 1][j + 1], rows[i][j + 1], outward);
          }
        }
      }
    }
  }

  return tb.toGeometry();
}

/**
 * A cylinder with chamfered top and bottom rims, built as a lathe so the
 * silhouette break at the rim is a real facet rather than a hard 90-degree
 * corner. Used for drums, bollards, pipe sections and tower footings.
 */
export function chamferedCylinder(
  radius: number,
  height: number,
  bevel = 0.03,
  radialSegments = 16,
  capped = true,
): THREE.BufferGeometry {
  const b = Math.max(1e-4, Math.min(bevel, Math.min(radius, height * 0.5) * 0.9));
  const pts: THREE.Vector2[] = [];
  if (capped) pts.push(new THREE.Vector2(0, 0));
  pts.push(new THREE.Vector2(radius - b, 0));
  pts.push(new THREE.Vector2(radius, b));
  pts.push(new THREE.Vector2(radius, height - b));
  pts.push(new THREE.Vector2(radius - b, height));
  if (capped) pts.push(new THREE.Vector2(0, height));
  const g = new THREE.LatheGeometry(pts, radialSegments);
  g.translate(0, -height * 0.5, 0);
  return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Architectural helpers
// ─────────────────────────────────────────────────────────────────────────────

/** A rectangular hole in a wall, in wall-local coordinates. */
export interface WallOpening {
  /** Centre of the opening along the wall's local X, relative to wall centre. */
  x: number;
  /** Height of the opening's bottom edge above the wall base. */
  y: number;
  width: number;
  height: number;
}

/** Sorted unique values, collapsing anything closer together than `eps`. */
function uniqSorted(values: number[], eps = 1e-4): number[] {
  const s = [...values].sort((a, b) => a - b);
  const out: number[] = [];
  for (const v of s) if (out.length === 0 || v - out[out.length - 1] > eps) out.push(v);
  return out;
}

/**
 * A wall panel with rectangular openings punched through it.
 *
 * No CSG. The opening edges induce a grid over the wall; every cell whose
 * centre is not inside an opening is solid, and the solid cells are greedily
 * merged back into maximal rectangles before being emitted as chamfered boxes.
 * That is exact for arbitrary openings — including *vertically stacked* windows
 * on the same pier and overlapping spans, which a naive left-to-right
 * pier/sill/lintel walk silently gets wrong (it fills the lower window in with
 * the upper window's sill).
 *
 * The greedy merge matters: the raw grid for a six-window facade is ~100 cells,
 * and merging collapses it to ~20 boxes, so correctness costs almost nothing.
 *
 * Every resulting piece is a chamferedBox, so the reveals around a window read
 * as real wall *thickness*. A window must be an inset; this is what makes it one.
 *
 * Returned geometry is centred on X and Z with its base at y = 0.
 */
export function wallWithOpenings(
  width: number,
  height: number,
  thickness: number,
  openings: WallOpening[] = [],
  bevel = 0.03,
): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const halfW = width * 0.5;

  const block = (x0: number, x1: number, y0: number, y1: number): void => {
    const w = x1 - x0;
    const h = y1 - y0;
    if (w <= 1e-3 || h <= 1e-3) return;
    const g = chamferedBox(w, h, thickness, Math.min(bevel, Math.min(w, h, thickness) * 0.4));
    g.translate((x0 + x1) * 0.5, (y0 + y1) * 0.5, 0);
    parts.push(g);
  };

  // Clip every opening to the wall and discard the degenerate ones.
  const ops = openings
    .map((o) => ({
      x0: Math.max(-halfW, o.x - o.width * 0.5),
      x1: Math.min(halfW, o.x + o.width * 0.5),
      y0: Math.max(0, o.y),
      y1: Math.min(height, o.y + o.height),
    }))
    .filter((o) => o.x1 - o.x0 > 1e-3 && o.y1 - o.y0 > 1e-3);

  if (ops.length === 0) {
    block(-halfW, halfW, 0, height);
    return parts.length ? mergeGeometries(parts.map(normalizeAttributes)) : new THREE.BufferGeometry();
  }

  const xs = uniqSorted([-halfW, halfW, ...ops.flatMap((o) => [o.x0, o.x1])]);
  const ys = uniqSorted([0, height, ...ops.flatMap((o) => [o.y0, o.y1])]);
  const cols = xs.length - 1;
  const rows = ys.length - 1;

  // Occupancy by cell-centre containment. Centres can never land on an opening
  // boundary because every boundary is itself a grid line.
  const solid: boolean[] = new Array(rows * cols).fill(true);
  for (let r = 0; r < rows; r++) {
    const cy = (ys[r] + ys[r + 1]) * 0.5;
    for (let c = 0; c < cols; c++) {
      const cx = (xs[c] + xs[c + 1]) * 0.5;
      for (const o of ops) {
        if (cx > o.x0 && cx < o.x1 && cy > o.y0 && cy < o.y1) {
          solid[r * cols + c] = false;
          break;
        }
      }
    }
  }

  // Greedy maximal-rectangle decomposition: grow right, then grow down while
  // the whole run stays solid and unclaimed.
  const used: boolean[] = new Array(rows * cols).fill(false);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      if (!solid[i] || used[i]) continue;
      let c1 = c;
      while (c1 + 1 < cols && solid[r * cols + c1 + 1] && !used[r * cols + c1 + 1]) c1++;
      let r1 = r;
      grow: while (r1 + 1 < rows) {
        for (let k = c; k <= c1; k++) {
          const j = (r1 + 1) * cols + k;
          if (!solid[j] || used[j]) break grow;
        }
        r1++;
      }
      for (let rr = r; rr <= r1; rr++) for (let cc = c; cc <= c1; cc++) used[rr * cols + cc] = true;
      block(xs[c], xs[c1 + 1], ys[r], ys[r1 + 1]);
    }
  }

  if (parts.length === 0) return new THREE.BufferGeometry();
  return mergeGeometries(parts.map(normalizeAttributes));
}

/**
 * A run of stairs climbing in +Z, base at y = 0, centred on X.
 *
 * Treads are chamfered so the nosing catches light — a staircase lit only by a
 * low sun is otherwise a flat grey wedge. Stringers on both sides close the
 * silhouette from the side, which matters because stairs are usually seen in
 * profile across a system.
 */
export function staircase(
  steps: number,
  totalRise: number,
  totalRun: number,
  width: number,
  bevel = 0.02,
): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const rise = totalRise / steps;
  const run = totalRun / steps;

  for (let i = 0; i < steps; i++) {
    // Each tread is a solid block from the ground up: cheap, and it means the
    // underside is never visible as a floating slab from a lower position.
    const h = rise * (i + 1);
    const g = chamferedBox(width, h, run + 0.03, bevel);
    g.translate(0, h * 0.5, run * (i + 0.5));
    parts.push(g);
  }
  // Side stringers: a raked kerb either side, tying the flight together and
  // closing the silhouette in profile (stairs are usually seen side-on across a
  // system). Length and centre are chosen so the top face finishes a few
  // centimetres proud of the last tread rather than sailing past it.
  for (const sx of [-1, 1]) {
    const g = chamferedBox(0.12, 0.26, Math.hypot(totalRun, totalRise), 0.02);
    g.rotateX(-Math.atan2(totalRise, totalRun));
    g.translate(sx * (width * 0.5 + 0.05), totalRise * 0.5 - 0.04, totalRun * 0.5);
    parts.push(g);
  }
  return mergeGeometries(parts.map(normalizeAttributes));
}

/**
 * A ramp / wedge: rectangular in plan, rising from y=0 at z=-depth/2 to
 * `height` at z=+depth/2. Used for rubble ramps and loading docks.
 */
export function rampGeometry(width: number, height: number, depth: number, bevel = 0.03): THREE.BufferGeometry {
  const len = Math.hypot(depth, height);
  const g = chamferedBox(width, 0.28, len, bevel);
  g.rotateX(-Math.atan2(height, depth));
  g.translate(0, height * 0.5 - 0.02, 0);
  return g;
}

/**
 * A sand drift: a soft sloped fillet that accumulates against a vertical
 * surface, running along X with the wall at +Z.
 *
 * This is the single highest-value-per-triangle detail in the whole map. A
 * building meeting the ground at a perfect 90-degree line looks like a render;
 * two triangles of wind-blown sand piled into that corner looks like a place.
 */
export function sandDrift(
  length: number,
  height: number,
  depth: number,
  rng: Rng,
  segments = 12,
): THREE.BufferGeometry {
  const tb = new TriBuilder();
  const rows: number[][] = [];
  // Two rows: the drift crest against the wall, and the feathered toe.
  for (let r = 0; r <= 1; r++) {
    const row: number[] = [];
    for (let i = 0; i <= segments; i++) {
      const u = i / segments;
      // Taper the drift toward both ends so it doesn't read as an extruded prism.
      const taper = Math.sin(Math.PI * u) ** 0.6;
      const wobble = 0.75 + 0.25 * Math.sin(u * 11.3 + rng.next() * 0.01);
      const h = r === 1 ? height * taper * wobble : 0;
      const z = r === 1 ? depth * 0.5 : -depth * 0.5 * (0.6 + 0.4 * taper);
      const x = (u - 0.5) * length;
      // Normal leans away from the wall and up: a shallow slope catching the sun.
      const n = norm3([0, depth, -height]);
      row.push(tb.vertex([x, h, z], n));
    }
    rows.push(row);
  }
  for (let i = 0; i < segments; i++) {
    tb.tri(rows[0][i], rows[0][i + 1], rows[1][i + 1], [0, 1, -0.4]);
    tb.tri(rows[0][i], rows[1][i + 1], rows[1][i], [0, 1, -0.4]);
  }
  return tb.toGeometry();
}

/**
 * A corrugated sheet in the XY plane, ribs running vertically, thickness in -Z.
 * Shipping containers, roller shutters and shanty roofing all depend on this;
 * a flat plane with a normal map would fall apart in silhouette against the
 * skyline, and silhouette is what you actually read at 40 m.
 */
export function corrugatedPanel(
  width: number,
  height: number,
  ribs: number,
  ribDepth = 0.02,
  thickness = 0.03,
): THREE.BufferGeometry {
  const tb = new TriBuilder();
  const cols = Math.max(2, ribs * 2);
  const front: number[] = [];
  const frontLo: number[] = [];
  for (let i = 0; i <= cols; i++) {
    const u = i / cols;
    const x = (u - 0.5) * width;
    // Trapezoidal wave, the real profile of shipping-container steel.
    const phase = (i % 2) === 0 ? 0 : 1;
    const z = phase * ribDepth;
    // Slanted normal on the rib flanks catches a rim light along each fold.
    const nx = (i % 2 === 0 ? -1 : 1) * 0.45;
    const n = norm3([nx, 0, 1]);
    front.push(tb.vertex([x, height * 0.5, z], n));
    frontLo.push(tb.vertex([x, -height * 0.5, z], n));
  }
  for (let i = 0; i < cols; i++) {
    tb.tri(front[i], frontLo[i], frontLo[i + 1], [0, 0, 1]);
    tb.tri(front[i], frontLo[i + 1], front[i + 1], [0, 0, 1]);
  }
  // Flat back plate + rim so the panel is a closed solid.
  const back = chamferedBox(width, height, thickness, 0.008);
  back.translate(0, 0, -thickness * 0.5);
  return mergeGeometries([normalizeAttributes(tb.toGeometry()), normalizeAttributes(back)]);
}

/**
 * A hanging cloth panel: attached along the -Z edge, sagging toward +Z.
 * `sag` is the droop of the free edge in metres.
 *
 * Awnings and laundry are the two cheapest ways to break up an otherwise rigid
 * orthogonal city, and cloth is the only thing in the scene allowed to curve.
 */
export function saggingCloth(
  width: number,
  depth: number,
  sag = 0.2,
  segsW = 8,
  segsD = 4,
  rng?: Rng,
): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(width, depth, segsW, segsD);
  g.rotateX(-Math.PI / 2); // into the XZ plane, +Z is the free edge
  const pos = g.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const v = (z / depth) + 0.5;          // 0 at attached edge, 1 at free edge
    const u = (x / width) + 0.5;
    const across = Math.sin(Math.PI * Math.min(1, Math.max(0, u)));
    const along = v * v;                   // droop accelerates toward the free edge
    const ripple = rng ? Math.sin(u * 9.1 + rng.next() * 0.001) * 0.012 : 0;
    pos.setY(i, -sag * across * along + ripple * across);
  }
  g.computeVertexNormals();
  return g;
}

/** Catenary-ish sag between two points; used for cables and laundry lines. */
export function catenary(a: THREE.Vector3, b: THREE.Vector3, sag: number, segments = 12): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const p = new THREE.Vector3().lerpVectors(a, b, t);
    p.y -= sag * 4 * t * (1 - t); // parabola: visually indistinguishable from a catenary
    pts.push(p);
  }
  return pts;
}

/** A swept tube through a polyline. Pipes, cables, handrails, guy-wires. */
export function tubeThrough(
  points: THREE.Vector3[],
  radius: number,
  radialSegments = 6,
  tubularSegments = 24,
): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.4);
  return new THREE.TubeGeometry(curve, tubularSegments, radius, radialSegments, false);
}

/**
 * A faceted rock / rubble chunk. Icosahedron with per-vertex radial noise, left
 * flat-shaded so every face is its own plane — smooth rocks look like potatoes.
 */
export function rockGeometry(radius: number, rng: Rng, detail = 0): THREE.BufferGeometry {
  const g = new THREE.IcosahedronGeometry(radius, detail);
  const pos = g.getAttribute('position');
  // Vertices are shared per-face in PolyhedronGeometry (it is non-indexed), so
  // we hash the position to keep the displacement consistent across faces and
  // avoid tearing the mesh apart.
  const cache = new Map<string, number>();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const key = `${x.toFixed(3)},${y.toFixed(3)},${z.toFixed(3)}`;
    let f = cache.get(key);
    if (f === undefined) {
      f = rng.range(0.62, 1.18);
      cache.set(key, f);
    }
    pos.setXYZ(i, x * f, y * f * rng.range(0.85, 0.95), z * f);
  }
  g.computeVertexNormals();
  return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Batching
// ─────────────────────────────────────────────────────────────────────────────

export interface BatchOptions {
  /** Keep the geometry's authored UVs instead of re-projecting from world space. */
  keepUV?: boolean;
}

/**
 * Accumulates static geometry per material key and merges each bucket into one
 * BufferGeometry.
 *
 * PERF: this is the difference between ~40 and ~2000 draw calls. Every wall,
 * pier, lintel, stair tread, roof coping and window reveal in the compound
 * shares one of a dozen materials, and there is no reason for the GPU to be
 * told about them separately. The trade is that merged geometry cannot be
 * culled individually — acceptable here because the map is 90 m across and the
 * player can see most of it anyway, so per-object culling would win nothing.
 *
 * World transforms are baked at `add` time, and UVs are projected immediately
 * afterwards, which is what guarantees world-consistent texel density.
 */
export class GeometryBatcher {
  private readonly buckets = new Map<string, THREE.BufferGeometry[]>();
  private readonly tileFor: (key: string) => number;

  constructor(tileFor: (key: string) => number) {
    this.tileFor = tileFor;
  }

  add(key: string, geo: THREE.BufferGeometry, matrix?: THREE.Matrix4, opts?: BatchOptions): void {
    const src = geo.getAttribute('position');
    if (!src || src.count === 0) return; // degenerate; nothing to merge
    const g = geo.clone();
    if (matrix) g.applyMatrix4(matrix);
    normalizeAttributes(g);
    if (!opts?.keepUV) applyBoxUV(g, this.tileFor(key));
    let list = this.buckets.get(key);
    if (!list) {
      list = [];
      this.buckets.set(key, list);
    }
    list.push(g);
  }

  addParts(parts: Array<{ key: string; geo: THREE.BufferGeometry; keepUV?: boolean }>, matrix?: THREE.Matrix4): void {
    for (const p of parts) this.add(p.key, p.geo, matrix, { keepUV: p.keepUV });
  }

  get keys(): string[] {
    return [...this.buckets.keys()];
  }

  /** Number of source geometries in a bucket; useful for perf reporting. */
  countIn(key: string): number {
    return this.buckets.get(key)?.length ?? 0;
  }

  /**
   * Merge every bucket and hand back one mesh per material key. Buckets are
   * cleared afterwards so the batcher can be reused for a second pass.
   */
  build(materialFor: (key: string) => THREE.Material): Array<{ key: string; mesh: THREE.Mesh }> {
    const out: Array<{ key: string; mesh: THREE.Mesh }> = [];
    for (const [key, list] of this.buckets) {
      if (list.length === 0) continue;
      const merged = list.length === 1 ? list[0] : mergeGeometries(list);
      if (!merged) {
        console.warn(`[Geometry] merge failed for bucket "${key}" (${list.length} parts)`);
        continue;
      }
      ensureUV1(merged);
      merged.computeBoundingBox();
      merged.computeBoundingSphere();
      const mesh = new THREE.Mesh(merged, materialFor(key));
      mesh.name = `static_${key}`;
      out.push({ key, mesh });
      // Merged copies are now redundant; free the intermediates.
      if (list.length > 1) for (const g of list) g.dispose();
    }
    this.buckets.clear();
    return out;
  }

  dispose(): void {
    for (const list of this.buckets.values()) for (const g of list) g.dispose();
    this.buckets.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Instancing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Collects transforms for one repeated prop and emits a single InstancedMesh.
 *
 * PERF: 400 sandbags at 40 triangles each is 16k triangles in one draw call.
 * The same sandbags as individual meshes would be 400 draw calls and would
 * single-handedly blow the frame budget on integrated graphics.
 */
export class InstanceBuilder {
  readonly key: string;
  readonly geometry: THREE.BufferGeometry;

  private readonly matrices: THREE.Matrix4[] = [];
  private readonly colors: THREE.Color[] = [];
  private useColor = false;

  constructor(key: string, geometry: THREE.BufferGeometry) {
    this.key = key;
    this.geometry = geometry;
  }

  push(matrix: THREE.Matrix4, color?: THREE.Color): void {
    this.matrices.push(matrix.clone());
    if (color) {
      this.useColor = true;
      this.colors.push(color.clone());
    } else {
      this.colors.push(new THREE.Color(1, 1, 1));
    }
  }

  pushTRS(
    position: THREE.Vector3,
    rotationY: number,
    scale: number | THREE.Vector3 = 1,
    tiltX = 0,
    tiltZ = 0,
    color?: THREE.Color,
  ): void {
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(tiltX, rotationY, tiltZ, 'YXZ'));
    const s = typeof scale === 'number' ? new THREE.Vector3(scale, scale, scale) : scale;
    this.push(new THREE.Matrix4().compose(position, q, s), color);
  }

  get count(): number {
    return this.matrices.length;
  }

  build(material: THREE.Material, castShadow = true, receiveShadow = true): THREE.InstancedMesh | null {
    const n = this.matrices.length;
    if (n === 0) return null;
    const geo = this.geometry;
    normalizeAttributes(geo);
    applyBoxUV(geo, 1.4);
    ensureUV1(geo);
    const mesh = new THREE.InstancedMesh(geo, material, n);
    for (let i = 0; i < n; i++) {
      mesh.setMatrixAt(i, this.matrices[i]);
      if (this.useColor) mesh.setColorAt(i, this.colors[i]);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = castShadow;
    mesh.receiveShadow = receiveShadow;
    mesh.name = `inst_${this.key}`;
    // An InstancedMesh without an explicit bounding volume falls back to the
    // *source geometry's* sphere, which sits at the origin — the whole batch
    // would pop out of view the moment the origin left the frustum.
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    mesh.frustumCulled = true;
    return mesh;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small transform conveniences
// ─────────────────────────────────────────────────────────────────────────────

/** Compose a TRS matrix. Saves a great deal of noise in the level script. */
export function trs(
  x: number,
  y: number,
  z: number,
  ry = 0,
  scale: number | THREE.Vector3 = 1,
  rx = 0,
  rz = 0,
): THREE.Matrix4 {
  const s = typeof scale === 'number' ? new THREE.Vector3(scale, scale, scale) : scale;
  return new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'YXZ')),
    s,
  );
}

/** Count triangles in a geometry, for the perf report. */
export function triCount(geo: THREE.BufferGeometry): number {
  const idx = geo.index;
  if (idx) return idx.count / 3;
  const pos = geo.getAttribute('position');
  return pos ? pos.count / 3 : 0;
}
