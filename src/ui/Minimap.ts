/**
 * NOISEFIELD — Tactical radar / minimap.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT IT DRAWS AND WHY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * FOOTPRINT. A real top-down schematic of the level, baked ONCE at init by
 * walking the collidable geometry and rasterising the *vertical* triangles —
 * i.e. the walls — into an occupancy grid. Filtering on triangle normal is the
 * whole trick: naively projecting every triangle fills the map solid because
 * the ground plane and every roof project to a filled rectangle. Keeping only
 * faces with |normal.y| below a threshold leaves precisely the wall lines,
 * which is exactly what an architectural floor plan is.
 *
 * The obvious cheaper alternative — one AABB per collidable — does not work
 * here: this level merges all its static geometry into ~14 material buckets,
 * so "the concrete bucket" has a single AABB covering the entire 90m map.
 *
 * ROTATING MAP. The genre default, and the right default: a rotating map means
 * "up on the minimap" is always "forward in the world", so reading it costs no
 * mental rotation. North-up is better for memorising a layout, which matters
 * in a strategy game and does not matter at all when someone is shooting you.
 *
 * ENEMY DOTS ONLY ON GUNFIRE. Permanent enemy dots are wallhacks and they
 * delete the tension from every corner in the level. The usual rule — you appear
 * on radar when you fire unsuppressed — turns the radar into a *conversation*:
 * it rewards listening, it makes silence meaningful, and it gives the player
 * information they must still act on rather than an answer. Fire pings decay
 * over a couple of seconds so the radar shows "where they were", never "where
 * they are", which keeps flanking viable.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * COST
 * ═══════════════════════════════════════════════════════════════════════════
 * Per frame: one clip, one gradient fill, one drawImage of the pre-baked
 * footprint under a rotation, a handful of rings, and up to 32 dots. The
 * expensive part (geometry traversal) happens exactly once, is capped, and is
 * wrapped so that a malformed or absent level degrades to an empty grid rather
 * than throwing during boot.
 */

import * as THREE from 'three';
import type { EngineContext, System } from '../core/Types';

/**
 * Structural view of the level system. Declared locally and entirely optional
 * because `Types.ts` has no entry for it —
 * every field is probed at runtime before use.
 */
interface LevelLike extends System {
  readonly bounds?: THREE.Box3;
  readonly collidables?: THREE.Object3D[];
  readonly root?: THREE.Object3D;
}

export type PingKind = 'fire' | 'threat' | 'boom';

interface Ping {
  x: number;
  z: number;
  /** Age in seconds. */
  t: number;
  life: number;
  kind: PingKind;
}

/** Occupancy grid resolution. 256 over a ~92m map ≈ 0.36m per cell — finer
 *  than the thinnest wall in the level, so nothing thin disappears. */
const GRID = 256;
/** Baked footprint texture size. 1024 is ~2.8x the on-screen pixel budget at
 *  4K, which keeps the rotated blit free of stair-stepping. */
const BAKE = 1024;
/** Hard ceiling on init-time geometry work, so a pathological level cannot
 *  add seconds to the loading screen. */
const MAX_TRIANGLES = 240_000;
const MAX_INSTANCES = 4_000;
/** |normal.y| above this is a floor or a ceiling, not a wall. */
const WALL_NORMAL_Y = 0.62;

/** Metres visible across the radar. ~46m covers a three-system arena's local
 *  engagement space without shrinking the player's own surroundings to mush. */
const VIEW_METRES = 46;

const PING_LIFE: Readonly<Record<PingKind, number>> = {
  // Short: the radar must report "shots came from there a moment ago", not
  // track a live target. Two and a half seconds is enough to turn and commit.
  fire: 2.6,
  // Being hit is higher-value information, so it lingers a little longer.
  threat: 3.4,
  boom: 4.0,
};

const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

export class Minimap {
  readonly el: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D | null;

  /** Baked level footprint; null until (and unless) a level was found. */
  private baked: HTMLCanvasElement | null = null;

  private readonly min = new THREE.Vector3(-50, 0, -50);
  private readonly max = new THREE.Vector3(50, 20, 50);
  private spanX = 100;
  private spanZ = 100;

  private size = 180;
  private dpr = 1;
  private s = 1;

  private px = 0;
  private pz = 0;
  private bearing = 0;
  private fovHalf = 0.6;

  private readonly pings: Ping[] = [];
  private hidden = false;

  constructor() {
    this.el = document.createElement('canvas');
    this.el.className = 'nf-minimap__canvas';
    this.el.setAttribute('aria-hidden', 'true');
    this.g = this.el.getContext('2d', { alpha: true });
  }

  mount(parent: HTMLElement): void {
    parent.appendChild(this.el);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Bake
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Build the footprint. Safe to call with no level present.
   *
   * Every access into the level is guarded: this runs during boot, systems
   * initialise in an order this module does not control, and a HUD must never
   * be the reason the game fails to start.
   */
  build(ctx: EngineContext): void {
    let level: LevelLike | undefined;
    try {
      level = ctx.get<LevelLike>('level');
    } catch {
      level = undefined;
    }

    const b = level?.bounds;
    const finite = (v: number): boolean => Number.isFinite(v);
    if (
      b &&
      finite(b.min.x) && finite(b.max.x) && finite(b.min.z) && finite(b.max.z) &&
      finite(b.min.y) && finite(b.max.y) &&
      b.max.x > b.min.x && b.max.z > b.min.z
    ) {
      this.min.copy(b.min);
      this.max.copy(b.max);
    }
    this.spanX = Math.max(1e-3, this.max.x - this.min.x);
    this.spanZ = Math.max(1e-3, this.max.z - this.min.z);

    const sources = level?.collidables;
    if (!Array.isArray(sources) || sources.length === 0) return;

    const grid = new Uint8Array(GRID * GRID);
    let budget = MAX_TRIANGLES;
    let instBudget = MAX_INSTANCES;

    for (const obj of sources) {
      if (budget <= 0) break;
      try {
        if ((obj as THREE.InstancedMesh).isInstancedMesh) {
          instBudget -= this.stampInstanced(obj as THREE.InstancedMesh, grid, instBudget);
        } else if ((obj as THREE.Mesh).isMesh) {
          budget -= this.stampMesh(obj as THREE.Mesh, grid, budget);
        }
      } catch {
        // A single malformed collidable must not abort the whole bake.
        continue;
      }
    }

    this.baked = this.rasterise(grid);
  }

  /**
   * Props are instanced and usually box-like, so per-instance AABB outlines are
   * both far cheaper than triangle walking and a better read: a crate should
   * appear on the radar as a small square, not as a scribble of edges.
   */
  private stampInstanced(mesh: THREE.InstancedMesh, grid: Uint8Array, budget: number): number {
    const geo = mesh.geometry;
    if (!geo) return 0;
    if (!geo.boundingBox) geo.computeBoundingBox();
    const bb = geo.boundingBox;
    if (!bb) return 0;

    mesh.updateWorldMatrix(true, false);
    const m = new THREE.Matrix4();
    const world = new THREE.Matrix4();
    const box = new THREE.Box3();
    const count = Math.min(mesh.count, Math.max(0, budget));

    for (let i = 0; i < count; i++) {
      mesh.getMatrixAt(i, m);
      world.multiplyMatrices(mesh.matrixWorld, m);
      box.copy(bb).applyMatrix4(world);
      // Ignore ground scatter (pebbles, kerbs, decals-as-geometry): it is not
      // cover and drawing it turns the radar into noise.
      //
      // Filter on the object's own VERTICAL EXTENT, not on its height above a
      // ground plane. `bounds` is padded by the level (min.y is -1, not 0), so
      // the original "max.y - bounds.min.y < 0.45" test could only ever be true
      // for geometry below y = -0.55 and therefore rejected nothing at all.
      // Extent needs no ground reference and cannot be broken by padding.
      if (box.max.y - box.min.y < 0.45) continue;
      this.stampRect(grid, box.min.x, box.min.z, box.max.x, box.max.z);
    }
    return count;
  }

  /** Walk triangles, keep the vertical ones, rasterise their XZ edges. */
  private stampMesh(mesh: THREE.Mesh, grid: Uint8Array, budget: number): number {
    const geo = mesh.geometry;
    const posAttr = geo?.getAttribute('position');
    if (!posAttr) return 0;

    mesh.updateWorldMatrix(true, false);
    const mw = mesh.matrixWorld;
    const index = geo.index;
    const triCount = Math.floor((index ? index.count : posAttr.count) / 3);
    if (triCount <= 0) return 0;

    // If a single mesh would blow the budget, sample it uniformly rather than
    // truncating — truncation would draw half a building and stop.
    const allowed = Math.max(1, Math.min(triCount, budget));
    const step = Math.max(1, Math.ceil(triCount / allowed));

    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    const n = new THREE.Vector3();

    let processed = 0;
    for (let t = 0; t < triCount; t += step) {
      const i0 = t * 3;
      const ia = index ? index.getX(i0) : i0;
      const ib = index ? index.getX(i0 + 1) : i0 + 1;
      const ic = index ? index.getX(i0 + 2) : i0 + 2;

      a.fromBufferAttribute(posAttr, ia).applyMatrix4(mw);
      b.fromBufferAttribute(posAttr, ib).applyMatrix4(mw);
      c.fromBufferAttribute(posAttr, ic).applyMatrix4(mw);
      processed++;

      // Skip anything too short to be cover — kerbs, thresholds, paint edges.
      // Measured as the triangle's own vertical extent rather than its height
      // above `bounds.min.y`, which is padded and made the original test inert.
      if (Math.max(a.y, b.y, c.y) - Math.min(a.y, b.y, c.y) < 0.4) continue;

      e1.subVectors(b, a);
      e2.subVectors(c, a);
      n.crossVectors(e1, e2);
      const len = n.length();
      if (len < 1e-9) continue;
      if (Math.abs(n.y) / len > WALL_NORMAL_Y) continue; // floor / roof

      this.stampLine(grid, a.x, a.z, b.x, b.z);
      this.stampLine(grid, b.x, b.z, c.x, c.z);
      this.stampLine(grid, c.x, c.z, a.x, a.z);
    }
    return processed;
  }

  private stampRect(grid: Uint8Array, x0: number, z0: number, x1: number, z1: number): void {
    this.stampLine(grid, x0, z0, x1, z0);
    this.stampLine(grid, x1, z0, x1, z1);
    this.stampLine(grid, x1, z1, x0, z1);
    this.stampLine(grid, x0, z1, x0, z0);
  }

  /** Bresenham into the occupancy grid, in world XZ. */
  private stampLine(grid: Uint8Array, wx0: number, wz0: number, wx1: number, wz1: number): void {
    // Scale by GRID, not GRID-1: the rasteriser below writes CELLS, and the
    // texture pass maps cell i to [i*BAKE/GRID, (i+1)*BAKE/GRID). Using GRID-1
    // here stretches the grid by one cell against the texture and puts every
    // wall about a pixel out of true.
    const gx = (w: number): number =>
      Math.floor(((w - this.min.x) / this.spanX) * GRID);
    const gz = (w: number): number =>
      Math.floor(((w - this.min.z) / this.spanZ) * GRID);

    let x0 = gx(wx0);
    let y0 = gz(wz0);
    const x1 = gx(wx1);
    const y1 = gz(wz1);

    const dx = Math.abs(x1 - x0);
    const dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;

    // Guard: a NaN vertex would otherwise spin here forever.
    if (!Number.isFinite(x0) || !Number.isFinite(y0) || !Number.isFinite(x1) || !Number.isFinite(y1)) return;

    let guard = GRID * 3;
    for (;;) {
      if (x0 >= 0 && x0 < GRID && y0 >= 0 && y0 < GRID) grid[y0 * GRID + x0] = 1;
      if (x0 === x1 && y0 === y1) break;
      if (--guard < 0) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  /** Turn the binary grid into a drawable texture. */
  private rasterise(grid: Uint8Array): HTMLCanvasElement {
    const cv = document.createElement('canvas');
    cv.width = BAKE;
    cv.height = BAKE;
    const g = cv.getContext('2d');
    if (!g) return cv;

    const cell = BAKE / GRID;
    // Two passes: a wide soft "glow" pass gives the walls a subtle bleed that
    // stops the map looking like a 1-bit fax, then a crisp pass on top.
    g.fillStyle = 'rgba(150, 205, 250, 0.14)';
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        if (grid[y * GRID + x] === 1) g.fillRect(x * cell - cell * 0.6, y * cell - cell * 0.6, cell * 2.2, cell * 2.2);
      }
    }
    g.fillStyle = 'rgba(196, 228, 252, 0.86)';
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        if (grid[y * GRID + x] === 1) g.fillRect(x * cell, y * cell, cell * 1.05, cell * 1.05);
      }
    }
    return cv;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Inputs
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * @param bearing Radians, 0 = facing world north (-Z), increasing clockwise
   *   toward east (+X). Matches the compass tape so the two never disagree.
   * @param hFovDeg HORIZONTAL field of view. The cone is a top-down sweep, so
   *   it must be the horizontal angle; three.js `camera.fov` is the VERTICAL
   *   one and feeding it straight in drew a cone ~30% narrower than what is
   *   actually on screen, which makes the radar lie about what the player can
   *   see. The caller does the aspect conversion.
   */
  setPlayer(x: number, z: number, bearing: number, hFovDeg: number): void {
    this.px = x;
    this.pz = z;
    this.bearing = bearing;
    this.fovHalf = THREE.MathUtils.degToRad(clamp(hFovDeg, 40, 160)) * 0.5;
  }

  /**
   * Register a contact. Nearby pings of the same kind are merged rather than
   * stacked, otherwise a single 30-round burst posts thirty dots on one spot
   * and the alpha piles up into an opaque blob.
   */
  ping(x: number, z: number, kind: PingKind): void {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return;
    const mergeSq = kind === 'boom' ? 16 : 6.25; // 4m / 2.5m
    for (const p of this.pings) {
      if (p.kind !== kind) continue;
      const dx = p.x - x;
      const dz = p.z - z;
      if (dx * dx + dz * dz < mergeSq) {
        p.x = x;
        p.z = z;
        p.t = 0;
        return;
      }
    }
    if (this.pings.length >= 32) this.pings.shift();
    this.pings.push({ x, z, t: 0, life: PING_LIFE[kind], kind });
  }

  setHidden(v: boolean): void {
    this.hidden = v;
    this.el.style.opacity = v ? '0' : '1';
  }

  clearPings(): void {
    this.pings.length = 0;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Layout + frame
  // ═══════════════════════════════════════════════════════════════════════════

  resize(vw: number, vh: number): void {
    const short = Math.min(vw, vh);
    this.s = clamp(short / 1080, 0.85, 2.1);
    const size = Math.round(clamp(short * 0.175, 128, 300));
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    this.size = size;
    this.dpr = dpr;

    // Compare against the ELEMENT. A cached field pre-set to a plausible value
    // looks identical to one that has actually been applied, so a first resize
    // that computed exactly that value skipped the work and left the canvas at
    // the 300x150 HTML default.
    const px = Math.round(size * dpr);
    if (this.el.width !== px || this.el.height !== px) {
      this.el.width = px;
      this.el.height = px;
    }
    this.el.style.width = `${size}px`;
    this.el.style.height = `${size}px`;
  }

  /** @returns the on-screen edge length in CSS px, so the HUD can stack under it. */
  get cssSize(): number {
    return this.size;
  }

  update(dt: number): void {
    const g = this.g;
    if (g === null) return;

    for (let i = this.pings.length - 1; i >= 0; i--) {
      const p = this.pings[i];
      p.t += dt;
      if (p.t >= p.life) this.pings.splice(i, 1);
    }

    if (this.hidden) return;

    const size = this.size;
    const R = size * 0.5;
    const ppm = size / VIEW_METRES;

    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, size * this.dpr, size * this.dpr);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const radius = Math.max(4, 10 * this.s);

    // ── Body ────────────────────────────────────────────────────────────────
    g.save();
    this.roundRect(g, 0, 0, size, size, radius);
    g.clip();

    const bg = g.createLinearGradient(0, 0, 0, size);
    bg.addColorStop(0, 'rgba(10, 18, 26, 0.72)');
    bg.addColorStop(1, 'rgba(5, 10, 15, 0.82)');
    g.fillStyle = bg;
    g.fillRect(0, 0, size, size);

    // ── Rotating world layer ────────────────────────────────────────────────
    g.save();
    g.translate(R, R);
    // -bearing puts the player's facing at the top of the radar. See the
    // class comment for why the map rotates rather than the arrow.
    g.rotate(-this.bearing);
    g.scale(ppm, ppm);
    g.translate(-this.px, -this.pz);

    // Reference grid, in world metres so it conveys real scale (10m squares).
    g.lineWidth = 1 / ppm;
    g.strokeStyle = 'rgba(140, 190, 235, 0.075)';
    g.beginPath();
    const g0x = Math.floor((this.px - VIEW_METRES) / 10) * 10;
    const g1x = this.px + VIEW_METRES;
    for (let x = g0x; x <= g1x; x += 10) {
      g.moveTo(x, this.pz - VIEW_METRES);
      g.lineTo(x, this.pz + VIEW_METRES);
    }
    const g0z = Math.floor((this.pz - VIEW_METRES) / 10) * 10;
    const g1z = this.pz + VIEW_METRES;
    for (let z = g0z; z <= g1z; z += 10) {
      g.moveTo(this.px - VIEW_METRES, z);
      g.lineTo(this.px + VIEW_METRES, z);
    }
    g.stroke();

    if (this.baked !== null) {
      g.imageSmoothingEnabled = true;
      // 'medium', not 'high'. This is a 1024x1024 source minified ~4:1 every
      // frame; 'high' buys a quality difference that is invisible on a
      // two-tone schematic and costs real time on the compositor.
      g.imageSmoothingQuality = 'medium';
      g.drawImage(this.baked, this.min.x, this.min.z, this.spanX, this.spanZ);
    }

    // Playable boundary — tells the player where the map physically ends.
    g.lineWidth = 1.6 / ppm;
    g.strokeStyle = 'rgba(160, 205, 245, 0.22)';
    g.strokeRect(this.min.x, this.min.z, this.spanX, this.spanZ);
    g.restore();

    // ── Screen-space overlays ───────────────────────────────────────────────
    this.drawRings(g, R);
    this.drawPings(g, R, ppm);
    this.drawCardinals(g, R);
    this.drawPlayer(g, R);

    g.restore();

    // ── Frame ───────────────────────────────────────────────────────────────
    g.lineWidth = Math.max(1, 1.4 * this.s);
    g.strokeStyle = 'rgba(207, 232, 255, 0.34)';
    this.roundRect(g, 0.7, 0.7, size - 1.4, size - 1.4, radius);
    g.stroke();
    // Inner dark keyline: the radar must read as a distinct object even when
    // the scene behind the safe margin is bright sand.
    g.lineWidth = Math.max(1, 1.2 * this.s);
    g.strokeStyle = 'rgba(0, 0, 0, 0.55)';
    this.roundRect(g, 2, 2, size - 4, size - 4, Math.max(2, radius - 1.4));
    g.stroke();
  }

  private drawRings(g: CanvasRenderingContext2D, R: number): void {
    g.strokeStyle = 'rgba(160, 205, 245, 0.13)';
    g.lineWidth = 1;
    for (const f of [0.34, 0.67]) {
      g.beginPath();
      g.arc(R, R, R * f, 0, Math.PI * 2);
      g.stroke();
    }
  }

  private drawCardinals(g: CanvasRenderingContext2D, R: number): void {
    const dirs: Array<[string, number, number]> = [
      ['N', 0, -1],
      ['E', 1, 0],
      ['S', 0, 1],
      ['W', -1, 0],
    ];
    const cos = Math.cos(-this.bearing);
    const sin = Math.sin(-this.bearing);
    const rr = R - Math.max(7, 9 * this.s);

    g.font = `700 ${Math.round(Math.max(8, 9.5 * this.s))}px 'Bahnschrift','DIN Alternate','Segoe UI',system-ui,sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';

    for (const [label, dx, dz] of dirs) {
      const x = R + (dx * cos - dz * sin) * rr;
      const y = R + (dx * sin + dz * cos) * rr;
      const isNorth = label === 'N';
      // North is the only labelled bearing that gets full contrast: it is the
      // one the player uses to relate the radar to the compass tape.
      g.lineWidth = 3;
      g.strokeStyle = 'rgba(0, 0, 0, 0.8)';
      g.strokeText(label, x, y);
      g.fillStyle = isNorth ? 'rgba(255, 180, 84, 0.95)' : 'rgba(207, 232, 255, 0.5)';
      g.fillText(label, x, y);
    }
  }

  private drawPings(g: CanvasRenderingContext2D, R: number, ppm: number): void {
    const cos = Math.cos(-this.bearing);
    const sin = Math.sin(-this.bearing);
    const rim = R - Math.max(8, 10 * this.s);

    for (const p of this.pings) {
      const u = p.t / p.life;
      // Hold full opacity for the first 55%, then fade. A ping that starts
      // fading immediately reads as unreliable; the player needs a beat of
      // certainty before it decays.
      const a = u < 0.55 ? 1 : 1 - (u - 0.55) / 0.45;
      if (a <= 0.01) continue;

      const dx = (p.x - this.px) * ppm;
      const dz = (p.z - this.pz) * ppm;
      let sx = dx * cos - dz * sin;
      let sy = dx * sin + dz * cos;

      const d = Math.hypot(sx, sy);
      const offRadar = d > rim;
      if (offRadar && d > 1e-4) {
        sx = (sx / d) * rim;
        sy = (sy / d) * rim;
      }

      const x = R + sx;
      const y = R + sy;
      const col = p.kind === 'boom' ? '255, 180, 84' : '255, 68, 54';
      const base = p.kind === 'boom' ? 4.2 : 3.4;

      // An expanding sonar ring on the freshest pings draws the eye to *new*
      // information; without it a static dot appearing at the edge of vision
      // is routinely missed.
      if (u < 0.45) {
        const e = u / 0.45;
        g.globalAlpha = (1 - e) * 0.5 * a;
        g.strokeStyle = `rgba(${col}, 1)`;
        g.lineWidth = Math.max(1, 1.4 * this.s);
        g.beginPath();
        g.arc(x, y, (base + e * 11) * this.s, 0, Math.PI * 2);
        g.stroke();
      }

      g.globalAlpha = a;
      if (offRadar) {
        // Off-radar contacts become a chevron on the rim pointing outward, so
        // direction survives even when range does not.
        const ang = Math.atan2(sy, sx);
        g.save();
        g.translate(x, y);
        g.rotate(ang);
        g.beginPath();
        g.moveTo(4.4 * this.s, 0);
        g.lineTo(-2.6 * this.s, 3.4 * this.s);
        g.lineTo(-2.6 * this.s, -3.4 * this.s);
        g.closePath();
        g.fillStyle = `rgba(${col}, 0.95)`;
        g.strokeStyle = 'rgba(0, 0, 0, 0.75)';
        g.lineWidth = 1.2;
        g.fill();
        g.stroke();
        g.restore();
      } else {
        g.beginPath();
        g.arc(x, y, base * this.s, 0, Math.PI * 2);
        g.fillStyle = `rgba(${col}, 0.98)`;
        g.fill();
        g.lineWidth = Math.max(1, 1.2 * this.s);
        g.strokeStyle = 'rgba(0, 0, 0, 0.7)';
        g.stroke();
      }
    }
    g.globalAlpha = 1;
  }

  private drawPlayer(g: CanvasRenderingContext2D, R: number): void {
    const s = this.s;

    // View cone. Communicates *what the player can see*, which is the single
    // most useful thing a radar can add to a first-person view.
    const r = R * 0.62;
    const grad = g.createRadialGradient(R, R, 0, R, R, r);
    grad.addColorStop(0, 'rgba(207, 232, 255, 0.2)');
    grad.addColorStop(1, 'rgba(207, 232, 255, 0)');
    g.fillStyle = grad;
    g.beginPath();
    g.moveTo(R, R);
    // -PI/2 is straight up on the canvas, which is "forward" on a rotating map.
    g.arc(R, R, r, -Math.PI / 2 - this.fovHalf, -Math.PI / 2 + this.fovHalf);
    g.closePath();
    g.fill();

    // Chevron, always centred and always pointing up.
    g.beginPath();
    g.moveTo(R, R - 6.4 * s);
    g.lineTo(R + 4.8 * s, R + 5.2 * s);
    g.lineTo(R, R + 2.6 * s);
    g.lineTo(R - 4.8 * s, R + 5.2 * s);
    g.closePath();
    g.fillStyle = 'rgba(242, 245, 247, 0.98)';
    g.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    g.lineWidth = Math.max(1, 1.5 * s);
    g.stroke();
    g.fill();
  }

  private roundRect(
    g: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    r: number,
  ): void {
    const rr = Math.min(r, w * 0.5, h * 0.5);
    g.beginPath();
    g.moveTo(x + rr, y);
    g.lineTo(x + w - rr, y);
    g.arcTo(x + w, y, x + w, y + rr, rr);
    g.lineTo(x + w, y + h - rr);
    g.arcTo(x + w, y + h, x + w - rr, y + h, rr);
    g.lineTo(x + rr, y + h);
    g.arcTo(x, y + h, x, y + h - rr, rr);
    g.lineTo(x, y + rr);
    g.arcTo(x, y, x + rr, y, rr);
    g.closePath();
  }

  dispose(): void {
    this.el.remove();
    this.el.width = 0;
    this.el.height = 0;
    if (this.baked) {
      this.baked.width = 0;
      this.baked.height = 0;
      this.baked = null;
    }
    this.pings.length = 0;
  }
}
