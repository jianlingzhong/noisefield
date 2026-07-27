/**
 * NOISEFIELD — Pooled decal system.
 *
 * Two responsibilities, one implementation:
 *
 *  1. DYNAMIC decals — bullet holes, scorch marks, blood — sprayed at runtime
 *     by the ballistics system through `IDecals.add()`. Budgeted and recycled.
 *  2. STATIC grunge — posters, graffiti, oil stains, tyre marks — placed once
 *     at level build time through `addGrunge()`. Never recycled.
 *
 * ARCHITECTURE. One `InstancedMesh` per decal *family*, not per decal. All
 * concrete bullet holes in the map are a single draw call of ~40-byte
 * instances; adding a decal is two matrix writes and a partial buffer upload,
 * not an allocation. Eleven families total, so the whole system costs eleven
 * draw calls no matter how much lead is in the air.
 *
 * ORIENTATION. Each quad is rotated from +Z onto the surface normal with a
 * random roll (otherwise every hole on a wall shares an axis and the repetition
 * is instantly visible), pushed 1.2 cm along the normal, and drawn with
 * `polygonOffset` and `depthWrite: false`. Belt and braces: the offset handles
 * co-planar z-fighting, the normal push handles curved and slightly-off
 * surfaces, and no depth writes means overlapping decals never punch holes in
 * each other.
 *
 * FADE. Alpha per instance comes from a custom instanced attribute injected
 * into the standard material with a two-line shader patch. Three has no
 * built-in per-instance opacity (instanceColor multiplies RGB only), and
 * scaling a decal to zero to "fade" it looks exactly like what it is.
 *
 * TEXTURES. All procedural, drawn to canvases at load. Each surface gets a
 * physically-motivated profile: concrete spalls grey and cracks, metal tears a
 * bright lip, wood throws splinters, glass radiates cracks, sand craters
 * without a hole at all.
 */

import * as THREE from 'three';
import type { EngineContext, FrameContext, IDecals, QualitySettings, SurfaceType, System } from '../core/Types';
import { Rng } from './Geometry';

// ─────────────────────────────────────────────────────────────────────────────
// Families
// ─────────────────────────────────────────────────────────────────────────────

/** Impact families. One pooled InstancedMesh each. */
export type DecalFamily = 'concrete' | 'metal' | 'wood' | 'sand' | 'glass' | 'blood' | 'scorch';

/** Pre-placed environment storytelling families. */
export type GrungeKind = 'poster' | 'graffiti' | 'stain' | 'tyremark' | 'bulletscar';

const IMPACT_FAMILIES: readonly DecalFamily[] = ['concrete', 'metal', 'wood', 'sand', 'glass', 'blood', 'scorch'];
const GRUNGE_KINDS: readonly GrungeKind[] = ['poster', 'graffiti', 'stain', 'tyremark', 'bulletscar'];

/**
 * Surface classification to decal family. `water` deliberately has no entry —
 * a bullet hole floating on water is a bug, not a feature.
 */
const SURFACE_TO_FAMILY: Readonly<Partial<Record<SurfaceType, DecalFamily>>> = {
  concrete: 'concrete',
  dirt: 'sand',
  sand: 'sand',
  metal: 'metal',
  rubber: 'metal',
  wood: 'wood',
  fabric: 'wood',
  foliage: 'wood',
  glass: 'glass',
  flesh: 'blood',
};

/** Aspect ratio (width / height) per grunge kind. Tyre marks are long. */
const GRUNGE_ASPECT: Readonly<Record<GrungeKind, number>> = {
  poster: 0.72,
  graffiti: 1.6,
  stain: 1.0,
  tyremark: 3.4,
  bulletscar: 1.35,
};

interface PoolEntry {
  family: DecalFamily;
  slot: number;
  /** Seconds remaining before the decal begins its natural fade. */
  life: number;
  /** Seconds the fade-out takes once `life` hits zero. */
  fadeTime: number;
  dying: boolean;
}

interface Pool {
  mesh: THREE.InstancedMesh;
  fade: THREE.InstancedBufferAttribute;
  /** Slots below `high` that are free for reuse. */
  free: number[];
  high: number;
  capacity: number;
  dirtyMatrix: boolean;
  dirtyFade: boolean;
}

interface PendingGrunge {
  position: THREE.Vector3;
  normal: THREE.Vector3;
  kind: GrungeKind;
  size: number;
  roll: number;
  opacity: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Procedural textures
// ─────────────────────────────────────────────────────────────────────────────

function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; c2d: CanvasRenderingContext2D } | null {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const c2d = canvas.getContext('2d');
  if (!c2d) return null;
  return { canvas, c2d };
}

/** A wobbly closed blob — the shape of almost every real damage feature. */
function blobPath(c2d: CanvasRenderingContext2D, cx: number, cy: number, r: number, wobble: number, rng: Rng): void {
  const n = 14;
  c2d.beginPath();
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2;
    const rr = r * (1 - wobble * 0.5 + rng.next() * wobble);
    const x = cx + Math.cos(a) * rr;
    const y = cy + Math.sin(a) * rr;
    if (i === 0) c2d.moveTo(x, y);
    else c2d.lineTo(x, y);
  }
  c2d.closePath();
}

function radialHalo(c2d: CanvasRenderingContext2D, cx: number, cy: number, r: number, color: string, alpha: number): void {
  const g = c2d.createRadialGradient(cx, cy, 0, cx, cy, r);
  g.addColorStop(0, `rgba(${color},${alpha})`);
  g.addColorStop(0.45, `rgba(${color},${alpha * 0.42})`);
  g.addColorStop(1, `rgba(${color},0)`);
  c2d.fillStyle = g;
  c2d.beginPath();
  c2d.arc(cx, cy, r, 0, Math.PI * 2);
  c2d.fill();
}

/**
 * Paint one impact decal. `S` is the canvas edge in pixels; everything below is
 * expressed as a fraction of it so the profiles are resolution independent.
 */
function paintImpact(c2d: CanvasRenderingContext2D, S: number, family: DecalFamily, rng: Rng): void {
  const C = S * 0.5;
  c2d.clearRect(0, 0, S, S);
  c2d.lineCap = 'round';

  switch (family) {
    case 'concrete': {
      // Dust bloom, grey spall crater, radiating hairline cracks, dark core.
      radialHalo(c2d, C, C, S * 0.48, '196,186,166', 0.30);
      c2d.fillStyle = 'rgba(186,180,166,0.80)';
      blobPath(c2d, C, C, S * 0.24, 0.55, rng);
      c2d.fill();
      c2d.fillStyle = 'rgba(150,144,131,0.85)';
      blobPath(c2d, C, C, S * 0.16, 0.5, rng);
      c2d.fill();
      c2d.strokeStyle = 'rgba(58,52,45,0.72)';
      for (let i = 0; i < 9; i++) {
        const a = rng.range(0, Math.PI * 2);
        const len = S * rng.range(0.16, 0.44);
        c2d.lineWidth = rng.range(0.8, 2.2);
        c2d.beginPath();
        c2d.moveTo(C, C);
        c2d.lineTo(C + Math.cos(a) * len * 0.55 + rng.range(-4, 4), C + Math.sin(a) * len * 0.55 + rng.range(-4, 4));
        c2d.lineTo(C + Math.cos(a) * len, C + Math.sin(a) * len);
        c2d.stroke();
      }
      c2d.fillStyle = 'rgba(24,21,18,0.96)';
      blobPath(c2d, C, C, S * 0.085, 0.45, rng);
      c2d.fill();
      break;
    }

    case 'metal': {
      // Small punched hole with a bright torn lip and scoured radial streaks.
      radialHalo(c2d, C, C, S * 0.42, '30,28,26', 0.24);
      c2d.strokeStyle = 'rgba(214,210,200,0.5)';
      for (let i = 0; i < 14; i++) {
        const a = rng.range(0, Math.PI * 2);
        c2d.lineWidth = rng.range(0.6, 1.6);
        c2d.beginPath();
        c2d.moveTo(C + Math.cos(a) * S * 0.09, C + Math.sin(a) * S * 0.09);
        c2d.lineTo(C + Math.cos(a) * S * rng.range(0.18, 0.4), C + Math.sin(a) * S * rng.range(0.18, 0.4));
        c2d.stroke();
      }
      // The lip: petals of torn steel, brighter than the plate around them.
      c2d.fillStyle = 'rgba(232,228,216,0.92)';
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2 + rng.range(-0.2, 0.2);
        const r0 = S * 0.075;
        const r1 = S * rng.range(0.12, 0.17);
        c2d.beginPath();
        c2d.moveTo(C + Math.cos(a - 0.28) * r0, C + Math.sin(a - 0.28) * r0);
        c2d.lineTo(C + Math.cos(a) * r1, C + Math.sin(a) * r1);
        c2d.lineTo(C + Math.cos(a + 0.28) * r0, C + Math.sin(a + 0.28) * r0);
        c2d.closePath();
        c2d.fill();
      }
      c2d.fillStyle = 'rgba(12,11,10,0.99)';
      blobPath(c2d, C, C, S * 0.07, 0.3, rng);
      c2d.fill();
      break;
    }

    case 'wood': {
      // Splinters: long pale slivers lifted out of a dark torn socket.
      radialHalo(c2d, C, C, S * 0.4, '96,68,42', 0.34);
      c2d.fillStyle = 'rgba(58,38,22,0.86)';
      blobPath(c2d, C, C, S * 0.15, 0.6, rng);
      c2d.fill();
      for (let i = 0; i < 16; i++) {
        const a = rng.range(0, Math.PI * 2);
        const r1 = S * rng.range(0.13, 0.40);
        const wdt = rng.range(0.05, 0.14);
        c2d.fillStyle = `rgba(${190 + rng.int(0, 30)},${150 + rng.int(0, 26)},${96 + rng.int(0, 24)},${rng.range(0.5, 0.9).toFixed(2)})`;
        c2d.beginPath();
        c2d.moveTo(C + Math.cos(a - wdt) * S * 0.06, C + Math.sin(a - wdt) * S * 0.06);
        c2d.lineTo(C + Math.cos(a) * r1, C + Math.sin(a) * r1);
        c2d.lineTo(C + Math.cos(a + wdt) * S * 0.06, C + Math.sin(a + wdt) * S * 0.06);
        c2d.closePath();
        c2d.fill();
      }
      c2d.fillStyle = 'rgba(18,12,7,0.95)';
      blobPath(c2d, C, C, S * 0.075, 0.4, rng);
      c2d.fill();
      break;
    }

    case 'sand': {
      // No hole at all: a shallow crater with a bright thrown rim and a dust
      // plume. Sand absorbs rounds, it does not spall.
      radialHalo(c2d, C, C, S * 0.5, '214,196,160', 0.42);
      c2d.fillStyle = 'rgba(232,216,182,0.55)';
      blobPath(c2d, C, C, S * 0.27, 0.4, rng);
      c2d.fill();
      c2d.fillStyle = 'rgba(120,102,74,0.5)';
      blobPath(c2d, C, C, S * 0.15, 0.45, rng);
      c2d.fill();
      c2d.fillStyle = 'rgba(88,74,54,0.45)';
      blobPath(c2d, C, C, S * 0.075, 0.5, rng);
      c2d.fill();
      // A few grains thrown clear of the crater.
      for (let i = 0; i < 30; i++) {
        const a = rng.range(0, Math.PI * 2);
        const r = S * rng.range(0.25, 0.47);
        c2d.fillStyle = `rgba(206,188,152,${rng.range(0.15, 0.5).toFixed(2)})`;
        c2d.beginPath();
        c2d.arc(C + Math.cos(a) * r, C + Math.sin(a) * r, rng.range(0.7, 2.4), 0, Math.PI * 2);
        c2d.fill();
      }
      break;
    }

    case 'glass': {
      // Radial fractures tied together by concentric rings; a white punch at
      // the centre where the pane pulverised.
      const spokes = 11;
      const angles: number[] = [];
      for (let i = 0; i < spokes; i++) angles.push((i / spokes) * Math.PI * 2 + rng.range(-0.16, 0.16));
      c2d.strokeStyle = 'rgba(238,246,250,0.78)';
      for (const a of angles) {
        c2d.lineWidth = rng.range(0.7, 1.9);
        c2d.beginPath();
        c2d.moveTo(C, C);
        let r = 0;
        let x = C;
        let y = C;
        while (r < S * 0.46) {
          r += S * rng.range(0.06, 0.12);
          const jitter = rng.range(-0.09, 0.09);
          x = C + Math.cos(a + jitter) * r;
          y = C + Math.sin(a + jitter) * r;
          c2d.lineTo(x, y);
        }
        c2d.stroke();
      }
      c2d.lineWidth = 1.0;
      for (const ringR of [0.13, 0.24, 0.36]) {
        c2d.strokeStyle = `rgba(238,246,250,${(0.6 - ringR).toFixed(2)})`;
        c2d.beginPath();
        for (let i = 0; i <= spokes; i++) {
          const a = angles[i % spokes];
          const rr = S * ringR * rng.range(0.88, 1.12);
          const x = C + Math.cos(a) * rr;
          const y = C + Math.sin(a) * rr;
          if (i === 0) c2d.moveTo(x, y);
          else c2d.lineTo(x, y);
        }
        c2d.closePath();
        c2d.stroke();
      }
      c2d.fillStyle = 'rgba(250,253,255,0.9)';
      blobPath(c2d, C, C, S * 0.055, 0.5, rng);
      c2d.fill();
      break;
    }

    case 'blood': {
      const base = 'rgba(96,10,10,';
      c2d.fillStyle = `${base}0.88)`;
      blobPath(c2d, C, C, S * 0.22, 0.7, rng);
      c2d.fill();
      c2d.fillStyle = `${base}0.55)`;
      blobPath(c2d, C, C, S * 0.33, 0.85, rng);
      c2d.fill();
      // Satellite droplets and a couple of runs.
      for (let i = 0; i < 22; i++) {
        const a = rng.range(0, Math.PI * 2);
        const r = S * rng.range(0.2, 0.48);
        c2d.fillStyle = `${base}${rng.range(0.3, 0.85).toFixed(2)})`;
        c2d.beginPath();
        c2d.ellipse(C + Math.cos(a) * r, C + Math.sin(a) * r, rng.range(1, 4.5), rng.range(1, 3), a, 0, Math.PI * 2);
        c2d.fill();
      }
      break;
    }

    case 'scorch': {
      radialHalo(c2d, C, C, S * 0.5, '18,14,12', 0.86);
      c2d.fillStyle = 'rgba(10,8,7,0.9)';
      blobPath(c2d, C, C, S * 0.2, 0.6, rng);
      c2d.fill();
      // Sooty tongues licking outward from the seat of the burn.
      for (let i = 0; i < 12; i++) {
        const a = rng.range(0, Math.PI * 2);
        const r1 = S * rng.range(0.28, 0.49);
        c2d.strokeStyle = `rgba(22,18,15,${rng.range(0.2, 0.6).toFixed(2)})`;
        c2d.lineWidth = rng.range(3, 11);
        c2d.beginPath();
        c2d.moveTo(C, C);
        c2d.lineTo(C + Math.cos(a) * r1, C + Math.sin(a) * r1);
        c2d.stroke();
      }
      break;
    }
  }
}

/** Paint an environment grunge decal onto a (possibly non-square) canvas. */
function paintGrunge(c2d: CanvasRenderingContext2D, w: number, h: number, kind: GrungeKind, rng: Rng): void {
  c2d.clearRect(0, 0, w, h);
  c2d.lineCap = 'round';

  switch (kind) {
    case 'poster': {
      // Sun-bleached bill posted on a wall, one corner peeled and torn.
      const pad = w * 0.06;
      c2d.fillStyle = 'rgba(224,212,186,0.93)';
      c2d.beginPath();
      c2d.moveTo(pad, pad);
      c2d.lineTo(w - pad, pad + rng.range(0, h * 0.02));
      c2d.lineTo(w - pad - rng.range(0, w * 0.16), h - pad);
      c2d.lineTo(pad + rng.range(0, w * 0.03), h - pad * 1.4);
      c2d.closePath();
      c2d.fill();
      // A saturated header band: the map's colour accents want a home.
      c2d.fillStyle = rng.chance(0.5) ? 'rgba(38,132,132,0.92)' : 'rgba(184,116,36,0.92)';
      c2d.fillRect(pad * 1.6, pad * 1.8, w - pad * 3.2, h * 0.14);
      // Body copy as blocks; real letterforms would be unreadable at any
      // distance the player actually sees this from.
      c2d.fillStyle = 'rgba(52,44,38,0.72)';
      let y = h * 0.28;
      while (y < h * 0.84) {
        const lw = (w - pad * 3.2) * rng.range(0.4, 1.0);
        c2d.fillRect(pad * 1.6, y, lw, h * rng.range(0.018, 0.033));
        y += h * rng.range(0.055, 0.085);
      }
      // Stains and abrasion.
      for (let i = 0; i < 7; i++) {
        c2d.fillStyle = `rgba(150,128,96,${rng.range(0.06, 0.2).toFixed(2)})`;
        blobPath(c2d, rng.range(0, w), rng.range(0, h), rng.range(w * 0.05, w * 0.2), 0.7, rng);
        c2d.fill();
      }
      break;
    }

    case 'graffiti': {
      const palette = ['rgba(226,74,52,', 'rgba(38,150,150,', 'rgba(232,196,60,', 'rgba(30,30,34,'];
      for (let s = 0; s < 4; s++) {
        c2d.strokeStyle = `${rng.pick(palette)}${rng.range(0.55, 0.9).toFixed(2)})`;
        c2d.lineWidth = rng.range(w * 0.02, w * 0.055);
        c2d.beginPath();
        let x = rng.range(w * 0.08, w * 0.3);
        let y = rng.range(h * 0.25, h * 0.75);
        c2d.moveTo(x, y);
        const steps = rng.int(3, 6);
        for (let i = 0; i < steps; i++) {
          const nx = x + rng.range(w * 0.08, w * 0.22);
          const ny = rng.range(h * 0.15, h * 0.85);
          c2d.quadraticCurveTo(x + rng.range(-20, 20), y + rng.range(-40, 40), nx, ny);
          x = nx;
          y = ny;
        }
        c2d.stroke();
        // Overspray speckle around the stroke.
        for (let i = 0; i < 40; i++) {
          c2d.fillStyle = `${rng.pick(palette)}${rng.range(0.05, 0.25).toFixed(2)})`;
          c2d.beginPath();
          c2d.arc(rng.range(0, w), rng.range(h * 0.1, h * 0.9), rng.range(0.6, 2.4), 0, Math.PI * 2);
          c2d.fill();
        }
      }
      break;
    }

    case 'stain': {
      // A dark seep with drip runs — under an AC unit, a drum, a wound.
      for (let i = 0; i < 5; i++) {
        c2d.fillStyle = `rgba(34,28,22,${rng.range(0.16, 0.4).toFixed(2)})`;
        blobPath(c2d, w * 0.5 + rng.range(-w * 0.12, w * 0.12), h * 0.42 + rng.range(-h * 0.1, h * 0.1), w * rng.range(0.16, 0.36), 0.75, rng);
        c2d.fill();
      }
      for (let i = 0; i < 9; i++) {
        const x = w * 0.5 + rng.range(-w * 0.3, w * 0.3);
        c2d.strokeStyle = `rgba(30,25,20,${rng.range(0.1, 0.32).toFixed(2)})`;
        c2d.lineWidth = rng.range(1.5, 6);
        c2d.beginPath();
        c2d.moveTo(x, h * 0.42);
        c2d.lineTo(x + rng.range(-6, 6), h * rng.range(0.6, 0.96));
        c2d.stroke();
      }
      break;
    }

    case 'tyremark': {
      // A skid: dense in the middle, feathering out at both ends.
      const grd = c2d.createLinearGradient(0, 0, w, 0);
      grd.addColorStop(0, 'rgba(26,24,22,0)');
      grd.addColorStop(0.22, 'rgba(26,24,22,0.42)');
      grd.addColorStop(0.6, 'rgba(26,24,22,0.5)');
      grd.addColorStop(1, 'rgba(26,24,22,0)');
      c2d.fillStyle = grd;
      c2d.fillRect(0, h * 0.24, w, h * 0.52);
      // Tread blocks.
      c2d.fillStyle = 'rgba(14,13,12,0.34)';
      for (let x = 0; x < w; x += w * 0.024) {
        const a = 1 - Math.abs(x / w - 0.5) * 2;
        if (rng.next() > a * 0.9) continue;
        c2d.fillRect(x, h * (0.27 + rng.range(0, 0.06)), w * 0.013, h * rng.range(0.32, 0.44));
      }
      break;
    }

    case 'bulletscar': {
      // A *burst* of impacts baked into one texture. Eight of these quads
      // around a doorway reads as sixty rounds of suppressing fire at the cost
      // of eight instances — and unlike runtime decals they never get evicted
      // by a firefight happening on the other side of the map.
      const holes = rng.int(7, 12);
      for (let i = 0; i < holes; i++) {
        // Cluster tighter toward the middle: a burst has a centre of mass.
        const cx = w * (0.5 + rng.gauss(0, 0.19));
        const cy = h * (0.5 + rng.gauss(0, 0.19));
        const r = Math.min(w, h) * rng.range(0.045, 0.085);
        radialHalo(c2d, cx, cy, r * 2.6, '186,176,156', 0.3);
        c2d.fillStyle = 'rgba(168,161,146,0.72)';
        blobPath(c2d, cx, cy, r * 1.5, 0.55, rng);
        c2d.fill();
        c2d.strokeStyle = 'rgba(60,54,46,0.5)';
        for (let k = 0; k < 4; k++) {
          const a = rng.range(0, Math.PI * 2);
          c2d.lineWidth = rng.range(0.6, 1.5);
          c2d.beginPath();
          c2d.moveTo(cx, cy);
          c2d.lineTo(cx + Math.cos(a) * r * rng.range(1.6, 3.2), cy + Math.sin(a) * r * rng.range(1.6, 3.2));
          c2d.stroke();
        }
        c2d.fillStyle = 'rgba(26,22,19,0.94)';
        blobPath(c2d, cx, cy, r * 0.55, 0.45, rng);
        c2d.fill();
      }
      break;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// System
// ─────────────────────────────────────────────────────────────────────────────

export class Decals implements IDecals, System {
  readonly name = 'decals';

  /** Parent for every decal mesh, so the whole system can be hidden at once. */
  readonly root = new THREE.Group();

  private readonly pools = new Map<DecalFamily, Pool>();
  private readonly grungePools = new Map<GrungeKind, Pool>();
  /** Insertion-ordered list of live impact decals; the front is the oldest. */
  private readonly live: PoolEntry[] = [];
  private readonly pendingGrunge: PendingGrunge[] = [];

  private readonly materials: THREE.MeshStandardMaterial[] = [];
  private readonly textures: THREE.Texture[] = [];
  private readonly rng = new Rng(0xdeca15);

  private budget = 256;
  private ready = false;

  /** Seconds a decal persists at full opacity before it begins to fade. */
  private readonly lifetime = 90;
  /** Seconds the natural fade-out takes. */
  private readonly naturalFade = 8;
  /** Seconds the forced fade takes when the budget evicts a decal. */
  private readonly evictFade = 0.35;
  /**
   * Hard ceiling above the budget. Eviction is a *fade*, so decals linger for a
   * moment after being retired; without a hard cap a fast enough fire rate
   * could outrun the fade and grow the pool without bound.
   */
  private readonly overshoot = 32;

  private readonly tmpQuat = new THREE.Quaternion();
  private readonly tmpQuatRoll = new THREE.Quaternion();
  private readonly tmpMat = new THREE.Matrix4();
  private readonly tmpPos = new THREE.Vector3();
  private readonly tmpScale = new THREE.Vector3();
  private readonly zeroScale = new THREE.Vector3(0, 0, 0);
  private readonly forward = new THREE.Vector3(0, 0, 1);

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    const settings: QualitySettings = ctx.settings;
    this.budget = Math.max(32, settings.decalBudget);
    const capacity = this.budget + this.overshoot;
    const aniso = Math.min(8, Math.max(1, settings.anisotropy));

    this.root.name = 'decals';
    // Decals are pure surface dressing scattered over the whole map; there is
    // no useful bounding volume for the group, and culling the group as a unit
    // would pop every decal at once. Cull per-mesh instead.
    this.root.frustumCulled = false;
    ctx.scene.add(this.root);

    for (const family of IMPACT_FAMILIES) {
      const tex = this.makeImpactTexture(family, aniso);
      const pool = this.makePool(`impact_${family}`, tex, capacity, 1);
      if (pool) this.pools.set(family, pool);
    }
    for (const kind of GRUNGE_KINDS) {
      const tex = this.makeGrungeTexture(kind, aniso);
      // Grunge is authored, not sprayed: 48 of each kind is far more than the
      // level actually places, and it never recycles.
      const pool = this.makePool(`grunge_${kind}`, tex, 48, GRUNGE_ASPECT[kind]);
      if (pool) this.grungePools.set(kind, pool);
    }

    this.ready = true;
    // Anything the level asked for before we existed.
    for (const g of this.pendingGrunge) this.placeGrunge(g);
    this.pendingGrunge.length = 0;
  }

  update(_ctx: EngineContext, frame: FrameContext): void {
    if (!this.ready || this.live.length === 0) return;
    const dt = frame.dt;
    if (dt <= 0) return;

    for (let i = this.live.length - 1; i >= 0; i--) {
      const e = this.live[i];
      e.life -= dt;
      if (e.life > 0) continue;

      const pool = this.pools.get(e.family);
      if (!pool) {
        this.live.splice(i, 1);
        continue;
      }
      const t = Math.max(0, 1 + e.life / e.fadeTime); // life is negative here
      if (t <= 0) {
        this.releaseSlot(pool, e.slot);
        this.live.splice(i, 1);
      } else {
        // Ease the tail so the decal does not visibly "switch off".
        pool.fade.setX(e.slot, t * t);
        pool.dirtyFade = true;
      }
    }

    for (const pool of this.pools.values()) this.flush(pool);
  }

  dispose(): void {
    for (const pool of [...this.pools.values(), ...this.grungePools.values()]) {
      pool.mesh.geometry.dispose();
      pool.mesh.dispose();
      this.root.remove(pool.mesh);
    }
    for (const m of this.materials) m.dispose();
    for (const t of this.textures) t.dispose();
    this.pools.clear();
    this.grungePools.clear();
    this.live.length = 0;
    this.root.removeFromParent();
    this.ready = false;
  }

  // ── IDecals ────────────────────────────────────────────────────────────────

  /**
   * Place an impact decal. `size` is the decal's width in metres — a 5.56 mm
   * hole in concrete wants roughly 0.09 m including its dust halo.
   */
  add(position: THREE.Vector3, normal: THREE.Vector3, surface: SurfaceType, size: number): void {
    if (!this.ready) return;
    const family = SURFACE_TO_FAMILY[surface];
    if (!family) return; // water, and anything unmapped: no decal by design
    const pool = this.pools.get(family);
    if (!pool) return;

    this.evictIfNeeded();

    const slot = this.acquireSlot(pool);
    if (slot < 0) return;

    // Roll about the surface normal. Without it every hole on a given wall
    // shares an axis and the texture repetition is immediately obvious.
    this.orient(normal, this.rng.range(0, Math.PI * 2));
    // 1.2 cm of standoff. Enough to clear the polygon and any small curvature,
    // small enough to be invisible at a 1.7 m eye height.
    this.tmpPos.copy(normal).multiplyScalar(0.012).add(position);
    const s = Math.max(0.02, size);
    this.tmpScale.set(s, s, 1);
    this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
    pool.mesh.setMatrixAt(slot, this.tmpMat);
    pool.fade.setX(slot, 1);
    pool.dirtyMatrix = true;
    pool.dirtyFade = true;
    this.flush(pool);

    this.live.push({ family, slot, life: this.lifetime, fadeTime: this.naturalFade, dying: false });
  }

  clear(): void {
    for (const e of this.live) {
      const pool = this.pools.get(e.family);
      if (pool) this.releaseSlot(pool, e.slot);
    }
    this.live.length = 0;
    for (const pool of this.pools.values()) this.flush(pool);
  }

  // ── Environment storytelling ───────────────────────────────────────────────

  /**
   * Place a permanent grunge decal. Called by the level at build time; safe to
   * call before `init`, in which case the request is queued.
   *
   * These never expire and never count against the impact budget: a poster is
   * set dressing, not a bullet hole, and it must never be evicted because a
   * firefight happened somewhere else on the map.
   */
  addGrunge(
    position: THREE.Vector3,
    normal: THREE.Vector3,
    kind: GrungeKind,
    size: number,
    roll = 0,
    opacity = 1,
  ): void {
    const req: PendingGrunge = {
      position: position.clone(),
      normal: normal.clone().normalize(),
      kind,
      size,
      roll,
      opacity,
    };
    if (!this.ready) {
      this.pendingGrunge.push(req);
      return;
    }
    this.placeGrunge(req);
  }

  /** Live impact decals, for the perf overlay. */
  get activeCount(): number {
    return this.live.length;
  }

  /** Draw calls contributed by this system. */
  get drawCalls(): number {
    return this.pools.size + this.grungePools.size;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private placeGrunge(req: PendingGrunge): void {
    const pool = this.grungePools.get(req.kind);
    if (!pool) return;
    const slot = this.acquireSlot(pool);
    if (slot < 0) return;
    this.orient(req.normal, req.roll);
    // Grunge sits closer to the surface than impacts so a bullet hole punched
    // through a poster still renders on top of it.
    this.tmpPos.copy(req.normal).multiplyScalar(0.006).add(req.position);
    this.tmpScale.set(req.size, req.size, 1);
    this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
    pool.mesh.setMatrixAt(slot, this.tmpMat);
    pool.fade.setX(slot, req.opacity);
    pool.dirtyMatrix = true;
    pool.dirtyFade = true;
    this.flush(pool);
  }

  /** Build the quad-to-surface rotation into `tmpQuat`. */
  private orient(normal: THREE.Vector3, roll: number): void {
    this.tmpQuat.setFromUnitVectors(this.forward, normal);
    this.tmpQuatRoll.setFromAxisAngle(normal, roll);
    this.tmpQuat.premultiply(this.tmpQuatRoll);
  }

  private acquireSlot(pool: Pool): number {
    const reused = pool.free.pop();
    if (reused !== undefined) return reused;
    if (pool.high >= pool.capacity) return -1;
    const slot = pool.high++;
    pool.mesh.count = pool.high;
    return slot;
  }

  private releaseSlot(pool: Pool, slot: number): void {
    // Collapse the instance to zero scale so it rasterises nothing at all; the
    // slot stays in the buffer but costs no fragments.
    this.tmpMat.compose(this.tmpPos.set(0, -1000, 0), this.tmpQuat.identity(), this.zeroScale);
    pool.mesh.setMatrixAt(slot, this.tmpMat);
    pool.fade.setX(slot, 0);
    pool.dirtyMatrix = true;
    pool.dirtyFade = true;
    pool.free.push(slot);
  }

  /**
   * Retire the oldest decals once the budget is exceeded. They fade rather than
   * vanish, which is invisible in practice; only if the fire rate outruns the
   * fade do we free a slot outright.
   */
  private evictIfNeeded(): void {
    let guard = 0;
    while (this.live.length >= this.budget && guard++ < 64) {
      const oldest = this.live[0];
      if (!oldest.dying) {
        oldest.dying = true;
        oldest.life = 0;
        oldest.fadeTime = this.evictFade;
      }
      if (this.live.length < this.budget + this.overshoot) break;
      const pool = this.pools.get(oldest.family);
      if (pool) this.releaseSlot(pool, oldest.slot);
      this.live.shift();
    }
  }

  private flush(pool: Pool): void {
    if (pool.dirtyMatrix) {
      pool.mesh.instanceMatrix.needsUpdate = true;
      pool.dirtyMatrix = false;
    }
    if (pool.dirtyFade) {
      pool.fade.needsUpdate = true;
      pool.dirtyFade = false;
    }
  }

  private makePool(name: string, texture: THREE.Texture | null, capacity: number, aspect: number): Pool | null {
    if (!texture) return null;
    const geo = new THREE.PlaneGeometry(1, 1 / aspect);
    const fade = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    fade.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aFade', fade);

    const mat = new THREE.MeshStandardMaterial({
      map: texture,
      transparent: true,
      // Trim the fully-transparent border so the quad does not pay for it and
      // so shadow/depth passes ignore it entirely.
      alphaTest: 0.02,
      depthWrite: false,
      // Pull the fragment toward the camera in depth so a decal lying exactly
      // on a wall wins the depth test without any geometric offset.
      polygonOffset: true,
      polygonOffsetFactor: -6,
      polygonOffsetUnits: -6,
      roughness: 0.92,
      metalness: 0.0,
      side: THREE.FrontSide,
    });
    this.patchFade(mat);
    this.materials.push(mat);

    const mesh = new THREE.InstancedMesh(geo, mat, capacity);
    mesh.name = `decal_${name}`;
    mesh.count = 0;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Decals move around the whole map as the pool recycles, so a static
    // bounding volume is meaningless — but a *large* one is correct and cheap.
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 4, 0), 90);
    mesh.frustumCulled = false;
    // Draw after opaque geometry, and after each other in insertion order.
    mesh.renderOrder = 2;
    this.root.add(mesh);

    return {
      mesh,
      fade,
      free: [],
      high: 0,
      capacity,
      dirtyMatrix: false,
      dirtyFade: false,
    };
  }

  /**
   * Inject a per-instance alpha multiplier.
   *
   * three offers `instanceColor`, which multiplies RGB and cannot fade a decal
   * out, so we add one float attribute and two lines of GLSL. Patching right
   * after `<map_fragment>` is the stable hook: at that point `diffuseColor.a`
   * holds the texture's alpha and every downstream chunk respects it.
   */
  private patchFade(mat: THREE.MeshStandardMaterial): void {
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', '#include <common>\nattribute float aFade;\nvarying float vFade;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n\tvFade = aFade;');
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', '#include <common>\nvarying float vFade;')
        .replace('#include <map_fragment>', '#include <map_fragment>\n\tdiffuseColor.a *= vFade;');
    };
    // Without a distinct cache key three could hand this material a program
    // compiled for an unpatched material with identical parameters.
    mat.customProgramCacheKey = () => 'noisefield:decal-fade';
  }

  private makeImpactTexture(family: DecalFamily, aniso: number): THREE.Texture | null {
    const S = 128;
    const made = makeCanvas(S, S);
    if (!made) return null;
    paintImpact(made.c2d, S, family, this.rng);
    return this.finishTexture(made.canvas, aniso);
  }

  private makeGrungeTexture(kind: GrungeKind, aniso: number): THREE.Texture | null {
    const base = 192;
    const aspect = GRUNGE_ASPECT[kind];
    const w = aspect >= 1 ? base : Math.round(base * aspect);
    const h = aspect >= 1 ? Math.round(base / aspect) : base;
    const made = makeCanvas(w, h);
    if (!made) return null;
    paintGrunge(made.c2d, w, h, kind, this.rng);
    return this.finishTexture(made.canvas, aniso);
  }

  private finishTexture(canvas: HTMLCanvasElement, aniso: number): THREE.Texture {
    const tex = new THREE.CanvasTexture(canvas);
    // Authored in sRGB on a 2D canvas; tell three so it linearises on sample.
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = aniso;
    // Clamp: a decal that tiles is a decal that has smeared across the wall.
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    this.textures.push(tex);
    return tex;
  }
}
