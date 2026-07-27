/**
 * NOISEFIELD — Core combat HUD.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 * Entirely event-driven: this module never reaches into another system for
 * state. The only two things it pulls from `EngineContext` are the camera
 * (for the compass heading and the field-of-view cone, which are *view*
 * properties and have no event) and, once at init, the level (for the minimap
 * footprint). Everything else arrives on the bus.
 *
 * The split between DOM and canvas is deliberate:
 *   • DOM  — text and discrete state (ammo, health, killfeed, score, toasts).
 *            Text layout, font fallback and shadowing are things the browser
 *            does better than any canvas code, and these elements change a few
 *            times per second, not every frame.
 *   • Canvas — everything that changes every frame or needs exact vector
 *            geometry (reticle, hitmarkers, damage arcs, compass tape,
 *            minimap). Doing these in DOM means a style recalculation per
 *            frame on the main thread, which is the classic way a browser
 *            game loses its frame budget to its own HUD.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * LEGIBILITY
 * ═══════════════════════════════════════════════════════════════════════════
 * The level is sunlit sand and deep shadow in the same frame. Every text run
 * carries a two-stage shadow and every canvas stroke is drawn over a wider
 * near-black keyline. Without that, half the HUD disappears half the time —
 * which is the single most common failure mode of a hand-rolled game HUD.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  FrameContext,
  GameEvents,
  IHUD,
  WeaponId,
} from '../core/Types';
import { Crosshair } from './Crosshair';
import { Minimap } from './Minimap';
import './hud.css';

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

const smoothstep = (edge0: number, edge1: number, x: number): number => {
  const t = clamp((x - edge0) / Math.max(1e-6, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
};

const damp = (a: number, b: number, tau: number, dt: number): number =>
  b + (a - b) * Math.exp(-dt / Math.max(1e-4, tau));

/** Wrap a radian delta into (-π, π]. */
const wrapPi = (a: number): number => {
  let x = a;
  while (x <= -Math.PI) x += Math.PI * 2;
  while (x > Math.PI) x -= Math.PI * 2;
  return x;
};

const SVG_NS = 'http://www.w3.org/2000/svg';

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls !== undefined) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgIcon(viewBox: string, cls: string, d: string, evenOdd = false): SVGSVGElement {
  const root = document.createElementNS(SVG_NS, 'svg');
  root.setAttribute('viewBox', viewBox);
  root.setAttribute('class', cls);
  root.setAttribute('aria-hidden', 'true');
  root.setAttribute('focusable', 'false');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', d);
  if (evenOdd) path.setAttribute('fill-rule', 'evenodd');
  root.appendChild(path);
  return root;
}

/**
 * Restart a CSS animation on an element.
 *
 * Reading `offsetWidth` forces a synchronous layout, which is normally the
 * thing to avoid — but it is the only reliable way to rewind a declarative
 * keyframe animation, it touches one small element, and it happens on discrete
 * gameplay events (a kill, a hit) rather than every frame. The alternative,
 * driving these from JS per frame, costs far more.
 */
function replayAnimation(node: HTMLElement, attr: string): void {
  node.removeAttribute(attr);
  void node.offsetWidth;
  node.setAttribute(attr, '1');
}

// ─────────────────────────────────────────────────────────────────────────────
// Procedural weapon glyphs
//
// Drawn as SVG path silhouettes rather than text or images: zero network cost,
// they scale to any DPI, and a silhouette is recognised faster than a word in
// peripheral vision — which is the whole reason a killfeed uses icons.
// ─────────────────────────────────────────────────────────────────────────────

const GLYPH_VIEWBOX = '0 0 44 14';

const WEAPON_GLYPHS: Readonly<Record<WeaponId, string>> = {
  ar_vector:
    'M0 5.1h6.2v3.9H0z M6.2 4.4h22.6v5.1H6.2z M28.8 6.1h13.4v1.9H28.8z ' +
    'M35.6 4.2h1.5v2.1h-1.5z M13.4 9.5h5.2l-1.3 4.5h-4.4z M9 9.5h2.9l-1.1 3.2H8z ' +
    'M16.5 2.6h8.6v1.8h-8.6z',
  smg_wasp:
    'M4 5.2h6.2v1.5H4z M4 5.2h1.5v3.6H4z M10 4.6h17.4v4.9H10z ' +
    'M27.4 6.2h6.8v1.7h-6.8z M16 9.5h4.6l-2 4.5h-4.3z M11.6 9.5h2.7l-1 3.1h-2.5z ' +
    'M18.4 2.9h5.6v1.7h-5.6z',
  sniper_longbow:
    'M0 4.8h6.6v4.6H0z M6.6 5.2h20.2v4.4H6.6z M26.8 6.3h16.4v1.7H26.8z ' +
    'M12.4 2.2h12.2v2.2H12.4z M14.4 4.4h1.4v1.1h-1.4z M21.2 4.4h1.4v1.1h-1.4z ' +
    'M25.4 4.1h3v1.4h-3z M15.4 9.6h4v3.1h-4z ' +
    'M35.2 7.9h1.1l2.3 4.5h-1.2z M36.5 7.9h1.1l-2.3 4.5h-1.2z',
  shotgun_breach:
    'M0 5h7.6v4.4H0z M7.6 4.9h16.4v4.6H7.6z M24 5.3h18.6v1.8H24z ' +
    'M24 7.6h15.6v1.6H24z M27.6 7.1h6.4v2.8h-6.4z M10.8 9.5h2.8l-1 3.1h-2.6z ' +
    'M40.2 4.4h1.3v1.4h-1.3z',
  pistol_sidearm:
    'M13 4.3h19.4v3.2H13z M32.4 5h1.9v1.7h-1.9z M14 7.5h11.8v1.7H14z ' +
    'M16 9.2h5.4l-1.9 4.8h-5.2z M22.4 9.2h3.4v1h-3.4z',
};

/**
 * Generic cartridge. Used for any kill the player did not make: the
 * `entity:killed` payload carries no weapon, and stamping the *player's*
 * current weapon on someone else's kill is a plain fabrication. A round says
 * "killed by gunfire" and claims nothing further.
 */
const GLYPH_GENERIC =
  'M14.8 4.5h1.5v5h-1.5z M16.6 4.5h10.6v5H16.6z ' +
  'M27.2 4.5c3 .6 4.8 1.8 5.8 2.5-1 .7-2.8 1.9-5.8 2.5z';

/** Skull. `evenodd` turns the eye and nose sub-paths into holes. */
const GLYPH_HEADSHOT =
  'M8 1.6c-3.4 0-6 2.4-6 5.6 0 1.9.9 3.4 2.3 4.3v2.5h1.9v-1.5h1.1v1.5h1.4v-1.5h1.1v1.5h1.9v-2.5' +
  'c1.4-.9 2.3-2.4 2.3-4.3 0-3.2-2.6-5.6-6-5.6z ' +
  'M5.9 6a1.7 1.7 0 100 3.4 1.7 1.7 0 000-3.4z ' +
  'M10.1 6a1.7 1.7 0 100 3.4 1.7 1.7 0 000-3.4z ' +
  'M8 9.5l-1.1 2.1h2.2z';

/** Shield with a cross cut out of it (evenodd). */
const GLYPH_HEALTH =
  'M8 1l6 2.2v5.1c0 3.5-2.5 5.7-6 6.7-3.5-1-6-3.2-6-6.7V3.2z ' +
  'M7 4.9h2v2.1h2.1v2H9v2.1H7V9H4.9V7H7z';

/**
 * Display names. Kept local rather than imported from the weapon system's
 * `Specs.ts`, so a balance-pass edit in that file can never break the HUD
 * build. `WeaponId` in `Types.ts` is the contract; anything else is a courtesy.
 */
const WEAPON_NAMES: Readonly<Record<WeaponId, string>> = {
  ar_vector: 'VECTOR-5',
  smg_wasp: 'WASP-9',
  sniper_longbow: 'LONGBOW .338',
  shotgun_breach: 'BREACH-12',
  pistol_sidearm: 'SIDEARM-9',
};

const prettyWeapon = (id: string): string =>
  WEAPON_NAMES[id as WeaponId] ?? id.replace(/[_-]+/g, ' ').toUpperCase();

// ─────────────────────────────────────────────────────────────────────────────
// Directional damage indicators
// ─────────────────────────────────────────────────────────────────────────────

interface ThreatArc {
  /** Screen-relative bearing, radians. 0 = dead ahead, +½π = right. */
  angle: number;
  /** World bearing, for the matching caret on the compass tape. */
  worldBearing: number;
  t: number;
  strength: number;
}

/**
 * Red arcs ringing the reticle, pointing at whoever just shot you.
 *
 * Placement is a compromise the genre settled on long ago: far enough from the
 * centre that it never masks the target, close enough that reading it needs no
 * eye movement. ~17% of the screen's short axis puts it just outside the
 * reticle's largest bloom and well inside the damage vignette at the frame
 * edge — the two never occupy the same pixels, which is why they stay legible
 * simultaneously.
 */
class ThreatArcs {
  readonly el: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D | null;
  private side = 400;
  private dpr = 1;
  private s = 1;
  private readonly arcs: ThreatArc[] = [];
  /** 0..1 current vignette intensity; drives contrast compensation. */
  private backdrop = 0;
  /** True when the canvas has ink on it. Lets an idle frame skip the clear —
   *  this backing store reaches 2676x2676 at 4K and clearing it every frame
   *  for the 99% of the time nothing is drawn is pure waste. */
  private dirty = false;

  /** 1.2s: long enough to turn and reacquire, short enough that stale threat
   *  information never accumulates into a permanent red ring. */
  private static readonly LIFE = 1.2;

  constructor() {
    this.el = document.createElement('canvas');
    this.el.className = 'nf-threat';
    this.el.setAttribute('aria-hidden', 'true');
    this.g = this.el.getContext('2d', { alpha: true, desynchronized: true });
  }

  mount(parent: HTMLElement): void {
    parent.appendChild(this.el);
  }

  resize(vw: number, vh: number): void {
    const short = Math.min(vw, vh);
    this.s = clamp(short / 1080, 0.85, 2.1);
    const side = Math.round(clamp(short * 0.62, 300, 1500) / 2) * 2;
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    this.side = side;
    this.dpr = dpr;
    // Compare against the ELEMENT, not a cached field — see Crosshair.resize.
    // A fresh canvas is 300x150, never 0, so an `el.width === 0` guard does not
    // catch the never-applied case.
    const px = Math.round(side * dpr);
    if (this.el.width !== px || this.el.height !== px) {
      this.el.width = px;
      this.el.height = px;
      this.dirty = true;
    }
    this.el.style.width = `${side}px`;
    this.el.style.height = `${side}px`;
    this.el.style.left = `${Math.round((vw - side) * 0.5)}px`;
    this.el.style.top = `${Math.round((vh - side) * 0.5)}px`;
  }

  setBackdrop(v: number): void {
    this.backdrop = clamp(v, 0, 1);
  }

  /**
   * Bursts from a single attacker merge instead of stacking: five arcs at the
   * same bearing is not five times as informative, it is just five times
   * brighter, and the brightness is what makes an indicator hard to localise.
   */
  add(angle: number, worldBearing: number, strength: number): void {
    for (const a of this.arcs) {
      if (Math.abs(wrapPi(a.angle - angle)) < 0.32) {
        a.angle = angle;
        a.worldBearing = worldBearing;
        a.t = 0;
        a.strength = Math.min(1, a.strength + strength * 0.6);
        return;
      }
    }
    if (this.arcs.length >= 6) this.arcs.shift();
    this.arcs.push({ angle, worldBearing, t: 0, strength: clamp(strength, 0.25, 1) });
  }

  /** Live arcs, so the compass can mirror them as carets. */
  get active(): readonly ThreatArc[] {
    return this.arcs;
  }

  static progress(t: number): number {
    return clamp(t / ThreatArcs.LIFE, 0, 1);
  }

  clear(): void {
    this.arcs.length = 0;
  }

  update(dt: number): void {
    const g = this.g;
    if (g === null) return;

    for (let i = this.arcs.length - 1; i >= 0; i--) {
      this.arcs[i].t += dt;
      if (this.arcs[i].t >= ThreatArcs.LIFE) this.arcs.splice(i, 1);
    }

    if (this.arcs.length === 0 && !this.dirty) return;

    const px = this.side * this.dpr;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, px, px);
    this.dirty = false;
    if (this.arcs.length === 0) return;
    this.dirty = true;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const c = this.side * 0.5;
    const baseR = this.side * 0.275;

    for (const a of this.arcs) {
      const u = a.t / ThreatArcs.LIFE;
      // 60ms punch-in: the indicator must land before the player has finished
      // reacting to the damage sound, or it feels like a delayed report.
      const inT = clamp(a.t / 0.06, 0, 1);
      // Slow start then accelerating fade, so it holds legible for ~0.6s and
      // then leaves quickly rather than lingering as a dim smear.
      const alpha = (1 - u * u * u) * inT * (0.55 + 0.45 * a.strength);
      if (alpha < 0.01) continue;

      const r = baseR * (1.16 - 0.16 * inT);
      // HALF-angle. ~28-33° of total arc: wide enough to be unambiguous about
      // which quadrant the shot came from, narrow enough to still read as an
      // arrow rather than as a ring closing in around the player.
      const half = 0.24 + 0.05 * a.strength;
      const thick = (4.2 + 4 * a.strength) * this.s;
      // Canvas 0 rad points +X; the reticle's "ahead" is -Y, hence the -½π.
      const mid = a.angle - Math.PI / 2;

      this.tapered(g, c, r, mid, half, thick, alpha);
    }
    g.globalAlpha = 1;
  }

  /**
   * A lens/chevron: thickness follows a sine profile so the arc tapers to
   * nothing at both ends. A constant-width stroke reads as a UI ring; a
   * tapered one reads as a direction, which is the whole job.
   */
  private tapered(
    g: CanvasRenderingContext2D,
    c: number,
    r: number,
    mid: number,
    half: number,
    thick: number,
    alpha: number,
  ): void {
    const N = 20;
    const ang = (u: number): number => mid + (u - 0.5) * 2 * half;
    // sin(πu) tapers symmetrically to zero at both tips; the 0.7 exponent
    // fattens the middle so the body stays solid instead of looking like a
    // hairline with two spikes.
    const t = (u: number): number => Math.pow(Math.sin(u * Math.PI), 0.7) * thick * 0.5;

    g.beginPath();
    for (let i = 0; i <= N; i++) {
      const u = i / N;
      const a = ang(u);
      const rr = r + t(u);
      const x = c + Math.cos(a) * rr;
      const y = c + Math.sin(a) * rr;
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    for (let i = N; i >= 0; i--) {
      const u = i / N;
      const a = ang(u);
      const rr = r - t(u);
      g.lineTo(c + Math.cos(a) * rr, c + Math.sin(a) * rr);
    }
    g.closePath();

    // Stroke the outline FIRST and fill over it, so the fill masks the inner
    // half of the stroke and what survives is a clean, uniform outer keyline.
    // Inflating a second path instead (the obvious approach) produces a
    // lopsided border, because a radial scale is not a constant-width offset.
    //
    // The keyline is what keeps the arc readable on top of the red low-health
    // vignette: it separates the two by luminance rather than by hue, and
    // luminance is the only channel peripheral vision resolves well. It is
    // pushed harder the redder the screen already is.
    g.globalAlpha = alpha * (0.6 + 0.3 * this.backdrop);
    g.lineJoin = 'round';
    g.lineWidth = Math.max(1.8, 2.4 * this.s);
    g.strokeStyle = 'rgba(0, 0, 0, 0.9)';
    g.stroke();

    g.globalAlpha = alpha;
    g.fillStyle = '#ff4436';
    g.fill();

    // Hot inner core along the arc's spine, boosted against a red backdrop.
    g.globalAlpha = alpha * (0.45 + 0.45 * this.backdrop);
    g.beginPath();
    for (let i = 0; i <= N; i++) {
      const u = i / N;
      const a = ang(u);
      const x = c + Math.cos(a) * r;
      const y = c + Math.sin(a) * r;
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.lineWidth = Math.max(1, 1.3 * this.s);
    g.strokeStyle = 'rgba(255, 230, 224, 0.95)';
    g.stroke();
  }

  dispose(): void {
    this.el.remove();
    this.el.width = 0;
    this.el.height = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compass tape
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A scrolling heading tape, the way an aircraft HUD or a shooter's top strip does it.
 *
 * Why a tape and not a rose: a tape maps heading to a horizontal position that
 * matches the direction the player is actually turning, so "the marker slid
 * left" and "I turned right" agree. A rotating rose requires a mental rotation
 * every time it is read.
 */
class CompassTape {
  readonly el: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D | null;
  private w = 420;
  private hgt = 30;
  private dpr = 1;
  private s = 1;

  /** Degrees visible across the whole tape. */
  private static readonly SPAN = 120;
  private static readonly CARDINALS: ReadonlyArray<[number, string]> = [
    [0, 'N'], [45, 'NE'], [90, 'E'], [135, 'SE'],
    [180, 'S'], [225, 'SW'], [270, 'W'], [315, 'NW'],
  ];

  constructor() {
    this.el = document.createElement('canvas');
    this.el.className = 'nf-compass__tape';
    this.el.setAttribute('aria-hidden', 'true');
    this.g = this.el.getContext('2d', { alpha: true });
  }

  resize(vw: number, vh: number): void {
    const short = Math.min(vw, vh);
    this.s = clamp(short / 1080, 0.85, 2.1);
    const w = Math.round(clamp(vw * 0.4, 300, 820));
    const hgt = Math.round(clamp(short * 0.032, 22, 46));
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    this.w = w;
    this.hgt = hgt;
    this.dpr = dpr;
    // Compare against the ELEMENT — see Crosshair.resize for why a cached
    // field cannot be trusted to reflect what was actually applied.
    const pw = Math.round(w * dpr);
    const ph = Math.round(hgt * dpr);
    if (this.el.width !== pw || this.el.height !== ph) {
      this.el.width = pw;
      this.el.height = ph;
    }
    this.el.style.width = `${w}px`;
    this.el.style.height = `${hgt}px`;
  }

  /**
   * @param headingDeg 0..360, 0 = north.
   * @param threats Screen-independent world bearings (radians) to mark.
   */
  draw(headingDeg: number, threats: readonly ThreatArc[]): void {
    const g = this.g;
    if (g === null) return;

    const w = this.w;
    const hg = this.hgt;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, w * this.dpr, hg * this.dpr);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const pxPerDeg = w / CompassTape.SPAN;
    const cx = w * 0.5;
    const baseY = hg - 1;

    // Backing bar. Subtle, but the tape is unreadable against a bright skyline
    // without something behind it.
    const bg = g.createLinearGradient(0, 0, 0, hg);
    bg.addColorStop(0, 'rgba(6, 11, 16, 0.0)');
    bg.addColorStop(0.55, 'rgba(6, 11, 16, 0.42)');
    bg.addColorStop(1, 'rgba(6, 11, 16, 0.12)');
    g.fillStyle = bg;
    g.fillRect(0, 0, w, hg);

    g.strokeStyle = 'rgba(207, 232, 255, 0.2)';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, baseY - 0.5);
    g.lineTo(w, baseY - 0.5);
    g.stroke();

    // Ticks every 5°, taller every 15°.
    const start = Math.floor((headingDeg - CompassTape.SPAN * 0.5) / 5) * 5;
    const end = headingDeg + CompassTape.SPAN * 0.5 + 5;
    g.lineWidth = Math.max(1, 1.1 * this.s);
    for (let d = start; d <= end; d += 5) {
      const x = cx + wrapDeg(d - headingDeg) * pxPerDeg;
      if (x < -6 || x > w + 6) continue;
      const major = ((d % 15) + 15) % 15 === 0;
      const len = major ? hg * 0.34 : hg * 0.19;
      g.strokeStyle = major ? 'rgba(207, 232, 255, 0.55)' : 'rgba(207, 232, 255, 0.28)';
      g.beginPath();
      g.moveTo(Math.round(x) + 0.5, baseY);
      g.lineTo(Math.round(x) + 0.5, baseY - len);
      g.stroke();
    }

    // Cardinal labels.
    const fs = Math.round(clamp(hg * 0.44, 9, 19));
    g.font = `700 ${fs}px 'Bahnschrift','DIN Alternate','Segoe UI',system-ui,sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'alphabetic';
    for (const [deg, label] of CompassTape.CARDINALS) {
      const x = cx + wrapDeg(deg - headingDeg) * pxPerDeg;
      if (x < -30 || x > w + 30) continue;
      const y = baseY - hg * 0.42;
      g.lineWidth = 3.4;
      g.strokeStyle = 'rgba(0, 0, 0, 0.88)';
      g.strokeText(label, x, y);
      // The four primary points are brighter than the intercardinals: an even
      // brightness makes the tape a picket fence with no anchor to read from.
      const primary = deg % 90 === 0;
      g.fillStyle = primary ? 'rgba(242, 245, 247, 0.95)' : 'rgba(207, 232, 255, 0.6)';
      g.fillText(label, x, y);
    }

    // Threat carets, reusing the damage-direction data so the compass and the
    // on-screen arcs can never disagree about where the shooter is.
    for (const t of threats) {
      const a = 1 - ThreatArcs.progress(t.t);
      if (a <= 0.02) continue;
      const deg = (t.worldBearing * 180) / Math.PI;
      const x = cx + wrapDeg(deg - headingDeg) * pxPerDeg;
      if (x < -10 || x > w + 10) continue;
      g.globalAlpha = a;
      g.beginPath();
      g.moveTo(x, baseY - hg * 0.06);
      g.lineTo(x - 4.4 * this.s, baseY - hg * 0.06 - 6.5 * this.s);
      g.lineTo(x + 4.4 * this.s, baseY - hg * 0.06 - 6.5 * this.s);
      g.closePath();
      g.fillStyle = '#ff4436';
      g.strokeStyle = 'rgba(0,0,0,0.8)';
      g.lineWidth = 1.2;
      g.fill();
      g.stroke();
      g.globalAlpha = 1;
    }

    // Centre index mark.
    g.beginPath();
    g.moveTo(cx, baseY - hg * 0.62);
    g.lineTo(cx - 4.6 * this.s, baseY - hg * 0.62 - 6.8 * this.s);
    g.lineTo(cx + 4.6 * this.s, baseY - hg * 0.62 - 6.8 * this.s);
    g.closePath();
    g.fillStyle = 'rgba(242, 245, 247, 0.95)';
    g.strokeStyle = 'rgba(0, 0, 0, 0.8)';
    g.lineWidth = 1.4;
    g.fill();
    g.stroke();
  }

  dispose(): void {
    this.el.remove();
    this.el.width = 0;
    this.el.height = 0;
  }
}

/** Wrap a degree delta into (-180, 180]. */
function wrapDeg(d: number): number {
  let x = d;
  while (x <= -180) x += 360;
  while (x > 180) x -= 360;
  return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// HUD
// ─────────────────────────────────────────────────────────────────────────────

interface Callout {
  title: string;
  sub: string;
}

const MULTI_KILL_NAMES = ['', '', 'DOUBLE KILL', 'TRIPLE KILL', 'QUAD KILL', 'PENTA KILL'];
/** Streak thresholds worth announcing. Below 3 it is noise. */
const STREAK_STEPS = [3, 5, 7, 10, 15, 20, 25, 30];
/** Seconds between kills that still count as the same multi-kill. */
const MULTI_KILL_WINDOW = 4;

export class HUD implements IHUD {
  readonly name = 'hud';

  private ctx: EngineContext | null = null;
  private root: HTMLElement | null = null;
  private readonly off: Array<() => void> = [];
  private readonly timers = new Set<number>();
  /**
   * At most one pending clear per (element, attribute).
   *
   * Every one-shot flash used to schedule an independent `setTimeout`, so two
   * hits inside the flash duration meant the FIRST timer stripped the
   * attribute partway through the SECOND animation — the retrigger visibly cut
   * itself short, which is worst exactly when hits are arriving fastest.
   */
  private readonly flashTimers = new Map<string, number>();

  private readonly crosshair = new Crosshair();
  private readonly minimap = new Minimap();
  private readonly threat = new ThreatArcs();
  private readonly compass = new CompassTape();

  // DOM handles, created in init.
  private vignette!: HTMLElement;
  private vigPulse!: HTMLElement;
  private hitflash!: HTMLElement;
  private deathwash!: HTMLElement;
  private deathBanner!: HTMLElement;
  private ammoBox!: HTMLElement;
  private ammoName!: HTMLElement;
  private ammoMag!: HTMLElement;
  private ammoReserve!: HTMLElement;
  private ammoPips!: HTMLElement;
  private ringFill!: SVGCircleElement;
  private healthBox!: HTMLElement;
  private healthNum!: HTMLElement;
  private healthFill!: HTMLElement;
  private healthGhost!: HTMLElement;
  private killfeed!: HTMLElement;
  private calloutBox!: HTMLElement;
  private calloutTitle!: HTMLElement;
  private calloutSub!: HTMLElement;
  private bearingText!: HTMLElement;
  private toasts!: HTMLElement;
  private scoreKills!: HTMLElement;
  private scoreDeaths!: HTMLElement;
  private scoreStreak!: HTMLElement;
  private scoreStreakCell!: HTMLElement;

  // State ---------------------------------------------------------------
  private health = 100;
  private maxHealth = 100;
  private displayHealth = 100;
  private vignetteLevel = 0;
  private alive = true;
  /** Lagging "red health" value. A field, not a data-attribute: stashing it on
   *  the DOM meant a string round-trip and an attribute mutation every frame. */
  private ghostHealth = 1;
  /** Last values pushed to the DOM, so a frame that changed nothing writes
   *  nothing. Style writes are cheap but not free, and these run at 60Hz. */
  private lastFillScale = -1;
  private lastGhostScale = -1;
  private lastVignette = -1;

  private mag = 0;
  private magSize = 0;
  private reserve = 0;
  private pipCount = 0;
  private weapon: WeaponId | null = null;

  private reloading = false;
  private reloadT = 0;
  private reloadDur = 1;

  private kills = 0;
  private deaths = 0;
  private streak = 0;
  private lastStreakAnnounced = 0;
  private comboCount = 0;
  private comboTimer = Infinity;

  private readonly calloutQueue: Callout[] = [];
  private calloutCooldown = 0;

  /**
   * Transient DOM entries (killfeed rows, toasts) aged on the UI clock instead
   * of on `setTimeout`.
   *
   * Two bugs died with the timers. First, a row evicted early by the six-row
   * cap left two `setTimeout`s still running against a node that was already
   * detached. Second, wall-clock expiry meant a five-second killfeed row could
   * disappear during a pause while every other part of the HUD was frozen.
   */
  private readonly transients: Array<{ node: HTMLElement; age: number; life: number; out: boolean }> = [];

  private readonly playerPos = new THREE.Vector3();
  private hasPlayerPos = false;
  private readonly tmpDir = new THREE.Vector3();
  private headingDeg = 0;

  private visible = true;
  private paused = false;

  /** Circumference of the reload ring path (r = 44 in a 100-unit viewBox). */
  private static readonly RING_C = 2 * Math.PI * 44;

  // ═══════════════════════════════════════════════════════════════════════════
  // Lifecycle
  // ═══════════════════════════════════════════════════════════════════════════

  init(ctx: EngineContext): void {
    this.ctx = ctx;

    const mount = document.getElementById('ui-root');
    if (mount === null) {
      // Never throw from a HUD: a missing overlay must not stop the game.
      console.warn('[HUD] #ui-root not found — HUD disabled.');
      return;
    }

    const root = h('div', 'nf-hud');
    root.setAttribute('aria-hidden', 'true');
    this.root = root;

    this.buildDamageLayers(root);
    this.threat.mount(root);
    this.crosshair.mount(root);
    this.buildCompass(root);
    this.buildTopLeft(root);
    this.buildKillfeed(root);
    this.buildCallout(root);
    this.buildToasts(root);
    this.buildAmmo(root);
    this.buildHealth(root);

    mount.appendChild(root);

    // Subscribe and lay out BEFORE the minimap's one-off geometry bake. The
    // bake is the only expensive thing in this init and the only plausible
    // thrower; the engine swallows init exceptions, so anything sequenced
    // after it would be silently skipped and leave a mounted but dead HUD.
    this.subscribe(ctx);
    this.resize(ctx.width, ctx.height);
    this.minimap.build(ctx);
  }

  // ── DOM construction ──────────────────────────────────────────────────────

  private buildDamageLayers(root: HTMLElement): void {
    this.vignette = h('div', 'nf-vignette');
    this.vigPulse = h('div', 'nf-vignette__pulse');
    this.vignette.appendChild(this.vigPulse);
    this.hitflash = h('div', 'nf-hitflash');
    this.deathwash = h('div', 'nf-deathwash');

    // Death needs a statement in the CENTRE of the screen, where the player is
    // already looking. A transient corner toast is not a state indicator: it
    // expires while the player is still dead, leaving only a red wash that
    // also means "nearly dead", so dead and almost-dead looked alike.
    this.deathBanner = h('div', 'nf-death');
    this.deathBanner.append(
      h('div', 'nf-death__title', 'ELIMINATED'),
      h('div', 'nf-death__sub', 'STAND BY FOR REDEPLOY'),
    );

    root.append(this.vignette, this.hitflash, this.deathwash, this.deathBanner);
  }

  private buildCompass(root: HTMLElement): void {
    const wrap = h('div', 'nf-compass');
    this.compass.resize(window.innerWidth, window.innerHeight);
    wrap.appendChild(this.compass.el);
    this.bearingText = h('div', 'nf-compass__bearing', '000');
    wrap.appendChild(this.bearingText);
    root.appendChild(wrap);
  }

  private buildTopLeft(root: HTMLElement): void {
    const wrap = h('div', 'nf-topleft');
    this.minimap.mount(wrap);

    const score = h('div', 'nf-score');
    const cell = (label: string): [HTMLElement, HTMLElement] => {
      const box = h('div', 'nf-score__cell');
      const value = h('b', undefined, '0');
      const cap = h('i', undefined, label);
      box.append(value, cap);
      return [box, value];
    };
    const [kBox, kVal] = cell('KILLS');
    const [dBox, dVal] = cell('DEATHS');
    const [sBox, sVal] = cell('STREAK');
    sBox.classList.add('nf-score__cell--streak');
    this.scoreKills = kVal;
    this.scoreDeaths = dVal;
    this.scoreStreak = sVal;
    this.scoreStreakCell = sBox;
    score.append(kBox, dBox, sBox);
    wrap.appendChild(score);
    root.appendChild(wrap);
  }

  private buildKillfeed(root: HTMLElement): void {
    this.killfeed = h('div', 'nf-killfeed');
    root.appendChild(this.killfeed);
  }

  private buildCallout(root: HTMLElement): void {
    this.calloutBox = h('div', 'nf-callout');
    this.calloutTitle = h('div', 'nf-callout__title');
    this.calloutSub = h('div', 'nf-callout__sub');
    this.calloutBox.append(this.calloutTitle, this.calloutSub);
    root.appendChild(this.calloutBox);
  }

  private buildToasts(root: HTMLElement): void {
    this.toasts = h('div', 'nf-toasts');
    root.appendChild(this.toasts);
  }

  private buildAmmo(root: HTMLElement): void {
    const box = h('div', 'nf-ammo');

    // Reload ring, to the left of the numbers so the eye can read progress
    // without leaving the ammo cluster.
    const ring = document.createElementNS(SVG_NS, 'svg');
    ring.setAttribute('viewBox', '0 0 100 100');
    ring.setAttribute('class', 'nf-ammo__ring');
    ring.setAttribute('aria-hidden', 'true');
    const track = document.createElementNS(SVG_NS, 'circle');
    track.setAttribute('class', 'nf-ammo__ring-track');
    track.setAttribute('cx', '50');
    track.setAttribute('cy', '50');
    track.setAttribute('r', '44');
    const fill = document.createElementNS(SVG_NS, 'circle');
    fill.setAttribute('class', 'nf-ammo__ring-fill');
    fill.setAttribute('cx', '50');
    fill.setAttribute('cy', '50');
    fill.setAttribute('r', '44');
    fill.setAttribute('stroke-dasharray', `${HUD.RING_C}`);
    fill.setAttribute('stroke-dashoffset', `${HUD.RING_C}`);
    ring.append(track, fill);
    this.ringFill = fill;

    const stack = h('div', 'nf-ammo__stack');
    this.ammoName = h('div', 'nf-ammo__name', '—');
    const row = h('div', 'nf-ammo__row');
    this.ammoMag = h('span', 'nf-ammo__mag', '0');
    const sep = h('span', 'nf-ammo__sep', '/');
    this.ammoReserve = h('span', 'nf-ammo__reserve', '0');
    row.append(this.ammoMag, sep, this.ammoReserve);
    this.ammoPips = h('div', 'nf-ammo__pips');
    stack.append(this.ammoName, row, this.ammoPips);

    box.append(ring, stack);
    this.ammoBox = box;
    root.appendChild(box);
  }

  private buildHealth(root: HTMLElement): void {
    const box = h('div', 'nf-health');
    const row = h('div', 'nf-health__row');
    row.appendChild(svgIcon('0 0 16 16', 'nf-health__icon', GLYPH_HEALTH, true));
    this.healthNum = h('div', 'nf-health__num', '100');
    row.append(this.healthNum, h('div', 'nf-health__label', 'VITALS'));

    const bar = h('div', 'nf-health__bar');
    this.healthGhost = h('i', 'nf-health__ghost');
    this.healthFill = h('i', 'nf-health__fill');
    bar.append(this.healthGhost, this.healthFill);

    box.append(row, bar);
    this.healthBox = box;
    root.appendChild(box);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Event wiring
  // ═══════════════════════════════════════════════════════════════════════════

  private subscribe(ctx: EngineContext): void {
    const bus = ctx.bus;
    const on = <K extends keyof GameEvents>(k: K, fn: (p: GameEvents[K]) => void): void => {
      this.off.push(bus.on(k, fn));
    };

    on('weapon:ammo', (p) => this.onAmmo(p));
    on('weapon:switch', (p) => this.onWeaponSwitch(p.to));
    on('weapon:reload:start', (p) => this.onReloadStart(p));
    on('weapon:reload:end', () => this.onReloadEnd());
    on('weapon:ads', (p) => this.crosshair.setAds(p.active));
    on('weapon:dryfire', () => this.flashAmmo());

    // `weapon:fire` is the authoritative shot signal for reticle bloom: it is
    // emitted once per ROUND leaving the barrel (so a burst raises it once per
    // round of the burst, which is exactly how bloom accumulates), and the
    // weapon system emits it before applying that round's own bloom. A magazine
    // decrement is not equivalent — the ammo event also fires for reloads and
    // for reserve top-ups.
    on('weapon:fire', () => this.crosshair.notifyShot());

    on('player:state', (p) => this.onPlayerState(p));
    on('damage:taken', (p) => this.onDamageTaken(p));
    on('damage:dealt', (p) => this.crosshair.notifyHit(p.headshot ? 'headshot' : 'body'));
    on('entity:killed', (p) => this.onKill(p));
    on('game:score', (p) => this.onScore(p));
    on('game:pause', (p) => this.onPause(p.paused));
    on('ui:notify', (p) => this.toast(p.text, p.kind));
    on('explosion', (p) => this.onExplosion(p));

    // Radar rule: an enemy that fires unsuppressed shows up. Tracers carry a
    // world-space muzzle origin, which is precisely the "someone fired here"
    // signal — and unlike a dedicated AI event it costs the AI system nothing.
    on('tracer', (p) => this.onTracer(p));
  }

  // ── Weapon / ammo ─────────────────────────────────────────────────────────

  private onWeaponSwitch(to: WeaponId): void {
    this.weapon = to;
    this.crosshair.setWeapon(to);
    this.ammoName.textContent = prettyWeapon(to);
    // A swap cancels any reload in flight; the weapon system may or may not send
    // a matching reload:end, so clearing here keeps the ring honest.
    this.setReloading(false);
  }

  private onAmmo(p: GameEvents['weapon:ammo']): void {
    this.mag = Math.max(0, Math.round(p.mag));
    this.magSize = Math.max(0, Math.round(p.magSize));
    this.reserve = Math.max(0, Math.round(p.reserve));
    // `weapon:switch` is the only authority on what is equipped. An ammo event
    // naming a different weapon is a reserve top-up for something stowed, and
    // treating it as a switch used to swap the displayed name, reset the
    // reticle profile and silently cancel an in-flight reload ring.
    //
    // The `weapon === null` branch is a belt-and-braces bootstrap only. In
    // practice the weapon system defers its opening broadcast to its first
    // UPDATE rather than to `init`, precisely so that late-registered
    // subscribers receive it, and that broadcast emits `weapon:switch` before
    // its `weapon:ammo`. This branch therefore should not normally run; it
    // exists so the readout is never permanently blank if that ordering
    // changes.
    if (this.weapon === null) this.onWeaponSwitch(p.weapon);
    else if (this.weapon !== p.weapon) return;

    this.ammoMag.textContent = String(this.mag);
    this.ammoReserve.textContent = String(this.reserve);
    this.syncPips();

    const frac = this.magSize > 0 ? this.mag / this.magSize : 1;
    // Thresholds are on *both* fraction and absolute count: 30% of a 60-round
    // drum is still 18 rounds (not urgent), while 30% of a 5-round bolt gun is
    // one and a half (very urgent). Whichever fires first wins.
    let state = 'ok';
    if (this.mag <= 0) state = 'empty';
    else if (frac <= 0.15 || this.mag <= 3) state = 'crit';
    else if (frac <= 0.35 || this.mag <= 8) state = 'low';
    this.ammoBox.setAttribute('data-ammo', state);
  }

  private syncPips(): void {
    // Pips exist for weapons where the exact remaining count is a decision on
    // every shot — shotguns, bolt guns, sidearms. Past about a dozen the strip
    // is no longer countable at a glance and degrades into texture sitting
    // next to a number that already says the same thing, so it switches itself
    // off and the magazine count carries the information alone.
    const want = this.magSize > 0 && this.magSize <= 12 ? this.magSize : 0;
    if (want !== this.pipCount) {
      this.ammoPips.replaceChildren();
      for (let i = 0; i < want; i++) this.ammoPips.appendChild(h('i', 'nf-ammo__pip'));
      this.pipCount = want;
    }
    if (want === 0) return;
    const children = this.ammoPips.children;
    for (let i = 0; i < children.length; i++) {
      const spent = i >= this.mag;
      const node = children[i] as HTMLElement;
      // Only touch the DOM when the value actually changes — this runs on
      // every shot and a blind write would dirty the whole strip each time.
      if ((node.getAttribute('data-spent') === '1') !== spent) {
        if (spent) node.setAttribute('data-spent', '1');
        else node.removeAttribute('data-spent');
      }
    }
  }

  private onReloadStart(p: GameEvents['weapon:reload:start']): void {
    const d = Number.isFinite(p.duration) && p.duration > 0.05 ? p.duration : 2;
    this.reloadDur = d;
    this.reloadT = 0;
    this.setReloading(true);
    this.ammoName.textContent = p.tactical ? 'RELOADING · TAC' : 'RELOADING';
  }

  private onReloadEnd(): void {
    // Only a reload that actually RAN TO COMPLETION settles the weapon. The
    // same event is emitted when a reload is cancelled by a weapon swap or a
    // sprint, and the weapon system does not reset its own bloom in that case —
    // so clearing the reticle unconditionally made it under-report the real
    // cone, which is the dangerous direction to be wrong in.
    const completed = this.reloading && this.reloadT >= this.reloadDur * 0.92;
    this.setReloading(false);
    if (completed) this.crosshair.resetBloom();
    if (this.weapon !== null) this.ammoName.textContent = prettyWeapon(this.weapon);
  }

  private setReloading(v: boolean): void {
    if (this.reloading === v) return;
    this.reloading = v;
    if (v) {
      this.ammoBox.setAttribute('data-reloading', '1');
      this.ammoName.setAttribute('data-reloading', '1');
    } else {
      this.ammoBox.removeAttribute('data-reloading');
      this.ammoName.removeAttribute('data-reloading');
      this.ringFill.setAttribute('stroke-dashoffset', `${HUD.RING_C}`);
      this.crosshair.setReload(false, 0);
      if (this.weapon !== null) this.ammoName.textContent = prettyWeapon(this.weapon);
    }
  }

  /**
   * Restart a one-shot CSS animation and schedule exactly one clear for it,
   * replacing any clear still pending for the same element and attribute.
   */
  private flash(key: string, node: HTMLElement, attr: string, ms: number): void {
    const prev = this.flashTimers.get(key);
    if (prev !== undefined) {
      window.clearTimeout(prev);
      this.timers.delete(prev);
    }
    replayAnimation(node, attr);
    const id = window.setTimeout(() => {
      node.removeAttribute(attr);
      this.flashTimers.delete(key);
      this.timers.delete(id);
    }, ms);
    this.flashTimers.set(key, id);
    this.timers.add(id);
  }

  /** Dry-fire nudge: re-punch the low-ammo pulse so the empty click is seen. */
  private flashAmmo(): void {
    this.flash('dry', this.ammoBox, 'data-dry', 300);
  }

  // ── Player state / health ─────────────────────────────────────────────────

  private onPlayerState(p: GameEvents['player:state']): void {
    this.playerPos.copy(p.position);
    this.hasPlayerPos = true;

    const speed = Math.hypot(p.velocity.x, p.velocity.z);
    this.crosshair.setMotion(speed, p.grounded, p.sprinting, p.stamina);

    this.maxHealth = p.maxHealth > 0 ? p.maxHealth : 100;
    const prev = this.health;
    this.health = clamp(p.health, 0, this.maxHealth);

    if (this.health > prev + 0.01) {
      this.healthBox.setAttribute('data-regen', '1');
    } else if (this.health >= this.maxHealth - 0.01) {
      this.healthBox.removeAttribute('data-regen');
    }

    const wasAlive = this.alive;
    this.alive = this.health > 0;
    if (wasAlive && !this.alive) this.onDeath();
    else if (!wasAlive && this.alive) this.onRespawn();
  }

  private onDeath(): void {
    this.deathwash.setAttribute('data-dead', '1');
    this.deathBanner.setAttribute('data-dead', '1');
    this.crosshair.setHidden(true);
    this.threat.clear();
    this.minimap.clearPings();
  }

  private onRespawn(): void {
    this.ghostHealth = 1;
    this.deathwash.removeAttribute('data-dead');
    this.deathBanner.removeAttribute('data-dead');
    this.crosshair.setHidden(this.paused);
    this.healthBox.removeAttribute('data-regen');
    this.comboCount = 0;
    this.comboTimer = Infinity;
  }

  private onDamageTaken(p: GameEvents['damage:taken']): void {
    const ctx = this.ctx;
    if (ctx === null) return;

    // The payload direction points at the attacker in WORLD space. Convert to
    // a screen-relative bearing using only the camera's yaw: a pitch-aware
    // indicator would swing wildly when the player looks at their feet, and
    // the information the player needs is "which way do I turn", which is a
    // horizontal question.
    ctx.camera.getWorldDirection(this.tmpDir);
    const fx = this.tmpDir.x;
    const fz = this.tmpDir.z;
    const flen = Math.hypot(fx, fz) || 1;
    const nfx = fx / flen;
    const nfz = fz / flen;

    const dx0 = p.direction.x;
    const dz0 = p.direction.z;
    const dlen = Math.hypot(dx0, dz0);
    const strength = clamp(p.amount / 34, 0.28, 1);

    // A purely vertical damage vector (fall damage, a blast directly overhead)
    // carries NO bearing. The first version substituted "behind the player",
    // which pointed an arrow at nothing and posted a phantom radar contact 14m
    // astern — worse than silence, because the player acts on it. Take the hit
    // feedback, skip the direction.
    if (dlen >= 1e-5) {
      const dx = dx0 / dlen;
      const dz = dz0 / dlen;
      const forward = dx * nfx + dz * nfz;
      // Camera-right in the horizontal plane is (-forward.z, forward.x).
      const right = dx * -nfz + dz * nfx;
      this.threat.add(Math.atan2(right, forward), Math.atan2(dx, -dz), strength);

      // Best-effort radar contact from the hit direction. Range is unknown, so
      // it is placed at a nominal 14m — deliberately approximate, and it decays
      // like any other ping so a stale guess never becomes trusted intel.
      if (this.hasPlayerPos) {
        this.minimap.ping(this.playerPos.x + dx * 14, this.playerPos.z + dz * 14, 'threat');
      }
    }

    this.flash('hit', this.hitflash, 'data-fire', 170);
    this.healthBox.removeAttribute('data-regen');
  }

  // ── Combat feedback ───────────────────────────────────────────────────────

  private onKill(p: GameEvents['entity:killed']): void {
    if (p.byPlayer) {
      // Kill marker outranks the body/headshot marker raised microseconds
      // earlier by damage:dealt — see Crosshair.notifyHit for the policy.
      this.crosshair.notifyHit('kill');

      this.comboCount = this.comboTimer <= MULTI_KILL_WINDOW ? this.comboCount + 1 : 1;
      this.comboTimer = 0;
      if (this.comboCount >= 2) {
        // Index, do not clamp. Clamping announced a six-kill streak as
        // "PENTA KILL"; past the named tiers the honest answer is the number.
        const named = MULTI_KILL_NAMES[this.comboCount];
        this.queueCallout({
          title: named ?? `${this.comboCount} KILLS`,
          sub: `+${this.comboCount * 50} XP`,
        });
      }
    }
    this.addKillRow(p.byPlayer ? 'YOU' : 'HOSTILE', p.name || `TARGET ${p.id}`, p.headshot, p.byPlayer, false);
  }

  private onScore(p: GameEvents['game:score']): void {
    if (p.kills !== this.kills) {
      this.kills = p.kills;
      this.bump(this.scoreKills, String(p.kills));
    }
    if (p.deaths !== this.deaths) {
      const died = p.deaths > this.deaths;
      this.deaths = p.deaths;
      this.bump(this.scoreDeaths, String(p.deaths));
      if (died) this.addKillRow('HOSTILE', 'YOU', false, false, true);
    }
    if (p.streak !== this.streak) {
      const grew = p.streak > this.streak;
      this.streak = p.streak;
      this.bump(this.scoreStreak, String(p.streak));
      if (p.streak >= 5) this.scoreStreakCell.setAttribute('data-hot', '1');
      else this.scoreStreakCell.removeAttribute('data-hot');

      if (grew && STREAK_STEPS.includes(p.streak) && p.streak > this.lastStreakAnnounced) {
        this.lastStreakAnnounced = p.streak;
        this.queueCallout({ title: `KILLSTREAK ×${p.streak}`, sub: 'UNBROKEN' });
      }
      if (p.streak === 0) this.lastStreakAnnounced = 0;
    }
  }

  private bump(node: HTMLElement, text: string): void {
    node.textContent = text;
    const cell = node.parentElement;
    if (cell === null) return;
    this.flash(`bump:${node.parentElement === null ? '' : text}`, cell, 'data-bump', 170);
  }

  private onExplosion(p: GameEvents['explosion']): void {
    this.minimap.ping(p.position.x, p.position.z, 'boom');
    if (!this.hasPlayerPos) return;
    // Only flash for blasts that could plausibly have been felt.
    const d = this.playerPos.distanceTo(p.position);
    if (d < p.radius * 1.6) this.flash('hit', this.hitflash, 'data-fire', 170);
  }

  private onTracer(p: GameEvents['tracer']): void {
    if (!this.hasPlayerPos) return;
    const dx = p.from.x - this.playerPos.x;
    const dz = p.from.z - this.playerPos.z;
    // The player's own muzzle sits well under 3m from their origin in the
    // horizontal plane. Anything further away is somebody else shooting — and
    // anybody closer than 3m is already visible, so nothing is lost.
    if (dx * dx + dz * dz < 9) return;
    this.minimap.ping(p.from.x, p.from.z, 'fire');
  }

  // ── Killfeed ──────────────────────────────────────────────────────────────

  private addKillRow(
    killer: string,
    victim: string,
    headshot: boolean,
    mine: boolean,
    myDeath: boolean,
  ): void {
    const row = h('div', 'nf-kf__row');
    if (mine) row.setAttribute('data-mine', '1');
    if (myDeath) row.setAttribute('data-death', '1');

    const a = h('span', `ap-kf__actor${mine ? ' ap-kf__actor--you' : ''}`, killer);
    // Only the player's own kills can be attributed to a known weapon.
    const weaponId = this.weapon ?? 'ar_vector';
    const path = mine ? WEAPON_GLYPHS[weaponId] ?? WEAPON_GLYPHS.ar_vector : GLYPH_GENERIC;
    const glyph = svgIcon(GLYPH_VIEWBOX, 'nf-kf__glyph', path);
    row.append(a, glyph);
    if (headshot) row.appendChild(svgIcon('0 0 16 16', 'nf-kf__hs', GLYPH_HEADSHOT, true));
    row.appendChild(h('span', `ap-kf__actor${myDeath ? ' ap-kf__actor--victim-you' : ''}`, victim));

    this.killfeed.prepend(row);
    this.transients.push({ node: row, age: 0, life: 5, out: false });
    // Six rows is roughly the most that can be read before the top one is
    // stale; beyond that the feed is just occluding the minimap.
    while (this.killfeed.childElementCount > 6) {
      const victim = this.killfeed.lastElementChild;
      if (victim === null) break;
      this.dropTransient(victim);
    }
  }

  /** Remove a transient node and forget its ageing entry in one place. */
  private dropTransient(node: Element): void {
    node.remove();
    const i = this.transients.findIndex((t) => t.node === node);
    if (i >= 0) this.transients.splice(i, 1);
  }

  /** Age killfeed rows and toasts on the same clock as everything else. */
  private pumpTransients(dt: number): void {
    for (let i = this.transients.length - 1; i >= 0; i--) {
      const t = this.transients[i];
      t.age += dt;
      if (!t.out && t.age >= t.life) {
        t.out = true;
        t.node.setAttribute('data-out', '1');
      } else if (t.out && t.age >= t.life + 0.36) {
        t.node.remove();
        this.transients.splice(i, 1);
      }
    }
  }

  // ── Callouts / toasts ─────────────────────────────────────────────────────

  private queueCallout(c: Callout): void {
    // Queued rather than shown immediately: a multi-kill and a streak
    // milestone routinely land on the same frame, and two overlapping banners
    // are unreadable. Serialising them costs a beat and reads far better.
    if (this.calloutQueue.length >= 3) this.calloutQueue.shift();
    this.calloutQueue.push(c);
  }

  private pumpCallouts(dt: number): void {
    this.calloutCooldown -= dt;
    if (this.calloutCooldown > 0 || this.calloutQueue.length === 0) return;
    const c = this.calloutQueue.shift();
    if (c === undefined) return;
    this.calloutTitle.textContent = c.title;
    this.calloutSub.textContent = c.sub;
    // Matches the 1850ms keyframe plus a short gap so consecutive banners do
    // not butt against each other.
    this.calloutCooldown = 1.97;
    this.flash('callout', this.calloutBox, 'data-show', 1860);
  }

  private toast(text: string, kind: GameEvents['ui:notify']['kind']): void {
    const node = h('div', 'nf-toast', text);
    node.setAttribute('data-kind', kind);
    this.toasts.appendChild(node);
    this.transients.push({ node, age: 0, life: 2.6, out: false });
    while (this.toasts.childElementCount > 3) {
      const victim = this.toasts.firstElementChild;
      if (victim === null) break;
      this.dropTransient(victim);
    }
  }

  // ── Pause / visibility ────────────────────────────────────────────────────

  private onPause(paused: boolean): void {
    this.paused = paused;
    if (this.root === null) return;
    if (paused) this.root.setAttribute('data-paused', '1');
    else this.root.removeAttribute('data-paused');
    this.crosshair.setHidden(paused || !this.alive);
  }

  setVisible(v: boolean): void {
    this.visible = v;
    if (this.root === null) return;
    if (v) this.root.removeAttribute('data-hidden');
    else this.root.setAttribute('data-hidden', '1');
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Frame
  // ═══════════════════════════════════════════════════════════════════════════

  update(ctx: EngineContext, frame: FrameContext): void {
    if (this.root === null) return;
    // TWO clocks, deliberately.
    //   dt   — simulation time. Exactly 0 while the engine is paused, so
    //          anything that represents live combat state (bloom, hitmarkers,
    //          radar pings, threat arcs, killfeed ageing) freezes with the
    //          world instead of decaying behind the pause menu.
    //   uiDt — wall clock, still running while paused. Presentation
    //          transitions must use this or they cannot animate during the one
    //          state they exist to react to. Clamped so an alt-tab does not
    //          deliver a single enormous step.
    const dt = frame.dt;
    const uiDt = Math.min(frame.rawDt, 0.1);

    // Heading. 0 = world north (-Z), increasing clockwise toward east (+X).
    // The minimap uses the identical convention so the two cannot drift apart.
    ctx.camera.getWorldDirection(this.tmpDir);
    const bearing = Math.atan2(this.tmpDir.x, -this.tmpDir.z);
    const deg = ((bearing * 180) / Math.PI + 360) % 360;
    this.headingDeg = deg;

    this.crosshair.update(dt, uiDt);
    this.threat.update(dt);
    this.compass.draw(deg, this.threat.active);

    const bearingLabel = String(Math.round(deg) % 360).padStart(3, '0');
    if (this.bearingText.textContent !== bearingLabel) this.bearingText.textContent = bearingLabel;

    const px = this.hasPlayerPos ? this.playerPos.x : ctx.camera.position.x;
    const pz = this.hasPlayerPos ? this.playerPos.z : ctx.camera.position.z;
    // `camera.fov` is three.js's VERTICAL field of view; the radar cone is a
    // top-down horizontal sweep, so convert. Without this the cone reads ~30%
    // narrower than what is actually on screen at 16:9 and the radar misleads
    // the player about whether a contact is in view.
    const aspect = ctx.camera.aspect > 0.01 ? ctx.camera.aspect : 16 / 9;
    const vHalf = THREE.MathUtils.degToRad(ctx.camera.fov) * 0.5;
    const hFovDeg = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(vHalf) * aspect));
    this.minimap.setPlayer(px, pz, bearing, hFovDeg);
    this.minimap.update(dt);

    this.updateHealth(uiDt);
    this.updateReload(dt);
    this.pumpCallouts(uiDt);
    this.pumpTransients(dt);

    if (this.comboTimer !== Infinity) {
      this.comboTimer += dt;
      if (this.comboTimer > MULTI_KILL_WINDOW) this.comboTimer = Infinity;
    }
  }

  private updateHealth(dt: number): void {
    const max = this.maxHealth > 0 ? this.maxHealth : 100;
    const frac = clamp(this.health / max, 0, 1);

    // Numeric readout eases toward the true value. An instantly-snapping
    // number is hard to read while it is changing; a ~90ms ease is still
    // "immediate" to the player but stays legible mid-transition.
    this.displayHealth = damp(this.displayHealth, this.health, 0.09, dt);
    const shown = Math.round(this.health <= 0 ? 0 : Math.max(1, this.displayHealth));
    const txt = String(shown);
    if (this.healthNum.textContent !== txt) this.healthNum.textContent = txt;

    // Driven from the DAMPED value, and CSS no longer transitions this
    // property: a per-frame JS write plus a CSS transition means every frame
    // restarts the transition, so the bar chases a target it never reaches.
    // One owner per property.
    const shownFrac = clamp(this.displayHealth / max, 0, 1);
    if (Math.abs(shownFrac - this.lastFillScale) > 0.0005) {
      this.lastFillScale = shownFrac;
      this.healthFill.style.transform = `scaleX(${shownFrac.toFixed(4)})`;
    }

    // The ghost only ever catches down to the real value, never up: its whole
    // job is to show how much was just lost. 0.55/s means a 40-point hit takes
    // roughly three quarters of a second to close, which is long enough to
    // register during the fight it happened in.
    this.ghostHealth = Math.max(shownFrac, this.ghostHealth - dt * 0.55);
    const ghost = clamp(this.ghostHealth, shownFrac, 1);
    if (Math.abs(ghost - this.lastGhostScale) > 0.0005) {
      this.lastGhostScale = ghost;
      this.healthGhost.style.transform = `scaleX(${ghost.toFixed(4)})`;
    }

    const state = frac <= 0.3 ? 'crit' : frac <= 0.6 ? 'hurt' : 'ok';
    if (this.healthBox.getAttribute('data-state') !== state) {
      this.healthBox.setAttribute('data-state', state);
    }

    // Vignette. Nothing at all above 72% health — a permanently tinted frame
    // is the fastest way to make a HUD feel oppressive and to desensitise the
    // player to the signal that actually matters.
    const target = smoothstep(0, 0.72, 1 - frac);
    this.vignetteLevel = damp(this.vignetteLevel, target, 0.12, dt);
    const v = this.vignetteLevel;
    // Quantised: --dmg is an unregistered custom property inside a
    // `background`, so every distinct value is a full-screen repaint. ~125
    // steps is smooth to the eye and cuts the repaints by an order of
    // magnitude versus writing every frame.
    if (Math.abs(v - this.lastVignette) > 0.008) {
      this.lastVignette = v;
      this.vignette.style.setProperty('--dmg', v.toFixed(3));
      this.vignette.style.opacity = v < 0.004 ? '0' : Math.min(1, v * 1.25).toFixed(3);
    }
    this.threat.setBackdrop(v);

    const critical = frac > 0 && frac <= 0.3;
    if ((this.vignette.getAttribute('data-critical') === '1') !== critical) {
      if (critical) this.vignette.setAttribute('data-critical', '1');
      else this.vignette.removeAttribute('data-critical');
    }
  }

  private updateReload(dt: number): void {
    if (!this.reloading) return;
    this.reloadT += dt;
    const p = clamp(this.reloadT / this.reloadDur, 0, 1);
    this.ringFill.setAttribute('stroke-dashoffset', (HUD.RING_C * (1 - p)).toFixed(2));
    this.crosshair.setReload(true, p);
    // Failsafe: if a reload:end is ever dropped (weapon swap mid-reload,
    // death, a system that forgets to emit) the ring must not stick on screen.
    if (this.reloadT > this.reloadDur + 1.5) this.setReloading(false);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Lifecycle tail
  // ═══════════════════════════════════════════════════════════════════════════

  resize(width: number, height: number): void {
    this.crosshair.resize(width, height);
    this.threat.resize(width, height);
    this.minimap.resize(width, height);
    this.compass.resize(width, height);
  }

  dispose(): void {
    for (const un of this.off) {
      try {
        un();
      } catch {
        /* a bus already torn down is not an error worth propagating */
      }
    }
    this.off.length = 0;
    for (const id of this.timers) window.clearTimeout(id);
    this.timers.clear();
    this.flashTimers.clear();
    this.transients.length = 0;
    this.calloutQueue.length = 0;
    this.threat.clear();
    this.crosshair.dispose();
    this.minimap.dispose();
    this.threat.dispose();
    this.compass.dispose();
    this.root?.remove();
    this.root = null;
    this.ctx = null;
  }

  /** Exposed for the pause menu, which dims but must not fight `setVisible`. */
  get isVisible(): boolean {
    return this.visible;
  }

  /** Current compass heading in degrees, for any consumer that wants it. */
  get heading(): number {
    return this.headingDeg;
  }
}
