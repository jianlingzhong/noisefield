/**
 * NOISEFIELD — Dynamic reticle, hitmarkers and the sprint/stamina arc.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY CANVAS AND NOT DOM
 * ═══════════════════════════════════════════════════════════════════════════
 * The reticle changes every single frame (bloom decays continuously) and the
 * hitmarker is a sub-300ms shape animation. Doing that with DOM nodes means
 * touching style on 4-12 elements per frame, which is a guaranteed style
 * recalculation every frame on the main thread. One canvas draw of ~30 paths
 * costs a fraction of that and gives exact control over stroke geometry —
 * which a 4-blade reticle with a tapered dark outline genuinely needs.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY A SMALL CANVAS, NOT A FULL-SCREEN ONE
 * ═══════════════════════════════════════════════════════════════════════════
 * A full-screen canvas at devicePixelRatio 2 on a 4K panel is a 7680×4320
 * backing store — ~132 MB of VRAM, re-uploaded whenever it is touched, to
 * light up a few hundred pixels in the middle. This canvas is a small square
 * pinned to the centre of the viewport instead, positioned on integer pixels
 * so a 1px centre dot stays crisp rather than being resampled across two
 * device pixels.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY EVERY STROKE IS DRAWN TWICE
 * ═══════════════════════════════════════════════════════════════════════════
 * A near-white reticle over sunlit sand is invisible. Every blade is therefore
 * laid down first as a wider near-black stroke and then as the bright stroke,
 * producing a hard dark keyline. This costs one extra path per blade and is
 * the single difference between a reticle that reads on every surface in the
 * level and one that vanishes half the time.
 */

import type { WeaponId } from '../core/Types';

/** Hit feedback tiers, ordered by how much the player needs to notice them. */
export type HitKind = 'body' | 'headshot' | 'kill';

/**
 * Per-weapon reticle tuning, in *reference pixels* at a 1080p short axis.
 * Everything is multiplied by a viewport scale at draw time so the reticle
 * subtends the same visual angle from 720p to 4K.
 *
 * These deliberately do NOT import the weapon system's `Specs.ts`: the reticle
 * communicates *felt* accuracy, which is a presentation decision, and coupling
 * it to ballistics constants would mean every balance tweak silently changes
 * the HUD. `WeaponId` from `Types.ts` is the only shared contract.
 */
interface ReticleProfile {
  /** Blade gap from centre at rest. */
  gap: number;
  /** Blade length. */
  len: number;
  /** Blade thickness. */
  thick: number;
  /** Gap added per shot — the "bloom". */
  bloomPerShot: number;
  /**
   * Bloom bled off per second. Higher = snappier recovery.
   *
   * Calibrated so `bloomMax / bloomDecay` matches the weapon system's real
   * recovery time (its own `spreadPerShot * BLOOM_CAP_SHOTS / spreadDecay`).
   * The reticle is allowed to differ from ballistics in SHAPE, but it must not
   * recover FASTER than the gun does: that tells the player they are accurate
   * while they are not, which is the one direction this error must never take.
   */
  bloomDecay: number;
  /** Ceiling on accumulated bloom, so a held trigger cannot fill the screen. */
  bloomMax: number;
  /** Additional gap at full sprint. */
  moveMax: number;
  /**
   * Shotguns get four corner brackets rather than four blades: a bracket reads
   * as "area of effect", which is exactly what a pellet spread is, and it
   * stops players treating a shotgun like a precision weapon.
   */
  style: 'blades' | 'bracket';
}

const DEFAULT_PROFILE: ReticleProfile = {
  gap: 7,
  len: 11,
  thick: 2.4,
  bloomPerShot: 1.6,
  bloomDecay: 7.9,
  bloomMax: 16,
  moveMax: 9,
  style: 'blades',
};

const PROFILES: Readonly<Record<WeaponId, ReticleProfile>> = {
  // Assault rifle: the baseline. Moderate bloom, moderate recovery.
  ar_vector: { gap: 7, len: 11, thick: 2.4, bloomPerShot: 1.55, bloomDecay: 7.9, bloomMax: 16, moveMax: 9, style: 'blades' },
  // SMG: tighter at rest (its identity is hip-fire) but blooms fast because it
  // cycles at 950 rpm — the reticle must punish spraying.
  smg_wasp: { gap: 5.5, len: 9.5, thick: 2.4, bloomPerShot: 1.2, bloomDecay: 7.4, bloomMax: 15, moveMax: 10, style: 'blades' },
  // Sniper: enormous hip cone, and one shot opens it completely. The reticle is
  // the game telling you plainly that hip-firing this weapon is not a plan.
  sniper_longbow: { gap: 16, len: 13.5, thick: 2.2, bloomPerShot: 11, bloomDecay: 28, bloomMax: 27, moveMax: 14, style: 'blades' },
  shotgun_breach: { gap: 15, len: 11, thick: 2.7, bloomPerShot: 3.2, bloomDecay: 18.8, bloomMax: 18, moveMax: 11, style: 'bracket' },
  pistol_sidearm: { gap: 6, len: 9, thick: 2.2, bloomPerShot: 1.7, bloomDecay: 13.3, bloomMax: 13, moveMax: 8, style: 'blades' },
};

// ── Hitmarker timing ────────────────────────────────────────────────────────
//
// ATTACK ~40ms. Below roughly 50ms the eye does not resolve the growth as
// motion; it reads as an instantaneous snap, and that snap is what makes the
// marker feel welded to the trigger pull. Anything slower feels like lag
// between the shot and the confirmation, which is the exact feeling that makes
// a shooter's gunplay read as "mushy".
//
// HOLD 55-100ms. Just long enough for a saccade that arrives late to still
// catch it.
//
// DECAY 200-260ms. Long enough to be caught in peripheral vision while the
// player is riding recoil, short enough that consecutive hits at 700 rpm
// (86ms apart) read as a rapid stutter instead of smearing into one blob.
interface MarkerTiming { attack: number; hold: number; decay: number }
const MARKER_TIMING: Readonly<Record<HitKind, MarkerTiming>> = {
  body: { attack: 0.04, hold: 0.055, decay: 0.2 },
  headshot: { attack: 0.038, hold: 0.07, decay: 0.22 },
  kill: { attack: 0.042, hold: 0.1, decay: 0.26 },
};

const MARKER_PRIORITY: Readonly<Record<HitKind, number>> = { body: 0, headshot: 1, kill: 2 };

const COL_PRIMARY = '#f2f5f7';
/* Distinctly cooler than COL_PRIMARY. The first pass used #eaf6ff, which is
 * within a few percent of the body-hit white and was indistinguishable in
 * practice — shape alone was carrying the whole signal. */
const COL_HEADSHOT = '#a9dcff';
const COL_KILL = '#ff4436';
const COL_WARN = '#ffb454';
const COL_ACCENT = '#cfe8ff';
const COL_INK = 'rgba(3, 6, 10, 0.92)';

const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

/**
 * Frame-rate independent exponential approach. `tau` is the time in seconds to
 * close ~63% of the remaining distance; using this instead of `a += (b-a)*0.1`
 * means the animation looks identical at 30, 60 and 240 fps.
 */
const damp = (a: number, b: number, tau: number, dt: number): number =>
  b + (a - b) * Math.exp(-dt / Math.max(1e-4, tau));

export class Crosshair {
  readonly el: HTMLCanvasElement;
  private readonly g: CanvasRenderingContext2D | null;

  /** Canvas edge length in CSS pixels. */
  private side = 300;
  private dpr = 1;
  /** Reference-pixel → CSS-pixel scale, derived from the viewport short axis. */
  private s = 1;

  private profile: ReticleProfile = PROFILES.ar_vector;

  private bloom = 0;
  private adsT = 0;
  private adsTarget = 0;

  private moveT = 0;
  private moveTarget = 0;
  private sprintT = 0;
  private sprintTarget = 0;

  private stamina = 1;
  private staminaVis = 0;
  private staminaTarget = 0;
  /**
   * Largest stamina value ever observed, used to normalise the gauge.
   *
   * `PlayerStateSnapshot.stamina` has no documented range in `Types.ts`, and
   * the player system actually publishes 0..100. Hard-coding `/100` would be a
   * guess that silently breaks if that constant ever changes; hard-coding
   * `clamp(0,1)` (the original) pinned the gauge to FULL for 99% of the real
   * range, which killed the fill, the amber warning and the depleted trigger
   * outright. Self-calibrating handles 0..1, 0..100 or anything else, and
   * converges on the first frame at full stamina — which is the spawn state.
   */
  private staminaScale = 1;

  private hidden = false;
  private hideT = 0;

  private reloading = false;
  private reloadP = 0;
  private reloadVis = 0;

  /**
   * Outer radius of the reticle in CSS px, refreshed every frame. The stamina
   * arc reads it so the two can never overlap: at full sprint a shotgun's
   * corner brackets reach ~25px, which is exactly where a fixed 30px arc sat,
   * and sprinting is precisely when both are on screen. Found in review.
   */
  private extent = 0;

  private markerKind: HitKind = 'body';
  private markerT = Infinity;

  /** Pre-resolved so the hot path never re-reads the record. */
  private markerTiming: MarkerTiming = MARKER_TIMING.body;

  constructor() {
    this.el = document.createElement('canvas');
    this.el.className = 'nf-reticle';
    this.el.setAttribute('aria-hidden', 'true');
    // `desynchronized` lets the compositor skip a frame of latency on the
    // overlay; `alpha` is required because the world renders behind it.
    this.g = this.el.getContext('2d', { alpha: true, desynchronized: true });
  }

  mount(parent: HTMLElement): void {
    parent.appendChild(this.el);
  }

  // ── Inputs ────────────────────────────────────────────────────────────────

  setWeapon(id: WeaponId): void {
    this.profile = PROFILES[id] ?? DEFAULT_PROFILE;
    // A weapon swap is a clean slate: carrying the previous weapon's bloom
    // across would make the new gun feel broken for its first half second.
    this.bloom = 0;
  }

  setAds(active: boolean): void {
    this.adsTarget = active ? 1 : 0;
  }

  /**
   * @param speedXZ Horizontal speed in m/s.
   * @param grounded False while airborne — jumping wrecks accuracy in this genre and
   *   the reticle says so immediately.
   */
  setMotion(speedXZ: number, grounded: boolean, sprinting: boolean, stamina: number): void {
    // 6.2 m/s is a touch above sprint speed, so a full sprint saturates this.
    const t = clamp(speedXZ / 6.2, 0, 1);
    this.moveTarget = clamp(grounded ? t : Math.max(t, 0.85), 0, 1);
    this.sprintTarget = sprinting ? 1 : 0;
    if (Number.isFinite(stamina) && stamina > this.staminaScale) this.staminaScale = stamina;
    this.stamina = clamp(stamina / this.staminaScale, 0, 1);
    // The arc is deliberately absent at full stamina: a gauge that is always
    // full is pure noise. It exists only while it is telling you something.
    this.staminaTarget = sprinting || this.stamina < 0.985 ? 1 : 0;
  }

  /** One trigger pull. Adds bloom. */
  notifyShot(): void {
    this.bloom = Math.min(this.profile.bloomMax, this.bloom + this.profile.bloomPerShot);
  }

  /** Called on reload completion / weapon ready — accuracy is restored. */
  resetBloom(): void {
    this.bloom = 0;
  }

  /**
   * Trigger the hitmarker.
   *
   * Retrigger policy: a *stronger* marker always replaces a weaker one, so the
   * kill confirmation is never swallowed by the body hit that landed 20ms
   * later. A weaker or equal hit arriving inside the first 140ms only
   * re-punches the existing marker (rewinds it to the end of its attack)
   * rather than downgrading it. Stacking independent markers was tried first
   * and is unreadable: at 950 rpm you get six overlapping X shapes and it
   * turns into a white smear.
   */
  notifyHit(kind: HitKind): void {
    const stronger = MARKER_PRIORITY[kind] > MARKER_PRIORITY[this.markerKind];
    const stale = this.markerT > 0.14;
    if (stronger || stale) {
      this.markerKind = kind;
      this.markerTiming = MARKER_TIMING[kind];
      this.markerT = 0;
    } else {
      this.markerT = Math.min(this.markerT, this.markerTiming.attack);
    }
  }

  /**
   * Reload progress, 0..1, mirrored at the reticle.
   *
   * The corner ring alone is not enough: during a reload the only question the
   * player has is "can I shoot yet", and answering it must not cost a glance
   * away from the target they are strafing around. The stamina gauge — far less
   * time-critical — already sits at the reticle, so putting the reload timer in
   * the corner had the priority exactly backwards.
   */
  setReload(active: boolean, progress: number): void {
    this.reloading = active;
    this.reloadP = clamp(progress, 0, 1);
  }

  /** Hide during death, pause and menus. */
  setHidden(v: boolean): void {
    this.hidden = v;
  }

  // ── Layout ────────────────────────────────────────────────────────────────

  resize(vw: number, vh: number): void {
    const short = Math.min(vw, vh);
    // 1080p is the reference. The 0.85 floor deliberately over-scales below
    // 1080p: a reticle that shrinks linearly with the viewport goes
    // hairline-thin in a small window, and a reticle you cannot see is worse
    // than one that is fractionally larger than geometrically "correct". The
    // 2.1 ceiling stops it becoming a billboard on a 4K panel.
    this.s = clamp(short / 1080, 0.85, 2.1);

    // Big enough for the largest kill marker plus the stamina arc plus glow.
    const side = Math.round(clamp(short * 0.34, 260, 660) / 2) * 2;
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    this.side = side;
    this.dpr = dpr;

    // Compare against the ELEMENT, never against a cached field. A field
    // pre-initialised to a plausible value is indistinguishable from one that
    // has actually been applied, so the very first resize whose computed size
    // happened to equal that initial value skipped the work and left the
    // canvas at the 300x150 HTML default — at which point the reticle is drawn
    // outside its own backing store and vanishes completely.
    const px = Math.round(side * dpr);
    if (this.el.width !== px || this.el.height !== px) {
      this.el.width = px;
      this.el.height = px;
    }
    this.el.style.width = `${side}px`;
    this.el.style.height = `${side}px`;

    // Integer positioning: a canvas on a half-pixel is resampled by the
    // compositor and every hairline in it goes soft.
    this.el.style.left = `${Math.round((vw - side) * 0.5)}px`;
    this.el.style.top = `${Math.round((vh - side) * 0.5)}px`;
  }

  // ── Frame ─────────────────────────────────────────────────────────────────

  /**
   * @param dt Simulation delta. Zero while the engine is paused, so anything
   *   representing live combat state (bloom, hitmarker, ADS) freezes with it.
   * @param uiDt Wall-clock delta, still advancing while paused. Presentation
   *   transitions must use this: the hide-on-pause animation was previously
   *   driven by `dt`, and `damp(a, b, tau, 0) === a`, so the one state it
   *   existed to react to was the one state in which it could not run.
   */
  update(dt: number, uiDt: number): void {
    const g = this.g;
    if (g === null) return;

    // Simulation ------------------------------------------------------------
    this.bloom = Math.max(0, this.bloom - this.profile.bloomDecay * dt);

    // 70ms ADS fade. The weapon system's ADS transition is ~200-300ms; the
    // reticle intentionally leaves *first* so it is gone before the optic
    // reticle arrives, instead of the two overlapping into visual mush.
    this.adsT = damp(this.adsT, this.adsTarget, 0.07, dt);
    this.moveT = damp(this.moveT, this.moveTarget, 0.09, dt);
    this.sprintT = damp(this.sprintT, this.sprintTarget, 0.1, dt);
    this.staminaVis = damp(this.staminaVis, this.staminaTarget, 0.11, dt);
    this.reloadVis = damp(this.reloadVis, this.reloading ? 1 : 0, 0.1, uiDt);
    this.hideT = damp(this.hideT, this.hidden ? 1 : 0, 0.08, uiDt);

    if (this.markerT !== Infinity) {
      const t = this.markerTiming;
      this.markerT += dt;
      if (this.markerT > t.attack + t.hold + t.decay) this.markerT = Infinity;
    }

    // Draw ------------------------------------------------------------------
    const px = this.side * this.dpr;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, px, px);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const c = this.side * 0.5;
    const world = 1 - this.hideT;
    if (world <= 0.003) return;

    g.lineCap = 'butt';
    g.lineJoin = 'miter';

    this.drawReticle(g, c, world);
    this.drawStamina(g, c, world);
    this.drawReload(g, c, world);
    // Hitmarkers are drawn even at full ADS: confirming a hit through an optic
    // is the entire point of the mechanic, and the genre keeps them for that reason.
    this.drawMarker(g, c, world);
  }

  // ── Reticle ───────────────────────────────────────────────────────────────

  private drawReticle(g: CanvasRenderingContext2D, c: number, world: number): void {
    const p = this.profile;
    const s = this.s;

    // Blades fade out under ADS; a low-alpha centre dot survives so the player
    // still has a precise reference during the transition frames before the
    // optic is aligned.
    const bladeAlpha = world * (1 - this.adsT) * (1 - 0.5 * this.sprintT) * 0.92;
    // Stays bright under ADS. The first pass faded it to 0.35 alpha at 1.35px,
    // which on sunlit sand is genuinely invisible — and if the optic is not
    // perfectly aligned that dot is the only aiming reference there is.
    const dotAlpha = world * (0.92 - 0.12 * this.adsT);

    const spread = (p.gap + this.bloom + this.moveT * p.moveMax) * s;
    const len = p.len * s;
    const th = Math.max(1.2, p.thick * s);
    // Brackets are inset to 78% of the gap and 82% of the length, so blades are
    // always the wider of the two styles; take the blade extent as the bound.
    this.extent = spread + len + th;

    if (bladeAlpha > 0.004) {
      g.globalAlpha = bladeAlpha;
      if (p.style === 'bracket') {
        this.drawBrackets(g, c, spread, len, th);
      } else {
        this.drawBlades(g, c, spread, len, th);
      }
    }

    // Centre dot. Drawn last so it sits over any blade overlap, and with its
    // own dark ring so it survives a bright muzzle flash directly behind it.
    if (dotAlpha > 0.004) {
      // Grows slightly under ADS as it takes over from the blades.
      const r = Math.max(1.1, (1.45 + 0.45 * this.adsT) * s);
      g.globalAlpha = dotAlpha * 0.92;
      g.fillStyle = COL_INK;
      g.beginPath();
      g.arc(c, c, r + 1.4 * s, 0, Math.PI * 2);
      g.fill();
      g.globalAlpha = dotAlpha;
      // Cyan-white rather than neutral white. Sand, concrete, rust and shadow
      // are all warm or achromatic; nothing in this level is this hue, so the
      // dot cannot be camouflaged by the background it lands on.
      g.fillStyle = this.adsT > 0.5 ? COL_ACCENT : COL_PRIMARY;
      g.beginPath();
      g.arc(c, c, r, 0, Math.PI * 2);
      g.fill();
    }
    g.globalAlpha = 1;
  }

  private drawBlades(g: CanvasRenderingContext2D, c: number, gap: number, len: number, th: number): void {
    // Top blade points *up* from the gap, and so on — four independent
    // segments rather than a cross, so the gap can breathe with the bloom.
    this.seg(g, c, c - gap, c, c - gap - len, th, COL_PRIMARY);
    this.seg(g, c, c + gap, c, c + gap + len, th, COL_PRIMARY);
    this.seg(g, c - gap, c, c - gap - len, c, th, COL_PRIMARY);
    this.seg(g, c + gap, c, c + gap + len, c, th, COL_PRIMARY);
  }

  private drawBrackets(g: CanvasRenderingContext2D, c: number, gap: number, len: number, th: number): void {
    const a = gap * 0.78;
    const l = len * 0.82;
    for (let i = 0; i < 4; i++) {
      const sx = i === 0 || i === 3 ? -1 : 1;
      const sy = i < 2 ? -1 : 1;
      const x = c + sx * a;
      const y = c + sy * a;
      this.seg(g, x, y, x + sx * l, y, th, COL_PRIMARY);
      this.seg(g, x, y, x, y + sy * l, th, COL_PRIMARY);
    }
  }

  /** Stroke a segment twice: dark keyline first, bright core second. */
  private seg(
    g: CanvasRenderingContext2D,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    th: number,
    colour: string,
  ): void {
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x2, y2);

    // The keyline must stay a MINORITY of the total width. An outline as wide
    // as the core dilutes the bright stroke through anti-aliasing and the
    // whole reticle turns grey — which is exactly the failure the outline was
    // supposed to prevent. Roughly 1.3px of dark either side is enough to
    // separate it from any background without eating the core.
    const alpha = g.globalAlpha;
    g.globalAlpha = alpha * 0.9;
    g.lineWidth = th + Math.max(1.4, 1.2 * this.s);
    g.strokeStyle = COL_INK;
    g.stroke();

    g.globalAlpha = alpha;
    g.lineWidth = th;
    g.strokeStyle = colour;
    g.stroke();
  }

  // ── Stamina / sprint arc ──────────────────────────────────────────────────

  private drawStamina(g: CanvasRenderingContext2D, c: number, world: number): void {
    const vis = this.staminaVis * world;
    if (vis < 0.005) return;

    const s = this.s;
    // Sit outside whatever the reticle currently occupies, never closer than a
    // sane minimum. A constant radius collides with a bloomed or sprinting
    // reticle — and sprinting is the state that shows both at once.
    const r = Math.max(30 * s, this.extent + 9 * s);
    const half = 0.82; // ≈47° each side of straight down.
    const a0 = Math.PI / 2 - half;
    const a1 = Math.PI / 2 + half;

    g.lineCap = 'round';

    // Track. Dark, so the arc has something to read against on a bright floor.
    g.globalAlpha = vis * 0.5;
    g.lineWidth = Math.max(2.4, 3.1 * s);
    g.strokeStyle = COL_INK;
    g.beginPath();
    g.arc(c, c, r, a0, a1);
    g.stroke();

    // Fill grows from the centre outward in both directions, which reads as a
    // symmetric "reserve" gauge rather than a directional progress bar.
    const half2 = half * this.stamina;
    if (half2 > 0.004) {
      g.globalAlpha = vis * 0.95;
      g.lineWidth = Math.max(1.5, 2 * s);
      g.strokeStyle = this.stamina < 0.26 ? COL_WARN : COL_ACCENT;
      g.beginPath();
      g.arc(c, c, r, Math.PI / 2 - half2, Math.PI / 2 + half2);
      g.stroke();
    }

    g.lineCap = 'butt';
    g.globalAlpha = 1;
  }

  // ── Reload arc ────────────────────────────────────────────────────────────

  /**
   * Mirrors the reload timer at the reticle, above it, with the stamina gauge
   * below. Unlike the stamina gauge this SWEEPS from one end to the other
   * rather than growing symmetrically, because it is a countdown rather than a
   * reserve — a sweep answers "how much longer" without being read as a
   * quantity.
   */
  private drawReload(g: CanvasRenderingContext2D, c: number, world: number): void {
    const vis = this.reloadVis * world;
    if (vis < 0.005) return;

    const s = this.s;
    const r = Math.max(30 * s, this.extent + 9 * s);
    const half = 0.82;
    const a0 = -Math.PI / 2 - half;
    const a1 = -Math.PI / 2 + half;

    g.lineCap = 'round';

    g.globalAlpha = vis * 0.5;
    g.lineWidth = Math.max(2.4, 3.1 * s);
    g.strokeStyle = COL_INK;
    g.beginPath();
    g.arc(c, c, r, a0, a1);
    g.stroke();

    const p = this.reloadP;
    if (p > 0.002) {
      g.globalAlpha = vis * 0.95;
      g.lineWidth = Math.max(1.6, 2.2 * s);
      // Amber the whole way, then flicking to the ready-cyan on the last 8%:
      // "nearly done" and "done" are different decisions and deserve different
      // colours.
      g.strokeStyle = p > 0.92 ? COL_ACCENT : COL_WARN;
      g.beginPath();
      g.arc(c, c, r, a0, a0 + (a1 - a0) * p);
      g.stroke();
    }

    g.lineCap = 'butt';
    g.globalAlpha = 1;
  }

  // ── Hitmarker ─────────────────────────────────────────────────────────────

  private drawMarker(g: CanvasRenderingContext2D, c: number, world: number): void {
    const t = this.markerT;
    if (t === Infinity) return;

    const tm = this.markerTiming;
    let alpha: number;
    let scale: number;

    if (t < tm.attack) {
      // easeOutCubic on the way in, with a scale overshoot past 1. The
      // overshoot is what gives the marker its "pop"; a linear or ease-in
      // attack lands flat and feels like a UI element rather than an impact.
      const u = t / tm.attack;
      const e = 1 - Math.pow(1 - u, 3);
      alpha = e;
      scale = 0.5 + e * 0.72; // 0.50 → 1.22
    } else if (t < tm.attack + tm.hold) {
      // Settle from the overshoot back to unity across the hold.
      const u = (t - tm.attack) / tm.hold;
      alpha = 1;
      scale = 1.22 - u * 0.22;
    } else {
      // easeInQuad fade, drifting very slightly outward. The outward drift
      // reads as the marker "releasing"; a fade at constant size reads as a
      // dropped frame.
      const u = clamp((t - tm.attack - tm.hold) / tm.decay, 0, 1);
      alpha = 1 - u * u;
      scale = 1 + u * 0.16;
    }

    alpha *= world;
    if (alpha < 0.004) return;

    const s = this.s * scale;
    const kind = this.markerKind;
    g.globalAlpha = alpha;
    g.lineCap = 'round';

    if (kind === 'kill') {
      // Bigger, thicker, red. A kill has to out-shout the body hit that
      // preceded it by a few tens of milliseconds, and size is the only
      // channel that works reliably in peripheral vision — hue alone does not.
      this.marker4(g, c, 5.4 * s, 13.5 * s, Math.max(2, 3.2 * s), COL_KILL);
      // A slack ring gives the kill marker a silhouette no other marker has.
      g.globalAlpha = alpha * 0.55;
      g.lineWidth = Math.max(1.2, 1.7 * s);
      g.strokeStyle = COL_KILL;
      g.beginPath();
      g.arc(c, c, 21 * s, 0, Math.PI * 2);
      g.stroke();
    } else if (kind === 'headshot') {
      // Sharper, not louder: a SEGMENTED X — thin inner blades plus a detached
      // outer tick on each arm.
      //
      // The first attempt added four axis-aligned ticks to make an 8-point
      // star, and it failed in review for a concrete reason: the reticle's own
      // four blades are axis-aligned and sit at almost the same radius, so the
      // star's distinguishing feature landed exactly on top of the crosshair
      // and the marker became unreadable at hip fire. Keeping every element of
      // the marker on the diagonals guarantees it can never alias with the
      // reticle, whatever the current bloom.
      this.marker4(g, c, 4.4 * s, 6.5 * s, Math.max(1.3, 1.9 * s), COL_HEADSHOT);
      this.marker4(g, c, 15.5 * s, 6.5 * s, Math.max(1.2, 1.7 * s), COL_HEADSHOT);
    } else {
      this.marker4(g, c, 4.2 * s, 9.5 * s, Math.max(1.5, 2.3 * s), COL_PRIMARY);
    }

    g.lineCap = 'butt';
    g.globalAlpha = 1;
  }

  /** Four diagonal strokes forming an X with a gap in the middle. */
  private marker4(
    g: CanvasRenderingContext2D,
    c: number,
    gap: number,
    len: number,
    th: number,
    colour: string,
  ): void {
    const k = Math.SQRT1_2;
    for (let i = 0; i < 4; i++) {
      const sx = i === 0 || i === 3 ? -1 : 1;
      const sy = i < 2 ? -1 : 1;
      const x1 = c + sx * gap * k;
      const y1 = c + sy * gap * k;
      const x2 = c + sx * (gap + len) * k;
      const y2 = c + sy * (gap + len) * k;
      this.seg(g, x1, y1, x2, y2, th, colour);
    }
  }

  dispose(): void {
    this.el.remove();
    // Collapse the backing store; some drivers hold the surface otherwise.
    this.el.width = 0;
    this.el.height = 0;
  }
}
