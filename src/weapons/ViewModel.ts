/**
 * NOISEFIELD — Procedural viewmodel animation.
 *
 * No keyframe assets, no skeletons. Every pose is composed at runtime from
 * additive channels: damped springs for sway, fire kick and landing;
 * sinusoids for bob and idle breathing; eased timelines for reload, switch and
 * inspect; and a pass-through for the recoil mirror.
 *
 * WHY SPRINGS INSTEAD OF LERPS
 * ----------------------------
 * A `lerp(current, target, k)` is an exponential decay: it can only ever
 * approach the target, never overshoot, and its apparent speed is tied to
 * framerate unless you fight it. A damped spring has MASS. Under-damp it
 * (ζ ≈ 0.55–0.78 here) and it overshoots and settles, which is precisely what
 * a two-and-a-half kilogram lump of steel does when you stop turning. That
 * settle is 90% of why a viewmodel feels weighty rather than glued to the
 * camera, and it cannot be faked with a lerp. The one exception is the sway
 * spring while aiming, which is driven to ζ = 1.0: an overshooting wobble on a
 * sight picture reads as broken rather than as weight.
 *
 * CHANNEL COMPOSITION
 * -------------------
 * Final transform = base pose (hip↔ADS) + Σ additive channels:
 *   sprint pose · sway · bob · idle breathing · fire kick · landing ·
 *   reload timeline · switch timeline · inspect timeline · recoil mirror
 * Every channel writes into its own vector and they are summed once per frame,
 * so channels never fight over the transform and any one of them can be scaled
 * to zero (ADS scales sway/bob/idle down by ~80%) without special cases.
 */

import * as THREE from 'three';
import type { EngineContext, WeaponId, WeaponSpec } from '../core/Types';
import {
  buildWeaponRig,
  cloneMagazineProp,
  disposeRig,
  hasForeignViewLight,
  installViewLighting,
  muzzleFlashTexture,
  type WeaponRig,
} from './WeaponModels';

// ─────────────────────────────────────────────────────────────────────────────
// Per-weapon handling
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Feel parameters that are about the ANIMATION rather than about the ballistics
 * (which live in WeaponSpec). Kept here because they are tuned by looking at
 * the screen, not by reading a damage table.
 */
interface Handling {
  /**
   * Inertia. Divides the sway target, so a HEAVIER weapon swings LESS: the
   * sniper (1.85) trails the camera by 54% of the reference amount, the pistol
   * (0.52) by 192%. It is the wrist that is being modelled — a heavy rifle is
   * braced against the shoulder and barely moves, a pistol whips around.
   */
  mass: number;
  /** Multiplier on the idle breathing figure-of-eight. */
  idleSway: number;
  /** Multiplier on walk/sprint bob. */
  bobScale: number;
  /** Per-shot rearward translation impulse, m/s. */
  kickBack: number;
  /** Per-shot vertical translation impulse, m/s. */
  kickUp: number;
  /** Per-shot muzzle-rise impulse, rad/s. */
  kickPitch: number;
  /** Per-shot random roll impulse, rad/s. Also drives yaw, at 45% of this. */
  kickRoll: number;
  /** Weapon raise/lower time when switching, seconds. */
  swapTime: number;
  /** Distance from the eye to the sight when aiming, metres. */
  adsDepth: number;
  /** Muzzle flash size in metres. */
  flashScale: number;
  /** Muzzle flash point-light intensity. */
  flashLight: number;
  /** Shell ejection delay — non-zero for manually cycled actions. */
  ejectDelay: number;
}

const HANDLING: Record<WeaponId, Handling> = {
  ar_vector: {
    mass: 1.00,
    idleSway: 1.00,
    bobScale: 1.00,
    kickBack: 0.62,
    kickUp: 0.16,
    kickPitch: 3.1,
    kickRoll: 1.5,
    swapTime: 0.62,
    adsDepth: 0.195,
    flashScale: 0.115,
    flashLight: 16,
    ejectDelay: 0,
  },
  smg_wasp: {
    mass: 0.72, // light: swings 39% wider than the rifle (mass divides sway)
    idleSway: 0.78,
    bobScale: 0.90,
    kickBack: 0.44,
    kickUp: 0.12,
    kickPitch: 2.5,
    kickRoll: 2.1, // ...but rolls more, because there is less mass to resist it
    swapTime: 0.50,
    adsDepth: 0.180,
    flashScale: 0.095,
    flashLight: 12,
    ejectDelay: 0,
  },
  sniper_longbow: {
    mass: 1.85, // heaviest thing in the kit
    idleSway: 3.40, // HUGE unsupported wander — the whole reason to go prone
    bobScale: 1.35,
    kickBack: 1.55,
    kickUp: 0.34,
    kickPitch: 7.4,
    kickRoll: 1.1,
    swapTime: 0.86,
    adsDepth: 0.115, // scope eye relief — much closer than a reflex
    flashScale: 0.185,
    flashLight: 34,
    ejectDelay: 0.34, // shell leaves on the bolt cycle, not on the shot
  },
  shotgun_breach: {
    mass: 1.45,
    idleSway: 1.20,
    bobScale: 1.12,
    kickBack: 1.30,
    kickUp: 0.28,
    kickPitch: 6.2,
    kickRoll: 1.8,
    swapTime: 0.70,
    adsDepth: 0.190,
    flashScale: 0.175,
    flashLight: 28,
    ejectDelay: 0.20, // ejects as the pump comes back
  },
  pistol_sidearm: {
    mass: 0.52,
    idleSway: 0.70,
    bobScale: 0.78,
    kickBack: 0.40,
    kickUp: 0.14,
    kickPitch: 3.6,
    kickRoll: 1.6,
    swapTime: 0.34, // fast swap is the sidearm's entire reason to exist
    adsDepth: 0.215,
    flashScale: 0.085,
    flashLight: 11,
    ejectDelay: 0,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Easing
// ─────────────────────────────────────────────────────────────────────────────

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}
/** Normalised progress through a sub-range of a timeline. */
function seg(u: number, a: number, b: number): number {
  return clamp01((u - a) / Math.max(1e-6, b - a));
}
function easeOutCubic(t: number): number {
  const i = 1 - t;
  return 1 - i * i * i;
}
function easeInCubic(t: number): number {
  return t * t * t;
}
function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
/** Rises to 1 and returns to 0 — one smooth "there and back" pulse. */
function pulse(t: number): number {
  return Math.sin(clamp01(t) * Math.PI);
}
/** Overshooting settle, for a magazine slamming home. */
function easeOutBack(t: number): number {
  const c1 = 1.9;
  const c3 = c1 + 1;
  const i = t - 1;
  return 1 + c3 * i * i * i + c1 * i * i;
}

// ─────────────────────────────────────────────────────────────────────────────
// Springs
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Normalised timing of the reload animation's beats, as fractions of the total
 * reload duration.
 *
 * SINGLE SOURCE OF TRUTH. `stepReload` reads these, and `WeaponSystem`
 * re-exports them so the audio system can schedule `reload_magout` /
 * `reload_magin` / `reload_charge` against the `duration` carried by
 * 'weapon:reload:start' without re-deriving private constants that would then
 * silently drift out of sync the first time the animation is retuned.
 */
export const RELOAD_BEATS = {
  /** Weapon finishes canting inboard; the shooter is looking at the magwell. */
  cantIn: 0.16,
  /** Magazine release — the old magazine falls free. */
  magOut: { tactical: 0.17, empty: 0.20 },
  /** New magazine begins rising into the well. */
  magInStart: { tactical: 0.42, empty: 0.46 },
  /** New magazine seats. */
  magIn: { tactical: 0.60, empty: 0.64 },
  /** Support hand is back on the weapon. */
  handBack: 0.62,
  /** Charging handle starts moving. Empty reloads only. Must FOLLOW magIn. */
  chargeStart: 0.66,
  /** Charging handle pulled fully rearward. Empty reloads only. */
  chargeBack: 0.74,
  /** Bolt released, slamming forward. Empty reloads only. */
  chargeRelease: 0.80,
  /** Weapon starts returning to the ready. Must follow every mechanical beat. */
  returnStart: 0.84,
} as const;

/**
 * Target substep for spring integration. The real step is `dt / ceil(dt·180)`
 * capped at `MAX_SPRING_STEPS`, so it is bounded rather than fixed — 5.6 ms at
 * 60 fps, 3.5 ms at 144 fps, growing past 1/180 s only once a frame exceeds
 * 33 ms. Bounding it is what stops the springs stiffening as frames lengthen;
 * semi-implicit Euler is never exactly framerate-invariant, but across the
 * range that matters this is indistinguishable.
 */
const SPRING_STEP = 1 / 180;
const MAX_SPRING_STEPS = 6;

/** Damped harmonic oscillator on a scalar. */
class Spring {
  value = 0;
  velocity = 0;
  stiffness: number;
  damping: number;

  constructor(stiffness: number, zeta: number) {
    this.stiffness = stiffness;
    this.damping = 2 * zeta * Math.sqrt(stiffness);
  }

  step(dt: number, target: number): number {
    // Semi-implicit Euler is only conditionally stable, so substep. Six 1/180 s
    // substeps covers a 30 ms frame at the stiffest constants used here.
    const steps = Math.min(MAX_SPRING_STEPS, Math.max(1, Math.ceil(dt / SPRING_STEP)));
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const a = -this.stiffness * (this.value - target) - this.damping * this.velocity;
      this.velocity += a * h;
      this.value += this.velocity * h;
    }
    return this.value;
  }

  impulse(v: number): void {
    this.velocity += v;
  }
}

/** Three independent springs sharing constants; used for position and Euler. */
class Spring3 {
  readonly value = new THREE.Vector3();
  readonly velocity = new THREE.Vector3();
  stiffness: number;
  damping: number;

  constructor(stiffness: number, zeta: number) {
    this.stiffness = stiffness;
    this.damping = 2 * zeta * Math.sqrt(stiffness);
  }

  configure(stiffness: number, zeta: number): void {
    this.stiffness = stiffness;
    this.damping = 2 * zeta * Math.sqrt(stiffness);
  }

  step(dt: number, tx: number, ty: number, tz: number): void {
    const steps = Math.min(MAX_SPRING_STEPS, Math.max(1, Math.ceil(dt / SPRING_STEP)));
    const h = dt / steps;
    const k = this.stiffness;
    const c = this.damping;
    const v = this.value;
    const w = this.velocity;
    for (let i = 0; i < steps; i++) {
      w.x += (-k * (v.x - tx) - c * w.x) * h;
      w.y += (-k * (v.y - ty) - c * w.y) * h;
      w.z += (-k * (v.z - tz) - c * w.z) * h;
      v.x += w.x * h;
      v.y += w.y * h;
      v.z += w.z * h;
    }
  }

  impulse(x: number, y: number, z: number): void {
    this.velocity.x += x;
    this.velocity.y += y;
    this.velocity.z += z;
  }

  reset(): void {
    this.value.set(0, 0, 0);
    this.velocity.set(0, 0, 0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Input snapshot
// ─────────────────────────────────────────────────────────────────────────────

/** Reused every frame by WeaponSystem — never allocate one of these in update. */
export interface ViewModelState {
  /** Horizontal speed, m/s. */
  speed: number;
  /**
   * Eased 0..1 low-ready blend, ALREADY INTEGRATED by WeaponSystem — the same
   * value its firing gate uses. Passed in rather than re-derived here so the
   * visible pose and the gate are the same number by construction, instead of
   * two integrators that agree only as long as nobody edits one of them.
   */
  lowReady: number;
  crouching: boolean;
  grounded: boolean;
  /** Eased 0..1 ADS progress. */
  ads: number;
  /** Look angular velocity this frame, rad/s. */
  lookYawRate: number;
  lookPitchRate: number;
  /** Current camera recoil offset, mirrored into the viewmodel. */
  viewKickPitch: number;
  viewKickYaw: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Falling magazine props
// ─────────────────────────────────────────────────────────────────────────────

interface MagProp {
  obj: THREE.Object3D;
  vel: THREE.Vector3;
  spin: THREE.Vector3;
  life: number;
}

/**
 * Offset the magazine travels through on its way in and out, relative to its
 * seated position.
 *
 * The Z component is FORWARD (negative). Backward looks natural on a rifle,
 * where there is nothing behind the magwell, but every weapon here has its
 * grip and trigger guard immediately aft of the magazine — a rearward arc
 * drives a shotgun shell 9 mm into the grip and a bolt-gun magazine 6 mm
 * through the trigger guard.
 */
const MAG_ARC = { x: 0.026, y: -0.150, z: -0.030 } as const;

/** How long an interrupted inspect takes to fade out of the pose. */
const INSPECT_FADE = 0.17;
/** How long an interrupted bolt / slide / pump stroke takes to ease home. */
const BOLT_SETTLE = 0.13;
/** How long an aborted reload takes to unwind its cant. */
const RELOAD_FADE = 0.18;

/**
 * Hip cant, in radians, blended out completely as the weapon comes to the eye.
 *
 * WHY THE WEAPON IS NOT AXIS-ALIGNED AT THE HIP. A viewmodel whose bore is
 * exactly parallel to the view axis reads as a prop bolted to the camera,
 * because nothing about a human holding a rifle produces that. A real hip carry
 * converges: the weapon sits to the shooter's right and below the eyeline, and
 * the muzzle is brought back INBOARD toward where they are looking, with the
 * receiver rolled slightly so the ejection-port flank turns toward the camera.
 * Three or four degrees on each axis is all it takes — enough that the top rail
 * and the right flank are both visible (which is what gives the receiver its
 * three-dimensional read) and far too little to look like a canted stance.
 *
 * The yaw carries a second job. At 60° of vertical FOV a rifle pointed straight
 * down the view axis is savagely foreshortened: the 20 cm of barrel and muzzle
 * device forward of the handguard projects into barely 20 px, so the front of
 * the weapon simply is not readable. Six degrees of inboard yaw swings the
 * muzzle roughly 60 mm across the frame, which separates it from the handguard
 * and lets the whole length of the weapon read — the same reason a photographer
 * shooting a rifle for a catalogue never shoots it end-on.
 *
 *   x  +0.036  ≈ 2.1° muzzle UP, converging on the crosshair from below.
 *   y  +0.078  ≈ 4.5° muzzle LEFT. Positive yaw rotates a −Z forward axis
 *               toward −X, i.e. inboard from the weapon's position on the
 *               right. The previous value was NEGATIVE, which splayed the
 *               muzzle outward, away from the centre of the screen.
 *   z  +0.058  ≈ 3.3° of roll, top toward the screen centre.
 */
const HIP_CANT = { x: 0.036, y: 0.078, z: 0.058 } as const;

const MAX_MAG_PROPS = 3;
/**
 * Viewmodel space is not world space: a magazine falling at a true 9.81 m/s²
 * leaves the 30 cm-tall frame in under a fifth of a second and reads as a
 * glitch. Scaling gravity down keeps it on screen long enough to register.
 */
const PROP_GRAVITY = 4.2;

// ─────────────────────────────────────────────────────────────────────────────
// Scratch
// ─────────────────────────────────────────────────────────────────────────────

const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();

// ─────────────────────────────────────────────────────────────────────────────

export class ViewModel {
  private readonly ctx: EngineContext;

  /** Rigs are cached: switching weapons must not rebuild geometry. */
  private readonly rigs = new Map<WeaponId, WeaponRig>();
  private rig: WeaponRig | null = null;
  private spec: WeaponSpec | null = null;
  private handling: Handling = HANDLING.ar_vector;

  // ── pose channels ──────────────────────────────────────────────────────────
  private readonly basePos = new THREE.Vector3();
  private readonly baseRot = new THREE.Vector3();
  private readonly adsPos = new THREE.Vector3();
  private readonly hipPos = new THREE.Vector3();
  private readonly outPos = new THREE.Vector3();
  private readonly outRot = new THREE.Vector3();
  /**
   * Accumulator for the reload/switch/inspect timelines. These MUST NOT be the
   * shared module scratch: `stepEject` runs in the same frame and would
   * clobber a half-composed pose.
   */
  private readonly tlPos = new THREE.Vector3();
  private readonly tlRot = new THREE.Vector3();
  /** Last eased ADS value, so `onFire` can damp the kick without a parameter. */
  private lastAds = 0;

  private readonly swayPos = new Spring3(105, 0.78);
  private readonly swayRot = new Spring3(96, 0.72);
  private readonly kickPos = new Spring3(520, 0.62);
  private readonly kickRot = new Spring3(440, 0.58);
  private readonly land = new Spring(190, 0.55);

  private bobPhase = 0;
  private idlePhase = 0;

  // ── timelines ──────────────────────────────────────────────────────────────
  private reloadTime = -1;
  private reloadDur = 0;
  private reloadTactical = false;
  /**
   * Reload pose weight, 0..1, and the value it is easing toward.
   *
   * A level rather than a one-way timer because the reload is the only channel
   * that can be RE-ENTERED while it is unwinding: abort with a tap of sprint,
   * then press reload again. A one-way fade had to be cleared to restart, which
   * dropped ~70 mm and 36° of pose in a single frame — the exact discontinuity
   * the fade exists to prevent. With a level, re-committing simply reverses it.
   */
  private reloadLevel = 1;
  private reloadTarget = 1;
  private magDropped = false;
  private magReinserted = false;

  private switchTime = -1;
  private switchDur = 0;
  private switchCallback: (() => void) | null = null;

  private inspectTime = -1;
  private readonly inspectDur = 1.7;
  /** ≥0 while an interrupted inspect is fading out; see `interrupt`. */
  private inspectFade = -1;

  private triggerTime = -1;
  private boltTime = -1;
  private boltDur = 0.06;
  /** ≥0 while an interrupted bolt stroke eases home; see `settleBolt`. */
  private boltSettle = -1;
  private readonly boltSettleFrom = new THREE.Vector3();

  // ── muzzle flash ───────────────────────────────────────────────────────────
  private readonly flashGroup = new THREE.Group();
  private flashTime = -1;
  private flashDur = 0.045;
  /** Per-shot flash shape, randomised in `onFire`, applied in `stepFlash`. */
  private flashSize = 1;
  private flashStretch = 1;
  private viewLight: THREE.PointLight | null = null;
  private worldLight: THREE.PointLight | null = null;

  // ── shell ejection ─────────────────────────────────────────────────────────
  private ejectTimer = -1;
  /** Cases still owed, so a multi-shot catch-up frame does not lose any. */
  private ejectPending = 0;

  // ── props ──────────────────────────────────────────────────────────────────
  private readonly props: MagProp[] = [];

  // ── cached rig-local anchors, refreshed on weapon change ───────────────────
  private readonly opticLocal = new THREE.Vector3();
  private readonly magHome = new THREE.Vector3();
  private readonly leftArmHome = new THREE.Vector3();
  private readonly chargeHome = new THREE.Vector3();
  private readonly boltHome = new THREE.Vector3();
  private triggerHomeX = 0;

  private lights: THREE.Object3D[] = [];
  /**
   * Set once the deferred double-lighting check has run. See
   * `hasForeignViewLight`: the engine's own viewmodel rig is installed after
   * this class's constructor, so the only place the conflict is visible is the
   * first frame of the update loop.
   */
  private lightingReconciled = false;
  /** True only if we installed viewScene.environment, so dispose is safe. */
  private ownsEnvironment = false;
  private rngSeed = 0x51ed270b;

  constructor(ctx: EngineContext) {
    this.ctx = ctx;
    const hadEnvironment = ctx.viewScene.environment !== null;
    this.lights = installViewLighting(ctx);
    this.ownsEnvironment = !hadEnvironment && ctx.viewScene.environment !== null;
    this.buildFlash();
  }

  // ── weapon lifecycle ───────────────────────────────────────────────────────

  private rigFor(id: WeaponId): WeaponRig {
    let r = this.rigs.get(id);
    if (!r) {
      r = buildWeaponRig(id, this.ctx);
      this.rigs.set(id, r);
    }
    return r;
  }

  /** Swap the visible weapon immediately, with no transition. */
  setWeapon(id: WeaponId, spec: WeaponSpec): void {
    this.inspectTime = -1;
    this.inspectFade = -1;
    if (this.rig) this.ctx.viewScene.remove(this.rig.root);

    const rig = this.rigFor(id);
    this.rig = rig;
    this.spec = spec;
    this.handling = HANDLING[id];
    this.ctx.viewScene.add(rig.root);

    // Rest transforms come from the rig's BUILD-time capture, never from the
    // live scene graph: re-selecting a weapon whose parts happen to be
    // mid-animation would otherwise bake that pose in as the new rest, and
    // every later "restore" would compound the error.
    // Hard reset: the model itself is changing, so there is nothing to ease.
    this.boltSettle = -1;
    this.lastAds = 0;
    this.ejectPending = 0;
    this.ejectTimer = -1;
    this.flashTime = -1;
    this.magHome.copy(rig.rest.magazine);
    this.leftArmHome.copy(rig.rest.leftArm);
    this.chargeHome.copy(rig.rest.chargingHandle);
    this.boltHome.copy(rig.rest.bolt);
    this.triggerHomeX = rig.rest.triggerX;
    this.resetParts(true);

    // ── Solve the ADS pose from the OPTIC, not from a magic number ───────────
    // Place the rig so the sight's optical axis lands exactly on the view
    // camera's axis at the weapon's eye-relief distance. Because opticAim is
    // authored on the true optical axis of the sight, the dot then projects to
    // the exact centre of the screen at any FOV or aspect ratio — which no
    // hand-tuned offset can promise once someone changes the viewmodel FOV.
    rig.root.position.set(0, 0, 0);
    rig.root.rotation.set(0, 0, 0);
    rig.root.updateMatrixWorld(true);
    rig.opticAim.getWorldPosition(this.opticLocal);
    rig.root.worldToLocal(this.opticLocal);
    this.adsPos.set(-this.opticLocal.x, -this.opticLocal.y, -this.handling.adsDepth - this.opticLocal.z);

    this.hipPos.set(spec.hipPosition[0], spec.hipPosition[1], spec.hipPosition[2]);

    // `spec.adsPosition` is the hand-authored NOMINAL aim pose, and serves as
    // the fallback. The solved value is authoritative — it is the only thing
    // that guarantees the reticle lands on the mathematical centre of the
    // screen at any FOV — but a rig whose optic node was authored in the wrong
    // place would park the weapon somewhere absurd with no obvious cause, so
    // sanity-check against the nominal pose and fall back loudly instead of
    // shipping a weapon the player cannot aim.
    _v0.set(spec.adsPosition[0], spec.adsPosition[1], spec.adsPosition[2]);
    const plausible =
      Number.isFinite(this.adsPos.x) &&
      Number.isFinite(this.adsPos.y) &&
      Number.isFinite(this.adsPos.z) &&
      this.adsPos.distanceTo(_v0) < 0.25;
    if (!plausible) {
      console.warn(
        `[weapons] optic node for "${id}" solves to an implausible ADS pose; ` +
          "falling back to spec.adsPosition. Check the rig's opticAim placement.",
      );
      this.adsPos.copy(_v0);
    }

    // Leave the rig at the hip pose rather than at the origin it was zeroed to
    // for the optic solve. `update` normally overwrites this immediately, but
    // it is skipped while the game is paused (dt === 0) — so a session that
    // boots straight into a pause menu would otherwise show the weapon
    // centred on and clipping through the camera until the player unpauses.
    rig.root.position.copy(this.hipPos);
    rig.root.rotation.set(HIP_CANT.x, HIP_CANT.y, HIP_CANT.z);
    // Same reason: `update` owns reticle visibility but is skipped while the
    // game is paused, and the reticle ignores depth — so a session that boots
    // into a pause menu would draw the dot through the hip-posed weapon.
    if (rig.reticle) rig.reticle.visible = false;

    // Re-parent the flash rig and its light to the new muzzle.
    rig.muzzle.add(this.flashGroup);
    if (this.viewLight) rig.muzzle.add(this.viewLight);

    this.cancelReload();
    this.kickPos.reset();
    this.kickRot.reset();
  }

  /**
   * Lower the current weapon off the bottom of the screen, swap, raise. The
   * callback fires at the bottom of the arc so the model change is hidden.
   */
  beginSwitch(onSwap: () => void, duration?: number): void {
    // Floor the duration: a zero would make `u` Infinity on the first frame.
    // `Math.max` passes NaN through, and a NaN duration would make `u` NaN,
    // permanently freeze the switch and never fire the swap callback.
    const requested = duration ?? this.handling.swapTime;
    this.switchDur = Number.isFinite(requested) ? Math.max(0.05, requested) : this.handling.swapTime;
    this.switchTime = 0;
    this.switchCallback = onSwap;
    this.cancelReload();
    // Fade, do NOT hard-clear. The switch arc's first frame moves the weapon
    // by well under a millimetre, so it masks nothing — clearing the inspect
    // here discarded up to 88.8° of yaw in a single frame. The 0.17 s fade
    // completes before the model swap at half the swap time on every weapon.
    this.interrupt();
    this.settleBolt();
  }

  get switching(): boolean {
    return this.switchTime >= 0;
  }

  // ── actions ────────────────────────────────────────────────────────────────

  beginReload(duration: number, tactical: boolean): void {
    // Always a FRESH timeline. Resuming a half-unwound one keeps the pose
    // continuous but strands the animation against `WeaponSystem`'s timer,
    // which restarts unconditionally — up to 3.2 s of the weapon sitting at the
    // ready unable to fire. Callers instead wait out the ≤180 ms unwind (see
    // `reloadUnwinding`), which keeps both in step.
    // `Math.max` is NaN-transparent, same trap as `beginSwitch`.
    this.reloadDur = Number.isFinite(duration) ? Math.max(0.2, duration) : 1.5;
    this.reloadTactical = tactical;
    this.reloadTarget = 1;
    this.reloadTime = 0;
    this.reloadLevel = 1;
    this.magDropped = false;
    this.magReinserted = false;
    // Deliberately does NOT clear `inspectTime`. Callers pair this with
    // `interrupt()`, which sets the inspect to its own ease-out; clearing it
    // here would discard that and teleport the weapon back. Both timelines are
    // additive into tlPos/tlRot, so the ease-out composes with the reload cant.
    // Only the parts the RELOAD timeline drives — the last round of the
    // magazine auto-reloads from inside its own shot, so a full reset here
    // would cancel the trigger and bolt animation that shot just started.
    this.restoreReloadParts();
  }

  cancelReload(): void {
    if (this.reloadTime < 0) {
      this.resetParts();
      return;
    }
    // Unwind rather than hard-clear. The cant envelope is flat at 1.0 across
    // 66% of the timeline, so clearing it discards 82 mm of translation and
    // 35.5° of rotation in one frame — and both callers (sprint-abort and
    // weapon switch) hit it with the weapon still square on screen.
    this.reloadTarget = 0;

    // If the magazine had already dropped, bring it back NOW at the bottom of
    // its arc so the unwind can carry it home. Left hidden it would leave an
    // empty magwell for the whole 180 ms and then pop a full magazine in on a
    // single frame — the abort is a cancelled reload, so the magazine is
    // supposed to still be in the weapon.
    const rig = this.rig;
    if (rig && this.magDropped && !this.magReinserted) {
      this.magReinserted = true;
      rig.magazine.visible = true;
      rig.magazine.position.set(
        this.magHome.x + MAG_ARC.x,
        this.magHome.y + MAG_ARC.y,
        this.magHome.z + MAG_ARC.z,
      );
    }
  }

  /**
   * Return every animated sub-part to its rest transform and clear the timers
   * that drive them.
   *
   * Called wherever an animation can be abandoned mid-flight (reload cancel,
   * weapon switch, weapon set). The trigger blade in particular has no other
   * path home: its 90 ms animation is currently always shorter than the
   * shortest swap, but relying on that timing margin rather than on logic is
   * how a part ends up permanently displaced the first time someone retunes a
   * swap speed.
   */
  /**
   * Restore only the parts the RELOAD timeline drives.
   *
   * Used at reload end and at abort completion, in place of the full
   * `resetParts`. It must not touch the trigger or the bolt, because a reload
   * finishes on precisely the frame a held or buffered trigger fires: the
   * accumulator is pinned at zero for the whole reload, so `canFire` and
   * `u >= 1` come true together, `onFire` runs first, and a full reset would
   * silently cancel the cycle it had just started.
   */
  private restoreReloadParts(): void {
    const rig = this.rig;
    if (!rig) return;
    rig.magazine.visible = true;
    rig.magazine.position.copy(this.magHome);
    rig.leftArm.position.copy(this.leftArmHome);
    if (rig.chargingHandle && rig.chargingHandle !== rig.bolt) {
      rig.chargingHandle.position.copy(this.chargeHome);
    }
    if (rig.bolt && this.boltTime < 0 && this.boltSettle < 0) {
      // Nothing else is animating it, so seat it. Covers both the shared
      // handle/bolt weapons and the rifle's separately-driven carrier.
      rig.bolt.position.copy(this.boltHome);
    }
  }

  private resetParts(hard = false): void {
    const rig = this.rig;
    if (!rig) return;
    rig.magazine.visible = true;
    rig.magazine.position.copy(this.magHome);
    rig.leftArm.position.copy(this.leftArmHome);
    if (rig.trigger) rig.trigger.rotation.x = this.triggerHomeX;
    this.triggerTime = -1;

    if (hard) {
      // The model itself is being replaced; there is nothing to ease.
      if (rig.chargingHandle) rig.chargingHandle.position.copy(this.chargeHome);
      if (rig.bolt) rig.bolt.position.copy(this.boltHome);
      this.boltTime = -1;
      this.boltSettle = -1;
      return;
    }

    // Settle FIRST, and never snap the charging handle when it is the SAME
    // OBJECT as the bolt — which it is on the shotgun, the bolt gun and the
    // pistol, i.e. exactly the three longest strokes. Snapping it first moved
    // the part to rest before `settleBolt` could read where it actually was,
    // so the eased return measured zero displacement and silently did nothing.
    // On the shotgun that is a 60 mm teleport of the pump AND the support arm
    // parented to it.
    this.settleBolt();
    if (rig.chargingHandle && rig.chargingHandle !== rig.bolt) {
      rig.chargingHandle.position.copy(this.chargeHome);
    }
  }

  /**
   * Ease a mid-stroke bolt / slide / pump back to rest instead of snapping it.
   *
   * The trigger blade is 0.38 rad over 90 ms and nobody would notice a snap;
   * the shotgun's pump is SIXTY MILLIMETRES over 400 ms, and the support arm is
   * parented to it, so a hard reset teleports the pump and the shooter's whole
   * left hand across the screen in one frame.
   */
  private settleBolt(): void {
    const rig = this.rig;
    this.boltTime = -1;
    this.boltSettle = -1;
    if (!rig?.bolt) return;
    if (rig.bolt.position.distanceToSquared(this.boltHome) < 1e-9) return;
    this.boltSettleFrom.copy(rig.bolt.position);
    this.boltSettle = 0;
  }

  get reloading(): boolean {
    return this.reloadTime >= 0;
  }

  /**
   * True while an aborted reload is still unwinding its pose. A new reload
   * must not start during this window — see `beginReload`.
   */
  get reloadUnwinding(): boolean {
    return this.reloadTime >= 0 && this.reloadTarget === 0;
  }

  inspect(): void {
    // Not while aiming: the reticle is hidden for the duration of an inspect,
    // so starting one from a sight picture would blank the dot for 1.7 s.
    if (this.reloading || this.switching || this.lastAds > 0.2) return;
    // Nor while one is already running or fading: restarting the timeline
    // snaps the envelope from wherever it is back to 0, which at the peak is
    // 96° of rotation and 94 mm of translation in a single frame.
    if (this.inspectTime >= 0) return;
    this.inspectTime = 0;
    this.inspectFade = -1;
  }

  get inspecting(): boolean {
    return this.inspectTime >= 0;
  }

  /**
   * Interrupt a showy animation because something more urgent happened.
   *
   * FREEZES the inspect where it is and fades its contribution out over
   * `INSPECT_FADE`. Two tempting alternatives are both wrong: clearing
   * `inspectTime` teleports the weapon back from up to 89° of yaw in a single
   * frame, and skipping to the timeline's own ease-out does exactly the same
   * thing, because that segment begins from a pose that is already ≈0. The only
   * continuous exit is to scale down from the current value.
   */
  interrupt(): void {
    if (this.inspectTime < 0 || this.inspectFade >= 0) return;
    this.inspectFade = 0;
  }

  onLand(impactVelocity: number): void {
    // Impulse scales with the square root of the fall so a small hop still
    // registers while a four-storey drop does not fold the weapon in half.
    const t = clamp01(Math.abs(impactVelocity) / 14);
    this.land.impulse(-0.55 * Math.sqrt(t));
  }

  /**
   * One shot fired. Drives the kick springs, the flash, the bolt/slide cycle,
   * the trigger blade and the shell ejection.
   */
  onFire(ads: number): void {
    // Firing ends an inspect. Without this the showy animation keeps playing
    // over the shot, and — because the reticle is hidden while inspecting —
    // the player can be fully aimed in with no dot on screen.
    this.interrupt();
    const h = this.handling;
    // Passed in rather than read from the last `update`: firing happens before
    // the viewmodel updates, so a stored value would be a frame stale on
    // exactly the frames where the aim state is changing.
    this.lastAds = clamp01(ads);
    const rnd = this.random();
    const rnd2 = this.random();

    // ADS damps the VISUAL kick hard: a sight that leaves the screen on every
    // round is unusable, and real shooters brace far harder when aiming.
    const adsDamp = 1 - 0.55 * this.lastAds;

    this.kickPos.impulse(
      (rnd - 0.5) * 0.10 * adsDamp,
      h.kickUp * adsDamp,
      h.kickBack * adsDamp,
    );
    this.kickRot.impulse(
      h.kickPitch * adsDamp,
      (rnd2 - 0.5) * h.kickRoll * 0.45 * adsDamp,
      // The random roll is what stops consecutive shots looking like a looping
      // GIF; without it a full-auto burst reads as a single stuttering frame.
      (rnd - 0.5) * h.kickRoll * adsDamp,
    );

    this.flashTime = 0;
    this.triggerTime = 0;
    this.boltTime = 0;
    // A manually cycled action ejects its case partway through the stroke, so
    // the stroke has to outlast `ejectDelay` — otherwise the bolt is already
    // home before the case appears and the two read as unrelated events. A
    // self-loading action just cycles at half its own shot interval.
    this.boltDur =
      h.ejectDelay > 0
        ? h.ejectDelay * 2
        : this.spec
          ? Math.min(0.09, 30 / Math.max(1, this.spec.rpm))
          : 0.06;
    // Counted, not overwritten: the fire loop can resolve more than one round
    // in a single long frame, and each one has a case to throw.
    this.ejectTimer = h.ejectDelay;
    this.ejectPending++;

    // Randomise the flash so no two muzzle blasts are the same shape.
    this.flashGroup.rotation.z = this.random() * Math.PI * 2;
    // Store the shape and let `stepFlash` modulate it over the burn-down.
    // Writing the scale directly here would be pointless — `onFire` runs on the
    // fire path and `stepFlash` runs later in the SAME frame, so it would
    // overwrite the value before anything drew it.
    this.flashSize = 0.78 + this.random() * 0.5;
    this.flashStretch = 0.8 + this.random() * 0.6;
    this.flashDur = 0.038 + this.random() * 0.014;
  }

  // ── main update ────────────────────────────────────────────────────────────

  update(dt: number, elapsed: number, state: ViewModelState): void {
    if (dt <= 0) return;

    // First frame only: if another system has since lit the viewmodel scene,
    // withdraw our DUPLICATE lights rather than stacking a second key on top
    // of theirs. Doing this here — not in the constructor — is the whole point;
    // by now every system has initialised. Cheap: one array scan, once.
    if (!this.lightingReconciled) {
      this.lightingReconciled = true;
      if (this.lights.length > 0 && hasForeignViewLight(this.ctx.viewScene, this.lights)) {
        const keep: THREE.Object3D[] = [];
        for (const l of this.lights) {
          // Drop the duplicates — our key and fill directional pair, which the
          // engine's rig already provides and which were doubling exposure.
          // KEEP the ambient and the rim: the engine's rig is one key plus a
          // hemisphere and has no rim at all, so without these the weapon's
          // shadow side crushes to black and its silhouette stops separating
          // from the background. Their intensities are re-tuned here for the
          // engine's much stronger key rather than for our own.
          if (l.name === 'vm:key' || l.name === 'vm:fill') {
            l.removeFromParent();
            continue;
          }
          const light = l as THREE.Light;
          if (l.name === 'vm:amb') light.intensity = 0.85;
          if (l.name === 'vm:rim') light.intensity = 0.55;
          keep.push(l);
        }
        this.lights = keep;
      }
    }

    // The switch timeline runs FIRST because it may swap `this.rig` and
    // `this.spec` mid-frame. Everything after it must see the new weapon, or
    // the freshly raised gun spends a frame posed with the old one's offsets.
    this.tlPos.set(0, 0, 0);
    this.tlRot.set(0, 0, 0);
    this.stepSwitch(dt, this.tlPos, this.tlRot);

    const rig = this.rig;
    const spec = this.spec;
    if (!rig || !spec) return;

    const h = this.handling;
    const ads = clamp01(state.ads);
    this.lastAds = ads;

    // ── base pose: hip ↔ solved-ADS ─────────────────────────────────────────
    this.basePos.copy(this.hipPos).lerp(this.adsPos, ads);
    this.baseRot.set(HIP_CANT.x * (1 - ads), HIP_CANT.y * (1 - ads), HIP_CANT.z * (1 - ads));

    // ── sprint low-ready ────────────────────────────────────────────────────
    const sprintT = clamp01(state.lowReady);

    // ── sway: the weapon LAGS the camera ────────────────────────────────────
    // Target is proportional to angular VELOCITY (rad/s), not to the frame's
    // delta, so the lag is identical at 30 and 240 fps. The spring then trails
    // it, overshoots slightly and settles — that trail is the weight.
    const swayScale = (1 - 0.80 * ads) / h.mass;
    const yawRate = THREE.MathUtils.clamp(state.lookYawRate, -14, 14);
    const pitchRate = THREE.MathUtils.clamp(state.lookPitchRate, -14, 14);

    // Stiffen and fully damp the spring when aiming: an under-damped wobble on
    // a sight picture feels broken, but the same wobble at the hip feels alive.
    const swayK = THREE.MathUtils.lerp(105, 340, ads);
    const swayZ = THREE.MathUtils.lerp(0.78, 1.0, ads);
    this.swayPos.configure(swayK, swayZ);
    this.swayRot.configure(swayK * 0.92, swayZ);

    // EVERY channel must LAG. Turn the camera right (yawRate < 0) and the
    // weapon should trail to the LEFT of centre — x negative, yaw positive.
    // Pitch up (pitchRate > 0) and it should trail LOW — y negative, pitch
    // negative. Three of these five were signed the other way, so the weapon
    // led the camera on those axes and the whole channel read as a wobble
    // rather than as weight.
    this.swayPos.step(
      dt,
      THREE.MathUtils.clamp(yawRate * 0.0075, -0.05, 0.05) * swayScale,
      THREE.MathUtils.clamp(-pitchRate * 0.0060, -0.04, 0.04) * swayScale,
      0,
    );
    this.swayRot.step(
      dt,
      THREE.MathUtils.clamp(-pitchRate * 0.030, -0.20, 0.20) * swayScale,
      THREE.MathUtils.clamp(-yawRate * 0.038, -0.26, 0.26) * swayScale,
      // Roll INTO the turn, the way a carried weapon banks.
      THREE.MathUtils.clamp(yawRate * 0.026, -0.20, 0.20) * swayScale,
    );

    // ── bob: lissajous driven by real speed ─────────────────────────────────
    // Phase advances per METRE travelled, not per second, so the cadence stays
    // locked to the stride at any speed and the vertical dip lands on the
    // footfall. `|sin|` gives two dips per cycle: one per foot.
    const speed = Math.max(0, state.speed);
    // Stride lengthens as the low-ready blend comes in — that IS the sprint.
    const stride = 1.55 + 0.50 * clamp01(state.lowReady);
    this.bobPhase += (speed / stride) * Math.PI * 2 * dt;
    // Modulo, not a loop: `Number.isFinite` bounds NaN but not magnitude, and a
    // physics blow-up delivering an absurd-but-finite speed would spin a
    // subtractive loop for millions of iterations.
    if (this.bobPhase > Math.PI * 2) this.bobPhase %= Math.PI * 2;

    const bobAmt =
      clamp01(speed / 5.2) * h.bobScale * (1 - 0.82 * ads) * (state.grounded ? 1 : 0.15) *
      (state.crouching ? 0.55 : 1);
    const sprintBoost = 1 + sprintT * 0.85;
    const bobX = Math.sin(this.bobPhase) * 0.0135 * bobAmt * sprintBoost;
    const bobY = -Math.abs(Math.sin(this.bobPhase)) * 0.0115 * bobAmt * sprintBoost + 0.0035 * bobAmt;
    const bobZ = Math.sin(this.bobPhase * 2) * 0.0055 * bobAmt;
    const bobRoll = -Math.sin(this.bobPhase) * 0.030 * bobAmt * sprintBoost;
    const bobPitch = Math.sin(this.bobPhase * 2 + 0.7) * 0.016 * bobAmt;

    // ── idle breathing: a slow figure-of-eight ──────────────────────────────
    // 1:2 lissajous, which traces an ∞. The sniper's multiplier is 3.4×, so
    // unsupported it visibly wanders off a distant target — the mechanical
    // reason to hold your breath, expressed purely through animation.
    this.idlePhase += dt * 0.95;
    if (this.idlePhase > Math.PI * 2) this.idlePhase %= Math.PI * 2;
    const idleAmt = h.idleSway * (1 - 0.88 * ads) * (1 - sprintT);
    const idleX = Math.sin(this.idlePhase) * 0.0021 * idleAmt;
    const idleY = Math.sin(this.idlePhase * 2) * 0.0013 * idleAmt;
    const idleYaw = Math.sin(this.idlePhase + 0.4) * 0.0075 * idleAmt;
    const idlePitch = Math.sin(this.idlePhase * 2 + 1.1) * 0.0048 * idleAmt;

    // ── springs that decay to rest ──────────────────────────────────────────
    this.kickPos.step(dt, 0, 0, 0);
    this.kickRot.step(dt, 0, 0, 0);
    this.land.step(dt, 0);

    // ── remaining timelines (all additive into tlPos/tlRot) ─────────────────
    this.stepReload(dt, rig, this.tlPos, this.tlRot);
    this.stepInspect(dt, this.tlPos, this.tlRot);
    this.stepParts(dt, rig);
    this.stepFlash(dt);
    this.stepProps(dt);
    this.stepEject(dt, rig);

    // ── compose ─────────────────────────────────────────────────────────────
    const tp = this.tlPos;
    const tr = this.tlRot;
    this.outPos.copy(this.basePos);
    this.outPos.x += 0.030 * sprintT + this.swayPos.value.x + bobX + idleX + this.kickPos.value.x + tp.x;
    this.outPos.y +=
      -0.042 * sprintT + this.swayPos.value.y + bobY + idleY + this.kickPos.value.y + this.land.value + tp.y;
    this.outPos.z += 0.052 * sprintT + this.swayPos.value.z + bobZ + this.kickPos.value.z + tp.z;

    // Sprint low-ready: ~17° down, ~25° left, ~24° of roll. Getting the roll in
    // is what separates "gun pointed down" from "carried at the low ready".
    this.outRot.copy(this.baseRot);
    this.outRot.x +=
      -0.300 * sprintT + this.swayRot.value.x + bobPitch + idlePitch + this.kickRot.value.x +
      state.viewKickPitch * 0.35 + tr.x;
    this.outRot.y +=
      0.440 * sprintT + this.swayRot.value.y + idleYaw + this.kickRot.value.y +
      state.viewKickYaw * 0.35 + tr.y;
    this.outRot.z += 0.420 * sprintT + this.swayRot.value.z + bobRoll + this.kickRot.value.z + tr.z;

    rig.root.position.copy(this.outPos);
    rig.root.rotation.set(this.outRot.x, this.outRot.y, this.outRot.z);

    // The reticle is drawn with `depthTest: false` so the sight body can never
    // hide it while aiming. The cost is that it would otherwise punch through
    // the weapon and the shooter's hand whenever the gun is canted away — every
    // reload, every inspect — so it is simply switched off when the shooter is
    // not looking through the sight.
    // 0.55, not 0.12: `ads` is already eased, so 0.12 is only 4% of the way
    // through the transition — the depth-ignoring dot would switch on while the
    // weapon is still essentially at the hip and draw over the receiver.
    // Gated on the reload POSE, not on the reload timer: during an abort the
    // weapon is back on target well before the timer clears, and blanking the
    // dot for that window is exactly when the player needs it.
    const reloadPosed = this.reloadTime >= 0 && this.reloadLevel > 0.2;
    if (rig.reticle) rig.reticle.visible = ads > 0.55 && !reloadPosed && this.inspectTime < 0;

    // Reticle scintillation, ±12% at ~11 Hz. A perfectly steady dot looks
    // printed on; a visible flicker reads as a live emitter.
    if (rig.dot) {
      const mat = rig.dot.material as THREE.MeshBasicMaterial;
      // `elapsed` comes from the engine clock; guard it because this writes to
      // a SHARED material that every weapon's reticle draws with.
      const t = Number.isFinite(elapsed) ? elapsed : 0;
      mat.opacity = 0.88 + 0.12 * Math.sin(t * 71.3);
    }
  }

  // ── reload timeline ────────────────────────────────────────────────────────

  /**
   * A full magazine change, sequenced as eased segments of normalised time so
   * it always fits the spec'd reload duration exactly, whatever that is.
   *
   * All beats are fractions of the reload duration and live in `RELOAD_BEATS`,
   * which the audio system also reads, so the two can never drift apart.
   *
   *              tactical   empty
   *   cant in    0.00–0.16  0.00–0.16   look down at the magwell
   *   mag out    0.17       0.20        old magazine falls free
   *   mag rises  0.42       0.46        new magazine enters the well
   *   mag seats  0.60       0.64        seats with an overshoot and a jolt
   *   charge     —          0.66–0.80   handle pulled and released
   *   return     0.84–1.00  0.84–1.00   back to the ready
   *
   * The tactical reload skips the charging handle — that difference is exactly
   * why a tactical reload is faster, and showing it teaches the player the
   * mechanic without a tooltip.
   */
  private stepReload(dt: number, rig: WeaponRig, pos: THREE.Vector3, rot: THREE.Vector3): void {
    if (this.reloadTime < 0) return;

    // The level eases toward its target; the timeline only advances while the
    // reload is committed. Every channel below is scaled by the level, so both
    // unwinding and re-committing are continuous from wherever the pose is.
    const step = dt / RELOAD_FADE;
    if (this.reloadLevel < this.reloadTarget) {
      this.reloadLevel = Math.min(this.reloadTarget, this.reloadLevel + step);
    } else if (this.reloadLevel > this.reloadTarget) {
      this.reloadLevel = Math.max(this.reloadTarget, this.reloadLevel - step);
    }
    if (this.reloadTarget <= 0 && this.reloadLevel <= 0) {
      this.reloadTime = -1;
      this.reloadLevel = 1;
      this.reloadTarget = 1;
      this.restoreReloadParts();
      return;
    }
    const fade = easeInOutCubic(this.reloadLevel);
    if (this.reloadTarget > 0) this.reloadTime += dt;

    const u = this.reloadTime / this.reloadDur;

    if (u >= 1) {
      this.reloadTime = -1;
      this.reloadLevel = 1;
      this.reloadTarget = 1;
      this.restoreReloadParts();
      return;
    }

    // Overall cant: in fast, hold, out smooth.
    const inT = easeOutCubic(seg(u, 0.0, RELOAD_BEATS.cantIn));
    const outT = easeInOutCubic(seg(u, RELOAD_BEATS.returnStart, 1.0));
    const cant = inT * (1 - outT) * fade;
    pos.x += 0.038 * cant;
    pos.y += -0.055 * cant;
    pos.z += 0.048 * cant;
    rot.x += 0.20 * cant;
    rot.y += 0.62 * cant;
    rot.z += 0.34 * cant;

    // A shove down as the magazine is stripped, and a jolt as the new one
    // seats. Both are keyed off the beats themselves so retuning the timeline
    // cannot leave the jolt firing before the magazine arrives.
    // Each accent STARTS at its beat and decays after it — a reaction cannot
    // precede the event it is reacting to.
    const outAt = this.reloadTactical ? RELOAD_BEATS.magOut.tactical : RELOAD_BEATS.magOut.empty;
    const seatAt = this.reloadTactical ? RELOAD_BEATS.magIn.tactical : RELOAD_BEATS.magIn.empty;
    rot.x += -0.14 * pulse(seg(u, outAt, outAt + 0.13)) * fade;
    pos.y += -0.012 * pulse(seg(u, seatAt, seatAt + 0.09)) * fade;

    // Magazine out.
    const dropAt = this.reloadTactical ? RELOAD_BEATS.magOut.tactical : RELOAD_BEATS.magOut.empty;
    if (!this.magDropped && u >= dropAt) {
      this.magDropped = true;
      rig.magazine.visible = false;
      this.spawnMagProp(rig);
    }

    // Magazine in: rises from below the frame and overshoots into the well.
    const insertA = this.reloadTactical ? RELOAD_BEATS.magInStart.tactical : RELOAD_BEATS.magInStart.empty;
    const insertB = this.reloadTactical ? RELOAD_BEATS.magIn.tactical : RELOAD_BEATS.magIn.empty;
    if (this.magDropped && u >= insertA) {
      if (!this.magReinserted) {
        this.magReinserted = true;
        rig.magazine.visible = true;
      }
      const t = seg(u, insertA, insertB);
      const rise = 1 - easeOutBack(t);
      rig.magazine.position.set(
        this.magHome.x + rise * MAG_ARC.x,
        this.magHome.y + rise * MAG_ARC.y,
        this.magHome.z + rise * MAG_ARC.z,
      );
    } else if (this.magDropped) {
      rig.magazine.position.set(
        this.magHome.x + MAG_ARC.x,
        this.magHome.y + MAG_ARC.y,
        this.magHome.z + MAG_ARC.z,
      );
    }

    // Support hand: dives off-screen for the fresh magazine, comes back with it.
    const handOut = pulse(seg(u, RELOAD_BEATS.cantIn, RELOAD_BEATS.handBack)) * fade;
    rig.leftArm.position.set(
      this.leftArmHome.x - 0.055 * handOut,
      this.leftArmHome.y - 0.180 * handOut,
      this.leftArmHome.z + 0.090 * handOut,
    );
    // The magazine unwinds toward its seated position with everything else,
    // rather than popping back when the abort completes.
    if (fade < 1) rig.magazine.position.lerp(this.magHome, 1 - fade);

    // Charging handle — empty reload only.
    if (!this.reloadTactical && rig.chargingHandle) {
      const pull = easeOutCubic(seg(u, RELOAD_BEATS.chargeStart, RELOAD_BEATS.chargeBack));
      const release = easeInCubic(seg(u, RELOAD_BEATS.chargeBack, RELOAD_BEATS.chargeRelease));
      const travel = (pull - release) * rig.chargingTravel;
      rig.chargingHandle.position.set(
        this.chargeHome.x,
        this.chargeHome.y,
        this.chargeHome.z + travel * fade,
      );
      // On the rifle and the SMG the bolt carrier is a separate object from the
      // handle that drags it, so it has to be driven too or it sits motionless
      // in the ejection port while the handle strokes past it.
      if (rig.bolt && rig.bolt !== rig.chargingHandle && this.boltTime < 0) {
        const f = rig.chargingTravel > 1e-6 ? (travel * fade) / rig.chargingTravel : 0;
        rig.bolt.position.set(
          this.boltHome.x + rig.boltTravel.x * f,
          this.boltHome.y + rig.boltTravel.y * f,
          this.boltHome.z + rig.boltTravel.z * f,
        );
      }
      // The bolt slamming forward snaps the whole weapon, starting on the
      // release beat rather than at a hardcoded time that could drift off it.
      rot.x +=
        -0.10 * pulse(seg(u, RELOAD_BEATS.chargeRelease, RELOAD_BEATS.chargeRelease + 0.07)) * fade;
    }
  }

  // ── switch timeline ────────────────────────────────────────────────────────

  private stepSwitch(dt: number, pos: THREE.Vector3, rot: THREE.Vector3): void {
    if (this.switchTime < 0) return;
    this.switchTime += dt;
    const u = this.switchTime / this.switchDur;

    if (u >= 1) {
      this.switchTime = -1;
      // A frame long enough to jump straight past the halfway point must still
      // perform the swap, or the weapon dips, rises, and is still the old one.
      const pending = this.switchCallback;
      this.switchCallback = null;
      if (pending) pending();
      return;
    }

    // Drop out of frame, swap at the bottom of the arc where nothing is
    // visible, then raise. Ease-in on the way down and ease-out on the way up,
    // so the weapon accelerates away and decelerates into the ready position.
    if (u < 0.5) {
      const t = easeInCubic(seg(u, 0.0, 0.5));
      pos.y += -0.26 * t;
      pos.z += 0.06 * t;
      rot.x += -0.72 * t;
      rot.z += 0.30 * t;
    } else {
      if (this.switchCallback) {
        const cb = this.switchCallback;
        this.switchCallback = null;
        cb();
      }
      const t = easeOutCubic(seg(u, 0.5, 1.0));
      const i = 1 - t;
      pos.y += -0.26 * i;
      pos.z += 0.06 * i;
      rot.x += -0.72 * i;
      rot.z += 0.30 * i;
    }
  }

  // ── inspect timeline ───────────────────────────────────────────────────────

  /** A showy turn-and-tilt: bring it in, swing left to present the LEFT flank,
   *  swing back through square to present the RIGHT flank (the ejection port
   *  side), then return. Purely cosmetic, entirely for the player. */
  private stepInspect(dt: number, pos: THREE.Vector3, rot: THREE.Vector3): void {
    if (this.inspectTime < 0) return;

    // While fading out the timeline is HELD and only the envelope moves, so
    // the pose shrinks continuously toward zero from wherever it was.
    let fade = 1;
    if (this.inspectFade >= 0) {
      this.inspectFade += dt;
      fade = 1 - easeInOutCubic(clamp01(this.inspectFade / INSPECT_FADE));
      if (fade <= 0) {
        this.inspectTime = -1;
        this.inspectFade = -1;
        return;
      }
    } else {
      this.inspectTime += dt;
    }

    const u = this.inspectTime / this.inspectDur;
    if (u >= 1) {
      this.inspectTime = -1;
      return;
    }

    const envelope = easeOutCubic(seg(u, 0.0, 0.14)) * (1 - easeInOutCubic(seg(u, 0.84, 1.0))) * fade;

    // Pull the weapon in toward the eye and up into the light.
    pos.x += -0.055 * envelope;
    pos.y += 0.030 * envelope;
    pos.z += -0.070 * envelope;

    // Beat 1: swing the muzzle hard left (yaw +1.55), which turns the LEFT
    //          flank toward the camera. (+Y rotation turns a −Z forward axis
    //          toward −X, so the +X face rotates away.)
    const a = easeInOutCubic(seg(u, 0.10, 0.42));
    // Beat 2: swing back through square to yaw −1.00, presenting the RIGHT
    //          flank — the ejection port, bolt and brass deflector.
    const b = easeInOutCubic(seg(u, 0.44, 0.72));
    // Beat 3: settle back square.
    const c = easeInOutCubic(seg(u, 0.72, 0.92));

    const yaw = (1.55 * a - 2.55 * b + 1.0 * c) * envelope;
    const roll = (-0.55 * a + 1.35 * b - 0.80 * c) * envelope;
    const pitch = (0.30 * a - 0.62 * b + 0.32 * c) * envelope;

    rot.x += pitch;
    rot.y += yaw;
    rot.z += roll;
  }

  // ── animated sub-parts ─────────────────────────────────────────────────────

  private stepParts(dt: number, rig: WeaponRig): void {
    // Trigger blade: breaks fast under the finger, returns slower on its
    // spring. The asymmetry is small but it is the difference between a blade
    // that is being pulled and one that is oscillating.
    if (this.triggerTime >= 0 && rig.trigger) {
      this.triggerTime += dt;
      const t = this.triggerTime / 0.09;
      if (t >= 1) {
        this.triggerTime = -1;
        rig.trigger.rotation.x = this.triggerHomeX;
      } else {
        const press = t < 0.3 ? easeOutCubic(t / 0.3) : 1 - easeInOutCubic((t - 0.3) / 0.7);
        rig.trigger.rotation.x = this.triggerHomeX - 0.38 * press;
      }
    }

    // An interrupted stroke eases home. Runs OUTSIDE the reload gate below,
    // because the interruption is usually the reload itself.
    if (this.boltSettle >= 0 && rig.bolt) {
      this.boltSettle += dt;
      const t = clamp01(this.boltSettle / BOLT_SETTLE);
      rig.bolt.position.lerpVectors(this.boltSettleFrom, this.boltHome, easeOutCubic(t));
      if (t >= 1) this.boltSettle = -1;
    }

    // Bolt / slide / pump reciprocation. Sharp out, slower in — a slide is
    // driven back by gas and returned by a spring, and the asymmetry shows.
    // Only suspended during a reload on weapons where the reload timeline OWNS
    // the bolt — i.e. where the bolt and the charging handle are the same
    // object. On the rifle they are separate, so freezing it there just left
    // the carrier motionless in the port for the whole animation.
    // Only from the charging beat onward. Claiming the bolt for the WHOLE
    // reload cancelled the cycle the magazine's last round had just started —
    // the auto-reload fires from inside that very shot.
    const reloadOwnsBolt =
      this.reloadTime >= 0 &&
      !this.reloadTactical && // a tactical reload never touches the handle
      rig.bolt === rig.chargingHandle &&
      this.reloadTime / this.reloadDur >= RELOAD_BEATS.chargeStart;
    if (this.boltTime >= 0 && rig.bolt && !reloadOwnsBolt && this.boltSettle < 0) {
      this.boltTime += dt;
      const t = this.boltTime / this.boltDur;
      if (t >= 1) {
        this.boltTime = -1;
        rig.bolt.position.copy(this.boltHome);
      } else {
        const back = t < 0.42 ? easeOutCubic(t / 0.42) : 1 - easeInOutCubic((t - 0.42) / 0.58);
        rig.bolt.position.set(
          this.boltHome.x + rig.boltTravel.x * back,
          this.boltHome.y + rig.boltTravel.y * back,
          this.boltHome.z + rig.boltTravel.z * back,
        );
      }
    }
  }

  // ── muzzle flash ───────────────────────────────────────────────────────────

  private buildFlash(): void {
    this.flashGroup.name = 'muzzleFlash';
    this.flashGroup.visible = false;
    this.flashGroup.frustumCulled = false;

    // The flash owns its own additive materials rather than borrowing the
    // shared weapon cache: they need `depthTest: false` and additive blending,
    // which would be wrong on every other part of the weapon.
    const flashMat = new THREE.MeshBasicMaterial({
      // Star-burst texture rather than a flat quad: the hard radial streaks are
      // what make a single frame of light read as a muzzle flash.
      map: muzzleFlashTexture(),
      color: 0xffd2a0,
      toneMapped: false,
      transparent: true,
      opacity: 1,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      // Depth-TESTED, unlike the reticle: the flash is a real object at the
      // muzzle, so it should be occluded by the barrel and hand-guard it wraps.
      // Ignoring depth made it draw over the optic on every shot while aiming.
      depthTest: true,
      side: THREE.DoubleSide,
    });
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0xfff0d2,
      toneMapped: false,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
    });

    // Two crossed quads: the flash then reads as volume from any viewing angle
    // instead of vanishing edge-on the way a single billboard does. The
    // rotation MUST be about Y (or X) — a PlaneGeometry lies in XY facing +Z,
    // so rotating about Z spins it within its own plane and leaves the two
    // quads coplanar, which is just a single quad at double brightness.
    const quad = new THREE.PlaneGeometry(1, 1);
    for (let i = 0; i < 2; i++) {
      const p = new THREE.Mesh(quad, flashMat);
      p.rotation.y = (i * Math.PI) / 2;
      p.position.z = -0.012;
      p.frustumCulled = false;
      p.renderOrder = 20;
      this.flashGroup.add(p);
    }
    // Short forward cone: the visible column of burning propellant.
    const cone = new THREE.ConeGeometry(0.36, 1.0, 7, 1, true);
    cone.rotateX(-Math.PI / 2);
    cone.translate(0, 0, -0.5);
    const coneMesh = new THREE.Mesh(cone, coreMat);
    coneMesh.frustumCulled = false;
    coneMesh.renderOrder = 20;
    this.flashGroup.add(coneMesh);

    // Two lights: one inside the viewmodel scene so the weapon and hands are
    // lit by their own muzzle blast, and one in the world scene so the
    // environment flashes too. Neither is ever added or removed at fire time —
    // they are modulated by intensity instead, because changing a scene's light
    // count forces every material in it to recompile, which would stutter on
    // every trigger pull. (`worldLight` is parented once here; `viewLight` is
    // re-parented to the muzzle on a weapon change, which is not a fire-path
    // operation.)
    this.viewLight = new THREE.PointLight(0xffc07a, 0, 2.4, 2);
    this.viewLight.name = 'vm:flash';
    this.viewLight.position.set(0, 0, -0.04);

    this.worldLight = new THREE.PointLight(0xffb066, 0, 14, 2);
    this.worldLight.name = 'vm:muzzleLight';
    this.ctx.scene.add(this.worldLight);
  }

  private stepFlash(dt: number): void {
    if (this.flashTime < 0) {
      if (this.flashGroup.visible) this.flashGroup.visible = false;
      if (this.viewLight && this.viewLight.intensity !== 0) this.viewLight.intensity = 0;
      if (this.worldLight && this.worldLight.intensity !== 0) this.worldLight.intensity = 0;
      return;
    }
    this.flashTime += dt;
    const t = this.flashTime / this.flashDur;
    if (t >= 1) {
      this.flashTime = -1;
      this.flashGroup.visible = false;
      if (this.viewLight) this.viewLight.intensity = 0;
      if (this.worldLight) this.worldLight.intensity = 0;
      return;
    }
    this.flashGroup.visible = true;
    // Instant onset, cubic decay: a real muzzle flash is 1–2 ms of ignition
    // followed by a fast burn-down, and the sharp attack is what makes it
    // register as a flash rather than as a fade.
    const i = 1 - t;
    const curve = i * i * i;
    const size = this.handling.flashScale * this.flashSize * (0.85 + 0.35 * curve);
    this.flashGroup.scale.set(size, size, size * this.flashStretch);
    if (this.viewLight) this.viewLight.intensity = this.handling.flashLight * curve;
    if (this.worldLight) {
      this.worldLight.intensity = this.handling.flashLight * 2.2 * curve;
      this.getMuzzleWorld(this.worldLight.position);
    }
  }

  // ── shell ejection ─────────────────────────────────────────────────────────

  private stepEject(dt: number, rig: WeaponRig): void {
    if (this.ejectTimer < 0 || this.ejectPending <= 0) return;
    this.ejectTimer -= dt;
    if (this.ejectTimer > 0) return;
    this.ejectTimer = -1;
    const owed = this.ejectPending;
    this.ejectPending = 0;
    if (!this.spec) return;

    // Port position in world space, and a velocity that throws the case out to
    // the shooter's right and slightly up and back — the real ejection arc for
    // a right-handed weapon.
    this.getEjectWorld(_v2);
    const cam = this.ctx.camera;
    cam.updateMatrixWorld();
    _v0.set(1, 0, 0).transformDirection(cam.matrixWorld); // camera right
    _v1.set(0, 1, 0).transformDirection(cam.matrixWorld); // camera up

    // Allocated fresh, not pooled: the shell-physics listener owns this vector
    // for the lifetime of the casing. This is a per-SHOT allocation, which the
    // no-allocation rule for the per-FRAME path does not cover.
    const vel = new THREE.Vector3();
    vel.addScaledVector(_v0, 2.1 + this.random() * 1.1);
    vel.addScaledVector(_v1, 1.5 + this.random() * 0.8);
    _v0.set(0, 0, 1).transformDirection(cam.matrixWorld);
    vel.addScaledVector(_v0, 0.35 + this.random() * 0.5);

    this.ctx.bus.emit('shell:eject', {
      position: _v2.clone(),
      velocity: vel,
      caliber: this.spec.caliber,
    });
    // Any rounds beyond the first in this frame get their own case, scattered
    // a little more so they do not stack into one sprite.
    for (let i = 1; i < owed; i++) {
      const extra = vel.clone();
      extra.x += (this.random() - 0.5) * 0.9;
      extra.y += (this.random() - 0.5) * 0.7;
      this.ctx.bus.emit('shell:eject', {
        position: _v2.clone(),
        velocity: extra,
        caliber: this.spec.caliber,
      });
    }
  }

  // ── magazine props ─────────────────────────────────────────────────────────

  private spawnMagProp(rig: WeaponRig): void {
    if (this.props.length >= MAX_MAG_PROPS) {
      const old = this.props.shift();
      if (old) old.obj.removeFromParent();
    }
    const obj = cloneMagazineProp(rig);
    // Start where the magazine actually was, in viewmodel-scene space, so the
    // hand-off from "in the weapon" to "falling" is seamless.
    //
    // `updateWorldMatrix(true, …)` walks UP through the ancestors, recomposing
    // the rig root's matrix from its current position/rotation. This frame's
    // pose has not been written yet (that happens at the end of `update`), so
    // this captures last frame's — one frame of lag on a prop that is about to
    // start tumbling, which is invisible. `updateMatrixWorld(true)` walks DOWN
    // to children instead and would not refresh the ancestors at all, leaving
    // the prop attached to a stale root matrix from an arbitrary earlier frame.
    rig.magazine.updateWorldMatrix(true, false);
    obj.position.setFromMatrixPosition(rig.magazine.matrixWorld);
    obj.quaternion.setFromRotationMatrix(rig.magazine.matrixWorld);
    this.ctx.viewScene.add(obj);

    this.props.push({
      obj,
      vel: new THREE.Vector3(
        -0.10 + this.random() * 0.22,
        -0.35 - this.random() * 0.25,
        0.10 + this.random() * 0.20,
      ),
      spin: new THREE.Vector3(
        (this.random() - 0.5) * 5.5,
        (this.random() - 0.5) * 3.5,
        (this.random() - 0.5) * 4.5,
      ),
      life: 0,
    });
  }

  private stepProps(dt: number): void {
    for (let i = this.props.length - 1; i >= 0; i--) {
      const p = this.props[i];
      p.life += dt;
      p.vel.y -= PROP_GRAVITY * dt;
      p.obj.position.addScaledVector(p.vel, dt);
      p.obj.rotation.x += p.spin.x * dt;
      p.obj.rotation.y += p.spin.y * dt;
      p.obj.rotation.z += p.spin.z * dt;
      if (p.life > 1.15) {
        p.obj.removeFromParent();
        this.props.splice(i, 1);
      }
    }
  }

  // ── space conversion ───────────────────────────────────────────────────────

  /**
   * Convert a point from viewmodel-scene space into game-world space.
   *
   * The viewmodel lives in its own scene viewed by its own camera, so a muzzle
   * position there means nothing to the world. Going view-scene → view-camera
   * local → world-camera works regardless of where either camera is placed,
   * which matters because this system does not own either of them.
   */
  private toWorld(out: THREE.Vector3): THREE.Vector3 {
    const vc = this.ctx.viewCamera;
    const wc = this.ctx.camera;
    vc.updateMatrixWorld(); // Camera.updateMatrixWorld also refreshes the inverse
    wc.updateMatrixWorld();
    out.applyMatrix4(vc.matrixWorldInverse);
    out.applyMatrix4(wc.matrixWorld);
    return out;
  }

  private getWorldOf(obj: THREE.Object3D, out: THREE.Vector3): THREE.Vector3 {
    obj.updateWorldMatrix(true, false);
    out.setFromMatrixPosition(obj.matrixWorld);
    return this.toWorld(out);
  }

  /** World-space muzzle position, for tracers, flash light and audio. */
  getMuzzleWorld(out: THREE.Vector3): THREE.Vector3 {
    const rig = this.rig;
    if (!rig) return out.copy(this.ctx.camera.position);
    return this.getWorldOf(rig.muzzle, out);
  }

  /** World-space ejection-port position. */
  getEjectWorld(out: THREE.Vector3): THREE.Vector3 {
    const rig = this.rig;
    if (!rig) return out.copy(this.ctx.camera.position);
    return this.getWorldOf(rig.ejectPort, out);
  }

  // ── misc ───────────────────────────────────────────────────────────────────

  /** xorshift; a dedicated stream so cosmetic randomness never perturbs
   *  the deterministic ballistics RNG. */
  private random(): number {
    let x = this.rngSeed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngSeed = x >>> 0;
    return (x >>> 0) / 4294967296;
  }

  dispose(): void {
    for (const p of this.props) p.obj.removeFromParent();
    this.props.length = 0;
    this.flashGroup.removeFromParent();
    this.flashGroup.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.geometry.dispose();
        const mm = mesh.material;
        if (Array.isArray(mm)) for (const x of mm) x.dispose();
        else mm.dispose();
      }
    });
    this.viewLight?.removeFromParent();
    this.viewLight?.dispose();
    this.worldLight?.removeFromParent();
    this.worldLight?.dispose();
    for (const l of this.lights) {
      l.removeFromParent();
      (l as THREE.Light).dispose?.();
    }
    this.lights.length = 0;
    // Only release the environment if WE installed it; another system may own
    // it. Clearing the reference matters as much as disposing the texture:
    // three.js caches a PMREM render target keyed on it, and leaving it set
    // would leak that target into the next ViewModel.
    if (this.ownsEnvironment) {
      this.ctx.viewScene.environment?.dispose();
      this.ctx.viewScene.environment = null;
      this.ownsEnvironment = false;
    }
    for (const rig of this.rigs.values()) disposeRig(rig);
    this.rigs.clear();
    this.rig = null;
  }
}
