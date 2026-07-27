/**
 * NOISEFIELD — Weapon system.
 *
 * Owns firing, ammunition, reloads, weapon switching, aim-down-sights, recoil
 * accumulation and recovery, and drives the viewmodel.
 *
 * Communicates almost entirely over the event bus so it can be developed and
 * run before the player, physics, VFX, audio and HUD systems exist.
 *
 * EMITS   weapon:fire · weapon:dryfire · weapon:reload:start · weapon:reload:end
 *         weapon:switch · weapon:ads · weapon:ammo · tracer · shell:eject
 *         camera:kick · camera:shake · impact · damage:dealt
 * CONSUMES player:state · player:land
 * USES    ctx.get<IPhysics>('physics')  (optional — falls back to a Raycaster)
 *         ctx.get<IMaterialLibrary>('materials') (optional)
 * OWNS    ctx.camera.fov and ctx.viewCamera.fov — both are written every frame
 *         by the aim-down-sights blend (see `updateAds`). Another system that
 *         also writes them will be silently overwritten; layer additive effects
 *         on `ctx.settings.fov` instead, which this system reads as its base.
 *         ctx.viewScene — the weapon rig, its lighting and, if nothing else has
 *         claimed it, the environment map.
 *
 * CAVEAT  'weapon:reload:end' fires on ABORT as well as on completion (a HUD
 *         bar must not hang), and `GameEvents` has no field to distinguish
 *         them. A listener that schedules sound against RELOAD_BEATS should
 *         treat any 'end' as "stop the sequence" and read `magAmmo` to learn
 *         whether the reload actually happened.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  FrameContext,
  PlayerStateSnapshot,
  System,
  WeaponId,
  WeaponSpec,
} from '../core/Types';
import { Ballistics, Rng } from './Ballistics';
import { PRIMARY_POOL, SIDEARM, WEAPON_ORDER, WEAPON_SPECS, shotInterval } from './Specs';
import { RELOAD_BEATS, ViewModel, type ViewModelState } from './ViewModel';
import { disposeSharedMaterials } from './WeaponModels';

// ─────────────────────────────────────────────────────────────────────────────
// Tuning constants
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Recovery hold after a shot, as a multiple of the weapon's own shot interval.
 *
 * For an AUTOMATIC weapon the hold must outlast the gap between rounds, or the
 * view sinks between shots, fights the player's own pull, and the pattern
 * becomes unlearnable. Keying it to the fire rate is the only way to satisfy
 * that for a 950 rpm SMG (63 ms between rounds) and a 700 rpm rifle (86 ms) at
 * once. Slow weapons are then capped by `RECOVER_HOLD_MAX`: the bolt gun
 * (1.09 s cycle) and the shotgun (0.67 s) SHOULD settle between shots, and
 * holding for a full cycle would leave the sights parked high. The pistol's
 * 143 ms interval falls below the cap, so it behaves like the automatics —
 * rapid fire climbs, deliberate fire settles.
 */
const RECOVER_HOLD_RATIO = 1.15;
/** ...but never hold longer than this, so slow weapons still settle promptly. */
const RECOVER_HOLD_MAX = 0.22;

/**
 * Hard ceiling on the accumulated view kick, and the reason the soft bleed in
 * `absorb` exists at all.
 *
 * The residual is subtracted from the player's own base pitch, which their
 * controller clamps to ±90°, so an unbounded residual progressively eats the
 * bottom of their look range. It IS reachable: a fully compensated 30-round
 * rifle magazine accumulates ≈0.28 rad, so two of them with no downward mouse
 * movement in between reach the ceiling; left entirely uncompensated it takes
 * about six, because only the 28% that is not auto-recovered settles
 * (0.2849 × 0.28 = 0.0798 rad per magazine). The
 * bleed is what makes either case pathological rather than routine; the clamp
 * is what makes it survivable. At 0.45 rad the player has lost 26° at the
 * bottom of their pitch range and no more.
 */
const MAX_VIEW_KICK = 0.45;

/**
 * The settled-residual bleed. See `updateRecoil` and `absorb` for the model.
 *
 * The bleed makes counter-input travel slightly further than the player asked
 * for, so it IS a directional sensitivity change while it is active and must be
 * rationed. It is therefore off entirely below `RESIDUAL_SOFT` and ramps to
 * `RESIDUAL_BLEED_MAX` only as the residual approaches `RESIDUAL_HARD`:
 *
 *   |kick| <= 0.10 rad (5.7 deg)  ->  no bleed at all. Covers a normal burst,
 *                                     so ordinary play has zero artifact.
 *   |kick| -> 0.30 rad (17.2 deg) ->  ramp saturates at its strongest, 25%.
 *                                     The hard bound stays `MAX_VIEW_KICK`.
 *
 * That bound matters because the residual is subtracted from the player's own
 * base pitch, and their pitch is clamped to +-90 deg: an unbounded residual
 * would progressively eat the bottom of their look range. Trading a mild,
 * transient sensitivity change for that is unambiguously the right way round.
 */
const RESIDUAL_BLEED_MAX = 0.25;
const RESIDUAL_SOFT = 0.10;
const RESIDUAL_HARD = 0.30;
/** Residual only bleeds once the player has stopped shooting. */
const RESIDUAL_BLEED_DELAY = 0.25;

/**
 * Trigger-release time after which the recoil pattern restarts, as a floor.
 * Scaled up for slow-firing weapons so a bolt gun cycled at its maximum rate
 * still walks its authored five-shot pattern instead of replaying shot 1.
 */
const PATTERN_RESET = 0.34;
const PATTERN_RESET_RATIO = 1.9;

/** Buffer window for a semi-auto trigger press that arrived early. */
const SEMI_BUFFER = 0.16;

/** One round in three is a tracer, as with real linked/loaded ammunition. */
const TRACER_EVERY = 3;

/** Safety valve on the catch-up loop after a very long frame. */
const MAX_SHOTS_PER_FRAME = 8;

/**
 * Bloom ceiling, expressed as a number of shots' worth of `spreadPerShot`.
 *
 * The value matters more than it looks. The SMG's identity is a tighter HIP
 * cone (2.0° vs the rifle's 2.6°) that it loses under sustained fire, and a
 * ceiling this low simply caps both weapons before that reversal completes:
 * at 7.5 the saturated cones are SMG 0.0604 rad against rifle 0.0619, i.e. the
 * SMG becomes the more accurate weapon again from about round 13 — the exact
 * opposite of the design. The crossover needs a ceiling above 8.75; at 12 the
 * saturated cones are SMG 0.0757 against rifle 0.0718 and the intended
 * relationship holds for the whole magazine.
 */
const BLOOM_CAP_SHOTS = 12;

/**
 * The engine's default `QualitySettings.adsFovScale`. Per-weapon `fovScale`
 * values are authored as absolute zooms, so the global setting is applied
 * RELATIVE to this baseline — that way a player moving the aim-FOV slider
 * affects every weapon proportionally instead of the setting being ignored.
 */
const DEFAULT_ADS_FOV_SCALE = 0.62;

// ─────────────────────────────────────────────────────────────────────────────
// Scratch — the fire path must not allocate
// ─────────────────────────────────────────────────────────────────────────────

const _origin = new THREE.Vector3();
const _aim = new THREE.Vector3();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _muzzle = new THREE.Vector3();

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
/** NaN/Infinity guard. Note `clamp` above deliberately does NOT do this — both
 *  of its comparisons are false for NaN, so it passes NaN straight through. */
function finiteOr(x: number, fallback: number): number {
  return Number.isFinite(x) ? x : fallback;
}
function easeOutCubic(t: number): number {
  const i = 1 - t;
  return 1 - i * i * i;
}
function easeInCubic(t: number): number {
  return t * t * t;
}

interface AmmoState {
  mag: number;
  reserve: number;
}

// ─────────────────────────────────────────────────────────────────────────────

export class WeaponSystem implements System {
  readonly name = 'weapons';

  /**
   * ═══ CONTRACT FOR THE PLAYER SYSTEM ═══
   *
   * An ABSOLUTE camera-orientation offset in RADIANS — not a per-frame delta.
   * Read it every frame and ADD it to the orientation you derive from the
   * mouse; do not integrate it, do not zero it, do not reset it.
   *
   *     // in the player system's update, AFTER integrating mouse input:
   *     this.basePitch -= input.lookDelta.y;          // your own aim
   *     this.baseYaw   -= input.lookDelta.x;
   *     this.basePitch  = clamp(this.basePitch, -PI/2, PI/2);
   *     camera.rotation.set(
   *       this.basePitch + weapons.viewKick.pitch,
   *       this.baseYaw   + weapons.viewKick.yaw,
   *       0, 'YXZ');
   *
   * SIGN CONVENTION. `viewKick.pitch` is POSITIVE for muzzle rise, matching a
   * pitch axis where positive tilts the view UP. That is the convention that
   * falls out of `basePitch -= input.lookDelta.y` given that `Input` reports
   * `lookDelta.y > 0` for downward mouse movement. If your player system uses
   * the opposite sign for pitch, negate BOTH `viewKick.pitch` when you apply it
   * AND `input.lookDelta.y` before it reaches this system's recoil absorption
   * (see `updateRecoil`) — otherwise recoil will still LOOK right while the
   * player's counter-pull inflates the recovery instead of cancelling it,
   * silently reintroducing the spring-back this model exists to prevent.
   *
   * ORDERING. Valid to read at any point in a frame, but for zero latency
   * register this system BEFORE the player system so the kick applied is the
   * current frame's.
   *
   * DO NOT ALSO APPLY `camera:kick`. That event carries the same per-shot
   * deltas purely as a notification for effects (audio ducking, spectator
   * cameras, killcams); they are ALREADY accumulated into `viewKick`, and
   * applying both doubles all recoil.
   *
   * This system never writes to `ctx.camera.rotation` — the Player owns it.
   */
  readonly viewKick = { pitch: 0, yaw: 0 };

  private ctx: EngineContext | null = null;
  private view: ViewModel | null = null;
  private ballistics: Ballistics | null = null;
  /** Ballistics stream: reseeded per shot from the published `seed`. */
  private readonly rng = new Rng(0x1f4a3c7d);
  /**
   * Recoil-jitter stream, kept separate from `rng`. The ballistics stream is
   * consumed a variable number of times depending on what the round hits, so
   * sharing it would couple the recoil to world geometry and break replay.
   */
  private readonly recoilRng = new Rng(0x7b19d3a5);

  // ── inventory ──────────────────────────────────────────────────────────────
  private current: WeaponId = 'ar_vector';
  private primaryIndex = 0;
  private readonly ammo = {} as Record<WeaponId, AmmoState>;

  // ── firing ─────────────────────────────────────────────────────────────────
  /** Seconds until the next round may leave the barrel. May go negative. */
  private shotCooldown = 0;
  private shotIndex = 0;
  private sinceLastShot = 99;
  private semiBuffer = -1;
  private burstRemaining = 0;
  private tracerCounter = 0;
  private bloom = 0;

  // ── recoil ─────────────────────────────────────────────────────────────────
  /**
   * How much of the applied kick is still eligible for automatic recovery.
   * The player's own downward mouse input is subtracted from this, which is
   * what stops the view springing back past where they pulled to.
   */
  private owedPitch = 0;
  private owedYaw = 0;
  private recoverDelay = 0;
  /** Purely cosmetic mirror of the kick, always decays fully to zero. */
  private mirrorPitch = 0;
  private mirrorYaw = 0;

  // ── ADS ────────────────────────────────────────────────────────────────────
  /** Linear 0..1 transit; the eased value is derived from it. */
  private adsRaw = 0;
  private adsEase = 0;
  private adsIn = false;
  private adsAnnounced = false;
  private viewFovBase = 60;

  // ── reload ─────────────────────────────────────────────────────────────────
  private reloadTimer = -1;
  private reloadDur = 0;
  private reloadTactical = false;
  /** A reload requested while a previous abort was still unwinding. */
  private reloadQueued = false;

  // ── movement state, mirrored from 'player:state' ───────────────────────────
  private speed = 0;
  private sprintingRaw = false;
  private crouching = false;
  private grounded = true;
  /**
   * Eased low-ready blend. `lowReadyIntent` is the single source of truth: it
   * is eased here into `lowReady` and that EASED value is handed to the
   * ViewModel as `ViewModelState.lowReady`, so the firing gate below and the
   * visible pose are literally the same number rather than two integrators
   * that agree only until someone edits one of them.
   */
  private lowReady = 0;
  /** This frame's low-ready intent, sampled once at the top of `update`. */
  private lowReadyNow = false;

  private readonly unsubscribe: Array<() => void> = [];
  /**
   * Canonical list of colliders our rounds ignore. Held here rather than only
   * in Ballistics because `dispose()` discards that instance, and the list has
   * to survive to be replayed into the next one.
   */
  private readonly pendingExclusions: THREE.Object3D[] = [];
  /** Deferred first-frame work — see `init` for why it cannot happen there. */
  private firstFrame = true;

  /** Reused every frame — allocating this would defeat the whole exercise. */
  private readonly vmState: ViewModelState = {
    speed: 0,
    lowReady: 0,
    crouching: false,
    grounded: true,
    ads: 0,
    lookYawRate: 0,
    lookPitchRate: 0,
    viewKickPitch: 0,
    viewKickYaw: 0,
  };

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    this.viewFovBase = ctx.viewCamera.fov;

    for (const id of WEAPON_ORDER) {
      const spec = WEAPON_SPECS[id];
      this.ammo[id] = { mag: spec.magSize, reserve: spec.reserveAmmo };
    }

    this.ballistics = new Ballistics(ctx);
    // Replayed, NOT drained: `dispose()` throws the Ballistics instance away,
    // so a dispose/init cycle has to be able to replay them again.
    for (const o of this.pendingExclusions) this.ballistics.addExclusion(o);

    this.view = new ViewModel(ctx);

    this.unsubscribe.push(
      ctx.bus.on('player:state', this.onPlayerState),
      ctx.bus.on('player:land', this.onPlayerLand),
    );

    // The first weapon is NOT built here, and the opening 'weapon:switch' /
    // 'weapon:ammo' are NOT emitted here. Engine.init runs systems in
    // registration order, so anything done during init is invisible to every
    // system registered after this one — a HUD would miss its starting ammo
    // count, and the material library might not exist yet to skin the rig.
    // Both are deferred to the first update, by which point every system has
    // initialised and subscribed. See `firstFrame`.
  }

  /** Emit the full current state. Safe to call at any time; use it if a system
   *  attaches late and needs to catch up. */
  broadcastState(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.bus.emit('weapon:switch', { from: null, to: this.current });
    this.emitAmmo();
  }

  private onPlayerState = (s: PlayerStateSnapshot): void => {
    // Copy scalars only: the snapshot's vectors may be reused by the sender.
    // Validate before storing: a single non-finite velocity from another system
    // would propagate into the bob phase and every spring, and NaN is
    // absorbing — the weapon would vanish for the rest of the session with no
    // error to trace it back to.
    const speed = Math.hypot(s.velocity.x, s.velocity.z);
    this.speed = Number.isFinite(speed) ? speed : 0;
    this.sprintingRaw = s.sprinting;
    this.crouching = s.crouching;
    this.grounded = s.grounded;
  };

  private onPlayerLand = (e: { velocity: number }): void => {
    // Guarded for the same reason as `player:state` velocity: a NaN impulse
    // into the landing spring can never be integrated back out, and that
    // spring feeds the viewmodel's Y offset — the weapon would simply vanish
    // for the rest of the session with nothing to trace it back to.
    this.view?.onLand(finiteOr(e.velocity, 0));
  };

  // ── main update ────────────────────────────────────────────────────────────

  update(ctx: EngineContext, frame: FrameContext): void {
    const view = this.view;
    if (!view) return;
    const spec = WEAPON_SPECS[this.current];

    if (this.firstFrame) {
      // Deferred out of init(): every other system has now initialised, so the
      // material library is available to skin the rig and every subscriber
      // will actually receive the opening state broadcast.
      //
      // This runs BEFORE the dt guard on purpose. `Engine` delivers dt === 0
      // for every frame while paused, so a game that boots straight into a
      // pause menu would otherwise never build a weapon, never broadcast, and
      // leave the whole system silently dead with no error anywhere.
      this.firstFrame = false;
      view.setWeapon(this.current, spec);
      this.broadcastState();
    }

    if (!(frame.dt > 0) || !Number.isFinite(frame.dt)) return;
    const dt = frame.dt;

    this.sinceLastShot += dt;
    const patternReset = Math.max(PATTERN_RESET, shotInterval(spec) * PATTERN_RESET_RATIO);
    if (this.sinceLastShot > patternReset) this.shotIndex = 0;
    if (this.semiBuffer >= 0) {
      this.semiBuffer -= dt;
      if (this.semiBuffer < 0) this.semiBuffer = -1;
    }

    this.updateSwitching(ctx);
    this.updateReload(ctx, dt, spec);
    // Retry a reload that arrived mid-unwind, now that the pose has settled.
    if (this.reloadQueued && !view.reloadUnwinding) this.startReload(ctx);
    // Sampled ONCE, and only after the switch and reload state for this frame
    // has settled — `lowReadyIntent` reads `reloadTimer`, which `updateReload`
    // mutates. Sampling earlier meant the pose could be told "not reloading,
    // drop to the low ready" on the very frame every other consumer had
    // already switched to "reloading".
    this.lowReadyNow = this.lowReadyIntent(ctx);
    this.updateStance(dt);
    this.updateAds(ctx, dt, spec);
    this.updateSpread(dt, spec);
    this.updateFiring(ctx, dt, spec);
    this.updateRecoil(ctx, dt, spec);

    // Gated here on THIS frame's eased value; the viewmodel's own copy is a
    // frame stale, which would let an inspect start on the frame ADS begins.
    if (ctx.input.pressed('inspect') && this.adsEase < 0.2) view.inspect();

    // ── drive the viewmodel ─────────────────────────────────────────────────
    const st = this.vmState;
    st.speed = this.speed;
    // The eased value, not the raw intent: the viewmodel and the firing gate
    // are then literally the same number.
    st.lowReady = this.lowReady;
    st.crouching = this.crouching;
    st.grounded = this.grounded;
    st.ads = this.adsEase;
    // Angular velocity rather than per-frame delta, so the sway spring behaves
    // identically at any framerate.
    st.lookYawRate = finiteOr(-ctx.input.lookDelta.x / dt, 0);
    st.lookPitchRate = finiteOr(-ctx.input.lookDelta.y / dt, 0);
    st.viewKickPitch = this.mirrorPitch;
    st.viewKickYaw = this.mirrorYaw;
    view.update(dt, frame.elapsed, st);
  }

  // ── stance ─────────────────────────────────────────────────────────────────

  /**
   * Whether the weapon should be in the sprint low-ready.
   *
   * Fire intent instantly cancels it: in mainstream shooters, pulling the trigger
   * mid-sprint raises the weapon rather than doing nothing, and the ~93 ms
   * that takes (rate 13.0 down to the 0.30 firing gate) is the real cost of
   * sprinting.
   */
  private lowReadyIntent(ctx: EngineContext): boolean {
    if (!this.sprintingRaw || this.speed < 1.6) return false;
    if (ctx.input.isDown('fire') || ctx.input.isDown('ads')) return false;
    if (this.reloadTimer >= 0) return false;
    return true;
  }

  private updateStance(dt: number): void {
    // Same asymmetric rates as the ViewModel so the firing gate and the visible
    // pose always agree: slow into the jog, fast out of it.
    const target = this.lowReadyNow ? 1 : 0;
    const rate = target > this.lowReady ? 7.5 : 13.0;
    this.lowReady += (target - this.lowReady) * (1 - Math.exp(-rate * dt));
  }

  // ── weapon switching ───────────────────────────────────────────────────────

  private updateSwitching(ctx: EngineContext): void {
    const view = this.view;
    if (!view) return;
    // Bail before touching `primaryIndex`: a press that cannot start a switch
    // must not advance the cycle, or spamming `1` mid-transition silently
    // skips weapons.
    if (view.switching) return;

    if (ctx.input.pressed('weapon1')) {
      // Two keys, five weapons: `1` selects the current primary, or cycles to
      // the next one if a primary is already up.
      if (this.current === SIDEARM) {
        this.switchTo(PRIMARY_POOL[this.primaryIndex]);
      } else {
        this.primaryIndex = (this.primaryIndex + 1) % PRIMARY_POOL.length;
        this.switchTo(PRIMARY_POOL[this.primaryIndex]);
      }
    } else if (ctx.input.pressed('weapon2')) {
      this.switchTo(SIDEARM);
    }
  }

  /** Begin a lower/swap/raise transition to another weapon. */
  switchTo(id: WeaponId): void {
    const view = this.view;
    const ctx = this.ctx;
    if (!view || !ctx || id === this.current || view.switching) return;
    // Before the first update there is no rig to lower, so animating a swap
    // would emit 'weapon:switch' for a weapon that was never on screen. Just
    // retarget; the deferred first-frame setup will raise the right weapon.
    if (this.firstFrame) {
      this.current = id;
      const idx = PRIMARY_POOL.indexOf(id);
      if (idx >= 0) this.primaryIndex = idx;
      return;
    }

    // `notify` is true: an aborted reload must still close out, or a HUD
    // reload bar started by 'weapon:reload:start' hangs forever.
    this.cancelReload(ctx, true);
    const from = this.current;
    // Keep the `weapon1` cycle in step with a direct external selection,
    // otherwise the next cycle press skips a weapon.
    const poolIndex = PRIMARY_POOL.indexOf(id);
    if (poolIndex >= 0) this.primaryIndex = poolIndex;
    view.beginSwitch(() => {
      this.current = id;
      view.setWeapon(id, WEAPON_SPECS[id]);
      // Reset the transients that are meaningfully per-WEAPON: spread bloom,
      // the recoil pattern index, the fire-rate accumulator and the aim state.
      // The recoil OFFSET and its recovery debt deliberately survive, because
      // they describe where the camera currently is, and snapping that on a
      // weapon swap would yank the player's aim.
      this.shotIndex = 0;
      this.bloom = 0;
      this.shotCooldown = 0;
      this.burstRemaining = 0;
      this.semiBuffer = -1;
      this.reloadQueued = false;
      this.tracerCounter = 0;
      this.adsRaw = 0;
      this.adsEase = 0;
      this.adsIn = false;
      // This callback runs from inside ViewModel.update, AFTER the caller has
      // already snapshotted vmState — so clear the snapshot too, or the newly
      // raised weapon is posed for one frame using the old weapon's aim state.
      this.vmState.ads = 0;
      ctx.bus.emit('weapon:switch', { from, to: id });
      this.emitAmmo();
    });
  }

  // ── reload ─────────────────────────────────────────────────────────────────

  private updateReload(ctx: EngineContext, dt: number, spec: WeaponSpec): void {
    if (ctx.input.pressed('reload')) this.startReload(ctx);

    if (this.reloadTimer < 0) return;

    // Sprinting aborts a reload — you cannot seat a magazine at a dead run.
    if (this.sprintingRaw && this.speed > 3.2) {
      this.cancelReload(ctx, true);
      return;
    }

    this.reloadTimer -= dt;
    if (this.reloadTimer > 0) return;

    this.reloadTimer = -1;
    // Match the abort path: a trigger press buffered before the reload must not
    // discharge a round the instant the magazine seats.
    this.burstRemaining = 0;
    this.semiBuffer = -1;
    const a = this.ammo[this.current];
    const take = Math.max(0, Math.min(a.reserve, spec.magSize - a.mag));
    a.mag += take;
    a.reserve -= take;
    ctx.bus.emit('weapon:reload:end', { weapon: this.current });
    this.emitAmmo();
  }

  /** Start a reload if one is possible. Safe to call redundantly. */
  startReload(context?: EngineContext): void {
    const ctx = context ?? this.ctx;
    const view = this.view;
    if (!ctx || !view || this.reloadTimer >= 0 || view.switching) return;
    // A previous abort is still unwinding its pose. Buffer the request rather
    // than dropping it or starting a timeline the viewmodel cannot match; the
    // window is at most RELOAD_FADE (180 ms) and `update` retries every frame.
    if (view.reloadUnwinding) {
      this.reloadQueued = true;
      return;
    }
    this.reloadQueued = false;
    // Same precondition the abort check uses. Without it, a reload keyed while
    // sprinting starts and cancels on the same tick, emitting a start/end pair
    // that makes a HUD bar flicker.
    if (this.sprintingRaw && this.speed > 3.2) return;
    const spec = WEAPON_SPECS[this.current];
    const a = this.ammo[this.current];
    if (a.mag >= spec.magSize || a.reserve <= 0) return;

    // A round left in the chamber means the bolt does not have to be cycled,
    // which is the entire reason a tactical reload is faster.
    this.reloadTactical = a.mag > 0;
    this.reloadDur = this.reloadTactical ? spec.reloadTacticalTime : spec.reloadTime;
    this.reloadTimer = this.reloadDur;
    view.interrupt();
    view.beginReload(this.reloadDur, this.reloadTactical);
    ctx.bus.emit('weapon:reload:start', {
      weapon: this.current,
      duration: this.reloadDur,
      tactical: this.reloadTactical,
    });
  }

  private cancelReload(ctx: EngineContext, notify: boolean): void {
    if (this.reloadTimer < 0) return;
    this.reloadTimer = -1;
    this.reloadQueued = false;
    // A burst queued before the reload must not resume afterwards with no
    // trigger pull behind it.
    this.burstRemaining = 0;
    this.semiBuffer = -1;
    this.view?.cancelReload();
    // The 'end' event is the HUD's cue to stop showing a reload bar; it fires
    // on an abort too, otherwise the bar would hang forever.
    if (notify) ctx.bus.emit('weapon:reload:end', { weapon: this.current });
  }

  // ── aim down sights ────────────────────────────────────────────────────────

  private updateAds(ctx: EngineContext, dt: number, spec: WeaponSpec): void {
    const view = this.view;
    if (!view) return;

    // Raising the sights cancels an inspect, the same way firing does — and
    // ADS is gated until that fade completes. Without the gate the player
    // reaches full zoom and full ADS accuracy while the weapon is swung ~89°
    // across the frame with its reticle hidden.
    const adsHeld = ctx.input.isDown('ads');
    if (adsHeld && view.inspecting) view.interrupt();

    const want =
      adsHeld &&
      this.reloadTimer < 0 &&
      !view.switching &&
      !view.inspecting &&
      this.lowReady < 0.5;

    if (want !== this.adsIn) {
      // Re-solve the linear parameter so the EASED value is continuous across a
      // direction change. Without this, flicking the aim button mid-transition
      // pops the weapon, because the in and out curves disagree everywhere
      // except the endpoints.
      this.adsRaw = want
        ? 1 - Math.cbrt(1 - clamp(this.adsEase, 0, 1))
        : Math.cbrt(clamp(this.adsEase, 0, 1));
      this.adsIn = want;
    }

    const rate = dt / Math.max(0.02, spec.adsTime);
    this.adsRaw = want ? Math.min(1, this.adsRaw + rate) : Math.max(0, this.adsRaw - rate);
    // Ease-out going in (fast off the mark, gentle settle = "snappy"), ease-in
    // coming out (the weapon drops away immediately when you release).
    this.adsEase = this.adsIn ? easeOutCubic(this.adsRaw) : easeInCubic(this.adsRaw);

    if (this.adsIn !== this.adsAnnounced) {
      this.adsAnnounced = this.adsIn;
      ctx.bus.emit('weapon:ads', { active: this.adsIn, weapon: this.current });
    }

    // ── FOV ─────────────────────────────────────────────────────────────────
    // `spec.fovScale` is the weapon's authored zoom. `settings.adsFovScale` is
    // the player's GLOBAL aim-FOV preference, expressed against the engine
    // default of 0.62; treating it as a relative multiplier keeps a settings
    // slider meaningful across all five weapons instead of being dead data,
    // while preserving the relative zoom between, say, a reflex and a scope.
    // The setting is a MULTIPLIER APPLIED TO FOV, so a LARGER value means LESS
    // zoom and 1.0 means none at all. The strength therefore has to be derived
    // from how far the setting sits below 1.0, not from its raw magnitude:
    //   adsFovScale 1.00 -> pref 0.00 -> no zoom on any weapon
    //   adsFovScale 0.62 -> pref 1.00 -> each weapon's authored zoom (default)
    //   adsFovScale 0.30 -> pref 1.84, clamped to the 1.60 maximum
    const pref = clamp(
      (1 - finiteOr(ctx.settings.adsFovScale, DEFAULT_ADS_FOV_SCALE)) / (1 - DEFAULT_ADS_FOV_SCALE),
      0,
      1.6,
    );
    // Floor the zoom: the sniper's 0.42 scale at maximum preference would
    // otherwise reach 0.072, and a scale of <= 0 is a degenerate projection.
    const zoom = Math.max(0.15, 1 + (spec.fovScale - 1) * this.adsEase * pref);

    // The world camera takes the full zoom; the viewmodel camera takes 45% of
    // it. Zooming both equally makes the weapon balloon and the sight picture
    // feel claustrophobic — the split is what shipped games do.
    const cam = ctx.camera;
    const targetFov = finiteOr(ctx.settings.fov, 80) * zoom;
    if (Math.abs(cam.fov - targetFov) > 1e-3) {
      cam.fov = targetFov;
      cam.updateProjectionMatrix();
    }
    const vcam = ctx.viewCamera;
    const targetViewFov = this.viewFovBase * (1 + (zoom - 1) * 0.45);
    if (Math.abs(vcam.fov - targetViewFov) > 1e-3) {
      vcam.fov = targetViewFov;
      vcam.updateProjectionMatrix();
    }
  }

  // ── spread ─────────────────────────────────────────────────────────────────

  private updateSpread(dt: number, spec: WeaponSpec): void {
    // Linear decay, matching the mainstream model: bloom is a budget that bleeds at a
    // fixed rate, so the break-even fire rate is a number a player can feel.
    this.bloom = Math.max(0, this.bloom - spec.spreadDecay * dt);
  }

  /**
   * Total cone half-angle for the next round, in radians.
   *
   *   base cone (hip → ADS, by aim progress)
   * + accumulated bloom (reduced 65% when fully aimed)
   * + movement penalty: `speed/6.5` of 55% of the hip cone, plus a flat 90% of
   *   the hip cone while airborne, all scaled by 0.55 when crouched and by
   *   0.25 when fully aimed
   *
   * The sum is then capped at `spreadHip × 2.6 + spreadAds`, so no combination
   * of bloom and movement can open the cone without limit.
   */
  private currentCone(spec: WeaponSpec): number {
    const base = spec.spreadHip + (spec.spreadAds - spec.spreadHip) * this.adsEase;
    const bloom = this.bloom * (1 - 0.65 * this.adsEase);

    let move = clamp(this.speed / 6.5, 0, 1) * spec.spreadHip * 0.55;
    if (!this.grounded) move += spec.spreadHip * 0.9;
    if (this.crouching) move *= 0.55;
    move *= 1 - 0.75 * this.adsEase;

    return Math.min(base + bloom + move, spec.spreadHip * 2.6 + spec.spreadAds);
  }

  /** Current cone half-angle — for a HUD crosshair that reflects real spread. */
  get spreadRadians(): number {
    return this.currentCone(WEAPON_SPECS[this.current]);
  }

  // ── firing ─────────────────────────────────────────────────────────────────

  private canFire(view: ViewModel): boolean {
    if (this.reloadTimer >= 0) return false;
    if (view.switching) return false;
    // The weapon has to be up before it can go off.
    if (this.lowReady > 0.30) return false;
    return this.ammo[this.current].mag > 0;
  }

  private updateFiring(ctx: EngineContext, dt: number, spec: WeaponSpec): void {
    const view = this.view;
    if (!view) return;
    const input = ctx.input;
    const interval = shotInterval(spec);

    // ── trigger discipline ──────────────────────────────────────────────────
    if (input.pressed('fire')) {
      if (spec.burst === 1) {
        // Buffer the press: if the trigger is pulled a few milliseconds before
        // the weapon is ready, honour it rather than eating the input.
        this.semiBuffer = SEMI_BUFFER;
        // NOTE: `burst > 1` is fully implemented but unused by the shipped
      // loadout — every weapon is either full-auto (0) or single-shot (1).
    } else if (spec.burst > 1 && this.burstRemaining <= 0) {
        this.burstRemaining = spec.burst;
      }
      const a = this.ammo[this.current];
      if (a.mag <= 0 && this.reloadTimer < 0 && !view.switching) {
        ctx.bus.emit('weapon:dryfire', { weapon: this.current });
        // Pulling the trigger on an empty magazine is the universal cue to
        // reload. (This branch is on the press EDGE, so it fires once.)
        if (a.reserve > 0) this.startReload(ctx);
      }
    }

    let wants: boolean;
    if (spec.burst === 0) wants = input.isDown('fire');
    else if (spec.burst === 1) wants = this.semiBuffer >= 0;
    else wants = this.burstRemaining > 0;

    // ── the accumulator ─────────────────────────────────────────────────────
    // Cooldown is DECREMENTED by dt and INCREMENTED by exactly one interval per
    // round fired — never assigned. The sub-frame remainder therefore carries
    // forward, so a 950 rpm weapon fires 15.833 rounds a second at 30 fps, at
    // 144 fps, and across a frame that took 90 ms. Assigning `cooldown =
    // interval` instead (the naive "can I shoot yet?" boolean) quantises the
    // rate to the frame time and silently drops rounds during a hitch.
    this.shotCooldown -= dt;
    const blocked = !wants || !this.canFire(view);
    if (blocked) {
      // Do not bank credit while the trigger is off or the weapon is busy, or
      // the instant a reload finishes the gun would dump its backlog.
      if (this.shotCooldown < 0) this.shotCooldown = 0;
      return;
    }
    // Cap catch-up so an enormous frame cannot empty a magazine at once. The
    // engine already clamps dt to 100 ms, so this never binds in practice.
    if (this.shotCooldown < -interval * 3) this.shotCooldown = -interval * 3;

    let fired = 0;
    while (this.shotCooldown <= 0 && fired < MAX_SHOTS_PER_FRAME) {
      if (!this.canFire(view)) break;
      this.fireOne(ctx, view, spec);
      this.shotCooldown += interval;
      fired++;

      if (spec.burst === 1) {
        this.semiBuffer = -1;
        break;
      }
      if (spec.burst > 1) {
        this.burstRemaining--;
        if (this.burstRemaining <= 0) {
          // Enforced pause between bursts, otherwise a burst weapon is just a
          // full-auto with extra steps.
          this.shotCooldown += interval * 2.2;
          break;
        }
      }
    }
  }

  private fireOne(ctx: EngineContext, view: ViewModel, spec: WeaponSpec): void {
    const ballistics = this.ballistics;
    if (!ballistics) return;

    const cam = ctx.camera;
    cam.updateMatrixWorld();
    _origin.setFromMatrixPosition(cam.matrixWorld);
    // Rounds originate at the eye and travel down the camera axis, so they go
    // exactly where the crosshair is. The muzzle is used only as the visual
    // origin of the tracer, which is how every first-person shooter since
    // no shooter has resolved the muzzle-vs-crosshair parallax problem.
    _aim.set(0, 0, -1).transformDirection(cam.matrixWorld);
    _right.set(1, 0, 0).transformDirection(cam.matrixWorld);
    _up.set(0, 1, 0).transformDirection(cam.matrixWorld);
    view.getMuzzleWorld(_muzzle);

    // Publishing the seed lets another system reproduce this shot's pellet
    // ROSETTE — the ring layout and its jitter — given the same cone angle and
    // basis. The cone itself has no field in `GameEvents`, so a consumer that
    // needs pixel-exact pellet directions must also read `spreadRadians`, which
    // is why the fire event is emitted before this round's bloom is applied.
    const seed = (this.rng.next() * 4294967295) >>> 0;
    this.rng.reseed(seed);

    const ammo = this.ammo[this.current];
    ammo.mag--;

    const tracer = this.tracerCounter % TRACER_EVERY === 0;
    this.tracerCounter++;

    const cone = this.currentCone(spec);

    // Emitted BEFORE the round is resolved, so listeners observe cause then
    // effect: 'weapon:fire' precedes the 'impact', 'tracer' and 'damage:dealt'
    // it produces. Also before bloom is applied, so a listener reading
    // `spreadRadians` from inside the handler sees the cone this round actually
    // used rather than one already widened by this round's own bloom.
    ctx.bus.emit('weapon:fire', {
      origin: _origin.clone(),
      direction: _aim.clone(),
      weapon: this.current,
      seed,
    });

    // Ammo before the round resolves, so a HUD never shows the old count
    // alongside this shot's own impacts.
    this.emitAmmo();

    ballistics.fireShot(spec, _origin, _aim, _right, _up, cone, _muzzle, this.rng, tracer);

    this.applyRecoil(ctx, spec);
    this.sinceLastShot = 0;
    view.onFire(this.adsEase);
    this.bloom = Math.min(this.bloom + spec.spreadPerShot, spec.spreadPerShot * BLOOM_CAP_SHOTS);

    // Big-bore weapons shake the whole camera, not just the sight picture.
    if (spec.pellets > 1 || spec.caliber === '.338') {
      ctx.bus.emit('camera:shake', {
        amplitude: spec.caliber === '.338' ? 0.030 : 0.022,
        duration: 0.20,
        frequency: 34,
      });
    }

    if (ammo.mag <= 0 && ammo.reserve > 0) this.startReload(ctx);
  }

  // ── recoil ─────────────────────────────────────────────────────────────────

  private applyRecoil(ctx: EngineContext, spec: WeaponSpec): void {
    const pattern = spec.recoil;
    const vi = Math.max(0, Math.min(this.shotIndex, pattern.vertical.length - 1));
    const hi = Math.max(0, Math.min(this.shotIndex, pattern.horizontal.length - 1));
    // Advance the pattern. Without this every shot replays index 0 and all the
    // hand-authored sequencing — the entire reason the patterns exist — is dead.
    this.shotIndex++;

    // Jitter is deliberately SMALL: ±7% vertical, ±16% horizontal. Enough that
    // two magazines never print the same hole, not enough to stop the pattern
    // being learnable. Randomising more than this throws away the entire point
    // of authoring a pattern.
    //
    // Drawn from a DEDICATED stream: the ballistics RNG is reseeded per shot
    // from the published seed and then consumed a data-dependent number of
    // times by the penetration walk, so sharing it would make the recoil jitter
    // depend on what the bullet hit.
    let v = (pattern.vertical[vi] ?? 0) * (1 + this.recoilRng.bipolar() * 0.07);
    let h = (pattern.horizontal[hi] ?? 0) * (1 + this.recoilRng.bipolar() * 0.16);
    // A malformed pattern must not poison the view offset forever: NaN survives
    // every subsequent multiply and clamp() passes it straight through.
    if (!Number.isFinite(v)) v = 0;
    if (!Number.isFinite(h)) h = 0;

    // Bracing modifiers.
    const braced = 1 - 0.22 * this.adsEase;
    v *= braced;
    h *= braced;
    if (this.crouching) {
      v *= 0.88;
      h *= 0.88;
    }
    if (!this.grounded) {
      v *= 1.5;
      h *= 1.6;
    }

    this.viewKick.pitch = clamp(this.viewKick.pitch + v, -MAX_VIEW_KICK, MAX_VIEW_KICK);
    this.viewKick.yaw = clamp(this.viewKick.yaw + h, -MAX_VIEW_KICK, MAX_VIEW_KICK);

    // Only `recoverFraction` of the kick is owed back. The remainder is
    // permanent climb the player must physically compensate for — that is the
    // knob that decides whether a weapon "walks" under sustained fire.
    // Clamped alongside `viewKick`. `owed` can never legitimately exceed the
    // offset it is scheduled to walk back, and letting it would drive the view
    // through zero and out the other side.
    this.owedPitch = clamp(this.owedPitch + v * pattern.recoverFraction, -MAX_VIEW_KICK, MAX_VIEW_KICK);
    this.owedYaw = clamp(this.owedYaw + h * pattern.recoverFraction, -MAX_VIEW_KICK, MAX_VIEW_KICK);
    this.recoverDelay = Math.min(RECOVER_HOLD_MAX, shotInterval(spec) * RECOVER_HOLD_RATIO);

    // Cosmetic mirror: unlike the aim-affecting kick this always returns to
    // zero, so the viewmodel never ends up permanently canted after a player
    // has manually compensated for a long burst.
    this.mirrorPitch += v;
    this.mirrorYaw += h;

    ctx.bus.emit('camera:kick', { pitch: v, yaw: h, duration: pattern.recoverTime });
  }

  /**
   * Recoil recovery, interruptible by the player.
   *
   * THE MODEL. Recoil is applied to `viewKick`, an offset the Player adds to
   * its own orientation. `owedPitch`/`owedYaw` track how much of that offset is
   * still scheduled to be walked back automatically.
   *
   * The important part is what happens when the player fights the recoil. A
   * naive implementation lets the view recover to zero regardless, so pulling
   * down 3° during a burst and then stopping leaves the crosshair 3° BELOW
   * where you started as the spring returns — the notorious "gun springs back
   * past your correction" bug.
   *
   * Instead, downward mouse input is subtracted from the OWED amount without
   * touching `viewKick`. The player's pull moves the view down exactly once
   * (through their own base pitch, which the Player system applies), and the
   * automatic recovery that would have duplicated that movement is cancelled.
   * Compensate perfectly and the crosshair is exactly back on target with no
   * spring-back at all; compensate not at all and the view walks itself back
   * down over `recoverTime`. Anything in between blends smoothly, which is
   * precisely how a modern military shooter feels to shoot.
   */
  private updateRecoil(ctx: EngineContext, dt: number, spec: WeaponSpec): void {
    // Player's intended orientation change this frame, in radians. Mouse-down
    // is +lookDelta.y and pitches the view DOWN, hence the negation.
    //
    // Validated, like `player:state` velocity, because this is the last
    // untrusted external value that reaches persistent state. NaN here is
    // unrecoverable: it flows into `owed`, then into `viewKick`, and `clamp`
    // passes NaN straight through (both of its comparisons are false), so the
    // camera offset the Player system adds to its rotation would be NaN for the
    // rest of the session with nothing to trace it back to.
    const pitchDelta = finiteOr(-ctx.input.lookDelta.y, 0);
    const yawDelta = finiteOr(-ctx.input.lookDelta.x, 0);

    // Between bursts the leftover offset also unwinds — see `absorb`. The rate
    // ramps in only once the residual is large enough to be worth paying for.
    const quiet = this.sinceLastShot > RESIDUAL_BLEED_DELAY;
    this.owedPitch = this.absorb(this.owedPitch, pitchDelta, quiet, 'pitch');
    this.owedYaw = this.absorb(this.owedYaw, yawDelta, quiet, 'yaw');

    // Cosmetic mirror always decays, independent of the player.
    const mirrorTau = Math.max(0.03, spec.recoil.recoverTime * 0.42);
    const mk = 1 - Math.exp(-dt / mirrorTau);
    this.mirrorPitch -= this.mirrorPitch * mk;
    this.mirrorYaw -= this.mirrorYaw * mk;

    // `viewKick` is a public mutable object, so `readonly` protects the
    // reference and not the fields. If anything outside this system has written
    // a non-finite value into it, no amount of recovery arithmetic will ever
    // clear it — reset instead of accumulating NaN forever.
    if (!Number.isFinite(this.viewKick.pitch) || !Number.isFinite(this.viewKick.yaw)) {
      this.viewKick.pitch = 0;
      this.viewKick.yaw = 0;
      this.owedPitch = 0;
      this.owedYaw = 0;
    }

    if (this.recoverDelay > 0) {
      this.recoverDelay -= dt;
      return;
    }

    // Exponential approach. τ = recoverTime / 3 puts ~95% of the recovery
    // inside the authored recoverTime, and being exponential it is exactly
    // framerate-independent — a fixed per-frame lerp would not be.
    const tau = Math.max(0.02, spec.recoil.recoverTime / 3);
    const k = 1 - Math.exp(-dt / tau);
    const dp = this.owedPitch * k;
    const dy = this.owedYaw * k;
    this.viewKick.pitch -= dp;
    this.owedPitch -= dp;
    this.viewKick.yaw -= dy;
    this.owedYaw -= dy;
  }

  /**
   * Apply one axis of player counter-input to the recoil state.
   *
   * Two distinct effects, and the split between them is the whole model:
   *
   * 1. COUNTER-INPUT CANCELS THE OWED RECOVERY (no bleed, during a burst).
   *    Input opposing the outstanding OWED RECOVERY — not the accumulated
   *    offset, which can carry the opposite sign once a horizontal pattern
   *    reverses — is subtracted from `owed`, and `viewKick` is left alone. The
   *    player's pull moves the view down exactly once, through their own base
   *    pitch, and the automatic recovery that would have duplicated that
   *    movement is cancelled. Input in the same direction as `owed` is the
   *    player riding the recoil and must not inflate the budget, so it is
   *    ignored.
   *
   * 2. EXCESS COUNTER-INPUT UNWINDS THE SETTLED RESIDUAL (between bursts).
   *    Once `owed` is exhausted, `viewKick` still holds whatever the player
   *    compensated for manually. That residual is aim-neutral — their base
   *    pitch is lower by exactly the same amount — but it is not FREE: base
   *    pitch is clamped to ±90°, so a residual that ratchets up over a match
   *    eats the bottom of the player's look range. Leftover counter-input
   *    therefore unwinds it. This DOES make that input travel further than the
   *    player asked for, so it is rationed: silent below `RESIDUAL_SOFT`,
   *    ramping to `RESIDUAL_BLEED_MAX` only as the residual approaches
   *    `RESIDUAL_HARD`, and disabled for `RESIDUAL_BLEED_DELAY` after a shot so
   *    it can never interfere with effect 1 during a burst.
   */
  private absorb(owed: number, delta: number, quiet: boolean, axis: 'pitch' | 'yaw'): number {
    if (delta === 0) return owed;
    let remaining = Math.abs(delta);
    let next = owed;

    // (1) Cancel the owed recovery. Gated on the sign of OWED, independently of
    //     the accumulated kick: on a pattern whose horizontal component
    //     reverses (every pattern in Specs.ts does) the two genuinely disagree
    //     for a while, and testing the wrong one silently stops the player's
    //     counter-input from cancelling anything.
    if (owed !== 0 && Math.sign(delta) !== Math.sign(owed)) {
      const eaten = Math.min(Math.abs(owed), remaining);
      next = owed - Math.sign(owed) * eaten;
      remaining -= eaten;
    }

    // (2) Unwind the settled residual with whatever counter-input is left.
    //     Gated on the sign of the KICK, again independently.
    const kick = axis === 'pitch' ? this.viewKick.pitch : this.viewKick.yaw;
    if (!quiet || remaining <= 0 || kick === 0 || Math.sign(delta) === Math.sign(kick)) {
      return next;
    }
    const mag = Math.abs(kick);
    if (mag <= RESIDUAL_SOFT) return next;
    // Guarded like every other divisor here, so retuning the two constants to
    // the same value cannot silently produce 0/0 and poison `viewKick`.
    const ramp = Math.min(1, (mag - RESIDUAL_SOFT) / Math.max(1e-4, RESIDUAL_HARD - RESIDUAL_SOFT));
    const unwind = Math.min(mag, remaining * RESIDUAL_BLEED_MAX * ramp);
    const settled = kick - Math.sign(kick) * unwind;
    if (axis === 'pitch') this.viewKick.pitch = settled;
    else this.viewKick.yaw = settled;
    return next;
  }

  // ── events / queries ───────────────────────────────────────────────────────

  private emitAmmo(): void {
    this.emitAmmoFor(this.current);
  }

  private emitAmmoFor(id: WeaponId): void {
    const ctx = this.ctx;
    const a = this.ammo[id];
    if (!ctx || !a) return;
    ctx.bus.emit('weapon:ammo', {
      mag: a.mag,
      magSize: WEAPON_SPECS[id].magSize,
      reserve: a.reserve,
      weapon: id,
    });
  }

  /**
   * Register a collider that this weapon's rounds must ignore — in practice the
   * player's own body/capsule, which otherwise absorbs every shot in the
   * no-physics fallback path. Safe to call before `init`; the registration is
   * buffered and replayed.
   */
  addRaycastExclusion(object: THREE.Object3D): void {
    // Always recorded here, whether or not Ballistics exists yet: this list is
    // the canonical one, and `dispose()` throws the Ballistics instance away.
    // Forwarding only, without recording, meant every exclusion registered
    // after init was silently lost on a dispose/init cycle — leaving the player
    // shooting their own collider, which is the exact failure this prevents.
    if (!this.pendingExclusions.includes(object)) this.pendingExclusions.push(object);
    this.ballistics?.addExclusion(object);
  }

  /**
   * World-space muzzle position, written into `out`. The 'weapon:fire' event
   * carries the EYE position as its origin (rounds are cast from the camera so
   * they land on the crosshair), so any effect that needs the actual barrel —
   * smoke, heat haze, a third-person tracer origin, positional fire audio —
   * should read it from here.
   */
  getMuzzleWorld(out: THREE.Vector3): THREE.Vector3 {
    if (!this.view) return out.copy(this.ctx?.camera.position ?? out);
    return this.view.getMuzzleWorld(out);
  }

  /**
   * Normalised timing of the reload animation's audible beats, as fractions of
   * the `duration` carried by 'weapon:reload:start'. Re-exported from the
   * viewmodel (the single source of truth) so the audio system can schedule
   * `reload_magout` / `reload_magin` / `reload_charge` without duplicating —
   * and then drifting out of sync with — the animation's private constants.
   */
  static readonly RELOAD_BEATS = RELOAD_BEATS;

  get weapon(): WeaponId {
    return this.current;
  }
  get spec(): WeaponSpec {
    return WEAPON_SPECS[this.current];
  }
  get magAmmo(): number {
    return this.ammo[this.current]?.mag ?? 0;
  }
  get reserveAmmo(): number {
    return this.ammo[this.current]?.reserve ?? 0;
  }
  /** Eased 0..1 aim-down-sights progress, for HUD crosshair fades. */
  get adsProgress(): number {
    return this.adsEase;
  }
  get isReloading(): boolean {
    return this.reloadTimer >= 0;
  }
  /** 0..1 reload progress, for a HUD reload bar. */
  get reloadProgress(): number {
    if (this.reloadTimer < 0 || this.reloadDur <= 0) return 0;
    return 1 - this.reloadTimer / this.reloadDur;
  }
  get isSwitching(): boolean {
    return this.view?.switching ?? false;
  }

  /** Top up a weapon's reserve, for pickups and debug. */
  giveAmmo(id: WeaponId, rounds: number): void {
    const a = this.ammo[id];
    if (!a) return;
    // `Math.max(0, NaN)` is NaN, which would propagate into the magazine on the
    // next reload and be published to every ammo listener.
    // Ceilinged at twice the loadout allowance so a pickup loop cannot inflate
    // the reserve without bound.
    const cap = WEAPON_SPECS[id].reserveAmmo * 2;
    a.reserve = Math.min(cap, Math.max(0, a.reserve + finiteOr(rounds, 0)));
    // Report the weapon that actually changed, not whatever is in hand.
    this.emitAmmoFor(id);
  }

  dispose(): void {
    const ctx = this.ctx;
    if (ctx) {
      // Close out anything still in flight, or a HUD reload bar hangs and the
      // ADS state machine is left believing the player is still aiming.
      this.cancelReload(ctx, true);
      if (this.adsAnnounced) {
        this.adsAnnounced = false;
        ctx.bus.emit('weapon:ads', { active: false, weapon: this.current });
      }
      // Hand the cameras back. The world camera goes to the CURRENT settings
      // value, because that is what `updateAds` derives from every frame — a
      // runtime FOV change should survive teardown, not be reverted to
      // whatever happened to be set at init.
      // `finiteOr`, matching the guarded read in `updateAds`: `NaN !== NaN` is
      // true, so an unguarded compare here would happily assign NaN.
      const restoreFov = finiteOr(ctx.settings.fov, 80);
      if (ctx.camera.fov !== restoreFov) {
        ctx.camera.fov = restoreFov;
        ctx.camera.updateProjectionMatrix();
      }
      const restoreViewFov = finiteOr(this.viewFovBase, 60);
      if (ctx.viewCamera.fov !== restoreViewFov) {
        ctx.viewCamera.fov = restoreViewFov;
        ctx.viewCamera.updateProjectionMatrix();
      }
    }
    // The published contract tells the Player never to zero `viewKick` itself,
    // so leaving a residual here would strand a permanent camera offset that
    // nothing is allowed to clear.
    this.viewKick.pitch = 0;
    this.viewKick.yaw = 0;
    this.owedPitch = 0;
    this.owedYaw = 0;
    this.mirrorPitch = 0;
    this.mirrorYaw = 0;
    this.recoverDelay = 0;
    // Every per-session transient, not just the recoil offset. Left behind,
    // these produce a genuinely baffling next session: the first shot replays
    // whatever recoil-pattern index the last one ended on, fires through a
    // stale spread cone, and the weapon snaps to the aimed pose on frame two.
    this.shotIndex = 0;
    this.sinceLastShot = 99;
    this.bloom = 0;
    this.shotCooldown = 0;
    this.semiBuffer = -1;
    this.burstRemaining = 0;
    this.tracerCounter = 0;
    this.adsRaw = 0;
    this.adsEase = 0;
    this.adsIn = false;
    this.lowReady = 0;
    this.vmState.ads = 0;
    // Loadout: without this the next session raises whichever weapon the last
    // one happened to end on, and the weapon-cycle key resumes mid-rotation.
    this.current = 'ar_vector';
    this.primaryIndex = 0;
    // Movement mirror. These are only refreshed when a 'player:state' arrives,
    // so a session disposed while sprinting or airborne would start the next
    // one with firing gated behind a phantom sprint and an airborne spread
    // penalty applied to the first shot.
    this.speed = 0;
    this.sprintingRaw = false;
    this.crouching = false;
    this.grounded = true;
    // Random streams, so a run is reproducible from its seed rather than from
    // however many shots the previous session happened to fire.
    this.rng.reseed(0x1f4a3c7d);
    this.recoilRng.reseed(0x7b19d3a5);
    // Re-arm the deferred setup so a dispose()/init() cycle is not silently
    // dead.
    this.firstFrame = true;
    for (const off of this.unsubscribe) off();
    this.unsubscribe.length = 0;
    this.view?.dispose();
    this.ballistics?.dispose();
    // The weapon material cache is module-scoped and shared by every rig, so
    // only the (single) owning system may release it.
    disposeSharedMaterials();
    this.view = null;
    this.ballistics = null;
    this.ctx = null;
  }
}
