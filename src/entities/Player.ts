/**
 * NOISEFIELD — Player character controller.
 *
 * Owns the world camera transform. Everything about the feel of this game
 * passes through this file, so the constants are documented rather than tuned
 * silently.
 *
 * ── Why gravity is doubled ───────────────────────────────────────────────────
 * Real gravity is −9.81 m/s². Shipped shooters run roughly twice that; most
 * land between −18 and −24 m/s². At 1 g a 1 m jump hangs
 * for 0.9 s, which reads as swimming; at 2 g the same jump height takes 0.46 s,
 * which reads as athletic. Jump *height* is the thing the player perceives, so
 * we specify the apex and derive the impulse — never hardcode a velocity, or
 * the height silently changes whenever gravity is retuned.
 *
 * ── Why coyote time and jump buffering ───────────────────────────────────────
 * Both hide the gap between what the player intended and what the simulation
 * saw. Coyote time (90 ms) forgives pressing jump a frame or two after walking
 * off a ledge — without it, running off a crate and pressing jump "does
 * nothing" and feels broken. Jump buffering (120 ms) forgives pressing jump
 * just before landing, so bunny-hopping down stairs works instead of eating
 * every third input. Both windows are short enough to be invisible and long
 * enough to cover human timing error plus a frame of latency.
 *
 * ── Recoil absorption ────────────────────────────────────────────────────────
 * Weapon kick is an additive offset on top of the player's own look angles, not
 * a mutation of them. When the player pulls down during recovery, their input
 * is spent cancelling the *pending* kick before it moves their base aim. That
 * is the mainstream-shooter behaviour: fighting the climb leaves the gun
 * where you pulled it instead of springing back below your point of aim.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  FrameContext,
  IEventBus,
  PlayerStateSnapshot,
  SurfaceType,
  System,
} from '../core/Types';
import {
  CapsuleMoveResult,
  ProbeHit,
  WorldProbe,
  getPlayerSpawn,
  moveCapsule,
  type AITarget,
} from './AI';

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────

const RADIUS = 0.35;
const HEIGHT_STAND = 1.8;
const HEIGHT_CROUCH = 1.1;
const EYE_STAND = 1.7;
const EYE_CROUCH = 1.05;
/** Seconds for a full stand↔crouch transition. */
const CROUCH_TIME = 0.18;

const SPEED_WALK = 4.6;
const SPEED_SPRINT = 7.4;
/** Tac sprint: double-tap sprint. ~20% over a normal sprint, on a short timer. */
const SPEED_TAC_SPRINT = 8.9;
const SPEED_CROUCH = 2.4;
const SPEED_ADS = 2.9;

const ACCEL_GROUND = 55;
const FRICTION = 9;
/** Below this, friction is applied at a constant rate so we stop dead instead
 *  of asymptotically creeping. The classic `stopspeed` term. */
const STOP_SPEED = 1.9;
/** Air control is ~35% of ground authority: reduced, but never zero — mid-air
 *  correction is essential for jump-shots and for landing a vault. */
const AIR_CONTROL = 0.35;
const ACCEL_AIR = ACCEL_GROUND * AIR_CONTROL;

const GRAVITY = -19.6;
const JUMP_APEX = 1.05;
/** v = sqrt(2·g·h). Derived so retuning gravity preserves jump height. */
const JUMP_VELOCITY = Math.sqrt(2 * -GRAVITY * JUMP_APEX);
const COYOTE_TIME = 0.09;
const JUMP_BUFFER = 0.12;

const STEP_HEIGHT = 0.4;
/** cos(50°): steeper than this and the player slides off. */
const SLOPE_LIMIT_COS = Math.cos(50 * Math.PI / 180);

const SLIDE_SPEED = 9.5;
const SLIDE_TIME = 0.9;
/** Exponential decay chosen so SLIDE_SPEED reaches crouch speed in SLIDE_TIME. */
const SLIDE_DECAY = Math.log(SLIDE_SPEED / SPEED_CROUCH) / SLIDE_TIME;
const SLIDE_MIN_ENTRY_SPEED = 4.0;
const SLIDE_COOLDOWN = 0.45;
const SLIDE_ROLL = 4 * Math.PI / 180;
const SLIDE_CAMERA_DROP = 0.12;

const MANTLE_MIN_HEIGHT = 0.4;
const MANTLE_MAX_HEIGHT = 1.4;

const STRAFE_ROLL = 1.2 * Math.PI / 180;
const BOB_AMPLITUDE = 0.021;
const PITCH_LIMIT = 89 * Math.PI / 180;

const MAX_HEALTH = 100;
/** Regenerating health: five quiet seconds, then a fast top-up. */
const REGEN_DELAY = 5.0;
const REGEN_RATE = 24;
const LOW_HEALTH_THRESHOLD = 35;

const STAMINA_MAX = 100;
const STAMINA_SPRINT_DRAIN = 11;
const STAMINA_TAC_DRAIN = 26;
const STAMINA_REGEN = 19;
const STAMINA_REGEN_DELAY = 0.9;

/** Falls faster than this hurt. ~10 m drop. */
const FALL_DAMAGE_SPEED = 14;
const FALL_DAMAGE_PER_MS = 6.5;

const RESPAWN_DELAY = 3.0;
const DOUBLE_TAP_WINDOW = 0.3;
/** Below this the player has left the map; treat it as a death, not a void. */
const WORLD_FLOOR_Y = -60;

// Scratch — the controller runs every frame and allocates nothing.
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _hit = new ProbeHit();
const _up = new THREE.Vector3(0, 1, 0);
const _down = new THREE.Vector3(0, -1, 0);

/** Cheap deterministic 1-D noise for camera shake. Three octaves of sine. */
function shakeNoise(t: number): number {
  return Math.sin(t) * 0.6 + Math.sin(t * 2.37 + 1.3) * 0.3 + Math.sin(t * 4.91 + 2.1) * 0.1;
}

function smoothstep01(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * A three.js camera with identity rotation looks down −Z, so after a yaw of θ
 * about +Y its forward is (−sinθ, 0, −cosθ) and its right is (cosθ, 0, −sinθ).
 * Getting this sign wrong makes W walk backwards, so both are derived here once
 * and used everywhere.
 */
function yawForward(yaw: number, out: THREE.Vector3): THREE.Vector3 {
  return out.set(-Math.sin(yaw), 0, -Math.cos(yaw));
}

function yawRight(yaw: number, out: THREE.Vector3): THREE.Vector3 {
  return out.set(Math.cos(yaw), 0, -Math.sin(yaw));
}

// ─────────────────────────────────────────────────────────────────────────────
// Player
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Implements `AITarget` so the enemy system can consume the player directly
 * through `ctx.get('player')` without a shim.
 */
export class Player implements System, AITarget {
  readonly name = 'player';

  /** Feet position. */
  readonly position = new THREE.Vector3(0, 0, 0);
  readonly velocity = new THREE.Vector3();
  /** World eye position, refreshed after the camera is placed. */
  readonly eye = new THREE.Vector3();
  readonly radius = RADIUS;

  health = MAX_HEALTH;
  readonly maxHealth = MAX_HEALTH;
  stamina = STAMINA_MAX;
  alive = true;

  // Look.
  private yaw = 0;
  private pitch = 0;
  private recoilPitch = 0;
  private recoilYaw = 0;
  private recoilPitchVel = 0;
  private recoilYawVel = 0;
  private recoilOmega = 16;
  private shakeAmp = 0;
  private shakePeak = 0;
  private shakeLeft = 0;
  private shakeDuration = 1;
  private shakeFreq = 22;
  private shakeT = 0;

  // Stance.
  private crouchT = 0;
  private sprinting = false;
  private tacSprint = false;
  private tacSprintTimer = 0;
  private lastSprintPress = -10;
  private ads = false;

  // Slide.
  private sliding = false;
  private slideTimer = 0;
  private slideCooldown = 0;
  private readonly slideDir = new THREE.Vector3(0, 0, 1);
  private slideAirTime = 0;

  // Mantle.
  private mantling = false;
  private mantleT = 0;
  private mantleDuration = 0.5;
  private readonly mantleStart = new THREE.Vector3();
  private readonly mantleEnd = new THREE.Vector3();
  private mantleCooldown = 0;

  // Ground / air.
  private grounded = false;
  private coyote = 0;
  private jumpBuffer = 0;
  private groundSurface: SurfaceType = 'concrete';
  private prevVelocityY = 0;

  // Camera dressing.
  private bobPhase = 0;
  private bobAmount = 0;
  private strafeRoll = 0;
  private landDip = 0;
  private landDipVel = 0;
  private stepSmooth = 0;
  private slideDrop = 0;
  private prevGroundY = 0;
  private prevGroundX = 0;
  private prevGroundZ = 0;
  private hadGround = false;
  private footDistance = 0;

  // Health.
  private timeSinceDamage = 99;
  private timeSinceSprint = 99;
  private respawnTimer = 0;
  /** 0 = healthy, 1 = about to die. Read by the post/HUD systems for the red-out. */
  lowHealthFactor = 0;

  private ctx!: EngineContext;
  private bus!: IEventBus;
  private probe!: WorldProbe;
  private readonly move = new CapsuleMoveResult();
  private readonly spawnPoint = new THREE.Vector3();
  /**
   * The level system may register after us, so the spawn resolved during init
   * can be a guess. Re-resolve once shortly after boot, but only if the player
   * has not taken control yet — teleporting someone mid-stride is worse than a
   * slightly wrong start.
   */
  private spawnFixTimer = 0.6;

  private unsubscribe: Array<() => void> = [];

  // ── System lifecycle ───────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    this.bus = ctx.bus;
    this.probe = new WorldProbe(ctx);

    // YXZ is the only order that gives a clean FPS camera: yaw about world up,
    // then pitch about the rotated right axis, then roll in view space. Any
    // other order accumulates unwanted roll as you look around.
    ctx.camera.rotation.order = 'YXZ';

    getPlayerSpawn(ctx, this.probe, this.spawnPoint);
    this.position.copy(this.spawnPoint);
    this.prevGroundY = this.position.y;
    this.eye.set(this.position.x, this.position.y + EYE_STAND, this.position.z);

    this.unsubscribe.push(
      ctx.bus.on('camera:kick', (p) => this.applyKick(p.pitch, p.yaw, p.duration)),
      ctx.bus.on('camera:shake', (p) => this.applyShake(p.amplitude, p.duration, p.frequency)),
      ctx.bus.on('weapon:ads', (p) => {
        this.ads = p.active;
      }),
    );
  }

  dispose(): void {
    for (const u of this.unsubscribe) u();
    this.unsubscribe = [];
  }

  // ── Weapon-system events ───────────────────────────────────────────────────

  /**
   * Punch the view. The offset is applied instantly (guns snap, they do not
   * ease up) and then recovered by a critically damped spring, so there is no
   * overshoot to fight on top of the recoil itself.
   *
   * SIGN CONTRACT for `camera:kick`, since `Types.ts` does not state one:
   *   • `pitch` POSITIVE kicks the view UP. A rifle should send positive pitch.
   *   • `yaw`   POSITIVE swings the view LEFT (it is a rotation about +Y).
   *   • `duration` is the settle time; it sets the recovery rate. It is clamped
   *     to a floor of 0.05 s (and the resulting ω to 60), so anything snappier
   *     than ≈0.097 s settles at the same rate. Longer means a lazier return.
   */
  private applyKick(pitch: number, yaw: number, duration: number): void {
    this.recoilPitch += pitch;
    this.recoilYaw += yaw;
    // Settling time of a critically damped system is ≈ 5.8/ω.
    this.recoilOmega = THREE.MathUtils.clamp(5.8 / Math.max(duration, 0.05), 5, 60);
  }

  private applyShake(amplitude: number, duration: number, frequency?: number): void {
    // Take the stronger of the two rather than summing: overlapping explosions
    // should not blind the player.
    if (amplitude <= this.shakeAmp) return;
    this.shakePeak = amplitude;
    this.shakeAmp = amplitude;
    this.shakeDuration = Math.max(duration, 0.05);
    this.shakeLeft = this.shakeDuration;
    this.shakeFreq = frequency ?? 22;
  }

  // ── AITarget ───────────────────────────────────────────────────────────────

  get height(): number {
    return THREE.MathUtils.lerp(HEIGHT_STAND, HEIGHT_CROUCH, smoothstep01(this.crouchT));
  }

  /** Called by enemies (and anything else that can hurt the player). */
  damage(amount: number, fromWorld: THREE.Vector3): void {
    if (!this.alive || amount <= 0) return;
    this.health = Math.max(0, this.health - amount);
    this.timeSinceDamage = 0;

    // Direction *toward the attacker*, so the HUD indicator points at the threat.
    const dir = new THREE.Vector3().subVectors(fromWorld, this.eye);
    if (dir.lengthSq() < 1e-8) dir.set(0, 0, -1);
    dir.normalize();
    this.bus.emit('damage:taken', { amount, direction: dir, health: this.health });

    if (this.health <= 0) this.die();
  }

  private die(): void {
    this.alive = false;
    this.respawnTimer = RESPAWN_DELAY;
    // Via endSlide, so a death mid-slide still closes the slide:start it
    // opened — otherwise looping slide audio never stops.
    this.endSlide();
    this.sprinting = false;
    this.bus.emit('ui:notify', { text: 'YOU WERE KILLED', kind: 'warn' });
  }

  private respawn(): void {
    this.alive = true;
    this.health = MAX_HEALTH;
    this.stamina = STAMINA_MAX;
    this.timeSinceDamage = 99;
    this.position.copy(this.spawnPoint);
    this.velocity.set(0, 0, 0);
    this.crouchT = 0;
    this.recoilPitch = 0;
    this.recoilYaw = 0;
    this.shakeAmp = 0;
    this.stepSmooth = 0;
    this.landDip = 0;
    this.prevGroundY = this.position.y;
    this.prevGroundX = this.position.x;
    this.prevGroundZ = this.position.z;
    // Traversal state must not survive a death, or respawning mid-mantle drags
    // the player straight back to the ledge they were climbing.
    this.mantling = false;
    this.mantleCooldown = 0;
    this.hadGround = false;
    this.bus.emit('ui:notify', { text: 'RESPAWNING', kind: 'info' });
  }

  // ── Main update ────────────────────────────────────────────────────────────

  update(ctx: EngineContext, frame: FrameContext): void {
    const dt = frame.dt;
    if (dt <= 0) {
      this.publish();
      return;
    }
    this.probe.tick(dt);

    if (!this.alive) {
      this.respawnTimer -= dt;
      // Keep looking around while dead so the death cam is not frozen.
      this.updateLook(dt, /*absorbRecoil*/ false);
      this.updateRecoil(dt);
      this.placeCamera(ctx, dt);
      if (this.respawnTimer <= 0) this.respawn();
      this.publish();
      return;
    }

    if (this.spawnFixTimer > 0) {
      this.spawnFixTimer -= dt;
      if (this.spawnFixTimer <= 0) this.resolveSpawn();
    }

    this.updateTimers(dt);
    this.updateLook(dt, true);
    this.updateRecoil(dt);

    if (this.mantling) {
      this.updateMantle(dt);
    } else {
      this.updateStance(dt);
      this.updateMovement(dt);
      this.tryMantle();
    }

    this.updateHealth(dt);
    this.updateCameraEffects(dt);
    this.placeCamera(ctx, dt);
    this.publish();
  }

  /** Re-query the level for a spawn now that every system has registered. */
  private resolveSpawn(): void {
    const previous = _v0.copy(this.spawnPoint);
    getPlayerSpawn(this.ctx, this.probe, this.spawnPoint);
    const untouched = this.position.distanceToSquared(previous) < 0.25;
    if (untouched && this.spawnPoint.distanceToSquared(previous) > 1e-4) {
      this.position.copy(this.spawnPoint);
      this.velocity.set(0, 0, 0);
      this.prevGroundY = this.position.y;
      this.stepSmooth = 0;
    }
  }

  private updateTimers(dt: number): void {
    this.slideCooldown = Math.max(0, this.slideCooldown - dt);
    this.mantleCooldown = Math.max(0, this.mantleCooldown - dt);
    this.timeSinceDamage += dt;
    this.timeSinceSprint += dt;
    this.jumpBuffer = Math.max(0, this.jumpBuffer - dt);
    if (this.grounded) this.coyote = COYOTE_TIME;
    else this.coyote = Math.max(0, this.coyote - dt);
  }

  // ── Look + recoil ──────────────────────────────────────────────────────────

  private updateLook(dt: number, absorbRecoil: boolean): void {
    const input = this.ctx.input;
    if (!input.pointerLocked) return;

    // Input arrives already sensitivity-scaled, in radians.
    let dYaw = -input.lookDelta.x;
    let dPitch = -input.lookDelta.y;

    if (absorbRecoil) {
      dPitch = this.absorb(dPitch, 'pitch');
      dYaw = this.absorb(dYaw, 'yaw');
    }

    // Mouse right (+movementX) must turn the view right, which is a *decrease*
    // in yaw about +Y; `dYaw` already carries that negation.
    this.yaw += dYaw;
    this.pitch = THREE.MathUtils.clamp(this.pitch + dPitch, -PITCH_LIMIT, PITCH_LIMIT);
    void dt;
  }

  /**
   * Spends player input against pending recoil before it moves the base aim.
   * Returns the portion of `delta` that should actually rotate the view.
   */
  private absorb(delta: number, axis: 'pitch' | 'yaw'): number {
    const pending = axis === 'pitch' ? this.recoilPitch : this.recoilYaw;
    if (pending === 0 || delta === 0) return delta;
    // Only input that opposes the pending kick can absorb it.
    if (Math.sign(delta) === Math.sign(pending)) return delta;

    const absorbable = Math.min(Math.abs(pending), Math.abs(delta));
    const remaining = Math.abs(pending) - absorbable;
    const scale = Math.abs(pending) > 1e-9 ? remaining / Math.abs(pending) : 0;

    if (axis === 'pitch') {
      this.recoilPitch = Math.sign(pending) * remaining;
      this.recoilPitchVel *= scale; // the player "caught" that share of the spring
    } else {
      this.recoilYaw = Math.sign(pending) * remaining;
      this.recoilYawVel *= scale;
    }
    return delta - Math.sign(delta) * absorbable;
  }

  private updateRecoil(dt: number): void {
    // Critically damped recovery: x'' = -ω²x - 2ωx'. No overshoot, so the view
    // never dips below the point of aim.
    //
    // Integrated ANALYTICALLY, not with Euler. Semi-implicit Euler on this
    // system is only stable while ω·dt < ~0.83; a snappy kick (duration 0.05 s
    // ⇒ ω = 60) blows up at 60 fps, and even a gentle ω = 16 diverges at the
    // engine's clamped worst-case dt of 0.1. The closed form
    //     x(t) = (x₀ + (v₀ + ω·x₀)·t)·e^(-ω·t)
    // is exact and unconditionally stable for any dt, which matters because a
    // single frame hitch must never permanently corrupt the player's aim.
    const w = this.recoilOmega;
    const e = Math.exp(-w * dt);
    {
      const x0 = this.recoilPitch;
      const b = this.recoilPitchVel + w * x0;
      this.recoilPitch = (x0 + b * dt) * e;
      this.recoilPitchVel = (b - w * (x0 + b * dt)) * e;
    }
    {
      const x0 = this.recoilYaw;
      const b = this.recoilYawVel + w * x0;
      this.recoilYaw = (x0 + b * dt) * e;
      this.recoilYawVel = (b - w * (x0 + b * dt)) * e;
    }
    if (Math.abs(this.recoilPitch) < 1e-5 && Math.abs(this.recoilPitchVel) < 1e-4) {
      this.recoilPitch = 0;
      this.recoilPitchVel = 0;
    }
    if (Math.abs(this.recoilYaw) < 1e-5 && Math.abs(this.recoilYawVel) < 1e-4) {
      this.recoilYaw = 0;
      this.recoilYawVel = 0;
    }

    // Shake honours the requested `duration` exactly, with a slightly convex
    // falloff so the tail is quiet rather than a hard linear cut.
    this.shakeT += dt * this.shakeFreq;
    if (this.shakeLeft > 0) {
      this.shakeLeft = Math.max(0, this.shakeLeft - dt);
      const t = this.shakeLeft / this.shakeDuration;
      this.shakeAmp = this.shakePeak * t * t;
      if (this.shakeAmp < 1e-5) this.shakeAmp = 0;
    } else {
      this.shakeAmp = 0;
    }
  }

  // ── Stance ─────────────────────────────────────────────────────────────────

  private updateStance(dt: number): void {
    const input = this.ctx.input;

    // Tactical sprint: a second sprint press inside the double-tap window.
    if (input.pressed('sprint')) {
      const now = performance.now() / 1000;
      if (now - this.lastSprintPress < DOUBLE_TAP_WINDOW && this.stamina > 25) {
        this.tacSprint = true;
        this.tacSprintTimer = 4.0;
      }
      this.lastSprintPress = now;
    }

    const forwardHeld = input.isDown('forward');
    const wantSprint = input.isDown('sprint') && forwardHeld && !this.ads && this.stamina > 1;
    this.sprinting = wantSprint && !this.sliding;
    if (!this.sprinting) this.tacSprint = false;
    if (this.tacSprint) {
      this.tacSprintTimer -= dt;
      if (this.tacSprintTimer <= 0 || this.stamina < 2) this.tacSprint = false;
    }

    if (this.sprinting) {
      this.stamina = Math.max(0, this.stamina - (this.tacSprint ? STAMINA_TAC_DRAIN : STAMINA_SPRINT_DRAIN) * dt);
      this.timeSinceSprint = 0;
    } else if (this.timeSinceSprint > STAMINA_REGEN_DELAY) {
      this.stamina = Math.min(STAMINA_MAX, this.stamina + STAMINA_REGEN * dt);
    }

    // Slide entry: sprinting + crouch, with real speed behind it.
    const crouchPressed = input.pressed('crouch') || input.pressed('slide');
    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    if (
      crouchPressed &&
      this.sprinting &&
      this.grounded &&
      !this.sliding &&
      this.slideCooldown <= 0 &&
      speed > SLIDE_MIN_ENTRY_SPEED
    ) {
      this.startSlide(speed);
    }

    // Crouch target. Sliding forces full crouch; otherwise it follows the key,
    // and standing back up is blocked by low ceilings.
    let wantCrouch = this.sliding || input.isDown('crouch');
    if (!wantCrouch && this.crouchT > 0 && !this.canStand()) wantCrouch = true;

    const rate = this.sliding ? dt / 0.09 : dt / CROUCH_TIME;
    this.crouchT = THREE.MathUtils.clamp(this.crouchT + (wantCrouch ? rate : -rate), 0, 1);
  }

  /** Is there room to extend back to full height? */
  private canStand(): boolean {
    const h = this.height;
    if (h >= HEIGHT_STAND - 1e-3) return true;
    _v0.set(this.position.x, this.position.y + h - 0.05, this.position.z);
    return !this.probe.raycast(_v0, _up, HEIGHT_STAND - h + 0.08, _hit);
  }

  private startSlide(entrySpeed: number): void {
    this.sliding = true;
    this.slideTimer = 0;
    this.slideAirTime = 0;
    this.slideDrop = SLIDE_CAMERA_DROP;
    _v0.set(this.velocity.x, 0, this.velocity.z);
    if (_v0.lengthSq() < 1e-6) yawForward(this.yaw, _v0);
    this.slideDir.copy(_v0).normalize();
    // A slide is a burst, not just a crouch: boost to at least SLIDE_SPEED.
    const target = Math.max(entrySpeed, SLIDE_SPEED);
    this.velocity.x = this.slideDir.x * target;
    this.velocity.z = this.slideDir.z * target;
    this.sprinting = false;
    this.tacSprint = false;
    this.bus.emit('player:slide:start', {});
  }

  private endSlide(): void {
    if (!this.sliding) return;
    this.sliding = false;
    this.slideCooldown = SLIDE_COOLDOWN;
    this.bus.emit('player:slide:end', {});
  }

  // ── Movement ───────────────────────────────────────────────────────────────

  private updateMovement(dt: number): void {
    const input = this.ctx.input;

    // Wish direction in world space from the camera yaw.
    yawForward(this.yaw, _v1);
    yawRight(this.yaw, _v2);
    const fx = _v1.x;
    const fz = _v1.z;
    const rx = _v2.x;
    const rz = _v2.z;
    let ix = 0;
    let iz = 0;
    if (input.isDown('forward')) iz += 1;
    if (input.isDown('back')) iz -= 1;
    if (input.isDown('right')) ix += 1;
    if (input.isDown('left')) ix -= 1;
    const il = Math.hypot(ix, iz);
    if (il > 1e-5) {
      ix /= il;
      iz /= il;
    }
    const wishX = fx * iz + rx * ix;
    const wishZ = fz * iz + rz * ix;

    // Jump buffering: remember the press even if we are a frame early.
    if (input.pressed('jump')) this.jumpBuffer = JUMP_BUFFER;

    if (this.sliding) {
      this.updateSlide(dt, wishX, wishZ);
    } else {
      const target = this.targetSpeed();
      if (this.grounded) {
        this.applyFriction(dt, wishX, wishZ, target);
        this.accelerate(wishX, wishZ, target, ACCEL_GROUND, dt);
      } else {
        this.accelerate(wishX, wishZ, target, ACCEL_AIR, dt);
      }
    }

    // Jump (also cancels a slide, which is the signature slide-cancel).
    if (this.jumpBuffer > 0 && (this.grounded || this.coyote > 0) && !this.move.hitCeiling) {
      if (this.sliding) this.endSlide();
      // Half-step gravity correction. A full frame of gravity is subtracted
      // below before the first integration, and semi-implicit Euler then
      // overshoots by ½·dt·v. Adding back ½·g·dt makes the DISCRETE apex equal
      // JUMP_APEX exactly, and — more importantly — makes it identical at 30,
      // 60 and 144 fps instead of scaling with frame time.
      this.velocity.y = JUMP_VELOCITY - GRAVITY * dt * 0.5;
      this.grounded = false;
      this.coyote = 0;
      this.jumpBuffer = 0;
      this.crouchT = Math.min(this.crouchT, 0.25); // pop back up out of a slide
    }

    this.velocity.y += GRAVITY * dt;
    if (this.velocity.y < -70) this.velocity.y = -70;

    this.prevVelocityY = this.velocity.y;

    moveCapsule(
      this.probe,
      this.position,
      this.velocity,
      RADIUS,
      this.height,
      this.sliding ? 0.2 : STEP_HEIGHT,
      dt,
      SLOPE_LIMIT_COS,
      this.move,
    );

    const wasGrounded = this.grounded;
    this.grounded = this.move.grounded;
    this.groundSurface = this.move.groundSurface;

    // Slopes steeper than the limit: keep accelerating downhill so the player
    // visibly slides off rather than standing on a 60° roof.
    if (this.move.onSteepSlope && !this.grounded) {
      const n = this.move.groundNormal;
      this.velocity.x += n.x * 14 * dt;
      this.velocity.z += n.z * 14 * dt;
    }

    // Stair smoothing: the collider snaps, the camera does not.
    //
    // Only *discontinuous* height changes get absorbed. Smoothing every
    // grounded Δy would also swallow the perfectly continuous rise and fall of
    // walking a ramp, which leaves the camera floating a fixed distance above
    // the head for the entire descent. So we predict the Δy implied by sliding
    // along the current ground plane and smooth only what exceeds it — which
    // is exactly a stair riser, a step-up, or a snap onto a ledge.
    if (this.grounded && this.hadGround && !this.mantling) {
      const dx = this.position.x - this.prevGroundX;
      const dz = this.position.z - this.prevGroundZ;
      const n = this.move.groundNormal;
      const slopeDy = n.y > 0.1 ? -(n.x * dx + n.z * dz) / n.y : 0;
      const discontinuity = (this.position.y - this.prevGroundY) - slopeDy;
      if ((this.move.steppedUp > 0 || Math.abs(discontinuity) > 0.012) && Math.abs(discontinuity) < 0.5) {
        this.stepSmooth = THREE.MathUtils.clamp(this.stepSmooth - discontinuity, -0.45, 0.45);
      }
    }
    this.prevGroundY = this.position.y;
    this.prevGroundX = this.position.x;
    this.prevGroundZ = this.position.z;
    this.hadGround = this.grounded;

    // Out-of-world recovery. Without this a player who tunnels or walks off the
    // map falls forever with no way back — there is no other path to respawn.
    if (this.position.y < WORLD_FLOOR_Y) {
      // Routed through `damage` rather than calling `die` directly: that is the
      // only path that raises `damage:taken`, and the scoreboard infers player
      // deaths from it. Killing silently here loses the death and the streak
      // reset forever.
      _v0.copy(this.eye).add(_up);
      this.damage(this.health + 1, _v0);
      return;
    }

    // Landing.
    if (this.grounded && !wasGrounded) this.onLand(-this.prevVelocityY);

    // Footsteps, distance based so cadence tracks actual travel.
    if (this.grounded && !this.sliding) {
      const sp = Math.hypot(this.velocity.x, this.velocity.z);
      this.footDistance += sp * dt;
      const stride = this.sprinting ? 2.5 : this.crouchT > 0.5 ? 1.75 : 2.05;
      if (this.footDistance >= stride && sp > 0.6) {
        this.footDistance = 0;
        this.bus.emit('player:footstep', {
          position: new THREE.Vector3(this.position.x, this.position.y, this.position.z),
          surface: this.groundSurface,
          sprinting: this.sprinting,
        });
      }
    }
  }

  private targetSpeed(): number {
    if (this.crouchT > 0.5) return SPEED_CROUCH;
    if (this.ads) return SPEED_ADS;
    if (this.sprinting) return this.tacSprint ? SPEED_TAC_SPRINT : SPEED_SPRINT;
    return SPEED_WALK;
  }

  /**
   * Directional acceleration: only the component of the wish
   * direction we are not already moving in gets added. Strafing therefore never
   * costs you speed, and stopping is instant when you release the key.
   */
  private accelerate(wishX: number, wishZ: number, wishSpeed: number, accel: number, dt: number): void {
    if (wishSpeed <= 0 || (wishX === 0 && wishZ === 0)) return;
    const current = this.velocity.x * wishX + this.velocity.z * wishZ;
    const add = wishSpeed - current;
    if (add <= 0) return;
    const step = Math.min(accel * dt, add);
    this.velocity.x += wishX * step;
    this.velocity.z += wishZ * step;
  }

  /**
   * Ground friction, with the driven part of the motion protected.
   *
   * Naive friction plus a linear (m/s²) acceleration cap has a nasty
   * emergent property: the two reach equilibrium at ACCEL/FRICTION, which caps
   * the real top speed regardless of what you asked for. With 55 m/s² and a
   * friction of 9 that ceiling is 6.11 m/s, so a 7.4 m/s sprint would silently
   * top out 18% slow while every constant in the file still *reads* correct.
   *
   * The fix is to exempt the speed the player is actively driving, up to their
   * wish speed. Anything above it (slide momentum, a launch pad) still bleeds
   * off, and releasing the stick still stops you dead in ~0.11 s.
   */
  private applyFriction(dt: number, wishX: number, wishZ: number, wishSpeed: number): void {
    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    if (speed < 1e-4) {
      this.velocity.x = 0;
      this.velocity.z = 0;
      return;
    }
    let floor = 0;
    if (wishSpeed > 0 && (wishX !== 0 || wishZ !== 0)) {
      const along = this.velocity.x * wishX + this.velocity.z * wishZ;
      floor = Math.max(0, Math.min(along, wishSpeed));
    }
    const control = Math.max(speed, STOP_SPEED);
    const drop = control * FRICTION * dt;
    const newSpeed = Math.max(floor, speed - drop);
    const scale = newSpeed / speed;
    this.velocity.x *= scale;
    this.velocity.z *= scale;
  }

  private updateSlide(dt: number, wishX: number, wishZ: number): void {
    this.slideTimer += dt;
    if (!this.grounded) this.slideAirTime += dt;
    else this.slideAirTime = 0;

    // Exponential speed bleed, plus a gravity assist so downhill slides carry.
    const decay = Math.exp(-SLIDE_DECAY * dt);
    this.velocity.x *= decay;
    this.velocity.z *= decay;
    if (this.grounded) {
      const n = this.move.groundNormal;
      // Downhill component of the surface: positive when the slide faces downhill.
      const downhill = -(n.x * this.slideDir.x + n.z * this.slideDir.z);
      if (downhill > 0) {
        this.velocity.x += this.slideDir.x * downhill * 26 * dt;
        this.velocity.z += this.slideDir.z * downhill * 26 * dt;
      }
    }

    // Limited steering — you can curve a slide, not turn it into a walk.
    if (wishX !== 0 || wishZ !== 0) {
      const steer = 2.6 * dt;
      this.velocity.x += wishX * steer;
      this.velocity.z += wishZ * steer;
    }

    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    if (speed > 1e-4) {
      this.slideDir.set(this.velocity.x / speed, 0, this.velocity.z / speed);
    }
    if (speed < SPEED_CROUCH + 0.2 || this.slideTimer > SLIDE_TIME * 1.25 || this.slideAirTime > 0.25) {
      this.endSlide();
    }
  }

  private onLand(impactSpeed: number): void {
    if (impactSpeed < 1.2) return;
    this.bus.emit('player:land', { velocity: impactSpeed });
    // Dip proportional to impact, capped so a long fall does not bury the camera.
    this.landDip = Math.min(0.16, impactSpeed / 14 * 0.16);
    this.landDipVel = 0;
    if (impactSpeed > FALL_DAMAGE_SPEED) {
      _v0.copy(this.eye).add(_up);
      this.damage((impactSpeed - FALL_DAMAGE_SPEED) * FALL_DAMAGE_PER_MS, _v0);
    }
  }

  // ── Mantle / vault ─────────────────────────────────────────────────────────

  /**
   * Ledge detection: a forward cast finds the face, a downward cast from beyond
   * and above it finds the top, and an upward cast confirms we would fit. Two
   * casts up, one down — cheap enough to run every frame we are pressed against
   * something.
   */
  private tryMantle(): void {
    if (this.sliding || this.mantleCooldown > 0) return;
    if (!this.ctx.input.isDown('forward')) return;
    // Only when actually pressed against geometry, or airborne next to it.
    if (!this.move.hitWall && this.grounded) return;
    // Not while plummeting: a long fall past a balcony should not snap you onto
    // it, and it would swallow fall damage.
    if (this.velocity.y < -9) return;

    yawForward(this.yaw, _v0);
    const fx = _v0.x;
    const fz = _v0.z;

    // Face cast: probe at knee AND chest height, because a ledge with an open
    // space beneath it (a railing, a table, a window sill) has no knee-height
    // face to hit at all.
    let faceDist = -1;
    for (let i = 0; i < 2; i++) {
      _v1.set(this.position.x, this.position.y + (i === 0 ? 0.3 : 1.0), this.position.z);
      if (!this.probe.raycast(_v1, _v0, RADIUS + 0.45, _hit)) continue;
      if (Math.abs(_hit.normal.y) > 0.6) continue; // a ramp, not a ledge
      faceDist = _hit.distance;
      break;
    }
    if (faceDist < 0) return;

    // Ledge cast: straight down from above and slightly past the face.
    const overshoot = faceDist + 0.42;
    _v1.set(
      this.position.x + fx * overshoot,
      this.position.y + MANTLE_MAX_HEIGHT + 0.5,
      this.position.z + fz * overshoot,
    );
    if (!this.probe.raycast(_v1, _down, MANTLE_MAX_HEIGHT + 0.5 - MANTLE_MIN_HEIGHT, _hit)) return;
    if (_hit.normal.y < 0.7) return; // top is not standable

    const ledgeY = _hit.point.y;
    const rise = ledgeY - this.position.y;
    if (rise < MANTLE_MIN_HEIGHT || rise > MANTLE_MAX_HEIGHT) return;

    // Headroom above the landing spot.
    _v2.copy(_hit.point);
    _v1.set(_v2.x, _v2.y + 0.1, _v2.z);
    if (this.probe.raycast(_v1, _up, HEIGHT_STAND - 0.15, _hit)) return;

    this.mantling = true;
    this.mantleT = 0;
    // Taller ledges take longer — a 1.4 m wall is a haul, a 0.5 m crate is a hop.
    this.mantleDuration = 0.34 + rise * 0.22;
    this.mantleStart.copy(this.position);
    this.mantleEnd.set(_v2.x + fx * (RADIUS * 0.9), _v2.y, _v2.z + fz * (RADIUS * 0.9));
    this.velocity.set(0, 0, 0);
    this.endSlide();
  }

  /**
   * Parametric arc: rise first, translate second. Doing both linearly would
   * drag the player's chest through the ledge corner.
   */
  private updateMantle(dt: number): void {
    this.mantleT += dt / this.mantleDuration;
    const t = Math.min(1, this.mantleT);

    // Vertical completes by 65% of the move, horizontal starts at 25%.
    const tv = Math.min(1, t / 0.65);
    const th = THREE.MathUtils.clamp((t - 0.25) / 0.75, 0, 1);
    const ev = 1 - (1 - tv) * (1 - tv) * (1 - tv); // easeOutCubic
    const eh = smoothstep01(th);

    this.position.x = THREE.MathUtils.lerp(this.mantleStart.x, this.mantleEnd.x, eh);
    this.position.z = THREE.MathUtils.lerp(this.mantleStart.z, this.mantleEnd.z, eh);
    this.position.y = THREE.MathUtils.lerp(this.mantleStart.y, this.mantleEnd.y, ev);

    // A little camera lean sells the effort.
    this.strafeRoll = THREE.MathUtils.damp(this.strafeRoll, Math.sin(t * Math.PI) * 0.045, 12, dt);

    if (t >= 1) {
      this.mantling = false;
      this.mantleCooldown = 0.25;
      this.grounded = true;
      this.coyote = COYOTE_TIME;
      this.prevGroundY = this.position.y;
      this.hadGround = true;
      this.velocity.set(0, 0, 0);
    }
  }

  // ── Health ─────────────────────────────────────────────────────────────────

  private updateHealth(dt: number): void {
    if (this.health < MAX_HEALTH && this.timeSinceDamage > REGEN_DELAY) {
      // Ramp in over the first half-second so regen does not click on abruptly.
      const ramp = Math.min(1, (this.timeSinceDamage - REGEN_DELAY) / 0.5);
      this.health = Math.min(MAX_HEALTH, this.health + REGEN_RATE * ramp * dt);
    }
    const target = this.health >= LOW_HEALTH_THRESHOLD
      ? 0
      : 1 - this.health / LOW_HEALTH_THRESHOLD;
    this.lowHealthFactor = THREE.MathUtils.damp(this.lowHealthFactor, target, 6, dt);
  }

  // ── Camera ─────────────────────────────────────────────────────────────────

  private updateCameraEffects(dt: number): void {
    const speed = Math.hypot(this.velocity.x, this.velocity.z);

    // Head bob: vertical at twice the lateral frequency, which is what a real
    // gait does. Amplitude scales with speed and is nearly suppressed in ADS —
    // a bobbing sight picture is the fastest way to make players motion-sick.
    if (this.grounded && !this.sliding && speed > 0.3) {
      this.bobPhase += (speed / 1.35) * Math.PI * 2 * dt;
      if (this.bobPhase > Math.PI * 2) this.bobPhase -= Math.PI * 2;
    }
    const bobTarget = this.grounded && !this.sliding
      ? Math.min(1, speed / SPEED_WALK) * (this.ads ? 0.25 : 1)
      : 0;
    this.bobAmount = THREE.MathUtils.damp(this.bobAmount, bobTarget, 8, dt);

    // Strafe roll: a small counter-lean when moving sideways. 1.2° is deliberate
    // — anything larger reads as drunk rather than athletic.
    const input = this.ctx.input;
    let lateral = 0;
    if (input.isDown('right')) lateral += 1;
    if (input.isDown('left')) lateral -= 1;
    let rollTarget = -lateral * STRAFE_ROLL * (this.ads ? 0.4 : 1);
    if (this.sliding) {
      // Slides always roll, biased by which way you are curving out of it.
      yawRight(this.yaw, _v1);
      const side = this.slideDir.x * _v1.x + this.slideDir.z * _v1.z;
      rollTarget = -SLIDE_ROLL * (0.55 + THREE.MathUtils.clamp(side * 2, -0.45, 0.45));
    }
    if (!this.mantling) this.strafeRoll = THREE.MathUtils.damp(this.strafeRoll, rollTarget, 9, dt);

    // Landing dip: a deliberately under-damped spring so the camera settles
    // with a small bounce. Sub-stepped at a fixed 1/120 s because explicit
    // integration of a stiff spring (ω ≈ 16) is unstable at large dt, and the
    // engine hands us up to 0.1 s after a hitch.
    let dipLeft = dt;
    while (dipLeft > 1e-6) {
      const h = Math.min(dipLeft, 1 / 120);
      this.landDipVel += (-this.landDip * 260 - this.landDipVel * 24) * h;
      this.landDip += this.landDipVel * h;
      dipLeft -= h;
    }

    this.slideDrop = THREE.MathUtils.damp(this.slideDrop, this.sliding ? SLIDE_CAMERA_DROP : 0, 10, dt);
    // Stair offset decays fast enough to stay imperceptible but slow enough to
    // remove the strobe.
    this.stepSmooth = THREE.MathUtils.damp(this.stepSmooth, 0, 14, dt);
  }

  private placeCamera(ctx: EngineContext, dt: number): void {
    const eyeLocal = THREE.MathUtils.lerp(EYE_STAND, EYE_CROUCH, smoothstep01(this.crouchT));

    const bobV = Math.sin(this.bobPhase * 2) * BOB_AMPLITUDE * this.bobAmount;
    const bobH = Math.sin(this.bobPhase) * BOB_AMPLITUDE * 1.15 * this.bobAmount;

    yawRight(this.yaw, _v1);
    const rx = _v1.x;
    const rz = _v1.z;

    this.eye.set(
      this.position.x + rx * bobH,
      this.position.y + eyeLocal + this.stepSmooth + bobV - this.landDip - this.slideDrop,
      this.position.z + rz * bobH,
    );
    ctx.camera.position.copy(this.eye);

    // Shake is transient and additive on all three axes; it must never be
    // absorbable by mouse input or aiming during an explosion would drift.
    let sp = 0;
    let sy = 0;
    let sr = 0;
    if (this.shakeAmp > 0) {
      sp = shakeNoise(this.shakeT) * this.shakeAmp;
      sy = shakeNoise(this.shakeT + 37.1) * this.shakeAmp;
      sr = shakeNoise(this.shakeT + 71.7) * this.shakeAmp * 0.7;
    }

    // The COMPOSED pitch is clamped, not just the base. With YXZ order an
    // X-rotation past vertical flips the horizon and mirrors yaw, so kick plus
    // shake must never be able to push the view over the top.
    ctx.camera.rotation.set(
      THREE.MathUtils.clamp(this.pitch + this.recoilPitch + sp, -PITCH_LIMIT, PITCH_LIMIT),
      this.yaw + this.recoilYaw + sy,
      this.strafeRoll + bobH * 0.35 + sr,
      'YXZ',
    );

    // Flush the camera's world matrix now rather than leaving it to the
    // renderer. Systems that run later in the same update phase — the weapon
    // system's `Raycaster.setFromCamera`, any frustum cull — read `matrixWorld`
    // directly, and would otherwise aim a whole frame behind the view during a
    // fast flick.
    ctx.camera.updateMatrixWorld(true);
    void dt;
  }

  // ── State broadcast ────────────────────────────────────────────────────────

  /**
   * A FRESH snapshot every frame, deliberately.
   *
   * Reusing one object would save three small allocations per frame, but every
   * consumer that legitimately holds last frame's state to diff against (HUD
   * damage flashes, speed-based FOV, footstep triggers) would silently compare
   * an object with itself and never see a change. That is a bug class that
   * costs hours to find; three short-lived objects a frame costs nothing next
   * to what the renderer already churns.
   */
  private publish(): void {
    this.bus.emit('player:state', {
      position: this.position.clone(),
      velocity: this.velocity.clone(),
      grounded: this.grounded,
      sprinting: this.sprinting,
      crouching: this.crouchT > 0.5,
      sliding: this.sliding,
      ads: this.ads,
      health: this.health,
      maxHealth: MAX_HEALTH,
      stamina: this.stamina,
    });
  }

  // ── Public accessors for other systems ───────────────────────────────────────

  get isSprinting(): boolean {
    return this.sprinting;
  }

  get isTacSprinting(): boolean {
    return this.tacSprint;
  }

  get isSliding(): boolean {
    return this.sliding;
  }

  get isMantling(): boolean {
    return this.mantling;
  }

  get isGrounded(): boolean {
    return this.grounded;
  }

  get lookYaw(): number {
    return this.yaw;
  }

  get lookPitch(): number {
    return this.pitch;
  }

  /** Exposed so the enemy system can share one probe rather than building a second. */
  get worldProbe(): WorldProbe {
    return this.probe;
  }

  /** Teleport (used by level scripting / debug). */
  setPosition(p: THREE.Vector3): void {
    this.position.copy(p);
    this.velocity.set(0, 0, 0);
    this.prevGroundY = p.y;
    this.prevGroundX = p.x;
    this.prevGroundZ = p.z;
    this.stepSmooth = 0;
    // Any in-flight traversal state must be cleared, or a teleport during a
    // mantle drags the player straight back to the ledge it was climbing.
    this.mantling = false;
    this.mantleCooldown = 0;
    this.hadGround = false;
    this.endSlide();
  }
}
