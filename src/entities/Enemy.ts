/**
 * NOISEFIELD — Enemy body.
 *
 * The "body" half of a soldier: locomotion, orientation, health, damage
 * response and the hand-off to the ragdoll. All decision making lives in
 * AI.ts; this class only ever executes commands from an `AIOutput`, which keeps
 * the two testable in isolation and lets the manager throttle thinking without
 * risking a half-applied transform.
 */

import * as THREE from 'three';
import type { EntityId, IEventBus, IMaterialLibrary } from '../core/Types';
import {
  AIController,
  AIOutput,
  CapsuleMoveResult,
  ProbeHit,
  moveCapsule,
  type AIBody,
  type AIShotResult,
  type AITarget,
  type NavGrid,
  type WorldProbe,
} from './AI';
import {
  CHAR_EYE_HEIGHT,
  CHAR_HEIGHT,
  CharacterModel,
  ProceduralAnimator,
  type AnimatorInput,
  type GroundSampler,
} from './CharacterModel';
import { Ragdoll } from './Ragdoll';

// ── Body tuning ──────────────────────────────────────────────────────────────

const ENEMY_RADIUS = 0.34;
/** Same doubled gravity as the player. See Player.ts for the rationale. */
const ENEMY_GRAVITY = -19.6;
const ENEMY_ACCEL = 17;
const ENEMY_FRICTION = 11;
const ENEMY_STEP = 0.42;
/** cos(52°) — enemies give up on slopes marginally sooner than the player. */
const ENEMY_SLOPE_COS = 0.616;
const TURN_RATE_AIM = 9.5;
const TURN_RATE_MOVE = 6.5;
const ENEMY_MAX_HEALTH = 100;
/** Beyond ~110° of backpedal the walk cycle stops reading; break off and turn. */
const BACKPEDAL_LIMIT = 110 * (Math.PI / 180);

export type EnemyPhase = 'inactive' | 'alive' | 'corpse';

export interface EnemyDeps {
  probe: WorldProbe;
  nav: NavGrid;
  bus: IEventBus;
  scene: THREE.Scene;
  materials?: IMaterialLibrary;
}

const _e0 = new THREE.Vector3();
const _e1 = new THREE.Vector3();
const _e2 = new THREE.Vector3();
const _eHit = new ProbeHit();

/** Shortest signed angular difference, wrapped to (-π, π]. */
function angleDelta(from: number, to: number): number {
  let d = (to - from) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

export class Enemy implements AIBody {
  id: EntityId = -1;
  phase: EnemyPhase = 'inactive';

  readonly position = new THREE.Vector3();
  readonly velocity = new THREE.Vector3();
  readonly eye = new THREE.Vector3();
  readonly muzzle = new THREE.Vector3();
  grounded = false;
  health = ENEMY_MAX_HEALTH;
  readonly maxHealth = ENEMY_MAX_HEALTH;
  yaw = 0;

  readonly model: CharacterModel;
  readonly ai: AIController;
  readonly ragdoll = new Ragdoll();

  private readonly animator: ProceduralAnimator;
  private readonly out = new AIOutput();
  /** Reused animator payload; rebuilding this literal every frame for every
   *  soldier is pure garbage in the hottest loop in the system. */
  private readonly animInput: AnimatorInput = {
    speed: 0, grounded: false, crouch: 0, aimAt: null, aimWeight: 0, firing: false,
  };
  private readonly move = new CapsuleMoveResult();
  private readonly deps: EnemyDeps;
  private crouch = 0;
  private aiAccum = 0;
  private animAccum = 0;
  private lastKiller: 'player' | 'world' = 'world';
  private diedFromHeadshot = false;
  /** Cached distance to the camera, written by the manager for LOD decisions. */
  viewDistance = 0;
  /**
   * Raised on the frame this soldier dies, cleared by the manager once it has
   * emitted `entity:killed`. Routing every death through one flag means a
   * soldier lost to the world (falling out of the map) raises exactly the same
   * event as one shot by the player — otherwise anything counting kills from
   * events drifts permanently out of sync with reality.
   */
  killEventPending = false;

  constructor(deps: EnemyDeps, seed: number) {
    this.deps = deps;
    this.model = new CharacterModel({ materials: deps.materials, castShadow: true });
    this.model.setVisible(false);
    deps.scene.add(this.model.root);

    this.animator = new ProceduralAnimator(this.model);
    // Foot planting needs a ground query; inject the probe rather than letting
    // the animator reach for a system it should not know about.
    const sampler: GroundSampler = (x, z, fromY, o) => {
      if (!deps.probe.ground(x, z, fromY, 1.6, _eHit)) return false;
      o.copy(_eHit.point);
      return true;
    };
    this.animator.groundSampler = sampler;

    this.ai = new AIController(seed, {
      probe: deps.probe,
      nav: deps.nav,
      onShot: (from, shot) => this.onShotFired(from, shot),
    });
  }

  get alive(): boolean {
    return this.phase === 'alive';
  }

  /** Shots emitted this tick. The manager uses it to propagate squad awareness. */
  get firedShots(): number {
    return this.out.shots;
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  spawn(id: EntityId, position: THREE.Vector3, yaw: number): void {
    this.id = id;
    this.phase = 'alive';
    this.health = ENEMY_MAX_HEALTH;
    this.position.copy(position);
    this.velocity.set(0, 0, 0);
    this.yaw = yaw;
    this.crouch = 0;
    this.grounded = false;
    this.aiAccum = 0;
    this.animAccum = 0;
    this.diedFromHeadshot = false;
    this.killEventPending = false;

    this.model.reset();
    this.model.setEntityId(id);
    this.model.root.position.copy(position);
    this.model.root.rotation.set(0, yaw, 0);
    this.model.setVisible(true);
    this.animator.reset();
    this.animator.footIK = true;
    this.ragdoll.reset();
    this.ai.reset(position);

    this.eye.set(position.x, position.y + CHAR_EYE_HEIGHT, position.z);
    this.muzzle.copy(this.eye);
    this.model.root.updateMatrixWorld(true);
  }

  despawn(): void {
    this.phase = 'inactive';
    this.model.setVisible(false);
    this.model.setHitboxesEnabled(false);
    this.ragdoll.reset();
    this.id = -1;
  }

  dispose(): void {
    this.deps.scene.remove(this.model.root);
    this.model.dispose();
  }

  // ── per-frame ──────────────────────────────────────────────────────────────

  /**
   * @param lod 0 = full logic + animation, 1 = reduced tick rate, 2 = minimal.
   * @returns false when a corpse has expired and the instance may be recycled.
   */
  update(dt: number, target: AITarget, lod: number): boolean {
    if (this.phase === 'inactive') return false;

    if (this.phase === 'corpse') {
      if (!this.ragdoll.update(dt)) {
        this.despawn();
        return false;
      }
      // Track the pelvis so the manager's distance culling stays meaningful.
      this.model.bones.pelvis.getWorldPosition(this.position);
      return true;
    }

    this.ai.lod = lod;

    // Sample the eye analytically rather than from the head bone: perception
    // runs before animation, and a one-frame-stale head would make the vision
    // cone lag the body during hard turns.
    const stance = 1 - this.crouch * 0.32;
    this.eye.set(this.position.x, this.position.y + CHAR_EYE_HEIGHT * stance, this.position.z);

    // Thinking rate falls off with LOD: distant soldiers still walk and turn
    // every frame (cheap) but only re-decide a few times a second.
    const aiPeriod = lod === 0 ? 0 : lod === 1 ? 1 / 12 : 1 / 4;
    this.aiAccum += dt;
    if (this.aiAccum >= aiPeriod) {
      this.ai.update(this.aiAccum, this, target, this.out);
      this.aiAccum = 0;
    } else {
      this.out.shots = 0;
    }

    this.crouch = THREE.MathUtils.damp(this.crouch, this.out.crouch, 7, dt);
    this.locomote(dt);
    // `locomote` can kill us (fell out of the world); do not then animate a
    // corpse whose ragdoll was just seeded.
    if (this.phase !== 'alive') return true;
    this.orient(dt);

    // Animation: full rate up close, 20 Hz at mid range, 8 Hz far away.
    const animPeriod = lod === 0 ? 0 : lod === 1 ? 1 / 20 : 1 / 8;
    this.animAccum += dt;
    if (this.animAccum >= animPeriod) {
      this.animate(this.animAccum);
      this.animAccum = 0;
    }

    // Matrices must be current before anyone raycasts our hitboxes this frame.
    this.model.root.updateMatrixWorld(true);
    this.model.rifleMuzzle.getWorldPosition(this.muzzle);
    return true;
  }

  private locomote(dt: number): void {
    const wishX = this.out.hasMoveTarget ? this.out.moveTarget.x - this.position.x : 0;
    const wishZ = this.out.hasMoveTarget ? this.out.moveTarget.z - this.position.z : 0;
    const wishLen = Math.hypot(wishX, wishZ);
    const speed = this.out.desiredSpeed;

    if (wishLen > 0.05 && speed > 0.05) {
      const tx = (wishX / wishLen) * speed;
      const tz = (wishZ / wishLen) * speed;
      const dvx = tx - this.velocity.x;
      const dvz = tz - this.velocity.z;
      const dvLen = Math.hypot(dvx, dvz);
      const maxDv = ENEMY_ACCEL * dt;
      if (dvLen > maxDv && dvLen > 1e-5) {
        this.velocity.x += (dvx / dvLen) * maxDv;
        this.velocity.z += (dvz / dvLen) * maxDv;
      } else {
        this.velocity.x = tx;
        this.velocity.z = tz;
      }
    } else {
      const drop = Math.max(0, 1 - ENEMY_FRICTION * dt);
      this.velocity.x *= drop;
      this.velocity.z *= drop;
    }

    this.velocity.y += ENEMY_GRAVITY * dt;
    if (this.velocity.y < -55) this.velocity.y = -55;

    moveCapsule(
      this.deps.probe,
      this.position,
      this.velocity,
      ENEMY_RADIUS,
      CHAR_HEIGHT,
      ENEMY_STEP,
      dt,
      ENEMY_SLOPE_COS,
      this.move,
    );
    this.grounded = this.move.grounded;

    // Safety net: a soldier that falls out of the level would otherwise stream
    // raycasts forever. Park it and let the manager recycle.
    if (this.position.y < -60) {
      this.health = 0;
      this.kill(this.position, this.position, false, false);
    }
  }

  private orient(dt: number): void {
    const vx = this.velocity.x;
    const vz = this.velocity.z;
    const moving = Math.hypot(vx, vz) > 0.25;

    let desired = this.yaw;
    let rate = TURN_RATE_MOVE;

    if (this.out.hasLookTarget && this.out.aimWeight > 0.3) {
      desired = Math.atan2(this.out.lookTarget.x - this.position.x, this.out.lookTarget.z - this.position.z);
      rate = TURN_RATE_AIM;
      if (moving) {
        // Never let them moonwalk: past the backpedal limit, rotate toward the
        // direction of travel instead of staying square to the player.
        const moveYaw = Math.atan2(vx, vz);
        const diff = angleDelta(desired, moveYaw);
        if (Math.abs(diff) > BACKPEDAL_LIMIT) {
          desired = moveYaw;
        }
      }
    } else if (moving) {
      desired = Math.atan2(vx, vz);
    }

    const d = angleDelta(this.yaw, desired);
    this.yaw += d * Math.min(1, rate * dt);
    this.model.root.position.copy(this.position);
    this.model.root.rotation.set(0, this.yaw, 0);
  }

  private animate(dt: number): void {
    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    // Scale the cadence by how much of the motion is actually forward, so a
    // sidestep does not drive a full-speed running cycle.
    let forwardness = 1;
    if (speed > 0.2) {
      const fx = Math.sin(this.yaw);
      const fz = Math.cos(this.yaw);
      forwardness = 0.45 + 0.55 * Math.abs((this.velocity.x * fx + this.velocity.z * fz) / speed);
    }

    this.animator.footIK = this.ai.lod === 0;
    const a = this.animInput;
    a.speed = speed * forwardness;
    a.grounded = this.grounded;
    a.crouch = this.crouch;
    a.aimAt = this.out.hasLookTarget ? this.out.lookTarget : null;
    a.aimWeight = this.out.aimWeight;
    a.firing = this.out.shots > 0;
    this.animator.update(dt, a);
  }

  // ── damage ─────────────────────────────────────────────────────────────────

  /**
   * Applies damage. The headshot multiplier is applied by the caller (the
   * weapon system owns the damage model); `headshot` here only drives feedback
   * and the death impulse.
   *
   * @returns true if this hit killed the soldier.
   */
  applyDamage(
    amount: number,
    headshot: boolean,
    point: THREE.Vector3,
    fromWorld: THREE.Vector3,
    byPlayer: boolean,
  ): boolean {
    if (this.phase !== 'alive') return false;

    this.health -= amount;
    this.ai.onDamaged(fromWorld);

    // Flinch: convert the impact direction into model space so the stagger
    // leans the right way.
    _e0.copy(point).sub(fromWorld);
    _e0.y = 0;
    if (_e0.lengthSq() > 1e-6) {
      _e0.normalize();
      const fx = Math.sin(this.yaw);
      const fz = Math.cos(this.yaw);
      const forward = _e0.x * fx + _e0.z * fz;
      const right = _e0.x * fz - _e0.z * fx;
      this.animator.hit(right, forward, THREE.MathUtils.clamp(amount / 45, 0.15, 0.9));
    }

    if (this.health <= 0) {
      this.kill(point, fromWorld, headshot, byPlayer);
      return true;
    }
    return false;
  }

  private kill(point: THREE.Vector3, fromWorld: THREE.Vector3, headshot: boolean, byPlayer: boolean): void {
    this.phase = 'corpse';
    this.health = 0;
    this.diedFromHeadshot = headshot;
    this.lastKiller = byPlayer ? 'player' : 'world';
    this.killEventPending = true;
    this.ai.kill();
    this.model.setHitboxesEnabled(false);

    // Death impulse as momentum (kg·m/s). A rifle round carries far less than
    // this in reality; games exaggerate because a physically-accurate ragdoll
    // barely twitches and reads as a bug.
    _e1.copy(point).sub(fromWorld);
    if (_e1.lengthSq() < 1e-8) _e1.set(0, 0, 1);
    _e1.normalize();
    _e1.y += 0.18; // a touch of lift so bodies fold rather than pancake
    _e1.normalize().multiplyScalar(headshot ? 145 : 85);

    this.ragdoll.activate(this.model, this.deps.probe, this.velocity, _e1, point);
  }

  get killedByPlayer(): boolean {
    return this.lastKiller === 'player';
  }

  get killedByHeadshot(): boolean {
    return this.diedFromHeadshot;
  }

  // ── feedback ───────────────────────────────────────────────────────────────

  /**
   * Emits world feedback for an AI shot. Called by the AI's `onShot` hook.
   *
   * NOTE for the audio/FX systems: enemy fire deliberately does NOT raise
   * `weapon:fire`. That event carries a `WeaponId` from the player's own
   * loadout and has no shooter field, so raising it here would drive the
   * player's ammo HUD and first-person weapon audio from an enemy's trigger
   * pull. `tracer` (whose `from` is the enemy muzzle) is the enemy-fire
   * signal; position gunshot audio from that.
   */
  private onShotFired(from: THREE.Vector3, shot: AIShotResult): void {
    const bus = this.deps.bus;
    // Fresh vectors: the bus is synchronous but handlers may retain payloads,
    // and AI fire rates are low enough that this is not a hot path.
    bus.emit('tracer', { from: from.clone(), to: shot.end.clone(), speed: 720 });

    // A round that reached max range without touching anything has no impact
    // point — emitting one would spawn sparks and a decal in mid-air.
    if (shot.outcome === 'air') return;

    bus.emit('impact', {
      position: shot.end.clone(),
      normal: shot.normal.clone(),
      incoming: shot.direction.clone(),
      surface: shot.surface,
      energy: shot.outcome === 'player' ? 0.7 : 0.55,
    });
  }

  /** Squad awareness: another soldier (or the player) fired nearby. */
  hearGunshot(position: THREE.Vector3, loudness = 1): void {
    if (this.phase !== 'alive') return;
    this.ai.onGunshotHeard(position, this.position, loudness);
  }

  /** A round cracked past — suppress without granting positional knowledge. */
  hearNearMiss(position: THREE.Vector3): void {
    if (this.phase !== 'alive') return;
    this.ai.onNearMiss(position, this.position);
  }

  /** Cheap pairwise separation so soldiers do not stand inside each other. */
  separateFrom(other: Enemy, dt: number): void {
    if (this.phase !== 'alive' || other.phase !== 'alive') return;
    const dx = this.position.x - other.position.x;
    const dz = this.position.z - other.position.z;
    const d2 = dx * dx + dz * dz;
    const minD = ENEMY_RADIUS * 2 + 0.12;
    if (d2 > minD * minD || d2 < 1e-8) return;
    const d = Math.sqrt(d2);
    const push = (minD - d) * 0.5 * Math.min(1, dt * 12);
    const nx = dx / d;
    const nz = dz / d;
    this.position.x += nx * push;
    this.position.z += nz * push;
    other.position.x -= nx * push;
    other.position.z -= nz * push;
  }
}

export { ENEMY_RADIUS, ENEMY_MAX_HEALTH };
