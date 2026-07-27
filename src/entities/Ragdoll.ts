/**
 * NOISEFIELD — Verlet ragdoll.
 *
 * A physically-reacting death beats a canned animation here: the killing shot's
 * direction, the surface underneath, and the momentum the soldier already had
 * all show up in the body, so no two deaths look alike and the player reads the
 * hit direction from the corpse.
 *
 * ── Why verlet ───────────────────────────────────────────────────────────────
 * Position-based (Verlet + distance constraints) rather than a rigid-body
 * solver: it is unconditionally stable at large timesteps, needs no inertia
 * tensors, is trivially cheap for ~16 particles, and — importantly — it does
 * not depend on the physics system existing. Joint limits are expressed as
 * one-sided distance constraints between non-adjacent joints, which is enough
 * to stop elbows and knees hyper-extending or the spine folding in half.
 *
 * All state lives in preallocated typed arrays; simulating a body allocates
 * nothing.
 */

import * as THREE from 'three';
import { ProbeHit, type WorldProbe } from './AI';
import { RIG_METRICS, type BoneName, type CharacterModel } from './CharacterModel';

// ─────────────────────────────────────────────────────────────────────────────
// Particle layout
// ─────────────────────────────────────────────────────────────────────────────

const P_HEAD = 0;
const P_NECK = 1;
const P_CHEST = 2;
const P_PELVIS = 3;
const P_SHOULDER_L = 4;
const P_ELBOW_L = 5;
const P_HAND_L = 6;
const P_SHOULDER_R = 7;
const P_ELBOW_R = 8;
const P_HAND_R = 9;
const P_HIP_L = 10;
const P_KNEE_L = 11;
const P_FOOT_L = 12;
const P_HIP_R = 13;
const P_KNEE_R = 14;
const P_FOOT_R = 15;
const PARTICLE_COUNT = 16;

/** Which bone (if any) each particle is initialised from, and the local offset. */
interface ParticleSeed {
  bone: BoneName;
  /** Offset along the bone's local -Y, i.e. down the limb. */
  down: number;
  radius: number;
  mass: number;
}

const SEEDS: ParticleSeed[] = [
  { bone: 'head', down: -0.09, radius: 0.11, mass: 5.0 }, // P_HEAD (above the bone)
  { bone: 'neck', down: 0.0, radius: 0.07, mass: 3.0 },
  { bone: 'chest', down: 0.0, radius: 0.16, mass: 16.0 },
  { bone: 'pelvis', down: 0.0, radius: 0.14, mass: 14.0 },
  { bone: 'upperArmL', down: 0.0, radius: 0.07, mass: 2.5 },
  { bone: 'lowerArmL', down: 0.0, radius: 0.06, mass: 1.8 },
  { bone: 'handL', down: 0.0, radius: 0.05, mass: 0.9 },
  { bone: 'upperArmR', down: 0.0, radius: 0.07, mass: 2.5 },
  { bone: 'lowerArmR', down: 0.0, radius: 0.06, mass: 1.8 },
  { bone: 'handR', down: 0.0, radius: 0.05, mass: 0.9 },
  { bone: 'upperLegL', down: 0.0, radius: 0.09, mass: 8.0 },
  { bone: 'lowerLegL', down: 0.0, radius: 0.08, mass: 4.0 },
  { bone: 'footL', down: 0.0, radius: 0.07, mass: 1.6 },
  { bone: 'upperLegR', down: 0.0, radius: 0.09, mass: 8.0 },
  { bone: 'lowerLegR', down: 0.0, radius: 0.08, mass: 4.0 },
  { bone: 'footR', down: 0.0, radius: 0.07, mass: 1.6 },
];

/** 0 = two-sided, 1 = minimum only (push apart), 2 = maximum only (pull together). */
const enum CMode {
  Stiff = 0,
  Min = 1,
  Max = 2,
}

interface ConstraintDef {
  a: number;
  b: number;
  /** Rest length multiplier applied to `chain`, or to the measured length. */
  scale: number;
  stiffness: number;
  mode: CMode;
  /**
   * Fixed reference length in metres, from the rig rather than the live pose.
   *
   * Structural (adjacent-joint) constraints can safely measure the death pose —
   * bone lengths are pose-invariant. Limit constraints that SPAN a bendable
   * joint cannot: a soldier killed mid-crouch has a folded leg, so a measured
   * "fully extended" limit would be the folded length and the corpse could
   * never straighten that leg again.
   */
  chain?: number;
}

const ARM_CHAIN = RIG_METRICS.upperArm + RIG_METRICS.lowerArm;
const LEG_CHAIN = RIG_METRICS.upperLeg + RIG_METRICS.lowerLeg;

/**
 * Skeleton constraints. The first block reproduces the bone lengths; the second
 * block is the "angle limit" set — non-adjacent pairs held apart or together so
 * joints cannot invert. That is the cheap trick that makes a distance-only
 * solver look like it has real joints.
 */
const CONSTRAINTS: ConstraintDef[] = [
  // Spine / neck.
  { a: P_HEAD, b: P_NECK, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_NECK, b: P_CHEST, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_CHEST, b: P_PELVIS, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  // Arms.
  { a: P_CHEST, b: P_SHOULDER_L, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_SHOULDER_L, b: P_ELBOW_L, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_ELBOW_L, b: P_HAND_L, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_CHEST, b: P_SHOULDER_R, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_SHOULDER_R, b: P_ELBOW_R, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_ELBOW_R, b: P_HAND_R, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  // Legs.
  { a: P_PELVIS, b: P_HIP_L, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_HIP_L, b: P_KNEE_L, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_KNEE_L, b: P_FOOT_L, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_PELVIS, b: P_HIP_R, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_HIP_R, b: P_KNEE_R, scale: 1, stiffness: 1.0, mode: CMode.Stiff },
  { a: P_KNEE_R, b: P_FOOT_R, scale: 1, stiffness: 1.0, mode: CMode.Stiff },

  // Torso rigidity — the ribcage is not a rope.
  { a: P_SHOULDER_L, b: P_SHOULDER_R, scale: 1, stiffness: 0.9, mode: CMode.Stiff },
  { a: P_HIP_L, b: P_HIP_R, scale: 1, stiffness: 0.9, mode: CMode.Stiff },
  { a: P_SHOULDER_L, b: P_PELVIS, scale: 1, stiffness: 0.55, mode: CMode.Stiff },
  { a: P_SHOULDER_R, b: P_PELVIS, scale: 1, stiffness: 0.55, mode: CMode.Stiff },
  { a: P_HEAD, b: P_CHEST, scale: 1, stiffness: 0.45, mode: CMode.Stiff },
  { a: P_CHEST, b: P_HIP_L, scale: 1, stiffness: 0.4, mode: CMode.Stiff },
  { a: P_CHEST, b: P_HIP_R, scale: 1, stiffness: 0.4, mode: CMode.Stiff },

  // Joint limits: elbows and knees may fold but never invert, and limbs cannot
  // pass through the torso.
  { a: P_SHOULDER_L, b: P_HAND_L, scale: 0.42, stiffness: 0.5, mode: CMode.Min, chain: ARM_CHAIN },
  { a: P_SHOULDER_R, b: P_HAND_R, scale: 0.42, stiffness: 0.5, mode: CMode.Min, chain: ARM_CHAIN },
  { a: P_HIP_L, b: P_FOOT_L, scale: 0.5, stiffness: 0.6, mode: CMode.Min, chain: LEG_CHAIN },
  { a: P_HIP_R, b: P_FOOT_R, scale: 0.5, stiffness: 0.6, mode: CMode.Min, chain: LEG_CHAIN },
  { a: P_SHOULDER_L, b: P_HAND_L, scale: 0.99, stiffness: 0.35, mode: CMode.Max, chain: ARM_CHAIN },
  { a: P_SHOULDER_R, b: P_HAND_R, scale: 0.99, stiffness: 0.35, mode: CMode.Max, chain: ARM_CHAIN },
  { a: P_HIP_L, b: P_FOOT_L, scale: 0.99, stiffness: 0.45, mode: CMode.Max, chain: LEG_CHAIN },
  { a: P_HIP_R, b: P_FOOT_R, scale: 0.99, stiffness: 0.45, mode: CMode.Max, chain: LEG_CHAIN },
  // Measured live and deliberately so: "do not compress further than you
  // already are" is meaningful from any death pose, whereas a fixed rig
  // distance would shove the arms out on a body that died with them tucked.
  { a: P_PELVIS, b: P_ELBOW_L, scale: 0.7, stiffness: 0.25, mode: CMode.Min },
  { a: P_PELVIS, b: P_ELBOW_R, scale: 0.7, stiffness: 0.25, mode: CMode.Min },
  { a: P_KNEE_L, b: P_KNEE_R, scale: 0.75, stiffness: 0.3, mode: CMode.Min },
  { a: P_FOOT_L, b: P_FOOT_R, scale: 0.6, stiffness: 0.25, mode: CMode.Min },
  { a: P_HEAD, b: P_PELVIS, scale: 0.55, stiffness: 0.5, mode: CMode.Min },
];

// ── Tuning ───────────────────────────────────────────────────────────────────

/** Same doubled gravity as the player, so corpses fall at the game's pace. */
const RAGDOLL_GRAVITY = -19.6;
const SOLVER_ITERATIONS = 7;
const SUBSTEPS = 2;
const AIR_DAMPING = 0.994;
const GROUND_FRICTION = 0.62;
const GROUND_RESTITUTION = 0.12;
/** Average particle speed (m/s) under which the body freezes and stops solving. */
const SLEEP_EPSILON = 0.22;
const SLEEP_TIME = 0.6;
export const RAGDOLL_LIFETIME = 12;
const RAGDOLL_FADE_START = 9;

// Scratch.
const _r0 = new THREE.Vector3();
const _r1 = new THREE.Vector3();
const _r2 = new THREE.Vector3();
const _r3 = new THREE.Vector3();
const _rq = new THREE.Quaternion();
const _rqPelvis = new THREE.Quaternion();
const _rqTorso = new THREE.Quaternion();
const _rm = new THREE.Matrix4();
const _hit = new ProbeHit();
const _DOWN = new THREE.Vector3(0, -1, 0);
const _UP = new THREE.Vector3(0, 1, 0);

/** Limb chains driven as pure two-point aims: [bone, parentBone, headP, tailP]. */
const LIMB_CHAINS: Array<[BoneName, BoneName | null, number, number]> = [
  ['upperArmL', 'chest', P_SHOULDER_L, P_ELBOW_L],
  ['lowerArmL', 'upperArmL', P_ELBOW_L, P_HAND_L],
  ['upperArmR', 'chest', P_SHOULDER_R, P_ELBOW_R],
  ['lowerArmR', 'upperArmR', P_ELBOW_R, P_HAND_R],
  ['upperLegL', 'pelvis', P_HIP_L, P_KNEE_L],
  ['lowerLegL', 'upperLegL', P_KNEE_L, P_FOOT_L],
  ['upperLegR', 'pelvis', P_HIP_R, P_KNEE_R],
  ['lowerLegR', 'upperLegR', P_KNEE_R, P_FOOT_R],
];

export class Ragdoll {
  active = false;
  /** Seconds since activation. Drives fade-out and despawn. */
  age = 0;

  private readonly pos = new Float32Array(PARTICLE_COUNT * 3);
  private readonly prev = new Float32Array(PARTICLE_COUNT * 3);
  private readonly invMass = new Float32Array(PARTICLE_COUNT);
  private readonly radius = new Float32Array(PARTICLE_COUNT);
  private readonly rest = new Float32Array(CONSTRAINTS.length);
  /** Cached floor height under each particle; refreshed round-robin. */
  private readonly groundY = new Float32Array(PARTICLE_COUNT);
  private readonly groundOk = new Uint8Array(PARTICLE_COUNT);
  private groundCursor = 0;
  /**
   * Launch velocity in m/s, applied on the first integration step.
   *
   * Verlet stores velocity implicitly as (pos - prev), which only means a
   * velocity relative to the step size actually used. Baking it at activate
   * time against an assumed 1/60 makes the launch 2x too strong at 60 fps and
   * ~5x at 144 fps, so it is kept explicitly and converted once h is known.
   */
  private readonly launchVel = new Float32Array(PARTICLE_COUNT * 3);
  private launchPending = false;

  private model: CharacterModel | null = null;
  private probe: WorldProbe | null = null;
  private sleepTimer = 0;
  private asleep = false;
  private opacity = 1;

  constructor() {
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      this.invMass[i] = 1 / SEEDS[i].mass;
      this.radius[i] = SEEDS[i].radius;
    }
  }

  /**
   * Snapshot the current pose into particles and let physics take over.
   *
   * @param bodyVelocity the soldier's velocity at the moment of death — a
   *        sprinting target must tumble forwards, not drop straight down.
   * @param impulse      world-space impulse of the killing shot (direction ×
   *        magnitude in m/s applied to the struck region).
   * @param impactPoint  where the shot landed; the nearest particles take most
   *        of the impulse so a headshot snaps the head back.
   */
  activate(
    model: CharacterModel,
    probe: WorldProbe,
    bodyVelocity: THREE.Vector3,
    impulse: THREE.Vector3,
    impactPoint: THREE.Vector3,
  ): void {
    this.model = model;
    this.probe = probe;
    this.active = true;
    this.age = 0;
    this.sleepTimer = 0;
    this.asleep = false;
    this.opacity = 1;
    this.groundCursor = 0;
    this.groundOk.fill(0);

    model.root.updateMatrixWorld(true);

    // Seed particle positions from the live pose so the transition is seamless.
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const seed = SEEDS[i];
      const bone = model.bones[seed.bone];
      bone.getWorldPosition(_r0);
      if (seed.down !== 0) {
        bone.getWorldQuaternion(_rq);
        _r1.copy(_DOWN).applyQuaternion(_rq).multiplyScalar(seed.down);
        _r0.add(_r1);
      }
      this.pos[i * 3] = _r0.x;
      this.pos[i * 3 + 1] = _r0.y;
      this.pos[i * 3 + 2] = _r0.z;
    }
    // Shoulders and hips are not bones in this rig; derive them from the arm
    // and leg roots, which already sit at the joint.
    // (upperArm* / upperLeg* bones ARE those joints, so nothing extra to do.)

    // Measure rest lengths from the live pose: the constraint set then encodes
    // whatever proportions the model actually has.
    for (let c = 0; c < CONSTRAINTS.length; c++) {
      const def = CONSTRAINTS[c];
      this.rest[c] = (def.chain ?? this.dist(def.a, def.b)) * def.scale;
    }

    // Record the launch velocity; `prev` is derived from it once the real step
    // size is known (see `launchVel`).
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      this.launchVel[i * 3] = bodyVelocity.x;
      this.launchVel[i * 3 + 1] = bodyVelocity.y;
      this.launchVel[i * 3 + 2] = bodyVelocity.z;
      this.prev[i * 3] = this.pos[i * 3];
      this.prev[i * 3 + 1] = this.pos[i * 3 + 1];
      this.prev[i * 3 + 2] = this.pos[i * 3 + 2];
    }
    this.launchPending = true;

    // Distribute the killing impulse with an inverse-square falloff from the
    // impact point, then a small global share so the whole body reacts.
    let totalW = 0;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const dx = this.pos[i * 3] - impactPoint.x;
      const dy = this.pos[i * 3 + 1] - impactPoint.y;
      const dz = this.pos[i * 3 + 2] - impactPoint.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      totalW += 1 / (0.09 + d2);
    }
    if (totalW < 1e-6) totalW = 1;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const dx = this.pos[i * 3] - impactPoint.x;
      const dy = this.pos[i * 3 + 1] - impactPoint.y;
      const dz = this.pos[i * 3 + 2] - impactPoint.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      const local = (1 / (0.09 + d2) / totalW) * 0.75;
      const share = local + 0.25 / PARTICLE_COUNT;
      // `impulse` is momentum (kg·m/s): dividing by mass is what makes a head
      // snap back hard while the torso barely shifts.
      const k = share * this.invMass[i];
      this.launchVel[i * 3] += impulse.x * k;
      this.launchVel[i * 3 + 1] += impulse.y * k;
      this.launchVel[i * 3 + 2] += impulse.z * k;
    }
  }

  reset(): void {
    this.active = false;
    this.model = null;
    this.probe = null;
    this.age = 0;
    this.opacity = 1;
    this.asleep = false;
    this.sleepTimer = 0;
    this.launchPending = false;
  }

  private dist(a: number, b: number): number {
    const dx = this.pos[a * 3] - this.pos[b * 3];
    const dy = this.pos[a * 3 + 1] - this.pos[b * 3 + 1];
    const dz = this.pos[a * 3 + 2] - this.pos[b * 3 + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-5;
  }

  /** @returns false once the body has expired and should be despawned. */
  update(dt: number): boolean {
    if (!this.active || !this.model) return false;

    this.age += dt;
    if (this.age >= RAGDOLL_LIFETIME) {
      this.active = false;
      return false;
    }

    if (!this.asleep) {
      const h = Math.min(dt, 1 / 45) / SUBSTEPS;
      if (this.launchPending) {
        // Now that h is known, express the launch velocity as a Verlet offset.
        this.launchPending = false;
        for (let i = 0; i < PARTICLE_COUNT; i++) {
          this.prev[i * 3] = this.pos[i * 3] - this.launchVel[i * 3] * h;
          this.prev[i * 3 + 1] = this.pos[i * 3 + 1] - this.launchVel[i * 3 + 1] * h;
          this.prev[i * 3 + 2] = this.pos[i * 3 + 2] - this.launchVel[i * 3 + 2] * h;
        }
      }
      for (let s = 0; s < SUBSTEPS; s++) {
        this.integrate(h);
        for (let it = 0; it < SOLVER_ITERATIONS; it++) this.solve();
        this.collide();
      }
      this.checkSleep(dt, h);
    }

    this.driveBones();

    // Fade out at the end of life. Corpses that pop out of existence read as a
    // bug; a three-second dissolve reads as intentional.
    if (this.age > RAGDOLL_FADE_START) {
      const t = (this.age - RAGDOLL_FADE_START) / (RAGDOLL_LIFETIME - RAGDOLL_FADE_START);
      const o = 1 - t * t;
      if (Math.abs(o - this.opacity) > 0.01) {
        this.opacity = o;
        this.model.setOpacity(o);
      }
    }
    return true;
  }

  private integrate(h: number): void {
    const g = RAGDOLL_GRAVITY * h * h;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;
      for (let k = 0; k < 3; k++) {
        const p = this.pos[i3 + k];
        const v = (p - this.prev[i3 + k]) * AIR_DAMPING;
        this.prev[i3 + k] = p;
        this.pos[i3 + k] = p + v + (k === 1 ? g : 0);
      }
    }
  }

  private solve(): void {
    for (let c = 0; c < CONSTRAINTS.length; c++) {
      const def = CONSTRAINTS[c];
      const rest = this.rest[c];
      const a3 = def.a * 3;
      const b3 = def.b * 3;
      const dx = this.pos[b3] - this.pos[a3];
      const dy = this.pos[b3 + 1] - this.pos[a3 + 1];
      const dz = this.pos[b3 + 2] - this.pos[a3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < 1e-6) continue;
      if (def.mode === CMode.Min && d >= rest) continue;
      if (def.mode === CMode.Max && d <= rest) continue;

      const wa = this.invMass[def.a];
      const wb = this.invMass[def.b];
      const wsum = wa + wb;
      if (wsum < 1e-8) continue;

      const diff = ((d - rest) / d) * def.stiffness;
      const ax = dx * diff * (wa / wsum);
      const ay = dy * diff * (wa / wsum);
      const az = dz * diff * (wa / wsum);
      const bx = dx * diff * (wb / wsum);
      const by = dy * diff * (wb / wsum);
      const bz = dz * diff * (wb / wsum);
      this.pos[a3] += ax;
      this.pos[a3 + 1] += ay;
      this.pos[a3 + 2] += az;
      this.pos[b3] -= bx;
      this.pos[b3 + 1] -= by;
      this.pos[b3 + 2] -= bz;
    }
  }

  /**
   * Ground contact. Floor heights are sampled round-robin (four particles per
   * frame) rather than every particle every frame: a corpse does not move fast
   * enough for the staleness to be visible, and it keeps the raycast budget at
   * a handful of rays per body.
   */
  private collide(): void {
    const probe = this.probe;
    if (probe) {
      for (let n = 0; n < 4; n++) {
        const i = this.groundCursor;
        this.groundCursor = (this.groundCursor + 1) % PARTICLE_COUNT;
        const i3 = i * 3;
        if (probe.ground(this.pos[i3], this.pos[i3 + 2], this.pos[i3 + 1] + 1.2, 6, _hit)) {
          this.groundY[i] = _hit.point.y;
          this.groundOk[i] = 1;
        } else {
          this.groundOk[i] = 0;
        }
      }
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      if (!probe) break;
      if (this.groundOk[i] === 0) continue;
      const i3 = i * 3;
      const floor = this.groundY[i] + this.radius[i];
      if (this.pos[i3 + 1] >= floor) continue;

      const vy = this.pos[i3 + 1] - this.prev[i3 + 1];
      this.pos[i3 + 1] = floor;
      // Kill most of the normal velocity and shear the tangential component:
      // limbs should slap down and stay, not skitter across the floor.
      this.prev[i3 + 1] = floor + vy * GROUND_RESTITUTION;
      const vx = this.pos[i3] - this.prev[i3];
      const vz = this.pos[i3 + 2] - this.prev[i3 + 2];
      this.prev[i3] = this.pos[i3] - vx * (1 - GROUND_FRICTION);
      this.prev[i3 + 2] = this.pos[i3 + 2] - vz * (1 - GROUND_FRICTION);
    }
  }

  private checkSleep(dt: number, h: number): void {
    let motion = 0;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;
      const dx = this.pos[i3] - this.prev[i3];
      const dy = this.pos[i3 + 1] - this.prev[i3 + 1];
      const dz = this.pos[i3 + 2] - this.prev[i3 + 2];
      motion += Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
    }
    // Normalised to a speed, so the sleep threshold means the same thing at
    // every frame rate rather than scaling with the substep size.
    if (motion / (PARTICLE_COUNT * Math.max(h, 1e-5)) < SLEEP_EPSILON) {
      this.sleepTimer += dt;
      if (this.sleepTimer > SLEEP_TIME) this.asleep = true;
    } else {
      this.sleepTimer = 0;
    }
  }

  private p(i: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(this.pos[i * 3], this.pos[i * 3 + 1], this.pos[i * 3 + 2]);
  }

  /**
   * Push particle positions back onto the bone hierarchy.
   *
   * The model root is forced to identity so model space equals world space;
   * every bone is then given a local rotation derived from its parent's world
   * rotation. Limb bones are pure two-point aims (twist is irrelevant on a
   * corpse); torso bones get a full orthonormal basis so shoulders and hips
   * stay square.
   */
  private driveBones(): void {
    const model = this.model;
    if (!model) return;
    const b = model.bones;

    model.root.position.set(0, 0, 0);
    model.root.quaternion.identity();
    model.root.scale.set(1, 1, 1);

    // ── Pelvis: full basis from (chest - pelvis) and (hipL - hipR).
    this.p(P_CHEST, _r0);
    this.p(P_PELVIS, _r1);
    this.p(P_HIP_L, _r2);
    this.p(P_HIP_R, _r3);
    b.pelvis.position.copy(_r1);
    basisQuat(_r0.sub(_r1), _r2.sub(_r3), _rqPelvis);
    b.pelvis.quaternion.copy(_rqPelvis);

    // ── Spine + chest: aim at the neck, square the shoulders.
    this.p(P_NECK, _r0);
    this.p(P_CHEST, _r1);
    this.p(P_SHOULDER_L, _r2);
    this.p(P_SHOULDER_R, _r3);
    basisQuat(_r0.sub(_r1), _r2.sub(_r3), _rqTorso);
    // chest is a grandchild of pelvis via spine; put the whole torso rotation on
    // `spine` and leave `chest` neutral so the two do not double up.
    b.spine.quaternion.copy(_rqPelvis).invert().multiply(_rqTorso);
    b.chest.quaternion.identity();

    // ── Neck + head.
    this.p(P_HEAD, _r0);
    this.p(P_NECK, _r1);
    _r0.sub(_r1);
    if (_r0.lengthSq() > 1e-8) {
      _r0.normalize().applyQuaternion(_rq.copy(_rqTorso).invert());
      b.neck.quaternion.setFromUnitVectors(_UP, _r0);
    }
    b.head.quaternion.identity();

    // Torso rotations are set; refresh world matrices so the limb pass below
    // reads correct parent frames.
    model.root.updateMatrixWorld(true);

    // ── Limbs (ordered parent-first, so each parent's matrixWorld is fresh).
    for (let i = 0; i < LIMB_CHAINS.length; i++) {
      const chain = LIMB_CHAINS[i];
      const bone = b[chain[0]];
      const parent = bone.parent;
      if (!parent) continue;
      this.p(chain[2], _r0);
      this.p(chain[3], _r1);
      _r1.sub(_r0);
      if (_r1.lengthSq() < 1e-8) continue;
      _r1.normalize();
      // Convert the world-space limb direction into the parent's local frame,
      // then align the bone's own -Y with it.
      parent.updateMatrixWorld();
      _rm.copy(parent.matrixWorld);
      _rm.setPosition(0, 0, 0);
      _rq.setFromRotationMatrix(_rm).invert();
      _r1.applyQuaternion(_rq);
      bone.quaternion.setFromUnitVectors(_DOWN, _r1);
      bone.updateMatrixWorld();
    }

    // Feet and hands relax; a corpse has no muscle tone.
    b.footL.rotation.set(0.45, 0, 0);
    b.footR.rotation.set(0.45, 0, 0);
    b.handL.rotation.set(0.2, 0, 0);
    b.handR.rotation.set(0.2, 0, 0);

    model.root.updateMatrixWorld(true);
  }
}

/**
 * Builds an orthonormal rotation from an up vector and a right hint.
 * The model faces +Z with +X to its right, so the basis is (right, up, fwd).
 */
function basisQuat(up: THREE.Vector3, rightHint: THREE.Vector3, out: THREE.Quaternion): void {
  if (up.lengthSq() < 1e-8) {
    out.identity();
    return;
  }
  up.normalize();
  if (rightHint.lengthSq() < 1e-8) rightHint.set(1, 0, 0);
  rightHint.normalize();
  // fwd = right × up, then re-derive right so the basis is exactly orthonormal.
  const fx = rightHint.y * up.z - rightHint.z * up.y;
  const fy = rightHint.z * up.x - rightHint.x * up.z;
  const fz = rightHint.x * up.y - rightHint.y * up.x;
  const fl = Math.hypot(fx, fy, fz) || 1;
  const nfx = fx / fl;
  const nfy = fy / fl;
  const nfz = fz / fl;
  const rx = up.y * nfz - up.z * nfy;
  const ry = up.z * nfx - up.x * nfz;
  const rz = up.x * nfy - up.y * nfx;
  _rm.set(
    rx, up.x, nfx, 0,
    ry, up.y, nfy, 0,
    rz, up.z, nfz, 0,
    0, 0, 0, 1,
  );
  out.setFromRotationMatrix(_rm);
}
