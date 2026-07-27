/**
 * NOISEFIELD — Lightweight rigid-body solver.
 *
 * Purpose-built for what an FPS actually throws around: shell casings, debris,
 * grenades, ragdoll segments and the occasional physics prop. Body-vs-*world*
 * is the whole job; body-vs-body is deliberately not simulated (see
 * `README` note at the bottom of this header).
 *
 * Design choices, and why:
 *
 *  - **Sphere-cluster shapes.** Every body is approximated by ≤ 8 spheres in
 *    local space. One shape kernel (sphere-vs-triangle) covers boxes, capsules
 *    and spheres, which keeps the narrow phase branch-free and lets a single
 *    BVH traversal per body feed every sphere.
 *
 *  - **Isotropic scalar inertia.** `I = k·m·R²` as a *scalar*, so `I⁻¹` needs no
 *    basis rotation each step. For tumbling debris the anisotropy of a real box
 *    tensor is visually undetectable, and this removes a 3×3 rotate + inverse
 *    from the innermost loop.
 *
 *  - **Semi-implicit Euler + sequential impulses.** Velocity is integrated
 *    before position, contacts are solved Gauss-Seidel with accumulated normal
 *    impulses so Coulomb friction clamps against the *total* normal impulse
 *    rather than one iteration's slice.
 *
 *  - **Positional projection with slop, not pure Baumgarte.** Baumgarte feeds
 *    correction energy back into velocity, which makes stacked casings buzz.
 *    Direct projection at `BAUMGARTE` strength with a `slop` band is stable and
 *    adds no energy.
 *
 *  - **Sleep.** Jittering casings that never settle are an instant quality
 *    tell, so bodies that stay slow *while supported* for `SLEEP_TIME` are
 *    frozen outright and cost zero until something wakes them.
 *
 * Body-vs-body is omitted on purpose: casings interpenetrating each other by a
 * few millimetres in a pile is invisible, whereas the broad phase + N² contact
 * budget is not. If it is ever needed, hash the sphere clusters into a uniform
 * grid and reuse `solveContact` — the impulse code is already shape-agnostic.
 */

import * as THREE from 'three';
import type { DynamicBodyOptions, EntityId, SurfaceType } from '../core/Types';
import {
  BVHWorld,
  closestPointOnTriangle,
  createRayResult,
  lastContactX,
  lastContactY,
  lastContactZ,
  type RayResult,
  type TriangleVisitor,
} from './BVHWorld';

// ─────────────────────────────────────────────────────────────────────────────
// Tunables — every one of these is a quality/stability tradeoff
// ─────────────────────────────────────────────────────────────────────────────

/** Max collision spheres per body. 8 = box corners. */
export const MAX_SPHERES = 8;

/** Contacts retained per body per step. Deeper contacts win. */
const MAX_CONTACTS = 10;

/** Gauss-Seidel velocity passes. 2 is enough for ≤ 10 contacts on one body. */
const VELOCITY_ITERATIONS = 2;

/** Projection passes. 3 converges corners (floor + two walls) without buzzing. */
const POSITION_ITERATIONS = 3;

/**
 * Allowed residual penetration. Below this we do not push, which is what stops
 * resting bodies from oscillating between "corrected" and "re-penetrating".
 * 4 mm is invisible on a crate but would swallow a 5 mm shell casing whole, so
 * the effective slop is additionally clamped to a fraction of the body radius.
 */
const PENETRATION_SLOP = 0.004;
const SLOP_RADIUS_FRACTION = 0.25;

/**
 * Continuous collision.
 *
 * A discrete solver samples collision only at the END of a step. A 6 mm shell
 * casing falling at 5 m/s travels 42 mm per 120 Hz step — seven times its own
 * radius — so it steps straight through a floor face and ends up *inside* the
 * solid, where no triangle is within its radius and it falls forever. This is
 * not a theoretical concern: it tunnelled 225 of 250 casings before the guard
 * existed.
 *
 * Whenever a body's per-step translation exceeds the smallest sphere in its
 * cluster, the translation is swept (exact swept-sphere against the BVH) and
 * clamped to the first impact. The body is then nudged marginally *past* the
 * touch point so the discrete narrow phase below still registers a contact and
 * can apply the restitution/friction impulse.
 */
const CCD_MOTION_RATIO = 0.5;
/** Fraction of the sphere radius of guaranteed overlap left after a CCD clamp. */
const CCD_OVERLAP_FRACTION = 0.35;

/** Fraction of the excess penetration removed per projection pass. */
const BAUMGARTE = 0.7;

/**
 * Below this approach speed restitution is forced to zero. Without it every
 * body micro-bounces forever and never satisfies the sleep test.
 */
const RESTITUTION_THRESHOLD = 0.55;

const SLEEP_LINEAR = 0.055;   // m/s
const SLEEP_TIME = 0.4;       // s of continuous rest while supported

/**
 * Contact persistence, in steps.
 *
 * A resting body's contact set flickers: a slowly rolling casing loses its
 * contact for a single step every ~25 steps as the cluster sphere rotates
 * through the slop band. Treating that as "became airborne" reset the rest
 * timer before it could ever reach `SLEEP_TIME`, so 42 of 250 casings stayed
 * awake indefinitely at 0.7 mm/s. Real solvers cache contacts across steps for
 * exactly this reason.
 */
const CONTACT_PERSIST_STEPS = 6;

/**
 * Angular rest is judged by SURFACE speed (ω·r), not by ω itself.
 *
 * A fixed rad/s threshold is not scale invariant: a 6 mm shell casing settles
 * into a shallow rocking limit cycle at ~0.7 rad/s, which is 4 mm/s at its
 * surface — visually dead still — yet a fixed 0.75 rad/s cut-off leaves it
 * awake forever (71 of 250 casings never slept). Conversely 0.75 rad/s on a
 * half-metre crate is a clearly visible tumble that must NOT sleep.
 */
const SLEEP_TANGENTIAL = 0.05;  // m/s at the body's surface
/** Absolute ceiling so a very small body cannot sleep while genuinely spinning. */
const SLEEP_ANGULAR_MAX = 12;   // rad/s

/** Speed clamps. A bad impulse must never tunnel a body out of the level. */
const MAX_LINEAR_SPEED = 140;
const MAX_ANGULAR_SPEED = 70;

/**
 * Rolling resistance, as a coefficient on the contact's normal impulse.
 *
 * A flat multiplicative bleed (`w *= 0.985`) is NOT scale invariant and cannot
 * do this job. The friction solve correctly converts sliding into rolling, so a
 * body settles at omega = v/r; on a 5 mm casing a 27 mm/s crawl is 5.4 rad/s,
 * which a fixed decay barely touches while it sails past any sleep threshold.
 * Resistance proportional to the normal impulse and the body radius scales
 * automatically: it kills a casing's residual spin in ~8 steps while barely
 * slowing a heavy crate, exactly as real rolling friction does.
 */
const ROLLING_RESISTANCE = 0.03;

/** Explosion shaping. Games always bias upward — flat radial looks wrong. */
const EXPLOSION_UP_BIAS = 0.35;
const EXPLOSION_TUMBLE = 0.55;

/**
 * Impact-report threshold, in impulse per kilogram (i.e. a velocity change in
 * m/s). Well above the per-step impulse needed to hold a body against gravity
 * at 120 Hz (~0.08 m/s), so resting bodies never report.
 */
const IMPACT_REPORT_IMPULSE = 0.6;

/** Broadphase padding around a body's swept bounds, in metres. */
const BROADPHASE_MARGIN = 0.02;

/**
 * Handle packing: 20 bits index | 8 bits generation | 4 bits type tag.
 *
 * The tag exists because rigid-body handles and entity-collider handles are
 * both plain `number`s in the public API. Without it the two spaces produce
 * identical values, and `removeBody(colliderHandle)` silently destroys an
 * unrelated body. Handle 0 is always invalid.
 */
// 16 bits of index (65 536 live objects) leaves 12 for the generation. With
// only 8 generation bits a recycled slot repeated its handle after 255 reuses,
// which a casing pool reaches in seconds, and a stale handle then drove a live
// unrelated body.
const HANDLE_INDEX_BITS = 16;
export const HANDLE_INDEX_MASK = (1 << HANDLE_INDEX_BITS) - 1;
const HANDLE_GEN_BITS = 12;
const HANDLE_GEN_MASK = (1 << HANDLE_GEN_BITS) - 1;
export const HANDLE_TAG_SHIFT = HANDLE_INDEX_BITS + HANDLE_GEN_BITS;
export const HANDLE_TAG_BODY = 1;
export const HANDLE_TAG_COLLIDER = 2;

export function packHandle(tag: number, generation: number, index: number): number {
  return (
    ((tag & 0xf) << HANDLE_TAG_SHIFT)
    | ((generation & HANDLE_GEN_MASK) << HANDLE_INDEX_BITS)
    | (index & HANDLE_INDEX_MASK)
  ) >>> 0;
}

export function handleTag(handle: number): number {
  return (handle >>> HANDLE_TAG_SHIFT) & 0xf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch (module scope — single-threaded, non-re-entrant)
// ─────────────────────────────────────────────────────────────────────────────

let _rx = 0, _ry = 0, _rz = 0;

/** v' = q ⊗ v ⊗ q*, written to `_rx/_ry/_rz`. */
function rotateByQuat(
  qx: number, qy: number, qz: number, qw: number,
  vx: number, vy: number, vz: number,
): void {
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  _rx = vx + qw * tx + (qy * tz - qz * ty);
  _ry = vy + qw * ty + (qz * tx - qx * tz);
  _rz = vz + qw * tz + (qx * ty - qy * tx);
}

// ─────────────────────────────────────────────────────────────────────────────
// Body
// ─────────────────────────────────────────────────────────────────────────────

export type BodyShape = DynamicBodyOptions['shape'];

export class RigidBody {
  handle = 0;
  generation = 1;
  alive = false;

  object: THREE.Object3D | null = null;

  readonly position = new THREE.Vector3();
  readonly quaternion = new THREE.Quaternion();
  readonly velocity = new THREE.Vector3();
  readonly angularVelocity = new THREE.Vector3();

  shape: BodyShape = 'sphere';
  mass = 1;
  invMass = 1;
  /** Scalar (isotropic) inverse inertia — see the file header. */
  invInertia = 1;

  restitution = 0.25;
  friction = 0.6;
  linearDamping = 0.02;
  angularDamping = 0.08;
  surface: SurfaceType = 'metal';

  /** Bounding radius of the whole cluster; used for broadphase and slop. */
  boundingRadius = 0.1;

  /** Local-space collision spheres. */
  sphereCount = 1;
  readonly sphereLocal = new Float32Array(MAX_SPHERES * 3);
  readonly sphereRadius = new Float32Array(MAX_SPHERES);

  sleeping = false;
  restTimer = 0;
  /** True when the body is resting on something (with contact persistence). */
  supported = false;
  /** Consecutive steps with no contact; see `CONTACT_PERSIST_STEPS`. */
  airborneSteps = 0;

  /** Set when the owning object carries `userData.entityId` (ragdoll segments). */
  entity: EntityId | undefined = undefined;

  reset(): void {
    this.object = null;
    this.alive = false;
    this.position.set(0, 0, 0);
    this.quaternion.set(0, 0, 0, 1);
    this.velocity.set(0, 0, 0);
    this.angularVelocity.set(0, 0, 0);
    this.sleeping = false;
    this.restTimer = 0;
    this.supported = false;
    this.airborneSteps = 0;
    this.entity = undefined;
  }

  wake(): void {
    this.sleeping = false;
    this.restTimer = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape construction
// ─────────────────────────────────────────────────────────────────────────────

const _localBox = new THREE.Box3();
const _childBox = new THREE.Box3();
const _localMat = new THREE.Matrix4();
const _invRoot = new THREE.Matrix4();
const _tmpVec = new THREE.Vector3();
const _parentInv = new THREE.Matrix4();
const _parentInvQuat = new THREE.Quaternion();
const _pPos = new THREE.Vector3();
const _pQuat = new THREE.Quaternion();
const _pScale = new THREE.Vector3();

/**
 * Union of every descendant's geometry bounds expressed in `root`'s local
 * frame. Used to derive collision extents when the caller does not supply them.
 */
function computeLocalBounds(root: THREE.Object3D, target: THREE.Box3): boolean {
  target.makeEmpty();
  root.updateWorldMatrix(true, true);
  _invRoot.copy(root.matrixWorld).invert();

  let found = false;
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const geom = mesh.geometry;
    if (!geom.boundingBox) geom.computeBoundingBox();
    const bb = geom.boundingBox;
    if (!bb) return;
    _childBox.copy(bb);
    _localMat.multiplyMatrices(_invRoot, mesh.matrixWorld);
    _childBox.applyMatrix4(_localMat);
    target.union(_childBox);
    found = true;
  });
  return found;
}

/** Fill a body's local sphere cluster from its shape and half-extents. */
function buildSphereCluster(body: RigidBody, hx: number, hy: number, hz: number): void {
  const L = body.sphereLocal;
  const R = body.sphereRadius;

  switch (body.shape) {
    case 'sphere': {
      const r = Math.max(hx, hy, hz);
      body.sphereCount = 1;
      L[0] = 0; L[1] = 0; L[2] = 0;
      R[0] = r;
      body.boundingRadius = r;
      break;
    }

    case 'capsule': {
      // Y-up capsule, the three.js convention for CapsuleGeometry.
      const r = Math.max(1e-4, Math.max(hx, hz));
      const half = Math.max(0, hy - r);
      body.sphereCount = 2;
      L[0] = 0; L[1] = half; L[2] = 0;
      L[3] = 0; L[4] = -half; L[5] = 0;
      R[0] = r; R[1] = r;
      body.boundingRadius = half + r;
      break;
    }

    case 'convex': {
      // No convex hull support by design: a bounding sphere is honest and
      // costs nothing. Callers wanting fidelity should use 'box'.
      const r = Math.sqrt(hx * hx + hy * hy + hz * hz);
      body.sphereCount = 1;
      L[0] = 0; L[1] = 0; L[2] = 0;
      R[0] = r;
      body.boundingRadius = r;
      break;
    }

    case 'box':
    default: {
      // Eight corner spheres inset by their own radius, so their union is
      // exactly the box with corners/edges filleted at radius r: faces are
      // exact, and there are no gaps anywhere on the surface.
      //
      // r MUST be a fraction of the smallest half-extent, not the whole thing.
      // At r = min(h) the inset is zero for any near-cubic box and all eight
      // spheres collapse onto the single inscribed sphere — a crate then rolls
      // like a ball and, resting on a corner, buries it 181 mm into the floor.
      // Half gives the classic rounded-box cluster; the residual corner fillet
      // is r*(sqrt(3)-1) and only shows when a box comes to rest exactly on a
      // corner. Exact OBB-vs-triangle contact is the upgrade path if that ever
      // matters.
      const r = Math.max(1e-4, Math.min(hx, hy, hz) * 0.5);
      const ix = Math.max(0, hx - r);
      const iy = Math.max(0, hy - r);
      const iz = Math.max(0, hz - r);
      let n = 0;
      for (let sx = -1; sx <= 1; sx += 2) {
        for (let sy = -1; sy <= 1; sy += 2) {
          for (let sz = -1; sz <= 1; sz += 2) {
            L[n * 3] = ix * sx;
            L[n * 3 + 1] = iy * sy;
            L[n * 3 + 2] = iz * sz;
            R[n] = r;
            n++;
          }
        }
      }
      body.sphereCount = n;
      body.boundingRadius = Math.sqrt(hx * hx + hy * hy + hz * hz);
      break;
    }
  }
}

/** Isotropic inertia coefficient `k` in `I = k·m·R²`. */
function inertiaCoefficient(shape: BodyShape): number {
  switch (shape) {
    case 'sphere': return 0.4;    // exact solid sphere
    case 'capsule': return 0.32;  // between a rod and a sphere
    case 'box': return 0.35;      // cube tensor averaged over axes
    default: return 0.4;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solver
// ─────────────────────────────────────────────────────────────────────────────

export interface SolverStats {
  bodies: number;
  awake: number;
  contacts: number;
  /** Swept (continuous) collision resolutions performed on the last step. */
  ccdSweeps: number;
}

export class RigidBodySolver {
  readonly gravity = new THREE.Vector3(0, -9.81, 0);

  private readonly bodies: RigidBody[] = [];
  private readonly free: number[] = [];
  private readonly pool: RigidBody[] = [];

  private awakeCount = 0;
  private contactCount = 0;

  // ── per-body contact scratch (flat, reused) ──────────────────────────────
  private readonly cNx = new Float32Array(MAX_CONTACTS);
  private readonly cNy = new Float32Array(MAX_CONTACTS);
  private readonly cNz = new Float32Array(MAX_CONTACTS);
  /** World contact point (on the triangle). */
  private readonly cPx = new Float32Array(MAX_CONTACTS);
  private readonly cPy = new Float32Array(MAX_CONTACTS);
  private readonly cPz = new Float32Array(MAX_CONTACTS);
  private readonly cDepth = new Float32Array(MAX_CONTACTS);
  /** Accumulated normal impulse, so friction clamps against the total. */
  private readonly cAccum = new Float32Array(MAX_CONTACTS);
  /** Which local sphere produced the contact. */
  private readonly cSphere = new Int32Array(MAX_CONTACTS);
  /** The contacting triangle, copied in (stride 9). */
  private readonly cTriVerts = new Float32Array(MAX_CONTACTS * 9);
  /** Restitution target velocity, computed once before the impulse iterations. */
  private readonly cBias = new Float32Array(MAX_CONTACTS);
  private nContacts = 0;

  /** World centres of the current body's cluster spheres (stride 3). */
  private readonly sphereWorld = new Float32Array(MAX_SPHERES * 3);
  /** Body currently being visited by `visitTriangle`. */
  private visitBody: RigidBody | null = null;
  private visitSurface: SurfaceType = 'concrete';

  /** Reused CCD sweep result — see `CCD_MOTION_RATIO`. */
  private readonly sweep: RayResult = createRayResult();
  private ccdSweeps = 0;

  /**
   * Impact reporting. Set by `Physics` each step; see `Physics.onBodyImpact`.
   * Only the hardest contact of a step is reported, and only above
   * `IMPACT_REPORT_IMPULSE`, so a body resting on the floor stays silent.
   */
  onImpact: ((
    handle: number,
    point: THREE.Vector3,
    normal: THREE.Vector3,
    impulse: number,
    surface: SurfaceType,
  ) => void) | null = null;
  private readonly impactPoint = new THREE.Vector3();
  private readonly impactNormal = new THREE.Vector3();
  /** Surface of the triangle that produced each contact. */
  private readonly cSurface: SurfaceType[] = new Array<SurfaceType>(MAX_CONTACTS).fill('concrete');

  constructor(private readonly world: BVHWorld) {}

  get stats(): SolverStats {
    return {
      bodies: this.bodies.length - this.free.length,
      awake: this.awakeCount,
      contacts: this.contactCount,
      ccdSweeps: this.ccdSweeps,
    };
  }

  /** Live bodies, including sleeping ones. Iterate with an `alive` guard. */
  get all(): readonly RigidBody[] { return this.bodies; }

  // ───────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ───────────────────────────────────────────────────────────────────────────

  add(object: THREE.Object3D, opts: DynamicBodyOptions): number {
    const idx = this.free.length > 0 ? this.free.pop()! : this.bodies.length;
    let body: RigidBody;
    if (idx < this.bodies.length) {
      body = this.bodies[idx];
      body.generation = (body.generation + 1) & HANDLE_GEN_MASK;
      if (body.generation === 0) body.generation = 1;
    } else {
      body = this.pool.pop() ?? new RigidBody();
      this.bodies.push(body);
    }
    body.reset();
    body.alive = true;
    body.handle = packHandle(HANDLE_TAG_BODY, body.generation, idx);

    body.object = object;
    body.shape = opts.shape;
    // `Math.max(1e-4, NaN)` is NaN and `1/Infinity` is 0 (an immovable body
    // that gravity still accelerates, i.e. one that falls through the world).
    // Clamp into a finite, invertible range instead.
    body.mass = Number.isFinite(opts.mass) ? Math.min(1e6, Math.max(1e-4, opts.mass)) : 1;
    body.invMass = 1 / body.mass;
    body.restitution = opts.restitution ?? 0.25;
    body.friction = opts.friction ?? 0.6;
    body.linearDamping = opts.linearDamping ?? 0.02;
    body.angularDamping = opts.angularDamping ?? 0.08;
    body.surface = opts.surface ?? 'metal';

    object.updateWorldMatrix(true, false);
    object.getWorldPosition(body.position);
    object.getWorldQuaternion(body.quaternion);

    // Extents: explicit userData wins, then derived geometry bounds, then a
    // small default so a bare proxy Object3D still behaves sanely.
    let hx = 0.05, hy = 0.05, hz = 0.05;
    const udHalf = object.userData['physicsHalfExtents'];
    const udRadius = object.userData['physicsRadius'];
    const udHeight = object.userData['physicsHeight'];
    if (udHalf instanceof THREE.Vector3) {
      hx = Math.abs(udHalf.x); hy = Math.abs(udHalf.y); hz = Math.abs(udHalf.z);
    } else if (typeof udRadius === 'number' && udRadius > 0) {
      hx = hz = udRadius;
      // `physicsHeight` is the capsule's total tip-to-tip length; without it a
      // casing degenerates to a single sphere, which is a valid but rounder
      // proxy than the caller probably wanted.
      hy = typeof udHeight === 'number' && udHeight > 0 ? udHeight * 0.5 : udRadius;
    } else if (computeLocalBounds(object, _localBox)) {
      _localBox.getSize(_tmpVec);
      hx = Math.max(1e-4, _tmpVec.x * 0.5);
      hy = Math.max(1e-4, _tmpVec.y * 0.5);
      hz = Math.max(1e-4, _tmpVec.z * 0.5);
    }
    // A zero-scaled or NaN-transformed object yields non-finite extents, which
    // become a NaN bounding radius and then a NaN position that never recovers.
    if (!Number.isFinite(hx) || hx <= 0) hx = 0.05;
    if (!Number.isFinite(hy) || hy <= 0) hy = 0.05;
    if (!Number.isFinite(hz) || hz <= 0) hz = 0.05;
    buildSphereCluster(body, hx, hy, hz);

    const k = inertiaCoefficient(body.shape);
    const rr = Math.max(1e-6, body.boundingRadius * body.boundingRadius);
    body.invInertia = 1 / (k * body.mass * rr);

    const eid = object.userData['entityId'];
    body.entity = typeof eid === 'number' ? eid : undefined;

    return body.handle;
  }

  remove(handle: number): void {
    const body = this.get(handle);
    if (!body) return;
    const idx = handle & HANDLE_INDEX_MASK;
    body.alive = false;
    body.object = null;
    body.handle = 0;
    this.free.push(idx);
  }

  get(handle: number): RigidBody | undefined {
    if (handle === 0 || handleTag(handle) !== HANDLE_TAG_BODY) return undefined;
    const idx = handle & HANDLE_INDEX_MASK;
    if (idx >= this.bodies.length) return undefined;
    const body = this.bodies[idx];
    if (!body.alive || body.handle !== handle) return undefined;
    return body;
  }

  /**
   * Kill every body but KEEP the slots and their generation counters.
   *
   * Truncating the array would restart generations at 1, so a handle held from
   * before the clear would be byte-identical to a handle minted after it and
   * would silently drive an unrelated body.
   */
  clear(): void {
    this.free.length = 0;
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      b.alive = false;
      b.object = null;
      b.handle = 0;
      b.generation = (b.generation + 1) & HANDLE_GEN_MASK;
      if (b.generation === 0) b.generation = 1;
      this.free.push(i);
    }
  }

  /** Wake everything. Called after the static world changes under the bodies. */
  wakeAll(): void {
    for (const b of this.bodies) if (b.alive) b.wake();
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Impulses
  // ───────────────────────────────────────────────────────────────────────────

  applyImpulse(handle: number, ix: number, iy: number, iz: number, px?: number, py?: number, pz?: number): void {
    const body = this.get(handle);
    if (!body) return;
    // A single non-finite impulse permanently poisons the body: NaN propagates
    // into position and orientation, `clampBody` cannot recover it (every
    // comparison against NaN is false), and the body then costs full solver
    // time forever while rendering nowhere. Reject at the boundary.
    if (!Number.isFinite(ix) || !Number.isFinite(iy) || !Number.isFinite(iz)) return;
    body.wake();

    const im = body.invMass;
    body.velocity.x += ix * im;
    body.velocity.y += iy * im;
    body.velocity.z += iz * im;

    if (
      px !== undefined && py !== undefined && pz !== undefined
      && Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)
    ) {
      // Off-centre application ⇒ torque τ = r × J.
      const rx = px - body.position.x;
      const ry = py - body.position.y;
      const rz = pz - body.position.z;
      const ii = body.invInertia;
      body.angularVelocity.x += (ry * iz - rz * iy) * ii;
      body.angularVelocity.y += (rz * ix - rx * iz) * ii;
      body.angularVelocity.z += (rx * iy - ry * ix) * ii;
    }
    this.clampBody(body);
  }

  /**
   * Explosion. Falloff is `(1 - d/R)²` rather than true inverse-square: the
   * physical law is singular at the origin and would launch anything sitting on
   * the charge into orbit. Quadratic reaches exactly zero at the radius so
   * bodies at the fringe do not twitch.
   */
  applyRadialImpulse(cx: number, cy: number, cz: number, radius: number, power: number): void {
    // Guarded at the boundary: this is wired straight to the `explosion` event,
    // so one malformed payload from any system would NaN every body in the
    // level irrecoverably (and cost an 8x permanent step-time blowup).
    if (!(radius > 0) || !Number.isFinite(radius) || !Number.isFinite(power)) return;
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cz)) return;
    const invR = 1 / radius;

    for (const body of this.bodies) {
      if (!body.alive) continue;
      const dx = body.position.x - cx;
      const dy = body.position.y - cy;
      const dz = body.position.z - cz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > radius * radius) continue;

      const d = Math.sqrt(d2);
      const f = 1 - d * invR;
      const fall = f * f;
      if (fall <= 0) continue;

      let ux: number, uy: number, uz: number;
      if (d < 1e-4) {
        ux = 0; uy = 1; uz = 0;
      } else {
        // Upward bias, expressed as a fraction of the blast radius so it scales
        // with the explosion instead of being a magic constant in metres.
        ux = dx; uy = dy + radius * EXPLOSION_UP_BIAS; uz = dz;
        const l = Math.sqrt(ux * ux + uy * uy + uz * uz);
        const inv = 1 / (l > 1e-6 ? l : 1);
        ux *= inv; uy *= inv; uz *= inv;
      }

      const j = power * fall;
      const im = body.invMass;
      body.velocity.x += ux * j * im;
      body.velocity.y += uy * j * im;
      body.velocity.z += uz * j * im;

      // Deterministic tumble: spin about an axis perpendicular to the blast
      // direction. Picking the reference axis by |uy| avoids a degenerate cross
      // product when the blast pushes straight up.
      const refX = Math.abs(uy) < 0.9 ? 0 : 1;
      const refY = Math.abs(uy) < 0.9 ? 1 : 0;
      let ax = uy * 0 - uz * refY;
      let ay = uz * refX - ux * 0;
      let az = ux * refY - uy * refX;
      const al = Math.sqrt(ax * ax + ay * ay + az * az);
      if (al > 1e-6) {
        const inv = 1 / al;
        ax *= inv; ay *= inv; az *= inv;
        const spin = j * body.invInertia * body.boundingRadius * EXPLOSION_TUMBLE;
        body.angularVelocity.x += ax * spin;
        body.angularVelocity.y += ay * spin;
        body.angularVelocity.z += az * spin;
      }

      body.wake();
      this.clampBody(body);
    }
  }

  private clampBody(body: RigidBody): void {
    const v = body.velocity;
    const w0 = body.angularVelocity;
    // Last line of defence. `x > max` is false for NaN, so the clamps below
    // cannot catch it; an explicit finite check is the only way back.
    if (!Number.isFinite(v.x) || !Number.isFinite(v.y) || !Number.isFinite(v.z)) v.set(0, 0, 0);
    if (!Number.isFinite(w0.x) || !Number.isFinite(w0.y) || !Number.isFinite(w0.z)) w0.set(0, 0, 0);
    const p = body.position;
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
      p.set(0, 0, 0);
      body.sleeping = true;
    }
    const s2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (s2 > MAX_LINEAR_SPEED * MAX_LINEAR_SPEED) {
      const k = MAX_LINEAR_SPEED / Math.sqrt(s2);
      v.x *= k; v.y *= k; v.z *= k;
    }
    const w = body.angularVelocity;
    const a2 = w.x * w.x + w.y * w.y + w.z * w.z;
    if (a2 > MAX_ANGULAR_SPEED * MAX_ANGULAR_SPEED) {
      const k = MAX_ANGULAR_SPEED / Math.sqrt(a2);
      w.x *= k; w.y *= k; w.z *= k;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Step
  // ───────────────────────────────────────────────────────────────────────────

  step(dt: number): void {
    // Negated so a NaN step is rejected (`NaN <= 0` is false).
    if (!(dt > 0)) return;
    this.awakeCount = 0;
    this.contactCount = 0;
    this.ccdSweeps = 0;

    const gx = this.gravity.x * dt;
    const gy = this.gravity.y * dt;
    const gz = this.gravity.z * dt;

    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      if (!body.alive || body.sleeping) continue;
      this.awakeCount++;

      const v = body.velocity;
      const w = body.angularVelocity;
      const p = body.position;
      const q = body.quaternion;

      // ── integrate velocity ────────────────────────────────────────────────
      v.x += gx; v.y += gy; v.z += gz;

      // Implicit damping (v /= 1 + k·dt) instead of the explicit (1 - k·dt).
      // The explicit form flips sign and explodes when k·dt > 1; this one is
      // unconditionally stable for any damping coefficient or step size.
      const ld = 1 / (1 + body.linearDamping * dt);
      v.x *= ld; v.y *= ld; v.z *= ld;
      const ad = 1 / (1 + body.angularDamping * dt);
      w.x *= ad; w.y *= ad; w.z *= ad;

      this.clampBody(body);

      // ── integrate position and orientation ────────────────────────────────
      const prevX = p.x, prevY = p.y, prevZ = p.z;
      p.x += v.x * dt; p.y += v.y * dt; p.z += v.z * dt;

      const hx = w.x * 0.5 * dt, hy = w.y * 0.5 * dt, hz = w.z * 0.5 * dt;
      const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
      let nqx = qx + (hx * qw + hy * qz - hz * qy);
      let nqy = qy + (hy * qw + hz * qx - hx * qz);
      let nqz = qz + (hz * qw + hx * qy - hy * qx);
      let nqw = qw + (-hx * qx - hy * qy - hz * qz);
      const ql = Math.sqrt(nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw);
      if (ql > 1e-9) {
        const inv = 1 / ql;
        nqx *= inv; nqy *= inv; nqz *= inv; nqw *= inv;
      } else {
        nqx = 0; nqy = 0; nqz = 0; nqw = 1;
      }
      q.set(nqx, nqy, nqz, nqw);

      // ── continuous collision, before the discrete narrow phase ───────────
      this.sweepGuard(body, prevX, prevY, prevZ);

      // ── collide against the static world ─────────────────────────────────
      this.collideBody(body, dt);
      this.contactCount += this.nContacts;

      if (this.nContacts > 0) {
        this.solveVelocity(body);
        this.reportImpact(body);
        this.solvePosition(body);

        // Rolling resistance, only while supported so airborne spin is kept.
        this.applyRollingResistance(body);
        body.supported = true;
        body.airborneSteps = 0;
      } else {
        body.airborneSteps++;
        body.supported = body.airborneSteps <= CONTACT_PERSIST_STEPS;
      }

      // ── sleep ─────────────────────────────────────────────────────────────
      const lin2 = v.x * v.x + v.y * v.y + v.z * v.z;
      const ang2 = w.x * w.x + w.y * w.y + w.z * w.z;
      const angLimit = Math.min(SLEEP_ANGULAR_MAX, SLEEP_TANGENTIAL / Math.max(1e-4, body.boundingRadius));
      if (body.supported && lin2 < SLEEP_LINEAR * SLEEP_LINEAR && ang2 < angLimit * angLimit) {
        body.restTimer += dt;
        if (body.restTimer >= SLEEP_TIME) {
          body.sleeping = true;
          v.set(0, 0, 0);
          w.set(0, 0, 0);
        }
      } else {
        body.restTimer = 0;
      }
    }
  }

  /**
   * Push every live body's transform onto its `Object3D`.
   *
   * The solver works in WORLD space but `Object3D.position/quaternion` are
   * LOCAL to the parent, so a body parented to anything other than the scene
   * root must be converted or it renders at double its offset. The parent's
   * inverse is cached across the loop because bodies are typically added in
   * batches that share a parent (a casing pool, a debris group).
   */
  syncTransforms(): void {
    let cachedParent: THREE.Object3D | null = null;

    for (let i = 0; i < this.bodies.length; i++) {
      const body = this.bodies[i];
      const obj = body.object;
      if (!body.alive || !obj) continue;
      // Sleeping bodies do not move, so skip the write and the matrix dirty.
      if (body.sleeping) continue;

      const parent = obj.parent;
      if (parent === null) {
        obj.position.copy(body.position);
        obj.quaternion.copy(body.quaternion);
      } else {
        if (parent !== cachedParent) {
          cachedParent = parent;
          parent.updateWorldMatrix(true, false);
          _parentInv.copy(parent.matrixWorld).invert();
          // Decompose for the rotation rather than reading it off the inverse
          // matrix: `setFromRotationMatrix` assumes a pure rotation, so a
          // scaled parent fed it a skewed basis and produced up to 96 degrees
          // of orientation error plus spurious object scale.
          parent.matrixWorld.decompose(_pPos, _pQuat, _pScale);
          _parentInvQuat.copy(_pQuat).invert();
        }
        obj.position.copy(body.position).applyMatrix4(_parentInv);
        obj.quaternion.copy(_parentInvQuat).multiply(body.quaternion);
      }

      if (!obj.matrixAutoUpdate) {
        obj.updateMatrix();
        obj.matrixWorldNeedsUpdate = true;
      }
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Narrow phase
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Clamp this step's translation to the first swept-sphere impact when the
   * body moved far enough to skip past a surface. See `CCD_MOTION_RATIO` for
   * the failure this prevents.
   *
   * Each cluster sphere is swept independently from its previous world position
   * (using the new orientation for both endpoints — rotation over a single
   * 1/120 s step is small enough that the error is far below the slop band).
   */
  private sweepGuard(body: RigidBody, prevX: number, prevY: number, prevZ: number): void {
    const p = body.position;
    const mx = p.x - prevX, my = p.y - prevY, mz = p.z - prevZ;
    const dist = Math.sqrt(mx * mx + my * my + mz * mz);
    if (dist < 1e-9) return;

    // Only the smallest sphere can slip through, so it sets the threshold.
    let minR = Infinity;
    for (let s = 0; s < body.sphereCount; s++) {
      const r = body.sphereRadius[s];
      if (r < minR) minR = r;
    }
    if (dist <= minR * CCD_MOTION_RATIO) return;

    const inv = 1 / dist;
    const dx = mx * inv, dy = my * inv, dz = mz * inv;
    const q = body.quaternion;

    let earliest = dist;
    let hitR = minR;
    for (let s = 0; s < body.sphereCount; s++) {
      const r = body.sphereRadius[s];
      rotateByQuat(q.x, q.y, q.z, q.w, body.sphereLocal[s * 3], body.sphereLocal[s * 3 + 1], body.sphereLocal[s * 3 + 2]);
      if (this.world.sphereCast(prevX + _rx, prevY + _ry, prevZ + _rz, r, dx, dy, dz, earliest, this.sweep)) {
        if (this.sweep.t < earliest) {
          earliest = this.sweep.t;
          hitR = r;
        }
      }
    }

    if (earliest >= dist) return;

    // Land marginally PAST the touch point, not short of it. Stopping exactly
    // at (or before) first contact leaves the sphere non-overlapping, the
    // discrete narrow phase finds nothing, no impulse is applied, and the body
    // hovers while gravity accumulates velocity forever. Overshooting by a
    // fraction of the radius guarantees a contact this same step and is still
    // far too small to punch through the far side of any real surface.
    const t = Math.min(dist, earliest + hitR * CCD_OVERLAP_FRACTION);
    p.set(prevX + dx * t, prevY + dy * t, prevZ + dz * t);
    this.ccdSweeps++;
  }

  private collideBody(body: RigidBody, dt: number): void {
    this.nContacts = 0;

    // One traversal for the whole cluster: bounds of the body plus the distance
    // it just travelled, so a fast casing cannot skip past a thin floor.
    const r = body.boundingRadius + BROADPHASE_MARGIN;
    const sx = Math.abs(body.velocity.x) * dt;
    const sy = Math.abs(body.velocity.y) * dt;
    const sz = Math.abs(body.velocity.z) * dt;
    const p = body.position;

    // Cluster sphere world centres are computed ONCE, outside the traversal,
    // so the per-triangle visitor is a tight loop over `sphereCount`.
    const q = body.quaternion;
    for (let s = 0; s < body.sphereCount; s++) {
      rotateByQuat(q.x, q.y, q.z, q.w, body.sphereLocal[s * 3], body.sphereLocal[s * 3 + 1], body.sphereLocal[s * 3 + 2]);
      this.sphereWorld[s * 3] = p.x + _rx;
      this.sphereWorld[s * 3 + 1] = p.y + _ry;
      this.sphereWorld[s * 3 + 2] = p.z + _rz;
    }
    this.visitBody = body;

    this.world.traverseBox(
      p.x - r - sx, p.y - r - sy, p.z - r - sz,
      p.x + r + sx, p.y + r + sy, p.z + r + sz,
      this,
    );

    this.visitBody = null;
  }

  /**
   * `TriangleVisitor`. Called for every triangle overlapping the body's swept
   * bounds; there is deliberately no candidate-triangle budget (see
   * `TriangleVisitor` in BVHWorld for why a budget causes bodies to fall
   * through densely tessellated floors).
   */
  visitTriangle(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
    surface: SurfaceType,
  ): void {
    const body = this.visitBody;
    if (!body) return;
    this.visitSurface = surface;

    for (let s = 0; s < body.sphereCount; s++) {
      const sr = body.sphereRadius[s];
      const wx = this.sphereWorld[s * 3];
      const wy = this.sphereWorld[s * 3 + 1];
      const wz = this.sphereWorld[s * 3 + 2];

      // Cheap sphere-AABB reject before the closest-point solve.
      if (wx + sr < Math.min(ax, bx, cx) || wx - sr > Math.max(ax, bx, cx)) continue;
      if (wy + sr < Math.min(ay, by, cy) || wy - sr > Math.max(ay, by, cy)) continue;
      if (wz + sr < Math.min(az, bz, cz) || wz - sr > Math.max(az, bz, cz)) continue;

      closestPointOnTriangle(wx, wy, wz, ax, ay, az, bx, by, bz, cx, cy, cz);
      const qx = lastContactX(), qy = lastContactY(), qz = lastContactZ();
      let dx = wx - qx, dy = wy - qy, dz = wz - qz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= sr * sr) continue;

      const d = Math.sqrt(d2);
      if (d > 1e-7) {
        const inv = 1 / d;
        dx *= inv; dy *= inv; dz *= inv;
      } else {
        // Centre coincident with the surface: fall back to the face normal so
        // the body is pushed out along a sane axis instead of NaN.
        const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        dx = e1y * e2z - e1z * e2y;
        dy = e1z * e2x - e1x * e2z;
        dz = e1x * e2y - e1y * e2x;
        const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (l < 1e-12) continue;
        const inv = 1 / l;
        dx *= inv; dy *= inv; dz *= inv;
      }

      this.pushContact(dx, dy, dz, qx, qy, qz, sr - d, s, ax, ay, az, bx, by, bz, cx, cy, cz);
    }
  }

  /** Keep the `MAX_CONTACTS` deepest contacts; deeper ones matter more. */
  private pushContact(
    nx: number, ny: number, nz: number,
    px: number, py: number, pz: number,
    depth: number, sphere: number,
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
  ): void {
    let slot = this.nContacts;
    if (slot >= MAX_CONTACTS) {
      let shallowest = 0;
      for (let i = 1; i < MAX_CONTACTS; i++) {
        if (this.cDepth[i] < this.cDepth[shallowest]) shallowest = i;
      }
      if (this.cDepth[shallowest] >= depth) return;
      slot = shallowest;
    } else {
      this.nContacts++;
    }
    this.cNx[slot] = nx; this.cNy[slot] = ny; this.cNz[slot] = nz;
    this.cPx[slot] = px; this.cPy[slot] = py; this.cPz[slot] = pz;
    this.cDepth[slot] = depth;
    this.cAccum[slot] = 0;
    this.cSphere[slot] = sphere;
    this.cSurface[slot] = this.visitSurface;
    // The triangle is copied into the contact rather than referenced by index:
    // the visitor streams triangles and nothing outlives the traversal, so a
    // stored index would dangle by the time the position solver re-evaluates.
    const o = slot * 9;
    const tv = this.cTriVerts;
    tv[o] = ax; tv[o + 1] = ay; tv[o + 2] = az;
    tv[o + 3] = bx; tv[o + 4] = by; tv[o + 5] = bz;
    tv[o + 6] = cx; tv[o + 7] = cy; tv[o + 8] = cz;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Contact solving
  // ───────────────────────────────────────────────────────────────────────────

  private solveVelocity(body: RigidBody): void {
    const v = body.velocity;
    const w = body.angularVelocity;
    const p = body.position;
    const im = body.invMass;
    const ii = body.invInertia;
    const n = this.nContacts;

    // Restitution must be baked into a FIXED target velocity before iterating.
    //
    // Recomputing `-(1+e)·vn` inside the loop silently destroys bouncing: after
    // the first iteration `vn` is positive (separating), the threshold test
    // then yields e=0, and the resulting negative impulse is subtracted from
    // the accumulator — cancelling exactly the bounce the first iteration
    // added. With two iterations restitution became a total no-op at every
    // value including 1.0, so grenades stuck to walls instead of bouncing.
    for (let i = 0; i < n; i++) {
      const nx = this.cNx[i], ny = this.cNy[i], nz = this.cNz[i];
      const rx = this.cPx[i] - p.x, ry = this.cPy[i] - p.y, rz = this.cPz[i] - p.z;
      const vpx = v.x + (w.y * rz - w.z * ry);
      const vpy = v.y + (w.z * rx - w.x * rz);
      const vpz = v.z + (w.x * ry - w.y * rx);
      const vn0 = vpx * nx + vpy * ny + vpz * nz;
      // Suppressed below the threshold so resting bodies settle instead of
      // grinding out endless millimetre bounces.
      this.cBias[i] = -vn0 > RESTITUTION_THRESHOLD ? body.restitution * -vn0 : 0;
    }

    for (let it = 0; it < VELOCITY_ITERATIONS; it++) {
      for (let i = 0; i < n; i++) {
        const nx = this.cNx[i], ny = this.cNy[i], nz = this.cNz[i];
        const rx = this.cPx[i] - p.x, ry = this.cPy[i] - p.y, rz = this.cPz[i] - p.z;

        // Velocity of the material point at the contact: v + ω × r.
        let vpx = v.x + (w.y * rz - w.z * ry);
        let vpy = v.y + (w.z * rx - w.x * rz);
        let vpz = v.z + (w.x * ry - w.y * rx);
        const vn = vpx * nx + vpy * ny + vpz * nz;

        // Effective mass along n: 1/m + (r×n)·I⁻¹(r×n), scalar I⁻¹.
        const rnx = ry * nz - rz * ny;
        const rny = rz * nx - rx * nz;
        const rnz = rx * ny - ry * nx;
        const invEff = im + (rnx * rnx + rny * rny + rnz * rnz) * ii;
        if (invEff <= 0) continue;

        let j = -(vn - this.cBias[i]) / invEff;

        // Accumulate and clamp so the total normal impulse is never negative
        // (contacts push, never pull).
        const old = this.cAccum[i];
        let next = old + j;
        if (next < 0) next = 0;
        j = next - old;
        this.cAccum[i] = next;

        if (j !== 0) {
          v.x += nx * j * im; v.y += ny * j * im; v.z += nz * j * im;
          w.x += rnx * j * ii; w.y += rny * j * ii; w.z += rnz * j * ii;
        }

        // ── friction ────────────────────────────────────────────────────────
        vpx = v.x + (w.y * rz - w.z * ry);
        vpy = v.y + (w.z * rx - w.x * rz);
        vpz = v.z + (w.x * ry - w.y * rx);
        const vn2 = vpx * nx + vpy * ny + vpz * nz;
        let tx = vpx - nx * vn2;
        let ty = vpy - ny * vn2;
        let tz = vpz - nz * vn2;
        const tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (tl < 1e-6) continue;
        const invTl = 1 / tl;
        tx *= invTl; ty *= invTl; tz *= invTl;

        const rtx = ry * tz - rz * ty;
        const rty = rz * tx - rx * tz;
        const rtz = rx * ty - ry * tx;
        const invEffT = im + (rtx * rtx + rty * rty + rtz * rtz) * ii;
        if (invEffT <= 0) continue;

        let jt = -tl / invEffT;
        // Coulomb cone against the ACCUMULATED normal impulse.
        const maxT = body.friction * this.cAccum[i];
        if (jt < -maxT) jt = -maxT;

        v.x += tx * jt * im; v.y += ty * jt * im; v.z += tz * jt * im;
        // This torque is what makes drums and casings ROLL rather than skid.
        w.x += rtx * jt * ii; w.y += rty * jt * ii; w.z += rtz * jt * ii;
      }
    }
    this.clampBody(body);
  }

  /**
   * Bleed spin in proportion to how hard the body is pressed into the surface.
   * Clamped so it can only ever reduce the magnitude, never reverse the spin.
   */
  private applyRollingResistance(body: RigidBody): void {
    const w = body.angularVelocity;
    const wLen = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    if (wLen < 1e-6) return;

    let jn = 0;
    for (let i = 0; i < this.nContacts; i++) jn += this.cAccum[i];
    if (jn <= 0) return;

    const dw = ROLLING_RESISTANCE * jn * body.boundingRadius * body.invInertia;
    const scale = dw >= wLen ? 0 : 1 - dw / wLen;
    w.x *= scale; w.y *= scale; w.z *= scale;
  }

  /** Emit at most one impact per body per step: the largest normal impulse. */
  private reportImpact(body: RigidBody): void {
    const cb = this.onImpact;
    if (!cb) return;
    let best = 0;
    let bestI = -1;
    for (let i = 0; i < this.nContacts; i++) {
      if (this.cAccum[i] > best) { best = this.cAccum[i]; bestI = i; }
    }
    // Resting contacts carry a steady gravity-support impulse; the threshold
    // scales with mass so a crate at rest is as silent as a casing at rest.
    if (bestI < 0 || best < IMPACT_REPORT_IMPULSE * body.mass) return;
    this.impactPoint.set(this.cPx[bestI], this.cPy[bestI], this.cPz[bestI]);
    this.impactNormal.set(this.cNx[bestI], this.cNy[bestI], this.cNz[bestI]);
    cb(body.handle, this.impactPoint, this.impactNormal, best, this.cSurface[bestI]);
  }

  private solvePosition(body: RigidBody): void {
    const p = body.position;
    const q = body.quaternion;
    const tris = this.cTriVerts;
    const n = this.nContacts;

    // Slop scaled to the body: a fixed 8 mm would let a 5 mm shell casing rest
    // fully inside the floor.
    const slop = Math.min(PENETRATION_SLOP, body.boundingRadius * SLOP_RADIUS_FRACTION);

    for (let it = 0; it < POSITION_ITERATIONS; it++) {
      let moved = false;
      for (let i = 0; i < n; i++) {
        const s = this.cSphere[i];
        const sr = body.sphereRadius[s];

        rotateByQuat(q.x, q.y, q.z, q.w, body.sphereLocal[s * 3], body.sphereLocal[s * 3 + 1], body.sphereLocal[s * 3 + 2]);
        const wx = p.x + _rx, wy = p.y + _ry, wz = p.z + _rz;

        const o = i * 9;
        closestPointOnTriangle(
          wx, wy, wz,
          tris[o], tris[o + 1], tris[o + 2],
          tris[o + 3], tris[o + 4], tris[o + 5],
          tris[o + 6], tris[o + 7], tris[o + 8],
        );
        const dx = wx - lastContactX(), dy = wy - lastContactY(), dz = wz - lastContactZ();
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const depth = sr - d;
        if (depth <= slop) continue;

        let nx = this.cNx[i], ny = this.cNy[i], nz = this.cNz[i];
        if (d > 1e-7) {
          const inv = 1 / d;
          nx = dx * inv; ny = dy * inv; nz = dz * inv;
        }

        const push = (depth - slop) * BAUMGARTE;
        p.x += nx * push; p.y += ny * push; p.z += nz * push;
        moved = true;
      }
      // Converged early — the common case for a body resting on one plane.
      if (!moved) break;
    }
  }
}
