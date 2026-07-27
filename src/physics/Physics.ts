/**
 * NOISEFIELD — Physics system.
 *
 * Two collision worlds behind one facade:
 *
 *   1. STATIC — merged, world-space triangle soup bucketed by `SurfaceType` and
 *      indexed with `three-mesh-bvh`. Exact to the triangle, and a hit *is* its
 *      material (see `BVHWorld`).
 *   2. DYNAMIC — a few dozen analytic colliders (sphere / OBB / capsule) for
 *      entity hitboxes and shootable ragdoll segments, plus the rigid-body
 *      solver for casings, debris, grenades and props (see `RigidBody`).
 *
 * No Rapier. An FPS needs pixel-exact hitscan against arbitrary triangles with
 * per-triangle material identification, and a handful of extremely simple
 * dynamics. A general WASM solver costs load time, gives worse raycast
 * ergonomics, and none of its constraint machinery would ever be used.
 *
 * ── ALLOCATION CONTRACT ──────────────────────────────────────────────────────
 * This subsystem allocates no results and no scratch per query. `raycast` hands
 * back a hit drawn from a 128-entry ring buffer, so a returned hit stays valid
 * for the next 127 casts — far longer than any frame-local use — but must be
 * copied if it is to outlive that. `raycastAll` returns one of 4 rotating
 * arrays, each with its own hit objects, so its results survive 3 further
 * `raycastAll` calls and are never disturbed by `raycast`.
 *
 * Residual, and outside our control: three-mesh-bvh's `BufferStack.setBuffer`
 * allocates three TypedArray views per BVH traversal. Buckets whose cached
 * root bounds the query misses are skipped entirely, which removes most of it,
 * but a query that touches N surface buckets still costs 3N small views. The
 * alternative is reimplementing the library's traversal against its private
 * node buffers, which is not a trade worth making.
 *
 * ── NORMAL ORIENTATION ───────────────────────────────────────────────────────
 * `raycast()`     → normal is flipped to oppose the incoming ray (always faces
 *                   the shooter, which is what decals and impact VFX want).
 * `raycastAll()`  → normal is the RAW geometric normal, and each hit carries
 *                   `backface`. This is deliberate: an *exit* wound's decal must
 *                   face away from the wall, so the unflipped normal is correct
 *                   for both entry and exit surfaces, and `backface` /
 *                   `normal · dir > 0` tells penetration code which is which.
 */

import * as THREE from 'three';
import type {
  DynamicBodyOptions,
  EngineContext,
  EntityId,
  FrameContext,
  IPhysics,
  PhysicsFilter,
  RaycastHit,
  SurfaceType,
} from '../core/Types';
import {
  BVHWorld,
  createRayResult,
  lastNormalX,
  lastNormalY,
  lastNormalZ,
  rayCapsule,
  raySphere,
  type RayResult,
} from './BVHWorld';
import {
  HANDLE_INDEX_MASK,
  HANDLE_TAG_COLLIDER,
  RigidBodySolver,
  handleTag,
  packHandle,
  type RigidBody,
} from './RigidBody';

// ─────────────────────────────────────────────────────────────────────────────
// Public extras (beyond IPhysics)
// ─────────────────────────────────────────────────────────────────────────────

export type EntityColliderShape = 'sphere' | 'box' | 'capsule';

export interface EntityColliderOptions {
  /** Defaults to 'capsule' — what almost every humanoid hitbox wants. */
  shape?: EntityColliderShape;
  /** Local radius. Derived from the object's bounds when omitted. */
  radius?: number;
  /** Total capsule height, tip to tip. Derived from bounds when omitted. */
  height?: number;
  /** Local box half-extents. Derived from bounds when omitted. */
  halfExtents?: THREE.Vector3;
  /** Local-space offset of the collider centre from the object origin. */
  offset?: THREE.Vector3;
  /** Falls back to `object.userData.entityId`. */
  entity?: EntityId;
  /** Falls back to `object.userData.hitbox`; `'head'` sets `headshot`. */
  hitbox?: string;
  /** Falls back to `'flesh'`. */
  surface?: SurfaceType;
}

/** A `RaycastHit` from `raycastAll`, carrying entry/exit facing. */
export interface PenetrationHit extends RaycastHit {
  /** True when this is an EXIT surface (ray struck the back face). */
  backface: boolean;
}

export interface PhysicsStats {
  staticTriangles: number;
  surfaceBuckets: number;
  entityColliders: number;
  bodies: number;
  awakeBodies: number;
  contacts: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────────────

const SHAPE_SPHERE = 0;
const SHAPE_BOX = 1;
const SHAPE_CAPSULE = 2;

/** Hits stay valid for this many subsequent `raycast` calls. */
const HIT_RING = 128;
/**
 * Rotating result arrays for `raycastAll`.
 *
 * These have their OWN hit objects rather than drawing from `hitRing`. Sharing
 * them meant a single `raycastAll` could consume 32+ ring slots, so a shotgun
 * loop that interleaved the two silently rewrote pellet hits it was still
 * holding (pellet[0] and pellet[4] became the same object).
 */
const LIST_RING = 4;

const HANDLE_GEN_MASK = 0xfff;

/**
 * Safety bound on the `PhysicsFilter.exclude` ancestor walk. Generous enough to
 * reach the root of any real rig — at 8 a deep skeleton's bones never matched
 * the excluded player root, so the shooter could hit their own hitboxes.
 */
const EXCLUDE_PARENT_DEPTH = 64;

/**
 * Head hitboxes almost always sit inside the torso capsule of a humanoid rig,
 * so a ray through the skull frequently strikes the (larger, nearer) torso
 * first and a legitimate headshot registers as a body shot. When a head and a
 * non-head box of the SAME entity are both struck within this distance of each
 * other, the head wins. Set `headshotPriorityMargin = 0` for strict
 * nearest-hit-wins behaviour.
 */
const DEFAULT_HEADSHOT_PRIORITY_MARGIN = 0.25;

/**
 * Fall-back collider sizes for hitbox nodes that carry no geometry (bones and
 * empty Object3Ds, which is the normal case for a skinned rig). Human scale.
 */
const DEFAULT_HEAD_RADIUS = 0.12;
const DEFAULT_LIMB_RADIUS = 0.1;
const DEFAULT_BODY_RADIUS = 0.25;
const DEFAULT_BODY_HALF_HEIGHT = 0.85;

const DEBUG_MAX_VERTS = 48000;
const DEBUG_BVH_DEPTH = 6;

interface EntityCollider {
  handle: number;
  generation: number;
  alive: boolean;
  object: THREE.Object3D;
  shape: number;

  // Local definition
  readonly offset: THREE.Vector3;
  radius: number;
  /** Half of the capsule's inner segment (excludes the caps). */
  halfSegment: number;
  readonly half: THREE.Vector3;

  entity: EntityId | undefined;
  headshot: boolean;
  surface: SurfaceType;

  // World cache, refreshed once per simulation step
  readonly center: THREE.Vector3;
  readonly segA: THREE.Vector3;
  readonly segB: THREE.Vector3;
  worldRadius: number;
  readonly matrix: THREE.Matrix4;
  readonly invMatrix: THREE.Matrix4;
  aMinX: number; aMinY: number; aMinZ: number;
  aMaxX: number; aMaxY: number; aMaxZ: number;
}

function makeHit(): PenetrationHit {
  return {
    point: new THREE.Vector3(),
    normal: new THREE.Vector3(0, 1, 0),
    distance: 0,
    surface: 'concrete',
    entity: undefined,
    headshot: false,
    object: undefined,
    backface: false,
  };
}

// Scratch — module scope, single-threaded, never re-entered.
const _decompPos = new THREE.Vector3();
const _decompQuat = new THREE.Quaternion();
const _decompScale = new THREE.Vector3();
const _localBounds = new THREE.Box3();
const _boundsSize = new THREE.Vector3();
const _boundsCenter = new THREE.Vector3();
const _childBox = new THREE.Box3();
const _relMat = new THREE.Matrix4();
const _invRoot = new THREE.Matrix4();

/** Union of descendant geometry bounds in `root`'s local frame. */
function computeLocalBounds(root: THREE.Object3D, target: THREE.Box3): boolean {
  target.makeEmpty();
  root.updateWorldMatrix(true, true);
  _invRoot.copy(root.matrixWorld).invert();
  let found = false;
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    const bb = mesh.geometry.boundingBox;
    if (!bb) return;
    _childBox.copy(bb);
    _relMat.multiplyMatrices(_invRoot, mesh.matrixWorld);
    _childBox.applyMatrix4(_relMat);
    target.union(_childBox);
    found = true;
  });
  return found;
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics
// ─────────────────────────────────────────────────────────────────────────────

export class Physics implements IPhysics {
  readonly name = 'physics';

  /** Static triangle world. Exposed for tools; prefer the query methods. */
  readonly world = new BVHWorld();
  /** Dynamic body solver. Exposed for tuning gravity and inspecting bodies. */
  readonly solver = new RigidBodySolver(this.world);

  private ctx: EngineContext | null = null;
  private unsubExplosion: (() => void) | null = null;

  // ── entity colliders ──────────────────────────────────────────────────────
  private readonly colliders: EntityCollider[] = [];
  private readonly freeColliders: number[] = [];
  private readonly bodyColliders = new Map<number, number>();
  private collidersDirty = true;
  private liveColliders = 0;

  // ── result pools ──────────────────────────────────────────────────────────
  private readonly hitRing: PenetrationHit[] = [];
  private ringIndex = 0;
  private readonly listRing: PenetrationHit[][] = [];
  /** Hit objects owned by each `listRing` slot; grown on demand, never shared. */
  private readonly listStore: PenetrationHit[][] = [];
  private listIndex = 0;

  private readonly staticResult: RayResult = createRayResult();
  private readonly sweepResult: RayResult = createRayResult();

  /** See `DEFAULT_HEADSHOT_PRIORITY_MARGIN`. Metres; 0 disables. */
  headshotPriorityMargin = DEFAULT_HEADSHOT_PRIORITY_MARGIN;

  /**
   * Called when a dynamic body strikes static geometry hard enough to matter.
   *
   * Without this a grenade cannot detonate on impact and a casing cannot play
   * its bounce sound — the body's own `surface` describes the casing, not the
   * floor it just hit, and polling positions cannot recover the impact speed.
   * `impulse` is the normal impulse magnitude, `surface` is what was struck.
   * Assign a single persistent handler; it is invoked inside `fixedUpdate`.
   */
  onBodyImpact: ((
    handle: number,
    point: THREE.Vector3,
    normal: THREE.Vector3,
    impulse: number,
    surface: SurfaceType,
  ) => void) | null = null;

  // ── debug ─────────────────────────────────────────────────────────────────
  private debugEnabled = false;
  private debugShowBVH = false;
  private debugObject: THREE.LineSegments | null = null;
  private debugPositions: Float32Array | null = null;
  private debugColors: Float32Array | null = null;
  private debugVerts = 0;
  private debugBVHDirty = true;

  constructor() {
    for (let i = 0; i < HIT_RING; i++) this.hitRing.push(makeHit());
    for (let i = 0; i < LIST_RING; i++) { this.listRing.push([]); this.listStore.push([]); }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ───────────────────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    // Debug may have been enabled before there was a scene to attach to.
    if (this.debugObject && !this.debugObject.parent) ctx.scene.add(this.debugObject);
    // Nobody else can move rigid bodies, so owning the blast response here
    // keeps explosions correct regardless of which system raised the event.
    this.unsubExplosion = ctx.bus.on('explosion', (e) => {
      this.applyRadialImpulse(e.position, e.radius, e.power);
    });
  }

  fixedUpdate(_ctx: EngineContext, step: number): void {
    // Entity transforms are refreshed once per step; the first query in the
    // step pays for it and the rest are free.
    this.collidersDirty = true;
    this.syncEntityColliders();
    this.solver.onImpact = this.onBodyImpact;
    this.solver.step(step);
    this.solver.syncTransforms();
  }

  update(_ctx: EngineContext, _frame: FrameContext): void {
    this.collidersDirty = true;
    if (this.debugEnabled) this.rebuildDebug();
  }

  dispose(): void {
    this.unsubExplosion?.();
    this.unsubExplosion = null;
    this.setDebug(false);
    this.clearDynamic();
    this.world.dispose();
    this.colliders.length = 0;
    this.freeColliders.length = 0;
    this.bodyColliders.clear();
    this.liveColliders = 0;
  }

  get stats(): PhysicsStats {
    const s = this.solver.stats;
    return {
      staticTriangles: this.world.triangleCount,
      surfaceBuckets: this.world.bucketCount,
      entityColliders: this.liveColliders,
      bodies: s.bodies,
      awakeBodies: s.awake,
      contacts: s.contacts,
    };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Static registration
  // ───────────────────────────────────────────────────────────────────────────

  addStatic(object: THREE.Object3D, surface: SurfaceType): void {
    this.world.addStatic(object, surface);
    // Geometry may have appeared underneath sleeping debris.
    this.solver.wakeAll();
    this.debugBVHDirty = true;
  }

  /** Drop all static geometry (level unload). */
  clearStatic(): void {
    this.world.clear();
    // Anything sleeping was resting on geometry that no longer exists; without
    // a wake it hangs frozen in mid-air forever.
    this.solver.wakeAll();
    this.debugBVHDirty = true;
  }

  /**
   * Remove every dynamic body AND the entity colliders paired with them.
   *
   * Prefer this over `solver.clear()`: the solver does not know about the
   * colliders `addDynamic` creates for ragdoll segments, so clearing it
   * directly strands them as unkillable ghost hitboxes.
   */
  clearDynamic(): void {
    for (const ch of this.bodyColliders.values()) this.removeEntityCollider(ch);
    this.bodyColliders.clear();
    this.solver.clear();
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Dynamic bodies
  // ───────────────────────────────────────────────────────────────────────────

  addDynamic(object: THREE.Object3D, opts: DynamicBodyOptions): number {
    const handle = this.solver.add(object, opts);
    const body = this.solver.get(handle);

    // Ragdoll segments carry an entityId and must be shootable. Plain debris
    // does not, and stays out of the raycast broadphase entirely — 200 shell
    // casings should never cost anything on an AI line-of-sight query.
    if (body && body.entity !== undefined) {
      const ch = this.addEntityCollider(object, {
        shape: 'sphere',
        radius: body.boundingRadius,
        entity: body.entity,
        surface: opts.surface ?? 'flesh',
      });
      this.bodyColliders.set(handle, ch);
    }
    return handle;
  }

  removeBody(handle: number): void {
    const ch = this.bodyColliders.get(handle);
    if (ch !== undefined) {
      this.removeEntityCollider(ch);
      this.bodyColliders.delete(handle);
    }
    this.solver.remove(handle);
  }

  applyImpulse(handle: number, impulse: THREE.Vector3, point?: THREE.Vector3): void {
    if (point) {
      this.solver.applyImpulse(handle, impulse.x, impulse.y, impulse.z, point.x, point.y, point.z);
    } else {
      this.solver.applyImpulse(handle, impulse.x, impulse.y, impulse.z);
    }
  }

  applyRadialImpulse(center: THREE.Vector3, radius: number, power: number): void {
    this.solver.applyRadialImpulse(center.x, center.y, center.z, radius, power);
  }

  /** Direct body access for tuning, inspection or manual teleports. */
  getBody(handle: number): RigidBody | undefined {
    return this.solver.get(handle);
  }

  /**
   * Teleport a body and wake it. Writing `body.position` directly on a sleeping
   * body has no visible effect until something else wakes it, which reads as
   * "the physics ignored me".
   */
  teleportBody(handle: number, position: THREE.Vector3, quaternion?: THREE.Quaternion): void {
    const body = this.solver.get(handle);
    if (!body) return;
    body.position.copy(position);
    if (quaternion) body.quaternion.copy(quaternion);
    body.velocity.set(0, 0, 0);
    body.angularVelocity.set(0, 0, 0);
    body.wake();
  }

  /**
   * Push a sphere out of any static geometry it is embedded in, writing the
   * corrected centre to `out`.
   *
   * A character controller needs this and a sweep cannot substitute: once a
   * shape is already inside a surface every sweep reports distance 0 with no
   * direction to escape. Slopes and corners embed the player capsule routinely,
   * so call this after integrating the player each step.
   *
   * @returns true when a correction was applied.
   */
  depenetrateSphere(center: THREE.Vector3, radius: number, out: THREE.Vector3): boolean {
    // A non-finite centre or radius produces a query box of NaN/Infinity, which
    // every bounds test accepts — the whole level is then walked triangle by
    // triangle (measured 757x slower) only to write NaN into `out`.
    if (
      !Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z)
      || !Number.isFinite(radius) || radius <= 0
    ) {
      out.copy(center);
      return false;
    }
    return this.world.depenetrateSphere(center.x, center.y, center.z, radius, out);
  }

  setGravity(x: number, y: number, z: number): void {
    this.solver.gravity.set(x, y, z);
    this.solver.wakeAll();
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Entity colliders
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Register a hittable collider that follows `object`. Dimensions are derived
   * from the object's geometry bounds unless supplied.
   */
  addEntityCollider(object: THREE.Object3D, opts: EntityColliderOptions = {}): number {
    const idx = this.freeColliders.length > 0 ? this.freeColliders.pop()! : this.colliders.length;

    let c: EntityCollider;
    if (idx < this.colliders.length) {
      c = this.colliders[idx];
      c.generation = (c.generation + 1) & HANDLE_GEN_MASK;
      if (c.generation === 0) c.generation = 1;
    } else {
      c = {
        handle: 0, generation: 1, alive: false,
        object, shape: SHAPE_CAPSULE,
        offset: new THREE.Vector3(),
        radius: 0.3, halfSegment: 0.5,
        half: new THREE.Vector3(0.3, 0.9, 0.3),
        entity: undefined, headshot: false, surface: 'flesh',
        center: new THREE.Vector3(),
        segA: new THREE.Vector3(), segB: new THREE.Vector3(),
        worldRadius: 0.3,
        matrix: new THREE.Matrix4(), invMatrix: new THREE.Matrix4(),
        aMinX: 0, aMinY: 0, aMinZ: 0, aMaxX: 0, aMaxY: 0, aMaxZ: 0,
      };
      this.colliders.push(c);
    }

    c.alive = true;
    c.object = object;
    c.handle = packHandle(HANDLE_TAG_COLLIDER, c.generation, idx);
    c.shape = opts.shape === 'box' ? SHAPE_BOX : opts.shape === 'sphere' ? SHAPE_SPHERE : SHAPE_CAPSULE;

    const hb = opts.hitbox ?? object.userData['hitbox'];
    c.headshot = hb === 'head';

    // Dimensions. Bounds come from geometry when there is any; a rig's head is
    // very often an empty bone node with none, and the old fall-back produced a
    // 0.9 m "head" sphere — everything within 90 cm scored as a headshot.
    const hasBounds = computeLocalBounds(object, _localBounds) && !_localBounds.isEmpty();
    let hx: number, hy: number, hz: number;
    let cx = 0, cy = 0, cz = 0;

    if (hasBounds) {
      _localBounds.getSize(_boundsSize);
      _localBounds.getCenter(_boundsCenter);
      hx = Math.max(1e-3, _boundsSize.x * 0.5);
      hy = Math.max(1e-3, _boundsSize.y * 0.5);
      hz = Math.max(1e-3, _boundsSize.z * 0.5);
      cx = _boundsCenter.x; cy = _boundsCenter.y; cz = _boundsCenter.z;
    } else if (c.headshot) {
      hx = hy = hz = DEFAULT_HEAD_RADIUS;
    } else if (c.shape === SHAPE_SPHERE) {
      hx = hy = hz = DEFAULT_LIMB_RADIUS;
    } else {
      hx = hz = DEFAULT_BODY_RADIUS;
      hy = DEFAULT_BODY_HALF_HEIGHT;
    }

    if (opts.halfExtents) {
      hx = Math.abs(opts.halfExtents.x);
      hy = Math.abs(opts.halfExtents.y);
      hz = Math.abs(opts.halfExtents.z);
    }
    c.half.set(hx, hy, hz);

    if (opts.radius !== undefined) {
      c.radius = Math.abs(opts.radius);
    } else if (c.shape === SHAPE_SPHERE) {
      // A sphere hugging the bounds uses the largest half-extent; with no
      // bounds the defaults above are already the radius.
      c.radius = hasBounds ? Math.max(hx, hy, hz) : hx;
    } else {
      c.radius = Math.max(hx, hz);
    }

    if (opts.height !== undefined) {
      c.halfSegment = Math.max(0, opts.height * 0.5 - c.radius);
    } else {
      c.halfSegment = Math.max(0, hy - c.radius);
    }

    if (opts.offset) c.offset.copy(opts.offset);
    else c.offset.set(cx, cy, cz);

    const eid = opts.entity ?? object.userData['entityId'];
    c.entity = typeof eid === 'number' ? eid : undefined;

    c.surface = opts.surface ?? 'flesh';

    this.liveColliders++;
    this.collidersDirty = true;
    this.syncCollider(c);
    return c.handle;
  }

  removeEntityCollider(handle: number): void {
    const c = this.getCollider(handle);
    if (!c) return;
    c.alive = false;
    c.handle = 0;
    this.freeColliders.push(handle & HANDLE_INDEX_MASK);
    this.liveColliders--;
  }

  /**
   * Convenience: register a collider for every descendant tagged with
   * `userData.hitbox`. `'head'` becomes a sphere, everything else a capsule.
   * Returns the created handles appended to `out`.
   */
  addEntity(root: THREE.Object3D, entityId?: EntityId, out?: number[]): number[] {
    const handles = out ?? [];
    const eid = entityId ?? (typeof root.userData['entityId'] === 'number'
      ? (root.userData['entityId'] as EntityId)
      : undefined);
    root.traverse((child) => {
      const hb = child.userData['hitbox'];
      if (typeof hb !== 'string') return;
      handles.push(this.addEntityCollider(child, {
        shape: hb === 'head' ? 'sphere' : 'capsule',
        entity: eid,
        hitbox: hb,
        surface: 'flesh',
      }));
    });
    if (handles.length === 0) {
      // No tagged hitboxes: fall back to one capsule around the whole entity.
      handles.push(this.addEntityCollider(root, { shape: 'capsule', entity: eid, surface: 'flesh' }));
    }
    return handles;
  }

  private getCollider(handle: number): EntityCollider | undefined {
    // The tag keeps collider handles and rigid-body handles in disjoint numeric
    // spaces. Both are plain `number` in the public API, and without the tag
    // they collide exactly (index 0 gen 1 in both spaces), so passing a
    // collider handle to `removeBody` silently destroyed an unrelated body.
    if (handle === 0 || handleTag(handle) !== HANDLE_TAG_COLLIDER) return undefined;
    const idx = handle & HANDLE_INDEX_MASK;
    if (idx >= this.colliders.length) return undefined;
    const c = this.colliders[idx];
    if (!c.alive || c.handle !== handle) return undefined;
    return c;
  }

  /**
   * Refresh cached world transforms. Automatic at the start of each simulation
   * step; call with `force` after moving an entity mid-step.
   */
  syncEntityColliders(force = false): void {
    if (!this.collidersDirty && !force) return;
    this.collidersDirty = false;
    for (let i = 0; i < this.colliders.length; i++) {
      const c = this.colliders[i];
      if (c.alive) this.syncCollider(c);
    }
  }

  private syncCollider(c: EntityCollider): void {
    const obj = c.object;
    obj.updateWorldMatrix(true, false);
    c.matrix.copy(obj.matrixWorld);
    c.invMatrix.copy(c.matrix).invert();
    c.matrix.decompose(_decompPos, _decompQuat, _decompScale);
    const scale = Math.max(Math.abs(_decompScale.x), Math.abs(_decompScale.y), Math.abs(_decompScale.z));

    // `userData` is authoritative and may change per frame (e.g. a crouching
    // AI shrinking its capsule), so it is re-read on every sync.
    const eid = obj.userData['entityId'];
    if (typeof eid === 'number') c.entity = eid;
    const hb = obj.userData['hitbox'];
    if (typeof hb === 'string') c.headshot = hb === 'head';

    c.center.copy(c.offset).applyMatrix4(c.matrix);
    c.worldRadius = c.radius * scale;

    const e = c.matrix.elements;

    if (c.shape === SHAPE_CAPSULE) {
      const hs = c.halfSegment;
      c.segA.set(c.offset.x, c.offset.y + hs, c.offset.z).applyMatrix4(c.matrix);
      c.segB.set(c.offset.x, c.offset.y - hs, c.offset.z).applyMatrix4(c.matrix);
      const r = c.worldRadius;
      c.aMinX = Math.min(c.segA.x, c.segB.x) - r;
      c.aMinY = Math.min(c.segA.y, c.segB.y) - r;
      c.aMinZ = Math.min(c.segA.z, c.segB.z) - r;
      c.aMaxX = Math.max(c.segA.x, c.segB.x) + r;
      c.aMaxY = Math.max(c.segA.y, c.segB.y) + r;
      c.aMaxZ = Math.max(c.segA.z, c.segB.z) + r;
    } else if (c.shape === SHAPE_SPHERE) {
      const r = c.worldRadius;
      c.aMinX = c.center.x - r; c.aMinY = c.center.y - r; c.aMinZ = c.center.z - r;
      c.aMaxX = c.center.x + r; c.aMaxY = c.center.y + r; c.aMaxZ = c.center.z + r;
    } else {
      // OBB world AABB: |M| · halfExtents (three stores column-major).
      const hx = c.half.x, hy = c.half.y, hz = c.half.z;
      const ex = Math.abs(e[0]) * hx + Math.abs(e[4]) * hy + Math.abs(e[8]) * hz;
      const ey = Math.abs(e[1]) * hx + Math.abs(e[5]) * hy + Math.abs(e[9]) * hz;
      const ez = Math.abs(e[2]) * hx + Math.abs(e[6]) * hy + Math.abs(e[10]) * hz;
      c.aMinX = c.center.x - ex; c.aMinY = c.center.y - ey; c.aMinZ = c.center.z - ez;
      c.aMaxX = c.center.x + ex; c.aMaxY = c.center.y + ey; c.aMaxZ = c.center.z + ez;
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Queries
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Nearest hit along `direction` (normalised internally).
   *
   * `normal` is oriented against the incoming ray, so impact VFX and decals can
   * use it directly. See the allocation contract in the file header for the
   * lifetime of the returned object.
   */
  raycast(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    maxDistance: number,
    filter?: PhysicsFilter,
  ): RaycastHit | null {
    this.syncEntityColliders();

    const dl = Math.sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    // Negated form: `NaN < 1e-9` is FALSE, so the naive guard lets a NaN
    // direction through, and the whole BVH is then walked with NaN bounds
    // tests (measured 1166x slower) only to return null anyway.
    if (!(dl > 1e-9) || !(maxDistance > 0)) return null;
    const inv = 1 / dl;
    const dx = direction.x * inv, dy = direction.y * inv, dz = direction.z * inv;
    const ox = origin.x, oy = origin.y, oz = origin.z;

    let best = maxDistance;
    let staticHit = false;

    if (!filter?.entitiesOnly) {
      const r = this.staticResult;
      if (this.world.raycastFirst(ox, oy, oz, dx, dy, dz, best, r)) {
        staticHit = true;
        best = r.t;
      }
    }

    const entityIdx = this.raycastEntities(ox, oy, oz, dx, dy, dz, best, filter);
    if (entityIdx >= 0) {
      const c = this.colliders[entityIdx];
      const hit = this.nextHit();
      hit.distance = this.entityT;
      hit.point.set(ox + dx * this.entityT, oy + dy * this.entityT, oz + dz * this.entityT);
      hit.normal.set(this.entityNx, this.entityNy, this.entityNz);
      if (hit.normal.x * dx + hit.normal.y * dy + hit.normal.z * dz > 0) hit.normal.negate();
      hit.surface = c.surface;
      hit.entity = c.entity;
      hit.headshot = c.headshot;
      hit.object = c.object;
      hit.backface = false;
      return hit;
    }

    if (!staticHit) return null;

    const r = this.staticResult;
    const hit = this.nextHit();
    hit.distance = r.t;
    hit.point.set(r.px, r.py, r.pz);
    hit.normal.set(r.nx, r.ny, r.nz);
    // Orient against the ray: a decal must never face into the wall.
    if (hit.normal.x * dx + hit.normal.y * dy + hit.normal.z * dz > 0) hit.normal.negate();
    hit.surface = r.surface;
    hit.entity = undefined;
    hit.headshot = false;
    hit.object = undefined;
    hit.backface = !r.frontFace;
    return hit;
  }

  /**
   * Every hit along the ray, near→far, INCLUDING back faces.
   *
   * Wall penetration reads this as an alternating entry/exit sequence: pair
   * hit[i] (entry, `backface === false`) with hit[i+1] (exit,
   * `backface === true`) and the material thickness is the distance between
   * them. Normals here are RAW (not flipped) so both the entry and the exit
   * decal orient correctly — see the file header.
   *
   * The returned array is pooled; copy anything that must outlive the next
   * few `raycastAll` calls.
   */
  raycastAll(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    maxDistance: number,
    filter?: PhysicsFilter,
  ): RaycastHit[] {
    return this.raycastPenetration(origin, direction, maxDistance, filter);
  }

  /**
   * Identical to `raycastAll` but statically typed as `PenetrationHit[]`, so
   * `hit.backface` is reachable without a cast. `IPhysics` fixes `raycastAll`'s
   * return type to `RaycastHit[]`, which does not carry `backface`; prefer this
   * method whenever you are writing penetration code.
   */
  raycastPenetration(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    maxDistance: number,
    filter?: PhysicsFilter,
  ): PenetrationHit[] {
    this.syncEntityColliders();

    const slot = this.listIndex;
    const list = this.listRing[slot];
    this.listIndex = (this.listIndex + 1) % LIST_RING;
    list.length = 0;
    // Cleared up-front so an entities-only query cannot report the previous
    // call's truncation state.
    this.world.penetrationTruncated = false;

    const dl = Math.sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    if (!(dl > 1e-9) || !(maxDistance > 0)) return list;
    const inv = 1 / dl;
    const dx = direction.x * inv, dy = direction.y * inv, dz = direction.z * inv;
    const ox = origin.x, oy = origin.y, oz = origin.z;

    if (!filter?.entitiesOnly) {
      const n = this.world.raycastAllRaw(ox, oy, oz, dx, dy, dz, maxDistance);
      for (let i = 0; i < n; i++) {
        const hit = this.listSlot(slot, list.length);
        hit.distance = this.world.hitT(i);
        hit.point.set(this.world.hitPX(i), this.world.hitPY(i), this.world.hitPZ(i));
        hit.normal.set(this.world.hitNX(i), this.world.hitNY(i), this.world.hitNZ(i));
        hit.surface = this.world.hitSurfaceAt(i);
        hit.entity = undefined;
        hit.headshot = false;
        hit.object = undefined;
        hit.backface = !this.world.hitFrontAt(i);
        list.push(hit);
      }
    }

    // Entity colliders are convex, so at most one entry per collider matters
    // for penetration; the analytic tests return the near surface.
    for (let i = 0; i < this.colliders.length; i++) {
      const c = this.colliders[i];
      if (!c.alive) continue;
      if (this.excluded(c, filter)) continue;
      const t = this.intersectCollider(c, ox, oy, oz, dx, dy, dz, maxDistance);
      if (t < 0) continue;
      const hit = this.listSlot(slot, list.length);
      hit.distance = t;
      hit.point.set(ox + dx * t, oy + dy * t, oz + dz * t);
      hit.normal.set(this.entityNx, this.entityNy, this.entityNz);
      hit.surface = c.surface;
      hit.entity = c.entity;
      hit.headshot = c.headshot;
      hit.object = c.object;
      hit.backface = false;
      list.push(hit);
    }

    // Stable insertion sort: n is small and it needs no comparator closure.
    for (let i = 1; i < list.length; i++) {
      const v = list[i];
      let j = i - 1;
      while (j >= 0 && list[j].distance > v.distance) {
        list[j + 1] = list[j];
        j--;
      }
      list[j + 1] = v;
    }
    return list;
  }

  /**
   * Swept sphere against STATIC geometry only.
   *
   * Static-only is deliberate: the primary caller is the character controller,
   * and the player's own entity collider would otherwise report a hit at t=0 on
   * every step. Use `sphereCastEx` when entity colliders should participate.
   *
   * This is an exact continuous test (plane → edge → vertex per triangle), not
   * a multi-ray approximation, so a capsule cannot slip through a wall corner.
   */
  sphereCast(
    origin: THREE.Vector3,
    radius: number,
    direction: THREE.Vector3,
    maxDistance: number,
  ): RaycastHit | null {
    return this.sphereCastEx(origin, radius, direction, maxDistance);
  }

  /** `sphereCast` with filtering and optional entity-collider participation. */
  sphereCastEx(
    origin: THREE.Vector3,
    radius: number,
    direction: THREE.Vector3,
    maxDistance: number,
    filter?: PhysicsFilter,
    includeEntities = false,
  ): RaycastHit | null {
    const dl = Math.sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    // Negated form: `NaN < 1e-9` is FALSE, so the naive guard lets a NaN
    // direction through, and the whole BVH is then walked with NaN bounds
    // tests (measured 1166x slower) only to return null anyway.
    if (!(dl > 1e-9) || !(maxDistance > 0)) return null;
    const inv = 1 / dl;
    const dx = direction.x * inv, dy = direction.y * inv, dz = direction.z * inv;
    const ox = origin.x, oy = origin.y, oz = origin.z;
    const r = Math.max(1e-4, radius);

    let best = maxDistance;
    let staticHit = false;
    if (!filter?.entitiesOnly) {
      staticHit = this.world.sphereCast(ox, oy, oz, r, dx, dy, dz, best, this.sweepResult);
      if (staticHit) best = this.sweepResult.t;
    }

    let entityIdx = -1;
    if (includeEntities) {
      this.syncEntityColliders();
      // Inflating the analytic collider by the sweep radius is exact for
      // spheres and capsules, and rounds an OBB's corners slightly — an
      // acceptable and conservative approximation for a body-sized box.
      entityIdx = this.raycastEntities(ox, oy, oz, dx, dy, dz, best, filter, r);
    }

    if (entityIdx >= 0) {
      const c = this.colliders[entityIdx];
      const hit = this.nextHit();
      hit.distance = this.entityT;
      hit.point.set(
        ox + dx * this.entityT - this.entityNx * r,
        oy + dy * this.entityT - this.entityNy * r,
        oz + dz * this.entityT - this.entityNz * r,
      );
      hit.normal.set(this.entityNx, this.entityNy, this.entityNz);
      if (hit.normal.x * dx + hit.normal.y * dy + hit.normal.z * dz > 0) hit.normal.negate();
      hit.surface = c.surface;
      hit.entity = c.entity;
      hit.headshot = c.headshot;
      hit.object = c.object;
      hit.backface = false;
      return hit;
    }

    if (!staticHit) return null;
    const sr = this.sweepResult;
    const hit = this.nextHit();
    hit.distance = sr.t;
    hit.point.set(sr.px, sr.py, sr.pz);
    hit.normal.set(sr.nx, sr.ny, sr.nz);
    hit.surface = sr.surface;
    hit.entity = undefined;
    hit.headshot = false;
    hit.object = undefined;
    hit.backface = false;
    return hit;
  }

  // ── entity broadphase ─────────────────────────────────────────────────────

  private entityT = 0;
  private entityNx = 0;
  private entityNy = 1;
  private entityNz = 0;

  /**
   * Linear scan with an AABB pre-reject. There are only a few dozen colliders,
   * so a grid's traversal setup costs more than the tests it saves; the AABB
   * slab reject already discards almost everything.
   */
  private raycastEntities(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
    filter?: PhysicsFilter,
    inflate = 0,
  ): number {
    let bestT = maxDist;
    let bestIdx = -1;
    let bnx = 0, bny = 1, bnz = 0;

    // Tracked separately so an overlapping torso capsule cannot mask a skull.
    const margin = this.headshotPriorityMargin;
    let headT = Infinity;
    let headIdx = -1;
    let hnx = 0, hny = 1, hnz = 0;

    for (let i = 0; i < this.colliders.length; i++) {
      const c = this.colliders[i];
      if (!c.alive) continue;
      if (this.excluded(c, filter)) continue;
      // Head candidates must be tested against the full range, not the running
      // best, or a nearer torso hit would cull them before they are considered.
      const limit = c.headshot && margin > 0 ? maxDist : bestT;
      const t = this.intersectCollider(c, ox, oy, oz, dx, dy, dz, limit, inflate);
      if (t < 0) continue;

      if (c.headshot && margin > 0 && t < headT) {
        headT = t;
        headIdx = i;
        hnx = this.entityNx; hny = this.entityNy; hnz = this.entityNz;
      }
      if (t < bestT) {
        bestT = t;
        bestIdx = i;
        bnx = this.entityNx; bny = this.entityNy; bnz = this.entityNz;
      }
    }

    if (
      headIdx >= 0 && headIdx !== bestIdx
      && headT - bestT <= margin
      && (bestIdx < 0 || this.colliders[headIdx].entity === this.colliders[bestIdx].entity)
    ) {
      bestIdx = headIdx;
      bestT = headT;
      bnx = hnx; bny = hny; bnz = hnz;
    }

    if (bestIdx >= 0) {
      this.entityT = bestT;
      this.entityNx = bnx; this.entityNy = bny; this.entityNz = bnz;
    }
    return bestIdx;
  }

  /** Exact analytic test. Writes the surface normal to `entityN*`. */
  private intersectCollider(
    c: EntityCollider,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
    inflate = 0,
  ): number {
    // AABB pre-reject (inflated by the sweep radius when spherecasting).
    const idx = 1 / (dx !== 0 ? dx : 1e-20);
    const idy = 1 / (dy !== 0 ? dy : 1e-20);
    const idz = 1 / (dz !== 0 ? dz : 1e-20);
    let tmin = 0;
    let tmax = maxDist;

    let t1 = (c.aMinX - inflate - ox) * idx;
    let t2 = (c.aMaxX + inflate - ox) * idx;
    if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmax < tmin) return -1;

    t1 = (c.aMinY - inflate - oy) * idy;
    t2 = (c.aMaxY + inflate - oy) * idy;
    if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmax < tmin) return -1;

    t1 = (c.aMinZ - inflate - oz) * idz;
    t2 = (c.aMaxZ + inflate - oz) * idz;
    if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmax < tmin) return -1;

    switch (c.shape) {
      case SHAPE_SPHERE: {
        const t = raySphere(ox, oy, oz, dx, dy, dz, c.center.x, c.center.y, c.center.z, c.worldRadius + inflate);
        if (t < 0 || t > maxDist) return -1;
        const r = c.worldRadius + inflate;
        let nx = ox + dx * t - c.center.x;
        let ny = oy + dy * t - c.center.y;
        let nz = oz + dz * t - c.center.z;
        const l = r > 1e-9 ? 1 / r : 0;
        nx *= l; ny *= l; nz *= l;
        const ll = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (ll > 1e-9) { const k = 1 / ll; nx *= k; ny *= k; nz *= k; }
        else { nx = -dx; ny = -dy; nz = -dz; }
        this.entityNx = nx; this.entityNy = ny; this.entityNz = nz;
        return t;
      }

      case SHAPE_CAPSULE: {
        const t = rayCapsule(
          ox, oy, oz, dx, dy, dz,
          c.segA.x, c.segA.y, c.segA.z,
          c.segB.x, c.segB.y, c.segB.z,
          c.worldRadius + inflate,
        );
        if (t < 0 || t > maxDist) return -1;
        this.entityNx = lastNormalX();
        this.entityNy = lastNormalY();
        this.entityNz = lastNormalZ();
        return t;
      }

      default:
        return this.intersectOBB(c, ox, oy, oz, dx, dy, dz, maxDist, inflate);
    }
  }

  /**
   * Ray vs oriented box. The ray is pushed into the collider's local space;
   * because `M · (localOrigin + t · localDir) = worldOrigin + t · worldDir`, the
   * local parameter `t` is already the world distance along a unit direction —
   * no rescaling needed, even with non-uniform scale.
   */
  private intersectOBB(
    c: EntityCollider,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
    inflate: number,
  ): number {
    const e = c.invMatrix.elements;
    const lox = e[0] * ox + e[4] * oy + e[8] * oz + e[12];
    const loy = e[1] * ox + e[5] * oy + e[9] * oz + e[13];
    const loz = e[2] * ox + e[6] * oy + e[10] * oz + e[14];
    const ldx = e[0] * dx + e[4] * dy + e[8] * dz;
    const ldy = e[1] * dx + e[5] * dy + e[9] * dz;
    const ldz = e[2] * dx + e[6] * dy + e[10] * dz;

    const minx = c.offset.x - c.half.x - inflate;
    const maxx = c.offset.x + c.half.x + inflate;
    const miny = c.offset.y - c.half.y - inflate;
    const maxy = c.offset.y + c.half.y + inflate;
    const minz = c.offset.z - c.half.z - inflate;
    const maxz = c.offset.z + c.half.z + inflate;

    const iX = 1 / (ldx !== 0 ? ldx : 1e-20);
    const iY = 1 / (ldy !== 0 ? ldy : 1e-20);
    const iZ = 1 / (ldz !== 0 ? ldz : 1e-20);

    let tmin = 0;
    let tmax = maxDist;
    let axis = 0;
    let sign = -1;

    let t1 = (minx - lox) * iX;
    let t2 = (maxx - lox) * iX;
    let s = -1;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 0; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmax < tmin) return -1;

    t1 = (miny - loy) * iY;
    t2 = (maxy - loy) * iY;
    s = -1;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 1; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmax < tmin) return -1;

    t1 = (minz - loz) * iZ;
    t2 = (maxz - loz) * iZ;
    s = -1;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = 2; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmax < tmin) return -1;

    if (tmin > maxDist) return -1;

    // Local axis normal → world via the inverse-transpose, (M⁻¹)ᵀ.
    //
    // For the unit basis vector e_axis this selects ROW `axis` of M⁻¹, not the
    // column. three stores column-major (`e[col*4 + row]`), so row `axis` is
    // e[axis], e[4+axis], e[8+axis]. Reading the column instead yields Rᵀ
    // rather than R, which for a yaw-rotated hitbox mirrors the normal — up to
    // 140° of error, and exactly 90° at yaw 45°.
    let nx = e[axis] * sign;
    let ny = e[4 + axis] * sign;
    let nz = e[8 + axis] * sign;
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l > 1e-9) { const k = 1 / l; nx *= k; ny *= k; nz *= k; }
    else { nx = -dx; ny = -dy; nz = -dz; }
    this.entityNx = nx; this.entityNy = ny; this.entityNz = nz;
    return tmin;
  }

  /** `PhysicsFilter.exclude`, matched against the collider and its ancestors. */
  private excluded(c: EntityCollider, filter?: PhysicsFilter): boolean {
    const ex = filter?.exclude;
    if (!ex || ex.length === 0) return false;
    let node: THREE.Object3D | null = c.object;
    for (let depth = 0; node && depth < EXCLUDE_PARENT_DEPTH; depth++) {
      for (let i = 0; i < ex.length; i++) if (ex[i] === node) return true;
      node = node.parent;
    }
    return false;
  }

  /** Stable hit object `k` belonging to `raycastAll` ring slot `slot`. */
  private listSlot(slot: number, k: number): PenetrationHit {
    const store = this.listStore[slot];
    while (store.length <= k) store.push(makeHit());
    return store[k];
  }

  /**
   * True when the last `raycastAll` found more surfaces than it can report.
   * Entry/exit pairing is unreliable when set, so penetration code should treat
   * the shot as fully absorbed rather than trusting the thickness it computes.
   */
  get penetrationTruncated(): boolean { return this.world.penetrationTruncated; }

  private nextHit(): PenetrationHit {
    const h = this.hitRing[this.ringIndex];
    this.ringIndex = (this.ringIndex + 1) % HIT_RING;
    return h;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Debug view
  // ───────────────────────────────────────────────────────────────────────────

  /** Wireframe overlay of every collider. `showBVH` adds static BVH bounds. */
  setDebug(enabled: boolean, showBVH = false): void {
    this.debugShowBVH = showBVH;
    if (enabled === this.debugEnabled) return;
    this.debugEnabled = enabled;

    if (!enabled) {
      if (this.debugObject && this.debugObject.parent) this.debugObject.parent.remove(this.debugObject);
      this.debugObject?.geometry.dispose();
      (this.debugObject?.material as THREE.Material | undefined)?.dispose();
      this.debugObject = null;
      this.debugPositions = null;
      this.debugColors = null;
      return;
    }

    const positions = new Float32Array(DEBUG_MAX_VERTS * 3);
    const colors = new Float32Array(DEBUG_MAX_VERTS * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true, opacity: 0.85 });
    const lines = new THREE.LineSegments(geom, mat);
    lines.name = 'PhysicsDebug';
    lines.frustumCulled = false;
    lines.renderOrder = 999;
    this.debugObject = lines;
    this.debugPositions = positions;
    this.debugColors = colors;
    this.debugBVHDirty = true;
    this.ctx?.scene.add(lines);
  }

  get debug(): boolean { return this.debugEnabled; }

  private rebuildDebug(): void {
    const pos = this.debugPositions;
    const col = this.debugColors;
    const obj = this.debugObject;
    if (!pos || !col || !obj) return;

    this.debugVerts = 0;
    this.syncEntityColliders(true);

    for (let i = 0; i < this.colliders.length; i++) {
      const c = this.colliders[i];
      if (!c.alive) continue;
      const r = c.headshot ? 1 : 0.1;
      const g = c.headshot ? 0.35 : 1;
      const b = 0.25;
      if (c.shape === SHAPE_SPHERE) {
        this.pushSphere(c.center.x, c.center.y, c.center.z, c.worldRadius, r, g, b);
      } else if (c.shape === SHAPE_CAPSULE) {
        this.pushSphere(c.segA.x, c.segA.y, c.segA.z, c.worldRadius, r, g, b);
        this.pushSphere(c.segB.x, c.segB.y, c.segB.z, c.worldRadius, r, g, b);
        this.pushSegment(c.segA.x, c.segA.y, c.segA.z, c.segB.x, c.segB.y, c.segB.z, r, g, b);
      } else {
        this.pushOBB(c, r, g, b);
      }
    }

    // Rigid bodies: draw their sphere clusters so a wrong shape is obvious.
    const bodies = this.solver.all;
    for (let i = 0; i < bodies.length; i++) {
      const body = bodies[i];
      if (!body.alive) continue;
      const g = body.sleeping ? 0.35 : 1;
      const q = body.quaternion;
      for (let s = 0; s < body.sphereCount; s++) {
        const lx = body.sphereLocal[s * 3], ly = body.sphereLocal[s * 3 + 1], lz = body.sphereLocal[s * 3 + 2];
        // Inline quaternion rotate; the debug path is cold but still allocation-free.
        const tx = 2 * (q.y * lz - q.z * ly);
        const ty = 2 * (q.z * lx - q.x * lz);
        const tz = 2 * (q.x * ly - q.y * lx);
        const wx = body.position.x + lx + q.w * tx + (q.y * tz - q.z * ty);
        const wy = body.position.y + ly + q.w * ty + (q.z * tx - q.x * tz);
        const wz = body.position.z + lz + q.w * tz + (q.x * ty - q.y * tx);
        this.pushSphere(wx, wy, wz, body.sphereRadius[s], 1, g, 0.15);
      }
    }

    if (this.debugShowBVH) {
      this.world.debugBounds(DEBUG_BVH_DEPTH, this.pushBox);
    }

    obj.geometry.setDrawRange(0, this.debugVerts);
    const pa = obj.geometry.getAttribute('position') as THREE.BufferAttribute;
    const ca = obj.geometry.getAttribute('color') as THREE.BufferAttribute;
    pa.needsUpdate = true;
    ca.needsUpdate = true;
    obj.geometry.computeBoundingSphere();
  }

  private pushSegment(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    r: number, g: number, b: number,
  ): void {
    const pos = this.debugPositions;
    const col = this.debugColors;
    if (!pos || !col || this.debugVerts + 2 > DEBUG_MAX_VERTS) return;
    let o = this.debugVerts * 3;
    pos[o] = ax; pos[o + 1] = ay; pos[o + 2] = az;
    col[o] = r; col[o + 1] = g; col[o + 2] = b;
    o += 3;
    pos[o] = bx; pos[o + 1] = by; pos[o + 2] = bz;
    col[o] = r; col[o + 1] = g; col[o + 2] = b;
    this.debugVerts += 2;
  }

  /** Three orthogonal great circles — reads as a sphere from any angle. */
  private pushSphere(cx: number, cy: number, cz: number, radius: number, r: number, g: number, b: number): void {
    const SEG = 12;
    const step = (Math.PI * 2) / SEG;
    for (let i = 0; i < SEG; i++) {
      const a0 = i * step;
      const a1 = a0 + step;
      const s0 = Math.sin(a0) * radius, c0 = Math.cos(a0) * radius;
      const s1 = Math.sin(a1) * radius, c1 = Math.cos(a1) * radius;
      this.pushSegment(cx + c0, cy + s0, cz, cx + c1, cy + s1, cz, r, g, b);
      this.pushSegment(cx + c0, cy, cz + s0, cx + c1, cy, cz + s1, r, g, b);
      this.pushSegment(cx, cy + c0, cz + s0, cx, cy + c1, cz + s1, r, g, b);
    }
  }

  private readonly obbCorner = new THREE.Vector3();
  private readonly obbPts: THREE.Vector3[] = Array.from({ length: 8 }, () => new THREE.Vector3());

  private pushOBB(c: EntityCollider, r: number, g: number, b: number): void {
    let n = 0;
    for (let sx = -1; sx <= 1; sx += 2) {
      for (let sy = -1; sy <= 1; sy += 2) {
        for (let sz = -1; sz <= 1; sz += 2) {
          this.obbCorner.set(
            c.offset.x + c.half.x * sx,
            c.offset.y + c.half.y * sy,
            c.offset.z + c.half.z * sz,
          ).applyMatrix4(c.matrix);
          this.obbPts[n++].copy(this.obbCorner);
        }
      }
    }
    // Corner order is (x,y,z) with z fastest: 0..7 = (---)(--+)(-+-)(-++)...
    const edges = [0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7];
    for (let i = 0; i < edges.length; i += 2) {
      const p = this.obbPts[edges[i]];
      const q = this.obbPts[edges[i + 1]];
      this.pushSegment(p.x, p.y, p.z, q.x, q.y, q.z, r, g, b);
    }
  }

  private readonly pushBox = (
    minx: number, miny: number, minz: number,
    maxx: number, maxy: number, maxz: number,
  ): void => {
    const r = 0.25, g = 0.45, b = 1;
    // bottom
    this.pushSegment(minx, miny, minz, maxx, miny, minz, r, g, b);
    this.pushSegment(maxx, miny, minz, maxx, miny, maxz, r, g, b);
    this.pushSegment(maxx, miny, maxz, minx, miny, maxz, r, g, b);
    this.pushSegment(minx, miny, maxz, minx, miny, minz, r, g, b);
    // top
    this.pushSegment(minx, maxy, minz, maxx, maxy, minz, r, g, b);
    this.pushSegment(maxx, maxy, minz, maxx, maxy, maxz, r, g, b);
    this.pushSegment(maxx, maxy, maxz, minx, maxy, maxz, r, g, b);
    this.pushSegment(minx, maxy, maxz, minx, maxy, minz, r, g, b);
    // uprights
    this.pushSegment(minx, miny, minz, minx, maxy, minz, r, g, b);
    this.pushSegment(maxx, miny, minz, maxx, maxy, minz, r, g, b);
    this.pushSegment(maxx, miny, maxz, maxx, maxy, maxz, r, g, b);
    this.pushSegment(minx, miny, maxz, minx, maxy, maxz, r, g, b);
  };
}
