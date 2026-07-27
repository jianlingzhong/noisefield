/**
 * NOISEFIELD — Enemy behaviour: perception, navigation, combat, cover.
 *
 * This file also hosts `WorldProbe` and `NavGrid`. The entities subsystem has a
 * fixed set of six files and has no "shared/" module to put them in; AI.ts is
 * the most world-query-centric of the six, so the abstraction lives here and is
 * imported by Player.ts, Enemy.ts and Ragdoll.ts.
 *
 * ── Design stance ────────────────────────────────────────────────────────────
 * The AI never touches the scene graph or the physics API directly. It reads an
 * `AIBody` (where am I), an `AITarget` (where is the player) and a `WorldProbe`
 * (what can I see / walk on), and writes an `AIOutput` (what I want to do). The
 * body in Enemy.ts is the only thing that moves transforms. That split keeps
 * the behaviour testable and lets the manager throttle AI at range without
 * risking a half-applied transform.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  EntityId,
  IMaterialLibrary,
  IPhysics,
  RaycastHit,
  SurfaceType,
} from '../core/Types';

// ─────────────────────────────────────────────────────────────────────────────
// WorldProbe — one world-query API over either IPhysics or a raw Raycaster
// ─────────────────────────────────────────────────────────────────────────────

/** Reusable hit record. Callers own the instance so queries never allocate. */
export class ProbeHit {
  readonly point = new THREE.Vector3();
  readonly normal = new THREE.Vector3(0, 1, 0);
  distance = 0;
  surface: SurfaceType = 'concrete';
  object: THREE.Object3D | null = null;
}

const _ph = new THREE.Vector3();
const _pd = new THREE.Vector3();
const _pn = new THREE.Vector3();
const _pm = new THREE.Matrix3();

/**
 * Collision / visibility queries with a graceful standalone fallback.
 *
 * When the physics system has registered an `IPhysics` system we use it. When it
 * has not (it may not be registered yet, depending on start-up order) we fall
 * back to a plain `THREE.Raycaster` over the scene, filtered so character
 * meshes and hitboxes never register as world geometry.
 */
export class WorldProbe {
  private physics: IPhysics | undefined;
  private materials: IMaterialLibrary | undefined;
  private readonly scene: THREE.Scene;
  private readonly ray = new THREE.Raycaster();
  private readonly hits: THREE.Intersection[] = [];
  private readonly ctx: EngineContext;
  private refreshTimer = 0;
  /**
   * Top-level scene children minus character roots, refreshed on a timer.
   * `intersectObjects` descends into everything it is handed; with a dozen
   * soldiers in the scene that is several hundred wasted node visits per ray,
   * and the nav build alone fires tens of thousands of rays.
   */
  private readonly roots: THREE.Object3D[] = [];
  private rootsTimer = 0;

  constructor(ctx: EngineContext) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.ray.far = 500;
    this.refresh();
  }

  /** Re-resolve optional systems; they may register after we did. */
  refresh(): void {
    this.physics = this.ctx.get<IPhysics>('physics');
    this.materials = this.ctx.get<IMaterialLibrary>('materials');
  }

  tick(dt: number): void {
    // Cheap poll so a late-registering physics system is picked up automatically.
    this.refreshTimer -= dt;
    if (this.refreshTimer <= 0) {
      this.refreshTimer = 1.0;
      if (!this.physics || !this.materials) this.refresh();
    }
    this.rootsTimer -= dt;
  }

  /** Cached, character-free list of scene roots for the Raycaster fallback. */
  private candidateRoots(): THREE.Object3D[] {
    if (this.rootsTimer > 0 && this.roots.length > 0) return this.roots;
    this.rootsTimer = 0.5;
    this.roots.length = 0;
    const children = this.scene.children;
    for (let i = 0; i < children.length; i++) {
      if (children[i].userData.character === true) continue;
      this.roots.push(children[i]);
    }
    return this.roots;
  }

  get hasPhysics(): boolean {
    return this.physics !== undefined;
  }

  /**
   * Nearest world hit along a ray. `direction` must be normalised.
   * Character meshes and entity hitboxes are ignored — this is a *world* query.
   */
  raycast(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number, out: ProbeHit): boolean {
    const phys = this.physics;
    if (phys) {
      const h = phys.raycast(origin, direction, maxDistance);
      if (!h) return false;
      if (!isEntityHit(h)) return fillFrom(h, out);
      // `PhysicsFilter` has no "exclude entities" option, so when the nearest
      // thing is a soldier we re-query and walk past every entity hit. Only
      // paid for when a character is actually in the way; without this, every
      // line-of-sight test, nav sample and capsule sweep would collide with
      // other soldiers the moment a real physics system is registered.
      const all = phys.raycastAll(origin, direction, maxDistance);
      for (let i = 0; i < all.length; i++) {
        if (!isEntityHit(all[i])) return fillFrom(all[i], out);
      }
      return false;
    }

    this.ray.set(origin, direction);
    this.ray.near = 0;
    this.ray.far = maxDistance;
    this.hits.length = 0;
    this.ray.intersectObjects(this.candidateRoots(), true, this.hits);
    for (let i = 0; i < this.hits.length; i++) {
      const it = this.hits[i];
      if (!isWorldCollider(it.object)) continue;
      out.point.copy(it.point);
      out.distance = it.distance;
      out.object = it.object;
      if (it.normal) {
        // Face normals arrive in object space; rotate into world space.
        _pm.getNormalMatrix(it.object.matrixWorld);
        out.normal.copy(it.normal).applyMatrix3(_pm).normalize();
      } else if (it.face) {
        _pm.getNormalMatrix(it.object.matrixWorld);
        out.normal.copy(it.face.normal).applyMatrix3(_pm).normalize();
      } else {
        out.normal.set(0, 1, 0);
      }
      out.surface = this.surfaceOf(it.object);
      return true;
    }
    return false;
  }

  /** Downcast helper: finds the floor under (x, z) starting at `fromY`. */
  ground(x: number, z: number, fromY: number, maxDrop: number, out: ProbeHit): boolean {
    _ph.set(x, fromY, z);
    _pd.set(0, -1, 0);
    return this.raycast(_ph, _pd, maxDrop, out);
  }

  /** True when nothing blocks the segment `from → to`. */
  lineOfSight(from: THREE.Vector3, to: THREE.Vector3, out: ProbeHit): boolean {
    _pn.copy(to).sub(from);
    const dist = _pn.length();
    if (dist < 1e-4) return true;
    _pn.multiplyScalar(1 / dist);
    // Pull the end in slightly so a target standing flush against a wall is
    // still visible rather than being occluded by the wall it touches.
    return !this.raycast(from, _pn, dist - 0.06, out);
  }

  private surfaceOf(object: THREE.Object3D): SurfaceType {
    const lib = this.materials;
    if (lib) {
      try {
        return lib.surfaceOf(object);
      } catch {
        /* fall through to the userData tag */
      }
    }
    let o: THREE.Object3D | null = object;
    for (let i = 0; i < 6 && o; i++) {
      const s = o.userData.surface;
      if (typeof s === 'string') return s as SurfaceType;
      o = o.parent;
    }
    return 'concrete';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Level introspection — duck-typed so the world system can evolve independently
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The shape we hope the level system exposes. Every member is optional:
 * the world system may name things differently, so we probe for
 * several plausible spellings and fall back to deriving everything from the
 * scene graph. This system must run standalone.
 */
export interface LevelLike {
  bounds?: THREE.Box3;
  getBounds?: () => THREE.Box3;
  spawnPoints?: THREE.Vector3[];
  getSpawnPoints?: () => THREE.Vector3[];
  playerSpawn?: THREE.Vector3;
  getPlayerSpawn?: () => THREE.Vector3;
}

/**
 * `_lvlChild` is used *inside* `getLevelBounds` while accumulating into the
 * caller's `out`. It must therefore never be the same object a caller passes as
 * `out` — `_lvlWork` exists so the helpers below have their own destination.
 * (Aliasing these two silently reduces the level bounds to the last child
 * examined, which then collapses every spawn radius.)
 */
const _lvlChild = new THREE.Box3();
const _lvlWork = new THREE.Box3();
const _lvlSize = new THREE.Vector3();
const _lvlCenter = new THREE.Vector3();
/** Hard cap so a skybox or ground plane cannot blow the nav grid out to km scale. */
const MAX_LEVEL_EXTENT = 220;

function levelSystem(ctx: EngineContext): (LevelLike & { name: string }) | undefined {
  return ctx.get<LevelLike & { name: string }>('level');
}

/** Best-effort world bounds. Always returns something usable. */
export function getLevelBounds(ctx: EngineContext, out: THREE.Box3): THREE.Box3 {
  const level = levelSystem(ctx);
  let got = false;
  if (level) {
    const b = level.getBounds ? level.getBounds() : level.bounds;
    if (b && !b.isEmpty()) {
      out.copy(b);
      got = true;
    }
  }

  if (!got) {
    out.makeEmpty();
    for (const child of ctx.scene.children) {
      if (child.userData.character === true) continue;
      if (!(child as THREE.Mesh).isMesh && child.children.length === 0) continue;
      _lvlChild.setFromObject(child);
      if (_lvlChild.isEmpty()) continue;
      _lvlChild.getSize(_lvlSize);
      // Skip skyboxes / infinite planes: they say nothing about the playfield.
      if (_lvlSize.x > 600 || _lvlSize.z > 600) continue;
      out.union(_lvlChild);
    }
  }

  if (out.isEmpty()) {
    out.min.set(-45, -3, -45);
    out.max.set(45, 20, 45);
    return out;
  }

  // Clamp the extent around the centre so the nav grid stays affordable.
  out.getCenter(_lvlCenter);
  out.getSize(_lvlSize);
  const hx = Math.min(_lvlSize.x, MAX_LEVEL_EXTENT) * 0.5;
  const hz = Math.min(_lvlSize.z, MAX_LEVEL_EXTENT) * 0.5;
  out.min.x = _lvlCenter.x - hx;
  out.max.x = _lvlCenter.x + hx;
  out.min.z = _lvlCenter.z - hz;
  out.max.z = _lvlCenter.z + hz;
  return out;
}

/**
 * Spawn points from the level, or a generated ring dropped onto the ground.
 * @returns the number of valid points written into `out`.
 */
export function getSpawnPoints(ctx: EngineContext, probe: WorldProbe, out: THREE.Vector3[]): number {
  const level = levelSystem(ctx);
  const supplied = level ? (level.getSpawnPoints ? level.getSpawnPoints() : level.spawnPoints) : undefined;
  if (supplied && supplied.length > 0) {
    ensureVec(out, supplied.length);
    for (let i = 0; i < supplied.length; i++) out[i].copy(supplied[i]);
    return supplied.length;
  }

  getLevelBounds(ctx, _lvlWork);
  _lvlWork.getCenter(_lvlCenter);
  _lvlWork.getSize(_lvlSize);

  // Spread points across the whole playfield rather than onto one ring.
  // A single ring couples the spawn radius to the level size, and if that
  // radius falls under the manager's minimum spawn distance *nothing ever
  // spawns* — a silent, permanent failure. Several rings, with a minimum outer
  // radius, keep a distant candidate available whatever the level size. The
  // manager still has a farthest-point fallback for the pathological case.
  // Floor of 34 m so even a tiny level produces an outer ring beyond the
  // manager's minimum spawn distance. Points that end up off the playable area
  // simply fail their ground probe and fall back to the level floor; the
  // manager scores them by distance and visibility either way.
  const halfExtent = Math.max(34, Math.min(_lvlSize.x, _lvlSize.z) * 0.5);
  const rings = [0.35, 0.6, 0.85];
  const perRing = 10;
  const drop = _lvlSize.y + 10;
  const hit = new ProbeHit();
  ensureVec(out, rings.length * perRing);

  let written = 0;
  for (let r = 0; r < rings.length; r++) {
    const radius = halfExtent * rings[r];
    // Offset alternate rings so points do not line up radially.
    const phase = (r * 0.5 * Math.PI * 2) / perRing;
    for (let i = 0; i < perRing; i++) {
      const a = phase + (i / perRing) * Math.PI * 2;
      const x = _lvlCenter.x + Math.cos(a) * radius;
      const z = _lvlCenter.z + Math.sin(a) * radius;
      if (probe.ground(x, z, _lvlWork.max.y + 2, drop, hit)) {
        out[written].set(x, hit.point.y, z);
      } else {
        out[written].set(x, _lvlWork.min.y, z);
      }
      written++;
    }
  }
  return written;
}

/** Best-effort player start. */
export function getPlayerSpawn(ctx: EngineContext, probe: WorldProbe, out: THREE.Vector3): THREE.Vector3 {
  const level = levelSystem(ctx);
  const p = level ? (level.getPlayerSpawn ? level.getPlayerSpawn() : level.playerSpawn) : undefined;
  if (p) {
    out.copy(p);
    return out;
  }
  getLevelBounds(ctx, _lvlWork);
  _lvlWork.getCenter(_lvlCenter);
  const hit = new ProbeHit();
  const drop = (_lvlWork.max.y - _lvlWork.min.y) + 8;
  if (probe.ground(_lvlCenter.x, _lvlCenter.z, _lvlWork.max.y + 2, drop, hit)) {
    out.set(_lvlCenter.x, hit.point.y, _lvlCenter.z);
  } else {
    out.set(_lvlCenter.x, Math.max(0, _lvlWork.min.y), _lvlCenter.z);
  }
  return out;
}

/** True when a physics hit belongs to a character rather than the world. */
function isEntityHit(h: RaycastHit): boolean {
  if (h.entity !== undefined) return true;
  return h.object !== undefined && !isWorldCollider(h.object);
}

function fillFrom(h: RaycastHit, out: ProbeHit): boolean {
  out.point.copy(h.point);
  out.normal.copy(h.normal);
  out.distance = h.distance;
  out.surface = h.surface;
  out.object = h.object ?? null;
  return true;
}

/** Filters out characters, hitboxes and explicitly non-colliding decoration. */
function isWorldCollider(object: THREE.Object3D): boolean {
  let o: THREE.Object3D | null = object;
  for (let i = 0; i < 6 && o; i++) {
    const ud = o.userData;
    if (ud.character === true || ud.hitbox !== undefined || ud.noCollide === true) return false;
    o = o.parent;
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kinematic capsule mover — shared by Player.ts and Enemy.ts
// ─────────────────────────────────────────────────────────────────────────────

export class CapsuleMoveResult {
  /** Blocked laterally this frame (drives step-up / mantle probing). */
  hitWall = false;
  /** Standing on something walkable. */
  grounded = false;
  /** Standing on something too steep to hold. */
  onSteepSlope = false;
  readonly groundNormal = new THREE.Vector3(0, 1, 0);
  groundSurface: SurfaceType = 'concrete';
  /** Vertical distance gained by the step-up pass; used for camera smoothing. */
  steppedUp = 0;
  /** Hit a ceiling while moving up. */
  hitCeiling = false;
}

const SKIN = 0.02;
const _m0 = new THREE.Vector3();
const _m1 = new THREE.Vector3();
const _m2 = new THREE.Vector3();
const _m3 = new THREE.Vector3();
const _mHit = new ProbeHit();
const _mHit2 = new ProbeHit();
/** Fixed outward directions for the depenetration pass; module-level, no alloc. */
const DEPEN_DIRS: THREE.Vector3[] = [];
for (let i = 0; i < 8; i++) {
  const a = (i / 8) * Math.PI * 2;
  DEPEN_DIRS.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
}

/**
 * Collide-and-slide for an upright capsule, driven entirely by rays.
 *
 * A swept-capsule test would be nicer, but rays are the only primitive we can
 * count on when the physics system is absent, and three probe rays per slide
 * iteration is accurate enough for a shooter at these speeds. Two details that
 * matter for feel:
 *   • the lower probe starts *above* `stepHeight`, so kerbs and stair treads do
 *     not register as walls at all — the step-up pass owns them;
 *   • `steppedUp` is reported rather than applied silently, so the caller can
 *     smooth the camera separately from the collider and stairs do not strobe.
 *
 * `position` is the feet position and is mutated in place, as is `velocity`.
 */
export function moveCapsule(
  probe: WorldProbe,
  position: THREE.Vector3,
  velocity: THREE.Vector3,
  radius: number,
  height: number,
  stepHeight: number,
  dt: number,
  slopeLimitCos: number,
  out: CapsuleMoveResult,
): void {
  out.hitWall = false;
  out.grounded = false;
  out.onSteepSlope = false;
  out.steppedUp = 0;
  out.hitCeiling = false;

  // ── Horizontal: up to three slide iterations.
  _m0.set(velocity.x * dt, 0, velocity.z * dt);
  const lowH = stepHeight + 0.05;
  const midH = Math.max(lowH + 0.05, height * 0.5);
  const topH = Math.max(midH + 0.05, height - radius - 0.02);

  for (let iter = 0; iter < 3; iter++) {
    const dist = Math.hypot(_m0.x, _m0.z);
    if (dist < 1e-5) break;
    _m1.set(_m0.x / dist, 0, _m0.z / dist);

    let nearest = -1;
    _m3.set(0, 0, 0);
    for (let h = 0; h < 3; h++) {
      const y = h === 0 ? lowH : h === 1 ? midH : topH;
      _m2.set(position.x, position.y + y, position.z);
      if (!probe.raycast(_m2, _m1, dist + radius + SKIN, _mHit)) continue;
      // Ignore floor-ish and ceiling-ish surfaces; those are the vertical pass's
      // job. The threshold is the CALLER's slope limit, not a separate constant:
      // hardcoding one here opens a band of ramps (44°-50°) that the vertical
      // pass calls walkable ground while this pass calls them a wall, so the
      // player is shoved back down every frame they try to climb.
      if (Math.abs(_mHit.normal.y) > slopeLimitCos) continue;
      if (nearest < 0 || _mHit.distance < nearest) {
        nearest = _mHit.distance;
        _m3.copy(_mHit.normal);
      }
    }

    if (nearest < 0) {
      position.x += _m0.x;
      position.z += _m0.z;
      break;
    }

    out.hitWall = true;
    const allowed = Math.max(0, nearest - radius - SKIN);
    position.x += _m1.x * allowed;
    position.z += _m1.z * allowed;

    // Consume the travelled distance immediately: both the step-up retry and
    // the slide below must operate on what is *left*, not the original delta.
    const remain = dist - allowed;
    _m0.set(_m1.x * remain, 0, _m1.z * remain);

    // Try to walk up the obstruction before giving up and sliding along it.
    if (iter === 0 && stepHeight > 0) {
      const climbed = tryStepUp(probe, position, _m1, radius, height, stepHeight);
      if (climbed > 0) {
        out.steppedUp = climbed;
        // Re-run the remaining delta now that we are standing higher.
        continue;
      }
    }

    // Project the remaining motion (and the velocity) onto the wall plane.
    const d = _m0.x * _m3.x + _m0.z * _m3.z;
    _m0.x -= _m3.x * d;
    _m0.z -= _m3.z * d;
    const vd = velocity.x * _m3.x + velocity.z * _m3.z;
    if (vd < 0) {
      velocity.x -= _m3.x * vd;
      velocity.z -= _m3.z * vd;
    }
  }

  // ── Depenetration: rays outward at mid height push us out of anything we
  //    ended up inside (spawning in geometry, a moving platform, tunnelling).
  _m2.set(position.x, position.y + midH, position.z);
  for (let i = 0; i < DEPEN_DIRS.length; i++) {
    const dir = DEPEN_DIRS[i];
    if (!probe.raycast(_m2, dir, radius, _mHit)) continue;
    if (Math.abs(_mHit.normal.y) > slopeLimitCos) continue;
    const push = radius - _mHit.distance;
    if (push <= 0) continue;
    position.x -= dir.x * push;
    position.z -= dir.z * push;
    _m2.x = position.x;
    _m2.z = position.z;
  }

  // ── Vertical.
  const yBefore = position.y;
  position.y += velocity.y * dt;

  if (velocity.y > 0) {
    // Swept from the PRE-integration head height. Casting from the already
    // advanced head with a length that also includes the travel over-reaches
    // above the head, so a ceiling merely near the apex would snap the player
    // up into it and cut the jump short. `Math.min` guarantees the resolution
    // can only ever move the player down.
    _m2.set(position.x, yBefore + height - 0.05, position.z);
    _m1.set(0, 1, 0);
    if (probe.raycast(_m2, _m1, 0.05 + Math.max(0, position.y - yBefore), _mHit)) {
      position.y = Math.min(position.y, _mHit.point.y - height);
      velocity.y = 0;
      out.hitCeiling = true;
    }
  }

  // Ground probe. The ray is SWEPT: it starts from where we were before the
  // vertical integration and reaches past where we ended up, so the whole
  // distance fallen this frame is covered. A fixed-length probe tunnels through
  // floors as soon as |vy|·dt exceeds the snap distance — which happens on any
  // long fall, and on an ordinary landing after a single dropped frame.
  const probeTop = Math.max(stepHeight, 0.3);
  const snap = velocity.y <= 0.01 ? 0.24 : 0;
  const descended = Math.max(0, yBefore - position.y);
  _m2.set(position.x, yBefore + probeTop, position.z);
  _m1.set(0, -1, 0);
  if (probe.raycast(_m2, _m1, probeTop + descended + snap + 0.02, _mHit2)) {
    const groundY = _mHit2.point.y;
    if (velocity.y <= 0.01 && groundY <= yBefore + probeTop && groundY >= position.y - snap - 0.02) {
      out.groundNormal.copy(_mHit2.normal);
      out.groundSurface = _mHit2.surface;
      if (_mHit2.normal.y >= slopeLimitCos) {
        out.grounded = true;
        position.y = groundY;
        if (velocity.y < 0) velocity.y = 0;
      } else {
        // Too steep to stand on: keep falling, but bias the slide downhill so
        // the player visibly slips off rather than sticking to the face.
        out.onSteepSlope = true;
        position.y = groundY;
        const n = _mHit2.normal;
        const into = velocity.x * n.x + velocity.y * n.y + velocity.z * n.z;
        if (into < 0) {
          velocity.x -= n.x * into;
          velocity.y -= n.y * into;
          velocity.z -= n.z * into;
        }
      }
    }
  }
}

/**
 * Attempts to lift the capsule onto a ledge no taller than `stepHeight`.
 * @returns the height gained, or 0 when the step is not viable.
 */
function tryStepUp(
  probe: WorldProbe,
  position: THREE.Vector3,
  dir: THREE.Vector3,
  radius: number,
  height: number,
  stepHeight: number,
): number {
  const ahead = radius + 0.12;
  // Is the space at head height ahead of us clear? If not this is a wall.
  _m2.set(position.x, position.y + stepHeight + 0.06, position.z);
  if (probe.raycast(_m2, dir, ahead + 0.05, _mHit)) return 0;

  // Probe down onto the ledge surface.
  _m2.set(position.x + dir.x * ahead, position.y + stepHeight + 0.06, position.z + dir.z * ahead);
  _m1.set(0, -1, 0);
  if (!probe.raycast(_m2, _m1, stepHeight + 0.12, _mHit)) return 0;
  if (_mHit.normal.y < 0.65) return 0;

  const rise = _mHit.point.y - position.y;
  if (rise <= 0.008 || rise > stepHeight) return 0;

  // Headroom above the new footing.
  _m2.set(_mHit.point.x, _mHit.point.y + 0.08, _mHit.point.z);
  _m1.set(0, 1, 0);
  if (probe.raycast(_m2, _m1, height - 0.12, _mHit)) return 0;

  position.y += rise;
  return rise;
}

// ─────────────────────────────────────────────────────────────────────────────
// NavGrid — sampled walkability grid + A* + string pulling
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum cells per axis. Caps both memory and the cost of the sampling pass. */
const NAV_MAX_DIM = 112;
const NAV_TARGET_CELL = 0.9;
/** Vertical clearance a soldier needs to occupy a cell. */
const NAV_HEADROOM = 1.85;
/** Two neighbouring cells are connected only if the step between them is small. */
const NAV_MAX_STEP = 0.45;

const _ng = new ProbeHit();
const _ngUp = new THREE.Vector3(0, 1, 0);
const _ngOrigin = new THREE.Vector3();

/**
 * A uniform walkability grid sampled from world geometry by raycast.
 *
 * There is no navmesh in this project, and building one at runtime is not worth
 * the complexity. A grid gives us exact, cheap A* and — crucially — it can be
 * built incrementally across frames so a large level never stalls the first
 * second of play. Until it is ready the AI runs on pure steering.
 */
export class NavGrid {
  cols = 0;
  rows = 0;
  cell = NAV_TARGET_CELL;
  originX = 0;
  originZ = 0;
  ready = false;

  private walkable = new Uint8Array(0);
  private floorY = new Float32Array(0);

  // A* working set — allocated once with the grid, never per search.
  private gScore = new Float32Array(0);
  private fScore = new Float32Array(0);
  private cameFrom = new Int32Array(0);
  private stamp = new Int32Array(0);
  private closed = new Uint8Array(0);
  private heap = new Int32Array(0);
  private heapSize = 0;
  private searchId = 0;

  private buildCursor = 0;
  private probe: WorldProbe | null = null;
  private sampleTopY = 0;
  private sampleDrop = 0;

  /** Sizes the grid to `bounds` and prepares the incremental sampling pass. */
  begin(probe: WorldProbe, bounds: THREE.Box3): void {
    const sx = Math.max(4, bounds.max.x - bounds.min.x);
    const sz = Math.max(4, bounds.max.z - bounds.min.z);
    const cell = Math.max(NAV_TARGET_CELL, Math.max(sx, sz) / NAV_MAX_DIM);

    this.cell = cell;
    this.cols = Math.min(NAV_MAX_DIM, Math.max(2, Math.ceil(sx / cell)));
    this.rows = Math.min(NAV_MAX_DIM, Math.max(2, Math.ceil(sz / cell)));
    this.originX = bounds.min.x;
    this.originZ = bounds.min.z;
    this.sampleTopY = bounds.max.y + 2;
    this.sampleDrop = (bounds.max.y - bounds.min.y) + 6;

    const n = this.cols * this.rows;
    this.walkable = new Uint8Array(n);
    this.floorY = new Float32Array(n);
    this.gScore = new Float32Array(n);
    this.fScore = new Float32Array(n);
    this.cameFrom = new Int32Array(n);
    this.stamp = new Int32Array(n);
    this.closed = new Uint8Array(n);
    // Lazy-deletion A*: a node is re-pushed whenever its g-score improves, so
    // the heap can hold more entries than there are cells. Sizing it at n+1
    // silently drops writes past the end (typed arrays do not throw), which
    // corrupts the search. 4n covers realistic grids; `heapPush` refuses to
    // overflow regardless.
    this.heap = new Int32Array(n * 4 + 16);
    this.buildCursor = 0;
    this.probe = probe;
    this.ready = false;
  }

  /**
   * Samples up to `budget` cells. Call every frame until it returns true.
   * Spreading the ~10k raycasts over ~25 frames keeps the hitch below a frame.
   */
  buildStep(budget: number): boolean {
    const probe = this.probe;
    if (!probe || this.ready) return this.ready;

    const total = this.cols * this.rows;
    const end = Math.min(total, this.buildCursor + budget);
    for (let i = this.buildCursor; i < end; i++) {
      const cx = i % this.cols;
      const cz = (i / this.cols) | 0;
      const wx = this.originX + (cx + 0.5) * this.cell;
      const wz = this.originZ + (cz + 0.5) * this.cell;

      if (!probe.ground(wx, wz, this.sampleTopY, this.sampleDrop, _ng)) continue;
      // Reject anything a soldier could not stand on. cos(50°) ≈ 0.643 matches
      // the player's own slope limit so AI and player agree on the playfield.
      if (_ng.normal.y < 0.643) continue;

      const fy = _ng.point.y;
      _ngOrigin.set(wx, fy + 0.12, wz);
      if (probe.raycast(_ngOrigin, _ngUp, NAV_HEADROOM, _ng)) continue;

      this.walkable[i] = 1;
      this.floorY[i] = fy;
    }
    this.buildCursor = end;
    if (end >= total) {
      this.ready = true;
      this.probe = null;
    }
    return this.ready;
  }

  index(cx: number, cz: number): number {
    return cz * this.cols + cx;
  }

  cellX(wx: number): number {
    return Math.floor((wx - this.originX) / this.cell);
  }

  cellZ(wz: number): number {
    return Math.floor((wz - this.originZ) / this.cell);
  }

  inBounds(cx: number, cz: number): boolean {
    return cx >= 0 && cz >= 0 && cx < this.cols && cz < this.rows;
  }

  isWalkable(cx: number, cz: number): boolean {
    return this.inBounds(cx, cz) && this.walkable[this.index(cx, cz)] === 1;
  }

  floorAt(cx: number, cz: number): number {
    return this.floorY[this.index(cx, cz)];
  }

  cellCenter(cx: number, cz: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(
      this.originX + (cx + 0.5) * this.cell,
      this.floorY[this.index(cx, cz)],
      this.originZ + (cz + 0.5) * this.cell,
    );
  }

  /** Nearest walkable cell index to a world point, or -1. Spiral search. */
  nearestCell(p: THREE.Vector3, maxRings = 6): number {
    const cx = this.cellX(p.x);
    const cz = this.cellZ(p.z);
    if (this.isWalkable(cx, cz)) return this.index(cx, cz);
    for (let r = 1; r <= maxRings; r++) {
      for (let dz = -r; dz <= r; dz++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
          if (this.isWalkable(cx + dx, cz + dz)) return this.index(cx + dx, cz + dz);
        }
      }
    }
    return -1;
  }

  // ── A* ─────────────────────────────────────────────────────────────────────

  private heapPush(node: number): void {
    // Refuse rather than write out of bounds: a full heap degrades the path to
    // whatever has already been expanded, which is recoverable. A silent OOB
    // write is not.
    if (this.heapSize + 2 >= this.heap.length) return;
    let i = ++this.heapSize;
    this.heap[i] = node;
    while (i > 1) {
      const parent = i >> 1;
      if (this.fScore[this.heap[parent]] <= this.fScore[this.heap[i]]) break;
      const t = this.heap[parent];
      this.heap[parent] = this.heap[i];
      this.heap[i] = t;
      i = parent;
    }
  }

  private heapPop(): number {
    const top = this.heap[1];
    this.heap[1] = this.heap[this.heapSize--];
    let i = 1;
    for (;;) {
      const l = i << 1;
      const r = l + 1;
      let best = i;
      if (l <= this.heapSize && this.fScore[this.heap[l]] < this.fScore[this.heap[best]]) best = l;
      if (r <= this.heapSize && this.fScore[this.heap[r]] < this.fScore[this.heap[best]]) best = r;
      if (best === i) break;
      const t = this.heap[best];
      this.heap[best] = this.heap[i];
      this.heap[i] = t;
      i = best;
    }
    return top;
  }

  /**
   * A* with an octile heuristic. Writes world-space waypoints into `out`
   * (reused Vector3s, grown on demand) and returns the waypoint count.
   * Returns 0 when no path exists or the grid is not ready.
   */
  findPath(from: THREE.Vector3, to: THREE.Vector3, out: THREE.Vector3[], maxExpansions = 3000): number {
    if (!this.ready) return 0;
    const start = this.nearestCell(from);
    const goal = this.nearestCell(to, 8);
    if (start < 0 || goal < 0) return 0;
    if (start === goal) {
      ensureVec(out, 1);
      const gx = goal % this.cols;
      const gz = (goal / this.cols) | 0;
      this.cellCenter(gx, gz, out[0]);
      return 1;
    }

    const id = ++this.searchId;
    this.heapSize = 0;
    const gx = goal % this.cols;
    const gz = (goal / this.cols) | 0;

    this.stamp[start] = id;
    this.closed[start] = 0;
    this.gScore[start] = 0;
    this.cameFrom[start] = -1;
    this.fScore[start] = this.heuristic(start % this.cols, (start / this.cols) | 0, gx, gz);
    this.heapPush(start);

    let expansions = 0;
    let found = false;
    while (this.heapSize > 0 && expansions < maxExpansions) {
      const cur = this.heapPop();
      if (this.closed[cur] === 1 && this.stamp[cur] === id) continue;
      this.closed[cur] = 1;
      if (cur === goal) {
        found = true;
        break;
      }
      expansions++;

      const cx = cur % this.cols;
      const cz = (cur / this.cols) | 0;
      const curY = this.floorY[cur];

      for (let k = 0; k < 8; k++) {
        const dx = NEIGH_DX[k];
        const dz = NEIGH_DZ[k];
        const nx = cx + dx;
        const nz = cz + dz;
        if (!this.isWalkable(nx, nz)) continue;
        // Never cut a diagonal corner through a blocked orthogonal neighbour —
        // that is the classic "AI clips the wall pillar" bug.
        if (dx !== 0 && dz !== 0) {
          if (!this.isWalkable(cx + dx, cz) || !this.isWalkable(cx, cz + dz)) continue;
        }
        const ni = this.index(nx, nz);
        const dy = this.floorY[ni] - curY;
        if (Math.abs(dy) > NAV_MAX_STEP) continue;

        const step = (dx !== 0 && dz !== 0 ? 1.41421356 : 1) * this.cell;
        // Climbing costs more, so squads prefer the flat route when both work.
        const tentative = this.gScore[cur] + step + Math.max(0, dy) * 1.6;

        const seen = this.stamp[ni] === id;
        if (seen && this.closed[ni] === 1) continue;
        if (!seen || tentative < this.gScore[ni]) {
          this.stamp[ni] = id;
          this.closed[ni] = 0;
          this.gScore[ni] = tentative;
          this.cameFrom[ni] = cur;
          this.fScore[ni] = tentative + this.heuristic(nx, nz, gx, gz) * 1.05; // slight weighting: faster, near-optimal
          this.heapPush(ni);
        }
      }
    }
    if (!found) return 0;

    // Walk the parent chain backwards, then reverse in place.
    let count = 0;
    for (let n = goal; n !== -1; n = this.cameFrom[n]) count++;
    ensureVec(out, count);
    let n = goal;
    for (let i = count - 1; i >= 0 && n !== -1; i--) {
      this.cellCenter(n % this.cols, (n / this.cols) | 0, out[i]);
      n = this.cameFrom[n];
    }
    return this.stringPull(out, count);
  }

  private heuristic(ax: number, az: number, bx: number, bz: number): number {
    const dx = Math.abs(ax - bx);
    const dz = Math.abs(az - bz);
    // Octile distance — admissible for 8-connected grids.
    return (Math.max(dx, dz) + 0.41421356 * Math.min(dx, dz)) * this.cell;
  }

  /**
   * String pulling: drop any waypoint that can be skipped by walking straight.
   * Grid paths are staircases; without this the soldiers zig-zag like Pac-Man
   * ghosts, which reads as robotic instantly.
   */
  private stringPull(path: THREE.Vector3[], count: number): number {
    if (count <= 2) return count;
    let write = 1;
    let anchor = 0;
    for (let probe = 2; probe < count; probe++) {
      if (!this.clearWalk(path[anchor], path[probe])) {
        path[write].copy(path[probe - 1]);
        anchor = probe - 1;
        write++;
      }
    }
    path[write].copy(path[count - 1]);
    return write + 1;
  }

  /** Supercover walk between two world points; true when every cell is walkable. */
  clearWalk(a: THREE.Vector3, b: THREE.Vector3): boolean {
    const ax = this.cellX(a.x);
    const az = this.cellZ(a.z);
    const bx = this.cellX(b.x);
    const bz = this.cellZ(b.z);
    let x = ax;
    let z = az;
    const dx = Math.abs(bx - ax);
    const dz = Math.abs(bz - az);
    const sx = ax < bx ? 1 : -1;
    const sz = az < bz ? 1 : -1;
    let err = dx - dz;
    let guard = dx + dz + 2;
    let prevY = this.isWalkable(x, z) ? this.floorAt(x, z) : a.y;

    while (guard-- > 0) {
      if (!this.isWalkable(x, z)) return false;
      const y = this.floorAt(x, z);
      if (Math.abs(y - prevY) > NAV_MAX_STEP) return false;
      prevY = y;
      if (x === bx && z === bz) return true;
      const e2 = err * 2;
      if (e2 > -dz) {
        err -= dz;
        x += sx;
        // Widen by one cell on the perpendicular so a path never grazes a corner.
        if (!this.isWalkable(x, z)) return false;
      }
      if (e2 < dx) {
        err += dx;
        z += sz;
        if (!this.isWalkable(x, z)) return false;
      }
    }
    return false;
  }
}

const NEIGH_DX = [1, -1, 0, 0, 1, 1, -1, -1];
const NEIGH_DZ = [0, 0, 1, -1, 1, -1, 1, -1];

function ensureVec(arr: THREE.Vector3[], n: number): void {
  while (arr.length < n) arr.push(new THREE.Vector3());
}

// ─────────────────────────────────────────────────────────────────────────────
// AI contracts
// ─────────────────────────────────────────────────────────────────────────────

export type AIState =
  | 'idle'
  | 'patrol'
  | 'alert'
  | 'seek'
  | 'engage'
  | 'takeCover'
  | 'reposition'
  | 'suppressed'
  | 'dead';

/** Read-only view of the soldier's own body, supplied by Enemy.ts. */
export interface AIBody {
  readonly id: EntityId;
  /** Feet position. */
  readonly position: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  readonly eye: THREE.Vector3;
  readonly muzzle: THREE.Vector3;
  readonly grounded: boolean;
  readonly health: number;
  readonly maxHealth: number;
  /** Current facing, radians, 0 = +Z. */
  readonly yaw: number;
}

/** Read-only view of the player. */
export interface AITarget {
  readonly position: THREE.Vector3;
  readonly eye: THREE.Vector3;
  readonly velocity: THREE.Vector3;
  readonly radius: number;
  readonly height: number;
  readonly alive: boolean;
  /** Applied when an AI bullet connects. */
  damage(amount: number, fromWorld: THREE.Vector3): void;
}

/** Commands written by the AI and consumed by the body. Reused every tick. */
export class AIOutput {
  readonly moveTarget = new THREE.Vector3();
  hasMoveTarget = false;
  /** Desired ground speed, m/s. */
  desiredSpeed = 0;
  /** 0..1 crouch blend. */
  crouch = 0;
  readonly lookTarget = new THREE.Vector3();
  hasLookTarget = false;
  /** 0..1 blend into the shouldered stance. */
  aimWeight = 0;
  /** Shots the body should emit this tick (muzzle flash / audio). */
  shots = 0;
  reloading = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────

const DEG = Math.PI / 180;

/** Half-angle of the vision cone. 110° total is roughly human useful vision. */
const VIEW_HALF_FOV = 55 * DEG;
const VIEW_RANGE = 60;
/** Inside this radius a soldier notices you regardless of facing. */
const PROXIMITY_RANGE = 4.5;
const HEAR_RANGE_GUNSHOT = 55;

/**
 * Accuracy ramp.
 *
 * A soldier that is perfectly accurate the instant it sees you is not
 * difficult, it is unfair — the player never gets the "I got shot at, take
 * cover" beat that makes a firefight readable. So the first burst of a fresh
 * contact sprays wide and the cone tightens over ~2 s of maintained line of
 * sight. Breaking LoS resets most of the progress, which rewards repositioning.
 */
const AIM_ERROR_MAX = 6.0 * DEG;
const AIM_ERROR_MIN = 0.6 * DEG;
const AIM_TIGHTEN_TAU = 1.9;
const AIM_RELAX_RATE = 0.55; // fraction of tightening lost per second without LoS

const WEAPON_RPM = 620;
const SHOT_INTERVAL = 60 / WEAPON_RPM;
const BURST_MIN = 3;
const BURST_MAX = 6;
const BURST_PAUSE_MIN = 0.5;
const BURST_PAUSE_MAX = 1.15;
const MAG_SIZE = 30;
const RELOAD_TIME = 2.6;
/** Per-bullet damage: 100 hp / 12 ≈ 9 hits to kill, so never a one-burst death. */
const BULLET_DAMAGE = 12;
const MUZZLE_VELOCITY = 720;
const ENGAGE_RANGE = 55;
/** Below this the soldier backs off rather than crowding the player. */
const MIN_ENGAGE_RANGE = 4;

const SPEED_PATROL = 1.7;
const SPEED_SEEK = 4.4;
const SPEED_COMBAT = 3.4;
const SPEED_SPRINT = 5.6;

const PATH_REFRESH = 0.7;
const COVER_REEVALUATE = 3.2;

// Scratch.
const _a0 = new THREE.Vector3();
const _a1 = new THREE.Vector3();
const _a2 = new THREE.Vector3();
const _a3 = new THREE.Vector3();
const _a4 = new THREE.Vector3();
const _hit = new ProbeHit();

/** Deterministic per-agent jitter so a squad never acts in lockstep. */
function hash01(seed: number): number {
  let x = Math.imul(seed ^ 0x9e3779b9, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

// ─────────────────────────────────────────────────────────────────────────────
// AIController
// ─────────────────────────────────────────────────────────────────────────────

/** What an AI round ended up doing, so the body can emit accurate FX. */
export interface AIShotResult {
  /** Where the round terminated. */
  readonly end: THREE.Vector3;
  /** 'player' hit the target, 'world' struck geometry, 'air' reached max range. */
  outcome: 'player' | 'world' | 'air';
  /** Surface normal at the impact point; only meaningful for 'world'. */
  readonly normal: THREE.Vector3;
  surface: SurfaceType;
  /** Unit direction the round was travelling. */
  readonly direction: THREE.Vector3;
}

export interface AIControllerDeps {
  probe: WorldProbe;
  nav: NavGrid;
  /** Emits tracer / impact feedback for AI fire. Optional. */
  onShot?: (from: THREE.Vector3, result: AIShotResult) => void;
}

export class AIController {
  state: AIState = 'idle';
  /** 0..1 certainty the player is a live threat. Gates idle → alert → engage. */
  awareness = 0;
  /** 0..1 how pinned down we are. High values force cover and blind fire. */
  suppression = 0;
  /** Set by the manager: 0 = full logic, 1 = reduced rate, 2 = minimal. */
  lod = 0;

  private readonly deps: AIControllerDeps;
  private readonly seed: number;
  private readonly rngBase: number;

  private stateTime = 0;
  private hasLoS = false;
  private losTime = 0;
  private lastSeenAt = 0;
  private readonly lastKnown = new THREE.Vector3();
  private hasLastKnown = false;

  // Navigation.
  private readonly path: THREE.Vector3[] = [];
  private pathLen = 0;
  private pathIdx = 0;
  private pathTimer = 0;
  private readonly destination = new THREE.Vector3();
  private hasDestination = false;
  private readonly steer = new THREE.Vector3();

  // Combat.
  private aimTightness = 0; // 0 = worst cone, 1 = best cone
  private shotTimer = 0;
  private burstLeft = 0;
  private burstPause = 0;
  private ammo = MAG_SIZE;
  private reloadTimer = 0;
  private readonly aimPoint = new THREE.Vector3();
  /** Reused shot record; the body copies anything it needs before returning. */
  private readonly shotResult: AIShotResult = {
    end: new THREE.Vector3(),
    outcome: 'air',
    normal: new THREE.Vector3(0, 1, 0),
    surface: 'concrete',
    direction: new THREE.Vector3(0, 0, 1),
  };

  // Cover.
  private readonly coverPos = new THREE.Vector3();
  private hasCover = false;
  private coverTimer = 0;
  private peekPhase = 0;
  private peeking = false;
  private readonly peekPos = new THREE.Vector3();

  // Patrol.
  private readonly patrolAnchor = new THREE.Vector3();
  private readonly patrolPoint = new THREE.Vector3();
  private patrolTimer = 0;

  constructor(seed: number, deps: AIControllerDeps) {
    this.seed = seed;
    this.rngBase = seed * 7919;
    this.deps = deps;
  }

  /** Called on (re)spawn. */
  reset(position: THREE.Vector3): void {
    this.state = 'patrol';
    this.awareness = 0;
    this.suppression = 0;
    this.stateTime = 0;
    this.hasLoS = false;
    this.losTime = 0;
    this.hasLastKnown = false;
    this.pathLen = 0;
    this.pathIdx = 0;
    this.pathTimer = 0;
    this.hasDestination = false;
    this.aimTightness = 0;
    this.ammo = MAG_SIZE;
    this.reloadTimer = 0;
    this.burstLeft = 0;
    this.burstPause = 0.4 + this.rand(0) * 0.6;
    this.hasCover = false;
    this.coverTimer = 0;
    this.peeking = false;
    this.peekPhase = 0;
    this.shotTimer = 0;
    this.lastSeenAt = 0;
    this.lod = 0;
    this.patrolAnchor.copy(position);
    this.patrolTimer = 0;
  }

  private rand(salt: number): number {
    return hash01(this.rngBase + salt * 2654435761 + ((this.stateTime * 1000) | 0));
  }

  /** Stable per-agent value in [0,1) that does not change frame to frame. */
  private trait(salt: number): number {
    return hash01(this.rngBase + salt * 40503);
  }

  // ── external stimuli ───────────────────────────────────────────────────────

  /** A gunshot was heard. Loud shots carry further and localise better. */
  onGunshotHeard(position: THREE.Vector3, bodyPos: THREE.Vector3, loudness = 1): void {
    if (this.state === 'dead') return;
    const d = bodyPos.distanceTo(position);
    const range = HEAR_RANGE_GUNSHOT * loudness;
    if (d > range) return;
    const strength = 1 - d / range;
    this.awareness = Math.min(1, this.awareness + 0.45 * strength + 0.2);
    this.lastKnown.copy(position);
    this.hasLastKnown = true;
    if (this.state === 'idle' || this.state === 'patrol') this.setState('alert');
  }

  /** A round cracked past or struck nearby: suppress without granting knowledge. */
  onNearMiss(position: THREE.Vector3, bodyPos: THREE.Vector3): void {
    if (this.state === 'dead') return;
    const d = bodyPos.distanceTo(position);
    if (d > 4.5) return;
    this.suppression = Math.min(1, this.suppression + 0.5 * (1 - d / 4.5));
    this.awareness = Math.min(1, this.awareness + 0.3);
  }

  /** Took a hit — always know roughly where it came from. */
  onDamaged(fromWorld: THREE.Vector3): void {
    if (this.state === 'dead') return;
    this.awareness = 1;
    this.suppression = Math.min(1, this.suppression + 0.35);
    this.lastKnown.copy(fromWorld);
    this.hasLastKnown = true;
    // Being shot while exposed should provoke a cover move, not a stand-and-trade.
    if (!this.hasCover && this.state !== 'takeCover') this.setState('takeCover');
  }

  kill(): void {
    this.setState('dead');
  }

  private setState(s: AIState): void {
    if (this.state === s) return;
    this.state = s;
    this.stateTime = 0;
  }

  // ── main tick ──────────────────────────────────────────────────────────────

  update(dt: number, body: AIBody, target: AITarget, out: AIOutput): void {
    out.shots = 0;
    out.hasMoveTarget = false;
    out.hasLookTarget = false;
    out.reloading = this.reloadTimer > 0;

    if (this.state === 'dead') {
      out.desiredSpeed = 0;
      out.aimWeight = 0;
      return;
    }

    this.stateTime += dt;
    this.suppression = Math.max(0, this.suppression - dt * 0.55);
    if (this.reloadTimer > 0) {
      this.reloadTimer -= dt;
      if (this.reloadTimer <= 0) this.ammo = MAG_SIZE;
    }

    // LOD 2: distant soldiers keep walking their path but stop perceiving,
    // pathing and shooting. Costs one vector op instead of ~6 raycasts.
    if (this.lod >= 2) {
      this.followPath(dt, body, out, SPEED_PATROL);
      out.aimWeight = 0;
      return;
    }

    this.perceive(dt, body, target);
    this.think(dt, body, target, out);
    this.act(dt, body, target, out);
  }

  // ── perception ─────────────────────────────────────────────────────────────

  private perceive(dt: number, body: AIBody, target: AITarget): void {
    if (!target.alive) {
      this.hasLoS = false;
      this.awareness = Math.max(0, this.awareness - dt * 0.3);
      return;
    }

    _a0.copy(target.eye).sub(body.eye);
    const dist = _a0.length();
    let visible = false;

    if (dist < VIEW_RANGE) {
      _a0.multiplyScalar(1 / Math.max(dist, 1e-4));
      // Facing test in the horizontal plane; vertical FOV is generous by design.
      const fwdX = Math.sin(body.yaw);
      const fwdZ = Math.cos(body.yaw);
      const flat = Math.hypot(_a0.x, _a0.z) || 1e-4;
      const dot = (_a0.x * fwdX + _a0.z * fwdZ) / flat;
      const inCone = dot > Math.cos(VIEW_HALF_FOV) || dist < PROXIMITY_RANGE;
      // LoS is the expensive half, so it runs only when the cone test passes.
      if (inCone) {
        visible = this.deps.probe.lineOfSight(body.eye, target.eye, _hit);
        if (!visible) {
          // A second ray at chest height catches targets peeking over cover.
          _a1.copy(target.position).setY(target.position.y + target.height * 0.55);
          visible = this.deps.probe.lineOfSight(body.eye, _a1, _hit);
        }
      }
    }

    this.hasLoS = visible;
    if (visible) {
      // Detection speed falls off with range: a sprinting player at 5 m is
      // spotted instantly, a crouched silhouette at 55 m takes a beat.
      const rate = THREE.MathUtils.lerp(3.2, 0.75, THREE.MathUtils.clamp(dist / VIEW_RANGE, 0, 1));
      this.awareness = Math.min(1, this.awareness + rate * dt);
      this.losTime += dt;
      this.lastSeenAt = 0;
      this.lastKnown.copy(target.position);
      this.hasLastKnown = true;
      this.aimTightness = 1 - (1 - this.aimTightness) * Math.exp(-dt / AIM_TIGHTEN_TAU);
    } else {
      this.losTime = 0;
      this.lastSeenAt += dt;
      this.awareness = Math.max(0, this.awareness - dt * 0.16);
      // Losing sight loosens the cone again — this is what makes breaking LoS
      // and re-peeking a genuinely useful player tactic.
      this.aimTightness = Math.max(0, this.aimTightness - AIM_RELAX_RATE * dt);
    }
  }

  // ── state machine ──────────────────────────────────────────────────────────

  private think(dt: number, body: AIBody, target: AITarget, out: AIOutput): void {
    const dist = body.position.distanceTo(target.position);

    // Reloading and heavy suppression override everything: get small, get safe.
    if (this.suppression > 0.75 && this.state !== 'suppressed' && this.state !== 'dead') {
      this.setState('suppressed');
    }

    switch (this.state) {
      case 'idle':
        if (this.awareness > 0.15) this.setState('alert');
        else if (this.stateTime > 2 + this.trait(1) * 3) this.setState('patrol');
        break;

      case 'patrol':
        if (this.awareness > 0.15) this.setState('alert');
        break;

      case 'alert':
        // Brief "did I just see something?" pause before committing. Without it
        // the AI snaps to combat the instant a pixel of the player appears.
        if (this.awareness >= 0.85 && this.hasLoS && dist < ENGAGE_RANGE) this.setState('engage');
        else if (this.stateTime > 0.6 + this.trait(2) * 0.5 && this.hasLastKnown) this.setState('seek');
        else if (this.awareness <= 0.02) this.setState('patrol');
        break;

      case 'seek':
        if (this.hasLoS && this.awareness > 0.6 && dist < ENGAGE_RANGE) this.setState('engage');
        else if (this.lastSeenAt > 12 || this.awareness <= 0.02) this.setState('patrol');
        break;

      case 'engage':
        if (this.reloadTimer > 0 || this.ammo === 0) this.setState('takeCover');
        else if (!this.hasLoS && this.lastSeenAt > 1.4) this.setState('seek');
        else if (dist > ENGAGE_RANGE * 1.15) this.setState('seek');
        // Rotate through cover so a firefight has movement in it rather than
        // two statues trading bullets.
        else if (this.stateTime > 5 + this.trait(3) * 4 && !this.hasCover) this.setState('takeCover');
        else if (dist < MIN_ENGAGE_RANGE) this.setState('reposition');
        break;

      case 'takeCover':
        if (this.hasCover && this.atDestination(body, 0.7)) {
          this.setState(this.reloadTimer > 0 ? 'takeCover' : 'engage');
          if (this.reloadTimer > 0) this.stateTime = 0;
        }
        if (this.stateTime > 7) this.setState('reposition');
        break;

      case 'reposition':
        if (this.atDestination(body, 0.8) || this.stateTime > 5) {
          this.setState(this.hasLoS ? 'engage' : 'seek');
        }
        break;

      case 'suppressed':
        if (this.suppression < 0.3) this.setState(this.hasCover ? 'engage' : 'takeCover');
        break;

      case 'dead':
        break;
    }
    void dt;
    void out;
  }

  // ── acting ─────────────────────────────────────────────────────────────────

  private act(dt: number, body: AIBody, target: AITarget, out: AIOutput): void {
    out.crouch = 0;
    out.aimWeight = 0;

    switch (this.state) {
      case 'idle': {
        out.desiredSpeed = 0;
        // Scan slowly so an idle guard is not a mannequin.
        const t = this.stateTime * 0.5 + this.trait(4) * 6;
        _a0.set(
          body.position.x + Math.sin(t) * 6,
          body.eye.y,
          body.position.z + Math.cos(t) * 6,
        );
        out.lookTarget.copy(_a0);
        out.hasLookTarget = true;
        break;
      }

      case 'patrol': {
        this.patrolTimer -= dt;
        if (this.patrolTimer <= 0 || this.atDestination(body, 1.0)) {
          this.patrolTimer = 4 + this.trait(5) * 5;
          this.pickPatrolPoint(body);
        }
        this.setDestination(this.patrolPoint);
        this.followPath(dt, body, out, SPEED_PATROL);
        break;
      }

      case 'alert': {
        out.desiredSpeed = 0;
        if (this.hasLastKnown) {
          out.lookTarget.copy(this.lastKnown).setY(this.lastKnown.y + 1.5);
          out.hasLookTarget = true;
        }
        out.aimWeight = 0.35;
        break;
      }

      case 'seek': {
        if (this.hasLastKnown) this.setDestination(this.lastKnown);
        this.followPath(dt, body, out, this.awareness > 0.5 ? SPEED_SPRINT : SPEED_SEEK);
        out.aimWeight = 0.5;
        if (this.hasLastKnown) {
          out.lookTarget.copy(this.lastKnown).setY(this.lastKnown.y + 1.5);
          out.hasLookTarget = true;
        }
        break;
      }

      case 'engage': {
        this.engage(dt, body, target, out);
        break;
      }

      case 'takeCover': {
        this.coverTimer -= dt;
        if (!this.hasCover || this.coverTimer <= 0) {
          // Per-agent jitter so a squad never runs the ~70-raycast cover scan
          // on the same frame.
          this.coverTimer = COVER_REEVALUATE + this.trait(24) * 1.8;
          this.findCover(body, target);
        }
        if (this.hasCover) this.setDestination(this.coverPos);
        this.followPath(dt, body, out, SPEED_SPRINT);
        out.aimWeight = 0.3;
        if (target.alive) {
          out.lookTarget.copy(target.eye);
          out.hasLookTarget = true;
        }
        // Duck once we are actually in the cover slot.
        if (this.atDestination(body, 1.2)) out.crouch = 0.85;
        break;
      }

      case 'reposition': {
        if (!this.hasDestination || this.atDestination(body, 0.8)) this.pickFlank(body, target);
        this.followPath(dt, body, out, SPEED_SPRINT);
        out.aimWeight = 0.45;
        if (target.alive) {
          out.lookTarget.copy(target.eye);
          out.hasLookTarget = true;
        }
        break;
      }

      case 'suppressed': {
        out.desiredSpeed = 0;
        out.crouch = 1;
        out.aimWeight = 0.2;
        if (this.hasCover) {
          _a0.copy(this.coverPos);
          if (body.position.distanceToSquared(_a0) > 1.4) {
            this.setDestination(_a0);
            this.followPath(dt, body, out, SPEED_SPRINT);
          }
        }
        // Occasional blind fire keeps pressure on without being lethal.
        if (target.alive && this.rand(9) > 0.985 && this.ammo > 0 && this.reloadTimer <= 0) {
          this.fireBullet(body, target, out, 9 * DEG);
        }
        if (target.alive) {
          out.lookTarget.copy(target.eye);
          out.hasLookTarget = true;
        }
        break;
      }

      case 'dead':
        out.desiredSpeed = 0;
        break;
    }
  }

  private engage(dt: number, body: AIBody, target: AITarget, out: AIOutput): void {
    if (!target.alive) {
      out.desiredSpeed = 0;
      return;
    }

    out.aimWeight = 1;
    out.lookTarget.copy(target.eye);
    out.hasLookTarget = true;

    const dist = body.position.distanceTo(target.position);

    // Peeking: alternate between the cover slot and a lateral firing position
    // instead of standing in the open. Only the peek window allows fire.
    if (this.hasCover) {
      this.peekPhase -= dt;
      if (this.peekPhase <= 0) {
        this.peeking = !this.peeking;
        this.peekPhase = this.peeking
          ? 1.1 + this.trait(6) * 1.4 // exposure window
          : 0.8 + this.trait(7) * 1.6; // recovery window
      }
      this.computePeekSlot(body, target);
      this.setDestination(this.peeking ? this.peekPos : this.coverPos);
      this.followPath(dt, body, out, SPEED_COMBAT);
      out.crouch = this.peeking ? 0 : 0.8;
    } else if (dist > ENGAGE_RANGE * 0.8) {
      // Too far to be effective: close the distance while firing.
      this.setDestination(target.position);
      this.followPath(dt, body, out, SPEED_COMBAT);
    } else {
      // Combat shuffle — a slow strafe makes them much harder to headshot and
      // reads as "trained soldier" rather than "turret".
      const t = this.stateTime * 1.1 + this.trait(8) * 6;
      _a0.copy(target.position).sub(body.position).setY(0).normalize();
      _a1.set(-_a0.z, 0, _a0.x).multiplyScalar(Math.sin(t) * 2.2);
      _a2.copy(body.position).add(_a1);
      this.setDestination(_a2);
      this.followPath(dt, body, out, SPEED_COMBAT * 0.65);
    }

    if (this.ammo <= 0) {
      this.reloadTimer = RELOAD_TIME;
      out.reloading = true;
      return;
    }
    if (this.reloadTimer > 0) return;
    // No firing while sprinting out of cover — soldiers stop to shoot.
    if (this.hasCover && !this.peeking) return;
    if (!this.hasLoS) return;

    this.runGun(dt, body, target, out);
  }

  /** Burst cadence + the accuracy ramp. */
  private runGun(dt: number, body: AIBody, target: AITarget, out: AIOutput): void {
    if (this.burstPause > 0) {
      this.burstPause -= dt;
      if (this.burstPause <= 0) {
        const span = BURST_MAX - BURST_MIN;
        this.burstLeft = BURST_MIN + Math.floor(this.rand(11) * (span + 1));
        this.shotTimer = 0;
      }
      return;
    }
    if (this.burstLeft <= 0) {
      this.burstPause = BURST_PAUSE_MIN + this.rand(12) * (BURST_PAUSE_MAX - BURST_PAUSE_MIN);
      return;
    }

    this.shotTimer -= dt;
    if (this.shotTimer > 0) return;
    this.shotTimer = SHOT_INTERVAL;
    this.burstLeft--;
    this.ammo--;

    // Cone: the ramp does most of the work; range and target speed add the rest.
    const dist = body.position.distanceTo(target.position);
    const rangeTerm = THREE.MathUtils.clamp(dist / ENGAGE_RANGE, 0, 1) * 1.4 * DEG;
    const motionTerm = THREE.MathUtils.clamp(target.velocity.length() / 7, 0, 1) * 1.6 * DEG;
    const suppressTerm = this.suppression * 3.5 * DEG;
    const base = THREE.MathUtils.lerp(AIM_ERROR_MAX, AIM_ERROR_MIN, this.aimTightness);
    this.fireBullet(body, target, out, base + rangeTerm + motionTerm + suppressTerm);

    if (this.ammo <= 0) this.reloadTimer = RELOAD_TIME;
  }

  /** Resolves a single shot: lead, scatter, LoS, analytic capsule test. */
  private fireBullet(body: AIBody, target: AITarget, out: AIOutput, cone: number): void {
    out.shots++;

    // Lead the target. Soldiers are decent but not perfect predictors, so the
    // lead is deliberately partial — a strafing player still gains an edge.
    const dist = body.muzzle.distanceTo(target.position);
    const flight = dist / MUZZLE_VELOCITY;
    const leadSkill = 0.55 + this.trait(13) * 0.3;
    _a0.copy(target.eye)
      .addScaledVector(target.velocity, flight * leadSkill)
      .sub(body.muzzle);
    // Aim centre-mass, not the head: AI headshots feel arbitrary to receive.
    _a0.y -= target.height * 0.22;

    const len = _a0.length();
    if (len < 1e-4) return;
    _a0.multiplyScalar(1 / len);

    // Scatter inside the cone using a stable orthonormal basis.
    _a1.set(-_a0.z, 0, _a0.x);
    if (_a1.lengthSq() < 1e-6) _a1.set(1, 0, 0);
    _a1.normalize();
    _a2.crossVectors(_a0, _a1);
    const ang = this.rand(14) * Math.PI * 2;
    // sqrt keeps the distribution uniform over the disc rather than centre-biased.
    const rad = Math.sqrt(this.rand(15)) * cone;
    _a3.copy(_a0)
      .addScaledVector(_a1, Math.cos(ang) * rad)
      .addScaledVector(_a2, Math.sin(ang) * rad)
      .normalize();

    // Blocked by geometry before reaching the player?
    const maxRange = ENGAGE_RANGE * 1.5;
    let hitDist = maxRange;
    let blocked = false;
    if (this.deps.probe.raycast(body.muzzle, _a3, hitDist, _hit)) {
      hitDist = _hit.distance;
      blocked = true;
    }

    const shot = this.shotResult;
    shot.direction.copy(_a3);
    shot.surface = 'concrete';
    shot.normal.set(0, 1, 0);

    const t = rayCapsule(body.muzzle, _a3, target.position, target.radius, target.height);
    const connected = t >= 0 && t < hitDist;
    if (connected) {
      target.damage(BULLET_DAMAGE, body.muzzle);
      shot.end.copy(body.muzzle).addScaledVector(_a3, t);
      shot.outcome = 'player';
      shot.surface = 'flesh';
      shot.normal.copy(_a3).negate();
    } else if (blocked) {
      shot.end.copy(_hit.point);
      shot.outcome = 'world';
      // The real surface normal, not the reversed bullet direction: decals and
      // spark cones are oriented from this and look wrong against a flat −dir.
      shot.normal.copy(_hit.normal);
      shot.surface = _hit.surface;
    } else {
      shot.end.copy(body.muzzle).addScaledVector(_a3, maxRange);
      shot.outcome = 'air';
    }
    this.deps.onShot?.(body.muzzle, shot);
  }

  // ── navigation helpers ─────────────────────────────────────────────────────

  private setDestination(p: THREE.Vector3): void {
    if (this.hasDestination && this.destination.distanceToSquared(p) < 0.6) return;
    this.destination.copy(p);
    this.hasDestination = true;
    this.pathTimer = 0; // force a repath on the next follow
  }

  private atDestination(body: AIBody, tolerance: number): boolean {
    if (!this.hasDestination) return true;
    _a0.copy(this.destination).sub(body.position);
    _a0.y = 0;
    return _a0.lengthSq() < tolerance * tolerance;
  }

  /**
   * Advances along the current path, repathing on a timer, and writes the next
   * steering target into `out`. Falls back to whisker-avoidance steering when
   * the nav grid is unavailable or A* fails.
   */
  private followPath(dt: number, body: AIBody, out: AIOutput, speed: number): void {
    if (!this.hasDestination) {
      out.desiredSpeed = 0;
      return;
    }

    this.pathTimer -= dt;
    // Distant soldiers repath far less often; the grid does not change anyway.
    const refresh = this.lod === 0 ? PATH_REFRESH : PATH_REFRESH * 3;
    if (this.pathTimer <= 0) {
      this.pathTimer = refresh + this.trait(16) * 0.25;
      this.pathLen = this.deps.nav.findPath(body.position, this.destination, this.path);
      this.pathIdx = 0;
      // Skip the first waypoint when we are already effectively on it.
      if (this.pathLen > 1 && this.path[0].distanceToSquared(body.position) < 1.0) this.pathIdx = 1;
    }

    if (this.pathLen > 0) {
      // Look-ahead: aim at the waypoint after the nearest one so corners are cut
      // smoothly instead of being clipped at right angles.
      while (this.pathIdx < this.pathLen - 1) {
        _a0.copy(this.path[this.pathIdx]).sub(body.position);
        _a0.y = 0;
        if (_a0.lengthSq() > 0.55 * 0.55) break;
        this.pathIdx++;
      }
      out.moveTarget.copy(this.path[this.pathIdx]);
      out.hasMoveTarget = true;
    } else {
      // No grid / no path: steer directly and let whiskers push us around walls.
      this.avoid(body, this.destination, this.steer);
      out.moveTarget.copy(this.steer);
      out.hasMoveTarget = true;
    }

    const remaining = body.position.distanceTo(this.destination);
    // Arrival easing so soldiers settle onto a slot instead of jittering on it.
    out.desiredSpeed = remaining < 1.2 ? speed * THREE.MathUtils.clamp(remaining / 1.2, 0.15, 1) : speed;
  }

  /** Whisker obstacle avoidance — the no-navmesh fallback. */
  private avoid(body: AIBody, goal: THREE.Vector3, out: THREE.Vector3): void {
    _a0.copy(goal).sub(body.position);
    _a0.y = 0;
    const dist = _a0.length();
    if (dist < 1e-4) {
      out.copy(goal);
      return;
    }
    _a0.multiplyScalar(1 / dist);

    _a1.copy(body.position).setY(body.position.y + 1.0);
    const probeLen = Math.min(3.2, dist);
    let steerX = _a0.x;
    let steerZ = _a0.z;

    for (let i = 0; i < WHISKERS.length; i++) {
      const a = WHISKERS[i];
      const c = Math.cos(a);
      const s = Math.sin(a);
      _a2.set(_a0.x * c - _a0.z * s, 0, _a0.x * s + _a0.z * c);
      if (this.deps.probe.raycast(_a1, _a2, probeLen, _hit)) {
        // Push away proportional to how close the obstruction is.
        const w = (1 - _hit.distance / probeLen) * (i === 0 ? 1.6 : 1.0);
        steerX -= _a2.x * w;
        steerZ -= _a2.z * w;
        // Add a tangential nudge so we slide along the wall rather than stall.
        steerX += -_a2.z * w * 0.9 * (i % 2 === 0 ? 1 : -1);
        steerZ += _a2.x * w * 0.9 * (i % 2 === 0 ? 1 : -1);
      }
    }

    const l = Math.hypot(steerX, steerZ) || 1;
    out.set(
      body.position.x + (steerX / l) * 2.5,
      body.position.y,
      body.position.z + (steerZ / l) * 2.5,
    );
  }

  private pickPatrolPoint(body: AIBody): void {
    const nav = this.deps.nav;
    const ang = this.rand(17) * Math.PI * 2;
    const r = 5 + this.rand(18) * 11;
    _a0.set(
      this.patrolAnchor.x + Math.cos(ang) * r,
      this.patrolAnchor.y,
      this.patrolAnchor.z + Math.sin(ang) * r,
    );
    if (nav.ready) {
      const ci = nav.nearestCell(_a0, 4);
      if (ci >= 0) {
        nav.cellCenter(ci % nav.cols, (ci / nav.cols) | 0, this.patrolPoint);
        return;
      }
    }
    this.patrolPoint.copy(_a0);
    void body;
  }

  /**
   * Cover scoring.
   *
   * Candidates are sampled on a ring around the soldier and scored on three
   * axes that matter: does it break the player's line of sight, is it
   * close enough to reach before dying, and can we still shoot from it (via a
   * lateral peek). A slot that hides you but offers no firing angle is a
   * hiding place, not cover, and scores poorly.
   */
  private findCover(body: AIBody, target: AITarget): void {
    if (!target.alive) {
      this.hasCover = false;
      return;
    }
    const nav = this.deps.nav;
    const probe = this.deps.probe;
    let bestScore = -Infinity;
    let found = false;

    const rings = COVER_RADII.length;
    const spokes = COVER_SPOKES;
    const angOffset = this.trait(19) * Math.PI * 2;

    for (let ri = 0; ri < rings; ri++) {
      const radius = COVER_RADII[ri];
      for (let si = 0; si < spokes; si++) {
        const ang = angOffset + (si / spokes) * Math.PI * 2;
        _a0.set(
          body.position.x + Math.cos(ang) * radius,
          body.position.y,
          body.position.z + Math.sin(ang) * radius,
        );

        // Snap to a walkable cell so we never pick a slot inside a wall.
        if (nav.ready) {
          const ci = nav.nearestCell(_a0, 1);
          if (ci < 0) continue;
          nav.cellCenter(ci % nav.cols, (ci / nav.cols) | 0, _a0);
        } else if (!probe.ground(_a0.x, _a0.z, body.position.y + 2.5, 5, _hit)) {
          continue;
        } else {
          _a0.y = _hit.point.y;
        }

        // (a) Does it break line of sight to a crouched head at that slot?
        _a1.copy(_a0).setY(_a0.y + 1.15);
        const hidden = !probe.lineOfSight(target.eye, _a1, _hit);
        if (!hidden) continue;

        // (b) Distance — near is better, and moving away from the player is
        //     better than crossing in front of them.
        const d = body.position.distanceTo(_a0);
        const towardPlayer = _a0.distanceTo(target.position) < body.position.distanceTo(target.position);
        let score = 12 - d * 0.9 - (towardPlayer ? 2.5 : 0);

        // (c) Firing angle: is there a lateral step that re-acquires the target?
        _a2.copy(target.position).sub(_a0).setY(0).normalize();
        _a3.set(-_a2.z, 0, _a2.x).multiplyScalar(0.85);
        _a4.copy(_a1).add(_a3);
        let canPeek = probe.lineOfSight(target.eye, _a4, _hit);
        if (!canPeek) {
          _a4.copy(_a1).sub(_a3);
          canPeek = probe.lineOfSight(target.eye, _a4, _hit);
        }
        if (canPeek) score += 6;

        // Prefer keeping a sensible engagement distance.
        const engDist = _a0.distanceTo(target.position);
        if (engDist < MIN_ENGAGE_RANGE) score -= 6;
        if (engDist > ENGAGE_RANGE) score -= 5;

        if (score > bestScore) {
          bestScore = score;
          this.coverPos.copy(_a0);
          found = true;
        }
      }
    }

    this.hasCover = found;
    if (found) {
      this.peeking = false;
      this.peekPhase = 0.5 + this.trait(20);
    }
  }

  /** Lateral firing step relative to the current cover slot. */
  private computePeekSlot(body: AIBody, target: AITarget): void {
    _a0.copy(target.position).sub(this.coverPos).setY(0);
    if (_a0.lengthSq() < 1e-6) _a0.set(0, 0, 1);
    _a0.normalize();
    const side = this.trait(21) > 0.5 ? 1 : -1;
    this.peekPos
      .copy(this.coverPos)
      .addScaledVector(_a1.set(-_a0.z, 0, _a0.x), side * 0.95)
      .addScaledVector(_a0, 0.25);
    void body;
  }

  /** Pick a flanking slot roughly 60–120° around the player. */
  private pickFlank(body: AIBody, target: AITarget): void {
    _a0.copy(body.position).sub(target.position).setY(0);
    const r = Math.max(6, Math.min(ENGAGE_RANGE * 0.6, _a0.length()));
    const base = Math.atan2(_a0.z, _a0.x);
    const side = this.trait(22) > 0.5 ? 1 : -1;
    const sweep = (60 + this.rand(23) * 60) * DEG * side;
    const ang = base + sweep;
    _a1.set(
      target.position.x + Math.cos(ang) * r,
      body.position.y,
      target.position.z + Math.sin(ang) * r,
    );
    const nav = this.deps.nav;
    if (nav.ready) {
      const ci = nav.nearestCell(_a1, 4);
      if (ci >= 0) nav.cellCenter(ci % nav.cols, (ci / nav.cols) | 0, _a1);
    }
    this.setDestination(_a1);
    this.hasCover = false;
  }
}

const WHISKERS = [0, 0.42, -0.42, 0.95, -0.95];
const COVER_RADII = [3, 5.5, 8.5];
const COVER_SPOKES = 8;

/**
 * Analytic ray vs. vertical capsule.
 *
 * Used instead of a scene raycast so AI hit resolution never depends on the
 * player having a collider mesh in the graph. Returns the hit distance along
 * `dir`, or -1.
 */
export function rayCapsule(
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  base: THREE.Vector3,
  radius: number,
  height: number,
): number {
  const yBot = base.y + radius;
  const yTop = base.y + height - radius;

  // Infinite-cylinder test in the XZ plane first.
  const ox = origin.x - base.x;
  const oz = origin.z - base.z;
  const a = dir.x * dir.x + dir.z * dir.z;
  const b = 2 * (ox * dir.x + oz * dir.z);
  const c = ox * ox + oz * oz - radius * radius;

  let best = -1;
  if (a > 1e-8) {
    const disc = b * b - 4 * a * c;
    if (disc >= 0) {
      const sq = Math.sqrt(disc);
      for (const t of [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]) {
        if (t < 0) continue;
        const y = origin.y + dir.y * t;
        if (y >= yBot && y <= yTop) {
          best = t;
          break;
        }
      }
    }
  }
  if (best >= 0) return best;

  // Cap spheres.
  for (const cy of [yBot, yTop]) {
    const dx = origin.x - base.x;
    const dy = origin.y - cy;
    const dz = origin.z - base.z;
    const sb = 2 * (dx * dir.x + dy * dir.y + dz * dir.z);
    const sc = dx * dx + dy * dy + dz * dz - radius * radius;
    const disc = sb * sb - 4 * sc;
    if (disc < 0) continue;
    const sq = Math.sqrt(disc);
    const t0 = (-sb - sq) / 2;
    const t1 = (-sb + sq) / 2;
    const t = t0 >= 0 ? t0 : t1;
    if (t >= 0 && (best < 0 || t < best)) best = t;
  }
  return best;
}
