/**
 * NOISEFIELD — Ballistics.
 *
 * Hitscan resolution with damage falloff, headshot detection and — the
 * signature mechanic of the modern military shooter — MULTI-LAYER WALL PENETRATION.
 *
 * The penetration model is intentionally simple but physically motivated:
 * every surface has a *hardness* (energy cost per metre of material) and the
 * round carries a normalised energy budget of 1.0. Passing through material
 * costs `thickness × hardness / (penetration × 2.2)`. When the budget runs out
 * the round stops; while it survives, damage is scaled by the remaining
 * energy, so a wallbang is a real but degraded hit rather than a free one.
 *
 * Everything here is allocation-conscious. Vectors used for intermediate maths
 * are module-scope scratch; the only per-shot allocations are the event
 * payloads themselves, which must be fresh because listeners are allowed to
 * retain them.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  IMaterialLibrary,
  IPhysics,
  PhysicsFilter,
  RaycastHit,
  SurfaceType,
  WeaponSpec,
} from '../core/Types';
import { falloffAt } from './Specs';

// ─────────────────────────────────────────────────────────────────────────────
// Material response tables
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Energy cost per metre of material traversed, before the weapon's
 * penetration rating is applied. Calibrated so a 0.55-penetration rifle just
 * barely defeats a 22 cm concrete wall and comfortably defeats 6 cm of wood.
 */
const SURFACE_HARDNESS: Record<SurfaceType, number> = {
  concrete: 5.0,
  metal: 9.0,
  wood: 1.6,
  sand: 3.0,
  dirt: 3.2,
  glass: 0.5,
  water: 1.2,
  flesh: 0.8,
  foliage: 0.15,
  fabric: 0.3,
  rubber: 1.8,
};

/**
 * Assumed thickness (metres) when the ray cannot find a back face — which is
 * the common case, because most level geometry is single-sided and a
 * `THREE.Raycaster` will not report the far wall of a closed volume. Without
 * this fallback, penetration silently never terminates on single-sided walls.
 */
const DEFAULT_THICKNESS: Record<SurfaceType, number> = {
  concrete: 0.22,
  metal: 0.02,
  wood: 0.06,
  sand: 0.50,
  dirt: 0.60,
  glass: 0.008,
  water: 1.00,
  flesh: 0.30,
  foliage: 0.10,
  fabric: 0.02,
  rubber: 0.05,
};

/** Beyond this many walls a round is not interesting any more. */
const MAX_PENETRATION_LAYERS = 3;
/**
 * How much wider the shot pattern is than the weapon's aim cone. Buckshot
 * leaving a cylinder-bore barrel opens faster than the sights suggest, and this
 * is the single number that expresses that.
 */
export const SHOT_CHOKE = 1.15;
/** Nudge past an exit face so the next cast does not re-hit the same triangle. */
const EXIT_EPSILON = 0.0025;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic RNG
// ─────────────────────────────────────────────────────────────────────────────

/**
 * mulberry32. Deterministic, seedable, allocation-free and fast enough to call
 * a few dozen times per trigger pull. Determinism matters because the fire
 * event carries a `seed`: any other system (VFX, netcode, replay) can
 * reproduce the exact pellet pattern from the seed alone.
 */
export class Rng {
  private state: number;

  constructor(seed = 0x9e3779b9) {
    this.state = seed >>> 0;
  }

  reseed(seed: number): void {
    this.state = seed >>> 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [-1, 1). */
  bipolar(): number {
    return this.next() * 2 - 1;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cone sampling
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sample a direction inside a cone of the given half-angle, uniformly over the
 * projected disc.
 *
 * THE SQRT IS THE WHOLE POINT. The area of an annulus at radius r grows like
 * r·dr, so drawing radius directly from a uniform variable puts far too many
 * samples near the centre: with `r = R·u` half of all shots land inside the
 * inner 50% of the cone, which is only 25% of its area. Drawing `r = R·√u`
 * makes the density uniform per unit area, which is what a real cone of fire
 * looks like and — importantly — makes the published cone angle honest.
 *
 * Small-angle approximation: for the cone half-angles we care about (< 10°)
 * offsetting the direction in the tangent plane and renormalising is within a
 * fraction of a percent of a true spherical-cap sample, and is much cheaper.
 *
 * @param out         receives the result (may alias nothing else)
 * @param dir         unit aim direction
 * @param right/up    orthonormal basis perpendicular to `dir`
 * @param halfAngle   cone half-angle, radians
 * @param u           uniform [0,1) driving radius
 * @param v           uniform [0,1) driving azimuth
 */
export function sampleCone(
  out: THREE.Vector3,
  dir: THREE.Vector3,
  right: THREE.Vector3,
  up: THREE.Vector3,
  halfAngle: number,
  u: number,
  v: number,
): THREE.Vector3 {
  out.copy(dir);
  if (halfAngle <= 1e-6) return out;
  const r = Math.tan(halfAngle) * Math.sqrt(u);
  const a = v * Math.PI * 2;
  out.addScaledVector(right, r * Math.cos(a));
  out.addScaledVector(up, r * Math.sin(a));
  return out.normalize();
}

/**
 * Fixed shotgun pattern, expressed as unit-disc offsets that are later scaled
 * by the pellet spread. Real buckshot does NOT scatter randomly: the shot cup
 * and the choke produce a repeatable rosette. A structured pattern also plays
 * better, because the player learns "centre the target and you get the core
 * pellets" instead of praying.
 *
 * Layout: 1 core pellet, an inner triad at 0.42 R, and an outer quad at 0.88 R.
 * Per-shot jitter (applied at fire time) keeps it from looking stamped.
 */
export const SHOT_PATTERN: ReadonlyArray<readonly [number, number]> = [
  [0.00, 0.00],
  [0.42, 0.0],
  [0.42, (Math.PI * 2) / 3],
  [0.42, (Math.PI * 4) / 3],
  [0.88, Math.PI / 4],
  [0.88, (Math.PI * 3) / 4],
  [0.88, (Math.PI * 5) / 4],
  [0.88, (Math.PI * 7) / 4],
];

// ─────────────────────────────────────────────────────────────────────────────
// Scratch — module scope so the fire path never allocates a vector
// ─────────────────────────────────────────────────────────────────────────────

const _origin = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _pelletDir = new THREE.Vector3();
const _exit = new THREE.Vector3();
const _end = new THREE.Vector3();
const _normalMat = new THREE.Matrix3();
const _worldNormal = new THREE.Vector3();

/** Reusable hit record for the non-physics fallback path. */
interface MutableHit {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  distance: number;
  surface: SurfaceType;
  entity?: number;
  headshot?: boolean;
  object?: THREE.Object3D;
}

function isFiniteVec(v: THREE.Vector3): boolean {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

function makeHit(): MutableHit {
  return {
    point: new THREE.Vector3(),
    normal: new THREE.Vector3(0, 1, 0),
    distance: 0,
    surface: 'concrete',
  };
}

// ─────────────────────────────────────────────────────────────────────────────

export interface ShotResult {
  /** World-space end of the visible trajectory (last impact, or max range). */
  readonly end: THREE.Vector3;
  /** True if anything at all was struck. */
  readonly hitSomething: boolean;
  /** Total damage delivered to entities by this trigger pull. */
  readonly damage: number;
}

/**
 * Resolves shots into impacts, damage and tracers.
 *
 * Built to run standalone: if no `physics` system is registered it falls back
 * to a plain `THREE.Raycaster` over `ctx.scene`, so this system is playable
 * before the physics system lands.
 */
export class Ballistics {
  private readonly ctx: EngineContext;
  private readonly raycaster = new THREE.Raycaster();

  /** Reused by the fallback raycaster; cleared, never reallocated. */
  private readonly intersections: THREE.Intersection[] = [];
  /** Pool of hit records reused across the fallback path. */
  private readonly hitPool: MutableHit[] = [];
  private readonly hits: RaycastHit[] = [];

  /** Objects the round must ignore (the shooter's own collider, viewmodel). */
  private readonly exclude: THREE.Object3D[] = [];

  /**
   * Separate stream for post-penetration tumble.
   *
   * The pellet pattern must be reproducible from the published seed, and the
   * penetration walk consumes a number of draws that depends on what each
   * pellet hit. Sharing one stream would therefore make pellet 7's direction
   * depend on how many walls pellet 0 went through — reproducible only if the
   * replaying peer reproduces every raycast result exactly.
   */
  private readonly tumbleRng = new Rng(0x2f6e15c3);

  private physics: IPhysics | undefined;
  private materials: IMaterialLibrary | undefined;
  private resolvedPhysics = false;
  private resolvedMaterials = false;
  /** Hoisted so the fire path does not allocate a filter object per cast. */
  private readonly filter: PhysicsFilter = {};

  private readonly result = {
    end: new THREE.Vector3(),
    hitSomething: false,
    damage: 0,
  };

  constructor(ctx: EngineContext) {
    this.ctx = ctx;
    // Sprites and lines are decoration; they must never eat a bullet.
    this.raycaster.params.Line.threshold = 0;
    this.raycaster.params.Points.threshold = 0;
    for (let i = 0; i < 8; i++) this.hitPool.push(makeHit());
  }

  /** Register a collider that rounds should pass straight through. */
  addExclusion(object: THREE.Object3D): void {
    if (!this.exclude.includes(object)) this.exclude.push(object);
  }

  /**
   * Late-bind sibling systems. Called lazily on the first shot rather than in
   * `init`, because system registration order is not guaranteed and the
   * physics system may come up after us.
   */
  private resolveSystems(): void {
    // Latched INDEPENDENTLY. Coupling them meant that in the supported
    // no-physics mode neither ever latched (so both were re-queried on every
    // trigger pull), while as soon as physics appeared the material library was
    // frozen to whatever existed at that instant.
    if (!this.resolvedPhysics) {
      this.physics = this.ctx.get<IPhysics>('physics');
      if (this.physics) this.resolvedPhysics = true;
    }
    if (!this.resolvedMaterials) {
      this.materials = this.ctx.get<IMaterialLibrary>('materials');
      if (this.materials) this.resolvedMaterials = true;
    }
  }

  private maxRange(spec: WeaponSpec): number {
    return spec.caliber === '.338' ? 500 : 250;
  }

  // ── public entry point ─────────────────────────────────────────────────────

  /**
   * Resolve one trigger pull.
   *
   * @param spec       weapon being fired
   * @param origin     eye position — rounds originate at the camera, not the
   *                   muzzle, so the round always goes exactly where the
   *                   crosshair is. The muzzle is only used for the tracer.
   * @param aim        unit aim direction (already recoil-offset by the caller)
   * @param right/up   orthonormal basis for cone sampling
   * @param coneHalf   total cone half-angle, radians
   * @param muzzle     world muzzle position, used as the tracer origin
   * @param rng        deterministic source, already seeded for this shot
   * @param tracer     whether this round is a tracer
   */
  fireShot(
    spec: WeaponSpec,
    origin: THREE.Vector3,
    aim: THREE.Vector3,
    right: THREE.Vector3,
    up: THREE.Vector3,
    coneHalf: number,
    muzzle: THREE.Vector3,
    rng: Rng,
    tracer: boolean,
  ): ShotResult {
    this.resolveSystems();

    this.result.hitSomething = false;
    this.result.damage = 0;
    const range = this.maxRange(spec);

    // Refresh world matrices ONCE per trigger pull, not once per cast: a
    // shotgun pull is up to 24 casts and a full-scene traversal each time would
    // cost more than everything else in the fire path combined. `Raycaster`
    // never does this itself — the renderer normally would, but firing happens
    // during `update`, before the frame is drawn, so without it the very first
    // shot of a session tests against identity matrices.
    if (!this.physics) this.ctx.scene.updateMatrixWorld(false);

    if (spec.pellets > 1) {
      // ── Shotgun: structured rosette, jittered ─────────────────────────────
      // The aim cone still applies (movement/bloom widen it) and the pattern
      // is laid out inside a pellet cone that is deliberately wider than the
      // aim cone, so the spread you see is dominated by the choke, not by
      // whether you were walking.
      // The choke widens the aim cone by a fixed factor rather than imposing a
      // floor. A fixed floor would have to be authored at the HIP cone (3.45°),
      // which is wider than the whole aimed cone (1.7°) — so it would swallow
      // the aimed value completely and aiming a shotgun would do nothing at
      // all. Scaling keeps it monotone: 3.45° hip → 1.955° aimed, a 68%
      // reduction in pattern area for aiming.
      const pelletCone = coneHalf * SHOT_CHOKE;
      const phase = rng.next() * Math.PI * 2;
      for (let i = 0; i < spec.pellets; i++) {
        const slot = SHOT_PATTERN[i % SHOT_PATTERN.length];
        // Jitter of ±0.13 R keeps consecutive shots from stamping identically
        // while preserving the recognisable ring structure.
        const rr = Math.min(1, Math.max(0, slot[0] + rng.bipolar() * 0.13));
        const aa = slot[1] + phase + rng.bipolar() * 0.22;
        _pelletDir.copy(aim);
        const r = Math.tan(pelletCone) * rr;
        _pelletDir.addScaledVector(right, r * Math.cos(aa));
        _pelletDir.addScaledVector(up, r * Math.sin(aa));
        _pelletDir.normalize();
        this.trace(spec, origin, _pelletDir, range);
        // The core pellet defines the visible trajectory — a tracer drawn to
        // an outlying pellet reads as a miss even when the pattern connected.
        if (i === 0) _end.copy(this.result.end);
      }
      // One tracer for the whole pattern; eight tracers would look like a
      // firework rather than a shotgun.
      // The documented "visible trajectory" is the CORE pellet's, not
      // whichever pellet the loop happened to finish on.
      this.result.end.copy(_end);
      if (tracer) this.emitTracer(spec, muzzle, _end);
      return this.result;
    }

    // ── Single projectile ───────────────────────────────────────────────────
    sampleCone(_pelletDir, aim, right, up, coneHalf, rng.next(), rng.next());
    this.trace(spec, origin, _pelletDir, range);
    if (tracer) this.emitTracer(spec, muzzle, this.result.end);
    return this.result;
  }

  // ── penetration walk ───────────────────────────────────────────────────────

  /**
   * Walk a single projectile through the world, spending energy on each layer
   * of material. Returns the distance travelled before it stopped.
   */
  private trace(
    spec: WeaponSpec,
    startOrigin: THREE.Vector3,
    direction: THREE.Vector3,
    maxRange: number,
  ): number {
    _origin.copy(startOrigin);
    _dir.copy(direction);

    let energy = 1;
    let travelled = 0;
    let remaining = maxRange;
    let layers = 0;
    // Local, because `result.hitSomething` spans a whole trigger pull: a
    // shotgun pellet that misses must still get its own max-range endpoint
    // rather than inheriting a previous pellet's impact point.
    let struck = false;

    let flewOn = false;
    while (layers < MAX_PENETRATION_LAYERS && remaining > 0.05 && energy > 0) {
      const count = this.castAll(_origin, _dir, remaining);
      if (count === 0) {
        // Survived a wall and then found open air: the trajectory continues to
        // max range, so the tracer must too. Stopping it at the exit face makes
        // every wallbang look like it hit the wall.
        flewOn = layers > 0;
        break;
      }

      const hit = this.hits[0];
      const entryDistance = travelled + hit.distance;
      // Physics implementations and material libraries are outside this system;
      // an out-of-vocabulary surface string would make every table lookup
      // `undefined` and propagate NaN into the impact events every VFX and
      // decal listener consumes. Clamp it at the boundary.
      const surface: SurfaceType = hit.surface in SURFACE_HARDNESS ? hit.surface : 'concrete';

      this.emitImpact(hit.point, hit.normal, _dir, surface, energy);
      struck = true;
      this.result.hitSomething = true;
      this.result.end.copy(hit.point);

      // Damage is applied on entry only. Energy at the moment of entry is what
      // the target absorbs; the exit-side damage of a through-and-through is
      // not modelled separately.
      if (hit.entity !== undefined) {
        const head = hit.headshot === true;
        const dmg =
          spec.damage *
          falloffAt(spec, entryDistance) *
          energy *
          (head ? spec.headshotMultiplier : 1);
        this.result.damage += dmg;
        this.ctx.bus.emit('damage:dealt', {
          target: hit.entity,
          amount: dmg,
          headshot: head,
          position: hit.point.clone(),
        });
      }

      // ── Find the far side of this material ────────────────────────────────
      let thickness = DEFAULT_THICKNESS[surface];
      let haveExit = false;
      for (let i = 1; i < count; i++) {
        const h = this.hits[i];
        // A back face (normal pointing along travel) or the next surface of
        // the same object is the exit. Cap at 3× the nominal thickness so a
        // distant unrelated wall is not mistaken for the far side of this one.
        const isBackFace = h.normal.dot(_dir) > 0;
        if (!isBackFace && h.object !== hit.object) continue;
        const t = h.distance - hit.distance;
        if (t > 1e-3 && t < thickness * 3) {
          thickness = t;
          _exit.copy(h.point);
          haveExit = true;
        }
        break;
      }
      if (!haveExit) {
        _exit.copy(hit.point).addScaledVector(_dir, thickness);
      }

      // ── Spend energy ──────────────────────────────────────────────────────
      // `penetration × 2.2` normalises the 0..1 design knob so that a 1.0 rated
      // weapon defeats about 44 cm of concrete and a 0.1 rated one defeats
      // nothing harder than drywall.
      const cost = (thickness * SURFACE_HARDNESS[surface]) / Math.max(0.05, spec.penetration * 2.2);
      energy -= cost;
      travelled = entryDistance + thickness;

      if (energy <= 0) {
        return travelled;
      }

      // Surviving rounds tumble slightly. A 0.6° deflection is enough to make
      // wallbangs feel imprecise without making them useless.
      const deflect = 0.010 * (1 - energy);
      _dir.x += this.tumbleRng.bipolar() * deflect;
      _dir.y += this.tumbleRng.bipolar() * deflect;
      _dir.z += this.tumbleRng.bipolar() * deflect;
      _dir.normalize();

      // Exit impact: spall and dust on the far side. This is the visual tell
      // that a wall is penetrable, and it is why players learn which walls to
      // shoot through. The exit face's outward normal points ALONG travel —
      // negating it here would bury the spall inside the wall.
      this.emitImpact(_exit, _worldNormal.copy(_dir), _dir, surface, energy);
      this.result.end.copy(_exit);

      _origin.copy(_exit).addScaledVector(_dir, EXIT_EPSILON);
      remaining = maxRange - travelled;
      layers++;
    }

    if (!struck) {
      this.result.end.copy(startOrigin).addScaledVector(direction, maxRange);
    } else if (flewOn) {
      this.result.end.copy(_origin).addScaledVector(_dir, remaining);
    }
    return travelled;
  }

  private emitImpact(
    point: THREE.Vector3,
    normal: THREE.Vector3,
    incoming: THREE.Vector3,
    surface: SurfaceType,
    energy: number,
  ): void {
    // Payloads are cloned: listeners (decals, particles) legitimately retain
    // them for the lifetime of the effect.
    this.ctx.bus.emit('impact', {
      position: point.clone(),
      normal: normal.clone(),
      incoming: incoming.clone(),
      surface,
      energy: Math.min(1, Math.max(0, energy)),
    });
  }

  private emitTracer(spec: WeaponSpec, from: THREE.Vector3, to: THREE.Vector3): void {
    this.ctx.bus.emit('tracer', {
      from: from.clone(),
      to: to.clone(),
      speed: spec.muzzleVelocity,
    });
  }

  // ── raycast abstraction ────────────────────────────────────────────────────

  /**
   * Fill `this.hits` with every intersection along the ray, near → far.
   * Prefers the physics system; falls back to `THREE.Raycaster` so this system
   * runs before the physics system exists.
   */
  private castAll(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number): number {
    this.hits.length = 0;

    const physics = this.physics;
    if (physics) {
      // Reuse the filter object; a fresh literal per cast is 8 allocations for
      // one shotgun trigger pull.
      this.filter.exclude = this.exclude.length > 0 ? this.exclude : undefined;
      const found = physics.raycastAll(origin, direction, maxDistance, this.filter);
      for (let i = 0; i < found.length; i++) {
        const h = found[i];
        // Physics belongs to another system. A non-finite distance would reach
        // `falloffAt` (whose range early-outs are both false for NaN) and be
        // published as a NaN `damage:dealt`, permanently corrupting an entity's
        // health; a non-finite point would reach every decal and VFX listener.
        // Drop the hit rather than propagate it.
        if (!Number.isFinite(h.distance) || !isFiniteVec(h.point) || !isFiniteVec(h.normal)) continue;
        // `typeof NaN === 'number'`, and a NaN id would be published as a
        // `damage:dealt` target that no entity lookup can ever match.
        if (h.entity !== undefined && !Number.isFinite(h.entity)) h.entity = undefined;
        this.hits.push(h);
      }
      return this.hits.length;
    }

    // ── Fallback ────────────────────────────────────────────────────────────
    // World matrices were refreshed once by `fireShot` for the whole pull.
    this.raycaster.set(origin, direction);
    this.raycaster.near = 0;
    this.raycaster.far = maxDistance;
    this.intersections.length = 0;
    this.raycaster.intersectObjects(this.ctx.scene.children, true, this.intersections);

    let n = 0;
    for (let i = 0; i < this.intersections.length && n < this.hitPool.length; i++) {
      const it = this.intersections[i];
      const obj = it.object;
      if (!(obj as THREE.Mesh).isMesh) continue;
      if (obj.userData['noHit'] === true) continue;
      if (this.isExcluded(obj)) continue;

      // Same finiteness screen as the physics path: a degenerate triangle can
      // produce a NaN intersection, which would reach `falloffAt` and be
      // published as NaN damage.
      if (!Number.isFinite(it.distance) || !isFiniteVec(it.point)) continue;

      const rec = this.hitPool[n];
      rec.point.copy(it.point);
      rec.distance = it.distance;
      rec.object = obj;

      // Geometric face normal, object-local → world. `it.normal` is flipped to
      // oppose the ray, which would destroy our ability to tell an entry face
      // from an exit face, so we deliberately use the unflipped face normal.
      if (it.face) {
        _normalMat.getNormalMatrix(obj.matrixWorld);
        rec.normal.copy(it.face.normal).applyMatrix3(_normalMat).normalize();
      } else {
        rec.normal.copy(direction).negate();
      }

      // A degenerate object matrix yields a non-finite normal, which would
      // reach every decal listener and silently break the exit-face test.
      if (!isFiniteVec(rec.normal)) rec.normal.copy(direction).negate();

      rec.surface = this.surfaceOf(obj);
      // `typeof NaN === 'number'`, and a NaN entity id would be published as a
      // `damage:dealt` target that no lookup can ever match.
      const entity = obj.userData['entity'];
      rec.entity = typeof entity === 'number' && Number.isFinite(entity) ? entity : undefined;
      rec.headshot = obj.userData['headshot'] === true;

      this.hits.push(rec);
      n++;
    }
    return n;
  }

  private isExcluded(obj: THREE.Object3D): boolean {
    if (this.exclude.length === 0) return false;
    for (let i = 0; i < this.exclude.length; i++) {
      const ex = this.exclude[i];
      let o: THREE.Object3D | null = obj;
      while (o) {
        if (o === ex) return true;
        o = o.parent;
      }
    }
    return false;
  }

  /** Best-effort surface classification without the material library. */
  private surfaceOf(obj: THREE.Object3D): SurfaceType {
    const lib = this.materials;
    if (lib) {
      try {
        return lib.surfaceOf(obj);
      } catch {
        /* library not ready yet — fall through to the heuristic */
      }
    }
    let o: THREE.Object3D | null = obj;
    while (o) {
      const s = o.userData['surface'];
      if (typeof s === 'string' && s in SURFACE_HARDNESS) return s as SurfaceType;
      o = o.parent;
    }
    return 'concrete';
  }

  dispose(): void {
    this.intersections.length = 0;
    this.hits.length = 0;
    this.exclude.length = 0;
  }
}
