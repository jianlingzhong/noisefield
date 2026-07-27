/**
 * NOISEFIELD — Enemy population manager.
 *
 * Owns the pool, the shared navigation grid, spawning/recycling, squad
 * awareness and AI level-of-detail. One system so the per-frame cost of the
 * whole enemy population is budgeted in a single place rather than scattered
 * across a dozen independent actors.
 */

import * as THREE from 'three';
import type {
  EngineContext,
  EntityId,
  FrameContext,
  IMaterialLibrary,
  System,
} from '../core/Types';
import {
  NavGrid,
  WorldProbe,
  getLevelBounds,
  getSpawnPoints,
  type AITarget,
} from './AI';
import { Enemy } from './Enemy';
import type { Player } from './Player';

// ── Population tuning ────────────────────────────────────────────────────────

/** Alive at once. Enough to feel like a firefight, few enough to stay readable. */
const TARGET_ALIVE = 10;
/** Pool size. Extra slots absorb corpses that are still ragdolling. */
const POOL_SIZE = 16;
/** Never respawn inside the player's comfort zone. */
const MIN_SPAWN_DISTANCE = 26;
const MAX_SPAWN_DISTANCE = 95;
/** Stagger respawns so a wipe does not repopulate in one frame. */
const SPAWN_INTERVAL = 0.7;
const RESPAWN_DELAY_MIN = 1.2;
const RESPAWN_DELAY_MAX = 3.4;

/** LOD ring radii, metres. */
const LOD1_DISTANCE = 42;
const LOD2_DISTANCE = 88;

/** Nav cells sampled per frame with an accelerated physics backend. */
const NAV_BUILD_BUDGET = 380;
/** Sampling rate when falling back to a plain Raycaster over the scene graph. */
const NAV_BUILD_BUDGET_SLOW = 90;

/** Radius within which an ally hears another soldier's rifle. */
const SQUAD_HEAR_RADIUS = 34;
const NEAR_MISS_RADIUS = 5;

const _m0 = new THREE.Vector3();
const _m1 = new THREE.Vector3();
const _bounds = new THREE.Box3();
const _projScreen = new THREE.Matrix4();

/**
 * Structural contract for the enemy registry, so consuming systems can depend on
 * a shape rather than on the concrete class. `Types.ts` is owned by the
 * integration contract and has no entry for this, so it is declared here.
 *
 * Usage from another system:
 *   const enemies = ctx.get<System & IEnemyRegistry>('enemies');
 *   const died = enemies?.damage(id, dmg, headshot, point) ?? false;
 */
export interface IEnemyRegistry {
  /**
   * Apply damage to an entity. The headshot MULTIPLIER is the caller's
   * responsibility; `headshot` here only selects feedback and death impulse.
   * @returns true if this hit killed the target.
   */
  damage(id: EntityId, amount: number, headshot: boolean, point: THREE.Vector3): boolean;
  getColliders(out: THREE.Object3D[]): number;
  readonly aliveCount: number;
}

/** Stand-in used until the player system exists, so enemies simply patrol. */
class NullTarget implements AITarget {
  readonly position = new THREE.Vector3(0, -9999, 0);
  readonly eye = new THREE.Vector3(0, -9999, 0);
  readonly velocity = new THREE.Vector3();
  readonly radius = 0.35;
  readonly height = 1.8;
  readonly alive = false;
  damage(): void {
    /* nothing to damage */
  }
}

interface PendingSpawn {
  timer: number;
}

export class EnemyManager implements System, IEnemyRegistry {
  readonly name = 'enemies';

  private ctx!: EngineContext;
  private probe!: WorldProbe;
  private readonly nav = new NavGrid();
  private readonly pool: Enemy[] = [];
  private readonly byId = new Map<EntityId, Enemy>();
  private readonly spawnPoints: THREE.Vector3[] = [];
  private spawnCount = 0;
  private nextId = 1;
  private spawnTimer = 0;
  private readonly pending: PendingSpawn[] = [];
  private navStarted = false;
  /** Bounds the current nav grid was sampled over, for the late re-resolve. */
  private readonly builtBounds = new THREE.Box3();
  /**
   * Systems register in an arbitrary order and the level may populate the scene
   * after we initialise, in which case the grid and spawn ring were derived
   * from a fallback box. Re-derive once shortly after boot so registration
   * order is not load-bearing.
   */
  private boundsFixTimer = 0.7;
  /**
   * True while stepping the enemies. Their own outgoing rounds raise `impact`
   * synchronously from inside that loop, and treating those as incoming near
   * misses makes a soldier suppress itself the moment it opens fire — pinning
   * it crouched in the open. Player fire arrives from the weapons system's own
   * update, outside this window, and is handled normally.
   */
  private steppingEnemies = false;

  private player: Player | undefined;
  private readonly nullTarget = new NullTarget();
  private readonly frustum = new THREE.Frustum();

  private kills = 0;
  private deaths = 0;
  private streak = 0;
  private unsubscribe: Array<() => void> = [];

  /** Live alive-count, useful for HUD / director logic. */
  get aliveCount(): number {
    let n = 0;
    for (const e of this.pool) if (e.alive) n++;
    return n;
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;

    // Share the player's probe when it exists: one cached physics lookup and
    // one raycaster instead of two.
    const player = ctx.get<Player>('player');
    this.probe = player?.worldProbe ?? new WorldProbe(ctx);
    this.player = player;

    getLevelBounds(ctx, _bounds);
    this.nav.begin(this.probe, _bounds);
    this.builtBounds.copy(_bounds);
    this.navStarted = true;

    this.spawnCount = getSpawnPoints(ctx, this.probe, this.spawnPoints);

    // Prime the population. Instances are created lazily in `trySpawn` so the
    // first frame does not build sixteen soldiers at once.
    for (let i = 0; i < TARGET_ALIVE; i++) this.pending.push({ timer: i * 0.12 });

    this.unsubscribe.push(
      // Player gunfire is the loudest cue in the game: everyone within earshot
      // gets a bearing on it.
      ctx.bus.on('weapon:fire', (p) => this.propagateGunshot(p.origin, 1)),
      // Rounds cracking into cover suppress without revealing the shooter.
      ctx.bus.on('impact', (p) => this.propagateNearMiss(p.position)),
      ctx.bus.on('explosion', (p) => {
        this.propagateGunshot(p.position, 2);
        this.propagateNearMiss(p.position);
      }),
      // The player has no death event of its own, so deaths are inferred from
      // the damage that reduced it to zero. Without this the scoreboard would
      // report `deaths: 0` forever and the streak would never reset.
      ctx.bus.on('damage:taken', (p) => {
        if (p.health > 0) return;
        this.deaths++;
        this.streak = 0;
        ctx.bus.emit('game:score', { kills: this.kills, deaths: this.deaths, streak: this.streak });
      }),
    );
  }

  dispose(): void {
    for (const u of this.unsubscribe) u();
    this.unsubscribe = [];
    for (const e of this.pool) e.dispose();
    this.pool.length = 0;
    this.byId.clear();
  }

  // ── public API for the weapon system ─────────────────────────────────────────

  /**
   * Apply damage to an entity.
   *
   * The headshot multiplier is the *caller's* responsibility (the weapon owns
   * the damage model); `headshot` here drives feedback and the ragdoll impulse.
   *
   * @returns true if this hit killed the target.
   */
  damage(id: EntityId, amount: number, headshot: boolean, point: THREE.Vector3): boolean {
    const enemy = this.byId.get(id);
    if (!enemy || !enemy.alive) return false;

    // The shooter is the player; their eye is the camera.
    _m0.copy(this.ctx.camera.position);
    const died = enemy.applyDamage(amount, headshot, point, _m0, true);
    // Drained here rather than next frame so the weapon system's hitmarker and
    // the kill feed land on the same frame as the shot.
    if (died) this.drainKillEvent(id, enemy);
    return died;
  }

  /**
   * Emits `entity:killed` for a soldier that has just died, whatever killed it.
   * Idempotent: the flag is cleared, so calling it twice cannot double-count.
   */
  private drainKillEvent(id: EntityId, enemy: Enemy): void {
    if (!enemy.killEventPending) return;
    enemy.killEventPending = false;

    if (enemy.killedByPlayer) {
      this.kills++;
      this.streak++;
    }
    this.byId.delete(id);
    this.ctx.bus.emit('entity:killed', {
      id,
      name: 'Soldier',
      headshot: enemy.killedByHeadshot,
      byPlayer: enemy.killedByPlayer,
    });
    this.ctx.bus.emit('game:score', {
      kills: this.kills,
      deaths: this.deaths,
      streak: this.streak,
    });
    this.scheduleRespawn();
  }

  /**
   * Every live hitbox, for systems that want their own acceleration structure.
   *
   * Enemy colliders are deliberately NOT registered with `IPhysics`: its API
   * offers `addStatic` (wrong — soldiers move) and `addDynamic` (wrong — that
   * hands the transform to the solver, and the animator owns it). Hitboxes are
   * kinematic, which the interface cannot express. Discovery is therefore by
   * scene raycast plus the `userData` tags, exactly as the integration contract
   * specifies; this accessor exists for systems that would rather not traverse.
   */
  getColliders(out: THREE.Object3D[]): number {
    out.length = 0;
    for (const e of this.pool) {
      if (!e.alive) continue;
      for (const h of e.model.hitboxes) out.push(h);
    }
    return out.length;
  }

  /** Nearest living enemy to a point, or undefined. Useful for UI / director. */
  nearestTo(point: THREE.Vector3, maxDistance = Infinity): Enemy | undefined {
    let best: Enemy | undefined;
    let bestD = maxDistance * maxDistance;
    for (const e of this.pool) {
      if (!e.alive) continue;
      const d = e.position.distanceToSquared(point);
      if (d < bestD) {
        bestD = d;
        best = e;
      }
    }
    return best;
  }

  // ── per-frame ──────────────────────────────────────────────────────────────

  update(ctx: EngineContext, frame: FrameContext): void {
    const dt = frame.dt;
    if (dt <= 0) return;

    this.probe.tick(dt);

    if (this.boundsFixTimer > 0) {
      this.boundsFixTimer -= dt;
      if (this.boundsFixTimer <= 0) this.reresolveLevel(ctx);
    }

    // Incremental navigation build. Doing this in slices keeps the ~10k sampling
    // raycasts off the critical path; until it finishes the AI steers instead.
    // Without a physics system each sample walks the scene graph, so the slice
    // has to be far smaller or the first second of play stutters.
    if (this.navStarted && !this.nav.ready) {
      this.nav.buildStep(this.probe.hasPhysics ? NAV_BUILD_BUDGET : NAV_BUILD_BUDGET_SLOW);
    }

    if (!this.player) this.player = ctx.get<Player>('player');
    const target: AITarget = this.player && this.player.alive ? this.player : this.nullTarget;

    // One frustum per frame, shared by every LOD decision.
    _projScreen.multiplyMatrices(ctx.camera.projectionMatrix, ctx.camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(_projScreen);
    const camPos = ctx.camera.position;

    this.updateSpawning(dt);

    this.steppingEnemies = true;
    for (let i = 0; i < this.pool.length; i++) {
      const e = this.pool[i];
      if (e.phase === 'inactive') continue;

      const d = e.position.distanceTo(camPos);
      e.viewDistance = d;

      let lod = d < LOD1_DISTANCE ? 0 : d < LOD2_DISTANCE ? 1 : 2;
      // Off-screen soldiers drop a tier: the player cannot see the animation
      // and does not need frame-accurate decisions from something behind them.
      if (lod < 2 && d > 12) {
        _m1.set(e.position.x, e.position.y + 1.0, e.position.z);
        if (!this.frustum.containsPoint(_m1)) lod = Math.min(2, lod + 1);
      }

      const idBefore = e.id;
      // `update` returns false on the frame a corpse expires and recycles. That
      // is the only place an id is released without going through `damage`, so
      // it is where the lookup map must be pruned or it leaks dead entities.
      const live = e.update(dt, target, lod);
      // Deaths not caused by the player (falling out of the map) surface here.
      if (idBefore >= 0 && e.killEventPending) this.drainKillEvent(idBefore, e);
      if (!live && idBefore >= 0) this.byId.delete(idBefore);
    }
    this.steppingEnemies = false;

    // Squad awareness: anyone who fired this frame is heard by nearby allies.
    for (let i = 0; i < this.pool.length; i++) {
      const e = this.pool[i];
      if (!e.alive || e.firedShots === 0) continue;
      for (let j = 0; j < this.pool.length; j++) {
        if (i === j) continue;
        const o = this.pool[j];
        if (!o.alive) continue;
        if (o.position.distanceToSquared(e.position) < SQUAD_HEAR_RADIUS * SQUAD_HEAR_RADIUS) {
          o.hearGunshot(e.position, 0.6);
        }
      }
    }

    // Pairwise separation. n ≤ 16, so the quadratic loop is ~120 cheap tests.
    for (let i = 0; i < this.pool.length; i++) {
      const a = this.pool[i];
      if (!a.alive) continue;
      for (let j = i + 1; j < this.pool.length; j++) {
        a.separateFrom(this.pool[j], dt);
      }
    }
  }

  /**
   * Re-derive level bounds and spawn points now that every system has
   * registered. Only restarts the (expensive) nav sample if the bounds actually
   * moved materially, so the common case where `init` already saw the level
   * costs one Box3 comparison.
   */
  private reresolveLevel(ctx: EngineContext): void {
    getLevelBounds(ctx, _bounds);
    const moved =
      _bounds.min.distanceToSquared(this.builtBounds.min) > 4 ||
      _bounds.max.distanceToSquared(this.builtBounds.max) > 4;
    if (!moved) return;

    this.builtBounds.copy(_bounds);
    this.nav.begin(this.probe, _bounds);
    this.spawnCount = getSpawnPoints(ctx, this.probe, this.spawnPoints);
  }

  // ── spawning ───────────────────────────────────────────────────────────────

  private scheduleRespawn(): void {
    this.pending.push({
      timer: RESPAWN_DELAY_MIN + Math.random() * (RESPAWN_DELAY_MAX - RESPAWN_DELAY_MIN),
    });
  }

  private updateSpawning(dt: number): void {
    this.spawnTimer -= dt;

    // Top up if attrition or a despawn left us short.
    const alive = this.aliveCount;
    if (alive + this.pending.length < TARGET_ALIVE) this.scheduleRespawn();

    if (this.pending.length === 0) return;
    const next = this.pending[0];
    next.timer -= dt;
    if (next.timer > 0 || this.spawnTimer > 0) return;
    if (alive >= TARGET_ALIVE) return;

    if (this.trySpawn()) {
      this.pending.shift();
      this.spawnTimer = SPAWN_INTERVAL;
    } else {
      // No valid spot right now (player standing on the only spawn); retry soon.
      next.timer = 0.5;
    }
  }

  private acquire(): Enemy | undefined {
    for (const e of this.pool) {
      if (e.phase === 'inactive') return e;
    }
    if (this.pool.length >= POOL_SIZE) return undefined;
    const e = new Enemy(
      {
        probe: this.probe,
        nav: this.nav,
        bus: this.ctx.bus,
        scene: this.ctx.scene,
        materials: this.ctx.get<IMaterialLibrary>('materials'),
      },
      this.pool.length * 977 + 13,
    );
    this.pool.push(e);
    return e;
  }

  /**
   * Chooses a spawn point that is far from the player and, ideally, out of
   * their view — materialising a soldier in someone's crosshair is the single
   * most immersion-breaking thing a spawner can do.
   */
  private trySpawn(): boolean {
    if (this.spawnCount === 0) return false;

    const playerPos = this.player ? this.player.position : _m0.set(0, 0, 0);
    let best = -1;
    let bestScore = -Infinity;
    // Fallback bookkeeping, so a level too small to satisfy MIN_SPAWN_DISTANCE
    // still fields enemies instead of silently fielding none forever.
    let farthest = -1;
    let farthestD = -1;
    let farthestHidden = -1;
    let farthestHiddenD = -1;

    for (let i = 0; i < this.spawnCount; i++) {
      const p = this.spawnPoints[i];
      const d = p.distanceTo(playerPos);
      _m1.set(p.x, p.y + 1.2, p.z);
      const visible = this.frustum.containsPoint(_m1);

      if (d > farthestD) {
        farthestD = d;
        farthest = i;
      }
      if (!visible && d > farthestHiddenD) {
        farthestHiddenD = d;
        farthestHidden = i;
      }

      if (d < MIN_SPAWN_DISTANCE || d > MAX_SPAWN_DISTANCE) continue;

      let score = 20 - Math.abs(d - 45) * 0.25;
      if (visible) score -= 14; // in the player's view: heavily penalised
      // Spread the squad out rather than stacking them on one door.
      for (const other of this.pool) {
        if (!other.alive) continue;
        if (other.position.distanceToSquared(p) < 36) score -= 5;
      }
      score += Math.random() * 3; // break ties so waves are not identical

      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }

    // Prefer a hidden distant point, then the most distant point of all. Never
    // return "no spawn" just because the map is smaller than the ideal ring.
    if (best < 0) best = farthestHidden >= 0 && farthestHiddenD > 8 ? farthestHidden : farthest;
    if (best < 0) return false;
    const spot = this.spawnPoints[best];

    // Acquire only once a destination is settled. Constructing the Enemy first
    // and then bailing out leaves a pooled soldier parked at the world origin
    // with a live set of raycastable hitboxes.
    const enemy = this.acquire();
    if (!enemy) return false;

    // Face roughly toward the player so they do not spawn staring at a wall.
    const yaw = Math.atan2(playerPos.x - spot.x, playerPos.z - spot.z);
    const id = this.nextId++;
    enemy.spawn(id, spot, yaw);
    this.byId.set(id, enemy);
    this.ctx.bus.emit('entity:spawned', { id, position: spot.clone() });
    return true;
  }

  // ── stimuli propagation ────────────────────────────────────────────────────

  private propagateGunshot(origin: THREE.Vector3, loudness: number): void {
    for (const e of this.pool) {
      if (e.alive) e.hearGunshot(origin, loudness);
    }
  }

  private propagateNearMiss(point: THREE.Vector3): void {
    if (this.steppingEnemies) return; // our own outgoing fire; see `steppingEnemies`
    const r2 = NEAR_MISS_RADIUS * NEAR_MISS_RADIUS;
    for (const e of this.pool) {
      if (e.alive && e.position.distanceToSquared(point) < r2) e.hearNearMiss(point);
    }
  }
}
