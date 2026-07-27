import * as THREE from 'three';
import type { EngineContext, EntityId, FrameContext, IAudio, System } from './Types';
import type { Physics } from '../physics/Physics';
import type { Level } from '../world/Level';
import type { EnemyManager } from '../entities/EnemyManager';
import type { Player } from '../entities/Player';

/**
 * Integration seams between independently-authored subsystems.
 *
 * Every subsystem was built against `Types.ts` alone, which means a handful of
 * couplings exist that no single system could own. They live here rather than
 * being smuggled into a subsystem that would then depend on another:
 *
 *  1. Level geometry → physics static collision. The level does not know about
 *     physics and physics does not know about the level; something has to
 *     introduce them.
 *  2. Enemy hitboxes → physics entity colliders. `EnemyManager` deliberately
 *     refuses to self-register (its hitboxes are kinematic — neither `addStatic`
 *     nor `addDynamic` expresses that), but `Ballistics` resolves targets via
 *     `physics.raycast`. Without this bridge enemies are literally unhittable.
 *  3. `damage:dealt` → `EnemyManager.damage`. Ballistics announces damage on the
 *     bus but has no reference to the entity registry, so nothing applies it.
 *     Without this bridge enemies are immortal.
 *  4. Camera → audio listener. `IAudio.setListener` must be driven every frame
 *     and fails silently if it is not (the listener stays at the origin).
 */
export class Bridge implements System {
  readonly name = 'bridge';

  private physics?: Physics;
  private enemies?: EnemyManager;
  private player?: Player;
  private audio?: IAudio;

  /** Live hitbox → physics collider handle. */
  private readonly registered = new Map<THREE.Object3D, number>();
  private readonly scratchColliders: THREE.Object3D[] = [];
  private readonly stale: THREE.Object3D[] = [];
  private reconcileTimer = 0;

  private readonly listenerPos = new THREE.Vector3();
  private readonly listenerFwd = new THREE.Vector3();
  private readonly listenerUp = new THREE.Vector3();
  private readonly listenerVel = new THREE.Vector3();

  private unsubscribe: Array<() => void> = [];

  init(ctx: EngineContext): void {
    this.physics = ctx.get<Physics>('physics');
    this.enemies = ctx.get<EnemyManager>('enemies');
    this.player = ctx.get<Player>('player');
    this.audio = ctx.get<IAudio>('audio');

    this.registerLevelCollision(ctx);

    // Seam 3: apply the damage the weapon system announces.
    const enemies = this.enemies;
    if (enemies) {
      this.unsubscribe.push(
        ctx.bus.on('damage:dealt', (p) => {
          // `damage:dealt` is also emitted for hits on non-entity geometry in
          // some paths; `target` is only meaningful when an entity was struck.
          if (typeof p.target !== 'number') return;
          if (!Number.isFinite(p.amount) || p.amount <= 0) return;
          enemies.damage(p.target as EntityId, p.amount, p.headshot, p.position);
        }),
      );
    }
  }

  /**
   * Seam 1. `BVHWorld.addStatic` resolves `userData.surface` per mesh and
   * expands `InstancedMesh`, so a single default surface is enough here — the
   * per-mesh tags the world system applied do the real work.
   */
  private registerLevelCollision(ctx: EngineContext): void {
    const level = ctx.get<Level>('level');
    const physics = this.physics;
    if (!level || !physics) {
      console.warn('[Bridge] level or physics missing; world collision disabled');
      return;
    }
    let count = 0;
    for (const obj of level.collidables) {
      physics.addStatic(obj, 'concrete');
      count++;
    }
    console.info(`[Bridge] registered ${count} static collision groups`);
  }

  update(ctx: EngineContext, frame: FrameContext): void {
    this.syncEnemyColliders(frame);
    this.updateListener();
  }

  /**
   * Seam 2. Reconcile the set of live hitboxes against registered colliders.
   *
   * Enemies spawn and die infrequently relative to the frame rate, so a full
   * reconcile every frame would be wasted work; instead it runs on a ~10Hz
   * timer and immediately whenever the hitbox count changes (which catches
   * every spawn and death on the frame it happens, so a soldier is never
   * briefly unhittable or briefly shootable after death).
   */
  private syncEnemyColliders(frame: FrameContext): void {
    const enemies = this.enemies;
    const physics = this.physics;
    if (!enemies || !physics) return;

    const count = enemies.getColliders(this.scratchColliders);
    this.reconcileTimer -= frame.dt;
    if (count === this.registered.size && this.reconcileTimer > 0) return;
    this.reconcileTimer = 0.1;

    // Add newcomers.
    for (let i = 0; i < count; i++) {
      const obj = this.scratchColliders[i];
      if (this.registered.has(obj)) continue;
      const eid = obj.userData['entityId'];
      const handle = physics.addEntityCollider(obj, {
        shape: obj.userData['hitbox'] === 'head' ? 'sphere' : 'capsule',
        entity: typeof eid === 'number' ? (eid as EntityId) : undefined,
        hitbox: typeof obj.userData['hitbox'] === 'string' ? (obj.userData['hitbox'] as string) : undefined,
        surface: 'flesh',
      });
      this.registered.set(obj, handle);
    }

    // Retire hitboxes that are no longer live. Collected first because deleting
    // from a Map while iterating it is legal but easy to get wrong.
    if (this.registered.size !== count) {
      this.stale.length = 0;
      for (const obj of this.registered.keys()) {
        let live = false;
        for (let i = 0; i < count; i++) {
          if (this.scratchColliders[i] === obj) {
            live = true;
            break;
          }
        }
        if (!live) this.stale.push(obj);
      }
      for (const obj of this.stale) {
        const handle = this.registered.get(obj);
        if (handle !== undefined) physics.removeEntityCollider(handle);
        this.registered.delete(obj);
      }
    }
  }

  /**
   * Seam 4. Drive the audio listener from the camera. Velocity comes from the
   * player rather than a finite-difference of the camera position, because the
   * camera also carries recoil and bob — differentiating it would feed doppler
   * a violently noisy signal on every shot.
   */
  private updateListener(): void {
    const audio = this.audio;
    if (!audio) return;
    const cam = this.camera;
    if (!cam) return;

    cam.getWorldPosition(this.listenerPos);
    cam.getWorldDirection(this.listenerFwd);
    this.listenerUp.set(0, 1, 0).applyQuaternion(cam.quaternion);
    if (this.player) this.listenerVel.copy(this.player.velocity);
    else this.listenerVel.set(0, 0, 0);

    audio.setListener(this.listenerPos, this.listenerFwd, this.listenerUp, this.listenerVel);
  }

  private camera: THREE.Camera | null = null;

  /** Called by main once the engine exists; avoids a per-frame ctx lookup. */
  attachCamera(camera: THREE.Camera): void {
    this.camera = camera;
  }

  dispose(): void {
    for (const u of this.unsubscribe) u();
    this.unsubscribe = [];
    const physics = this.physics;
    if (physics) {
      for (const handle of this.registered.values()) physics.removeEntityCollider(handle);
    }
    this.registered.clear();
  }
}
