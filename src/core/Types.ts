/**
 * NOISEFIELD — Shared contracts.
 *
 * This file is the integration boundary between subsystems. Every subsystem is
 * authored against these interfaces so modules can be developed independently
 * and swapped without touching call sites.
 *
 * This file is the integration contract between subsystems. Changing a type
 * here affects every subsystem that implements or consumes it, so treat edits
 * as an API change rather than a local one.
 */

import type * as THREE from 'three';

// ─────────────────────────────────────────────────────────────────────────────
// Engine lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/** Per-frame timing, produced once per tick and shared by all systems. */
export interface FrameContext {
  /** Seconds since previous frame, clamped to avoid spiral-of-death. */
  readonly dt: number;
  /** Unclamped seconds since previous frame. */
  readonly rawDt: number;
  /** Seconds since engine start. */
  readonly elapsed: number;
  /** Monotonically increasing integer frame counter. */
  readonly frame: number;
}

/**
 * A subsystem participating in the engine lifecycle. All hooks are optional so
 * implementations only pay for what they use.
 */
export interface System {
  readonly name: string;
  /** Called once, after the renderer and scene exist. May be async. */
  init?(ctx: EngineContext): void | Promise<void>;
  /** Fixed-rate simulation step. Called 0..n times per frame. */
  fixedUpdate?(ctx: EngineContext, step: number): void;
  /** Variable-rate update, called exactly once per frame before render. */
  update?(ctx: EngineContext, frame: FrameContext): void;
  /** Called after the main render, for systems that need post-render work. */
  lateUpdate?(ctx: EngineContext, frame: FrameContext): void;
  /** Viewport size changed. */
  resize?(width: number, height: number): void;
  /** Release GPU/CPU resources. */
  dispose?(): void;
}

/** Everything a subsystem is allowed to reach for. */
export interface EngineContext {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  /** Separate scene rendered on top with a cleared depth buffer (viewmodel). */
  readonly viewScene: THREE.Scene;
  readonly viewCamera: THREE.PerspectiveCamera;
  readonly bus: IEventBus;
  readonly input: IInput;
  readonly settings: QualitySettings;
  readonly width: number;
  readonly height: number;
  /** Registry for cross-system lookup by name. Prefer events where possible. */
  get<T extends System>(name: string): T | undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// Events
// ─────────────────────────────────────────────────────────────────────────────

/** Payloads for every cross-system event. Extend here, never inline. */
export interface GameEvents {
  'weapon:fire': { origin: THREE.Vector3; direction: THREE.Vector3; weapon: WeaponId; seed: number };
  'weapon:dryfire': { weapon: WeaponId };
  'weapon:reload:start': { weapon: WeaponId; duration: number; tactical: boolean };
  'weapon:reload:end': { weapon: WeaponId };
  'weapon:switch': { from: WeaponId | null; to: WeaponId };
  'weapon:ads': { active: boolean; weapon: WeaponId };
  'weapon:ammo': { mag: number; magSize: number; reserve: number; weapon: WeaponId };

  'impact': ImpactEvent;
  'tracer': { from: THREE.Vector3; to: THREE.Vector3; speed: number };
  'shell:eject': { position: THREE.Vector3; velocity: THREE.Vector3; caliber: Caliber };

  'damage:dealt': { target: EntityId; amount: number; headshot: boolean; position: THREE.Vector3 };
  'damage:taken': { amount: number; direction: THREE.Vector3; health: number };
  'entity:killed': { id: EntityId; name: string; headshot: boolean; byPlayer: boolean };
  'entity:spawned': { id: EntityId; position: THREE.Vector3 };

  'player:land': { velocity: number };
  'player:footstep': { position: THREE.Vector3; surface: SurfaceType; sprinting: boolean };
  'player:slide:start': Record<string, never>;
  'player:slide:end': Record<string, never>;
  'player:state': PlayerStateSnapshot;

  'explosion': { position: THREE.Vector3; radius: number; power: number };

  'camera:shake': { amplitude: number; duration: number; frequency?: number };
  'camera:kick': { pitch: number; yaw: number; duration: number };

  'game:start': Record<string, never>;
  'game:pause': { paused: boolean };
  'game:score': { kills: number; deaths: number; streak: number };
  'ui:notify': { text: string; kind: 'info' | 'kill' | 'warn' };
}

export type EventName = keyof GameEvents;
export type EventHandler<K extends EventName> = (payload: GameEvents[K]) => void;

export interface IEventBus {
  on<K extends EventName>(event: K, handler: EventHandler<K>): () => void;
  once<K extends EventName>(event: K, handler: EventHandler<K>): () => void;
  off<K extends EventName>(event: K, handler: EventHandler<K>): void;
  emit<K extends EventName>(event: K, payload: GameEvents[K]): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain vocabulary
// ─────────────────────────────────────────────────────────────────────────────

export type EntityId = number;
export type WeaponId = 'ar_vector' | 'smg_wasp' | 'sniper_longbow' | 'shotgun_breach' | 'pistol_sidearm';

export type Caliber = '5.56' | '9mm' | '.338' | '12ga';

/**
 * Surface classification. Drives impact VFX, decal choice, footstep audio and
 * ballistic penetration. Every collider in the world must map to one of these.
 */
export type SurfaceType =
  | 'concrete'
  | 'metal'
  | 'wood'
  | 'sand'
  | 'dirt'
  | 'glass'
  | 'water'
  | 'flesh'
  | 'foliage'
  | 'fabric'
  | 'rubber';

export interface ImpactEvent {
  position: THREE.Vector3;
  /** Unit surface normal at the point of impact. */
  normal: THREE.Vector3;
  /** Incoming unit direction of the projectile. */
  incoming: THREE.Vector3;
  surface: SurfaceType;
  /** 0..1 relative energy, drives spark count / decal size / audio gain. */
  energy: number;
}

export interface PlayerStateSnapshot {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  grounded: boolean;
  sprinting: boolean;
  crouching: boolean;
  sliding: boolean;
  ads: boolean;
  health: number;
  maxHealth: number;
  stamina: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Input
// ─────────────────────────────────────────────────────────────────────────────

export type ActionName =
  | 'forward' | 'back' | 'left' | 'right'
  | 'jump' | 'crouch' | 'sprint' | 'slide'
  | 'fire' | 'ads' | 'reload' | 'melee' | 'grenade'
  | 'weapon1' | 'weapon2' | 'inspect' | 'pause';

export interface IInput {
  /** Held this frame. */
  isDown(action: ActionName): boolean;
  /** Transitioned up→down this frame. */
  pressed(action: ActionName): boolean;
  /** Transitioned down→up this frame. */
  released(action: ActionName): boolean;
  /** Accumulated mouse delta for this frame, in radians (already sensitivity-scaled). */
  readonly lookDelta: { x: number; y: number };
  readonly pointerLocked: boolean;
  requestPointerLock(): void;
  /** Consume per-frame edge state. Called by the engine at end of frame. */
  endFrame(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Quality / settings
// ─────────────────────────────────────────────────────────────────────────────

export type QualityTier = 'low' | 'medium' | 'high' | 'ultra';

export interface QualitySettings {
  tier: QualityTier;
  /** Render scale multiplier applied to the drawing buffer. */
  renderScale: number;
  shadows: boolean;
  shadowMapSize: number;
  /** Number of cascaded shadow map splits. */
  shadowCascades: number;
  ssao: boolean;
  ssaoQuality: 'low' | 'medium' | 'high';
  bloom: boolean;
  motionBlur: boolean;
  volumetrics: boolean;
  antialias: 'none' | 'fxaa' | 'smaa' | 'taa';
  anisotropy: number;
  /** Max simultaneous GPU particles. */
  particleBudget: number;
  /** Max decals retained before recycling. */
  decalBudget: number;
  fov: number;
  /** Multiplier applied to fov when aiming down sights. */
  adsFovScale: number;
  sensitivity: number;
  masterVolume: number;
}

export const DEFAULT_SETTINGS: QualitySettings = {
  tier: 'ultra',
  renderScale: 1.0,
  shadows: true,
  shadowMapSize: 2048,
  shadowCascades: 3,
  ssao: true,
  ssaoQuality: 'high',
  bloom: true,
  motionBlur: true,
  volumetrics: true,
  antialias: 'taa',
  anisotropy: 16,
  particleBudget: 20000,
  decalBudget: 256,
  fov: 80,
  adsFovScale: 0.62,
  sensitivity: 0.0022,
  masterVolume: 0.8,
};

// ─────────────────────────────────────────────────────────────────────────────
// Material / world description
// ─────────────────────────────────────────────────────────────────────────────

/** A fully-populated PBR material set produced by the texture forge. */
export interface PBRTextureSet {
  map: THREE.Texture;
  normalMap: THREE.Texture;
  roughnessMap: THREE.Texture;
  aoMap?: THREE.Texture;
  metalnessMap?: THREE.Texture;
  displacementMap?: THREE.Texture;
}

export interface MaterialDescriptor {
  id: string;
  surface: SurfaceType;
  /** World-space size in metres that one texture tile covers. */
  tileMeters: number;
  material: THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial;
}

/**
 * Contract for the procedural material library. Consumers ask for a surface and
 * receive a shared, cached material instance.
 */
export interface IMaterialLibrary extends System {
  get(id: string): THREE.Material;
  /** Returns a material appropriate for a surface class. */
  forSurface(surface: SurfaceType): THREE.Material;
  /** Surface classification for a mesh, used by ballistics/VFX. */
  surfaceOf(object: THREE.Object3D): SurfaceType;
  /** Tag an object so `surfaceOf` can resolve it later. */
  tag(object: THREE.Object3D, surface: SurfaceType): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Physics
// ─────────────────────────────────────────────────────────────────────────────

export interface RaycastHit {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  distance: number;
  surface: SurfaceType;
  /** Entity struck, if the collider belongs to one. */
  entity?: EntityId;
  /** True when the hit region is a head hitbox. */
  headshot?: boolean;
  object?: THREE.Object3D;
}

export interface IPhysics extends System {
  raycast(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number, filter?: PhysicsFilter): RaycastHit | null;
  /** All hits along a ray, sorted near→far. Used for wall penetration. */
  raycastAll(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number, filter?: PhysicsFilter): RaycastHit[];
  /** Sphere-cast used by the character controller and grenades. */
  sphereCast(origin: THREE.Vector3, radius: number, direction: THREE.Vector3, maxDistance: number): RaycastHit | null;
  /** Register static level geometry for collision. */
  addStatic(object: THREE.Object3D, surface: SurfaceType): void;
  /** Register a dynamic body; returns a handle used to drive/read it. */
  addDynamic(object: THREE.Object3D, opts: DynamicBodyOptions): number;
  removeBody(handle: number): void;
  applyImpulse(handle: number, impulse: THREE.Vector3, point?: THREE.Vector3): void;
  /** Radial impulse, used by explosions. */
  applyRadialImpulse(center: THREE.Vector3, radius: number, power: number): void;
}

export interface PhysicsFilter {
  /** Skip these objects (e.g. the shooter's own collider). */
  exclude?: THREE.Object3D[];
  /** Only consider entity colliders. */
  entitiesOnly?: boolean;
}

export interface DynamicBodyOptions {
  shape: 'box' | 'sphere' | 'capsule' | 'convex';
  mass: number;
  restitution?: number;
  friction?: number;
  linearDamping?: number;
  angularDamping?: number;
  surface?: SurfaceType;
}

// ─────────────────────────────────────────────────────────────────────────────
// Weapons
// ─────────────────────────────────────────────────────────────────────────────

export interface RecoilPattern {
  /** Vertical kick in radians per shot, indexed by shot number in the burst. */
  vertical: number[];
  /** Horizontal kick in radians per shot; sign encodes direction. */
  horizontal: number[];
  /** Seconds for the view to recover toward centre. */
  recoverTime: number;
  /** How much of the kick is recovered automatically, 0..1. */
  recoverFraction: number;
}

export interface WeaponSpec {
  id: WeaponId;
  displayName: string;
  caliber: Caliber;
  /** Rounds per minute. */
  rpm: number;
  damage: number;
  headshotMultiplier: number;
  /** Damage falloff start/end in metres. */
  falloffStart: number;
  falloffEnd: number;
  /** Damage multiplier at and beyond falloffEnd. */
  falloffMin: number;
  magSize: number;
  reserveAmmo: number;
  reloadTime: number;
  /** Reload time when a round remains chambered. */
  reloadTacticalTime: number;
  adsTime: number;
  /** Cone half-angle in radians at hip and at ADS. */
  spreadHip: number;
  spreadAds: number;
  /** Additional spread accumulated per shot, and decay per second. */
  spreadPerShot: number;
  spreadDecay: number;
  recoil: RecoilPattern;
  /** Muzzle velocity in m/s; drives tracer speed and travel time. */
  muzzleVelocity: number;
  /** Number of pellets per trigger pull (>1 for shotguns). */
  pellets: number;
  /** Rounds fired per trigger pull for burst weapons; 0 = full auto. */
  burst: number;
  /** Wall-penetration power, 0..1. */
  penetration: number;
  /** Viewmodel offsets in view space. */
  hipPosition: [number, number, number];
  adsPosition: [number, number, number];
  fovScale: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// VFX
// ─────────────────────────────────────────────────────────────────────────────

export interface IParticles extends System {
  /** Emit a burst. `kind` selects a preconfigured emitter profile. */
  burst(kind: ParticleKind, position: THREE.Vector3, normal: THREE.Vector3, intensity: number): void;
  /** Emit a single long-lived particle (smoke puff, ember). */
  emit(kind: ParticleKind, position: THREE.Vector3, velocity: THREE.Vector3, life: number, size: number): void;
  readonly activeCount: number;
}

export type ParticleKind =
  | 'spark' | 'dust' | 'smoke' | 'blood' | 'debris'
  | 'muzzleSmoke' | 'shell' | 'ember' | 'glass' | 'water' | 'explosion';

export interface IDecals extends System {
  add(position: THREE.Vector3, normal: THREE.Vector3, surface: SurfaceType, size: number): void;
  clear(): void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio
// ─────────────────────────────────────────────────────────────────────────────

export type SoundId =
  | 'fire_ar' | 'fire_smg' | 'fire_sniper' | 'fire_shotgun' | 'fire_pistol'
  | 'fire_distant' | 'tail_indoor' | 'tail_outdoor'
  | 'reload_magout' | 'reload_magin' | 'reload_charge' | 'dryfire'
  | 'impact_concrete' | 'impact_metal' | 'impact_wood' | 'impact_sand'
  | 'impact_glass' | 'impact_flesh' | 'impact_water'
  | 'footstep_concrete' | 'footstep_sand' | 'footstep_metal' | 'footstep_wood'
  | 'shell_drop' | 'hitmarker' | 'headshot' | 'kill'
  | 'ads_in' | 'ads_out' | 'slide' | 'jump' | 'land'
  | 'explosion' | 'whizby' | 'damage' | 'lowhealth' | 'heartbeat';

export interface IAudio extends System {
  /** Non-positional UI/feedback sound. */
  play(id: SoundId, gain?: number, rate?: number): void;
  /** Positional world sound. */
  play3D(id: SoundId, position: THREE.Vector3, gain?: number, rate?: number): void;
  /** Update listener from the camera. */
  setListener(position: THREE.Vector3, forward: THREE.Vector3, up: THREE.Vector3, velocity: THREE.Vector3): void;
  /** Ducking / temporary deafening after loud events. */
  duck(amount: number, duration: number): void;
  resume(): Promise<void>;
  readonly ready: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────

export interface IHUD extends System {
  setVisible(v: boolean): void;
}
