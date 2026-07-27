/**
 * NOISEFIELD — impact, explosion and incidental VFX orchestration.
 *
 * This system owns *art direction*, not simulation: it listens for gameplay
 * events and translates them into particle emissions on {@link Particles},
 * pooled dynamic lights, decal requests and camera shake.
 *
 * WHY PER-SURFACE EFFECTS MATTER
 * ------------------------------
 * A player reads the world through its impacts. Sparks mean metal, grey dust
 * means stone, splinters mean wood, a tan plume means sand. Getting that
 * vocabulary right is worth more than any amount of extra particle count, so
 * every SurfaceType gets a hand-tuned recipe rather than a recoloured generic
 * puff. Two rules are enforced everywhere:
 *   • Concrete NEVER sparks. Sparks on stone is the classic tell of a generic
 *     impact system and instantly reads as cheap.
 *   • Sparks are ALWAYS velocity-stretched and blackbody-cooled. Round, evenly
 *     coloured spark dots look like confetti.
 *
 * BUDGET DISCIPLINE
 * -----------------
 * Every emission is multiplied by {@link Impacts.budgetScale}, which folds in
 * the quality tier, distance from the camera and whether the event is even
 * roughly in view. Off-screen impacts at 40 m emit a token amount; impacts
 * beyond ~140 m emit nothing. Nothing here allocates after `init`.
 *
 * DEFENSIVE INTEGRATION
 * ---------------------
 * `physics`, `decals` and `particles` are all resolved lazily through
 * `ctx.get` and every optional call is guarded. If the physics system never
 * ships, ricochet ground-scatter simply does not happen; if the decals system
 * never ships, no decals are requested. Nothing throws.
 *
 */

import * as THREE from 'three';
import type {
  EngineContext,
  FrameContext,
  GameEvents,
  IDecals,
  ImpactEvent,
  IPhysics,
  QualityTier,
  SurfaceType,
  System,
} from '../core/Types';
import { AtlasCell } from './textures';
import { PBatch, PFlags, Particles, randomCone, Rng } from './Particles';

const TAU = Math.PI * 2;

/**
 * Palette. Constructed from sRGB hex, so `.r/.g/.b` are already in the
 * renderer's linear working space (three's ColorManagement is on by default)
 * and can be written straight into particle attributes.
 *
 * Art direction: sun-baked North-African compound at golden hour.
 */
const C = {
  sand: new THREE.Color(0xc2a878),
  sunlight: new THREE.Color(0xffd2a1),
  shadow: new THREE.Color(0x1a2028),
  concreteDust: new THREE.Color(0xa39a8c),
  concreteChip: new THREE.Color(0x857f76),
  woodDust: new THREE.Color(0xc0a475),
  woodChip: new THREE.Color(0x9d7844),
  dirt: new THREE.Color(0x8d7551),
  smoke: new THREE.Color(0x8d8880),
  sparkWarm: new THREE.Color(0xffb060),
  blood: new THREE.Color(0x6e0c07),
  bloodMist: new THREE.Color(0x8e1a10),
  glass: new THREE.Color(0xdaeef7),
  water: new THREE.Color(0xbdd9e6),
  foliage: new THREE.Color(0x6f9145),
  fire: new THREE.Color(0xffa040),
  fabric: new THREE.Color(0xa9997f),
  rubber: new THREE.Color(0x3b3b3d),
} as const;

/** Dynamic light slots per tier. Zero on low: every extra light costs shader work. */
const LIGHT_COUNT: Record<QualityTier, number> = { low: 0, medium: 3, high: 6, ultra: 6 };

// ─────────────────────────────────────────────────────────────────────────────
// Pooled dynamic lights
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A small fixed pool of PointLights for muzzle flashes, spark showers and
 * explosions.
 *
 * WHY FIXED AND ALWAYS-VISIBLE: three bakes the light count into every
 * material's shader program. Adding or hiding a light mid-game triggers a
 * full shader recompile and a multi-frame hitch. The pool therefore creates
 * all its lights up front, leaves them permanently in the scene and
 * permanently `visible`, and animates only `intensity` — which is free.
 */
class LightPool {
  private readonly lights: THREE.PointLight[] = [];
  private readonly ttl: Float32Array;
  private readonly dur: Float32Array;
  private readonly peak: Float32Array;

  constructor(scene: THREE.Scene, count: number) {
    this.ttl = new Float32Array(count);
    this.dur = new Float32Array(count);
    this.peak = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const l = new THREE.PointLight(0xffffff, 0, 8, 2);
      l.castShadow = false;
      l.name = `noisefield.fx.dynLight${i}`;
      scene.add(l);
      this.lights.push(l);
    }
  }

  get size(): number {
    return this.lights.length;
  }

  /**
   * Lights `position` for `duration` seconds. Steals the slot with the least
   * remaining life when the pool is saturated, so the newest (and therefore
   * most gameplay-relevant) event always wins.
   */
  flash(
    position: THREE.Vector3,
    color: THREE.Color,
    intensity: number,
    duration: number,
    distance: number,
  ): void {
    const n = this.lights.length;
    if (n === 0) return;
    let idx = 0;
    let worst = Infinity;
    for (let i = 0; i < n; i++) {
      const t = this.ttl[i];
      if (t <= 0) {
        idx = i;
        worst = -1;
        break;
      }
      if (t < worst) {
        worst = t;
        idx = i;
      }
    }
    const l = this.lights[idx];
    l.position.copy(position);
    l.color.copy(color);
    l.distance = distance;
    l.intensity = intensity;
    this.peak[idx] = intensity;
    this.dur[idx] = Math.max(0.01, duration);
    this.ttl[idx] = this.dur[idx];
  }

  update(dt: number): void {
    for (let i = 0; i < this.lights.length; i++) {
      if (this.ttl[i] <= 0) continue;
      this.ttl[i] -= dt;
      if (this.ttl[i] <= 0) {
        this.ttl[i] = 0;
        this.lights[i].intensity = 0;
        continue;
      }
      // Quadratic falloff: a muzzle flash should snap off, not dim linearly.
      const k = this.ttl[i] / this.dur[i];
      this.lights[i].intensity = this.peak[i] * k * k;
    }
  }

  dispose(): void {
    for (const l of this.lights) l.removeFromParent();
    this.lights.length = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// System
// ─────────────────────────────────────────────────────────────────────────────

export class Impacts implements System {
  readonly name = 'impacts';

  private ctx: EngineContext | null = null;
  private particles: Particles | null = null;
  private decals: IDecals | null = null;
  private physics: IPhysics | null = null;
  /** Latches after the first throw so a broken system is probed exactly once. */
  private physicsUsable = true;
  private decalsUsable = true;

  private lights: LightPool | null = null;
  private readonly rng = new Rng(0x51f3a7);
  private readonly unsubscribe: Array<() => void> = [];

  // Scratch — this system must not allocate during play.
  private readonly dir = new THREE.Vector3();
  private readonly refl = new THREE.Vector3();
  private readonly tmpA = new THREE.Vector3();
  private readonly tmpB = new THREE.Vector3();
  private readonly tmpC = new THREE.Vector3();
  private readonly down = new THREE.Vector3(0, -1, 0);
  private readonly up = new THREE.Vector3(0, 1, 0);
  private readonly flashColor = new THREE.Color();
  /** Reused camera-shake payload; explosions must not allocate. */
  private readonly shakePayload = { amplitude: 0, duration: 0, frequency: 22 };

  /** Remembers the last flesh hit so `entity:killed` can punctuate it. */
  private readonly lastFlesh = new THREE.Vector3();
  private readonly lastFleshNormal = new THREE.Vector3(0, 1, 0);
  private lastFleshTime = -1e9;
  private time = 0;

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    this.lights = new LightPool(ctx.scene, LIGHT_COUNT[ctx.settings.tier] ?? 4);

    const bus = ctx.bus;
    this.unsubscribe.push(
      bus.on('impact', this.onImpact),
      bus.on('explosion', this.onExplosion),
      bus.on('weapon:fire', this.onWeaponFire),
      bus.on('player:footstep', this.onFootstep),
      bus.on('entity:killed', this.onKilled),
    );
  }

  update(_ctx: EngineContext, frame: FrameContext): void {
    this.time = frame.elapsed;
    this.lights?.update(frame.dt);
  }

  dispose(): void {
    for (const off of this.unsubscribe) off();
    this.unsubscribe.length = 0;
    this.lights?.dispose();
    this.lights = null;
  }

  // ── lazy system resolution ───────────────────────────────────────────────────

  private get P(): Particles | null {
    if (!this.particles) this.particles = this.ctx?.get<Particles>('particles') ?? null;
    return this.particles;
  }

  private get physicsSystem(): IPhysics | null {
    if (!this.physicsUsable) return null;
    if (!this.physics) this.physics = this.ctx?.get<IPhysics>('physics') ?? null;
    return this.physics;
  }

  /**
   * Emission multiplier for an event at `pos`: folds together quality tier,
   * distance attenuation and a rough in-view test. Returns 0 when the effect
   * is not worth spawning at all.
   */
  private budgetScale(pos: THREE.Vector3): number {
    const ctx = this.ctx;
    if (!ctx) return 0;
    const cam = ctx.camera;
    const dx = pos.x - cam.position.x;
    const dy = pos.y - cam.position.y;
    const dz = pos.z - cam.position.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > 140 * 140) return 0;

    const dist = Math.sqrt(d2) || 1e-3;
    cam.getWorldDirection(this.tmpB);
    const facing = (dx * this.tmpB.x + dy * this.tmpB.y + dz * this.tmpB.z) / dist;
    // Behind the player and far away: nobody will ever see it.
    if (facing < -0.3 && dist > 16) return 0;

    let s = 1;
    if (dist > 26) s *= Math.max(0.15, 1 - (dist - 26) / 95);
    // Peripheral events still emit — you notice motion in your periphery — but
    // at a fraction of the count.
    if (facing < 0.15) s *= 0.5;
    return s * (this.P?.spawnScale ?? 1);
  }

  /** out = incoming reflected about `normal`. */
  private reflectDir(incoming: THREE.Vector3, normal: THREE.Vector3, out: THREE.Vector3): void {
    const d = incoming.x * normal.x + incoming.y * normal.y + incoming.z * normal.z;
    out.set(incoming.x - 2 * d * normal.x, incoming.y - 2 * d * normal.y, incoming.z - 2 * d * normal.z);
    if (out.lengthSq() < 1e-8) out.copy(normal);
    else out.normalize();
    // Bias the ejecta cone away from the surface so nothing sprays into it.
    out.addScaledVector(normal, 0.35).normalize();
  }

  private requestDecal(position: THREE.Vector3, normal: THREE.Vector3, surface: SurfaceType, size: number): void {
    if (!this.decalsUsable) return;
    if (!this.decals) this.decals = this.ctx?.get<IDecals>('decals') ?? null;
    if (!this.decals) return;
    try {
      this.decals.add(position, normal, surface, size);
    } catch {
      // A misbehaving decal system must never break combat feedback.
      this.decalsUsable = false;
    }
  }

  // ── impact ─────────────────────────────────────────────────────────────────

  private onImpact = (e: ImpactEvent): void => {
    const P = this.P;
    if (!P) return;
    const scale = this.budgetScale(e.position);
    if (scale <= 0.02) return;
    const energy = Math.max(0.05, Math.min(1.5, e.energy));

    // Normalise defensively: a zero-length normal from a degenerate collider
    // would make every cone emitter produce NaNs.
    this.tmpC.copy(e.normal);
    if (this.tmpC.lengthSq() < 1e-8) this.tmpC.set(0, 1, 0);
    else this.tmpC.normalize();
    this.reflectDir(e.incoming, this.tmpC, this.refl);

    // Spawn a hair off the surface so nothing z-fights the wall it came from.
    this.tmpA.copy(e.position).addScaledVector(this.tmpC, 0.02);

    switch (e.surface) {
      case 'concrete':
        this.stone(P, energy, scale, C.concreteDust, C.concreteChip);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.09 + 0.1 * energy);
        break;
      case 'metal':
        this.metal(P, energy, scale);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.06 + 0.07 * energy);
        break;
      case 'wood':
        this.wood(P, energy, scale);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.08 + 0.09 * energy);
        break;
      case 'sand':
        this.granular(P, energy, scale, C.sand, 1.0);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.13 + 0.14 * energy);
        break;
      case 'dirt':
        this.granular(P, energy, scale, C.dirt, 0.85);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.12 + 0.13 * energy);
        break;
      case 'glass':
        this.glass(P, energy, scale);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.1 + 0.12 * energy);
        break;
      case 'flesh':
        this.flesh(P, e, energy, scale);
        break;
      case 'water':
        this.water(P, energy, scale);
        break;
      case 'foliage':
        this.foliage(P, energy, scale);
        break;
      case 'fabric':
        this.soft(P, energy, scale, C.fabric);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.07 + 0.06 * energy);
        break;
      case 'rubber':
        this.soft(P, energy, scale, C.rubber);
        this.requestDecal(e.position, this.tmpC, e.surface, 0.06 + 0.05 * energy);
        break;
      default:
        this.stone(P, energy, scale, C.concreteDust, C.concreteChip);
        break;
    }
  };

  // ── per-surface recipes ────────────────────────────────────────────────────

  /**
   * Concrete / stone: expanding grey dust, a handful of chips on real
   * ballistic arcs, and a surface-hugging spall ring. Deliberately no sparks.
   */
  private stone(P: Particles, energy: number, scale: number, dust: THREE.Color, chip: THREE.Color): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nDust = Math.max(2, Math.round(8 * energy * scale));
    for (let i = 0; i < nDust; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      // Mixing two sprites stops a burst reading as N copies of one shape.
      d.cell = rng.next() < 0.32 ? AtlasCell.Smoke : AtlasCell.Dust;
      d.setPos(p);
      randomCone(n, 1.2, rng, this.dir);
      const sp = rng.range(0.7, 3.0) * (0.55 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.15, 0.8), this.dir.z * sp);
      d.life = rng.range(0.9, 1.9) * (0.7 + 0.5 * energy);
      d.size0 = rng.range(0.05, 0.11) * (0.7 + energy);
      // Real scale check: a rifle round on concrete throws a puff a few tens
      // of centimetres across, not a metre and a half.
      d.size1 = rng.range(0.28, 0.62) * (0.7 + energy);
      // Peak alpha must account for OVERLAP: eight of these on top of one
      // another at 0.4 each compound to a 94%-opaque grey disc.
      d.setColor(dust.r, dust.g, dust.b, rng.range(0.13, 0.26));
      d.gravity = 0.05;
      d.drag = 2.2;
      d.turbulence = 0.26;
      d.flags = PFlags.Lit;
      d.fadePow = 2.0;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(1.0);
      d.setPlane(p, n, 0.35);
      P.commit();
    }

    const nChip = Math.max(1, Math.round(7 * energy * scale));
    for (let i = 0; i < nChip; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = rng.next() < 0.3 ? AtlasCell.Grain : AtlasCell.Debris;
      d.setPos(p);
      randomCone(this.refl, 0.9, rng, this.dir);
      const sp = rng.pow(2.5, 11, 1.6) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.5, 2.5), this.dir.z * sp);
      d.life = rng.range(0.9, 1.8);
      d.size0 = rng.range(0.022, 0.06) * (0.7 + 0.6 * energy);
      d.size1 = d.size0 * 0.9;
      d.setColor(chip.r, chip.g, chip.b, 1);
      d.gravity = 1;
      d.drag = 0.28;
      d.flags = PFlags.Lit;
      d.fadePow = 0.7;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(9);
      P.commit();
    }

    // Spall ring: a fast, low, surface-aligned shock of dust around the crater.
    if (energy > 0.25) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Ring;
      d.setPos(p);
      d.life = 0.3;
      d.size0 = 0.05 * energy;
      d.size1 = 0.16 + 0.22 * energy;
      d.setColor(dust.r, dust.g, dust.b, 0.34 * energy);
      d.flags = PFlags.Lit | PFlags.Oriented;
      d.fadePow = 1.5;
      d.rot = rng.range(0, TAU);
      d.setOrientation(n);
      P.commit();
    }
  }

  /**
   * Metal: an orange-white spark shower with velocity streaks, a cooling hot
   * point, a wisp of grey smoke and a pooled light. Optionally a second,
   * delayed wave scattering off the floor for a real ricochet arc.
   */
  private metal(P: Particles, energy: number, scale: number): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nSpark = Math.max(4, Math.round(24 * energy * scale));
    for (let i = 0; i < nSpark; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Spark;
      d.setPos(p);
      randomCone(this.refl, 0.9, rng, this.dir);
      // Power-biased speed: most sparks are slow, a few fly. `rng.pow` maps
      // rand^p, so an exponent ABOVE 1 biases toward the low end — an
      // exponent below 1 would do the opposite and give an even, unconvincing
      // fan of uniformly fast sparks.
      const sp = rng.pow(3.5, 19, 1.9) * (0.45 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0, 1.5), this.dir.z * sp);
      d.life = rng.range(0.2, 0.8);
      d.size0 = rng.range(0.04, 0.085);
      d.size1 = d.size0 * 0.22;
      d.setColor(1, 0.8, 0.46, rng.range(1.5, 3.2));
      d.gravity = 1;
      d.drag = rng.range(0.9, 2.3);
      d.stretch = rng.range(0.6, 1.25);
      d.flags = PFlags.Blackbody | PFlags.Flicker;
      d.fadePow = 1.0;
      P.commit();
    }

    this.ricochetScatter(P, energy, scale);

    // Hot point: white → orange → dark red over ~0.4 s. This tiny detail is
    // what makes a metal hit feel like it deposited energy.
    {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Circle;
      d.setPos(p);
      d.life = 0.42;
      d.size0 = 0.075 * (0.6 + energy);
      d.size1 = 0.02;
      d.setColor(1, 1, 1, 2.4);
      d.flags = PFlags.Blackbody;
      d.fadePow = 1.3;
      P.commit();
    }

    const nSmoke = Math.max(1, Math.round(2.5 * energy * scale));
    for (let i = 0; i < nSmoke; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.SmokeWisp;
      d.setPos(p);
      randomCone(n, 0.8, rng, this.dir);
      const sp = rng.range(0.4, 1.5);
      d.setVel(this.dir.x * sp, this.dir.y * sp + 0.5, this.dir.z * sp);
      d.life = rng.range(0.7, 1.3);
      d.size0 = 0.06;
      d.size1 = rng.range(0.3, 0.55);
      d.setColor(C.smoke.r, C.smoke.g, C.smoke.b, 0.24);
      d.gravity = -0.05;
      d.drag = 2.0;
      d.turbulence = 0.4;
      d.flags = PFlags.Lit;
      d.fadePow = 2.2;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(1.4);
      d.setPlane(p, n, 0.3);
      P.commit();
    }

    // Spark showers genuinely light their surroundings; the pooled light is
    // the cheapest way to make a ricochet feel hot.
    this.flashColor.copy(C.sparkWarm);
    this.lights?.flash(p, this.flashColor, 3.5 + 9 * energy, 0.1, 4.5);
  }

  /**
   * Second-bounce sparks. Probes straight down for a floor and scatters a
   * delayed wave off it, timed by the free-fall interval so the two waves read
   * as one continuous ricochet. Skipped entirely without a physics system.
   */
  private ricochetScatter(P: Particles, energy: number, scale: number): void {
    if (energy < 0.3 || scale < 0.35) return;
    const phys = this.physicsSystem;
    if (!phys) return;

    let hitY: number | null = null;
    try {
      const hit = phys.raycast(this.tmpA, this.down, 6);
      if (hit) hitY = hit.point.y;
    } catch {
      // Physics is not ready (or not implemented). Never ask again.
      this.physicsUsable = false;
      this.physics = null;
      return;
    }
    if (hitY === null) return;

    const drop = Math.max(0.05, this.tmpA.y - hitY);
    const fall = Math.sqrt((2 * drop) / 9.81);
    const rng = this.rng;
    const count = Math.max(2, Math.round(6 * energy * scale));
    for (let i = 0; i < count; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Spark;
      d.px = this.tmpA.x + rng.sym(0.25);
      d.py = hitY + 0.015;
      d.pz = this.tmpA.z + rng.sym(0.25);
      randomCone(this.up, 1.25, rng, this.dir);
      const sp = rng.range(0.8, 3.4);
      d.setVel(this.dir.x * sp, Math.abs(this.dir.y) * sp * 0.7, this.dir.z * sp);
      d.delay = fall * rng.range(0.75, 1.15);
      d.life = rng.range(0.12, 0.4);
      d.size0 = rng.range(0.025, 0.05);
      d.size1 = d.size0 * 0.3;
      // Already cooled on the way down: start further along the ramp.
      d.setColor(1, 0.6, 0.24, rng.range(0.9, 1.8));
      d.gravity = 1;
      d.drag = 2.6;
      d.stretch = 0.7;
      d.flags = PFlags.Blackbody | PFlags.Flicker;
      d.fadePow = 1.1;
      P.commit();
    }
  }

  /** Wood: elongated splinters that tumble, plus a warm tan dust puff. */
  private wood(P: Particles, energy: number, scale: number): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nSpl = Math.max(2, Math.round(9 * energy * scale));
    for (let i = 0; i < nSpl; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Splinter;
      d.setPos(p);
      randomCone(this.refl, 0.85, rng, this.dir);
      const sp = rng.pow(2.5, 10, 1.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.4, 2.2), this.dir.z * sp);
      d.life = rng.range(0.9, 1.7);
      d.size0 = rng.range(0.05, 0.13) * (0.7 + 0.5 * energy);
      d.size1 = d.size0;
      d.setColor(C.woodChip.r, C.woodChip.g, C.woodChip.b, 1);
      d.gravity = 1;
      d.drag = 0.45;
      d.flags = PFlags.Lit;
      d.fadePow = 0.8;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(12);
      P.commit();
    }

    const nDust = Math.max(1, Math.round(5 * energy * scale));
    for (let i = 0; i < nDust; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Dust;
      d.setPos(p);
      randomCone(n, 1.0, rng, this.dir);
      const sp = rng.range(0.5, 2.2);
      d.setVel(this.dir.x * sp, this.dir.y * sp + 0.4, this.dir.z * sp);
      d.life = rng.range(0.7, 1.4);
      d.size0 = rng.range(0.06, 0.13);
      d.size1 = rng.range(0.35, 0.7);
      d.setColor(C.woodDust.r, C.woodDust.g, C.woodDust.b, rng.range(0.12, 0.22));
      d.gravity = 0.1;
      d.drag = 2.4;
      d.turbulence = 0.22;
      d.flags = PFlags.Lit;
      d.fadePow = 2.0;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(1.2);
      d.setPlane(p, n, 0.3);
      P.commit();
    }
  }

  /**
   * Sand / dirt: a big soft plume that hangs and settles slowly, plus a spray
   * of individual grains. The plume is the hero — it should linger and be
   * strongly backlit at golden hour.
   */
  private granular(P: Particles, energy: number, scale: number, tint: THREE.Color, grainMul: number): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nPlume = Math.max(3, Math.round(12 * energy * scale));
    for (let i = 0; i < nPlume; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = rng.next() < 0.45 ? AtlasCell.Smoke : AtlasCell.Dust;
      d.setPos(p);
      randomCone(n, 1.0, rng, this.dir);
      const sp = rng.range(0.6, 3.4) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.4, 1.6), this.dir.z * sp);
      d.life = rng.range(1.6, 3.2) * (0.7 + 0.4 * energy);
      d.size0 = rng.range(0.1, 0.26) * (0.7 + energy);
      d.size1 = rng.range(0.9, 2.0) * (0.7 + energy);
      d.setColor(tint.r, tint.g, tint.b, rng.range(0.12, 0.24));
      d.gravity = 0.05;
      d.drag = 2.6;
      d.turbulence = 0.34;
      d.flags = PFlags.Lit;
      d.fadePow = 2.4;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(0.8);
      d.setPlane(p, n, 0.45);
      P.commit();
    }

    const nGrain = Math.max(2, Math.round(14 * energy * scale * grainMul));
    for (let i = 0; i < nGrain; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Grain;
      d.setPos(p);
      randomCone(this.refl, 0.85, rng, this.dir);
      const sp = rng.pow(2, 9, 1.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(1.0, 3.5), this.dir.z * sp);
      d.life = rng.range(0.6, 1.3);
      d.size0 = rng.range(0.012, 0.032);
      d.size1 = d.size0;
      d.setColor(tint.r * 1.05, tint.g * 1.02, tint.b, 1);
      d.gravity = 1;
      d.drag = 0.5;
      d.flags = PFlags.Lit;
      d.fadePow = 0.8;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(6);
      P.commit();
    }
  }

  /** Glass: shards, a shatter ring and a cloud of glittering fragments. */
  private glass(P: Particles, energy: number, scale: number): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nShard = Math.max(3, Math.round(14 * energy * scale));
    for (let i = 0; i < nShard; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Glass;
      d.setPos(p);
      randomCone(this.refl, 1.0, rng, this.dir);
      const sp = rng.pow(2.5, 12, 1.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.3, 2.0), this.dir.z * sp);
      d.life = rng.range(1.0, 1.8);
      d.size0 = rng.range(0.03, 0.09);
      d.size1 = d.size0;
      d.setColor(C.glass.r, C.glass.g, C.glass.b, rng.range(0.55, 0.95));
      d.gravity = 1;
      d.drag = 0.3;
      // Fast tumble is what makes shards catch the light in flashes.
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(16);
      d.flags = PFlags.Lit;
      d.fadePow = 0.9;
      P.commit();
    }

    // Additive glitter: the specular "sparkle" pass sold separately from the
    // shard bodies, because a lit alpha shard can never out-shine the sky.
    const nGlint = Math.max(4, Math.round(18 * energy * scale));
    for (let i = 0; i < nGlint; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Glint;
      d.setPos(p);
      randomCone(this.refl, 1.1, rng, this.dir);
      const sp = rng.pow(2, 11, 1.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.3, 2.2), this.dir.z * sp);
      d.life = rng.range(0.4, 1.2);
      d.size0 = rng.range(0.03, 0.07);
      d.size1 = d.size0 * 0.5;
      d.setColor(0.85, 0.95, 1.0, rng.range(0.5, 1.6));
      d.gravity = 1;
      d.drag = 0.6;
      d.flags = PFlags.Flicker;
      d.fadePow = 1.4;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(4);
      P.commit();
    }

    {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Ring;
      d.setPos(p);
      d.life = 0.28;
      d.size0 = 0.06;
      d.size1 = 0.3 + 0.4 * energy;
      d.setColor(0.8, 0.92, 1.0, 0.9);
      d.flags = PFlags.Oriented;
      d.fadePow = 1.6;
      d.setOrientation(n);
      P.commit();
    }
  }

  /**
   * Flesh: a short blood mist cone thrown along the direction of travel plus
   * arcing droplets. Deliberately restrained and stylised — brief, dark, and
   * over quickly.
   */
  private flesh(P: Particles, e: ImpactEvent, energy: number, scale: number): void {
    const rng = this.rng;
    const p = this.tmpA;

    // Mist continues in the bullet's direction (exit side), not the normal.
    this.dir.copy(e.incoming);
    if (this.dir.lengthSq() < 1e-8) this.dir.copy(this.tmpC).negate();
    else this.dir.normalize();
    this.tmpB.copy(this.dir);

    const nMist = Math.max(2, Math.round(9 * energy * scale));
    for (let i = 0; i < nMist; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.SmokeWisp;
      d.setPos(p);
      randomCone(this.tmpB, 0.55, rng, this.dir);
      const sp = rng.range(1.5, 5.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0, 0.6), this.dir.z * sp);
      d.life = rng.range(0.22, 0.45);
      d.size0 = rng.range(0.05, 0.11);
      d.size1 = rng.range(0.22, 0.42);
      d.setColor(C.bloodMist.r, C.bloodMist.g, C.bloodMist.b, rng.range(0.3, 0.55));
      d.gravity = 0.35;
      d.drag = 3.2;
      d.turbulence = 0.2;
      d.fadePow = 1.6;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(2.5);
      P.commit();
    }

    const nDrop = Math.max(2, Math.round(11 * energy * scale));
    for (let i = 0; i < nDrop; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Blood;
      d.setPos(p);
      randomCone(this.tmpB, 0.8, rng, this.dir);
      const sp = rng.pow(1.5, 7, 1.4) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.4, 2.2), this.dir.z * sp);
      d.life = rng.range(0.5, 1.0);
      d.size0 = rng.range(0.025, 0.055);
      d.size1 = d.size0 * 0.85;
      d.setColor(C.blood.r, C.blood.g, C.blood.b, 0.95);
      d.gravity = 1;
      d.drag = 0.6;
      d.stretch = 0.55;
      d.fadePow = 1.1;
      P.commit();
    }

    // Droplets land somewhere: ask the decal system for a splatter under the hit.
    if (scale > 0.4 && this.decalsUsable) {
      const phys = this.physicsSystem;
      if (phys) {
        try {
          const hit = phys.raycast(p, this.down, 3.0);
          if (hit) this.requestDecal(hit.point, hit.normal, 'flesh', 0.25 + 0.35 * energy);
        } catch {
          this.physicsUsable = false;
          this.physics = null;
        }
      }
    }

    this.lastFlesh.copy(e.position);
    this.lastFleshNormal.copy(this.tmpC);
    this.lastFleshTime = this.time;
  }

  /** Water: a splash crown, a surface ripple ring and a little fine mist. */
  private water(P: Particles, energy: number, scale: number): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nDrop = Math.max(4, Math.round(18 * energy * scale));
    for (let i = 0; i < nDrop; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Water;
      d.setPos(p);
      // Narrow cone around the normal = a crown rather than a dome.
      randomCone(n, 0.45, rng, this.dir);
      const sp = rng.pow(2.5, 9, 1.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, Math.abs(this.dir.y) * sp + rng.range(1.5, 4.5), this.dir.z * sp);
      d.life = rng.range(0.5, 1.0);
      d.size0 = rng.range(0.03, 0.075);
      d.size1 = d.size0 * 0.8;
      d.setColor(C.water.r, C.water.g, C.water.b, rng.range(0.55, 0.95));
      d.gravity = 1;
      d.drag = 0.5;
      d.stretch = 0.5;
      d.fadePow = 1.2;
      P.commit();
    }

    const nMist = Math.max(2, Math.round(5 * energy * scale));
    for (let i = 0; i < nMist; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Dust;
      d.setPos(p);
      randomCone(n, 0.8, rng, this.dir);
      const sp = rng.range(0.6, 2.2);
      d.setVel(this.dir.x * sp, this.dir.y * sp + 0.8, this.dir.z * sp);
      d.life = rng.range(0.5, 1.1);
      d.size0 = 0.08;
      d.size1 = rng.range(0.35, 0.7);
      d.setColor(0.86, 0.93, 0.97, rng.range(0.16, 0.3));
      d.gravity = 0.25;
      d.drag = 3.0;
      d.turbulence = 0.3;
      d.flags = PFlags.Lit;
      d.fadePow = 2.0;
      P.commit();
    }

    {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Ring;
      d.setPos(p);
      d.life = 0.8;
      d.size0 = 0.1;
      d.size1 = 0.45 + 0.55 * energy;
      d.setColor(0.9, 0.96, 1.0, 0.38);
      d.flags = PFlags.Oriented | PFlags.Lit;
      d.fadePow = 1.5;
      d.setOrientation(n);
      P.commit();
    }
  }

  /** Foliage: torn leaf fragments that flutter down on turbulence. */
  private foliage(P: Particles, energy: number, scale: number): void {
    const rng = this.rng;
    const p = this.tmpA;
    const nLeaf = Math.max(2, Math.round(10 * energy * scale));
    for (let i = 0; i < nLeaf; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Leaf;
      d.setPos(p);
      randomCone(this.refl, 1.2, rng, this.dir);
      const sp = rng.range(0.8, 4.5) * (0.5 + energy);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0, 1.2), this.dir.z * sp);
      d.life = rng.range(1.4, 2.8);
      d.size0 = rng.range(0.04, 0.1);
      d.size1 = d.size0;
      d.setColor(C.foliage.r, C.foliage.g, C.foliage.b, 1);
      // Low gravity + high drag + turbulence = the flutter of a falling leaf.
      d.gravity = 0.32;
      d.drag = 1.5;
      d.turbulence = 0.55;
      d.flags = PFlags.Lit;
      d.fadePow = 1.2;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(7);
      P.commit();
    }
  }

  /** Fabric / rubber: a muted puff and a couple of torn fibres. No sparks. */
  private soft(P: Particles, energy: number, scale: number, tint: THREE.Color): void {
    const rng = this.rng;
    const n = this.tmpC;
    const p = this.tmpA;

    const nPuff = Math.max(1, Math.round(5 * energy * scale));
    for (let i = 0; i < nPuff; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Dust;
      d.setPos(p);
      randomCone(n, 1.0, rng, this.dir);
      const sp = rng.range(0.4, 1.8);
      d.setVel(this.dir.x * sp, this.dir.y * sp + 0.3, this.dir.z * sp);
      d.life = rng.range(0.5, 1.1);
      d.size0 = rng.range(0.05, 0.11);
      d.size1 = rng.range(0.22, 0.5);
      d.setColor(tint.r, tint.g, tint.b, rng.range(0.16, 0.3));
      d.gravity = 0.08;
      d.drag = 2.6;
      d.turbulence = 0.2;
      d.flags = PFlags.Lit;
      d.fadePow = 2.0;
      d.setPlane(p, n, 0.28);
      P.commit();
    }

    const nBits = Math.max(1, Math.round(3 * energy * scale));
    for (let i = 0; i < nBits; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Splinter;
      d.setPos(p);
      randomCone(this.refl, 1.0, rng, this.dir);
      const sp = rng.range(1, 5);
      d.setVel(this.dir.x * sp, this.dir.y * sp + 0.8, this.dir.z * sp);
      d.life = rng.range(0.7, 1.4);
      d.size0 = rng.range(0.02, 0.055);
      d.size1 = d.size0;
      d.setColor(tint.r * 0.85, tint.g * 0.85, tint.b * 0.85, 1);
      d.gravity = 0.9;
      d.drag = 1.1;
      d.flags = PFlags.Lit;
      d.fadePow = 1.0;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(10);
      P.commit();
    }
  }

  // ── explosion ──────────────────────────────────────────────────────────────

  /**
   * Layered explosion. The layers are deliberately staged in time using the
   * particle system's free `delay` field so the event reads as a sequence —
   * flash, fireball, shock, then the slow smoke column — rather than one
   * simultaneous pop.
   */
  private onExplosion = (ev: GameEvents['explosion']): void => {
    const P = this.P;
    if (!P) return;
    const pos = ev.position;
    const radius = Math.max(0.5, ev.radius);
    const power = Math.max(0.1, ev.power);
    const rng = this.rng;

    const ctx = this.ctx;
    const camDist = ctx ? ctx.camera.position.distanceTo(pos) : 30;

    // Camera shake is emitted regardless of visibility: you feel a blast you
    // cannot see. Inverse-square-ish falloff, clamped so a point-blank grenade
    // does not detach the player's head.
    const shake = Math.min(1.6, (power * radius * 0.9) / Math.max(2.5, camDist));
    if (shake > 0.02) {
      this.shakePayload.amplitude = shake;
      this.shakePayload.duration = 0.45 + 0.25 * shake;
      ctx?.bus.emit('camera:shake', this.shakePayload);
    }

    // Radial impulse for any dynamic bodies (shell casings, props).
    const phys = this.physicsSystem;
    if (phys) {
      try {
        phys.applyRadialImpulse(pos, radius * 2.2, power * 12);
      } catch {
        this.physicsUsable = false;
        this.physics = null;
      }
    }

    // Cull FIRST, then apply the floor. Flooring before the test made the
    // guard unreachable and let an explosion a kilometre away still emit a
    // full effect. Explosions get a higher floor than bullet impacts because
    // a blast you can barely see should still register.
    const rawScale = this.budgetScale(pos);
    if (rawScale <= 0.02) return;
    const scale = Math.max(0.25, rawScale);

    // 1. Flash core — two frames of blown-out white, gone before you can look.
    for (let i = 0; i < 2; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = i === 0 ? AtlasCell.Flash : AtlasCell.Circle;
      d.setPos(pos);
      d.life = 0.1 + 0.05 * i;
      d.size0 = radius * (0.7 + 0.5 * i);
      d.size1 = radius * (2.0 + 1.2 * i);
      d.setColor(1, 0.95, 0.82, 5.5 - 2.0 * i);
      d.fadePow = 1.6;
      d.rot = rng.range(0, TAU);
      P.commit();
    }

    // 2. Fireball — turbulent, staggered, cooling.
    const nFire = Math.max(6, Math.round(20 * scale));
    for (let i = 0; i < nFire; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Ember;
      d.px = pos.x + rng.sym(radius * 0.25);
      d.py = pos.y + rng.sym(radius * 0.25);
      d.pz = pos.z + rng.sym(radius * 0.25);
      randomCone(this.up, 1.55, rng, this.dir);
      const sp = rng.range(1.5, 8) * (0.4 + 0.5 * power);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.5, 3.5), this.dir.z * sp);
      d.delay = rng.range(0, 0.09);
      // Short life + steep fade on purpose. A blackbody ramp that is still
      // visible at 80% of its life renders as big saturated RED blobs hanging
      // in the air half a second after the blast, which reads as coloured
      // smoke rather than cooling fire. The smoke column takes over instead.
      d.life = rng.range(0.28, 0.62);
      d.size0 = radius * rng.range(0.25, 0.55);
      d.size1 = radius * rng.range(0.85, 1.6);
      d.setColor(1, 0.55, 0.2, rng.range(1.6, 3.2));
      d.gravity = -0.35;
      d.drag = 2.4;
      d.turbulence = 1.4;
      d.flags = PFlags.Flicker | PFlags.Blackbody;
      d.fadePow = 2.8;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(1.5);
      P.commit();
    }

    // 3. Shockwave — a ground-hugging ring plus an air-blast ring. Cheap and
    //    far more legible than a screen-space distortion at this scale.
    {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Ring;
      d.setPos(pos);
      d.life = 0.32;
      d.size0 = radius * 0.5;
      d.size1 = radius * 3.4;
      d.setColor(1, 0.88, 0.7, 1.5);
      d.flags = PFlags.Oriented;
      d.fadePow = 1.9;
      d.rot = rng.range(0, TAU);
      d.setOrientation(this.up);
      P.commit();
    }
    {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Ring;
      d.setPos(pos);
      d.life = 0.24;
      d.size0 = radius * 0.4;
      d.size1 = radius * 2.6;
      d.setColor(1, 0.93, 0.85, 1.1);
      d.fadePow = 2.2;
      d.rot = rng.range(0, TAU);
      P.commit();
    }

    // 4. Ground dust ring — horizontal, fast, low. Sells the blast footprint.
    const nRing = Math.max(5, Math.round(16 * scale));
    for (let i = 0; i < nRing; i++) {
      const a = (i / nRing) * TAU + rng.sym(0.3);
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Smoke;
      d.px = pos.x + Math.cos(a) * radius * 0.3;
      d.py = pos.y - radius * 0.25;
      d.pz = pos.z + Math.sin(a) * radius * 0.3;
      const sp = rng.range(4, 11);
      d.setVel(Math.cos(a) * sp, rng.range(0.2, 1.4), Math.sin(a) * sp);
      d.delay = rng.range(0, 0.06);
      d.life = rng.range(1.6, 3.0);
      d.size0 = radius * 0.4;
      d.size1 = radius * rng.range(1.5, 2.6);
      d.setColor(C.sand.r, C.sand.g, C.sand.b, rng.range(0.22, 0.4));
      d.gravity = 0.04;
      d.drag = 2.2;
      d.turbulence = 0.5;
      d.flags = PFlags.Lit;
      d.fadePow = 2.2;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(0.7);
      P.commit();
    }

    // 5. Rising smoke column — the long tail of the effect.
    const nCol = Math.max(6, Math.round(20 * scale));
    for (let i = 0; i < nCol; i++) {
      const t = i / nCol;
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = rng.next() < 0.5 ? AtlasCell.Smoke : AtlasCell.SmokeWisp;
      d.px = pos.x + rng.sym(radius * 0.4);
      d.py = pos.y + t * radius * 0.5;
      d.pz = pos.z + rng.sym(radius * 0.4);
      d.setVel(rng.sym(1.2), rng.range(1.4, 4.2), rng.sym(1.2));
      d.delay = t * 0.55 + rng.range(0, 0.1);
      d.life = rng.range(2.8, 5.0);
      d.size0 = radius * rng.range(0.35, 0.7);
      d.size1 = radius * rng.range(1.8, 3.2);
      // `t` is emission order, so t=0 is emitted first and by the time the
      // column has formed it is the oldest, highest and most thinned-out
      // smoke — hence the LIGHTEST. The freshest smoke at the base is dark.
      const k = 0.82 - 0.5 * t;
      d.setColor(C.smoke.r * k, C.smoke.g * k, C.smoke.b * k, rng.range(0.3, 0.5));
      d.gravity = -0.16;
      d.drag = 1.0;
      d.turbulence = 0.8;
      d.flags = PFlags.Lit;
      d.fadePow = 2.4;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(0.6);
      P.commit();
    }

    // 6. Debris and embers.
    const nDeb = Math.max(4, Math.round(18 * scale));
    for (let i = 0; i < nDeb; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = rng.next() < 0.4 ? AtlasCell.Grain : AtlasCell.Debris;
      d.setPos(pos);
      randomCone(this.up, 1.35, rng, this.dir);
      const sp = rng.pow(4, 22, 1.4) * (0.4 + 0.6 * power);
      d.setVel(this.dir.x * sp, Math.abs(this.dir.y) * sp * 0.8 + rng.range(1, 6), this.dir.z * sp);
      d.life = rng.range(1.2, 2.6);
      d.size0 = rng.range(0.03, 0.11);
      d.size1 = d.size0;
      d.setColor(C.concreteChip.r, C.concreteChip.g, C.concreteChip.b, 1);
      d.gravity = 1;
      d.drag = 0.22;
      d.flags = PFlags.Lit;
      d.fadePow = 0.7;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(11);
      P.commit();
    }

    const nEmber = Math.max(4, Math.round(14 * scale));
    for (let i = 0; i < nEmber; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Ember;
      d.setPos(pos);
      randomCone(this.up, 1.4, rng, this.dir);
      const sp = rng.range(2, 12);
      d.setVel(this.dir.x * sp, Math.abs(this.dir.y) * sp + rng.range(0.5, 3), this.dir.z * sp);
      d.delay = rng.range(0, 0.15);
      d.life = rng.range(0.8, 2.4);
      d.size0 = rng.range(0.05, 0.13);
      d.size1 = rng.range(0.015, 0.04);
      d.setColor(1, 0.6, 0.25, rng.range(1.2, 2.4));
      d.gravity = 0.55;
      d.drag = 1.3;
      d.stretch = 0.35;
      d.turbulence = 0.5;
      d.flags = PFlags.Blackbody | PFlags.Flicker;
      d.fadePow = 1.5;
      P.commit();
    }

    this.flashColor.setRGB(1.0, 0.62, 0.28);
    this.lights?.flash(pos, this.flashColor, 40 * power, 0.28, radius * 6);

    this.scorchDecal(pos, radius);
  };

  /**
   * Drops the scorch mark on the ground beneath the blast rather than at the
   * blast's own altitude — an airburst centred 2 m up would otherwise request
   * a floating, up-facing decal that no decal system can place sensibly.
   */
  private scorchDecal(pos: THREE.Vector3, radius: number): void {
    const phys = this.physicsSystem;
    if (phys) {
      try {
        const hit = phys.raycast(pos, this.down, radius * 2 + 3);
        if (hit) this.requestDecal(hit.point, hit.normal, 'concrete', radius * 1.2);
        return;
      } catch {
        this.physicsUsable = false;
        this.physics = null;
      }
    }
    // No physics system: only place a scorch when the blast was near the deck,
    // where assuming a flat floor underneath is safe.
    if (pos.y < 1.5) {
      this.tmpB.set(pos.x, pos.y - Math.min(pos.y, 0.4), pos.z);
      this.requestDecal(this.tmpB, this.up, 'concrete', radius * 1.2);
    }
  }

  // ── incidental effects ─────────────────────────────────────────────────────

  /**
   * Muzzle byproducts. The weapon system owns the viewmodel flash; this adds
   * the *world-space* residue — propellant smoke, a couple of embers and a
   * short light — so firing lights the surroundings and leaves a haze.
   */
  private onWeaponFire = (ev: GameEvents['weapon:fire']): void => {
    const P = this.P;
    if (!P) return;
    const rng = this.rng;
    const scale = this.P?.spawnScale ?? 1;

    // `weapon:fire.direction` carries no unit-length guarantee, and
    // randomCone's orthonormal basis assumes one. Normalise defensively, the
    // same way onImpact() guards its surface normal.
    this.tmpC.copy(ev.direction);
    if (this.tmpC.lengthSq() < 1e-8) this.tmpC.set(0, 0, -1);
    else this.tmpC.normalize();
    this.tmpA.copy(ev.origin).addScaledVector(this.tmpC, 0.22);

    const nSmoke = Math.max(1, Math.round(4 * scale));
    for (let i = 0; i < nSmoke; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.SmokeWisp;
      d.setPos(this.tmpA);
      randomCone(this.tmpC, 0.4, rng, this.dir);
      const sp = rng.range(1.2, 4.0);
      d.setVel(this.dir.x * sp, this.dir.y * sp + rng.range(0.1, 0.6), this.dir.z * sp);
      d.life = rng.range(0.5, 1.1);
      d.size0 = rng.range(0.04, 0.09);
      d.size1 = rng.range(0.25, 0.55);
      d.setColor(0.72, 0.69, 0.64, rng.range(0.1, 0.22));
      d.gravity = -0.08;
      d.drag = 3.0;
      d.turbulence = 0.5;
      d.flags = PFlags.Lit;
      d.fadePow = 2.4;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(2.2);
      P.commit();
    }

    // A few unburnt-propellant embers spat out with the gas.
    const nEmber = Math.max(1, Math.round(3 * scale));
    for (let i = 0; i < nEmber; i++) {
      const d = P.next();
      d.batch = PBatch.Additive;
      d.cell = AtlasCell.Spark;
      d.setPos(this.tmpA);
      randomCone(this.tmpC, 0.5, rng, this.dir);
      const sp = rng.range(3, 11);
      d.setVel(this.dir.x * sp, this.dir.y * sp, this.dir.z * sp);
      d.life = rng.range(0.1, 0.3);
      d.size0 = rng.range(0.02, 0.045);
      d.size1 = d.size0 * 0.3;
      d.setColor(1, 0.72, 0.35, rng.range(1.2, 2.4));
      d.gravity = 1;
      d.drag = 2.5;
      d.stretch = 0.9;
      d.flags = PFlags.Blackbody | PFlags.Flicker;
      d.fadePow = 1.0;
      P.commit();
    }

    this.flashColor.setRGB(1.0, 0.78, 0.45);
    this.lights?.flash(this.tmpA, this.flashColor, 9, 0.055, 7);
  };

  /** A puff of surface-appropriate dust under the player's feet. */
  private onFootstep = (ev: GameEvents['player:footstep']): void => {
    const P = this.P;
    if (!P) return;
    let tint: THREE.Color;
    let amount: number;
    switch (ev.surface) {
      case 'sand':
        tint = C.sand;
        amount = 1.0;
        break;
      case 'dirt':
        tint = C.dirt;
        amount = 0.85;
        break;
      case 'concrete':
        tint = C.concreteDust;
        amount = 0.35;
        break;
      case 'wood':
        tint = C.woodDust;
        amount = 0.3;
        break;
      case 'foliage':
        tint = C.foliage;
        amount = 0.4;
        break;
      case 'water':
        tint = C.water;
        amount = 0.7;
        break;
      // Metal, glass, fabric and rubber do not kick up dust.
      default:
        return;
    }
    if (ev.sprinting) amount *= 1.7;

    const rng = this.rng;
    const scale = P.spawnScale;
    const n = Math.max(1, Math.round(4 * amount * scale));
    for (let i = 0; i < n; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.Dust;
      d.px = ev.position.x + rng.sym(0.1);
      d.py = ev.position.y + 0.03;
      d.pz = ev.position.z + rng.sym(0.1);
      d.setVel(rng.sym(0.7), rng.range(0.15, 0.7), rng.sym(0.7));
      d.life = rng.range(0.8, 1.7);
      d.size0 = rng.range(0.06, 0.14);
      d.size1 = rng.range(0.35, 0.8) * amount;
      d.setColor(tint.r, tint.g, tint.b, rng.range(0.1, 0.24) * amount);
      d.gravity = 0.03;
      d.drag = 2.4;
      d.turbulence = 0.22;
      d.flags = PFlags.Lit;
      d.fadePow = 2.2;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(0.8);
      d.setPlane(ev.position, this.up, 0.3);
      P.commit();
    }
  };

  /**
   * `entity:killed` carries no position, so we punctuate the most recent flesh
   * impact instead — which is very nearly always the killing shot.
   */
  private onKilled = (ev: GameEvents['entity:killed']): void => {
    const P = this.P;
    if (!P) return;
    if (this.time - this.lastFleshTime > 0.5) return;
    const scale = this.budgetScale(this.lastFlesh);
    if (scale <= 0.02) return;

    const rng = this.rng;
    const boost = ev.headshot ? 1.6 : 1.0;
    const n = Math.max(2, Math.round(8 * scale * boost));
    for (let i = 0; i < n; i++) {
      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = AtlasCell.SmokeWisp;
      d.setPos(this.lastFlesh);
      randomCone(this.lastFleshNormal, 1.3, rng, this.dir);
      const sp = rng.range(1.0, 4.0) * boost;
      d.setVel(this.dir.x * sp, this.dir.y * sp + 0.4, this.dir.z * sp);
      d.life = rng.range(0.3, 0.6);
      d.size0 = rng.range(0.06, 0.14);
      d.size1 = rng.range(0.28, 0.55);
      d.setColor(C.bloodMist.r, C.bloodMist.g, C.bloodMist.b, rng.range(0.25, 0.45));
      d.gravity = 0.4;
      d.drag = 3.0;
      d.turbulence = 0.25;
      d.fadePow = 1.6;
      d.rot = rng.range(0, TAU);
      d.spin = rng.sym(2.0);
      P.commit();
    }
  };
}
