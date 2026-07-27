/**
 * NOISEFIELD — tracer rounds and ejected shell casings.
 *
 * TWO UNRELATED EFFECTS, ONE SYSTEM
 * ---------------------------------
 * Both are consequences of firing a weapon, both are pooled, and both want to
 * be single draw calls, so they share a lifecycle here.
 *
 * TRACERS — GPU-ANALYTIC RIBBONS
 * ------------------------------
 * A tracer is a straight-line, constant-speed event: everything about it is
 * known at spawn. So, exactly like the main particle system, we upload origin,
 * direction, distance, speed and start time once and let the vertex shader
 * derive the streak's head and tail from `t`. The CPU never touches a tracer
 * after it is fired. One instanced draw call covers every tracer in flight.
 *
 * The ribbon is built between the head and the tail and twisted to face the
 * camera, so a round passing across your view is a bright streak and a round
 * fired away from you is a shrinking dot — both correct. A screen-space
 * minimum width keeps distant tracers from disintegrating into sub-pixel
 * shimmer, which is the usual reason tracers look like fireflies.
 *
 * SHELL CASINGS — INSTANCED LIT GEOMETRY
 * --------------------------------------
 * Casings are real metal, so they are real `MeshStandardMaterial` geometry in
 * one `InstancedMesh`: they take the sun, they take the environment map, and
 * they glint as they tumble. That glint is a disproportionately large part of
 * why firing a rifle feels good, and it cannot be faked with a sprite.
 *
 * Simulation prefers the physics system (`addDynamic`) when one is present and
 * falls back to a self-contained ballistic integrator with a ground bounce.
 * The fallback is not a degraded mode — it is deterministic, allocation-free
 * and costs ~120 float updates a frame.
 *
 */

import * as THREE from 'three';
import type {
  Caliber,
  DynamicBodyOptions,
  EngineContext,
  FrameContext,
  GameEvents,
  IPhysics,
  QualityTier,
  System,
} from '../core/Types';
import { AtlasCell } from './textures';
import { PBatch, PFlags, Particles, Rng } from './Particles';

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────

const MAX_TRACERS = 192;

const MAX_SHELLS: Record<QualityTier, number> = { low: 24, medium: 64, high: 120, ultra: 120 };

/** Mass of a rifle casing in kg; also the impulse scale at the physics boundary. */
const SHELL_MASS = 0.012;

/**
 * Rigid-body description for a casing. Hoisted to module scope because every
 * field is constant and `shell:eject` fires up to ~10x/second on full auto —
 * allocating a fresh options literal per shot is exactly the kind of drip-feed
 * garbage this system is supposed to avoid.
 */
const SHELL_BODY: DynamicBodyOptions = {
  shape: 'box',
  mass: SHELL_MASS,
  restitution: 0.35,
  friction: 0.7,
  linearDamping: 0.04,
  angularDamping: 0.06,
  surface: 'metal',
};

/** Seconds a casing survives before it is recycled (shrinking out at the end). */
const SHELL_LIFE = 11;
const SHELL_FADE = 0.6;

/**
 * Visual travel speed is clamped. A 5.56 round leaves the muzzle at ~900 m/s
 * and crosses a 40 m courtyard in 44 ms — under three frames at 60 Hz, i.e.
 * invisible. Every shooter slows the *visual* tracer while leaving ballistics
 * alone; this is that clamp, and it is the difference between readable
 * suppressing fire and an occasional white flicker.
 */
const TRACER_SPEED_MIN = 60;
const TRACER_SPEED_MAX = 420;

/** Per-caliber casing scale (relative to a 5.56 case) and tint. */
const CALIBER: Record<Caliber, { scale: number; r: number; g: number; b: number }> = {
  '5.56': { scale: 1.0, r: 1.0, g: 1.0, b: 1.0 },
  '9mm': { scale: 0.78, r: 1.0, g: 0.96, b: 0.86 },
  '.338': { scale: 1.55, r: 1.0, g: 0.98, b: 0.9 },
  // Shotgun hulls are red plastic with a brass head; tint the whole case.
  '12ga': { scale: 1.7, r: 0.85, g: 0.22, b: 0.16 },
};

// ─────────────────────────────────────────────────────────────────────────────
// Tracer shaders
// ─────────────────────────────────────────────────────────────────────────────

const TRACER_VERT = /* glsl */ `
precision highp float;

attribute vec4 tFrom;   // xyz = muzzle position, w = start time
attribute vec4 tDir;    // xyz = unit direction, w = travel distance
attribute vec4 tParam;  // x = speed, y = streak length, z = width, w = seed
attribute vec4 tColor;  // rgb = tint, a = HDR intensity

uniform float uTime;
uniform float uMinAngular;

varying vec2  vUv;
varying vec4  vColor;
varying float vFade;

void main() {
  float t    = uTime - tFrom.w;
  float dist = tDir.w;
  float s    = t * tParam.x;

  // The streak is the segment of the flight path between (s - streakLen) and
  // s, clipped to [0, dist]. Once the tail reaches the target the round has
  // fully arrived and the instance retires.
  float headT = min(max(s, tParam.y * 0.25), dist);
  float tailT = clamp(s - tParam.y, 0.0, dist);

  if (t < 0.0 || tailT >= dist - 1e-4 || headT <= tailT + 1e-4) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(0.0);
    vColor = vec4(0.0);
    vFade = 0.0;
    return;
  }

  vec3  head    = tFrom.xyz + tDir.xyz * headT;
  vec3  tail    = tFrom.xyz + tDir.xyz * tailT;
  vec3  mid     = (head + tail) * 0.5;
  vec3  seg     = head - tail;
  float segLen  = length(seg);
  vec3  axis    = segLen > 1e-5 ? seg / segLen : tDir.xyz;

  vec3  toCam   = cameraPosition - mid;
  float camDist = length(toCam);
  vec3  viewDir = camDist > 1e-4 ? toCam / camDist : vec3(0.0, 0.0, 1.0);

  vec3  side = cross(axis, viewDir);
  float sl   = length(side);
  // Looking straight down the barrel of the tracer collapses the ribbon; pick
  // an arbitrary perpendicular so it degrades to a dot instead of vanishing.
  side = sl > 1e-4 ? side / sl : normalize(cross(axis, vec3(0.017, 1.0, 0.013)));

  // Screen-space minimum width. A 3 cm ribbon at 90 m is sub-pixel and
  // shimmers; widening with distance keeps it a stable bright line.
  float w = max(tParam.z, camDist * uMinAngular);

  vec2 corner = position.xy;               // -0.5 .. 0.5
  vec3 world  = mid + axis * (corner.y * segLen) + side * (corner.x * w * 2.0);

  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);

  vUv    = uv;
  vColor = tColor;
  // Dim over the last quarter of flight so the round does not simply blink out
  // at the impact point.
  float u = tailT / max(dist, 1e-3);
  vFade   = 1.0 - smoothstep(0.72, 1.0, u);
}
`;

const TRACER_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

varying vec2  vUv;
varying vec4  vColor;
varying float vFade;

void main() {
  // Cross-section: a tight white-hot core inside a broad soft halo. The halo
  // is a cheap stand-in for bloom so the tracer still reads when the post
  // stack's bloom pass is disabled on lower tiers.
  float x    = vUv.x * 2.0 - 1.0;
  float core = exp(-x * x * 16.0);
  float halo = exp(-x * x * 2.0) * 0.30;

  // Along the streak: dim at the tail, brightest at the head.
  float along    = pow(clamp(vUv.y, 0.0, 1.0), 1.5);
  float headGlow = smoothstep(0.80, 1.0, vUv.y);

  float a = ((core + halo) * (0.22 + 0.85 * along) + headGlow * core * 0.8) * vFade;
  if (a < 0.004) discard;

  vec3 col = mix(vColor.rgb, vec3(1.0), clamp(core * 0.8 + headGlow * 0.6, 0.0, 1.0));
  gl_FragColor = vec4(col * vColor.a, a);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// System
// ─────────────────────────────────────────────────────────────────────────────

export class Tracers implements System {
  readonly name = 'tracers';

  private ctx: EngineContext | null = null;
  private particles: Particles | null = null;
  private physics: IPhysics | null = null;
  private physicsUsable = true;
  private readonly rng = new Rng(0x2c91d3);
  private readonly unsubscribe: Array<() => void> = [];
  private time = 0;

  // ── tracer batch ───────────────────────────────────────────────────────────
  private tracerGeo: THREE.InstancedBufferGeometry | null = null;
  private tracerMat: THREE.ShaderMaterial | null = null;
  private tracerMesh: THREE.Mesh | null = null;
  private tFrom = new Float32Array(0);
  private tDir = new Float32Array(0);
  private tParam = new Float32Array(0);
  private tColor = new Float32Array(0);
  private tracerAttrs: THREE.InstancedBufferAttribute[] = [];
  private tracerExpiry = new Float32Array(0);
  private tracerHead = 0;
  private tracerHigh = 0;
  private tracerDirtyLo = -1;
  private tracerDirtyHi = -1;

  private readonly tracerUniforms: Record<string, THREE.IUniform> = {
    uTime: { value: 0 },
    // ~2.2 mrad ≈ 2 px at 1080p / 80° fov: the floor below which a bright
    // line starts to alias badly.
    uMinAngular: { value: 0.0022 },
  };

  // ── shell pool ─────────────────────────────────────────────────────────────
  private shellMesh: THREE.InstancedMesh | null = null;
  private shellCount = 0;
  /** pos(3) vel(3) quat(4) angVel(3) floorY(1) = 14 floats per shell. */
  private shellData = new Float32Array(0);
  private shellAge = new Float32Array(0);
  private shellScale = new Float32Array(0);
  private shellBounces = new Uint8Array(0);
  /** 0 = free, 1 = simulated locally, 2 = settled, 3 = driven by physics. */
  private shellState = new Uint8Array(0);
  /** 1 once a settled casing's final matrix has been written to the GPU. */
  private shellSettled = new Uint8Array(0);
  private shellHandle = new Int32Array(0);
  private shellProxy: Array<THREE.Object3D | null> = [];
  private shellCursor = 0;
  private shellPhysicsBodies = 0;

  private readonly mtx = new THREE.Matrix4();
  private readonly vPos = new THREE.Vector3();
  private readonly vScale = new THREE.Vector3();
  private readonly qRot = new THREE.Quaternion();
  private readonly qTmp = new THREE.Quaternion();
  private readonly tmpA = new THREE.Vector3();
  private readonly down = new THREE.Vector3(0, -1, 0);
  private readonly shellColor = new THREE.Color();

  private static readonly STRIDE = 14;
  /** Maximum concurrent rigid bodies handed to the physics system. */
  private static readonly MAX_PHYSICS_SHELLS = 48;

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    this.buildTracers(ctx);
    this.buildShells(ctx);

    this.unsubscribe.push(
      ctx.bus.on('tracer', this.onTracer),
      ctx.bus.on('shell:eject', this.onShellEject),
    );
  }

  update(_ctx: EngineContext, frame: FrameContext): void {
    this.time = frame.elapsed;
    this.tracerUniforms.uTime.value = this.time;
    this.flushTracers();
    this.stepShells(frame.dt);
  }

  dispose(): void {
    for (const off of this.unsubscribe) off();
    this.unsubscribe.length = 0;

    if (this.tracerMesh) this.tracerMesh.removeFromParent();
    this.tracerGeo?.dispose();
    this.tracerMat?.dispose();
    this.tracerGeo = null;
    this.tracerMat = null;
    this.tracerMesh = null;

    if (this.shellMesh) {
      this.shellMesh.removeFromParent();
      // InstancedMesh owns instanceMatrix/instanceColor GPU buffers that the
      // geometry does not; only InstancedMesh.dispose() releases those.
      this.shellMesh.dispose();
      this.shellMesh.geometry.dispose();
      (this.shellMesh.material as THREE.Material).dispose();
      this.shellMesh = null;
    }
    this.releaseAllPhysicsBodies();
  }

  // ── tracers ────────────────────────────────────────────────────────────────

  private buildTracers(ctx: EngineContext): void {
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.instanceCount = 0;

    const mk = (name: string) => {
      const data = new Float32Array(MAX_TRACERS * 4);
      const attr = new THREE.InstancedBufferAttribute(data, 4);
      attr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute(name, attr);
      this.tracerAttrs.push(attr);
      return data;
    };
    this.tFrom = mk('tFrom');
    this.tDir = mk('tDir');
    this.tParam = mk('tParam');
    this.tColor = mk('tColor');
    for (let i = 0; i < MAX_TRACERS; i++) this.tFrom[i * 4 + 3] = -1e9;
    this.tracerExpiry = new Float32Array(MAX_TRACERS);

    this.tracerMat = new THREE.ShaderMaterial({
      uniforms: this.tracerUniforms,
      vertexShader: TRACER_VERT,
      fragmentShader: TRACER_FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      toneMapped: true,
    });
    this.tracerMat.name = 'noisefield.fx.tracers';

    this.tracerGeo = geo;
    this.tracerMesh = new THREE.Mesh(geo, this.tracerMat);
    this.tracerMesh.name = 'noisefield.fx.tracers';
    this.tracerMesh.frustumCulled = false;
    this.tracerMesh.matrixAutoUpdate = false;
    // Above smoke (10) and level with hot particles so glow composites last.
    this.tracerMesh.renderOrder = 13;
    ctx.scene.add(this.tracerMesh);
  }

  private onTracer = (ev: GameEvents['tracer']): void => {
    if (!this.tracerGeo) return;
    const ctx = this.ctx;

    this.tmpA.copy(ev.to).sub(ev.from);
    const dist = this.tmpA.length();
    if (dist < 0.25) return;
    this.tmpA.multiplyScalar(1 / dist);

    // Cheap relevance cull: a tracer whose whole path is far behind the camera
    // cannot contribute anything.
    if (ctx) {
      const mx = (ev.from.x + ev.to.x) * 0.5 - ctx.camera.position.x;
      const my = (ev.from.y + ev.to.y) * 0.5 - ctx.camera.position.y;
      const mz = (ev.from.z + ev.to.z) * 0.5 - ctx.camera.position.z;
      if (mx * mx + my * my + mz * mz > 200 * 200) return;
    }

    const speed = Math.max(TRACER_SPEED_MIN, Math.min(TRACER_SPEED_MAX, ev.speed || 300));
    const streak = Math.max(2.0, Math.min(14, speed * 0.045));

    const slot = this.tracerHead;
    this.tracerHead = (this.tracerHead + 1) % MAX_TRACERS;
    if (this.tracerHead === 0) this.tracerHigh = MAX_TRACERS;
    else this.tracerHigh = Math.max(this.tracerHigh, slot + 1);
    if (this.tracerDirtyLo < 0) {
      this.tracerDirtyLo = slot;
      this.tracerDirtyHi = slot;
    } else {
      if (slot < this.tracerDirtyLo) this.tracerDirtyLo = slot;
      if (slot > this.tracerDirtyHi) this.tracerDirtyHi = slot;
    }

    const o = slot * 4;
    this.tFrom[o] = ev.from.x;
    this.tFrom[o + 1] = ev.from.y;
    this.tFrom[o + 2] = ev.from.z;
    this.tFrom[o + 3] = this.time;

    this.tDir[o] = this.tmpA.x;
    this.tDir[o + 1] = this.tmpA.y;
    this.tDir[o + 2] = this.tmpA.z;
    this.tDir[o + 3] = dist;

    this.tParam[o] = speed;
    this.tParam[o + 1] = streak;
    this.tParam[o + 2] = 0.028;
    this.tParam[o + 3] = this.rng.next();

    // Warm amber core; slight per-round variation stops a burst reading as a
    // single repeated sprite.
    const warm = this.rng.range(0.55, 0.78);
    this.tColor[o] = 1.0;
    this.tColor[o + 1] = warm;
    this.tColor[o + 2] = warm * 0.42;
    this.tColor[o + 3] = this.rng.range(2.6, 4.2);

    this.tracerExpiry[slot] = this.time + (dist + streak) / speed;
  };

  private flushTracers(): void {
    if (!this.tracerGeo) return;
    if (this.tracerDirtyLo >= 0) {
      const start = this.tracerDirtyLo * 4;
      const count = (this.tracerDirtyHi - this.tracerDirtyLo + 1) * 4;
      for (const attr of this.tracerAttrs) {
        attr.clearUpdateRanges();
        attr.addUpdateRange(start, count);
        attr.needsUpdate = true;
      }
      this.tracerDirtyLo = -1;
      this.tracerDirtyHi = -1;
    }
    // Collapse the draw range once every round has landed — the common case
    // between engagements.
    let any = false;
    for (let i = 0; i < this.tracerHigh; i++) {
      if (this.tracerExpiry[i] > this.time) {
        any = true;
        break;
      }
    }
    if (!any) {
      this.tracerHigh = 0;
      this.tracerHead = 0;
    }
    this.tracerGeo.instanceCount = this.tracerHigh;
  }

  // ── shells ─────────────────────────────────────────────────────────────────

  private buildShells(ctx: EngineContext): void {
    const cap = MAX_SHELLS[ctx.settings.tier] ?? 96;
    this.shellCount = cap;
    this.shellData = new Float32Array(cap * Tracers.STRIDE);
    this.shellAge = new Float32Array(cap);
    this.shellScale = new Float32Array(cap);
    this.shellBounces = new Uint8Array(cap);
    this.shellState = new Uint8Array(cap);
    this.shellSettled = new Uint8Array(cap);
    this.shellHandle = new Int32Array(cap).fill(-1);
    this.shellProxy = new Array<THREE.Object3D | null>(cap).fill(null);

    // Low-poly brass case. 8 radial segments is plenty: these are 2 cm long
    // and the read is entirely from the specular glint, not the silhouette.
    const geo = new THREE.CylinderGeometry(0.0047, 0.0042, 0.0195, 8, 1, false);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff, // per-instance colour supplies the brass/plastic tint
      // Deliberately NOT 1.0. A fully metallic surface has zero diffuse
      // response, so if the render system never sets `scene.environment` the
      // casings render as near-black specks and the sun glint — the whole
      // reason these are lit geometry rather than sprites — disappears.
      // 0.86 keeps them unmistakably metallic while retaining a little
      // diffuse fallback under a bare directional light.
      metalness: 0.86,
      roughness: 0.29,
      envMapIntensity: 1.3,
    });
    mat.name = 'noisefield.fx.brass';

    const mesh = new THREE.InstancedMesh(geo, mat, cap);
    mesh.name = 'noisefield.fx.shells';
    // Instance transforms are written every frame from our own integrator, so
    // three's cached bounds are always stale. Culling 120 tiny instances is
    // not worth a bounds recompute.
    mesh.frustumCulled = false;
    mesh.castShadow = ctx.settings.shadows && (ctx.settings.tier === 'high' || ctx.settings.tier === 'ultra');
    mesh.receiveShadow = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    // Park every instance at zero scale; free slots then cost four degenerate
    // triangles each and never need a separate visibility list.
    this.mtx.makeScale(0, 0, 0);
    for (let i = 0; i < cap; i++) {
      mesh.setMatrixAt(i, this.mtx);
      mesh.setColorAt(i, BRASS);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    this.shellMesh = mesh;
    ctx.scene.add(mesh);
  }

  private get P(): Particles | null {
    if (!this.particles) this.particles = this.ctx?.get<Particles>('particles') ?? null;
    return this.particles;
  }

  private get physicsSystem(): IPhysics | null {
    if (!this.physicsUsable) return null;
    if (!this.physics) this.physics = this.ctx?.get<IPhysics>('physics') ?? null;
    return this.physics;
  }

  private onShellEject = (ev: GameEvents['shell:eject']): void => {
    const mesh = this.shellMesh;
    if (!mesh || this.shellCount === 0) return;

    // Round-robin recycle: the oldest casing on screen always yields first.
    const i = this.shellCursor;
    this.shellCursor = (this.shellCursor + 1) % this.shellCount;
    this.retire(i);

    const cal = CALIBER[ev.caliber] ?? CALIBER['5.56'];
    const s = i * Tracers.STRIDE;
    const d = this.shellData;
    const rng = this.rng;

    d[s] = ev.position.x;
    d[s + 1] = ev.position.y;
    d[s + 2] = ev.position.z;
    // A little scatter so a burst does not eject a perfectly straight line.
    d[s + 3] = ev.velocity.x + rng.sym(0.4);
    d[s + 4] = ev.velocity.y + rng.sym(0.3);
    d[s + 5] = ev.velocity.z + rng.sym(0.4);
    // Random initial orientation.
    d[s + 6] = rng.sym(1);
    d[s + 7] = rng.sym(1);
    d[s + 8] = rng.sym(1);
    d[s + 9] = rng.sym(1);
    const qn =
      1 / Math.max(1e-4, Math.hypot(d[s + 6], d[s + 7], d[s + 8], d[s + 9]));
    d[s + 6] *= qn;
    d[s + 7] *= qn;
    d[s + 8] *= qn;
    d[s + 9] *= qn;
    // Fast end-over-end tumble — this is what makes the glint flash.
    d[s + 10] = rng.sym(26);
    d[s + 11] = rng.sym(12);
    d[s + 12] = rng.sym(26);
    d[s + 13] = this.probeFloor(ev.position);

    this.shellAge[i] = 0;
    this.shellScale[i] = cal.scale;
    this.shellBounces[i] = 0;
    this.shellState[i] = 1;
    this.shellSettled[i] = 0;

    this.shellColor.setRGB(cal.r, cal.g, cal.b).multiply(BRASS);
    mesh.setColorAt(i, this.shellColor);
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    this.tryAttachPhysics(i, ev);
    this.ejectGlint(ev);
  };

  /**
   * Finds the floor under an ejection point so casings land on the actual
   * geometry. Without a physics system we assume the compound's ground plane,
   * which is correct for the overwhelming majority of shots.
   */
  private probeFloor(position: THREE.Vector3): number {
    const phys = this.physicsSystem;
    if (phys) {
      try {
        const hit = phys.raycast(position, this.down, 5);
        if (hit) return hit.point.y;
      } catch {
        this.physicsUsable = false;
        this.physics = null;
      }
    }
    return position.y > 0.4 ? 0 : position.y - 1.0;
  }

  /**
   * Hands the casing to the physics system if one exists. Any failure demotes
   * this casing (and, for a hard error, every future casing) to the built-in
   * integrator — the visual result is nearly identical, so a missing or
   * half-built physics system is a non-event.
   */
  private tryAttachPhysics(i: number, ev: GameEvents['shell:eject']): void {
    if (this.shellPhysicsBodies >= Tracers.MAX_PHYSICS_SHELLS) return;
    const phys = this.physicsSystem;
    if (!phys || typeof phys.addDynamic !== 'function') return;

    let proxy = this.shellProxy[i];
    if (!proxy) {
      proxy = new THREE.Object3D();
      proxy.matrixAutoUpdate = false;
      this.shellProxy[i] = proxy;
    }
    proxy.position.copy(ev.position);
    const s = i * Tracers.STRIDE;
    proxy.quaternion.set(
      this.shellData[s + 6],
      this.shellData[s + 7],
      this.shellData[s + 8],
      this.shellData[s + 9],
    );
    proxy.scale.setScalar(1);
    proxy.updateMatrix();

    try {
      const handle = phys.addDynamic(proxy, SHELL_BODY);
      if (typeof handle !== 'number' || handle < 0) return;
      // Impulse is mass * delta-v. Passing the raw velocity here over-drives
      // the casing by 1/mass (~83x at 12 g) and fires it across the map.
      this.tmpA.copy(ev.velocity).multiplyScalar(SHELL_MASS);
      // Book-keep only AFTER the impulse lands. Recording the handle first and
      // then throwing would orphan a live body in the physics world that
      // retire() could never remove.
      phys.applyImpulse(handle, this.tmpA);
      this.shellHandle[i] = handle;
      this.shellState[i] = 3;
      this.shellPhysicsBodies++;
    } catch {
      this.physicsUsable = false;
      this.physics = null;
      this.shellHandle[i] = -1;
      this.shellState[i] = 1;
    }
  }

  /** A brief brass catchlight as the case clears the ejection port. */
  private ejectGlint(ev: GameEvents['shell:eject']): void {
    const P = this.P;
    if (!P) return;
    const d = P.next();
    d.batch = PBatch.Additive;
    d.cell = AtlasCell.Glint;
    d.setPos(ev.position);
    d.setVel(ev.velocity.x * 0.4, ev.velocity.y * 0.4, ev.velocity.z * 0.4);
    d.life = 0.16;
    d.size0 = 0.05;
    d.size1 = 0.012;
    d.setColor(1.0, 0.86, 0.55, 1.1);
    d.gravity = 1;
    d.drag = 1.5;
    d.flags = PFlags.Flicker;
    d.fadePow = 1.4;
    P.commit();
  }

  private retire(i: number): void {
    if (this.shellHandle[i] >= 0) {
      const phys = this.physicsSystem;
      if (phys) {
        try {
          phys.removeBody(this.shellHandle[i]);
        } catch {
          this.physicsUsable = false;
          this.physics = null;
        }
      }
      this.shellHandle[i] = -1;
      this.shellPhysicsBodies = Math.max(0, this.shellPhysicsBodies - 1);
    }
    this.shellState[i] = 0;
  }

  private releaseAllPhysicsBodies(): void {
    for (let i = 0; i < this.shellCount; i++) this.retire(i);
  }

  /**
   * Advances every live casing and writes the instance matrices.
   *
   * The local integrator is semi-implicit Euler with an analytic ground
   * bounce. Casings are tiny and short-lived, so accuracy is irrelevant; what
   * matters is that they tumble fast, bounce twice with a low restitution and
   * then lie still, because a casing that keeps jittering reads as a bug.
   */
  private stepShells(dt: number): void {
    const mesh = this.shellMesh;
    if (!mesh || dt <= 0) return;
    const d = this.shellData;
    const S = Tracers.STRIDE;
    const rng = this.rng;
    let dirty = false;

    for (let i = 0; i < this.shellCount; i++) {
      const state = this.shellState[i];
      if (state === 0) continue;

      const age = (this.shellAge[i] += dt);
      if (age >= SHELL_LIFE) {
        this.retire(i);
        this.mtx.makeScale(0, 0, 0);
        mesh.setMatrixAt(i, this.mtx);
        dirty = true;
        continue;
      }

      const s = i * S;

      if (state === 3) {
        // Physics-driven: read back whatever the system wrote to our proxy.
        const proxy = this.shellProxy[i];
        if (proxy) {
          d[s] = proxy.position.x;
          d[s + 1] = proxy.position.y;
          d[s + 2] = proxy.position.z;
          d[s + 6] = proxy.quaternion.x;
          d[s + 7] = proxy.quaternion.y;
          d[s + 8] = proxy.quaternion.z;
          d[s + 9] = proxy.quaternion.w;
        }
      } else if (state === 1) {
        // Gravity + light air drag.
        d[s + 3] *= 1 - 0.35 * dt;
        d[s + 4] = d[s + 4] - 9.81 * dt;
        d[s + 5] *= 1 - 0.35 * dt;
        d[s] += d[s + 3] * dt;
        d[s + 1] += d[s + 4] * dt;
        d[s + 2] += d[s + 5] * dt;

        const rest = d[s + 13] + 0.006;
        if (d[s + 1] <= rest && d[s + 4] < 0) {
          d[s + 1] = rest;
          const impact = -d[s + 4];
          this.shellBounces[i]++;
          if (impact < 0.55 || this.shellBounces[i] > 3) {
            // Settle: kill all motion and lay the case on its side.
            d[s + 3] = 0;
            d[s + 4] = 0;
            d[s + 5] = 0;
            d[s + 10] = 0;
            d[s + 11] = 0;
            d[s + 12] = 0;
            // Random yaw about the world up axis, otherwise every casing on
            // the floor points the same way and the pile reads as a decal.
            this.qRot.setFromAxisAngle(AXIS_Y, rng.range(0, Math.PI * 2));
            this.qTmp.setFromAxisAngle(AXIS_X, Math.PI * 0.5);
            this.qRot.multiply(this.qTmp);
            d[s + 6] = this.qRot.x;
            d[s + 7] = this.qRot.y;
            d[s + 8] = this.qRot.z;
            d[s + 9] = this.qRot.w;
            this.shellState[i] = 2;
          } else {
            d[s + 4] = impact * 0.32;
            d[s + 3] *= 0.55;
            d[s + 5] *= 0.55;
            d[s + 10] *= 0.45;
            d[s + 11] *= 0.45;
            d[s + 12] *= 0.45;
          }
        }

        // Quaternion integration: q += ½·dt·(ω ⊗ q), then renormalise.
        const hx = 0.5 * dt * d[s + 10];
        const hy = 0.5 * dt * d[s + 11];
        const hz = 0.5 * dt * d[s + 12];
        const qx = d[s + 6];
        const qy = d[s + 7];
        const qz = d[s + 8];
        const qw = d[s + 9];
        let nx = qx + (hx * qw + hy * qz - hz * qy);
        let ny = qy + (hy * qw + hz * qx - hx * qz);
        let nz = qz + (hz * qw + hx * qy - hy * qx);
        let nw = qw - (hx * qx + hy * qy + hz * qz);
        const inv = 1 / Math.max(1e-6, Math.hypot(nx, ny, nz, nw));
        d[s + 6] = nx * inv;
        d[s + 7] = ny * inv;
        d[s + 8] = nz * inv;
        d[s + 9] = nw * inv;
      }

      // Scale-out is the fade. Shrinking an opaque metal instance is free and
      // avoids making the whole InstancedMesh transparent (which would cost a
      // sort and break its depth write).
      let k = this.shellScale[i];
      const remaining = SHELL_LIFE - age;
      const fading = remaining < SHELL_FADE;
      if (fading) k *= Math.max(0, remaining / SHELL_FADE);

      // A settled casing that is not yet fading has not moved, so re-composing
      // and re-uploading its matrix would push up to 120 unchanged matrices to
      // the GPU every frame for the ~10 s it lies there.
      if (state === 2 && !fading && this.shellSettled[i] === 1) continue;
      if (state === 2) this.shellSettled[i] = 1;

      this.vPos.set(d[s], d[s + 1], d[s + 2]);
      this.qRot.set(d[s + 6], d[s + 7], d[s + 8], d[s + 9]);
      this.vScale.setScalar(k);
      this.mtx.compose(this.vPos, this.qRot, this.vScale);
      mesh.setMatrixAt(i, this.mtx);
      dirty = true;
    }

    if (dirty) mesh.instanceMatrix.needsUpdate = true;
  }
}

const AXIS_X = new THREE.Vector3(1, 0, 0);
const AXIS_Y = new THREE.Vector3(0, 1, 0);
/** Cartridge brass, authored in sRGB and converted by ColorManagement. */
const BRASS = new THREE.Color(0xb0873a);
