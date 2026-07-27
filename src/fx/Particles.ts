/**
 * NOISEFIELD — GPU particle system.
 *
 * ARCHITECTURE: ANALYTIC GPU SIMULATION
 * =====================================
 * There is exactly ONE mesh per blend mode (two total) for the entire particle
 * budget. Every particle is an instance of a unit quad in an
 * `InstancedBufferGeometry`, and the CPU does **zero per-particle work per
 * frame**. At spawn time we upload eight vec4 attributes describing the
 * particle's whole trajectory; from then on the vertex shader evaluates its
 * state analytically from `t = uTime - spawnTime`:
 *
 *     dv/dt = g - k*v                        (gravity + linear drag)
 *   ⇒ v(t)  = (v0 - g/k)·e^(-k·t) + g/k
 *   ⇒ p(t)  = p0 + (v0 - g/k)·(1 - e^(-k·t))/k + (g/k)·t
 *
 * with the k→0 limit falling back to the familiar `p0 + v0·t + ½·g·t²`. Using
 * the exact drag integral rather than per-frame Euler matters: sparks and dust
 * decelerate on a proper exponential curve, which is what makes a spark shower
 * read as hot metal rather than confetti.
 *
 * Consequences of this design:
 *   • CPU cost is O(spawns), not O(live particles). 20 000 live particles cost
 *     the CPU nothing at all on a frame with no emission. (The one unavoidable
 *     per-frame allocation is inside three itself: BufferAttribute
 *     .addUpdateRange pushes a {start,count} literal. That is 8 small objects
 *     on a frame that spawned anything, and zero on a frame that did not.)
 *   • Particles cannot collide or be steered after spawn. That is the correct
 *     trade for combat VFX — everything here is ballistic-and-forget.
 *   • Expiry is free: the vertex shader emits a clipped position for dead
 *     slots, so the rasteriser discards them before any fragment work. No
 *     compaction pass, no sorted free list.
 *
 * ALLOCATION: RING BUFFER
 * -----------------------
 * Slots are handed out from a ring. Overflow silently overwrites the oldest
 * particle, which is exactly the behaviour you want under a hard budget: a
 * grenade in a firefight steals from the tail of the dust cloud rather than
 * dropping the grenade's own effect. `instanceCount` tracks a high-water mark
 * and collapses back to zero whenever the system fully drains, so the idle
 * cost is a single empty draw call.
 *
 * BLENDING
 * --------
 * Two batches, because you cannot mix these correctly in one:
 *   • ALPHA (NormalBlending, depthWrite off, depthTest on) — smoke, dust,
 *     debris, blood, glass. Occludes what is behind it.
 *   • ADDITIVE (AdditiveBlending, depthWrite off, depthTest on) — sparks,
 *     embers, muzzle flash, glints. Never darkens, always glows.
 * Additive is drawn after alpha (renderOrder) so hot bits composite over smoke.
 * Within a batch, particles draw in slot order with no per-particle depth sort;
 * for additive that is order-independent and exact, and for alpha the low
 * per-particle opacity makes the sorting error imperceptible. Sorting 20k
 * quads per frame on the CPU would defeat the entire architecture.
 *
 * SOFT PARTICLES
 * --------------
 * See `SOFT_DEPTH` in the fragment shader. If the render system hands us a scene
 * depth texture via {@link Particles.setDepthTexture} we do a true depth-buffer
 * fade. Otherwise we fall back to a per-particle *surface plane* fade plus a
 * camera-proximity fade — documented in detail at the fallback site.
 *
 */

import * as THREE from 'three';
import type { EngineContext, FrameContext, IParticles, ParticleKind, QualityTier } from '../core/Types';
import { AtlasCell, getSharedAtlas, releaseSharedAtlas, type SpriteAtlas } from './textures';

// ─────────────────────────────────────────────────────────────────────────────
// Public vocabulary
// ─────────────────────────────────────────────────────────────────────────────

/** Which blend batch a particle belongs to. */
export const PBatch = { Alpha: 0, Additive: 1 } as const;
export type PBatchId = (typeof PBatch)[keyof typeof PBatch];

/** Per-particle shader feature bits, packed into a single float attribute. */
export const PFlags = {
  None: 0,
  /** Apply sun wrap-lighting + forward scattering. Smoke and dust only. */
  Lit: 1,
  /** Modulate colour by a cooling blackbody ramp over the particle's life. */
  Blackbody: 2,
  /** Cheap per-particle brightness flicker. Embers, sparks, fire. */
  Flicker: 4,
  /**
   * Lie the quad IN the plane carried by `iPlane` instead of facing the
   * camera, offset a few millimetres along its normal. This is what makes
   * spall rings, ground shockwaves and scorch flashes read as decals painted
   * on the surface rather than cards floating in front of it. Mutually
   * exclusive with the plane soft-fade (an in-plane quad has zero plane
   * distance everywhere, so the fade would erase it).
   */
  Oriented: 8,
} as const;

/** Deterministic, allocation-free PRNG (mulberry32). */
export class Rng {
  private s: number;
  constructor(seed = 0x9e3779b9) {
    this.s = seed >>> 0;
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** Uniform in [a,b). */
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }
  /** Uniform in [-a,a). */
  sym(a: number): number {
    return (this.next() * 2 - 1) * a;
  }
  /** Uniform in [a,b) raised to `p` — biases toward `a` for p>1. */
  pow(a: number, b: number, p: number): number {
    return a + (b - a) * Math.pow(this.next(), p);
  }
}

/**
 * Mutable spawn descriptor.
 *
 * WHY A SHARED MUTABLE OBJECT: emitting a 40-spark ricochet must not allocate
 * 40 objects. Callers obtain the single shared instance from
 * {@link Particles.next}, fill the fields they care about, and call
 * {@link Particles.commit}. Every field is reset to a sane default by `next()`,
 * so a caller only writes what differs.
 */
export class SpawnDesc {
  batch: PBatchId = PBatch.Alpha;
  cell: number = AtlasCell.Circle;

  px = 0;
  py = 0;
  pz = 0;
  vx = 0;
  vy = 0;
  vz = 0;

  /** Seconds. */
  life = 1;
  /**
   * Seconds to wait before the particle becomes visible. Costs nothing: the
   * shader already clips particles whose age is negative, so staging an
   * explosion's smoke column over half a second is free.
   */
  delay = 0;
  /** World-space diameter at birth and at death; interpolated with an ease. */
  size0 = 0.2;
  size1 = 0.4;

  /** Linear (not sRGB) tint multiplied with the sprite. */
  r = 1;
  g = 1;
  b = 1;
  /** Peak opacity / additive gain. */
  a = 1;

  /** Multiplier on 9.81 m/s². Negative values float (buoyant smoke). */
  gravity = 0;
  /** Linear drag coefficient, 1/s. ~0.2 heavy debris, ~2.5 fine dust. */
  drag = 0;
  /** Curl-noise displacement amplitude, m/s. Smoke churn. */
  turbulence = 0;
  /** >0 aligns the quad to velocity and elongates it. Sparks, tracers, rain. */
  stretch = 0;

  /** Initial roll (radians) and spin rate (radians/s). */
  rot = 0;
  spin = 0;

  /** Exponent on the (1-u) opacity ramp. >1 lingers, <1 vanishes early. */
  fadePow = 1.6;
  /** Bitwise OR of {@link PFlags}. */
  flags = 0;

  /**
   * Soft-particle surface plane, `dot(n, x) = d`, in world space. Leave the
   * normal at zero length to disable the plane fade.
   */
  planeNx = 0;
  planeNy = 0;
  planeNz = 0;
  planeD = 0;
  /** Soft-fade distance in metres, for both the depth and plane paths. */
  soft = 0.3;

  reset(): this {
    this.batch = PBatch.Alpha;
    this.cell = AtlasCell.Circle;
    this.px = this.py = this.pz = 0;
    this.vx = this.vy = this.vz = 0;
    this.life = 1;
    this.delay = 0;
    this.size0 = 0.2;
    this.size1 = 0.4;
    this.r = this.g = this.b = this.a = 1;
    this.gravity = 0;
    this.drag = 0;
    this.turbulence = 0;
    this.stretch = 0;
    this.rot = 0;
    this.spin = 0;
    this.fadePow = 1.6;
    this.flags = 0;
    this.planeNx = this.planeNy = this.planeNz = this.planeD = 0;
    this.soft = 0.3;
    return this;
  }

  setPos(v: THREE.Vector3): this {
    this.px = v.x;
    this.py = v.y;
    this.pz = v.z;
    return this;
  }
  setVel(x: number, y: number, z: number): this {
    this.vx = x;
    this.vy = y;
    this.vz = z;
    return this;
  }
  setColor(r: number, g: number, b: number, a = 1): this {
    this.r = r;
    this.g = g;
    this.b = b;
    this.a = a;
    return this;
  }
  /**
   * Anchors the soft-particle fade to the surface that `point`/`normal` came
   * from.
   *
   * The plane is placed half a soft-radius BEHIND the surface, which centres
   * the fade band ON the intersection. A fragment exactly at the impact point
   * evaluates to 0.5 opacity, rises to 1 half a radius in front, and reaches 0
   * half a radius behind — so the billboard dissolves smoothly through the
   * wall instead of being sliced by it. Anchoring the plane at the surface
   * itself would make every impact puff spawn fully transparent; anchoring it
   * a full radius behind (the previous behaviour) put the entire gradient
   * inside the geometry, where the depth test has already hidden it.
   */
  setPlane(point: THREE.Vector3, normal: THREE.Vector3, soft = 0.3): this {
    this.planeNx = normal.x;
    this.planeNy = normal.y;
    this.planeNz = normal.z;
    this.planeD = normal.x * point.x + normal.y * point.y + normal.z * point.z - soft * 0.5;
    this.soft = soft;
    return this;
  }

  /** Sets the surface plane used by {@link PFlags.Oriented} without any fade. */
  setOrientation(normal: THREE.Vector3): this {
    this.planeNx = normal.x;
    this.planeNy = normal.y;
    this.planeNz = normal.z;
    this.planeD = 0;
    return this;
  }
}

/**
 * Builds a unit vector inside a cone of half-angle `spread` (radians) around
 * `axis`. Allocation free — writes into `out`.
 */
export function randomCone(
  axis: THREE.Vector3,
  spread: number,
  rng: Rng,
  out: THREE.Vector3,
): THREE.Vector3 {
  // Branchless orthonormal basis (Duff et al. 2017) — no degenerate case at
  // the poles, unlike the naive "cross with up" construction.
  const sign = axis.z >= 0 ? 1 : -1;
  const a = -1 / (sign + axis.z);
  const b = axis.x * axis.y * a;
  const t1x = 1 + sign * axis.x * axis.x * a;
  const t1y = sign * b;
  const t1z = -sign * axis.x;
  const t2x = b;
  const t2y = sign + axis.y * axis.y * a;
  const t2z = -axis.y;

  const cosMax = Math.cos(Math.min(spread, Math.PI));
  const cosT = 1 - rng.next() * (1 - cosMax);
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  const phi = rng.next() * Math.PI * 2;
  const cx = Math.cos(phi) * sinT;
  const cy = Math.sin(phi) * sinT;

  out.set(
    t1x * cx + t2x * cy + axis.x * cosT,
    t1y * cx + t2y * cy + axis.y * cosT,
    t1z * cx + t2z * cy + axis.z * cosT,
  );
  return out.normalize();
}

// ─────────────────────────────────────────────────────────────────────────────
// Shaders
// ─────────────────────────────────────────────────────────────────────────────

const PARTICLE_VERT = /* glsl */ `
precision highp float;

attribute vec4 iPosLife;   // xyz = spawn position, w = lifetime
attribute vec4 iVelSpawn;  // xyz = initial velocity, w = spawn time
attribute vec4 iSizeRot;   // x = size0, y = size1, z = roll0, w = spin
attribute vec4 iColor;     // rgb tint (linear), a = peak opacity
attribute vec4 iPhys;      // x = gravityScale, y = drag, z = turbulence, w = stretch
attribute vec4 iAtlas;     // xy = uv origin, zw = uv size
attribute vec4 iMisc;      // x = seed, y = flags, z = fadePow, w = softRadius
attribute vec4 iPlane;     // xyz = surface normal, w = plane constant

uniform float uTime;
uniform float uSizeScale;

varying vec2  vUv;
varying vec2  vLocal;
varying vec4  vColor;
varying float vLife;
varying float vFlags;
varying vec3  vWorld;
varying vec4  vPlane;
varying float vSoft;
varying float vViewDepth;
varying float vSeed;

const float GRAV = 9.81;

/**
 * Analytic curl of the potential field
 *   P(q) = ( sin(q.y)cos(q.z), sin(q.z)cos(q.x), sin(q.x)cos(q.y) ).
 * Being an exact curl it is divergence-free, so the flow swirls and folds
 * instead of inflating — that is the difference between smoke that churns and
 * smoke that just gets bigger.
 */
vec3 curlField(vec3 q) {
  float sx = sin(q.x), cx = cos(q.x);
  float sy = sin(q.y), cy = cos(q.y);
  float sz = sin(q.z), cz = cos(q.z);
  return vec3(
    -sx * sy - cz * cx,
    -sy * sz - cx * cy,
    -sz * sx - cy * cz
  );
}

void main() {
  float life  = max(iPosLife.w, 1e-4);
  float t     = uTime - iVelSpawn.w;
  float u     = t / life;

  // Dead or unborn: emit a position guaranteed to fail the clip test. This is
  // the cheapest possible retirement — no CPU compaction, no index rebuild.
  if (t < 0.0 || u >= 1.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(0.0);
    vLocal = vec2(0.0);
    vColor = vec4(0.0);
    vLife = 1.0;
    vFlags = 0.0;
    vWorld = vec3(0.0);
    vPlane = vec4(0.0);
    vSoft = 1.0;
    vViewDepth = 1.0;
    vSeed = 0.0;
    return;
  }

  // ── analytic ballistics with exact linear drag ───────────────────────────
  vec3  g = vec3(0.0, -GRAV * iPhys.x, 0.0);
  float k = iPhys.y;
  vec3  p;
  vec3  vel;
  if (k > 1e-3) {
    vec3  term = g / k;
    float e    = exp(-k * t);
    p   = iPosLife.xyz + (iVelSpawn.xyz - term) * ((1.0 - e) / k) + term * t;
    vel = (iVelSpawn.xyz - term) * e + term;
  } else {
    p   = iPosLife.xyz + iVelSpawn.xyz * t + 0.5 * g * t * t;
    vel = iVelSpawn.xyz + g * t;
  }

  // ── turbulence ───────────────────────────────────────────────────────────
  // Displacement grows with age so a puff leaves the muzzle coherent and only
  // dissolves into churn as it ages.
  float turb = iPhys.z;
  if (turb > 0.0) {
    vec3 q = p * 0.55 + vec3(iMisc.x * 17.3, iMisc.x * 29.1, iMisc.x * 41.7);
    vec3 c = curlField(q + vec3(uTime * 0.28));
    p   += c * turb * t;
    vel += c * turb * 0.5;
  }

  float s = mix(iSizeRot.x, iSizeRot.y, pow(u, 0.55)) * uSizeScale;

  float fadeIn  = smoothstep(0.0, 0.06, u);
  float fadeOut = pow(1.0 - u, max(iMisc.z, 0.05));
  float alpha   = iColor.a * fadeIn * fadeOut;

  // Camera basis in world space. viewMatrix is world→view, so the columns of
  // its transpose (i.e. these rows) are the camera's axes expressed in world
  // space. Cheaper and more precise than inverting the matrix.
  vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 camUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);

  // Surface-aligned mode: build the quad basis from the carried plane normal
  // so rings and scorch flashes sit ON the geometry.
  float flags    = floor(iMisc.y + 0.5);
  float oriented = mod(floor(flags / 8.0), 2.0);
  float pnLen    = length(iPlane.xyz);
  vec3  sn       = pnLen > 1e-4 ? iPlane.xyz / pnLen : vec3(0.0, 1.0, 0.0);
  vec3  tang     = abs(sn.y) < 0.985
                 ? normalize(cross(vec3(0.0, 1.0, 0.0), sn))
                 : vec3(1.0, 0.0, 0.0);
  vec3  bitan    = cross(sn, tang);
  vec3  basisR   = mix(camRight, tang, oriented);
  vec3  basisU   = mix(camUp, bitan, oriented);

  vec2 corner = position.xy;                 // -0.5 .. 0.5
  float rot   = iSizeRot.z + iSizeRot.w * t;
  float cr    = cos(rot), sr = sin(rot);
  vec2  spun  = vec2(corner.x * cr - corner.y * sr, corner.x * sr + corner.y * cr) * s;

  // Velocity-aligned elongation. Round dots for sparks look wrong; real sparks
  // are motion-blurred streaks, so we align local +Y to the screen-space
  // velocity and scale the quad's length by speed.
  float stretch = iPhys.w;
  float speed   = length(vel);
  vec2  sv      = vec2(dot(vel, basisR), dot(vel, basisU));
  float svLen   = length(sv);
  vec2  axis    = svLen > 1e-4 ? sv / svLen : vec2(0.0, 1.0);
  vec2  perp    = vec2(-axis.y, axis.x);
  float longLen = s * (1.0 + stretch * min(speed, 120.0) * 0.14);
  vec2  streak  = perp * (corner.x * s) + axis * (corner.y * longLen);

  vec2 off   = mix(spun, streak, step(0.001, stretch));
  // 12 mm normal offset keeps oriented quads off the surface they hug without
  // being visible as a gap even at grazing angles.
  vec3 world = p + basisR * off.x + basisU * off.y + sn * (oriented * 0.012);

  vec4 mv = viewMatrix * vec4(world, 1.0);
  gl_Position = projectionMatrix * mv;

  vUv        = iAtlas.xy + uv * iAtlas.zw;
  vLocal     = uv * 2.0 - 1.0;
  vColor     = vec4(iColor.rgb, alpha);
  vLife      = u;
  vFlags     = iMisc.y;
  vWorld     = world;
  vPlane     = iPlane;
  vSoft      = max(iMisc.w, 1e-3);
  vViewDepth = -mv.z;
  vSeed      = iMisc.x;
}
`;

const PARTICLE_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform sampler2D uAtlas;
uniform vec3  uSunDir;      // unit vector pointing TOWARD the sun
uniform vec3  uSunColor;    // linear radiance
uniform vec3  uAmbient;     // linear sky/bounce term
uniform float uWrap;        // wrap-lighting softness, 0..1
uniform float uNearFade;    // metres over which we fade in front of the camera
uniform vec2  uResolution;  // drawing-buffer size in pixels
uniform float uNear;
uniform float uFar;

#ifdef SOFT_DEPTH
uniform sampler2D uDepth;
#endif

varying vec2  vUv;
varying vec2  vLocal;
varying vec4  vColor;
varying float vLife;
varying float vFlags;
varying vec3  vWorld;
varying vec4  vPlane;
varying float vSoft;
varying float vViewDepth;
varying float vSeed;

bool hasFlag(float flags, float bit) {
  return mod(floor(flags / bit), 2.0) >= 0.5;
}

/** Cheap blackbody-ish cooling ramp: 1 = white hot, 0 = black. */
vec3 blackbody(float t) {
  return vec3(
    clamp(t * 3.2, 0.0, 1.0),
    clamp(t * t * 2.1 - 0.10, 0.0, 1.0),
    clamp(pow(t, 3.5) * 2.4 - 0.22, 0.0, 1.0)
  );
}

void main() {
  vec4 tex = texture2D(uAtlas, vUv);

  float flags    = floor(vFlags + 0.5);
  float oriented = mod(floor(flags / 8.0), 2.0);

  // ── soft particles ───────────────────────────────────────────────────────
  float soft = 1.0;
#ifdef SOFT_DEPTH
  // True depth-buffer soft particles: fade out as the billboard approaches the
  // opaque surface behind it, killing the hard intersection line.
  vec2  suv = gl_FragCoord.xy / uResolution;
  float dz  = texture2D(uDepth, suv).x;
  float ndc = dz * 2.0 - 1.0;
  float sceneZ = (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
  soft = clamp((sceneZ - vViewDepth) / vSoft, 0.0, 1.0);
#else
  // FALLBACK — DOCUMENTED LIMITATION.
  // The render system does not (yet) publish a scene depth texture to us, so a
  // true depth comparison is impossible without an extra full-scene depth
  // prepass, which we refuse to pay for. Instead each impact-spawned particle
  // carries the PLANE of the surface it was born on and we fade the fragment
  // as it crosses behind that plane. This removes the hard straight cut where
  // a dust puff meets its own wall or floor — empirically the ~90% case — at
  // zero extra bandwidth and one dot product. Wire up the real thing with
  // Particles.setDepthTexture() and this branch compiles out.
  float planeDist = dot(vWorld, vPlane.xyz) - vPlane.w;
  // Surface-aligned quads live IN the plane, so their plane distance is ~0
  // everywhere and the fade would erase them. Exclude them explicitly.
  float usePlane  = step(0.5, length(vPlane.xyz)) * (1.0 - oriented);
  soft = mix(1.0, clamp(planeDist / vSoft, 0.0, 1.0), usePlane);
#endif
  // Always fade particles that get very close to the near plane; otherwise the
  // camera "swallows" a puff and the screen flashes a full-screen grey card.
  soft *= smoothstep(0.0, uNearFade, vViewDepth);

  float a = tex.a * vColor.a * soft;
  if (a < 0.003) discard;

  vec3 col = vColor.rgb * tex.rgb;

  if (hasFlag(flags, 2.0)) {
    // Hot fragments cool over their lifetime: white → orange → deep red.
    col *= blackbody(1.0 - vLife);
  }

  if (hasFlag(flags, 1.0)) {
    vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
    vec3 camUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
    vec3 camFwd   = vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);

    // Fake hemisphere normal across the billboard so the puff shades like a
    // sphere rather than a flat card. Costs one normalize. Surface-aligned
    // quads use their real plane normal instead.
    float r2 = min(dot(vLocal, vLocal), 1.0);
    vec3  sphereN = normalize(camRight * vLocal.x + camUp * vLocal.y + camFwd * (sqrt(1.0 - r2) * 0.9));
    vec3  n = mix(sphereN, normalize(vPlane.xyz + vec3(0.0, 1e-5, 0.0)), oriented);

    float ndl  = dot(n, uSunDir);
    float wrap = clamp((ndl + uWrap) / (1.0 + uWrap), 0.0, 1.0);

    // Forward scattering. Looking through a puff toward a low sun makes it
    // glow — this single term is what sells backlit golden-hour dust.
    // viewDir points camera -> fragment, and uSunDir points toward the sun,
    // so looking THROUGH a puff at the sun makes these agree. Negating either
    // one inverts the effect and front-lights the smoke instead, which is the
    // exact opposite of the backlit golden-hour look.
    vec3  viewDir = normalize(vWorld - cameraPosition);
    float fwd     = pow(clamp(dot(uSunDir, viewDir), 0.0, 1.0), 5.0);

    // The sprite's RGB doubles as an internal density field, giving cheap
    // self-shadowing: dense lumps take more sun, thin edges stay ambient.
    float dens = tex.r;
    vec3  lit  = uAmbient * (0.55 + 0.45 * dens)
               + uSunColor * (wrap * (0.35 + 0.85 * dens) + fwd * 1.35);
    col *= lit;
  }

  if (hasFlag(flags, 4.0)) {
    // Hash on the PER-PARTICLE seed, not on the flag word: flags are constant
    // across a whole emitter, so hashing them makes every ember in a burst
    // flicker in lockstep, which reads as a strobe rather than as fire.
    col *= 0.70 + 0.42 * fract(sin(vLife * 21.0 + vSeed * 53.7) * 43758.5453);
  }

  gl_FragColor = vec4(col, a);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Batch
// ─────────────────────────────────────────────────────────────────────────────

/** Number of vec4 instanced attributes per particle. */
const ATTRS = 8;
const FLOATS_PER_PARTICLE = ATTRS * 4;

interface BatchArrays {
  posLife: Float32Array;
  velSpawn: Float32Array;
  sizeRot: Float32Array;
  color: Float32Array;
  phys: Float32Array;
  atlas: Float32Array;
  misc: Float32Array;
  plane: Float32Array;
}

/**
 * One blend-mode's worth of particles: a ring-allocated instanced quad mesh.
 * Everything is preallocated; `alloc` never grows and never allocates.
 */
class ParticleBatch {
  readonly geometry: THREE.InstancedBufferGeometry;
  readonly material: THREE.ShaderMaterial;
  readonly mesh: THREE.Mesh;
  readonly capacity: number;

  private readonly attrs: THREE.InstancedBufferAttribute[] = [];
  private readonly arr: BatchArrays;
  /** Absolute time at which each slot dies; used for liveness accounting. */
  private readonly expiry: Float32Array;

  private head = 0;
  private wrapped = false;
  private dirtyLo = -1;
  private dirtyHi = -1;
  private dirtyFull = false;
  private live = 0;

  constructor(capacity: number, additive: boolean, uniforms: Record<string, THREE.IUniform>) {
    this.capacity = capacity;

    const geo = new THREE.InstancedBufferGeometry();
    // Indexed unit quad: 4 vertices, 2 triangles. Fewer vertex invocations than
    // a 6-vertex soup, which matters at 20k instances (80k vs 120k).
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.instanceCount = 0;

    const mk = (name: string): Float32Array => {
      const data = new Float32Array(capacity * 4);
      const attr = new THREE.InstancedBufferAttribute(data, 4);
      attr.setUsage(THREE.DynamicDrawUsage);
      geo.setAttribute(name, attr);
      this.attrs.push(attr);
      return data;
    };

    this.arr = {
      posLife: mk('iPosLife'),
      velSpawn: mk('iVelSpawn'),
      sizeRot: mk('iSizeRot'),
      color: mk('iColor'),
      phys: mk('iPhys'),
      atlas: mk('iAtlas'),
      misc: mk('iMisc'),
      plane: mk('iPlane'),
    };

    // Lifetime 0 + spawn time far in the past ⇒ every slot starts retired.
    for (let i = 0; i < capacity; i++) this.arr.velSpawn[i * 4 + 3] = -1e9;

    this.expiry = new Float32Array(capacity);

    this.material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: PARTICLE_VERT,
      fragmentShader: PARTICLE_FRAG,
      transparent: true,
      depthTest: true,
      // Particles never write depth: they are volumetric and unsorted, so
      // depth writes would make them occlude each other in slot order.
      depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      side: THREE.DoubleSide,
      // Matches built-in materials: three disables tone mapping automatically
      // when rendering into a post-processing target, so this is correct for
      // both the direct-to-canvas and EffectComposer paths.
      toneMapped: true,
    });
    this.material.name = additive ? 'noisefield.fx.particles.additive' : 'noisefield.fx.particles.alpha';

    this.geometry = geo;
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.name = this.material.name;
    // Positions are synthesised in the vertex shader, so three's bounding
    // sphere is meaningless — culling it would pop the whole system out.
    this.mesh.frustumCulled = false;
    // The vertex shader assumes modelMatrix == identity (it transforms world
    // positions straight through viewMatrix). Freeze the transform.
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = additive ? 12 : 10;
  }

  get liveCount(): number {
    return this.live;
  }

  /** Claims the next ring slot. Always succeeds; overwrites the oldest entry. */
  alloc(): number {
    const slot = this.head;
    this.head++;
    if (this.head >= this.capacity) {
      this.head = 0;
      this.wrapped = true;
      this.dirtyFull = true;
    }
    if (this.dirtyLo < 0) {
      this.dirtyLo = slot;
      this.dirtyHi = slot;
    } else {
      if (slot < this.dirtyLo) this.dirtyLo = slot;
      if (slot > this.dirtyHi) this.dirtyHi = slot;
    }
    return slot;
  }

  /**
   * @param spawnTime absolute time the particle becomes visible; may be in the
   *   future (the shader clips negative ages), which is how staged effects
   *   such as an explosion's rising smoke column are scheduled for free.
   */
  write(slot: number, d: SpawnDesc, spawnTime: number, rect: THREE.Vector4, seed: number): void {
    const o = slot * 4;
    const a = this.arr;

    a.posLife[o] = d.px;
    a.posLife[o + 1] = d.py;
    a.posLife[o + 2] = d.pz;
    a.posLife[o + 3] = d.life;

    a.velSpawn[o] = d.vx;
    a.velSpawn[o + 1] = d.vy;
    a.velSpawn[o + 2] = d.vz;
    a.velSpawn[o + 3] = spawnTime;

    a.sizeRot[o] = d.size0;
    a.sizeRot[o + 1] = d.size1;
    a.sizeRot[o + 2] = d.rot;
    a.sizeRot[o + 3] = d.spin;

    a.color[o] = d.r;
    a.color[o + 1] = d.g;
    a.color[o + 2] = d.b;
    a.color[o + 3] = d.a;

    a.phys[o] = d.gravity;
    a.phys[o + 1] = d.drag;
    a.phys[o + 2] = d.turbulence;
    a.phys[o + 3] = d.stretch;

    a.atlas[o] = rect.x;
    a.atlas[o + 1] = rect.y;
    a.atlas[o + 2] = rect.z;
    a.atlas[o + 3] = rect.w;

    a.misc[o] = seed;
    a.misc[o + 1] = d.flags;
    a.misc[o + 2] = d.fadePow;
    a.misc[o + 3] = d.soft;

    a.plane[o] = d.planeNx;
    a.plane[o + 1] = d.planeNy;
    a.plane[o + 2] = d.planeNz;
    a.plane[o + 3] = d.planeD;

    this.expiry[slot] = spawnTime + d.life;
  }

  /** Uploads only the slots touched this frame and updates the draw count. */
  flush(): void {
    if (this.dirtyLo >= 0) {
      const lo = this.dirtyFull ? 0 : this.dirtyLo;
      const hi = this.dirtyFull ? this.capacity - 1 : this.dirtyHi;
      const start = lo * 4;
      const count = (hi - lo + 1) * 4;
      for (const attr of this.attrs) {
        attr.clearUpdateRanges();
        attr.addUpdateRange(start, count);
        attr.needsUpdate = true;
      }
      this.dirtyLo = -1;
      this.dirtyHi = -1;
      this.dirtyFull = false;
    }
    this.geometry.instanceCount = this.wrapped ? this.capacity : this.head;
  }

  /**
   * Recounts live particles and collapses the ring when the system drains.
   *
   * This is the only per-particle CPU touch in the whole system, it is a flat
   * typed-array compare with no branching on object state, and callers run it
   * on a stride (every Nth frame). At 20k slots it is ~20 µs.
   */
  recount(now: number): void {
    const n = this.wrapped ? this.capacity : this.head;
    let live = 0;
    const e = this.expiry;
    for (let i = 0; i < n; i++) if (e[i] > now) live++;
    this.live = live;
    if (live === 0 && this.dirtyLo < 0) {
      this.head = 0;
      this.wrapped = false;
      this.geometry.instanceCount = 0;
    }
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kind profiles
// ─────────────────────────────────────────────────────────────────────────────

interface KindProfile {
  batch: PBatchId;
  cell: number;
  /** Base sizes; scaled by intensity at the call site. */
  size0: number;
  size1: number;
  life: number;
  speed: number;
  spread: number;
  gravity: number;
  drag: number;
  turbulence: number;
  stretch: number;
  flags: number;
  fadePow: number;
  /** Linear RGB tint. */
  r: number;
  g: number;
  b: number;
  a: number;
  /** How many particles a `burst` of intensity 1 emits. */
  count: number;
}

/**
 * Emitter presets behind the coarse {@link IParticles} API. Impacts and
 * Volumetrics bypass these and drive {@link SpawnDesc} directly for finer art
 * control; these exist so any other system can fire a decent-looking effect with
 * one call.
 */
const PROFILES: Record<ParticleKind, KindProfile> = {
  spark: {
    batch: PBatch.Additive, cell: AtlasCell.Spark,
    size0: 0.05, size1: 0.012, life: 0.55, speed: 7.5, spread: 0.95,
    gravity: 1.0, drag: 1.6, turbulence: 0, stretch: 0.9,
    flags: PFlags.Blackbody | PFlags.Flicker, fadePow: 1.1,
    r: 1.0, g: 0.72, b: 0.34, a: 2.4, count: 14,
  },
  ember: {
    batch: PBatch.Additive, cell: AtlasCell.Ember,
    size0: 0.09, size1: 0.03, life: 1.6, speed: 2.2, spread: 1.2,
    gravity: -0.12, drag: 1.1, turbulence: 0.55, stretch: 0.25,
    flags: PFlags.Blackbody | PFlags.Flicker, fadePow: 1.5,
    r: 1.0, g: 0.55, b: 0.2, a: 1.5, count: 8,
  },
  dust: {
    batch: PBatch.Alpha, cell: AtlasCell.Dust,
    size0: 0.22, size1: 1.1, life: 1.5, speed: 2.0, spread: 1.05,
    gravity: 0.06, drag: 2.0, turbulence: 0.30, stretch: 0,
    flags: PFlags.Lit, fadePow: 1.9,
    r: 0.76, g: 0.66, b: 0.47, a: 0.42, count: 12,
  },
  smoke: {
    batch: PBatch.Alpha, cell: AtlasCell.Smoke,
    size0: 0.35, size1: 2.4, life: 2.8, speed: 1.4, spread: 0.9,
    gravity: -0.05, drag: 1.1, turbulence: 0.55, stretch: 0,
    flags: PFlags.Lit, fadePow: 2.0,
    r: 0.58, g: 0.55, b: 0.52, a: 0.40, count: 8,
  },
  muzzleSmoke: {
    batch: PBatch.Alpha, cell: AtlasCell.SmokeWisp,
    size0: 0.10, size1: 0.72, life: 1.05, speed: 2.6, spread: 0.35,
    gravity: -0.08, drag: 2.6, turbulence: 0.45, stretch: 0,
    flags: PFlags.Lit, fadePow: 2.2,
    r: 0.70, g: 0.66, b: 0.60, a: 0.30, count: 6,
  },
  blood: {
    batch: PBatch.Alpha, cell: AtlasCell.Blood,
    size0: 0.05, size1: 0.03, life: 0.85, speed: 5.0, spread: 0.55,
    gravity: 1.0, drag: 0.7, turbulence: 0, stretch: 0.5,
    flags: 0, fadePow: 1.3,
    r: 0.42, g: 0.035, b: 0.03, a: 1.0, count: 12,
  },
  debris: {
    batch: PBatch.Alpha, cell: AtlasCell.Debris,
    size0: 0.045, size1: 0.035, life: 1.5, speed: 5.5, spread: 0.85,
    gravity: 1.0, drag: 0.28, turbulence: 0, stretch: 0,
    flags: PFlags.Lit, fadePow: 0.9,
    r: 0.60, g: 0.56, b: 0.50, a: 1.0, count: 8,
  },
  glass: {
    batch: PBatch.Alpha, cell: AtlasCell.Glass,
    size0: 0.05, size1: 0.04, life: 1.4, speed: 6.0, spread: 0.9,
    gravity: 1.0, drag: 0.3, turbulence: 0, stretch: 0,
    flags: 0, fadePow: 1.0,
    r: 0.85, g: 0.94, b: 1.0, a: 0.9, count: 14,
  },
  water: {
    batch: PBatch.Alpha, cell: AtlasCell.Water,
    size0: 0.06, size1: 0.04, life: 0.9, speed: 5.5, spread: 0.7,
    gravity: 1.0, drag: 0.55, turbulence: 0, stretch: 0.45,
    flags: 0, fadePow: 1.2,
    r: 0.72, g: 0.83, b: 0.9, a: 0.85, count: 16,
  },
  explosion: {
    batch: PBatch.Additive, cell: AtlasCell.Ember,
    size0: 0.8, size1: 4.0, life: 0.7, speed: 7.0, spread: 1.4,
    gravity: -0.3, drag: 2.2, turbulence: 1.2, stretch: 0,
    flags: PFlags.Flicker, fadePow: 1.8,
    r: 1.0, g: 0.5, b: 0.18, a: 2.2, count: 16,
  },
  /**
   * Shell casings are real lit geometry owned by Tracers, not sprites. This
   * profile only exists so a stray `burst('shell', …)` degrades to a plausible
   * brass glint instead of throwing.
   */
  shell: {
    batch: PBatch.Additive, cell: AtlasCell.Glint,
    size0: 0.05, size1: 0.02, life: 0.4, speed: 2.5, spread: 0.6,
    gravity: 1.0, drag: 0.6, turbulence: 0, stretch: 0.2,
    flags: 0, fadePow: 1.2,
    r: 1.0, g: 0.85, b: 0.5, a: 1.2, count: 2,
  },
};

/** Per-tier ceilings. The engine's `particleBudget` is an upper bound on these. */
const TIER_CAP: Record<QualityTier, number> = {
  low: 2500,
  medium: 7000,
  high: 14000,
  ultra: 20000,
};

const TIER_SPAWN_SCALE: Record<QualityTier, number> = {
  low: 0.32,
  medium: 0.6,
  high: 0.85,
  ultra: 1.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// System
// ─────────────────────────────────────────────────────────────────────────────

export class Particles implements IParticles {
  readonly name = 'particles';

  /** Shared sprite atlas; Tracers and Volumetrics reuse this texture. */
  atlas: SpriteAtlas | null = null;

  /**
   * Multiplier applied by emitters to their requested particle counts, derived
   * from the quality tier. Public so Impacts/Volumetrics scale consistently.
   */
  spawnScale = 1;

  readonly rng = new Rng(0xa9e1);

  private ctx: EngineContext | null = null;
  private alpha: ParticleBatch | null = null;
  private additive: ParticleBatch | null = null;
  private readonly desc = new SpawnDesc();
  private pendingBatch: PBatchId = PBatch.Alpha;

  private time = 0;
  private frameIndex = 0;
  private cachedActive = 0;
  /** Latched once another system calls setSun(); stops the auto-probe fighting it. */
  private sunOverridden = false;
  /** Latched when setDepthTexture() supplied an explicit near/far pair. */
  private depthRangeOverridden = false;

  private readonly rect = new THREE.Vector4();
  private readonly tmpDir = new THREE.Vector3();
  private readonly tmpVec = new THREE.Vector3();
  private readonly drawBuffer = new THREE.Vector2(1, 1);

  /** Shared uniform block — one object referenced by both batch materials. */
  private readonly uniforms: Record<string, THREE.IUniform> = {
    uAtlas: { value: null },
    uTime: { value: 0 },
    uSizeScale: { value: 1 },
    uSunDir: { value: new THREE.Vector3(-0.42, 0.28, -0.86).normalize() },
    uSunColor: { value: new THREE.Color(0xffd2a1).multiplyScalar(2.1) },
    uAmbient: { value: new THREE.Color(0x1a2028).multiplyScalar(1.35) },
    uWrap: { value: 0.62 },
    uNearFade: { value: 0.55 },
    uResolution: { value: new THREE.Vector2(1, 1) },
    uNear: { value: 0.05 },
    uFar: { value: 800 },
    uDepth: { value: null },
  };

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    const tier = ctx.settings.tier;
    const budget = Math.max(
      256,
      Math.min(ctx.settings.particleBudget | 0, TIER_CAP[tier] ?? TIER_CAP.high),
    );
    this.spawnScale = TIER_SPAWN_SCALE[tier] ?? 1;

    this.atlas = getSharedAtlas(tier === 'low' || tier === 'medium' ? 128 : 256);
    this.uniforms.uAtlas.value = this.atlas.texture;
    this.uniforms.uNear.value = ctx.camera.near;
    this.uniforms.uFar.value = ctx.camera.far;

    // 55/45 split: smoke and dust are the volume hogs, sparks are numerous but
    // short-lived. Both batches share one uniform block so a single sun update
    // reaches everything.
    const alphaCap = Math.max(128, Math.round(budget * 0.55));
    const addCap = Math.max(128, budget - alphaCap);

    this.alpha = new ParticleBatch(alphaCap, false, this.uniforms);
    this.additive = new ParticleBatch(addCap, true, this.uniforms);
    ctx.scene.add(this.alpha.mesh, this.additive.mesh);

    this.syncResolution(ctx);
    this.findSun(ctx);
  }

  update(ctx: EngineContext, frame: FrameContext): void {
    this.time = frame.elapsed;
    this.uniforms.uTime.value = this.time;
    this.frameIndex = frame.frame;

    // The drawing buffer can change without a window resize (renderScale), so
    // refresh it cheaply rather than relying on resize() alone.
    if ((frame.frame & 15) === 0) this.syncResolution(ctx);
    // Re-probe the sun occasionally: the world/render system may add or retarget
    // its directional light after we initialised.
    if (frame.frame % 47 === 0 && !this.sunOverridden) this.findSun(ctx);

    this.alpha?.flush();
    this.additive?.flush();

    // Liveness accounting on a stride — this is the only per-particle CPU work
    // in the system and it exists purely to report activeCount and to collapse
    // the draw range when combat stops.
    if ((frame.frame & 3) === 0) {
      this.alpha?.recount(this.time);
      this.additive?.recount(this.time);
      this.cachedActive = (this.alpha?.liveCount ?? 0) + (this.additive?.liveCount ?? 0);
    }
  }

  resize(): void {
    if (this.ctx) this.syncResolution(this.ctx);
  }

  dispose(): void {
    if (this.alpha) {
      this.alpha.mesh.removeFromParent();
      this.alpha.dispose();
    }
    if (this.additive) {
      this.additive.mesh.removeFromParent();
      this.additive.dispose();
    }
    this.alpha = null;
    this.additive = null;
    if (this.atlas) {
      releaseSharedAtlas();
      this.atlas = null;
    }
  }

  // ── IParticles ─────────────────────────────────────────────────────────────

  get activeCount(): number {
    return this.cachedActive;
  }

  burst(kind: ParticleKind, position: THREE.Vector3, normal: THREE.Vector3, intensity: number): void {
    const p = PROFILES[kind];
    if (!p) return;
    const inten = Math.max(0, Math.min(4, intensity));
    const n = Math.max(1, Math.round(p.count * inten * this.spawnScale));
    const axis = this.tmpVec.copy(normal);
    if (axis.lengthSq() < 1e-6) axis.set(0, 1, 0);
    else axis.normalize();

    for (let i = 0; i < n; i++) {
      const d = this.next();
      this.applyProfile(d, p, inten);
      d.px = position.x;
      d.py = position.y;
      d.pz = position.z;
      randomCone(axis, p.spread, this.rng, this.tmpDir);
      const sp = p.speed * this.rng.range(0.45, 1.35) * (0.6 + 0.5 * inten);
      d.vx = this.tmpDir.x * sp;
      d.vy = this.tmpDir.y * sp;
      d.vz = this.tmpDir.z * sp;
      this.commit();
    }
  }

  emit(kind: ParticleKind, position: THREE.Vector3, velocity: THREE.Vector3, life: number, size: number): void {
    const p = PROFILES[kind];
    if (!p) return;
    const d = this.next();
    this.applyProfile(d, p, 1);
    d.px = position.x;
    d.py = position.y;
    d.pz = position.z;
    d.vx = velocity.x;
    d.vy = velocity.y;
    d.vz = velocity.z;
    d.life = life;
    d.size0 = size;
    d.size1 = size * (p.size1 / Math.max(p.size0, 1e-4));
    this.commit();
  }

  // ── low-level emission API (used by Impacts / Tracers / Volumetrics) ───────

  /**
   * Returns the shared, freshly reset {@link SpawnDesc}. Fill it and call
   * {@link commit}. Never retain the returned object across a commit.
   */
  next(): SpawnDesc {
    return this.desc.reset();
  }

  /** Writes the descriptor filled by {@link next} into its ring slot. */
  commit(): void {
    const d = this.desc;
    const batch = d.batch === PBatch.Additive ? this.additive : this.alpha;
    if (!batch || !this.atlas) return;
    this.pendingBatch = d.batch;
    const slot = batch.alloc();
    this.atlas.rect(d.cell, this.rect);
    batch.write(slot, d, this.time + d.delay, this.rect, this.rng.next());
  }

  /** Current simulation time used as the spawn stamp. */
  get now(): number {
    return this.time;
  }

  /** Total slots across both batches; useful for perf overlays. */
  get capacity(): number {
    return (this.alpha?.capacity ?? 0) + (this.additive?.capacity ?? 0);
  }

  /** Last committed batch, exposed for debugging/telemetry. */
  get lastBatch(): PBatchId {
    return this.pendingBatch;
  }

  // ── integration hooks for other systems ──────────────────────────────────────

  /**
   * Wires a real scene depth texture, enabling true depth-buffer soft
   * particles. The render system should call this once its depth attachment
   * exists. Passing `null` reverts to the plane-fade fallback.
   */
  setDepthTexture(depth: THREE.Texture | null, near?: number, far?: number): void {
    this.uniforms.uDepth.value = depth;
    this.depthRangeOverridden = near !== undefined || far !== undefined;
    if (near !== undefined) this.uniforms.uNear.value = near;
    if (far !== undefined) this.uniforms.uFar.value = far;
    for (const b of [this.alpha, this.additive]) {
      if (!b) continue;
      if (depth) b.material.defines.SOFT_DEPTH = '';
      else delete b.material.defines.SOFT_DEPTH;
      b.material.needsUpdate = true;
    }
  }

  /**
   * Overrides the lighting environment used by lit particles. Volumetrics
   * calls this so smoke, dust motes and fog agree on the sun.
   */
  setSun(direction: THREE.Vector3, color: THREE.Color, ambient: THREE.Color): void {
    this.sunOverridden = true;
    this.applySun(direction, color, ambient);
  }

  /**
   * Non-latching sun update used by the atmospherics system to keep smoke,
   * fog and dust motes in agreement.
   *
   * WHY SEPARATE FROM setSun: Volumetrics re-probes the scene light on a
   * timer and forwards the result here. If that path latched the override it
   * would permanently disable our own fallback probe, and it would silently
   * outrank a deliberate setSun() from a third system. Callers should check
   * {@link sunLocked} first.
   */
  applySun(direction: THREE.Vector3, color: THREE.Color, ambient: THREE.Color): void {
    (this.uniforms.uSunDir.value as THREE.Vector3).copy(direction).normalize();
    (this.uniforms.uSunColor.value as THREE.Color).copy(color);
    (this.uniforms.uAmbient.value as THREE.Color).copy(ambient);
  }

  /** True once a system has taken explicit ownership of the sun via setSun(). */
  get sunLocked(): boolean {
    return this.sunOverridden;
  }

  /** Unit vector pointing toward the sun, in world space. */
  get sunDirection(): THREE.Vector3 {
    return this.uniforms.uSunDir.value as THREE.Vector3;
  }

  /** Linear sun radiance used for particle lighting. */
  get sunColor(): THREE.Color {
    return this.uniforms.uSunColor.value as THREE.Color;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private applyProfile(d: SpawnDesc, p: KindProfile, inten: number): void {
    const s = 0.6 + 0.55 * inten;
    d.batch = p.batch;
    d.cell = p.cell;
    d.size0 = p.size0 * s;
    d.size1 = p.size1 * s;
    d.life = p.life * this.rng.range(0.75, 1.25);
    d.gravity = p.gravity;
    d.drag = p.drag;
    d.turbulence = p.turbulence;
    d.stretch = p.stretch;
    d.flags = p.flags;
    d.fadePow = p.fadePow;
    d.r = p.r;
    d.g = p.g;
    d.b = p.b;
    d.a = p.a;
    d.rot = this.rng.range(0, Math.PI * 2);
    d.spin = this.rng.sym(2.0);
  }

  private syncResolution(ctx: EngineContext): void {
    ctx.renderer.getDrawingBufferSize(this.drawBuffer);
    (this.uniforms.uResolution.value as THREE.Vector2).copy(this.drawBuffer);
    // Do not stomp a depth range the render system gave us with its depth texture.
    if (this.depthRangeOverridden) return;
    this.uniforms.uNear.value = ctx.camera.near;
    this.uniforms.uFar.value = ctx.camera.far;
  }

  /**
   * Best-effort discovery of the scene's key light. The render and world systems own
   * the sun; rather than couple to their type we walk the graph for the
   * brightest DirectionalLight. Cheap and runs on a long stride.
   */
  private findSun(ctx: EngineContext): void {
    let best: THREE.DirectionalLight | null = null;
    ctx.scene.traverse((o) => {
      const l = o as THREE.DirectionalLight;
      if (l.isDirectionalLight === true && (!best || l.intensity > best.intensity)) best = l;
    });
    // TS narrows `best` to `never` through the traverse closure; re-widen.
    const sun = best as THREE.DirectionalLight | null;
    if (!sun) return;
    const dir = this.uniforms.uSunDir.value as THREE.Vector3;
    // A DirectionalLight shines from `position` toward `target`, so the vector
    // pointing *at* the sun is position − target.
    sun.getWorldPosition(this.tmpVec);
    dir.copy(this.tmpVec);
    sun.target.getWorldPosition(this.tmpVec);
    dir.sub(this.tmpVec);
    if (dir.lengthSq() < 1e-8) dir.set(-0.42, 0.28, -0.86);
    dir.normalize();
    const col = this.uniforms.uSunColor.value as THREE.Color;
    col.copy(sun.color).multiplyScalar(Math.max(0.25, Math.min(4, sun.intensity)) * 0.9);
  }
}
