/**
 * NOISEFIELD — atmospherics.
 *
 * After lighting, atmosphere is the single largest lever on perceived
 * production value. A sun-baked compound at golden hour should feel *thick*:
 * shafts of light, suspended dust catching the sun, haze stacking up with
 * distance. This system delivers five layers, all pooled and all GPU-animated:
 *
 *   1. GOD RAYS      — a camera-locked additive radial-shaft overlay anchored
 *                      to the sun's screen position, masked by an occlusion
 *                      estimate so it does not bleed when the sun is hidden.
 *   2. DUST MOTES    — thousands of tiny backlit particles in a box that
 *                      follows the camera, wrapped in the vertex shader so
 *                      they respawn for free and are always present.
 *   3. GROUND FOG    — soft, warm-lit haze cards drifting near the floor.
 *   4. HORIZON HAZE  — a distance band that ties the geometry into the sky.
 *   5. HEAT SHIMMER  — a subtle animated warm band at the horizon line.
 *
 *   plus an occasional DUST DEVIL that sweeps across the map, and a public
 *   {@link Volumetrics.addLightShaft} for the world system to place explicit
 *   shafts under windows and alley openings.
 *
 * WHY THE GOD RAYS ARE NOT A POST PASS
 * ------------------------------------
 * The textbook implementation is a screen-space radial blur of an occlusion
 * buffer, which requires inserting a pass into the post-processing chain. This
 * system does not own the render pipeline and must not edit it. The overlay here
 * reproduces the *look* — radial noise-modulated shafts emanating from the
 * sun's projected position, attenuated by distance from it — and derives its
 * occlusion mask from a handful of physics raycasts toward the sun rather than
 * from a depth buffer. Cost is ~3 raycasts and one full-screen additive quad
 * per frame; at golden hour with the sun low behind architecture it reads very
 * close to the real thing. The honest limitation: occlusion is sampled at the
 * camera, not per pixel, so shafts fade as a whole rather than being carved by
 * individual silhouettes.
 *
 * DEGRADATION
 * -----------
 * `settings.volumetrics === false` disables god rays, fog cards, shimmer and
 * dust devils, leaving only a reduced mote field and the horizon band, which
 * are nearly free. Every layer additionally scales with the quality tier.
 *
 */

import * as THREE from 'three';
import type {
  EngineContext,
  FrameContext,
  IPhysics,
  QualityTier,
  System,
} from '../core/Types';
import {
  AtlasCell,
  getSharedAtlas,
  getSharedNoise,
  releaseSharedAtlas,
  releaseSharedNoise,
  type SpriteAtlas,
} from './textures';
import { PBatch, PFlags, Particles, Rng } from './Particles';

// ─────────────────────────────────────────────────────────────────────────────
// Tuning
// ─────────────────────────────────────────────────────────────────────────────

const MOTE_COUNT: Record<QualityTier, number> = { low: 0, medium: 1100, high: 2400, ultra: 3600 };
const FOG_CARDS: Record<QualityTier, number> = { low: 0, medium: 10, high: 20, ultra: 28 };

/** Edge length of the mote box that follows the camera, in metres. */
const MOTE_BOX = 22;
/** Radius of the horizon haze cylinder. Must sit inside the camera's far plane. */
const HORIZON_RADIUS = 380;

/** Golden-hour defaults used until a DirectionalLight is discovered. */
const DEFAULT_SUN = new THREE.Vector3(-0.46, 0.24, -0.85).normalize();
const DEFAULT_SUN_COLOR = new THREE.Color(0xffd2a1);
const DEFAULT_AMBIENT = new THREE.Color(0x1a2028);
const HAZE_COLOR = new THREE.Color(0xc9a97e);
const SAND = new THREE.Color(0xc2a878);

/** Fixed jitter pattern for the sun occlusion probe (unit-disc offsets). */
const OCC_OFFSETS: readonly number[] = [
  0.0, 0.0, 0.05, 0.0, -0.05, 0.0, 0.0, 0.05, 0.0, -0.05, 0.035, 0.035, -0.035, 0.035, 0.035, -0.035,
];
const OCC_SAMPLES_PER_FRAME = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Shaders
// ─────────────────────────────────────────────────────────────────────────────

/** Shared by every camera-locked overlay: a unit quad with plain screen UVs. */
const OVERLAY_VERT = /* glsl */ `
precision highp float;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const GODRAY_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform sampler2D uNoise;
uniform vec2  uSunScreen;   // sun position in 0..1 screen space
uniform float uIntensity;   // CPU-side mask: on-screen × facing × unoccluded
uniform vec3  uColor;
uniform float uTime;
uniform float uAspect;

varying vec2 vUv;

void main() {
  if (uIntensity < 0.002) discard;

  vec2 d = vUv - uSunScreen;
  d.x *= uAspect;                    // keep shafts radially symmetric
  float r   = length(d);
  float ang = atan(d.y, d.x);

  // Sample the noise around a CIRCLE in texture space rather than by raw
  // angle. atan() jumps by 2*PI across the -X axis, and feeding that
  // discontinuity straight into a texture coordinate cuts a hard radial seam
  // through the shafts. A closed circular path has no endpoints, and the
  // noise texture is built to tile, so the field is continuous all the way
  // round. Two radii give a coarse shaft count plus finer structure.
  vec2  cir = vec2(cos(ang), sin(ang));
  float s1 = texture2D(uNoise, cir * 0.46 + vec2(0.5) + vec2(uTime * 0.008, 0.0)).r;
  float s2 = texture2D(uNoise, cir * 1.13 + vec2(0.5) + vec2(0.0, uTime * 0.013)).g;
  float streak = pow(clamp(s1 * 0.72 + s2 * 0.58, 0.0, 1.0), 1.7);

  float core  = exp(-r * 6.0);           // tight glow around the disc
  float shaft = exp(-r * 2.6) * streak;  // shafts reach further out

  // Gains are deliberately conservative. This overlay is ADDITIVE across the
  // WHOLE screen, so an over-eager value clips a third of the frame to white
  // and buries the level behind a starburst filter. Measured target: under 3%
  // of pixels fully clipped even when staring straight into the sun.
  float v = (core * 0.38 + shaft * 0.62) * uIntensity;
  // Suppress the singularity exactly at the sun centre.
  v *= smoothstep(0.0, 0.04, r);
  if (v < 0.002) discard;

  gl_FragColor = vec4(uColor * v, 1.0);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const SHIMMER_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform sampler2D uNoise;
uniform float uTime;
uniform float uHorizon;    // horizon line in 0..1 screen space
uniform float uStrength;
uniform vec3  uColor;

varying vec2 vUv;

void main() {
  if (uStrength < 0.001) discard;

  // Concentrate the effect in a band hugging the horizon, which is where a
  // long grazing path through hot air actually accumulates.
  float k    = (vUv.y - uHorizon) / 0.085;
  float band = exp(-k * k);
  if (band < 0.01) discard;

  // Two counter-scrolling noise layers read as rising convection cells.
  float n1 = texture2D(uNoise, vec2(vUv.x * 2.4, vUv.y * 9.0 - uTime * 0.30)).b;
  float n2 = texture2D(uNoise, vec2(vUv.x * 5.1 + 0.37, vUv.y * 15.0 - uTime * 0.53)).a;
  float w  = (n1 * 0.6 + n2 * 0.4) - 0.5;

  // NOTE: a true heat haze refracts the framebuffer. That needs a post pass we
  // do not own, so this is a brightness/warmth modulation instead — at the
  // amplitudes used here the eye reads it as shimmer.
  float v = band * uStrength * max(0.0, 0.45 + w * 1.6);
  if (v < 0.002) discard;

  gl_FragColor = vec4(uColor * v, 1.0);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const MOTE_VERT = /* glsl */ `
precision highp float;

attribute vec4 mSeed;   // xyz = normalised position in the box, w = seed

uniform vec3  uCamPos;
uniform vec3  uBox;
uniform vec3  uWind;
uniform float uTime;
uniform float uSize;
uniform float uMinAngular;
uniform vec3  uSunDir;
uniform float uBackBright;
uniform float uBaseBright;

varying vec2  vUv;
varying float vAlpha;
varying float vWarm;

void main() {
  float ph = mSeed.w * 6.28318;

  // Slow wind plus a per-mote Lissajous wobble so they never look like a rigid
  // translating lattice.
  vec3 p = mSeed.xyz * uBox
         + uWind * uTime
         + vec3(sin(uTime * 0.50 + ph)        * 0.30,
                sin(uTime * 0.31 + ph * 1.7)  * 0.22,
                cos(uTime * 0.42 + ph * 2.3)  * 0.30);

  // Wrap into a box centred on the camera. GLSL mod() is always non-negative,
  // so this is a branchless infinite field: motes that leave one face reappear
  // on the opposite one. Zero CPU cost, no respawn bookkeeping, and the field
  // is always populated wherever the player walks.
  p = uCamPos + mod(p - uCamPos + uBox * 0.5, uBox) - uBox * 0.5;

  vec3  toCam = cameraPosition - p;
  float dist  = length(toCam);

  // Screen-space size floor: a 2 cm mote at 15 m is sub-pixel and flickers.
  float s = max(uSize, dist * uMinAngular);

  vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  vec3 camUp    = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
  vec3 world    = p + camRight * (position.x * s) + camUp * (position.y * s);

  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);

  // Backlighting. Suspended dust is barely visible against a lit wall and
  // blazes when you look into the sun; that contrast is the whole effect.
  vec3  viewDir = -toCam / max(dist, 1e-4);
  float fwd     = pow(clamp(dot(viewDir, uSunDir), 0.0, 1.0), 6.0);

  float near = smoothstep(0.35, 1.4, dist);
  float far  = 1.0 - smoothstep(uBox.x * 0.32, uBox.x * 0.5, dist);
  float rnd  = 0.35 + 0.65 * fract(mSeed.w * 91.7);

  vAlpha = (uBaseBright + fwd * uBackBright) * near * far * rnd;
  vWarm  = fwd;
  vUv    = uv;
}
`;

const MOTE_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform sampler2D uAtlas;
uniform vec4  uRect;      // atlas rect of the soft circle cell
uniform vec3  uSunTint;
uniform vec3  uAmbient;

varying vec2  vUv;
varying float vAlpha;
varying float vWarm;

void main() {
  float a = texture2D(uAtlas, uRect.xy + vUv * uRect.zw).a * vAlpha;
  if (a < 0.004) discard;
  // Cool in shadow, warm when backlit — the same golden-hour split the rest of
  // the scene uses.
  vec3 col = mix(uAmbient * 1.6, uSunTint * 1.15, clamp(vWarm * 1.4, 0.0, 1.0));
  gl_FragColor = vec4(col, a);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const FOG_VERT = /* glsl */ `
precision highp float;

attribute vec4 fSeed;   // xy = normalised XZ in the box, z = height jitter, w = seed

uniform vec3  uCamPos;
uniform vec2  uBox;
uniform vec2  uWind;
uniform float uTime;
uniform float uFloorY;
uniform float uCardW;
uniform float uCardH;

varying vec2  vUv;
varying vec3  vWorld;
varying float vDist;
varying float vSeed;

void main() {
  vec2 xz = fSeed.xy * uBox + uWind * uTime;
  // Same free wrap as the motes, in the horizontal plane only.
  xz = uCamPos.xz + mod(xz - uCamPos.xz + uBox * 0.5, uBox) - uBox * 0.5;

  float ph = fSeed.w * 6.28318;
  float y  = uFloorY + 0.7 + fSeed.z * 2.6 + sin(uTime * 0.13 + ph) * 0.25;
  vec3  p  = vec3(xz.x, y, xz.y);

  vec3  toCam = cameraPosition - p;
  float dist  = length(toCam);

  vec3 camRight = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
  // Cards stay world-upright and only yaw toward the camera: a fog bank that
  // rolls with the pitch of your head reads as a sprite immediately.
  vec3 up       = vec3(0.0, 1.0, 0.0);
  vec3 right    = normalize(vec3(camRight.x, 0.0, camRight.z) + vec3(1e-5, 0.0, 0.0));

  float w = uCardW * (0.7 + 0.6 * fract(fSeed.w * 37.1));
  float h = uCardH * (0.7 + 0.6 * fract(fSeed.w * 13.7));
  vec3 world = p + right * (position.x * w) + up * (position.y * h);

  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
  vUv    = uv;
  vWorld = world;
  vDist  = dist;
  vSeed  = fSeed.w;
}
`;

const FOG_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform sampler2D uAtlas;
uniform sampler2D uNoise;
uniform vec4  uRect;
uniform vec3  uSunDir;
uniform vec3  uSunTint;
uniform vec3  uAmbient;
uniform float uFloorY;
uniform float uDensity;
uniform float uTime;
uniform float uFar;

varying vec2  vUv;
varying vec3  vWorld;
varying float vDist;
varying float vSeed;

void main() {
  float mask = texture2D(uAtlas, uRect.xy + vUv * uRect.zw).a;
  // A slow noise multiply keeps the card from reading as one static sprite.
  float n = texture2D(uNoise, vUv * 0.7 + vec2(uTime * 0.008 + vSeed, vSeed * 2.3)).b;
  float a = mask * (0.45 + 0.9 * n) * uDensity;

  // Hide the part of the card that hangs below the floor. Assumes a locally
  // flat ground plane, which holds for a walled compound.
  a *= smoothstep(uFloorY - 1.6, uFloorY - 0.05, vWorld.y);
  /*
   * Height falloff: haze pools low.
   *
   * Was smoothstep(1.4, 4.5), which is not "low" — it is full strength up to
   * chest height and still contributing at 4.5 m, i.e. straight across the
   * 1.7 m eye line and over everything the player is looking at. Because each
   * card applies this as a world-Y gradient on a billboard, every one of the
   * eight to twelve cards stacked along a view ray laid its own soft horizontal
   * band across the frame; together they washed solid geometry out badly enough
   * that walls behind crates showed through, and the compound read as though it
   * were rendered on glass.
   *
   * 0.25 → 1.8 keeps the fog genuinely pooled at and below knee height, where
   * ground fog belongs, so it adds depth to the floor plane without ever
   * crossing the sightline.
   */
  a *= 1.0 - smoothstep(0.25, 1.8, vWorld.y - uFloorY);
  // Do not let a card engulf the camera, and fade the far edge of the box.
  a *= smoothstep(3.0, 11.0, vDist) * (1.0 - smoothstep(uFar * 0.6, uFar, vDist));
  if (a < 0.004) discard;

  // Same treatment as the particle system: forward scattering toward a low sun
  // gives the haze a hot rim on the sun side and leaves it cool elsewhere.
  vec3  viewDir = normalize(vWorld - cameraPosition);
  float fwd     = pow(clamp(dot(viewDir, uSunDir), 0.0, 1.0), 4.0);
  vec3  col     = mix(uAmbient * 1.3, uSunTint * 0.95, clamp(0.28 + 0.35 * n + fwd * 0.8, 0.0, 1.0));

  gl_FragColor = vec4(col, a);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const HAZE_VERT = /* glsl */ `
precision highp float;
varying float vY;
varying vec3  vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  vY = position.y;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`;

const HAZE_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform vec3  uHaze;
uniform vec3  uSunTint;
uniform vec3  uSunDir;
uniform float uDensity;
uniform float uBottom;
uniform float uTop;

varying float vY;
varying vec3  vWorld;

void main() {
  // Dense at the base, clear well before the zenith.
  float t = clamp((vY - uBottom) / max(uTop - uBottom, 1e-3), 0.0, 1.0);
  float a = uDensity * pow(1.0 - t, 2.6);
  if (a < 0.004) discard;

  vec3  viewDir = normalize(vWorld - cameraPosition);
  float fwd     = pow(clamp(dot(viewDir, uSunDir), 0.0, 1.0), 3.0);
  vec3  col     = mix(uHaze, uSunTint * 1.05, fwd * 0.65);

  gl_FragColor = vec4(col, a);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

const SHAFT_VERT = /* glsl */ `
precision highp float;
varying vec3  vNormal;
varying vec3  vWorld;
varying float vT;
void main() {
  vec4 w  = modelMatrix * vec4(position, 1.0);
  vWorld  = w.xyz;
  vNormal = normalize(mat3(modelMatrix) * normal);
  // Unit cylinder spans y in [-0.5, 0.5]; 0 = entry aperture, 1 = far end.
  vT = 0.5 - position.y;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`;

const SHAFT_FRAG = /* glsl */ `
precision highp float;

// NOTE: three injects <tonemapping_pars_fragment> and <colorspace_pars_fragment>
// into every ShaderMaterial's fragment prefix. Including them here as well is a
// GLSL redefinition error. Only the statement-level chunks belong in main().
#include <common>

uniform vec3  uSunTint;
uniform vec3  uSunDir;
uniform float uDensity;

varying vec3  vNormal;
varying vec3  vWorld;
varying float vT;

void main() {
  vec3  viewDir = normalize(vWorld - cameraPosition);
  // Facing ratio: the silhouette edges of a cylinder have the longest path
  // through the volume, so brightening them makes a hollow tube read as a
  // solid beam of lit air.
  float edge = 1.0 - abs(dot(normalize(vNormal), viewDir));
  edge = pow(clamp(edge, 0.0, 1.0), 1.4);
  // Beams dim along their length as the air scatters the light away.
  float along = pow(clamp(1.0 - vT, 0.0, 1.0), 1.6);
  // Forward scattering: brightest looking INTO the sun. viewDir points
  // camera -> fragment and uSunDir points toward the sun, so they agree when
  // you are looking through the shaft at the sun. Do not negate either.
  float fwd = 0.35 + 0.9 * pow(clamp(dot(viewDir, uSunDir), 0.0, 1.0), 3.0);

  float a = edge * along * fwd * uDensity;
  if (a < 0.003) discard;
  gl_FragColor = vec4(uSunTint * a, 1.0);

  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// System
// ─────────────────────────────────────────────────────────────────────────────

export class Volumetrics implements System {
  readonly name = 'volumetrics';

  private ctx: EngineContext | null = null;
  private particles: Particles | null = null;
  private physics: IPhysics | null = null;
  private physicsUsable = true;
  private atlas: SpriteAtlas | null = null;
  private readonly rng = new Rng(0x77c1b5);

  private time = 0;
  private enabled = true;
  /** Latched once another system calls setSun(); stops the auto-probe fighting it. */
  private sunOverridden = false;

  // Shared uniforms — the SAME IUniform objects are referenced by every
  // material, so updating the sun once updates all of them.
  private readonly uTime: THREE.IUniform<number> = { value: 0 };
  private readonly uSunDir: THREE.IUniform<THREE.Vector3> = { value: DEFAULT_SUN.clone() };
  /**
   * True sun radiance, including the light's intensity. Correct for *lighting*
   * — it is what Particles shades smoke with — and wrong for everything here.
   */
  private readonly uSunColor: THREE.IUniform<THREE.Color> = { value: DEFAULT_SUN_COLOR.clone() };
  /**
   * Hue-normalised sun colour (largest channel == 1).
   *
   * WHY THIS EXISTS: a physically-lit scene runs its DirectionalLight around
   * intensity 3, so `uSunColor` lands near (2.7, 2.2, 1.5). Feeding that into
   * an ADDITIVE full-screen overlay adds ~3.0 of linear radiance per layer and
   * blows the entire frame to white. Atmospheric layers want the sun's *hue*
   * at an art-directed magnitude, so every one of them reads this and applies
   * its own gain.
   */
  private readonly uSunTint: THREE.IUniform<THREE.Color> = { value: DEFAULT_SUN_COLOR.clone() };
  private readonly uAmbient: THREE.IUniform<THREE.Color> = { value: DEFAULT_AMBIENT.clone() };

  // Layers.
  private godray: THREE.Mesh | null = null;
  private godrayMat: THREE.ShaderMaterial | null = null;
  private shimmer: THREE.Mesh | null = null;
  private shimmerMat: THREE.ShaderMaterial | null = null;
  private motes: THREE.Mesh | null = null;
  private moteMat: THREE.ShaderMaterial | null = null;
  private fog: THREE.Mesh | null = null;
  private fogMat: THREE.ShaderMaterial | null = null;
  private haze: THREE.Mesh | null = null;
  private hazeMat: THREE.ShaderMaterial | null = null;

  // Optional explicit light shafts placed by the world system.
  private shaftGroup: THREE.Group | null = null;
  private shaftGeo: THREE.CylinderGeometry | null = null;
  private shaftMat: THREE.ShaderMaterial | null = null;
  private readonly shafts: THREE.Mesh[] = [];
  private static readonly MAX_SHAFTS = 14;

  // Sun occlusion state.
  private occCursor = 0;
  private occSmooth = 1;
  private sunIntensity = 0;

  // Dust devil state machine.
  private gustNext = 26;
  private gustEnd = -1;
  private readonly gustPos = new THREE.Vector3();
  private readonly gustVel = new THREE.Vector3();

  private floorY = 0;

  // Scratch.
  private readonly tmpA = new THREE.Vector3();
  private readonly tmpB = new THREE.Vector3();
  private readonly tmpC = new THREE.Vector3();
  private readonly tmpD = new THREE.Vector3();
  private readonly tmpQ = new THREE.Quaternion();
  private readonly rect = new THREE.Vector4();
  private readonly down = new THREE.Vector3(0, -1, 0);
  private readonly up = new THREE.Vector3(0, 1, 0);

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    this.enabled = ctx.settings.volumetrics;
    const tier = ctx.settings.tier;
    this.atlas = getSharedAtlas(tier === 'low' || tier === 'medium' ? 128 : 256);
    const noise = getSharedNoise();

    this.floorY = Math.min(0, ctx.camera.position.y - 1.7);

    this.buildMotes(ctx, noise);
    this.buildHaze(ctx);
    if (this.enabled) {
      this.buildGodRays(ctx, noise);
      this.buildShimmer(ctx, noise);
      this.buildFog(ctx, noise);
    }
    this.findSun(ctx);
  }

  update(ctx: EngineContext, frame: FrameContext): void {
    this.time = frame.elapsed;
    this.uTime.value = this.time;

    if (frame.frame % 41 === 0) {
      if (!this.sunOverridden) this.findSun(ctx);
      this.sampleSkyColour(ctx);
    }

    this.updateMotes(ctx);
    this.updateHaze(ctx);
    if (this.enabled) {
      this.updateGodRays(ctx);
      this.updateShimmer(ctx);
      this.updateFog(ctx);
      this.updateGust(ctx, frame.dt);
    }
  }

  dispose(): void {
    const kill = (m: THREE.Mesh | null, mat: THREE.Material | null): void => {
      if (m) {
        m.removeFromParent();
        m.geometry.dispose();
      }
      mat?.dispose();
    };
    kill(this.godray, this.godrayMat);
    kill(this.shimmer, this.shimmerMat);
    kill(this.motes, this.moteMat);
    kill(this.fog, this.fogMat);
    kill(this.haze, this.hazeMat);
    this.clearLightShafts();
    this.shaftGroup?.removeFromParent();
    this.shaftGeo?.dispose();
    this.shaftMat?.dispose();
    this.godray = null;
    this.shimmer = null;
    this.motes = null;
    this.fog = null;
    this.haze = null;
    if (this.atlas) {
      releaseSharedAtlas();
      releaseSharedNoise();
      this.atlas = null;
    }
  }

  // ── public API for other systems ─────────────────────────────────────────────

  /** Overrides the sun used by every atmospheric layer (and by Particles). */
  setSun(direction: THREE.Vector3, color: THREE.Color, ambient?: THREE.Color): void {
    this.sunOverridden = true;
    this.uSunDir.value.copy(direction).normalize();
    this.uSunColor.value.copy(color);
    if (ambient) this.uAmbient.value.copy(ambient);
    this.refreshSunTint();
    this.pushSunToParticles();
  }

  /** Y coordinate the fog cards and dust devils treat as ground level. */
  setFloor(y: number): void {
    this.floorY = y;
  }

  /**
   * Places an explicit world-space light shaft: a cone of lit air falling from
   * `origin` along the sun direction. Intended for the world system to drop
   * under windows, roof holes and alley gaps, where a screen-space
   * approximation cannot know the aperture exists.
   */
  addLightShaft(origin: THREE.Vector3, length: number, radius: number, density = 0.22): void {
    // NOTE: all shafts share one material, so `density` is a global that the
    // most recent call wins. Per-shaft density would need an instanced
    // attribute, which is not worth a draw call for at most 14 shafts.
    const ctx = this.ctx;
    if (!ctx || !this.enabled) return;
    if (this.shafts.length >= Volumetrics.MAX_SHAFTS) return;

    if (!this.shaftGeo) {
      // Slightly conical: shafts widen as they descend, which matches how a
      // finite aperture actually projects.
      this.shaftGeo = new THREE.CylinderGeometry(0.78, 1.0, 1, 14, 1, true);
    }
    if (!this.shaftMat) {
      this.shaftMat = new THREE.ShaderMaterial({
        uniforms: {
          uSunTint: this.uSunTint,
          uSunDir: this.uSunDir,
          uDensity: { value: density },
        },
        vertexShader: SHAFT_VERT,
        fragmentShader: SHAFT_FRAG,
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        toneMapped: true,
      });
      this.shaftMat.name = 'noisefield.fx.lightShaft';
    }
    this.shaftMat.uniforms.uDensity.value = density;
    if (!this.shaftGroup) {
      this.shaftGroup = new THREE.Group();
      this.shaftGroup.name = 'noisefield.fx.lightShafts';
      ctx.scene.add(this.shaftGroup);
    }

    // Clamp before scaling: a zero radius collapses the cylinder's side
    // normals, and GLSL normalize(vec3(0.0)) is undefined (NaN on most
    // drivers), which would paint the whole shaft with garbage.
    const r = Math.max(0.02, radius);
    const len = Math.max(0.05, length);
    const mesh = new THREE.Mesh(this.shaftGeo, this.shaftMat);
    mesh.name = 'noisefield.fx.lightShaft';
    mesh.frustumCulled = true;
    mesh.renderOrder = 9;
    // Cylinder's local +Y becomes the direction the light travels FROM, so
    // align +Y with the sun direction and hang the body below the aperture.
    this.tmpQ.setFromUnitVectors(this.up, this.uSunDir.value);
    mesh.quaternion.copy(this.tmpQ);
    mesh.scale.set(r, len, r);
    mesh.position.copy(origin).addScaledVector(this.uSunDir.value, -len * 0.5);
    this.shaftGroup.add(mesh);
    this.shafts.push(mesh);
  }

  /** Removes every shaft added through {@link addLightShaft}. */
  clearLightShafts(): void {
    for (const s of this.shafts) s.removeFromParent();
    this.shafts.length = 0;
  }

  /** Current CPU-side god-ray mask, 0..1. Useful for HUD/debug overlays. */
  get sunVisibility(): number {
    return this.sunIntensity;
  }

  // ── systems ──────────────────────────────────────────────────────────────────

  private get P(): Particles | null {
    if (!this.particles) this.particles = this.ctx?.get<Particles>('particles') ?? null;
    return this.particles;
  }

  private get physicsSystem(): IPhysics | null {
    if (!this.physicsUsable) return null;
    if (!this.physics) this.physics = this.ctx?.get<IPhysics>('physics') ?? null;
    return this.physics;
  }

  private pushSunToParticles(): void {
    const p = this.P;
    // applySun, not setSun: forwarding our periodic auto-probe must not latch
    // Particles' override flag (which would disable its own fallback probe),
    // and must never outrank a deliberate setSun() from another system.
    if (p && !p.sunLocked) p.applySun(this.uSunDir.value, this.uSunColor.value, this.uAmbient.value);
  }

  /** Recomputes {@link uSunTint} from {@link uSunColor}; see their docs. */
  private refreshSunTint(): void {
    const c = this.uSunColor.value;
    const m = Math.max(c.r, c.g, c.b, 1e-4);
    this.uSunTint.value.setRGB(c.r / m, c.g / m, c.b / m);
  }

  /** Best-effort discovery of the scene's key light; see Particles.findSun. */
  private sunProbe: THREE.DirectionalLight | null = null;
  /** Hoisted so the periodic probe does not allocate a closure every 41 frames. */
  private readonly visitLight = (o: THREE.Object3D): void => {
    const l = o as THREE.DirectionalLight;
    if (l.isDirectionalLight !== true) return;
    if (!this.sunProbe || l.intensity > this.sunProbe.intensity) this.sunProbe = l;
  };

  private findSun(ctx: EngineContext): void {
    this.sunProbe = null;
    ctx.scene.traverse(this.visitLight);
    // traverse() mutates sunProbe through a callback, which TypeScript's
    // control-flow analysis cannot see; it still believes the field is the
    // `null` we just assigned. Re-widen explicitly.
    const sun = this.sunProbe as THREE.DirectionalLight | null;
    if (!sun) {
      this.pushSunToParticles();
      return;
    }
    sun.getWorldPosition(this.tmpA);
    sun.target.getWorldPosition(this.tmpB);
    this.tmpA.sub(this.tmpB);
    if (this.tmpA.lengthSq() < 1e-8) this.tmpA.copy(DEFAULT_SUN);
    this.uSunDir.value.copy(this.tmpA).normalize();
    this.uSunColor.value.copy(sun.color).multiplyScalar(Math.max(0.25, Math.min(4, sun.intensity)) * 0.9);
    this.refreshSunTint();
    this.pushSunToParticles();
  }

  /** Matches the haze band to whatever the render system is using for the sky. */
  private sampleSkyColour(ctx: EngineContext): void {
    if (!this.hazeMat) return;
    const bg = ctx.scene.background;
    const target = this.hazeMat.uniforms.uHaze.value as THREE.Color;
    if (bg instanceof THREE.Color) {
      // Warm the sky slightly: haze is dustier and more saturated than air.
      target.copy(bg).lerp(HAZE_COLOR, 0.45);
    } else if (ctx.scene.fog) {
      target.copy(ctx.scene.fog.color).lerp(HAZE_COLOR, 0.3);
    } else {
      target.copy(HAZE_COLOR);
    }
  }

  // ── overlays ───────────────────────────────────────────────────────────────

  private makeOverlay(name: string, material: THREE.ShaderMaterial, order: number): THREE.Mesh {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
    mesh.name = name;
    mesh.frustumCulled = false;
    mesh.renderOrder = order;
    return mesh;
  }

  /** Parks a camera-locked overlay exactly covering the near frustum. */
  private placeOverlay(mesh: THREE.Mesh, ctx: EngineContext): void {
    const cam = ctx.camera;
    const dist = cam.near * 6;
    const h = 2 * dist * Math.tan((cam.fov * Math.PI) / 360);
    const w = h * cam.aspect;
    cam.getWorldQuaternion(this.tmpQ);
    cam.getWorldPosition(this.tmpA);
    this.tmpB.set(0, 0, -1).applyQuaternion(this.tmpQ);
    mesh.position.copy(this.tmpA).addScaledVector(this.tmpB, dist);
    mesh.quaternion.copy(this.tmpQ);
    // 2% oversize hides any rounding gap at the frustum edge.
    mesh.scale.set(w * 1.02, h * 1.02, 1);
  }

  private buildGodRays(ctx: EngineContext, noise: THREE.Texture): void {
    this.godrayMat = new THREE.ShaderMaterial({
      uniforms: {
        uNoise: { value: noise },
        uSunScreen: { value: new THREE.Vector2(0.5, 0.5) },
        uIntensity: { value: 0 },
        uColor: { value: new THREE.Color(0xffd2a1).multiplyScalar(1.15) },
        uTime: this.uTime,
        uAspect: { value: 1 },
      },
      vertexShader: OVERLAY_VERT,
      fragmentShader: GODRAY_FRAG,
      transparent: true,
      // Atmospheric scattering happens in front of everything, so the overlay
      // ignores depth entirely and draws last.
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: true,
    });
    this.godrayMat.name = 'noisefield.fx.godrays';
    this.godray = this.makeOverlay('noisefield.fx.godrays', this.godrayMat, 900);
    ctx.scene.add(this.godray);
  }

  private updateGodRays(ctx: EngineContext): void {
    const mesh = this.godray;
    const mat = this.godrayMat;
    if (!mesh || !mat) return;
    this.placeOverlay(mesh, ctx);

    const cam = ctx.camera;
    const sun = this.uSunDir.value;
    cam.getWorldDirection(this.tmpB);
    const facing = this.tmpB.dot(sun);

    let intensity = 0;
    if (facing > 0.02) {
      // Project a point a long way along the sun direction; with the sun in
      // front of us this is a stable proxy for the disc's screen position.
      this.tmpC.copy(cam.position).addScaledVector(sun, 600).project(cam);
      const sx = this.tmpC.x * 0.5 + 0.5;
      const sy = this.tmpC.y * 0.5 + 0.5;
      (mat.uniforms.uSunScreen.value as THREE.Vector2).set(sx, sy);

      // Allow the shafts to persist a little past the screen edge — real ones
      // do — but roll off before the projection becomes meaningless.
      const off = Math.max(Math.abs(sx - 0.5), Math.abs(sy - 0.5)) * 2;
      const onScreen = 1 - smootherstep(0.9, 1.9, off);
      const facingMask = smootherstep(0.02, 0.45, facing);

      const occ = this.sampleSunOcclusion(ctx, sun);
      // Heavy temporal smoothing: walking behind a pillar should dim the
      // shafts over ~0.3 s, not strobe them off for a single frame.
      this.occSmooth += (occ - this.occSmooth) * 0.11;
      // No shafts once the sun has actually set.
      const above = smootherstep(-0.04, 0.13, sun.y);
      intensity = onScreen * facingMask * this.occSmooth * above * 0.42;
    } else {
      this.occSmooth += (0 - this.occSmooth) * 0.11;
    }

    this.sunIntensity = intensity;
    mat.uniforms.uIntensity.value = intensity;
    mat.uniforms.uAspect.value = cam.aspect;
    (mat.uniforms.uColor.value as THREE.Color).copy(this.uSunTint.value).multiplyScalar(0.85);
  }

  /**
   * Estimates how much of the sky toward the sun is unoccluded, by casting a
   * few jittered rays. Costs 3 raycasts a frame and is the only thing keeping
   * the god rays from bleeding through solid walls.
   */
  private sampleSunOcclusion(ctx: EngineContext, sun: THREE.Vector3): number {
    const phys = this.physicsSystem;
    // With no physics system we cannot know: assume clear sky rather than
    // suppressing the effect entirely.
    if (!phys) return 1;

    // Tangent basis around the sun direction for the jitter pattern.
    if (Math.abs(sun.y) < 0.95) this.tmpA.copy(this.up).cross(sun).normalize();
    else this.tmpA.set(1, 0, 0);
    this.tmpB.copy(sun).cross(this.tmpA).normalize();

    let clear = 0;
    for (let i = 0; i < OCC_SAMPLES_PER_FRAME; i++) {
      const k = ((this.occCursor + i) % (OCC_OFFSETS.length / 2)) * 2;
      this.tmpC
        .copy(sun)
        .addScaledVector(this.tmpA, OCC_OFFSETS[k])
        .addScaledVector(this.tmpB, OCC_OFFSETS[k + 1])
        .normalize();
      // Start slightly out along the ray so we never hit the player's own
      // collider and report a permanently occluded sky.
      this.tmpD.copy(ctx.camera.position).addScaledVector(this.tmpC, 0.6);
      try {
        const hit = phys.raycast(this.tmpD, this.tmpC, 80);
        if (!hit) clear++;
      } catch {
        this.physicsUsable = false;
        this.physics = null;
        return 1;
      }
    }
    this.occCursor = (this.occCursor + OCC_SAMPLES_PER_FRAME) % (OCC_OFFSETS.length / 2);
    return clear / OCC_SAMPLES_PER_FRAME;
  }

  private buildShimmer(ctx: EngineContext, noise: THREE.Texture): void {
    this.shimmerMat = new THREE.ShaderMaterial({
      uniforms: {
        uNoise: { value: noise },
        uTime: this.uTime,
        uHorizon: { value: 0.5 },
        // Overwritten every frame by `updateShimmer`; this is only the value
        // used for the first frame before that runs.
        uStrength: { value: 0.006 },
        uColor: { value: new THREE.Color(0xffc890) },
      },
      vertexShader: OVERLAY_VERT,
      fragmentShader: SHIMMER_FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: true,
    });
    this.shimmerMat.name = 'noisefield.fx.heatShimmer';
    this.shimmer = this.makeOverlay('noisefield.fx.heatShimmer', this.shimmerMat, 898);
    ctx.scene.add(this.shimmer);
  }

  private updateShimmer(ctx: EngineContext): void {
    const mesh = this.shimmer;
    const mat = this.shimmerMat;
    if (!mesh || !mat) return;
    this.placeOverlay(mesh, ctx);

    // Project a distant point at eye height to find the horizon line, then
    // anchor the shimmer band to it so it tracks the player's pitch.
    const cam = ctx.camera;
    cam.getWorldDirection(this.tmpB);
    this.tmpB.y = 0;
    if (this.tmpB.lengthSq() < 1e-6) this.tmpB.set(0, 0, -1);
    else this.tmpB.normalize();
    this.tmpC.copy(cam.position).addScaledVector(this.tmpB, 220).project(cam);
    mat.uniforms.uHorizon.value = this.tmpC.y * 0.5 + 0.5;

    /*
     * Shimmer only really exists when the sun is up and shining on the ground.
     *
     * Strength was 0.02 + 0.075·sun, i.e. ~0.095 at midday. This overlay is a
     * FULL-SCREEN ADDITIVE quad with `depthTest: false`, anchored to the horizon
     * band — so that 0.095 of warm light was being added on top of every distant
     * building, every wall and every crate, with nothing to attenuate it. It
     * lifted the blacks out of the entire horizon band and was the reason solid
     * geometry read as though it were painted on glass: you could see a wall's
     * window line straight through a crate in front of it.
     *
     * The deeper problem is that heat haze is REFRACTION — it displaces what is
     * behind it — and this element cannot refract, because it is a scene overlay
     * with no access to the colour buffer. Additive light is the one thing a
     * shimmer must not be. Doing it properly means moving it into the post chain
     * as a UV distortion, which is a render-system change; until then the honest
     * thing is to keep it as a faint horizon warmth that cannot wash the frame.
     *
     * Peak 0.013, about a seventh of what it was.
     */
    const sunUp = Math.max(0, this.uSunDir.value.y);
    mat.uniforms.uStrength.value = 0.003 + 0.010 * Math.min(1, sunUp * 3.2);
    (mat.uniforms.uColor.value as THREE.Color).copy(this.uSunTint.value).multiplyScalar(0.9);
  }

  // ── dust motes ─────────────────────────────────────────────────────────────

  private buildMotes(ctx: EngineContext, _noise: THREE.Texture): void {
    const tier = ctx.settings.tier;
    let count = MOTE_COUNT[tier] ?? 1500;
    if (!this.enabled) count = Math.round(count * 0.4);
    // Motes are billed against a slice of the particle budget so a huge mote
    // field can never crowd out combat VFX on a constrained device.
    count = Math.min(count, Math.max(0, Math.round(ctx.settings.particleBudget * 0.22)));
    if (count <= 0 || !this.atlas) return;

    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.instanceCount = count;

    // Static attribute: the field never changes after init. All motion is
    // derived in the vertex shader from uTime, so there is no per-frame upload.
    const seeds = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      seeds[i * 4] = this.rng.next();
      seeds[i * 4 + 1] = this.rng.next();
      seeds[i * 4 + 2] = this.rng.next();
      seeds[i * 4 + 3] = this.rng.next();
    }
    geo.setAttribute('mSeed', new THREE.InstancedBufferAttribute(seeds, 4));

    this.atlas.rect(AtlasCell.Circle, this.rect);

    this.moteMat = new THREE.ShaderMaterial({
      uniforms: {
        uAtlas: { value: this.atlas.texture },
        uRect: { value: this.rect.clone() },
        uCamPos: { value: new THREE.Vector3() },
        uBox: { value: new THREE.Vector3(MOTE_BOX, MOTE_BOX * 0.7, MOTE_BOX) },
        uWind: { value: new THREE.Vector3(0.28, 0.05, 0.16) },
        uTime: this.uTime,
        uSize: { value: 0.017 },
        uMinAngular: { value: 0.0016 },
        uSunDir: this.uSunDir,
        uSunTint: this.uSunTint,
        uAmbient: this.uAmbient,
        uBackBright: { value: 0.85 },
        uBaseBright: { value: 0.055 },
      },
      vertexShader: MOTE_VERT,
      fragmentShader: MOTE_FRAG,
      transparent: true,
      // Depth tested so motes are correctly hidden behind walls, but never
      // depth writing — they are volumetric and unsorted.
      depthTest: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: true,
    });
    this.moteMat.name = 'noisefield.fx.dustMotes';

    this.motes = new THREE.Mesh(geo, this.moteMat);
    this.motes.name = 'noisefield.fx.dustMotes';
    this.motes.frustumCulled = false;
    this.motes.matrixAutoUpdate = false;
    this.motes.renderOrder = 11;
    ctx.scene.add(this.motes);
  }

  private updateMotes(ctx: EngineContext): void {
    if (!this.moteMat) return;
    (this.moteMat.uniforms.uCamPos.value as THREE.Vector3).copy(ctx.camera.position);
  }

  // ── ground fog ─────────────────────────────────────────────────────────────

  private buildFog(ctx: EngineContext, noise: THREE.Texture): void {
    const count = FOG_CARDS[ctx.settings.tier] ?? 12;
    if (count <= 0 || !this.atlas) return;

    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3),
    );
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.instanceCount = count;

    const seeds = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      seeds[i * 4] = this.rng.next();
      seeds[i * 4 + 1] = this.rng.next();
      seeds[i * 4 + 2] = this.rng.next();
      seeds[i * 4 + 3] = this.rng.next();
    }
    geo.setAttribute('fSeed', new THREE.InstancedBufferAttribute(seeds, 4));

    this.atlas.rect(AtlasCell.Smoke, this.rect);

    this.fogMat = new THREE.ShaderMaterial({
      uniforms: {
        uAtlas: { value: this.atlas.texture },
        uNoise: { value: noise },
        uRect: { value: this.rect.clone() },
        uCamPos: { value: new THREE.Vector3() },
        // Box tuned so most cards fall inside the visible fade band: a bigger
        // box with the same count just wastes instances on invisible cards.
        uBox: { value: new THREE.Vector2(62, 62) },
        uWind: { value: new THREE.Vector2(0.35, 0.2) },
        uTime: this.uTime,
        uFloorY: { value: this.floorY },
        uCardW: { value: 17 },
        // 4.0 m cards reached far above the band the fog is now allowed to
        // occupy, so most of each card was clipped to zero alpha anyway; 2.6 m
        // fits the visible band and wastes less overdraw.
        uCardH: { value: 2.6 },
        uSunDir: this.uSunDir,
        uSunTint: this.uSunTint,
        uAmbient: this.uAmbient,
        // Per-card alpha must account for OVERLAP: ~8-12 cards can stack
        // along a single view ray, so 0.12 each compounds to a near-opaque
        // tan wall. Even 0.035 compounds to roughly 30% coverage, which is a
        // third of the scene's contrast gone; measured against a volumetrics-off
        // reference that was the single largest quality regression in the
        // renderer. 0.014 leaves the floor plane hazy and the sightline clear.
        uDensity: { value: 0.014 },
        uFar: { value: 38 },
      },
      vertexShader: FOG_VERT,
      fragmentShader: FOG_FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      side: THREE.DoubleSide,
      toneMapped: true,
    });
    this.fogMat.name = 'noisefield.fx.groundFog';

    this.fog = new THREE.Mesh(geo, this.fogMat);
    this.fog.name = 'noisefield.fx.groundFog';
    this.fog.frustumCulled = false;
    this.fog.matrixAutoUpdate = false;
    // Behind the particle batches so combat smoke reads in front of the haze.
    this.fog.renderOrder = 8;
    ctx.scene.add(this.fog);
  }

  private updateFog(ctx: EngineContext): void {
    if (!this.fogMat) return;
    (this.fogMat.uniforms.uCamPos.value as THREE.Vector3).copy(ctx.camera.position);
    this.fogMat.uniforms.uFloorY.value = this.floorY;
  }

  // ── horizon haze ───────────────────────────────────────────────────────────

  private buildHaze(ctx: EngineContext): void {
    // Open cylinder well beyond the playable area: it only ever appears
    // against the sky, which is exactly what a horizon band should do. True
    // distance fog over geometry belongs to the render system's scene.fog.
    const geo = new THREE.CylinderGeometry(HORIZON_RADIUS, HORIZON_RADIUS, 150, 40, 1, true);
    this.hazeMat = new THREE.ShaderMaterial({
      uniforms: {
        uHaze: { value: HAZE_COLOR.clone() },
        uSunTint: this.uSunTint,
        uSunDir: this.uSunDir,
        uDensity: { value: 0.32 },
        // Anchored just below eye level: haze must be densest AT the horizon
        // line and clear out well before the zenith, or it either does nothing
        // or paints the whole sky.
        uBottom: { value: -2 },
        uTop: { value: 52 },
      },
      vertexShader: HAZE_VERT,
      fragmentShader: HAZE_FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      side: THREE.BackSide,
      toneMapped: true,
    });
    this.hazeMat.name = 'noisefield.fx.horizonHaze';

    this.haze = new THREE.Mesh(geo, this.hazeMat);
    this.haze.name = 'noisefield.fx.horizonHaze';
    this.haze.frustumCulled = false;
    this.haze.renderOrder = 2;
    ctx.scene.add(this.haze);
    this.sampleSkyColour(ctx);
  }

  private updateHaze(ctx: EngineContext): void {
    if (!this.haze) return;
    // Follow the camera horizontally so the band is always at the horizon;
    // keep it anchored vertically to the world so it does not slide with jumps.
    this.haze.position.set(ctx.camera.position.x, 0, ctx.camera.position.z);
  }

  // ── dust devil / wind gust ─────────────────────────────────────────────────

  /**
   * Occasional wind event: a loose column of sand that drifts across the map.
   * Emitted straight into the shared particle system so it costs no extra draw
   * call and is automatically bounded by the same budget as everything else.
   */
  private updateGust(ctx: EngineContext, dt: number): void {
    if (dt <= 0) return;
    const P = this.P;
    if (!P) return;
    const rng = this.rng;

    if (this.gustEnd < this.time) {
      if (this.time >= this.gustNext) {
        // Spawn upwind of the camera and sweep past it.
        const ang = rng.range(0, Math.PI * 2);
        const dist = rng.range(16, 34);
        this.gustPos.set(
          ctx.camera.position.x + Math.cos(ang) * dist,
          this.floorY,
          ctx.camera.position.z + Math.sin(ang) * dist,
        );
        const travel = rng.range(0, Math.PI * 2);
        const speed = rng.range(3.5, 7.0);
        this.gustVel.set(Math.cos(travel) * speed, 0, Math.sin(travel) * speed);
        this.gustEnd = this.time + rng.range(7, 12);
        this.gustNext = this.gustEnd + rng.range(34, 70);
      }
      return;
    }

    this.gustPos.addScaledVector(this.gustVel, dt);

    const n = Math.max(1, Math.round(3 * P.spawnScale));
    for (let i = 0; i < n; i++) {
      // Swirl: tangential velocity around the column axis, rising with height.
      const h = rng.pow(0, 7, 1.4); // biased low: the column is densest at its base
      const radius = 0.6 + h * 0.55;
      const a = rng.range(0, Math.PI * 2);
      const tang = 3.4 + h * 0.7;

      const d = P.next();
      d.batch = PBatch.Alpha;
      d.cell = rng.next() < 0.5 ? AtlasCell.Dust : AtlasCell.Smoke;
      d.px = this.gustPos.x + Math.cos(a) * radius;
      d.py = this.floorY + h;
      d.pz = this.gustPos.z + Math.sin(a) * radius;
      d.setVel(
        -Math.sin(a) * tang + this.gustVel.x,
        rng.range(1.4, 3.6),
        Math.cos(a) * tang + this.gustVel.z,
      );
      d.life = rng.range(2.0, 4.0);
      d.size0 = rng.range(0.4, 1.0);
      d.size1 = rng.range(1.6, 3.4);
      d.setColor(SAND.r, SAND.g, SAND.b, rng.range(0.07, 0.17));
      d.gravity = 0.02;
      d.drag = 0.9;
      d.turbulence = 0.9;
      d.flags = PFlags.Lit;
      d.fadePow = 2.4;
      d.rot = rng.range(0, Math.PI * 2);
      d.spin = rng.sym(0.9);
      P.commit();
    }
  }
}

/** Ken Perlin's smootherstep — a C² ramp, used for masks that must not pop. */
function smootherstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * t * (t * (t * 6 - 15) + 10);
}
