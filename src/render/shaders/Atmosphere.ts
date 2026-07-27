/**
 * NOISEFIELD — Atmospheric scattering model (shared GPU + CPU).
 *
 * This module is the single source of truth for the game's atmosphere. The very
 * same single-scattering model is compiled into the sky shader (GPU, per-pixel)
 * and evaluated on the CPU at a handful of directions to derive:
 *
 *   • the sun's transmitted colour           → `Lighting.sunColor`
 *   • the zenith / horizon sky colours       → hemisphere fill + fog tint
 *   • the ground bounce colour               → warm up-fill in shadowed areas
 *
 * Keeping one model means the key light, the ambient fill, the fog and the
 * visible sky can never drift out of agreement, which is the single most common
 * reason hobby renderers read as "fake" — a blue sky with a white sun and grey
 * fog is instantly wrong to the eye.
 *
 * MODEL
 * -----
 * Classic single-scatter ray march through a spherical, radially symmetric
 * atmosphere (Nishita-style, with Bruneton's coefficient set):
 *
 *   L(v) = Σ_i  T_view(i) · T_sun(i) · [ βR·ρR(i)·PR(µ) + βM·ρM(i)·PM(µ,g) ] · Δt
 *
 *   T(a→b) = exp( −∫ (βR_ext·ρR + βM_ext·ρM + βO·ρO) ds )
 *
 * ρR / ρM are exponential with scale heights 8 km / 1.2 km; ρO is a triangular
 * "tent" centred at 25 km, which is what turns the twilight zenith a deeper,
 * less muddy blue (ozone absorbs the green/orange the Rayleigh term leaves
 * behind on long horizon paths). Multiple scattering is not simulated; instead
 * a cheap ambient term approximates the second bounce, which for a clear
 * golden-hour sky is visually indistinguishable and ~100× cheaper.
 *
 * PERF: the march uses quadratically distributed samples (t ∝ x²) so that the
 * dense low atmosphere — where essentially all of the scattering happens — is
 * finely sampled even for a 600 km horizon ray. With uniform spacing you need
 * ~64 samples for a clean horizon; with quadratic spacing 12 is enough.
 */

import * as THREE from 'three';

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants (metres, m^-1). Shared by the GLSL and the TS evaluator.
// ─────────────────────────────────────────────────────────────────────────────

/** Planet radius. Earth-like; controls how fast the horizon curves away. */
export const R_GROUND = 6_360_000;
/** Top of the modelled atmosphere. */
export const R_TOP = 6_420_000;
/** Rayleigh (air molecule) density scale height. */
export const H_RAYLEIGH = 8_000;
/** Mie (aerosol / dust) density scale height. Low → dust hugs the ground. */
export const H_MIE = 1_200;

/** Rayleigh scattering coefficient at sea level, per metre, per RGB channel. */
export const BETA_R: readonly [number, number, number] = [5.802e-6, 13.558e-6, 33.1e-6];
/**
 * Mie scattering coefficient (grey — aerosols are wavelength-agnostic).
 *
 * Roughly 3x the clean-maritime reference value: this is a North-African
 * compound with dust in the air, and the heavier aerosol load is what produces
 * the milky band above the horizon and the strong aureole around a low sun.
 */
export const BETA_M = 1.2e-5;
/**
 * Mie extinction. Slightly larger than scattering because real aerosols also
 * absorb; the ratio (~0.9 albedo) is what stops haze looking like glowing milk.
 */
export const BETA_M_EXT = 1.34e-5;
/** Ozone absorption coefficient (no scattering — pure absorption). */
export const BETA_O: readonly [number, number, number] = [0.65e-6, 1.881e-6, 0.085e-6];

/** Mie asymmetry. 0.78 gives the tight forward aureole around a low sun. */
export const MIE_G = 0.78;

/**
 * Multiple-scattering boost.
 *
 * A pure single-scatter integral is famously ~3× too dark at the zenith and
 * gives no sky light at all to surfaces the sun cannot see, because in reality
 * most of the light arriving from a patch of clear sky has bounced more than
 * once. Rather than pay for a full multi-scattering LUT, the single-scatter
 * integral is re-injected with an isotropic phase (1/4π) and this gain. That
 * lifts the zenith and the shadow-side sky — where the directional phase terms
 * are smallest — far more than the forward-scatter aureole, which is exactly
 * the shape of the error being corrected.
 */
export const MULTI_SCATTER = 2.2;

/**
 * Scalar turning the unitless scatter integral into scene units — i.e. the
 * solar irradiance. Every other radiometric quantity in the renderer (the key
 * light, the hemisphere fill, the ground bounce, the fog) is derived from this
 * one number, so sun and sky can never drift apart photometrically.
 *
 * It is deliberately NOT the exposure control. Scaling it scales the sky by
 * exactly the same factor, so it cannot change the sun-to-sky *ratio* — the
 * only thing that actually sets how deep a shadow is. Overall image brightness
 * belongs to `RenderPipeline.exposure`, which sits after the whole light
 * transport and before the tone mapper, exactly where a camera's exposure sits.
 */
export const SUN_INTENSITY = 5.2;

/** Rec.709 luminance weights, for comparing differently coloured radiances. */
export const LUMA_WEIGHTS: readonly [number, number, number] = [0.2126, 0.7152, 0.0722];

/**
 * Ground albedo used by the analytic planet surface below the horizon. Matches
 * the art direction's sand (#C2A878) so the IBL's lower hemisphere pushes a
 * warm bounce into every downward-facing normal.
 */
export const GROUND_ALBEDO: readonly [number, number, number] = [0.42, 0.33, 0.22];

/**
 * Flat ambient term added to the analytic planet surface, standing in for the
 * sky light it receives. Without it the shadowed side of the horizon reads as
 * a hard black band in the IBL's lower hemisphere.
 */
export const GROUND_SKY_AMBIENT = 0.28;

// Golden-hour default. 17.5° puts the sun low enough for long raking shadows
// and a warm transmitted colour, but high enough that the compound is not in
// full silhouette. Azimuth is arbitrary but fixed so the level can be authored
// against a known shadow direction.
export const DEFAULT_SUN_ELEVATION_DEG = 17.5;
export const DEFAULT_SUN_AZIMUTH_DEG = 118;

// March quality. Exposed as GLSL defines so the pipeline can dial them down.
export const VIEW_STEPS_HIGH = 14;
export const VIEW_STEPS_LOW = 8;
export const LIGHT_STEPS_HIGH = 5;
export const LIGHT_STEPS_LOW = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Sun direction helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Builds the engine-convention sun vector: normalized and pointing **from the
 * sun toward the scene** (so its `y` is negative for a sun above the horizon).
 */
export function sunDirectionFromAngles(
  elevationDeg: number,
  azimuthDeg: number,
  target: THREE.Vector3,
): THREE.Vector3 {
  const el = THREE.MathUtils.degToRad(elevationDeg);
  const az = THREE.MathUtils.degToRad(azimuthDeg);
  // Vector pointing at the sun, then negated to face the scene.
  target.set(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
  return target.normalize().negate();
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The scattering model as GLSL. Injected verbatim into the sky fragment shader.
 * Callers must `#define ATM_VIEW_STEPS` and `ATM_LIGHT_STEPS` before including
 * it (GLSL ES 1.0 requires constant loop bounds and we keep that constraint so
 * the shader also compiles on a WebGL1 fallback context).
 */
export const ATMOSPHERE_GLSL = /* glsl */ `
#ifndef PI
#define PI 3.141592653589793
#endif

#ifndef ATM_VIEW_STEPS
#define ATM_VIEW_STEPS ${VIEW_STEPS_HIGH}
#endif
#ifndef ATM_LIGHT_STEPS
#define ATM_LIGHT_STEPS ${LIGHT_STEPS_HIGH}
#endif

#define ATM_R_GROUND ${R_GROUND.toFixed(1)}
#define ATM_R_TOP    ${R_TOP.toFixed(1)}
#define ATM_H_RAY    ${H_RAYLEIGH.toFixed(1)}
#define ATM_H_MIE    ${H_MIE.toFixed(1)}

const vec3  ATM_BETA_R     = vec3(${BETA_R[0]}, ${BETA_R[1]}, ${BETA_R[2]});
const float ATM_BETA_M     = float(${BETA_M});
const float ATM_BETA_M_EXT = float(${BETA_M_EXT});
const vec3  ATM_BETA_O     = vec3(${BETA_O[0]}, ${BETA_O[1]}, ${BETA_O[2]});

/** Rayleigh phase: gentle, symmetric — this is what makes the sky blue. */
float atmRayleighPhase(float mu) {
  return (3.0 / (16.0 * PI)) * (1.0 + mu * mu);
}

/**
 * Henyey-Greenstein Mie phase. Strongly forward-peaked (g ~ 0.78), which is
 * exactly the bright aureole hugging a low sun and the whitening of the sky
 * near the horizon.
 */
float atmMiePhase(float mu, float g) {
  float g2 = g * g;
  float d = max(1e-4, 1.0 + g2 - 2.0 * g * mu);
  return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * d * sqrt(d));
}

/** (rayleigh, mie, ozone) relative densities at altitude h. */
vec3 atmDensity(float h) {
  float a = max(h, 0.0);
  return vec3(
    exp(-a / ATM_H_RAY),
    exp(-a / ATM_H_MIE),
    max(0.0, 1.0 - abs(a - 25000.0) / 15000.0)
  );
}

/** Far intersection with a sphere centred at the origin (we are always inside). */
float atmRaySphereFar(vec3 o, vec3 d, float R) {
  float b = dot(o, d);
  float c = dot(o, o) - R * R;
  float disc = b * b - c;
  if (disc < 0.0) return -1.0;
  return -b + sqrt(disc);
}

/** Nearest positive intersection, or -1. Used for the ground test. */
float atmRaySphereNear(vec3 o, vec3 d, float R) {
  float b = dot(o, d);
  float c = dot(o, o) - R * R;
  float disc = b * b - c;
  if (disc < 0.0) return -1.0;
  float s = sqrt(disc);
  float t0 = -b - s;
  return t0 > 0.0 ? t0 : -1.0;
}

/** Transmittance from p toward the top of the atmosphere along dir. */
vec3 atmTransmittance(vec3 p, vec3 dir) {
  float tTop = atmRaySphereFar(p, dir, ATM_R_TOP);
  if (tTop <= 0.0) return vec3(1.0);
  // Shadowed by the planet: the sun is below the local horizon.
  if (atmRaySphereNear(p, dir, ATM_R_GROUND) > 0.0) return vec3(0.0);

  vec3 od = vec3(0.0);
  float prev = 0.0;
  for (int i = 0; i < ATM_LIGHT_STEPS; i++) {
    float x = float(i + 1) / float(ATM_LIGHT_STEPS);
    float t1 = tTop * x * x;                 // quadratic clustering near p
    float seg = t1 - prev;
    vec3 q = p + dir * (prev + seg * 0.5);
    od += atmDensity(length(q) - ATM_R_GROUND) * seg;
    prev = t1;
  }
  return exp(-(ATM_BETA_R * od.x + ATM_BETA_M_EXT * od.y + ATM_BETA_O * od.z));
}

/**
 * Single-scatter radiance along rayDir.
 *
 * @param sunDir       unit vector pointing TOWARD the sun
 * @param eyeAltitude  metres above the planet surface
 * @param outViewT     transmittance eye→end-of-ray (used for the sun disc)
 * @param outGroundT   distance to the planet surface, or -1 if the ray escapes
 * @param outGroundW   0..1 weight the caller must apply to its ground shading
 */
vec3 atmScatter(vec3 rayDir, vec3 sunDir, float eyeAltitude, float mieG,
                out vec3 outViewT, out float outGroundT, out float outGroundW) {
  vec3 o = vec3(0.0, ATM_R_GROUND + eyeAltitude, 0.0);
  float tTop = atmRaySphereFar(o, rayDir, ATM_R_TOP);
  float tGround = atmRaySphereNear(o, rayDir, ATM_R_GROUND);
  outGroundT = tGround;

  // The integration limit is violently discontinuous at the horizon: a ray one
  // pixel above it marches ~600 km of atmosphere, a ray one pixel below stops
  // at the surface. Left alone that rules a hard line across the frame.
  // Blending the two limits over a ~0.7 degree band — and fading the caller's
  // ground shading in over that same band — turns the horizon into a gradient.
  // Inside the band the long path is geometrically underground, which is
  // meaningless physically but is precisely the interpolation we want, and the
  // band is well under a degree wide.
  float tSky = max(tTop, 1.0);
  float groundW = tGround > 0.0 ? smoothstep(0.0, 0.012, -rayDir.y) : 0.0;
  outGroundW = groundW;

  float tMax = mix(tSky, max(tGround, 1.0), groundW);

  float mu = dot(rayDir, sunDir);
  float pr = atmRayleighPhase(mu);
  float pm = atmMiePhase(mu, mieG);

  vec3 od = vec3(0.0);
  vec3 sumR = vec3(0.0);
  vec3 sumM = vec3(0.0);
  float prev = 0.0;

  for (int i = 0; i < ATM_VIEW_STEPS; i++) {
    float x = float(i + 1) / float(ATM_VIEW_STEPS);
    float t1 = tMax * x * x;
    float seg = t1 - prev;
    vec3 p = o + rayDir * (prev + seg * 0.5);
    prev = t1;

    vec3 dens = atmDensity(length(p) - ATM_R_GROUND) * seg;
    od += dens;

    // Transmittance eye→sample, then sample→sun. Product = light that both
    // reaches the sample and survives the trip back to the eye.
    vec3 tView = exp(-(ATM_BETA_R * od.x + ATM_BETA_M_EXT * od.y + ATM_BETA_O * od.z));
    vec3 tSun = atmTransmittance(p, sunDir);
    vec3 t = tView * tSun;

    sumR += t * dens.x;
    sumM += t * dens.y;
  }

  outViewT = exp(-(ATM_BETA_R * od.x + ATM_BETA_M_EXT * od.y + ATM_BETA_O * od.z));

  // Isotropic re-injection standing in for multiple scattering; see
  // MULTI_SCATTER in Atmosphere.ts for why this is not optional.
  vec3 ms = (sumR * ATM_BETA_R + sumM * ATM_BETA_M) * (${MULTI_SCATTER} / (4.0 * PI));

  return sumR * ATM_BETA_R * pr + sumM * ATM_BETA_M * pm + ms;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CPU evaluator — a direct transcription of the GLSL above.
// ─────────────────────────────────────────────────────────────────────────────

type Vec3 = [number, number, number];

const _p: Vec3 = [0, 0, 0];
const _od: Vec3 = [0, 0, 0];
const _t: Vec3 = [0, 0, 0];
const _tSun: Vec3 = [0, 0, 0];
const _sumR: Vec3 = [0, 0, 0];
const _sumM: Vec3 = [0, 0, 0];
const _viewT: Vec3 = [0, 0, 0];
const _dens: Vec3 = [0, 0, 0];
/**
 * Separate scratch for the inner light march. `transmittance` is called from
 * inside `evaluateSky`'s view loop, so sharing `_dens` between them silently
 * corrupts the outer loop's density with the last high-altitude light sample.
 */
const _densInner: Vec3 = [0, 0, 0];

function density(h: number, out: Vec3): Vec3 {
  const a = Math.max(h, 0);
  out[0] = Math.exp(-a / H_RAYLEIGH);
  out[1] = Math.exp(-a / H_MIE);
  out[2] = Math.max(0, 1 - Math.abs(a - 25000) / 15000);
  return out;
}

function raySphereFar(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, R: number): number {
  const b = ox * dx + oy * dy + oz * dz;
  const c = ox * ox + oy * oy + oz * oz - R * R;
  const disc = b * b - c;
  if (disc < 0) return -1;
  return -b + Math.sqrt(disc);
}

function raySphereNear(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, R: number): number {
  const b = ox * dx + oy * dy + oz * dz;
  const c = ox * ox + oy * oy + oz * oz - R * R;
  const disc = b * b - c;
  if (disc < 0) return -1;
  const s = Math.sqrt(disc);
  const t0 = -b - s;
  return t0 > 0 ? t0 : -1;
}

function transmittance(px: number, py: number, pz: number, dx: number, dy: number, dz: number, steps: number, out: Vec3): Vec3 {
  const tTop = raySphereFar(px, py, pz, dx, dy, dz, R_TOP);
  if (tTop <= 0) {
    out[0] = out[1] = out[2] = 1;
    return out;
  }
  if (raySphereNear(px, py, pz, dx, dy, dz, R_GROUND) > 0) {
    out[0] = out[1] = out[2] = 0;
    return out;
  }
  let odR = 0;
  let odM = 0;
  let odO = 0;
  let prev = 0;
  for (let i = 0; i < steps; i++) {
    const x = (i + 1) / steps;
    const t1 = tTop * x * x;
    const seg = t1 - prev;
    const tm = prev + seg * 0.5;
    prev = t1;
    const qx = px + dx * tm;
    const qy = py + dy * tm;
    const qz = pz + dz * tm;
    const h = Math.hypot(qx, qy, qz) - R_GROUND;
    density(h, _densInner);
    odR += _densInner[0] * seg;
    odM += _densInner[1] * seg;
    odO += _densInner[2] * seg;
  }
  out[0] = Math.exp(-(BETA_R[0] * odR + BETA_M_EXT * odM + BETA_O[0] * odO));
  out[1] = Math.exp(-(BETA_R[1] * odR + BETA_M_EXT * odM + BETA_O[1] * odO));
  out[2] = Math.exp(-(BETA_R[2] * odR + BETA_M_EXT * odM + BETA_O[2] * odO));
  return out;
}

/**
 * Transmittance of the atmosphere along the path to the sun, seen from ground
 * level. Multiplying the solar irradiance by this gives the physically correct
 * warm, dimmed key-light colour at golden hour — no hand-picked hex needed.
 */
export function sunTransmittance(sunDirectionToScene: THREE.Vector3, target: THREE.Color): THREE.Color {
  // Sun vector points from sun → scene; the ray we integrate goes the other way.
  const dx = -sunDirectionToScene.x;
  const dy = -sunDirectionToScene.y;
  const dz = -sunDirectionToScene.z;
  transmittance(0, R_GROUND + 2, 0, dx, dy, dz, 12, _tSun);
  return target.setRGB(_tSun[0], _tSun[1], _tSun[2], THREE.LinearSRGBColorSpace);
}

/**
 * Evaluates the sky radiance for one direction, matching `atmScatter` in GLSL.
 * Deliberately uses generous sample counts — this is called a handful of times
 * per sun update, never per frame per pixel.
 */
export function evaluateSky(
  rayDir: THREE.Vector3,
  sunDirectionToScene: THREE.Vector3,
  target: THREE.Color,
  eyeAltitude = 2,
  viewSteps = 24,
  lightSteps = 8,
): THREE.Color {
  const ox = 0;
  const oy = R_GROUND + eyeAltitude;
  const oz = 0;
  const dx = rayDir.x;
  const dy = rayDir.y;
  const dz = rayDir.z;
  const sx = -sunDirectionToScene.x;
  const sy = -sunDirectionToScene.y;
  const sz = -sunDirectionToScene.z;

  const tTop = raySphereFar(ox, oy, oz, dx, dy, dz, R_TOP);
  const tGround = raySphereNear(ox, oy, oz, dx, dy, dz, R_GROUND);
  // Same horizon band as the GLSL; see `atmScatter`.
  const tSky = Math.max(tTop, 1);
  const groundW = tGround > 0 ? THREE.MathUtils.smoothstep(-dy, 0, 0.012) : 0;
  const tMax = THREE.MathUtils.lerp(tSky, Math.max(tGround, 1), groundW);

  const mu = dx * sx + dy * sy + dz * sz;
  const pr = (3 / (16 * Math.PI)) * (1 + mu * mu);
  const g2 = MIE_G * MIE_G;
  const dd = Math.max(1e-4, 1 + g2 - 2 * MIE_G * mu);
  const pm = ((3 / (8 * Math.PI)) * ((1 - g2) * (1 + mu * mu))) / ((2 + g2) * dd * Math.sqrt(dd));

  _od[0] = _od[1] = _od[2] = 0;
  _sumR[0] = _sumR[1] = _sumR[2] = 0;
  _sumM[0] = _sumM[1] = _sumM[2] = 0;
  let prev = 0;

  for (let i = 0; i < viewSteps; i++) {
    const x = (i + 1) / viewSteps;
    const t1 = tMax * x * x;
    const seg = t1 - prev;
    const tm = prev + seg * 0.5;
    prev = t1;

    _p[0] = ox + dx * tm;
    _p[1] = oy + dy * tm;
    _p[2] = oz + dz * tm;
    const h = Math.hypot(_p[0], _p[1], _p[2]) - R_GROUND;
    density(h, _dens);
    _od[0] += _dens[0] * seg;
    _od[1] += _dens[1] * seg;
    _od[2] += _dens[2] * seg;

    transmittance(_p[0], _p[1], _p[2], sx, sy, sz, lightSteps, _tSun);

    for (let c = 0; c < 3; c++) {
      const tv = Math.exp(-(BETA_R[c] * _od[0] + BETA_M_EXT * _od[1] + BETA_O[c] * _od[2]));
      _t[c] = tv * _tSun[c];
      _sumR[c] += _t[c] * _dens[0] * seg;
      _sumM[c] += _t[c] * _dens[1] * seg;
    }
  }

  for (let c = 0; c < 3; c++) {
    _viewT[c] = Math.exp(-(BETA_R[c] * _od[0] + BETA_M_EXT * _od[1] + BETA_O[c] * _od[2]));
  }

  // Isotropic multiple-scattering re-injection, matching the GLSL exactly.
  const ms = MULTI_SCATTER / (4 * Math.PI);
  let r = (_sumR[0] * BETA_R[0] * (pr + ms) + _sumM[0] * BETA_M * (pm + ms)) * SUN_INTENSITY;
  let g = (_sumR[1] * BETA_R[1] * (pr + ms) + _sumM[1] * BETA_M * (pm + ms)) * SUN_INTENSITY;
  let b = (_sumR[2] * BETA_R[2] * (pr + ms) + _sumM[2] * BETA_M * (pm + ms)) * SUN_INTENSITY;

  if (groundW > 0) {
    // Analytic planet surface: lambert sand lit by the transmitted sun.
    const gx = (ox + dx * tGround) / R_GROUND;
    const gy = (oy + dy * tGround) / R_GROUND;
    const gz = (oz + dz * tGround) / R_GROUND;
    const ndl = Math.max(gx * sx + gy * sy + gz * sz, 0);
    transmittance(ox + dx * tGround, oy + dy * tGround, oz + dz * tGround, sx, sy, sz, lightSteps, _tSun);
    // Lambert: L = albedo · E · cosθ / π, plus a flat sky-ambient term.
    const k = (ndl * SUN_INTENSITY) / Math.PI;
    r += GROUND_ALBEDO[0] * (_tSun[0] * k + GROUND_SKY_AMBIENT) * _viewT[0] * groundW;
    g += GROUND_ALBEDO[1] * (_tSun[1] * k + GROUND_SKY_AMBIENT) * _viewT[1] * groundW;
    b += GROUND_ALBEDO[2] * (_tSun[2] * k + GROUND_SKY_AMBIENT) * _viewT[2] * groundW;
  }

  return target.setRGB(r, g, b, THREE.LinearSRGBColorSpace);
}

/** Rec.709 relative luminance of a linear-space colour. */
export function luminance(c: THREE.Color): number {
  return c.r * LUMA_WEIGHTS[0] + c.g * LUMA_WEIGHTS[1] + c.b * LUMA_WEIGHTS[2];
}

/** Scratch for `skyIrradiance`; the integrator must not allocate per sample. */
const _irrDir = new THREE.Vector3();
const _irrSample = new THREE.Color();

/** Van der Corput radical inverse, base 2 — the second Hammersley coordinate. */
function radicalInverse2(i: number): number {
  let bits = i >>> 0;
  bits = ((bits << 16) | (bits >>> 16)) >>> 0;
  bits = (((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >>> 1)) >>> 0;
  bits = (((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >>> 2)) >>> 0;
  bits = (((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >>> 4)) >>> 0;
  bits = (((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >>> 8)) >>> 0;
  return bits * 2.3283064365386963e-10;
}

/**
 * Diffuse sky irradiance landing on an upward-facing horizontal surface:
 *
 *     E = ∫_Ω+ L(ω) · cosθ dω
 *
 * in exactly the units `SUN_INTENSITY` is expressed in, so it can be compared
 * directly against the direct solar irradiance.
 *
 * WHY THIS EXISTS: this is the number that decides how dark a shadow is. A
 * fragment the sun cannot see is lit by this and nothing else, so
 * `E_sky / E_sun` *is* the shadow depth, and it has to be measured rather than
 * guessed — the same scattering model that draws the dome also feeds the PMREM
 * environment, and its absolute magnitude changes every time the sun moves.
 * Evaluating it here lets `Lighting` solve for the environment intensity that
 * hits a chosen sun-to-sky ratio instead of hard-coding a number that is only
 * correct for one sun position.
 *
 * Cosine-weighted Hammersley quadrature: samples are distributed ∝ cosθ, so the
 * estimator is the plain mean times π and, usefully, the numerically fragile
 * grazing-horizon directions (where the 600 km slant path is badly
 * under-resolved by any affordable march) are exactly the ones the cosine
 * weight sends to zero.
 */
export function skyIrradiance(
  sunDirectionToScene: THREE.Vector3,
  target: THREE.Color,
  samples = 64,
): THREE.Color {
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < samples; i++) {
    // Stratified in the first coordinate, van der Corput in the second.
    const u1 = (i + 0.5) / samples;
    const u2 = radicalInverse2(i);
    const cosT = Math.sqrt(1 - u1);
    const sinT = Math.sqrt(u1);
    const phi = 2 * Math.PI * u2;
    _irrDir.set(sinT * Math.cos(phi), cosT, sinT * Math.sin(phi));
    // Cheap march: the result is a single integrated scalar per channel, so
    // per-direction error averages out and buying more steps buys nothing.
    evaluateSky(_irrDir, sunDirectionToScene, _irrSample, 2, 10, 4);
    r += _irrSample.r;
    g += _irrSample.g;
    b += _irrSample.b;
  }
  const k = Math.PI / samples;
  return target.setRGB(r * k, g * k, b * k, THREE.LinearSRGBColorSpace);
}
