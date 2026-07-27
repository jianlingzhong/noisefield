/**
 * NOISEFIELD — Sky shader.
 *
 * Two vertex programs share one fragment program:
 *
 *   SCREEN mode — a single full-screen triangle drawn last among the opaques
 *     with `depthTest: true, depthWrite: false` and `gl_Position.z = w`. Early-Z
 *     therefore rejects every pixel already covered by geometry, so the (fairly
 *     expensive) scattering march is only paid for on actually-visible sky. The
 *     view ray is rebuilt from the inverse projection so it is exact for any
 *     FOV, including while the player is aiming down sights.
 *
 *   CUBE mode — an inverted unit box rendered by PMREMGenerator into the IBL.
 *     A box works because linear interpolation of the corner positions across a
 *     planar face *is* the exact ray direction from the origin.
 *
 * The sun disc is authored as radiance, not as a sprite: it is the atmospheric
 * transmittance along the view ray times a large constant, which means the disc
 * automatically reddens and dims as it approaches the horizon without any
 * artist intervention. Bloom then does the rest.
 *
 * NOTE: never put a backtick inside these template literals.
 */

import { ATMOSPHERE_GLSL, GROUND_SKY_AMBIENT } from './Atmosphere';

/** Full-screen-triangle vertex program (mode: screen). */
export const SKY_VERTEX_SCREEN = /* glsl */ `
precision highp float;

uniform mat4 uInvProjection;
uniform mat4 uCameraWorld;

varying vec3 vRayDir;

void main() {
  // Unproject the clip-space corner to view space, then rotate (not translate)
  // into world space: a direction has no origin.
  vec4 viewPos = uInvProjection * vec4(position.xy, 1.0, 1.0);
  vRayDir = mat3(uCameraWorld) * (viewPos.xyz / viewPos.w);

  // z = w puts the fragment exactly on the far plane so LEQUAL depth testing
  // keeps it behind everything while still passing against a cleared buffer.
  gl_Position = vec4(position.xy, 1.0, 1.0);
}
`;

/** Inverted-box vertex program (mode: cube / IBL capture). */
export const SKY_VERTEX_CUBE = /* glsl */ `
precision highp float;

varying vec3 vRayDir;

void main() {
  vRayDir = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/**
 * Shared fragment program. Expects ATMOSPHERE_GLSL to be prepended and the
 * following defines to be supplied by the material factory:
 *   ATM_VIEW_STEPS, ATM_LIGHT_STEPS
 */
export const SKY_FRAGMENT = /* glsl */ `
precision highp float;

${ATMOSPHERE_GLSL}

uniform vec3  uSunDirection;      // unit, pointing TOWARD the sun
uniform float uSunIntensity;
uniform float uSunDiscIntensity;
uniform float uMieG;
uniform float uEyeAltitude;
uniform float uTime;
uniform float uCloudAmount;       // 0 disables the cloud layer entirely
uniform float uCloudCoverage;     // 0..1 fraction of sky covered
uniform vec3  uGroundAlbedo;

varying vec3 vRayDir;

// Angular radius of the solar disc, radians (~0.53 degrees across).
#define SUN_ANGULAR_RADIUS 0.004675

// Altitude of the cirrus/altocumulus deck, metres.
#define CLOUD_HEIGHT 2600.0

// ── cheap value noise (no textures — the project ships zero assets) ─────────

float skyHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float skyNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);      // smoothstep interpolation
  float a = skyHash(i);
  float b = skyHash(i + vec2(1.0, 0.0));
  float c = skyHash(i + vec2(0.0, 1.0));
  float d = skyHash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float skyFbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  // Rotating each octave breaks up the axis-aligned grid of value noise, which
  // is otherwise very visible against a smooth sky gradient.
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 5; i++) {
    v += a * skyNoise(p);
    p = rot * p * 2.03;
    a *= 0.5;
  }
  return v;
}

/**
 * Flat-plane cloud deck. Returns rgb = lit cloud radiance, a = coverage.
 *
 * A true volumetric march is far too costly for a background that is on screen
 * every frame; a single plane with fbm coverage plus a horizontally offset
 * second lookup for self-shadowing buys ~90% of the read for ~2% of the cost.
 *
 * @param sunIrradiance  transmitted solar irradiance in the same units the
 *                       scattering integral is expressed in. Radiance is
 *                       albedo·E·cosθ/PI, exactly like any other Lambert
 *                       surface — using irradiance directly here is what makes
 *                       a cloud deck blow the whole frame out to white.
 */
vec4 skyClouds(vec3 rayDir, vec3 sunDir, vec3 skyColor, vec3 sunIrradiance) {
  if (uCloudAmount <= 0.0 || rayDir.y <= 0.012) return vec4(0.0);

  float t = (CLOUD_HEIGHT - uEyeAltitude) / rayDir.y;
  vec2 uv = rayDir.xz * t * 0.00016;
  uv += vec2(uTime * 0.0022, uTime * 0.0009);   // slow wind drift

  float n = skyFbm(uv * 2.1);
  // Stretch the noise along the wind axis to get banded, streaky cirrus rather
  // than blobby cumulus, which suits a hot, dry, high-pressure sky.
  n = mix(n, skyFbm(uv * vec2(0.55, 3.4)), 0.45);

  // Five octaves of value noise land in roughly [0.2, 0.8] with a mean near
  // 0.48, so the threshold has to live in that window; comparing against
  // (1 - coverage) directly produces an empty sky for any sane coverage value.
  float thr = mix(0.74, 0.30, uCloudCoverage);
  float cover = smoothstep(thr, thr + 0.20, n);

  // Fade the deck out toward the horizon: at grazing angles the plane
  // approximation breaks down and the noise would smear into stripes.
  cover *= smoothstep(0.012, 0.16, rayDir.y);
  // ...and out at the zenith-facing far distance so it does not tile visibly.
  cover *= 1.0 - smoothstep(60000.0, 190000.0, t);
  cover *= uCloudAmount;

  if (cover <= 0.001) return vec4(0.0);

  // Self-shadow: compare against the deck a short distance sunward. Where the
  // cloud is thicker toward the sun the sample is darker.
  vec2 sunOffset = normalize(sunDir.xz + vec2(1e-4)) * 0.06;
  float toward = skyFbm((uv + sunOffset) * 2.1);
  float shade = clamp(1.0 - (toward - n) * 1.9, 0.25, 1.0);

  // Forward scattering through the thin edges gives the silver lining.
  float mu = dot(rayDir, sunDir);
  float silver = pow(clamp(mu, 0.0, 1.0), 12.0) * (1.0 - cover) * 1.6;

  // Lambert cloud top: albedo 0.82, cos of the sun's elevation on a flat deck.
  float ndl = max(sunDir.y, 0.0);
  vec3 lit = sunIrradiance * (0.82 / PI) * ndl * (shade + silver) + skyColor * 0.9;
  return vec4(lit, cover);
}

void main() {
  vec3 rayDir = normalize(vRayDir);
  vec3 sunDir = normalize(uSunDirection);

  vec3 viewT;
  float groundT;
  float groundW;
  vec3 color = atmScatter(rayDir, sunDir, uEyeAltitude, uMieG, viewT, groundT, groundW) * uSunIntensity;

  // Transmitted solar irradiance. Reused for the disc, the clouds and the
  // ground, and mirrored exactly on the CPU for the DirectionalLight colour, so
  // every light in the game traces back to this one number.
  vec3 eye = vec3(0.0, ATM_R_GROUND + uEyeAltitude, 0.0);
  vec3 sunIrradiance = atmTransmittance(eye, sunDir) * uSunIntensity;

  if (groundW > 0.0) {
    // Analytic planet surface. Only ever visible below the true horizon, but
    // it is what stops downward-facing normals sampling black from the IBL.
    vec3 n = normalize(eye + rayDir * groundT);
    float ndl = max(dot(n, sunDir), 0.0);
    vec3 gsun = atmTransmittance(eye + rayDir * groundT, sunDir);
    vec3 ground = uGroundAlbedo * (ndl * gsun * uSunIntensity / PI + vec3(${GROUND_SKY_AMBIENT}));
    // groundW is the same band atmScatter used to blend its integration
    // limit, so the surface and the sky behind it fade in together.
    color += ground * viewT * groundW;
  } else {
    // Solar disc, with a physically motivated limb darkening profile.
    float cosTheta = clamp(dot(rayDir, sunDir), -1.0, 1.0);
    float theta = acos(cosTheta);
    float r = theta / SUN_ANGULAR_RADIUS;
    if (r < 1.06) {
      float edge = 1.0 - smoothstep(0.985, 1.03, r);
      float mu = sqrt(max(0.0, 1.0 - min(r * r, 1.0)));
      // I(mu) = 1 - u(1 - mu); u ~ 0.6 in the visible band.
      float limb = 1.0 - 0.6 * (1.0 - mu);
      color += viewT * uSunDiscIntensity * uSunIntensity * edge * limb;
    }

    vec4 clouds = skyClouds(rayDir, sunDir, color, sunIrradiance);
    color = mix(color, clouds.rgb, clouds.a);
  }

  // Half-float render targets top out around 65504; keep a wide margin so a
  // bloom mip chain cannot produce Inf and poison the whole frame with NaN.
  color = min(color, vec3(6000.0));

  gl_FragColor = vec4(color, 1.0);
}
`;
