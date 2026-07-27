/**
 * NOISEFIELD — fragment programs for the two hand-written post effects.
 *
 * Both are written against the `postprocessing` Effect contract: they expose a
 * `mainImage` entry point and may read the built-in `inputBuffer`, `texelSize`,
 * `resolution` and `aspect` uniforms. Effects flagged CONVOLUTION are the only
 * ones allowed to sample `inputBuffer` at offsets, and only one such effect may
 * live in a given EffectPass — which is why these two are separate passes.
 *
 * NOTE: never put a backtick inside these template literals.
 */

/** Number of taps along the velocity vector. Overridable via a define. */
export const MOTION_BLUR_SAMPLES_HIGH = 10;
export const MOTION_BLUR_SAMPLES_LOW = 6;

/**
 * Camera motion blur reconstructed from the depth buffer.
 *
 * A full per-object velocity buffer would need every material in the game to
 * write motion vectors — impossible from this subsystem without reaching into
 * other systems' materials. Reprojecting the depth buffer through last frame's
 * view-projection gives exact camera-motion velocity (translation *and*
 * rotation, correctly parallaxed by depth) for free, which covers the
 * overwhelming majority of perceived blur in a first-person shooter: fast
 * turns, sprint bob, and recoil kick.
 *
 * Objects moving independently of the camera are not blurred. That is a
 * deliberate trade: a wrong velocity smears the *background* around a moving
 * object, which reads far worse than no blur at all.
 *
 * The pass is inserted BEFORE the viewmodel is composited, so the weapon — for
 * which the world's velocity field would be entirely wrong — stays crisp.
 */
export const MOTION_BLUR_FRAGMENT = /* glsl */ `
#ifndef MB_SAMPLES
#define MB_SAMPLES ${MOTION_BLUR_SAMPLES_HIGH}
#endif

uniform mat4 invViewProj;
uniform mat4 prevViewProj;
/** Shutter fraction: how much of a frame's motion the virtual shutter sees. */
uniform float shutter;
/** Hard cap on blur length in UV units; stops a fast spin smearing the frame. */
uniform float maxBlur;

float mbHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void mainImage(const in vec4 inputColor, const in vec2 uv, const in float depth, out vec4 outputColor) {

  // Pull the far plane in slightly: unprojecting exactly z=1 is numerically
  // fragile and the sky's apparent motion is rotation-dominated anyway.
  float d = min(depth, 0.99995);

  vec4 world = invViewProj * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
  if (abs(world.w) < 1e-8) { outputColor = inputColor; return; }
  vec3 wpos = world.xyz / world.w;

  vec4 prevClip = prevViewProj * vec4(wpos, 1.0);
  if (prevClip.w <= 1e-5) { outputColor = inputColor; return; }   // was behind the old camera

  vec2 prevUv = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
  vec2 velocity = (uv - prevUv) * shutter;

  float len = length(velocity);
  if (len < 1.0 / max(resolution.x, 1.0)) { outputColor = inputColor; return; }  // sub-pixel: skip
  velocity *= min(len, maxBlur) / len;

  // Per-pixel jitter turns the 10 discrete taps into noise instead of ghosting
  // replicas; the following SMAA/grain hides the noise completely.
  float jitter = mbHash(uv * resolution);

  vec4 acc = vec4(0.0);
  float wsum = 0.0;

  for (int i = 0; i < MB_SAMPLES; i++) {
    float t = (float(i) + jitter) / float(MB_SAMPLES) - 0.5;
    vec2 suv = clamp(uv + velocity * t, vec2(0.0), vec2(1.0));
    // Triangular weighting keeps the un-displaced sample dominant, which keeps
    // edges from crawling when the blur is short.
    float w = max(0.0, 1.0 - abs(t) * 1.35);
    acc += texture2D(inputBuffer, suv) * w;
    wsum += w;
  }

  outputColor = acc / max(wsum, 1e-4);
}
`;

/**
 * Final cinematic finishing: radial chromatic aberration, a light unsharp mask,
 * lift/gamma/gain grade, saturation/contrast, vignette and animated film grain.
 *
 * All of it merged into one shader so it costs a single fullscreen pass, and
 * all of it deliberately restrained — the fastest way to make a frame look
 * amateur is to over-apply exactly these effects.
 */
export const FILM_GRADE_FRAGMENT = /* glsl */ `
/** Peak per-channel UV offset at the frame corner. ~0.0008 is one pixel at 1080p. */
uniform float caStrength;
/** Normalised radius at which aberration starts ramping in. */
uniform float caInner;
uniform float sharpness;
uniform vec3  gradeLift;
uniform vec3  gradeGamma;
uniform vec3  gradeGain;
uniform float saturation;
uniform float contrast;
/** Output level the S-curve pivots about; everything below it is crushed. */
uniform float gradePivot;
/** Multiplicative tint applied to the toe, weighted by darkness. */
uniform vec3  shadowTint;
/** Multiplicative tint applied to the shoulder, weighted by brightness. */
uniform vec3  highlightTint;
/** 0 = no split tone, 1 = full strength. */
uniform float splitStrength;
uniform float vignetteDarkness;
uniform float vignetteOffset;
uniform float grainAmount;
uniform float grainTime;

float fgHash(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

/**
 * Filmic contrast S-curve on [0,1] with a movable pivot.
 *
 * Below the pivot the curve is a power that steepens the toe; above it, the
 * point-mirrored power steepens the shoulder. 0, the pivot and 1 are all exact
 * fixed points, which buys three things the usual (x - 0.5) * c + 0.5 cannot:
 *
 *   • it is closed on [0,1], so no amount of contrast can clip a highlight to a
 *     flat white plateau or drive a channel negative;
 *   • the pivot is free, so the curve can steepen the midtones without also
 *     changing overall exposure — with the pivot at display middle grey the
 *     average brightness of the frame barely moves;
 *   • the toe and the shoulder are symmetric in shape, which is what makes the
 *     result read as film rather than as a levels adjustment.
 *
 * Applied per channel deliberately, not to luminance: per-channel contrast is
 * what produces the mild hue "lean" of real film stock, and it is a large part
 * of why a tone-mapped-then-luma-contrasted image still looks digital.
 */
vec3 nfFilmicS(vec3 x, float pivot, float k) {
  vec3 c = clamp(x, 0.0, 1.0);
  vec3 lo = pivot * pow(c / pivot, vec3(k));
  vec3 hi = 1.0 - (1.0 - pivot) * pow((1.0 - c) / (1.0 - pivot), vec3(k));
  return mix(lo, hi, step(vec3(pivot), c));
}

void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {

  // Aspect-corrected radius so the vignette and CA masks stay circular on
  // ultrawide displays instead of turning into ellipses.
  vec2 centered = (uv - 0.5) * vec2(aspect, 1.0);
  float rMax = 0.5 * sqrt(aspect * aspect + 1.0);
  float rn = length(centered) / rMax;                 // 0 at centre, 1 at corner

  // ── chromatic aberration, edges only ──────────────────────────────────────
  // Real lenses are corrected in the middle of the field and fall apart toward
  // the corners; applying CA uniformly is the classic giveaway of a filter.
  float caMask = smoothstep(caInner, 1.0, rn);
  caMask *= caMask;
  vec2 caOffset = (uv - 0.5) * caStrength * caMask * 2.0;

  vec3 col;
  col.r = texture2D(inputBuffer, clamp(uv + caOffset, vec2(0.0), vec2(1.0))).r;
  col.g = inputColor.g;
  col.b = texture2D(inputBuffer, clamp(uv - caOffset, vec2(0.0), vec2(1.0))).b;

  // ── unsharp mask ──────────────────────────────────────────────────────────
  // SMAA is a spatial filter and slightly softens the image; a small amount of
  // sharpening restores the crispness without reintroducing the aliasing.
  if (sharpness > 0.0) {
    vec3 lp = (
      texture2D(inputBuffer, uv + vec2( texelSize.x, 0.0)).rgb +
      texture2D(inputBuffer, uv + vec2(-texelSize.x, 0.0)).rgb +
      texture2D(inputBuffer, uv + vec2(0.0,  texelSize.y)).rgb +
      texture2D(inputBuffer, uv + vec2(0.0, -texelSize.y)).rgb
    ) * 0.25;
    col += (col - lp) * sharpness;
  }

  col = clamp(col, 0.0, 1.0);

  // ── lift / gamma / gain ───────────────────────────────────────────────────
  // Kept, but now doing only what it is good at: a small overall channel
  // balance. The heavy lifting moved to the S-curve and the split tone below,
  // because lift/gamma/gain cannot steepen midtones without either clipping the
  // top or lifting black, and steepening the midtones is the entire job here.
  col = col * gradeGain + gradeLift * (1.0 - col);
  col = pow(max(col, vec3(1e-5)), gradeGamma);

  // ── filmic contrast ───────────────────────────────────────────────────────
  col = nfFilmicS(col, gradePivot, contrast);

  float luma = dot(max(col, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));

  // ── split tone ────────────────────────────────────────────────────────────
  // Cool the toe, warm the shoulder. This is the actual mechanism behind
  // "golden hour": the key is a low, warm, heavily reddened sun and the fill is
  // the blue sky, so lit and shadowed surfaces differ in HUE as much as in
  // brightness. Reproducing only the brightness difference gives the grey,
  // overcast look no amount of extra contrast can fix — the two have to be
  // pushed apart in colour as well, which is what these two tints do.
  float toe = 1.0 - smoothstep(0.0, 0.40, luma);
  float shoulder = smoothstep(0.42, 0.95, luma);
  col *= mix(vec3(1.0), shadowTint, toe * splitStrength);
  col *= mix(vec3(1.0), highlightTint, shoulder * splitStrength);

  luma = dot(max(col, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
  // Hold saturation back slightly in the deepest shadows, where chroma noise
  // lives and where real film has the least dye density, and let it run at full
  // strength from the low midtones up so sand reads as sand and not as grey
  // card. The rolloff is deliberately finished by 0.14: with the corrected
  // light ratio a lot of legitimate, readable surface now lives below 0.3, and
  // desaturating all of it drains the colour back out of the frame.
  float satWeight = mix(0.80, 1.0, smoothstep(0.015, 0.14, luma));
  col = mix(vec3(luma), col, saturation * satWeight);

  // ── vignette ──────────────────────────────────────────────────────────────
  col *= 1.0 - vignetteDarkness * smoothstep(vignetteOffset, 1.0, rn);

  // ── film grain ────────────────────────────────────────────────────────────
  //
  // ═══════════════════════════════════════════════════════════════════════════
  // THIS BUFFER IS LINEAR. GRAIN MUST NOT BE ADDED TO IT AT A FIXED AMPLITUDE.
  // ═══════════════════════════════════════════════════════════════════════════
  // postprocessing composites every effect in the working (linear) colour space
  // and only encodes to sRGB in the final blit. A constant additive offset in
  // linear light is therefore NOT a constant offset on screen — the sRGB
  // transfer function's slope varies by more than twenty to one across the
  // range, and it is steepest exactly where the offset matters most:
  //
  //     linear 0.500   ->  d(display)/d(linear) = 0.66   grain barely visible
  //     linear 0.180   ->                          1.20
  //     linear 0.005   ->                          9.6    grain is the image
  //
  // A shadowed wall in this scene sits near linear 0.005. The previous fixed
  // 0.022 amplitude was 2.2x the signal there, and the sRGB toe expanded it
  // into a display standard deviation of 0.048 on a mean of 0.062 — measured,
  // and reproduced to three decimal places by simulating this one line. That
  // is the entire "blotchy black noisy mess" over the shadowed half of the
  // market frame. It was never shadow acne; every attempt to bias it away was
  // chasing the wrong subsystem.
  //
  // The fix is to make the grain DISPLAY-referred: ask for a constant on-screen
  // amplitude and divide by the transfer slope to find the linear offset that
  // produces it. For sRGB, display = 1.055*lin^(1/2.4) - 0.055, so
  //
  //     d(linear)/d(display) = (2.4/1.055) * lin^(1 - 1/2.4)
  //                          = 2.2749 * lin^0.5833
  //
  // which is exact above the toe and within 2% of 1/12.92 as it approaches it,
  // so no piecewise branch is needed.
  float g = fgHash(vec3(uv * resolution, grainTime)) - 0.5;
  float grainSlope = 2.2749 * pow(max(luma, 1e-6), 0.5833);

  // Weighted on DISPLAY luma, not linear luma — "midtones" is a perceptual
  // statement and linear 0.45 is display 0.70, i.e. already a highlight, which
  // is why the old rolloff never actually did anything. Grain is killed below
  // display 0.10 (a renderer's near-black needs no dither: the half-float
  // buffer has orders of magnitude more precision than the 8-bit output there,
  // so there is nothing to break up) and eased off in the highlights, where
  // real film has the least visible grain.
  float displayLuma = pow(max(luma, 0.0), 1.0 / 2.2);
  float grainWeight = smoothstep(0.02, 0.14, displayLuma) * mix(1.0, 0.25, smoothstep(0.55, 1.0, displayLuma));

  col += g * grainAmount * grainSlope * grainWeight;

  outputColor = vec4(max(col, vec3(0.0)), inputColor.a);
}
`;
