/**
 * NOISEFIELD — Lighting.
 *
 * Owns everything about how the world is lit:
 *
 *   • the golden-hour key light and its physically derived colour
 *   • cascaded shadow maps (stabilised, slope-biased, distance-softened)
 *   • the sky/ground hemisphere fill that keeps shadows blue instead of black
 *   • exponential height fog and its aerial-perspective tint
 *
 * All four are driven from the single scattering model in
 * `shaders/Atmosphere.ts`, so moving the sun updates the key colour, the fill,
 * the fog and (via `Sky`) the visible dome and the IBL together.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * CASCADED SHADOW MAPS
 * ─────────────────────────────────────────────────────────────────────────────
 * A single 2048² frustum stretched over a 100 m arena gives ~7 cm texels: every
 * contact shadow is a staircase. Instead N directional lights share one
 * direction, each covering one slice of the view frustum, and a patched
 * `lights_fragment_begin` (see `shaders/CSMShader.ts`) lets exactly one slice
 * light each fragment, cross-faded at the boundaries.
 *
 * Split scheme — practical (Zhang et al.), lambda 0.6:
 *
 *     d_i = lerp( n + (f-n)·i/N ,  n·(f/n)^(i/N) ,  0.6 )
 *
 * The logarithmic term is optimal for perspective projections but wastes almost
 * the entire first cascade on the 5 cm in front of the near plane; the uniform
 * term is the opposite. 0.6 leans toward logarithmic, which is right for an FPS
 * where the interesting shadows are 2–30 m away.
 *
 * Stabilisation — two independent techniques, both mandatory:
 *
 *   1. The ortho extent comes from the *bounding sphere* of the frustum slice,
 *      computed analytically. A sphere's radius depends only on (near, far,
 *      fov, aspect) — never on camera orientation — so the shadow projection
 *      never changes scale as the player turns. Bounding-box fitting (what most
 *      three.js CSM code does) changes size every frame and makes shadow edges
 *      visibly boil.
 *   2. The sphere centre is snapped to whole shadow-map texels in light space,
 *      so sub-texel camera translation cannot shift which texel a given world
 *      point lands in. Without this, walking forward makes every shadow edge
 *      crawl — the single most recognisable "this is a WebGL demo" tell.
 *
 * Bias — `shadow.normalBias` is scaled to the cascade's world texel size, which
 * makes it a true slope-scaled bias (the required offset is proportional to
 * texel footprint, and the offset is along the normal so flat-on surfaces pay
 * nothing). `shadow.bias` then only has to absorb depth quantisation, so it can
 * stay small enough that nothing peter-pans.
 *
 * Penumbra — `shadow.radius` grows per cascade *and* the world size of a texel
 * grows per cascade, so the filter footprint in metres grows superlinearly with
 * distance: sharp contact shadows up close, soft shadows far away. This is a
 * cheap stand-in for PCSS that costs literally one extra uniform.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * INTEGRATION NOTES / ASSUMPTIONS
 * ─────────────────────────────────────────────────────────────────────────────
 *  • three sorts shadow-casting lights first when building light uniforms, and
 *    Array#sort is stable, so the cascade lights occupy directional indices
 *    0..N-1 as long as no *other* subsystem adds a shadow-casting
 *    DirectionalLight to the main scene. Non-shadowing directional lights added
 *    by other systems are fine: they land at index >= N and take the stock path.
 *  • Materials must be patched with the `NF_CSM_CASCADES` / `NF_FOG`
 *    defines. `Lighting` does this automatically by rescanning the scene every
 *    few frames, and `registerMaterial()` is available for anything created
 *    and rendered within a single frame.
 */

import * as THREE from 'three';
import type { EngineContext, FrameContext, System } from '../core/Types';
import {
  DEFAULT_SUN_AZIMUTH_DEG,
  DEFAULT_SUN_ELEVATION_DEG,
  evaluateSky,
  GROUND_ALBEDO,
  luminance,
  skyIrradiance,
  SUN_INTENSITY,
  sunDirectionFromAngles,
  sunTransmittance,
} from './shaders/Atmosphere';
import {
  CSM_DEFINE,
  CSM_UNIFORM_FADE,
  FOG_DEFINE,
  FOG_UNIFORM_PARAMS,
  FOG_UNIFORM_SUN_COLOR,
  FOG_UNIFORM_SUN_DIR,
  FOG_UNIFORM_UP_COLOR,
  installShaderChunkPatches,
  SHADOW_UNIFORM_TEXEL_DEPTH,
} from './shaders/CSMShader';

/**
 * Structural view of the material flags we duck-type against. three tags its
 * built-ins with `isMeshXxxMaterial` booleans; using those instead of
 * `instanceof` avoids breaking if a system imports three through a different
 * module specifier and ends up with a second copy of the class objects.
 */
interface MaterialFlags {
  readonly isMeshStandardMaterial?: boolean;
  readonly isMeshPhysicalMaterial?: boolean;
  readonly isMeshPhongMaterial?: boolean;
  readonly isMeshLambertMaterial?: boolean;
  readonly isMeshToonMaterial?: boolean;
  readonly isShaderMaterial?: boolean;
  readonly lights?: boolean;
}

/**
 * How far from the camera cascaded shadows are rendered at all, metres.
 *
 * Deliberately much shorter than `camera.far`. With an 80 degree vertical FOV
 * on a 16:9 display the horizontal field is ~122 degrees, and the bounding
 * sphere of a frustum that wide has a radius of roughly 2x its far distance —
 * so every extra metre of shadow range costs texel density quadratically.
 * 85 m covers a 100 m arena end to end; beyond that the height fog has already
 * taken over and a missing shadow is invisible.
 */
const DEFAULT_SHADOW_DISTANCE = 78;

/**
 * Lower bound on the near plane used for split fitting. The camera's true near
 * is 5 cm, which is inside the player's own body; feeding it to the logarithmic
 * term collapses the first split to centimetres and wastes an entire cascade.
 */
const SPLIT_MIN_NEAR = 0.5;

/**
 * Hard ceiling on a cascade's ortho radius, metres.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THIS IS THE SECOND HALF OF THE ACNE FIX. It is worth reading.
 * ─────────────────────────────────────────────────────────────────────────────
 * The minimal enclosing sphere of a frustum slice is, for a wide field of view,
 * dominated entirely by the FAR CAP: with an 80 degree vertical FOV on 16:9 the
 * horizontal field is ~122 degrees, k = sqrt(tanH^2 + tanV^2) = 1.71, and the
 * analytic centre always lands past the far plane, so the fit collapses to
 * "centre on the far plane, radius = far * k". At a 78 m shadow distance that
 * is a 133 m radius — a 267 m ortho box over a 90 m arena, i.e. more than nine
 * tenths of the shadow map's AREA aimed at empty desert outside the level.
 *
 * Every acne term is proportional to the world size of a texel, so that waste
 * is not merely a sharpness loss: at 2048 it put the outer cascade at 13 cm per
 * texel, needing over a metre of normal offset to be acne-free, which is why
 * the outer cascades stippled no matter how the bias was tuned. Clamping the
 * radius is what makes the whole bias budget tractable.
 *
 * 72 m is chosen from the level, not by eye: the arena is 90 x 90 m, so its
 * half-diagonal is 63.6 m, and a 72 m box centred on the middle of the cascade
 * slice reaches every corner of the level from anywhere inside it. Geometry
 * outside that box is off-map and has nothing to shadow.
 */
const MAX_CASCADE_RADIUS = 72;

/**
 * World-space normal offset for the shadow lookup, in texels of the cascade's
 * own footprint.
 *
 * With the receiver-plane depth bias doing the slope work (see
 * `shaders/CSMShader.ts`), this term no longer has to scale with the sun's
 * elevation — it only has to cover the sub-texel error left by interpolating
 * the shadow coordinate across a triangle, which is about one texel. That is
 * the entire reason it could drop from 3.4 texels to 1: a normal offset is the
 * ONLY remaining term that peter-pans, since it physically lifts the receiver
 * off its own surface, and it does so by offset/tan(elevation) = 3.2x its own
 * size horizontally at this sun angle.
 */
const NORMAL_BIAS_TEXELS = 1.0;

/**
 * Ceiling on the world-space normal bias, metres. Only binds on the outermost
 * cascade. 8 cm of offset there is ~25 cm of horizontal shadow contraction at
 * a 17.5 degree sun, which is under two pixels at 40 m.
 */
const MAX_NORMAL_BIAS = 0.08;

/**
 * Residual constant depth bias, metres of light-space depth.
 *
 * The shadow map is a 24-bit depth texture over a ~170 m range, so its own
 * quantum is ~10 microns; this exists only to absorb rasteriser rounding on
 * polygons that lie exactly in the light plane. Deliberately tiny — the
 * receiver-plane and normal-offset terms are the ones doing real work, and a
 * fat constant here would peter-pan everything uniformly, near shadows worst.
 */
const CONST_DEPTH_BIAS = 0.004;

/**
 * Extra depth in front of each cascade so off-screen casters still occlude.
 *
 * Sized from the level: the tallest structure is the ~14 m comms tower and the
 * buildings top out near 10 m, so 32 m of headroom above a cascade sphere whose
 * own radius is already 16-72 m cannot miss a caster. The previous 55 m only
 * widened the shadow camera's depth range for nothing.
 */
const CASTER_EXTENT = 32;
/**
 * Practical split blend factor: 0 = uniform, 1 = logarithmic.
 *
 * Raised from 0.6. Logarithmic splits put more of the shadow map where the
 * player is looking; at 0.6 cascade 0 stretched to 13 m and carried 2.2 cm
 * texels, at 0.72 it stops at 9.3 m and carries 1.6 cm texels. The far cascade
 * is unaffected (its far plane is the shadow distance either way), so this is
 * a free sharpening of everything inside 25 m.
 */
const SPLIT_LAMBDA = 0.72;
/** Cross-fade band around a split, as a fraction of the cascade's own length. */
const SPLIT_BLEND_FRACTION = 0.11;
/** Frames between automatic scene rescans for unpatched materials. */
const MATERIAL_SCAN_INTERVAL = 24;

/**
 * Direct solar irradiance, in exactly the units the scattering integral uses.
 * Sharing the constant is what keeps the sun, the sky and the fog on the same
 * photometric scale — pick them independently and the sky will always look
 * either pasted-on bright or dead next to the lit geometry.
 */
const SUN_IRRADIANCE = SUN_INTENSITY;

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * THE SUN-TO-SKY RATIO — the single most important number in this file
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Diffuse sky irradiance on a horizontal surface, as a fraction of the DIRECT
 * solar irradiance on that same surface.
 *
 * A fragment the sun cannot see is lit by the ambient and nothing else, so this
 * ratio *is* the depth of every shadow in the game:
 *
 *     shadowed / lit  =  r / (1 + r)
 *
 * Clear-sky measurements at ~17 degrees solar elevation put the diffuse
 * horizontal component at roughly a third of the direct horizontal component
 * (~50 W/m² against ~150 W/m²), i.e. r ≈ 0.33 → shadows at 25% of lit. We aim a
 * little under that because two later stages lift the low end again: the ACES
 * toe raises dark values more than bright ones, and N8AO's occlusion is applied
 * multiplicatively on top.
 *
 * WHAT WAS WRONG BEFORE: the PMREM sky environment ran at `environmentIntensity
 * = 1`, delivering the entire hemispherical sky irradiance, and a HemisphereLight
 * carrying another 32% of it was added on top — the same sky counted twice, for
 * r ≈ 1.2. That puts shadows at ~55% of lit in linear light and ~80% after tone
 * mapping, which is precisely the flat, shadowless, overcast image this constant
 * exists to prevent. Measured before the fix: sky irradiance 0.586 against a
 * direct 0.991 from the environment ALONE, plus 0.413 more from the hemisphere,
 * for r = 1.01 — the same sky, counted twice.
 *
 * TUNED VALUE: 0.26, predicting a 0.21 shadow-to-lit ratio. The first pass used
 * 0.20 (predicting 0.167); it measured beautifully and looked wrong. Inside the
 * market street, where a wall sees very little sky and takes nearly all its
 * light from this term, 26% of the frame fell below 0.10 display luma and the
 * left half of the shot became a featureless black hole. 0.26 is also nearer
 * the real clear-sky figure, so the honest reading is that the physics was
 * right and the first estimate was simply pessimistic.
 */
const AMBIENT_TO_DIRECT = 0.26;

/**
 * Share of the ambient budget delivered by the HemisphereLight rather than by
 * the PMREM environment.
 *
 * Neither can be zero. The environment carries all the *directionality* (a wall
 * facing the bright sun-side horizon must be warmer than one facing the dark
 * anti-sun sky, which a two-colour hemisphere cannot express) and is what makes
 * metal reflect a real sky. The hemisphere carries the sand bounce into
 * downward-facing normals with far more control than the IBL's analytic ground,
 * and is the only fill that reaches materials which ignore `scene.environment`.
 * A third/two-thirds split keeps the environment dominant while leaving the
 * hemisphere strong enough to be a usable shadow-colour knob.
 */
const HEMI_SHARE = 0.34;

/**
 * How much brighter the hemisphere's lower (sand bounce) half is than its upper
 * (sky) half.
 *
 * Dry sand at ~0.32 average albedo under a nearly-vertical-facing surface
 * returns more light than the clear sky above it does at golden hour, and that
 * warm up-fill on chins, undersides and the lower half of every wall is a large
 * part of why desert footage reads as desert. Capped well below the physical
 * ratio because the IBL's analytic ground already supplies some of it.
 */
const HEMI_GROUND_BOOST = 1.9;

/**
 * Elevation, in degrees, at which the sky is sampled for the fog colours.
 *
 * NOT the horizon. Two independent reasons:
 *
 *  1. NUMERICS. A ray at 2-3 degrees traverses ~500 km of atmosphere. The
 *     quadratically spaced march resolves that badly, and it resolves it
 *     differently on the CPU (24 steps) than in the sky shader (14), so the
 *     grazing sample disagrees with the dome the fog is supposed to blend into.
 *     Sampling at 2.6 degrees is what produced the previous #d7e0c5 fog: a
 *     green-biased colour that appears nowhere in the rendered sky, which is
 *     measurably blue (B > G > R) at every elevation actually on screen.
 *  2. FRAMING. Distant geometry in a compound of 4-10 m buildings seen from
 *     20-90 m away is silhouetted against sky roughly 3-10 degrees up, not
 *     against the true horizon — which is behind the buildings and never
 *     visible. Matching what is on screen is the whole point of fog colour.
 *
 * 6, not 9. The previous value was chosen for the sky the fog blends into when
 * you look UP past a rooftop; but the pixels that actually carry haze are the
 * mid-ground silhouettes, and the sky directly behind those sits at 2-8
 * degrees, where Mie scattering has already made it measurably brighter. Fog
 * sampled 3 degrees too high is fog that is darker than the sky it abuts, which
 * is a band along every distant roofline.
 */
const FOG_SAMPLE_ELEVATION_DEG = 6;

/**
 * Elevation sampled for the warm in-scatter colour on the sun side. Lower than
 * the away-side sample because the aureole that makes looking toward a low sun
 * glow is concentrated within a few degrees of the disc.
 */
const FOG_SUN_SAMPLE_ELEVATION_DEG = 5;

/**
 * Extinction coefficient at the fog base height, per metre.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY THIS IS SO MUCH SMALLER THAN IT LOOKS LIKE IT SHOULD BE
 * ─────────────────────────────────────────────────────────────────────────────
 * The trap in this number is that fogFactor is NOT the strength of the effect.
 * The fog colour here is sky radiance, and at golden hour the low sky is around
 * six times brighter in linear terms than a shadowed stucco wall. Mixing in
 * even 20% of something 6x brighter does not veil the wall, it nearly doubles
 * it — and it does the same to its neighbour, so the CONTRAST between them
 * collapses far faster than the mix factor suggests. At the previous 0.0034 the
 * 40-70 m buildings measured 0.49-0.62 display luma across their entire depth,
 * an 0.13 range where the near ones span 0.2-0.9: the mid-ground had no
 * structure left at all.
 *
 * Sized instead from what the frame needs to stay readable:
 *
 *     30 m (gameplay silhouettes)     4%   — separation, no visible veil
 *     60 m (far side of the compound)  8%   — reads as "behind"
 *     90 m (the far wall)             12%   — depth cue, detail intact
 *
 * 1 - exp(-0.0015·d) gives exactly that. The arena is only 90 m across; on a
 * map this size aerial perspective has to be a hint, not a curtain.
 */
const FOG_DENSITY = 0.0015;

/**
 * Fog height falloff, 1/m. 1/0.03 = a 33 m scale height, so dust pools in the
 * streets and thins out above the rooftops instead of fogging the sky.
 */
const FOG_HEIGHT_FALLOFF = 0.03;

/** Strength of the warm forward in-scatter lobe blended into the fog colour. */
const FOG_INSCATTER = 1.0;

/**
 * Fraction of the fog's colour taken from the sunlit ground rather than the sky.
 *
 * A near-horizontal ray a metre or two above the desert has the GROUND filling
 * the lower half of every scattering event's incoming hemisphere, not the sky.
 * Colouring near-ground haze purely with sky radiance therefore overstates its
 * brightness by roughly a factor of two and gives it the sky's blue, which is
 * the second half of why the mid-ground was washing out — and why it washed out
 * toward a cool grey instead of the warm dust the rest of the frame is made of.
 * Blending a third of the sand bounce in both darkens the haze to something
 * physically sane and warms it into the palette.
 */
const FOG_GROUND_SHARE = 0.33;

/**
 * Extra ambient fill for the first-person viewmodel, as a multiple of the
 * world's.
 *
 * A deliberate, conventional cheat. The weapon is a large near-black object a
 * few centimetres from the lens; at the world's honest sun-to-sky ratio its
 * shadow side goes to almost nothing and the gun reads as a silhouette cut out
 * of the frame. Every shipping FPS lifts the viewmodel's fill for exactly this
 * reason. Kept modest so the weapon still swings dark when the player turns
 * away from the sun.
 */
const VIEWMODEL_FILL_BOOST = 2.2;

// Scratch — this class must never allocate in `update`.
const _fwd = new THREE.Vector3();
const _center = new THREE.Vector3();
const _tmp = new THREE.Vector3();
const _basis = new THREE.Matrix4();
const _basisInv = new THREE.Matrix4();
const _up = new THREE.Vector3(0, 1, 0);
const _dirUp = new THREE.Vector3(0, 1, 0);
const _dirHorizonSun = new THREE.Vector3();
const _dirHorizonOpp = new THREE.Vector3();
/** Measured hemispherical sky irradiance; drives the whole ambient budget. */
const _skyIrradiance = new THREE.Color();
const _chroma = new THREE.Color();
/** Dedicated scratch for the viewmodel rig so it can never alias the CSM maths. */
const _vmSun = new THREE.Vector3();
const _color = new THREE.Color();

export class Lighting implements System {
  readonly name = 'lighting';

  // ── public contract ────────────────────────────────────────────────────────

  /** Unit vector pointing FROM the sun TOWARD the scene (so `y < 0`). */
  readonly sunDirection = new THREE.Vector3();
  /** Linear-space colour of the direct sun, derived from atmospheric transmittance. */
  sunColor = new THREE.Color(1, 1, 1);
  /** Primary key light. Also cascade 0 when CSM is active. */
  readonly sun: THREE.DirectionalLight;

  /** All cascade lights, near → far. `cascades[0] === sun`. */
  readonly cascades: THREE.DirectionalLight[] = [];
  /** Sky/ground fill. Public so a night-vision or flashbang effect can dim it. */
  readonly hemisphere: THREE.HemisphereLight;

  /**
   * Key light for the first-person viewmodel scene.
   *
   * `viewScene` is a separate scene with its own camera, so none of the world's
   * lights reach it — a weapon rendered there with only the environment map is
   * flat and obviously pasted on. This light mirrors the sun, re-expressed in
   * the viewmodel scene's space every frame, so the weapon's highlights swing
   * across the receiver as the player turns, exactly as they should.
   *
   * Tagged `userData.nfViewLight` so the weapon system can find or replace it.
   */
  readonly viewKey: THREE.DirectionalLight;
  /** Ambient fill for the viewmodel scene, matching the world hemisphere. */
  readonly viewFill: THREE.HemisphereLight;

  /** Zenith sky radiance, linear. Consumed by fog / UI / other systems. */
  readonly zenithColor = new THREE.Color();
  /** Low-sky radiance away from the sun. Base (cool) fog colour. */
  readonly horizonColor = new THREE.Color();
  /** Low-sky radiance toward the sun. Warm aerial-perspective in-scatter. */
  readonly horizonSunColor = new THREE.Color();
  /** Warm light bouncing off the sand, used as the hemisphere's lower colour. */
  readonly groundBounceColor = new THREE.Color();

  /**
   * Hemispherical sky irradiance on an up-facing surface at
   * `environmentIntensity = 1`. Exposed because it is the measurement the whole
   * ambient budget is solved from, and being able to read it back is the only
   * way to verify the sun-to-sky ratio without eyeballing a screenshot.
   */
  readonly skyIrradianceUp = new THREE.Color();
  /** Direct solar irradiance on a horizontal surface, Rec.709 luminance. */
  directIrradiance = 0;
  /** Total ambient irradiance on that same surface, Rec.709 luminance. */
  ambientIrradiance = 0;
  /**
   * Solved `scene.environmentIntensity`. `Sky` applies this whenever it
   * regenerates the PMREM, so the two stay in lockstep as the sun moves.
   */
  environmentIntensity = 1;
  /**
   * Predicted shadowed/lit luminance ratio on flat ground, ignoring AO and tone
   * mapping. The number the whole rebalance is tuned against; target 0.15-0.30.
   */
  shadowLitRatio = 1;

  /**
   * Bumped whenever the sun (and therefore the sky) changes. `Sky` polls this
   * to know when the IBL needs regenerating, which is far cheaper than
   * re-rendering the PMREM every frame.
   */
  atmosphereVersion = 0;

  // ── internals ──────────────────────────────────────────────────────────────

  private readonly root = new THREE.Group();
  private readonly viewRoot = new THREE.Group();
  private elevationDeg = DEFAULT_SUN_ELEVATION_DEG;
  private azimuthDeg = DEFAULT_SUN_AZIMUTH_DEG;

  private ctx: EngineContext | null = null;
  private cascadeCount = 1;
  private shadowMapSize = 2048;
  private shadowsEnabled = true;
  private csmActive = false;
  private fogActive = false;
  private shadowBiasActive = false;

  /** Split distances, length cascadeCount + 1: [near, d1, ..., far]. */
  private readonly splits: number[] = [];
  /** Per-cascade (nearStart, nearEnd, farStart, farEnd) in metres of view depth. */
  private readonly fadeWindows: THREE.Vector4[] = [];

  /**
   * Per-cascade NDC-depth-per-shadow-texel, fed to the receiver-plane bias.
   *
   * Kept exactly `cascadeCount` long. If another system ever adds its own
   * shadow-casting DirectionalLight the GLSL array grows past this, and three
   * will upload only the prefix — leaving the extra light with a zero clamp,
   * i.e. the plane fit disabled and stock constant biasing. Degrading is fine;
   * over-sizing the JS array would be a WebGL error.
   */
  private readonly shadowTexelDepth: number[] = [];

  private readonly fadeUniform = new THREE.Uniform<THREE.Vector4[]>([]);
  private readonly shadowTexelDepthUniform = new THREE.Uniform<number[]>(this.shadowTexelDepth);
  private readonly fogSunColorUniform = new THREE.Uniform<THREE.Color>(new THREE.Color());
  private readonly fogUpColorUniform = new THREE.Uniform<THREE.Color>(new THREE.Color());
  private readonly fogSunDirUniform = new THREE.Uniform<THREE.Vector3>(new THREE.Vector3());
  /** (density, heightFalloff, baseHeight, inScatterStrength) */
  private readonly fogParamsUniform = new THREE.Uniform<THREE.Vector4>(
    new THREE.Vector4(FOG_DENSITY, FOG_HEIGHT_FALLOFF, 0.0, FOG_INSCATTER),
  );

  private readonly patched = new WeakSet<THREE.Material>();
  private scanCountdown = 0;
  private disposed = false;

  constructor() {
    // Cascade 0 exists from construction so `sun` is never null for consumers
    // that grab a reference before `init` runs.
    this.sun = new THREE.DirectionalLight(0xffffff, SUN_IRRADIANCE);
    this.sun.name = 'noisefield.sun.cascade0';
    this.cascades.push(this.sun);

    // Colours carry the magnitude; intensity stays at 1 so the sky and ground
    // halves can be scaled independently (three multiplies BOTH by intensity).
    this.hemisphere = new THREE.HemisphereLight(0x6e9cc4, 0xc2a878, 1);
    this.hemisphere.name = 'noisefield.hemisphere';

    this.viewKey = new THREE.DirectionalLight(0xffffff, SUN_IRRADIANCE * 0.9);
    this.viewKey.name = 'noisefield.viewmodel.key';
    this.viewKey.userData['nfViewLight'] = true;
    this.viewFill = new THREE.HemisphereLight(0x6e9cc4, 0xc2a878, 1);
    this.viewFill.name = 'noisefield.viewmodel.fill';
    this.viewFill.userData['nfViewLight'] = true;

    sunDirectionFromAngles(this.elevationDeg, this.azimuthDeg, this.sunDirection);
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;

    // Patch the shader chunks BEFORE any material compiles. Both patches are
    // best-effort: a failure degrades quality, never correctness.
    const patches = installShaderChunkPatches();
    this.fogActive = patches.fog;
    this.shadowBiasActive = patches.shadowBias;

    const s = ctx.settings;
    this.shadowsEnabled = s.shadows;
    this.shadowMapSize = THREE.MathUtils.clamp(
      THREE.MathUtils.floorPowerOfTwo(s.shadowMapSize || 2048),
      512,
      4096,
    );
    this.cascadeCount = this.shadowsEnabled
      ? THREE.MathUtils.clamp(Math.round(s.shadowCascades) || 3, 1, 4)
      : 1;
    this.csmActive = patches.csm && this.shadowsEnabled && this.cascadeCount > 1;

    // three r185 removed PCFSoftShadowMap from the shader define table: setting
    // it silently downgrades to a single hard tap. PCFShadowMap is the good
    // path here — a hardware sampler2DShadow with a Vogel-disk kernel whose
    // width is driven by `shadow.radius`, which is exactly what the per-cascade
    // penumbra widening needs.
    ctx.renderer.shadowMap.enabled = this.shadowsEnabled;
    ctx.renderer.shadowMap.type = THREE.PCFShadowMap;

    for (let i = this.cascades.length; i < this.cascadeCount; i++) {
      const light = new THREE.DirectionalLight(0xffffff, SUN_IRRADIANCE);
      light.name = `noisefield.sun.cascade${i}`;
      this.cascades.push(light);
    }

    for (let i = 0; i < this.cascades.length; i++) {
      const light = this.cascades[i];
      light.castShadow = this.shadowsEnabled;
      light.shadow.mapSize.set(this.shadowMapSize, this.shadowMapSize);
      light.shadow.autoUpdate = true;
      // Shadow "intensity" < 1 would grey out the umbra; we want a true black
      // occlusion here and let the hemisphere + IBL fill provide the lift, so
      // that ambient occlusion and shadow do not double-darken.
      light.shadow.intensity = 1;
      this.root.add(light);
      this.root.add(light.target);
      this.fadeWindows.push(new THREE.Vector4(-2, -1, 1e6, 1e6 + 1e3));
      // Zero until the first fit; a zero clamp disables the plane fit rather
      // than letting it run against an uninitialised scale.
      this.shadowTexelDepth.push(0);
    }
    this.fadeUniform.value = this.fadeWindows;

    // Only cascade 0 contributes when CSM is off, otherwise a 3-cascade rig
    // would add the sun three times to every unpatched material.
    if (!this.csmActive) {
      for (let i = 1; i < this.cascades.length; i++) this.cascades[i].visible = false;
    }

    this.root.add(this.hemisphere);
    this.root.name = 'noisefield.lighting';
    ctx.scene.add(this.root);

    this.viewRoot.name = 'noisefield.lighting.viewmodel';
    this.viewRoot.add(this.viewKey, this.viewKey.target, this.viewFill);
    ctx.viewScene.add(this.viewRoot);

    this.refreshAtmosphere();
    this.updateCascades(ctx.camera);
    this.refreshMaterials(ctx.scene);
  }

  update(ctx: EngineContext, frame: FrameContext): void {
    if (this.disposed) return;
    this.updateCascades(ctx.camera);
    this.updateViewmodelRig(ctx);

    // Amortised rescan so materials created after boot (world streaming, decal
    // pools, ragdolls) still pick up the CSM and fog defines.
    if (--this.scanCountdown <= 0) {
      this.scanCountdown = MATERIAL_SCAN_INTERVAL;
      this.refreshMaterials(ctx.scene);
    }
    void frame;
  }

  dispose(): void {
    this.disposed = true;
    for (const light of this.cascades) {
      light.shadow.map?.dispose();
      light.dispose();
    }
    this.hemisphere.dispose();
    this.viewKey.dispose();
    this.viewFill.dispose();
    this.root.removeFromParent();
    this.viewRoot.removeFromParent();
  }

  // ── public API ─────────────────────────────────────────────────────────────

  get sunElevationDeg(): number {
    return this.elevationDeg;
  }

  get sunAzimuthDeg(): number {
    return this.azimuthDeg;
  }

  /**
   * Repositions the sun. Elevation is clamped away from the horizon and the
   * zenith: below ~2° the shadow frusta degenerate into infinitely long slivers
   * and above ~85° the light basis' `up` vector becomes ambiguous.
   */
  setSunAngles(elevationDeg: number, azimuthDeg: number): void {
    this.elevationDeg = THREE.MathUtils.clamp(elevationDeg, 2, 85);
    this.azimuthDeg = azimuthDeg;
    sunDirectionFromAngles(this.elevationDeg, this.azimuthDeg, this.sunDirection);
    this.refreshAtmosphere();
  }

  /** Fog density at the fog base height. 0 disables fog entirely. */
  setFog(density: number, heightFalloff: number, baseHeight: number, inScatter = 0.85): void {
    this.fogParamsUniform.value.set(density, heightFalloff, baseHeight, inScatter);
    const ctx = this.ctx;
    if (ctx && ctx.scene.fog instanceof THREE.FogExp2) {
      // Exactly matched, not approximately: the patched fallback branch uses
      // the same 1 - exp(-rho*d) curve, so sharing the density makes an
      // unpatched material haze identically to a patched one on any horizontal
      // ray. Mismatched curves here were visible as a tone break between
      // neighbouring surfaces at the same distance.
      ctx.scene.fog.density = density;
    }
  }

  /**
   * Opts a material into the CSM + height-fog shader paths. Safe to call more
   * than once; the second call is a no-op.
   */
  registerMaterial(material: THREE.Material): void {
    if (this.patched.has(material)) return;
    if (!this.isLitMaterial(material)) return;
    this.patched.add(material);

    const defines = (material.defines ??= {});
    if (this.csmActive) defines[CSM_DEFINE] = this.cascadeCount;
    if (this.fogActive) defines[FOG_DEFINE] = '';

    // Chain rather than replace: another subsystem may already have installed a
    // hook (procedural triplanar mapping, dissolve, etc.) and clobbering it
    // would silently break their material.
    const previous = material.onBeforeCompile.bind(material);
    material.onBeforeCompile = (shader, renderer): void => {
      previous(shader, renderer);
      // Assigned by reference so one uniform object drives every material.
      shader.uniforms[CSM_UNIFORM_FADE] = this.fadeUniform;
      shader.uniforms[SHADOW_UNIFORM_TEXEL_DEPTH] = this.shadowTexelDepthUniform;
      shader.uniforms[FOG_UNIFORM_SUN_COLOR] = this.fogSunColorUniform;
      shader.uniforms[FOG_UNIFORM_UP_COLOR] = this.fogUpColorUniform;
      shader.uniforms[FOG_UNIFORM_SUN_DIR] = this.fogSunDirUniform;
      shader.uniforms[FOG_UNIFORM_PARAMS] = this.fogParamsUniform;
    };

    // If the material had already compiled, force a rebuild with the new
    // defines. Cheap: it happens at most once per material.
    material.needsUpdate = true;
  }

  /** Walks a subtree and registers every lit material it finds. */
  refreshMaterials(root: THREE.Object3D): void {
    // The shadow-bias patch counts too: its uniform is declared globally by the
    // chunk, so a material that is never registered compiles with the uniform
    // left at zero, which silently switches the receiver-plane fit off for that
    // material alone — the hardest kind of bug to see in a screenshot.
    if (!this.csmActive && !this.fogActive && !this.shadowBiasActive) return;
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (!mat) return;
      if (Array.isArray(mat)) {
        for (const m of mat) this.registerMaterial(m);
      } else {
        this.registerMaterial(mat);
      }
    });
  }

  // ── atmosphere-derived colours ─────────────────────────────────────────────

  /**
   * Recomputes the key colour, the fill colours and the fog tint from the
   * scattering model. Only runs when the sun moves.
   */
  private refreshAtmosphere(): void {
    // Direct sun: solar irradiance filtered by the slant path through the air.
    sunTransmittance(this.sunDirection, this.sunColor);

    // Guard against a fully occluded sun (elevation clamped, but be safe) so we
    // never hand three a black key light and a pitch-black scene.
    const peak = Math.max(this.sunColor.r, this.sunColor.g, this.sunColor.b);
    if (peak < 1e-3) this.sunColor.setRGB(0.02, 0.02, 0.03, THREE.LinearSRGBColorSpace);

    for (const light of this.cascades) {
      light.color.copy(this.sunColor);
      light.intensity = SUN_IRRADIANCE;
    }

    // Sky samples. Cheap enough to evaluate on the CPU because it happens only
    // when the sun moves, not per frame.
    evaluateSky(_dirUp, this.sunDirection, this.zenithColor);

    // Low sky toward the sun and 180° away from it — these bracket the fog's
    // aerial-perspective gradient. Both are sampled well above the true horizon;
    // see FOG_SAMPLE_ELEVATION_DEG for why that is not a shortcut.
    const ySun = Math.tan(THREE.MathUtils.degToRad(FOG_SUN_SAMPLE_ELEVATION_DEG));
    const yOpp = Math.tan(THREE.MathUtils.degToRad(FOG_SAMPLE_ELEVATION_DEG));
    _dirHorizonSun.set(-this.sunDirection.x, 0, -this.sunDirection.z);
    if (_dirHorizonSun.lengthSq() < 1e-8) _dirHorizonSun.set(0, 0, 1);
    _dirHorizonSun.normalize().setY(ySun).normalize();
    evaluateSky(_dirHorizonSun, this.sunDirection, this.horizonSunColor);

    _dirHorizonOpp.set(-_dirHorizonSun.x, yOpp, -_dirHorizonSun.z).normalize();
    evaluateSky(_dirHorizonOpp, this.sunDirection, this.horizonColor);

    // ── the ambient budget ───────────────────────────────────────────────────
    // Measure what the sky actually delivers, then scale it to hit the chosen
    // sun-to-sky ratio. Solving rather than hard-coding is what keeps shadow
    // depth constant as the sun moves: at 40° elevation the direct term nearly
    // triples while the sky barely changes, and a fixed environment intensity
    // would make every shadow at noon three times deeper than at dusk.
    const cosSunOnGround = Math.max(0, -this.sunDirection.y);
    skyIrradiance(this.sunDirection, this.skyIrradianceUp);
    _skyIrradiance.copy(this.skyIrradianceUp);

    const directLum = luminance(this.sunColor) * SUN_IRRADIANCE * cosSunOnGround;
    const skyLum = Math.max(luminance(_skyIrradiance), 1e-5);
    this.directIrradiance = directLum;

    // Total ambient we are prepared to spend, in luminance units.
    const ambientLum = AMBIENT_TO_DIRECT * directLum;
    this.ambientIrradiance = ambientLum;
    // A sun below the horizon would otherwise divide the scene to black.
    this.shadowLitRatio = directLum > 1e-5 ? ambientLum / (ambientLum + directLum) : 1;

    // The environment map already has the right *colour*; it only needs the
    // right magnitude. Clamped so a degenerate sky can never blow up or vanish.
    this.environmentIntensity = THREE.MathUtils.clamp(
      (ambientLum * (1 - HEMI_SHARE)) / skyLum,
      0.01,
      8,
    );

    // Ground bounce radiance: Lambert sand lit by the transmitted sun *and* by
    // the sky it sits under — at golden hour the sky term is a fifth of the
    // total, which is exactly what stops the bounce reading as pure orange.
    //   L = albedo · E / π
    _color.setRGB(GROUND_ALBEDO[0], GROUND_ALBEDO[1], GROUND_ALBEDO[2], THREE.LinearSRGBColorSpace);
    this.groundBounceColor
      .copy(this.sunColor)
      .multiplyScalar(SUN_IRRADIANCE * cosSunOnGround)
      .add(_chroma.copy(_skyIrradiance).multiplyScalar(ambientLum / skyLum))
      .multiplyScalar(1 / Math.PI)
      .multiply(_color);

    // Hemisphere: cool sky from above, warm sand bounce from below. This is
    // what stops shadowed geometry reading as flat black holes, and — more
    // importantly at golden hour — what makes a shadow read as *cool* rather
    // than merely *dark*, which is the whole warm-key/cool-fill contrast the
    // hour is famous for.
    //
    // three uses `mix(groundColor, skyColor, w)` directly AS irradiance, so the
    // colours must already carry the magnitude. Both halves are scaled by the
    // same `intensity`, which is why the magnitude lives in the colours and
    // `intensity` stays pinned at 1 — scaling intensity to normalise the sky
    // half would silently blow the ground half up by the same factor.
    const hemiLum = ambientLum * HEMI_SHARE;
    this.applyChroma(this.hemisphere.color, _skyIrradiance, hemiLum);
    this.applyChroma(this.hemisphere.groundColor, this.groundBounceColor, hemiLum * HEMI_GROUND_BOOST);
    this.hemisphere.intensity = 1;

    // The weapon lives in the same world; light it with the same numbers. Its
    // fill runs richer than the world's: the viewmodel is a large, close,
    // mostly-vertical object filling a third of the frame, and at the world's
    // ratio its shadow side falls to near black and reads as a silhouette
    // pasted over the scene rather than an object inside it.
    this.viewKey.color.copy(this.sunColor);
    this.viewKey.intensity = SUN_IRRADIANCE * 0.9;
    this.applyChroma(this.viewFill.color, _skyIrradiance, hemiLum * VIEWMODEL_FILL_BOOST);
    this.applyChroma(
      this.viewFill.groundColor,
      this.groundBounceColor,
      hemiLum * HEMI_GROUND_BOOST * VIEWMODEL_FILL_BOOST,
    );
    this.viewFill.intensity = 1;

    // ── fog ──────────────────────────────────────────────────────────────────
    // Three colours, not one. Away from the sun the air is the cool low sky;
    // toward it, the warm forward-scattered aureole; looking up, the deep blue
    // zenith. The shader blends between them per fragment from the view ray, so
    // a wall 60 m away on the sun side hazes warm while the same wall on the
    // shadow side hazes cool — that separation is what aerial perspective
    // actually looks like, and a single constant can never produce it.
    // ...and all three are pulled toward the sand bounce by FOG_GROUND_SHARE,
    // most strongly for the near-horizontal directions where the ground really
    // does fill half the sky. Looking up, the ground contributes far less, so
    // the zenith blend keeps only a third of the share.
    this.fogSunColorUniform.value
      .copy(this.horizonSunColor)
      .lerp(this.groundBounceColor, FOG_GROUND_SHARE);
    this.fogUpColorUniform.value
      .copy(this.zenithColor)
      .lerp(this.horizonColor, 0.35)
      .lerp(this.groundBounceColor, FOG_GROUND_SHARE * 0.33);
    this.fogSunDirUniform.value.copy(this.sunDirection).negate();

    const ctx = this.ctx;
    if (ctx) {
      const params = this.fogParamsUniform.value;
      if (!(ctx.scene.fog instanceof THREE.FogExp2)) {
        ctx.scene.fog = new THREE.FogExp2(0x000000, params.x);
      }
      // Assigned by copy, never through getHex(): getHex clamps to [0,1] and
      // quantises to 8 bits, which silently crushes an HDR fog colour.
      //
      // This is BOTH the base tint of the height fog (mixed per fragment
      // against the sun-side and up-facing colours in the shader) AND the whole
      // colour used by the fallback branch, so it has to carry the same ground
      // share as the other two or unpatched surfaces haze a different colour
      // from their neighbours.
      ctx.scene.fog.color.copy(this.horizonColor).lerp(this.groundBounceColor, FOG_GROUND_SHARE);
      // NOT density * 0.9. The patched chunk's fallback branch now uses the
      // same 1 - exp(-rho*d) curve as the height fog (see CSMShader.ts), so the
      // two agree exactly at eye level when they share a density. The old 0.9
      // was compensating for a squared-distance curve that no longer exists.
      ctx.scene.fog.density = params.x;
    }

    this.atmosphereVersion++;
  }

  /**
   * Writes `source`'s chromaticity into `target` at a chosen Rec.709 luminance.
   *
   * Every ambient term in this file is specified as "the sky's colour, at N
   * units of brightness"; separating the two is what lets one budget scalar
   * drive four differently coloured lights without any of them drifting hue.
   */
  private applyChroma(target: THREE.Color, source: THREE.Color, targetLuminance: number): void {
    const l = luminance(source);
    if (l < 1e-6 || targetLuminance <= 0) {
      target.setRGB(0, 0, 0, THREE.LinearSRGBColorSpace);
      return;
    }
    target.copy(source).multiplyScalar(targetLuminance / l);
  }

  /**
   * Re-expresses the world sun in the viewmodel scene's space.
   *
   * The chain is world -> main camera view space -> viewmodel scene space. Going
   * through the main camera is what makes the weapon's key light rotate with
   * the player's head; going on through `viewCamera.matrixWorld` keeps it
   * correct even if the weapons system moves or rolls that camera for ADS, sway
   * or a lean.
   */
  private updateViewmodelRig(ctx: EngineContext): void {
    _vmSun.copy(this.sunDirection).transformDirection(ctx.camera.matrixWorldInverse);
    _vmSun.transformDirection(ctx.viewCamera.matrixWorld);

    // Light travels along +_vmSun, so the lamp must sit on the opposite side.
    this.viewKey.position.copy(ctx.viewCamera.position).addScaledVector(_vmSun, -4);
    this.viewKey.target.position.copy(ctx.viewCamera.position);
    this.viewKey.target.updateMatrixWorld();
  }

  // ── cascade fitting ────────────────────────────────────────────────────────

  private updateCascades(camera: THREE.PerspectiveCamera): void {
    const count = this.cascades.length;
    const near = Math.max(camera.near, SPLIT_MIN_NEAR);
    const far = Math.max(near + 1, Math.min(camera.far, DEFAULT_SHADOW_DISTANCE));

    this.computeSplits(near, far, count);

    // Light-space basis. Rotation only: `Matrix4.lookAt(eye, target, up)` builds
    // the same basis `Object3D.lookAt` gives the shadow camera, so snapping in
    // this space snaps in exactly the space the shadow map is rasterised in.
    _basis.identity().lookAt(_tmp.set(0, 0, 0), this.sunDirection, _up);
    _basisInv.copy(_basis).transpose();

    camera.getWorldDirection(_fwd);

    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
    const tanH = tanV * camera.aspect;
    const k2 = tanH * tanH + tanV * tanV;

    for (let i = 0; i < count; i++) {
      const light = this.cascades[i];
      if (!light.visible) continue;

      const n = this.splits[i];
      const f = this.splits[i + 1];

      // Bounding sphere of the frustum slice, on the view axis at z = zc.
      //   r² = (n·k)² + (zc − n)² = (f·k)² + (zc − f)²
      //   ⇒ zc = (n + f)(k² + 1) / 2
      let zc = (n + f) * (k2 + 1) * 0.5;
      let radius: number;
      if (zc >= f) {
        // Sphere would sit past the far plane: the far corners alone bound it.
        zc = f;
        radius = f * Math.sqrt(k2);
      } else {
        const dz = zc - f;
        radius = Math.sqrt(f * f * k2 + dz * dz);
      }
      // A hair of slack absorbs the up-to-one-texel offset introduced by the
      // snap below, so nothing can fall outside the map.
      radius *= 1.02;

      // Clamp to the level. See MAX_CASCADE_RADIUS — for a 122 degree
      // horizontal field the honest minimal enclosing sphere is several times
      // wider than the whole arena, and paying for that in texel size is what
      // made the outer cascades impossible to bias. When the clamp bites, the
      // centre moves from the far plane to the middle of the slice's on-axis
      // span, which is the placement that keeps the most of the slice inside a
      // now-smaller box; what falls outside is off-map desert.
      //
      // Both the clamp and the re-centring depend only on (n, f, fov, aspect),
      // never on camera orientation, so the anti-swim guarantee at the top of
      // this file is preserved exactly.
      if (radius > MAX_CASCADE_RADIUS) {
        radius = MAX_CASCADE_RADIUS;
        zc = (n + f) * 0.5;
      }

      const texel = (radius * 2) / this.shadowMapSize;

      _center.copy(camera.position).addScaledVector(_fwd, zc);
      // Snap to the shadow texel grid in light space — the anti-swim step.
      _center.applyMatrix4(_basisInv);
      _center.x = Math.floor(_center.x / texel) * texel;
      _center.y = Math.floor(_center.y / texel) * texel;
      _center.applyMatrix4(_basis);

      const back = radius + CASTER_EXTENT;
      light.position.copy(_center).addScaledVector(this.sunDirection, -back);
      light.target.position.copy(_center);
      light.target.updateMatrixWorld();

      const cam = light.shadow.camera;
      cam.left = -radius;
      cam.right = radius;
      cam.top = radius;
      cam.bottom = -radius;
      // Fitted to the cascade, not left at a generic 0.05. The light plane sits
      // `back` in front of the sphere centre, so the nearest thing that can
      // possibly be in this cascade's depth range is CASTER_EXTENT away and the
      // farthest is back + radius. A small margin in front absorbs the texel
      // snap; anything nearer than that is behind the light.
      cam.near = Math.max(0.05, CASTER_EXTENT * 0.5);
      cam.far = back + radius;
      cam.updateProjectionMatrix();
      const depthRange = cam.far - cam.near;

      // ── bias budget ────────────────────────────────────────────────────────
      // The slope term — historically the whole problem here — is no longer
      // paid for with bias at all. `shaders/CSMShader.ts` fits the receiver's
      // plane from screen-space derivatives of the shadow coordinate and offsets
      // every PCF tap along that plane, which cancels texel·tanθ exactly, for
      // any incidence angle, with no peter-panning. What is left for these two
      // constants is only the sub-texel residue, so both can be small.
      //
      // For the record, this is what they had to absorb before: acne-free with
      // a pure normal offset needs b >= texel · sinθ / cos²θ, and on flat ground
      // under a 17.5° sun (θ = 72.5°) that factor is 10.5 — times the PCF
      // kernel's reach of 1.5-3.3 texels on top. At 2.2-14.5 cm texels that is
      // 35 cm to 5 m of required offset. No value can satisfy that and still
      // keep a crate's shadow touching the crate; the previous 3.4 texels was
      // simply the least-bad point on an impossible trade, and the residual
      // 3-5x shortfall is exactly the speckle in the reference frames.
      light.shadow.normalBias = Math.min(texel * NORMAL_BIAS_TEXELS, MAX_NORMAL_BIAS);
      // Constant term, converted from metres of light-space depth into the
      // [0,1] NDC the comparison happens in. Negative = toward the light.
      light.shadow.bias = -CONST_DEPTH_BIAS / depthRange;
      // Penumbra widens with cascade index; combined with the growing world
      // texel size this gives sharp contact shadows and soft distant ones.
      // Slightly tighter than before because the receiver-plane bias now makes
      // wide kernels safe, so the width can be chosen for looks rather than as
      // a compromise against acne: 1.3/2.1/2.9 texels is 2 cm, 9 cm and 21 cm
      // of penumbra at this fit, which reads as a real distance falloff.
      light.shadow.radius = 1.3 + i * 0.8;

      // NDC depth gained per texel by a receiver tilted 45° to the light. The
      // shader multiplies this by a plain tan(θ) limit to clamp the plane fit;
      // keeping the unit conversion here means the shader never has to know
      // the cascade's metres-to-depth scale.
      this.shadowTexelDepth[i] = texel / depthRange;
    }
  }

  /** Practical (uniform/logarithmic blend) split scheme with lambda 0.6. */
  private computeSplits(near: number, far: number, count: number): void {
    this.splits.length = count + 1;
    this.splits[0] = near;
    this.splits[count] = far;
    const ratio = far / near;
    for (let i = 1; i < count; i++) {
      const t = i / count;
      const uniform = near + (far - near) * t;
      const logarithmic = near * Math.pow(ratio, t);
      this.splits[i] = THREE.MathUtils.lerp(uniform, logarithmic, SPLIT_LAMBDA);
    }

    if (!this.csmActive) {
      // Single cascade: cover everything, no fade windows needed.
      const w = this.fadeWindows[0];
      if (w) w.set(-2, -1, 1e6, 1e6 + 1e3);
      return;
    }

    // Blend bands. Cascade i's outgoing window is byte-identical to cascade
    // i+1's incoming window, which makes the two smoothsteps exact complements
    // and guarantees the weights sum to one across the transition.
    for (let i = 0; i < count; i++) {
      const w = this.fadeWindows[i];
      if (!w) continue;
      const lo = this.splits[i];
      const hi = this.splits[i + 1];

      if (i === 0) {
        w.x = -2;
        w.y = -1;
      } else {
        const prevLen = this.splits[i] - this.splits[i - 1];
        const h = THREE.MathUtils.clamp(prevLen * SPLIT_BLEND_FRACTION, 0.35, 7);
        w.x = lo - h;
        w.y = lo + h;
      }

      if (i === count - 1) {
        w.z = 1e6;
        w.w = 1e6 + 1e3;
      } else {
        const len = hi - lo;
        const h = THREE.MathUtils.clamp(len * SPLIT_BLEND_FRACTION, 0.35, 7);
        w.z = hi - h;
        w.w = hi + h;
      }
    }
  }

  // ── helpers ────────────────────────────────────────────────────────────────

  private isLitMaterial(material: THREE.Material): boolean {
    if (material.userData['nfNoPatch'] === true) return false;
    const flags = material as unknown as MaterialFlags;
    if (
      flags.isMeshStandardMaterial === true ||
      flags.isMeshPhysicalMaterial === true ||
      flags.isMeshPhongMaterial === true ||
      flags.isMeshLambertMaterial === true ||
      flags.isMeshToonMaterial === true
    ) {
      return true;
    }
    // A raw ShaderMaterial only pulls in lights_fragment_begin when it opts in.
    return flags.isShaderMaterial === true && flags.lights === true;
  }
}
