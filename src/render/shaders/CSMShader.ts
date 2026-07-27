/**
 * NOISEFIELD — global ShaderChunk patches for cascaded shadow maps,
 * receiver-plane shadow biasing, exponential height fog and specular
 * antialiasing.
 *
 * WHY PATCH THE CHUNKS AT ALL
 * ---------------------------
 * three renders one shadow frustum per DirectionalLight. Over a 100 m arena a
 * single 2048² frustum gives ~5 cm texels — blocky contact shadows and visible
 * swimming. The standard fix is N lights that share one direction, each
 * covering a slice of the view frustum, plus a shader-side selector that lets
 * exactly one slice contribute. That selector has to live inside
 * `lights_fragment_begin`, which is a built-in chunk, so it must be patched.
 *
 * The patch is *surgical*: we locate the directional-light section by two
 * stable marker strings and swap only that section. Everything else — point,
 * spot, rect-area, IBL — is left byte-identical to whatever three shipped. If
 * either marker is missing (three changed the chunk) we bail out and report
 * failure so the caller can fall back to a single non-cascaded shadow rather
 * than compiling a broken shader.
 *
 * CASCADE BLENDING
 * ----------------
 * Each cascade gets a vec4 fade window: (nearStart, nearEnd, farStart, farEnd)
 * in metres of view depth. Weight is
 *
 *     w_i = smoothstep(nearStart, nearEnd, z) * (1 - smoothstep(farStart, farEnd, z))
 *
 * Cascade i's (farStart, farEnd) is identical to cascade i+1's (nearStart,
 * nearEnd), so the two smoothsteps are exact complements: the weights sum to
 * one everywhere and the transition is invisible. A hard `if (z < split)`
 * selector — which is what most three CSM implementations do — leaves a
 * visible resolution seam sweeping across the ground as the player walks.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY THE SHADOW FILTER IS ALSO REPLACED — the acne story
 * ─────────────────────────────────────────────────────────────────────────────
 * three r185's PCF `getShadow` takes five Vogel-disk taps at
 * `shadowRadius` texels, and rotates the disk by an interleaved-gradient-noise
 * angle that is DIFFERENT FOR EVERY PIXEL. All five taps are compared against
 * the *fragment's own* depth.
 *
 * That is only correct when the receiver is perpendicular to the light. On a
 * receiver tilted by θ the true depth at a tap r texels away differs from the
 * fragment's by
 *
 *     Δz = r · texelWorldSize · tan θ
 *
 * At golden hour the dominant receiver — flat ground under a 17.5° sun — sits
 * at θ = 72.5°, so tan θ = 3.17, and with a 2-3 texel kernel the comparison is
 * wrong by three to ten texel-widths of depth. Neither a constant bias nor a
 * normal offset can absorb that: sizing them to cover it detaches every contact
 * shadow, and sizing them to keep contact shadows makes half the taps
 * self-occlude. Because the disk rotation is randomised per pixel, the half
 * that self-occludes changes from pixel to pixel — which is why the artefact
 * reads as dense monochrome NOISE rather than as the textbook striped moiré,
 * and why it survives every "increase the bias" attempt.
 *
 * The fix is a RECEIVER-PLANE DEPTH BIAS (Isidoro, GDC 2006): reconstruct
 * ∂z/∂u and ∂z/∂v of the shadow coordinate from screen-space derivatives, and
 * offset each tap's comparison depth by the receiver plane's own depth at that
 * tap. The slope term then cancels exactly, at any incidence angle and any
 * cascade resolution, and — crucially — it introduces NO peter-panning,
 * because the offset moves the comparison along the receiver's own surface
 * rather than away from it.
 *
 * Two caveats, both handled below:
 *   • dFdx/dFdy are undefined in non-uniform control flow, so the plane fit is
 *     computed before the cascade-weight branch, never inside it.
 *   • at a silhouette the two derivatives straddle two different surfaces and
 *     the fitted plane is meaningless, so the slope is clamped to a physically
 *     motivated maximum incidence (see NF_RPDB_MAX_TAN).
 *
 * NOTE: never put a backtick inside these template literals.
 */

import * as THREE from 'three';

/** Marker that opens the directional-light section of lights_fragment_begin. */
const DIR_SECTION_START = '#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )';
/** Marker that opens the section immediately after it. */
const DIR_SECTION_END = '#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )';

/** Guard in `worldpos_vertex` that decides whether `worldPosition` is computed. */
const WORLDPOS_GUARD =
  '#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0';

/** Per-material define that switches the CSM path on. Value = cascade count. */
export const CSM_DEFINE = 'NF_CSM_CASCADES';
/** Per-material define that switches the height-fog path on. */
export const FOG_DEFINE = 'NF_FOG';

/** Uniform names injected by `Lighting` / `Sky` through `onBeforeCompile`. */
export const CSM_UNIFORM_FADE = 'nfCsmFade';
export const FOG_UNIFORM_SUN_COLOR = 'nfFogSunColor';
export const FOG_UNIFORM_UP_COLOR = 'nfFogUpColor';
export const FOG_UNIFORM_SUN_DIR = 'nfFogSunDir';
export const FOG_UNIFORM_PARAMS = 'nfFogParams';
/**
 * Per-shadow scalar: the NDC depth a receiver gains over ONE shadow texel of
 * lateral movement when it is tilted 45° to the light, i.e.
 * `texelWorldSize / (shadowCamera.far - shadowCamera.near)`.
 *
 * It is the natural unit for everything the shadow filter has to reason about:
 * the receiver-plane slope clamp is expressed as a multiple of it (a plain
 * `tan θ`), and it stays correct as the cascade fit, the map resolution or the
 * depth range change, which a hard-coded epsilon never does.
 */
export const SHADOW_UNIFORM_TEXEL_DEPTH = 'nfShadowTexelDepth';

// ─────────────────────────────────────────────────────────────────────────────
// GLSL fragments
// ─────────────────────────────────────────────────────────────────────────────

const CSM_PARS = /* glsl */ `
#if defined( ${CSM_DEFINE} )

	uniform vec4 ${CSM_UNIFORM_FADE}[ ${CSM_DEFINE} ];

	// (nearStart, nearEnd, farStart, farEnd) in metres of view depth.
	float nfCascadeWeight( const in float viewDepth, const in vec4 fade ) {

		return smoothstep( fade.x, fade.y, viewDepth ) * ( 1.0 - smoothstep( fade.z, fade.w, viewDepth ) );

	}

#endif
`;

/**
 * Receiver-plane-biased PCF, appended to `shadowmap_pars_fragment`.
 *
 * Guarded on SHADOWMAP_TYPE_PCF because `sampler2DShadow` only exists on that
 * path; if anything ever switches the shadow type, the call site falls back to
 * three's stock `getShadow` and the renderer degrades rather than failing to
 * compile.
 */
const SHADOW_PARS = /* glsl */ `
#if defined( USE_SHADOWMAP ) && defined( SHADOWMAP_TYPE_PCF ) && NUM_DIR_LIGHT_SHADOWS > 0

	uniform float ${SHADOW_UNIFORM_TEXEL_DEPTH}[ NUM_DIR_LIGHT_SHADOWS ];

	// Taps per shadow lookup. Each is a hardware 2x2 comparison fetch, so eight
	// Vogel samples are ~32 filtered taps. Five (three's stock count) leaves
	// visible dither in a penumbra once the kernel is wide enough to matter.
	#define NF_PCF_TAPS 8

	// Largest receiver tilt the plane fit is trusted for, as tan(theta).
	// 14 is theta = 85.9 degrees: a surface steeper than that relative to the
	// light receives essentially no direct light, so clamping there costs
	// nothing visible, while an unclamped fit at a silhouette — where the two
	// derivatives straddle two different surfaces — would extrapolate a wild
	// plane and punch a lit hole through the middle of a real shadow.
	#define NF_RPDB_MAX_TAN 14.0

	// Safety margin on top of the exact plane compensation, in texels.
	// The plane fit cancels the slope between the FRAGMENT and each TAP, but a
	// shadow texel stores the depth rasterised at its own centre, which is up
	// to half a texel diagonal (0.71 texels) away from wherever the tap landed.
	// 0.85 covers that with a little slack and is the only remaining term that
	// can peter-pan; at cascade 0 it is ~4 cm of depth, i.e. a ~4 cm gap at the
	// foot of a crate under a 17.5-degree sun, which the contact AO conceals.
	#define NF_SLOPE_BIAS_TEXELS 0.85

	/**
	 * Depth gradient of the receiver plane, expressed as NDC depth per shadow
	 * texel along u and v.
	 *
	 * Solves the 2x2 system
	 *     [ du.x  dv.x ] [ dz/du ]   [ du.z ]
	 *     [ du.y  dv.y ] [ dz/dv ] = [ dv.z ]
	 * where du = dFdx(shadowCoord), dv = dFdy(shadowCoord). MUST be called from
	 * uniform control flow — derivatives inside a branch are undefined.
	 */
	vec2 nfReceiverPlaneSlope( const in vec3 coord, const in vec2 texelSize, const in float maxSlope ) {

		vec3 du = dFdx( coord );
		vec3 dv = dFdy( coord );

		float det = du.x * dv.y - dv.x * du.y;

		vec2 dzduv = vec2( 0.0 );

		// A degenerate determinant means the shadow UV is stationary across the
		// pixel (an exactly light-facing receiver); the slope is genuinely zero.
		if ( abs( det ) > 1e-12 ) {

			dzduv.x = ( dv.y * du.z - du.y * dv.z ) / det;
			dzduv.y = ( du.x * dv.z - dv.x * du.z ) / det;

		}

		// Per unit UV -> per texel, then clamp to a believable incidence.
		return clamp( dzduv * texelSize, vec2( - maxSlope ), vec2( maxSlope ) );

	}

	float nfGetShadowRPDB( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity,
		float shadowBias, float shadowRadius, vec3 coord, vec2 dzPerTexel ) {

		float shadow = 1.0;

		// Slope-scaled depth bias taken straight from the fitted plane. Note
		// that |dz/du| + |dz/dv| IS texel * tan(theta) in depth units, so this
		// is an exact tan-of-incidence bias with no trigonometry and no
		// dependence on a normal that may have been perturbed by a normal map.
		float slopeBias = ( abs( dzPerTexel.x ) + abs( dzPerTexel.y ) ) * NF_SLOPE_BIAS_TEXELS;
		float z = coord.z + shadowBias - slopeBias;

		bool inFrustum = coord.x >= 0.0 && coord.x <= 1.0 && coord.y >= 0.0 && coord.y <= 1.0;

		if ( inFrustum && z <= 1.0 ) {

			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;

			float sum = 0.0;

			for ( int s = 0; s < NF_PCF_TAPS; s ++ ) {

				// Offset in TEXELS, so the same vector drives both the UV step
				// and the receiver-plane depth correction.
				vec2 offset = vogelDiskSample( s, NF_PCF_TAPS, phi ) * shadowRadius;
				float zt = clamp( z + dot( offset, dzPerTexel ), 0.0, 1.0 );
				sum += texture( shadowMap, vec3( coord.xy + offset * texelSize, zt ) );

			}

			shadow = sum / float( NF_PCF_TAPS );

		}

		return mix( 1.0, shadow, shadowIntensity );

	}

#endif
`;

/**
 * Replacement for the directional-light section. Semantics outside the CSM
 * define are identical to stock three.
 */
const CSM_DIRECTIONAL_SECTION = /* glsl */ `#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )

	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif

	#if defined( ${CSM_DEFINE} )
	// Declared outside the loop: three unrolls the body by plain concatenation,
	// so a declaration inside would be emitted once per cascade in one scope.
	float nfCsmDepth = vViewPosition.z;
	float nfCsmW;
	#endif

	#if defined( USE_SHADOWMAP ) && defined( SHADOWMAP_TYPE_PCF ) && NUM_DIR_LIGHT_SHADOWS > 0
	vec3 nfShadowNdc;
	vec2 nfShadowSlope;
	#endif

	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {

		directionalLight = directionalLights[ i ];

		getDirectionalLightInfo( directionalLight, directLight );

	#if defined( USE_SHADOWMAP ) && defined( SHADOWMAP_TYPE_PCF ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )

		// HOISTED ON PURPOSE. nfReceiverPlaneSlope takes screen-space
		// derivatives, which are undefined inside the non-uniform
		// "if ( nfCsmW > 0.0 )" below. Computing it here — in the unrolled,
		// unconditional part of the loop body — is the only correct place.
		nfShadowNdc = vDirectionalShadowCoord[ i ].xyz / vDirectionalShadowCoord[ i ].w;
		nfShadowSlope = nfReceiverPlaneSlope(
			nfShadowNdc,
			vec2( 1.0 ) / directionalLightShadows[ i ].shadowMapSize,
			${SHADOW_UNIFORM_TEXEL_DEPTH}[ i ] * NF_RPDB_MAX_TAN
		);

	#endif

	#if defined( ${CSM_DEFINE} ) && ( UNROLLED_LOOP_INDEX < ${CSM_DEFINE} )

		nfCsmW = nfCascadeWeight( nfCsmDepth, ${CSM_UNIFORM_FADE}[ UNROLLED_LOOP_INDEX ] );

		if ( nfCsmW > 0.0 ) {

			#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
			directionalLightShadow = directionalLightShadows[ i ];
			#if defined( SHADOWMAP_TYPE_PCF )
			directLight.color *= ( directLight.visible && receiveShadow ) ? nfGetShadowRPDB( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, nfShadowNdc, nfShadowSlope ) : 1.0;
			#else
			directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
			#endif
			#endif

			// RE_Direct is linear in directLight.color, so scaling here gives a
			// mathematically exact cross-fade of the shaded result.
			directLight.color *= nfCsmW;

			RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );

		}

	#else

		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		#if defined( SHADOWMAP_TYPE_PCF )
		directLight.color *= ( directLight.visible && receiveShadow ) ? nfGetShadowRPDB( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, nfShadowNdc, nfShadowSlope ) : 1.0;
		#else
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		#endif

		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );

	#endif

	}
	#pragma unroll_loop_end

#endif

`;

const FOG_PARS_VERTEX = /* glsl */ `
#ifdef USE_FOG

	varying float vFogDepth;

	#ifdef ${FOG_DEFINE}
	varying vec3 vNfFogWorld;
	#endif

#endif
`;

const FOG_VERTEX = /* glsl */ `
#ifdef USE_FOG

	vFogDepth = - mvPosition.z;

	#ifdef ${FOG_DEFINE}
	// worldPosition is guaranteed by the worldpos_vertex guard patch below.
	vNfFogWorld = worldPosition.xyz;
	#endif

#endif
`;

const FOG_PARS_FRAGMENT = /* glsl */ `
#ifdef USE_FOG

	uniform vec3 fogColor;
	varying float vFogDepth;

	#ifdef FOG_EXP2

		uniform float fogDensity;

	#else

		uniform float fogNear;
		uniform float fogFar;

	#endif

	#ifdef ${FOG_DEFINE}

		varying vec3 vNfFogWorld;

		uniform vec3  ${FOG_UNIFORM_SUN_COLOR}; // warm low sky, sun side
		uniform vec3  ${FOG_UNIFORM_UP_COLOR};  // cool high sky, looking up
		uniform vec3  ${FOG_UNIFORM_SUN_DIR};   // unit, pointing TOWARD the sun
		// x = density, y = height falloff (1/m), z = base height (m),
		// w = in-scatter strength toward the sun.
		uniform vec4  ${FOG_UNIFORM_PARAMS};

		/**
		 * Closed-form integral of an exponentially decaying height density along
		 * the eye ray:
		 *
		 *     tau = d0 * exp(-k*(yEye - y0)) * L * (1 - exp(-k*dy)) / (k*dy)
		 *
		 * Doing it analytically rather than as a per-fragment march is what makes
		 * height fog affordable on every single material in the scene.
		 */
		float nfHeightFogTau( const in vec3 worldPos, const in vec3 camPos ) {

			vec3 delta = worldPos - camPos;
			float dist = length( delta );
			float dy = delta.y;
			float k = max( ${FOG_UNIFORM_PARAMS}.y, 1e-5 );

			// Clamped so standing below the fog base cannot explode the exponent.
			float base = ${FOG_UNIFORM_PARAMS}.x * exp( - k * clamp( camPos.y - ${FOG_UNIFORM_PARAMS}.z, -40.0, 400.0 ) );

			float kdy = k * dy;
			float ramp = abs( kdy ) > 1e-4 ? ( 1.0 - exp( - kdy ) ) / kdy : 1.0;

			return max( base * dist * ramp, 0.0 );

		}

	#endif

#endif
`;

const FOG_FRAGMENT = /* glsl */ `
#ifdef USE_FOG

	#ifdef ${FOG_DEFINE}

		float fogFactor = 1.0 - exp( - nfHeightFogTau( vNfFogWorld, cameraPosition ) );

		// ── aerial perspective ────────────────────────────────────────────────
		// The colour of the air is the colour of the sky behind it, and that is
		// emphatically NOT one constant: air lit from behind by a low sun is far
		// brighter and warmer than the same air seen at 90 degrees to it, and air
		// seen against the zenith is bluer than air seen against the low sky.
		// Collapsing all three into a single fog colour is what makes distant
		// geometry recede into a flat gauze that matches nothing on screen.
		vec3 nfFogView = normalize( vNfFogWorld - cameraPosition );
		float nfMu = max( dot( nfFogView, ${FOG_UNIFORM_SUN_DIR} ), 0.0 );

		// Two lobes standing in for a Henyey-Greenstein phase: a broad forward
		// bias that warms a good third of the sky around the sun, plus a tight
		// aureole that only fires within ~20 degrees of the disc. The single
		// pow(mu, 6) this replaces was too narrow to be seen anywhere except
		// staring straight down the sun vector.
		float nfBroad = nfMu * nfMu;
		float nfTight = pow( nfMu, 11.0 );
		float nfSunAmount = clamp( nfBroad * 0.42 + nfTight * 0.72, 0.0, 1.0 );
		nfSunAmount = clamp( nfSunAmount * ${FOG_UNIFORM_PARAMS}.w, 0.0, 1.0 );

		// Cool the haze as the ray tilts upward. Matters most on rooftops and on
		// anything silhouetted against open sky.
		float nfUpAmount = smoothstep( 0.0, 0.55, nfFogView.y );

		vec3 nfFogTint = mix( fogColor, ${FOG_UNIFORM_UP_COLOR}, nfUpAmount );
		nfFogTint = mix( nfFogTint, ${FOG_UNIFORM_SUN_COLOR}, nfSunAmount );

		gl_FragColor.rgb = mix( gl_FragColor.rgb, nfFogTint, clamp( fogFactor, 0.0, 1.0 ) );

	#else

		#ifdef FOG_EXP2

			// DELIBERATELY NOT three's exp(-(rho*d)^2).
			//
			// This branch runs for every material that never received the
			// NF_FOG define — unlit helpers, sprites, anything created and
			// drawn inside a single frame. three's squared-distance curve and
			// the analytic height integral above disagree violently at short
			// range: with rho matched at 60 m the squared form is only a fifth
			// as strong at 20 m. Two neighbouring surfaces fogged by two
			// different curves is exactly the "some geometry is hazed
			// differently" seam this renderer had. Using the same
			// 1 - exp(-rho*d) shape as the height fog makes the two agree for
			// any horizontal ray, and Lighting.setFog feeds this branch the
			// height fog's own base density so they agree in magnitude too.
			float fogFactor = 1.0 - exp( - fogDensity * vFogDepth );

		#else

			float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );

		#endif

		gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, clamp( fogFactor, 0.0, 1.0 ) );

	#endif

#endif
`;

/**
 * Marker inside `lights_physical_fragment` where three folds its
 * geometry-curvature roughness in. Replaced to add normal-map specular
 * antialiasing.
 */
const ROUGHNESS_MARKER = 'material.roughness += geometryRoughness;';

/**
 * Geometric specular antialiasing (Tokuyoshi & Kaplanyan 2019).
 *
 * three's own `geometryRoughness` is derived from `nonPerturbedNormal`, i.e.
 * the interpolated GEOMETRIC normal, so it is identically zero on a flat plane
 * and sees nothing the normal map does. That is precisely the aliasing case:
 * flat sand, a sharp procedural normal map, a grazing view and a low sun pack
 * many normal-map periods into one pixel, and a roughness that dips to 0.30 on
 * the damp patches turns each of them into an isolated white firefly — the
 * "wet glitter" on the courtyard floor.
 *
 * Estimating the pixel-footprint variance of the PERTURBED normal and adding it
 * to alpha (= roughness²) is the standard fix: variance sums in alpha, so this
 * is the correct convolution of the NDF with the pixel filter rather than a
 * fudge, it costs two derivatives, and it is exactly self-cancelling — it does
 * nothing at all where the normal is locally smooth.
 */
const ROUGHNESS_SPECULAR_AA = /* glsl */ `material.roughness += geometryRoughness;

// ── specular antialiasing (see ROUGHNESS_SPECULAR_AA in CSMShader.ts) ────────
{
	vec3 nfNdx = dFdx( normal );
	vec3 nfNdy = dFdy( normal );
	// 0.25 is Tokuyoshi's screen-space variance for a box pixel filter; the
	// factor 2 converts a normal variance into the equivalent NDF widening.
	float nfNormalVariance = 0.25 * ( dot( nfNdx, nfNdx ) + dot( nfNdy, nfNdy ) );
	// 0.18 caps how much roughness a single pixel may invent. Without it a
	// silhouette pixel — where the normal flips across the whole hemisphere in
	// one step — would be pushed to fully rough and read as a bright matte
	// fringe tracing every edge in the frame.
	float nfKernelRoughness = min( 2.0 * nfNormalVariance, 0.18 );
	material.roughness = sqrt( saturate( pow2( material.roughness ) + nfKernelRoughness ) );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Installation
// ─────────────────────────────────────────────────────────────────────────────

interface PatchState {
  csm: boolean;
  fog: boolean;
  /** Receiver-plane-biased PCF installed (the acne fix). */
  shadowBias: boolean;
  /** Normal-map specular antialiasing installed. */
  specularAA: boolean;
}

let installed: PatchState | null = null;

/**
 * Patches the global ShaderChunk table. Idempotent, and safe to call before any
 * material has been compiled (which is why `Lighting.init` runs it first).
 *
 * @returns which patches actually applied, so callers can degrade gracefully.
 */
export function installShaderChunkPatches(): PatchState {
  if (installed !== null) return installed;

  const state: PatchState = { csm: false, fog: false, shadowBias: false, specularAA: false };

  // ── receiver-plane shadow bias ─────────────────────────────────────────────
  // Installed BEFORE the CSM patch because the replacement directional section
  // calls nfGetShadowRPDB, which is declared here. (Order does not actually
  // matter to the compiler — both chunks are assembled per program — but the
  // dependency is real and worth making obvious.)
  try {
    THREE.ShaderChunk.shadowmap_pars_fragment += SHADOW_PARS;
    state.shadowBias = true;
  } catch (err) {
    console.error('[Lighting] Receiver-plane shadow bias patch failed:', err);
  }

  // ── CSM ────────────────────────────────────────────────────────────────────
  try {
    const src = THREE.ShaderChunk.lights_fragment_begin;
    const a = src.indexOf(DIR_SECTION_START);
    const b = src.indexOf(DIR_SECTION_END);
    if (a >= 0 && b > a && state.shadowBias) {
      THREE.ShaderChunk.lights_fragment_begin = src.slice(0, a) + CSM_DIRECTIONAL_SECTION + src.slice(b);
      THREE.ShaderChunk.lights_pars_begin = CSM_PARS + THREE.ShaderChunk.lights_pars_begin;
      state.csm = true;
    } else {
      console.warn(
        '[Lighting] Could not locate the directional-light section of lights_fragment_begin; ' +
          'falling back to a single non-cascaded shadow frustum.',
      );
    }
  } catch (err) {
    console.error('[Lighting] CSM shader patch failed:', err);
  }

  // ── specular antialiasing ──────────────────────────────────────────────────
  try {
    const src = THREE.ShaderChunk.lights_physical_fragment;
    const at = src.indexOf(ROUGHNESS_MARKER);
    if (at >= 0) {
      THREE.ShaderChunk.lights_physical_fragment =
        src.slice(0, at) + ROUGHNESS_SPECULAR_AA + src.slice(at + ROUGHNESS_MARKER.length);
      state.specularAA = true;
    } else {
      console.warn(
        '[Lighting] Could not locate the roughness marker in lights_physical_fragment; ' +
          'specular aliasing will not be filtered.',
      );
    }
  } catch (err) {
    console.error('[Lighting] Specular antialiasing patch failed:', err);
  }

  // ── height fog ─────────────────────────────────────────────────────────────
  try {
    const wp = THREE.ShaderChunk.worldpos_vertex;
    if (wp.indexOf(WORLDPOS_GUARD) >= 0) {
      // Force `worldPosition` to exist whenever height fog is enabled; without
      // shadows or an env map three would otherwise skip computing it.
      THREE.ShaderChunk.worldpos_vertex = wp.replace(
        WORLDPOS_GUARD,
        `#if defined( ${FOG_DEFINE} ) || ${WORLDPOS_GUARD.slice('#if '.length)}`,
      );
      THREE.ShaderChunk.fog_pars_vertex = FOG_PARS_VERTEX;
      THREE.ShaderChunk.fog_vertex = FOG_VERTEX;
      THREE.ShaderChunk.fog_pars_fragment = FOG_PARS_FRAGMENT;
      THREE.ShaderChunk.fog_fragment = FOG_FRAGMENT;
      state.fog = true;
    } else {
      console.warn('[Lighting] worldpos_vertex guard not found; using stock exponential fog.');
    }
  } catch (err) {
    console.error('[Lighting] Height-fog shader patch failed:', err);
  }

  installed = state;
  return state;
}

/** Test hook: reports whether the chunks have been patched in this session. */
export function shaderPatchState(): Readonly<PatchState> | null {
  return installed;
}
