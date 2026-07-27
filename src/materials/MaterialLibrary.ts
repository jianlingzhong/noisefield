/**
 * NOISEFIELD — procedural material library.
 *
 * Owns one shared `THREE.Material` per surface id.
 *
 * Sharing matters for three reasons, none of which is automatic batching —
 * three.js issues one draw call per object regardless. It (a) lets the renderer
 * skip the program/uniform/texture rebind between consecutive objects, which is
 * what the render-list sort is for, (b) is a precondition for folding objects
 * into an `InstancedMesh` or `BatchedMesh`, and (c) keeps one copy of each
 * texture in VRAM instead of one per wall.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * CONSUMER QUICK REFERENCE
 * ─────────────────────────────────────────────────────────────────────────────
 *   const lib = new MaterialLibrary();
 *   engine.add(lib);                       // init() bakes everything
 *   await lib.whenReady();                 // resolves once baking is done
 *
 *   mesh.material = lib.get('steel_painted');
 *   lib.tag(mesh, 'metal');                // so surfaceOf() can resolve it
 *
 * UV RULE — materials are SHARED, so `texture.repeat` must stay at (1,1) and
 * tiling has to live in the geometry. Author UVs as `worldPosition / tileMeters`
 * (equivalently `worldPosition * lib.uvScale(id)`). Using the wrong tile size is
 * the single most common integration mistake, so the value is published three
 * ways: `lib.tileMeters(id)`, `lib.descriptor(id).tileMeters`, and — for code
 * that only holds a `THREE.Material` — `material.userData.tileMeters`.
 *
 * `IMaterialLibrary` only declares get/forSurface/surfaceOf/tag, and there are
 * more material ids (17) than there are `SurfaceType` values (11), so
 * `forSurface()` cannot reach `plaster`, `brick`, `cinderblock`, `asphalt`,
 * `steel_painted` or `glass_broken`. Ask for those by id via `get()`, and use
 * `ids()` to enumerate. `material.userData` carries `{ id, surface, tileMeters,
 * triplanar }` for any material this library produced.
 *
 * Do NOT `clone()` a material from here: three's clone resets `defines` and
 * copies neither `onBeforeCompile` nor `customProgramCacheKey`, so the clone
 * silently loses the detail/macro/triplanar patch. Use `variant()` instead.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHAT MAKES THIS LOOK LIKE A SURFACE AND NOT A TILED TEXTURE
 * ─────────────────────────────────────────────────────────────────────────────
 * Three things, all applied through a single `onBeforeCompile` patch:
 *
 *  1. DETAIL NORMAL at ~8-16x UV rate, blended with Reoriented Normal Mapping.
 *     A 1024^2 map covering 2.5 m has ~2.4 mm texels; at 30 cm from a wall the
 *     player is well inside that, and without a detail layer the surface goes
 *     smooth and plastic exactly when they are looking hardest.
 *
 *  2. MACRO VARIATION at ~1/8 UV rate, multiplying albedo and biasing
 *     roughness. The base tile repeats every 2.5 m; the macro field repeats
 *     every 20 m, so the two never line up inside a single view and the
 *     repetition stops being legible.
 *
 *  3. TRIPLANAR projection for ground materials, where mesh UVs are unreliable
 *     and world-space projection is strictly better.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * TEXTURE PACKING
 * ─────────────────────────────────────────────────────────────────────────────
 * Occlusion, roughness and metalness share ONE texture (the glTF ORM
 * convention). three.js reads `.r` for aoMap, `.g` for roughnessMap and `.b`
 * for metalnessMap, so assigning the same texture to all three slots is both
 * correct and saves two thirds of the memory and sampler bandwidth.
 */

import * as THREE from 'three';

import type {
  EngineContext,
  FrameContext,
  IMaterialLibrary,
  MaterialDescriptor,
  PBRTextureSet,
  QualityTier,
  SurfaceType,
} from '../core/Types';
import { PASS_ALBEDO, PASS_ORM, TextureForge } from './TextureForge';
import { DETAIL_NORMAL_GLSL, MACRO_VARIATION_GLSL } from './surfaces/detail';
import {
  ALL_RECIPES,
  DEFAULT_TUNING,
  SURFACE_DEFAULT_ID,
  type RecipeTuning,
  type SurfaceRecipe,
} from './surfaces/index';

/** Every legal `SurfaceType`, for validating whatever is in `userData`. */
const SURFACE_TYPES: ReadonlySet<string> = new Set<SurfaceType>([
  'concrete', 'metal', 'wood', 'sand', 'dirt', 'glass', 'water', 'flesh', 'foliage', 'fabric', 'rubber',
]);

/** Soft ceiling; we log loudly if a tier's bake would exceed it. */
const VRAM_BUDGET_BYTES = 256 * 1024 * 1024;

type PatchableMaterial = THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial;

interface Patch {
  detailScale: { value: number };
  detailStrength: { value: number };
  detailScale2: { value: number };
  detailStrength2: { value: number };
  detailAlbedo: { value: number };
  macroScale: { value: number };
  macroStrength: { value: number };
  macroRough: { value: number };
  warp: { value: THREE.Vector2 };
  /** x = grime band height in metres, y = grime strength, z = top-dust strength. */
  weather: { value: THREE.Vector3 };
  detailOffset: { value: THREE.Vector2 };
  triScale: { value: number };
}

/**
 * Slope applied when deriving the shared detail normal from its height field.
 *
 * This number is the difference between a detail layer that works and one that
 * is invisible past arm's length, so it gets a named constant and a paragraph.
 *
 * `bakeNormalFromHeight` builds `normalize( vec3( -dh/du * S, -dh/dv * S, 1 ) )`.
 * The 1/f detail field in `detail.ts` has |grad| ~ 6 after the Sobel's implicit
 * low-pass, so S = 0.18 puts the steepest micro-facets at atan(1.08) ~ 47
 * degrees and the typical facet nearer 25. That is aggressive but not
 * saturated, which is the whole point: the previous value of 0.85 drove every
 * texel past 85 degrees, and a field of near-horizontal normals pointing in
 * random directions averages to flat in the very first mip. Detail that only
 * exists at mip 0 is detail the player never sees.
 */
const NF_DETAIL_SLOPE = 0.18;

// ─────────────────────────────────────────────────────────────────────────────
// GLSL injected into every patched material
// ─────────────────────────────────────────────────────────────────────────────

const PATCH_FRAG_PARS = /* glsl */ `
#ifdef NF_DETAIL
  uniform sampler2D nfDetailNormal;
  uniform float nfDetailScale;
  uniform float nfDetailStrength;
  uniform vec2  nfDetailOffset;
  #ifdef NF_DETAIL2
    uniform float nfDetailScale2;
    uniform float nfDetailStrength2;
  #endif
  #ifdef NF_DETAILALB
    uniform float nfDetailAlbedo;
  #endif
#endif
#ifdef NF_MACRO
  uniform sampler2D nfMacro;
  uniform float nfMacroScale;
  uniform float nfMacroStrength;
  uniform float nfMacroRough;
#endif
#ifdef NF_UVWARP
  // x = displacement in base-UV units, y = UV multiplier for the warp lookup.
  uniform vec2 nfWarp;
#endif
#ifdef NF_WEATHER
  // x = grime band height (m), y = grime strength, z = top-dust strength.
  uniform vec3 nfWeather;
#endif
#ifdef NF_TRIPLANAR
  uniform float nfTriScale;
#endif
#ifdef NF_WORLDSPACE
  varying vec3 vNfWPos;
  varying vec3 vNfWNrm;
#endif

#ifdef NF_TRIPLANAR
  /**
   * Projection weights. Raising |n| to a high power narrows the band where two
   * planar projections cross-fade; too low and everything looks smeared, too
   * high and the seam becomes a visible line. 6 is the usual sweet spot.
   */
  vec3 nfTriWeights( vec3 n ) {
    vec3 w = pow( abs( n ), vec3( 6.0 ) );
    return w / max( w.x + w.y + w.z, 1e-4 );
  }

  // Per-axis UVs. The sign flip makes opposite faces of a box agree on
  // handedness; EVERY map must use the same construction, or the albedo ends up
  // mirrored relative to the normal on half the faces.
  vec2 nfTriUvX( vec3 wp, vec3 sgn, float s ) { return wp.zy * s * vec2(  sgn.x, 1.0 ); }
  vec2 nfTriUvY( vec3 wp, vec3 sgn, float s ) { return wp.xz * s * vec2(  sgn.y, 1.0 ); }
  vec2 nfTriUvZ( vec3 wp, vec3 sgn, float s ) { return wp.xy * s * vec2( -sgn.z, 1.0 ); }
#endif

#if defined( NF_DETAIL ) || defined( NF_TRIPLANAR )
  /**
   * Reoriented Normal Mapping (Barre-Brisebois & Hill, 2012).
   *
   * Builds the rotation that maps +Z onto the base normal and applies it to
   * the detail normal, so detail is expressed IN the base normal's frame. A
   * naive add/lerp flattens the base map wherever the detail is strong; RNM
   * preserves the base shape exactly and only perturbs it.
   *
   * The canonical form is t * dot(t,u)/t.z - u; multiplying through by t.z
   * (always positive, since t.z = base.z + 1 > 0) avoids the divide and is
   * identical after normalisation.
   */
  vec3 nfRNM( vec3 base, vec3 detail ) {
    vec3 t = base + vec3( 0.0, 0.0, 1.0 );
    vec3 u = detail * vec3( -1.0, -1.0, 1.0 );
    return normalize( t * dot( t, u ) - u * t.z );
  }
#endif

#ifdef NF_DETAIL
  /**
   * Fetch and apply BOTH detail taps to a base tangent-space normal.
   *
   * Two taps five-ish times apart, not one. A single tap only supplies relief
   * in one octave band, so it fixes the surface at exactly one viewing
   * distance: crank it for the 30 cm case and it is mip-averaged into nothing
   * at 3 m; tune it for 3 m and the player's face goes through a smooth plane.
   * The coarse tap also has a period unrelated to the base tile's, so it
   * perturbs the base map's own repeat for free.
   */
  vec3 nfApplyDetail( vec3 baseN, vec2 uv ) {
    vec3 d = texture2D( nfDetailNormal, uv * nfDetailScale + nfDetailOffset ).xyz * 2.0 - 1.0;
    d.xy *= nfDetailStrength;
    vec3 n = nfRNM( baseN, d );
    #ifdef NF_DETAIL2
      vec3 d2 = texture2D( nfDetailNormal, uv * nfDetailScale2 + nfDetailOffset ).xyz * 2.0 - 1.0;
      d2.xy *= nfDetailStrength2;
      n = nfRNM( n, d2 );
    #endif
    return n;
  }
#endif
`;

const PATCH_VERT_PARS = /* glsl */ `
#ifdef NF_WORLDSPACE
  varying vec3 vNfWPos;
  varying vec3 vNfWNrm;
#endif
`;

/**
 * Injected immediately before `project_vertex`.
 *
 * three applies `instanceMatrix`/`batchingMatrix` to `mvPosition` INSIDE
 * `project_vertex`, and to `transformedNormal` inside `defaultnormal_vertex` —
 * never to `transformed` or `objectNormal`. Reading those raw would give every
 * instance of an `InstancedMesh` the same world position, so the triplanar
 * projection would collapse onto instance zero and slide as the mesh moved.
 * We therefore replay the same transforms here, including three's
 * inverse-scale correction for the normal.
 */
const PATCH_VERT_BODY = /* glsl */ `
#ifdef NF_WORLDSPACE
  vec4 nfObjPos = vec4( transformed, 1.0 );
  vec3 nfObjNrm = objectNormal;

  #ifdef USE_BATCHING
    nfObjPos = batchingMatrix * nfObjPos;
    mat3 nfBm = mat3( batchingMatrix );
    nfObjNrm /= vec3( dot( nfBm[ 0 ], nfBm[ 0 ] ), dot( nfBm[ 1 ], nfBm[ 1 ] ), dot( nfBm[ 2 ], nfBm[ 2 ] ) );
    nfObjNrm = nfBm * nfObjNrm;
  #endif

  #ifdef USE_INSTANCING
    nfObjPos = instanceMatrix * nfObjPos;
    mat3 nfIm = mat3( instanceMatrix );
    nfObjNrm /= vec3( dot( nfIm[ 0 ], nfIm[ 0 ] ), dot( nfIm[ 1 ], nfIm[ 1 ] ), dot( nfIm[ 2 ], nfIm[ 2 ] ) );
    nfObjNrm = nfIm * nfObjNrm;
  #endif

  vNfWPos = ( modelMatrix * nfObjPos ).xyz;
  // mat3(modelMatrix), not its inverse-transpose: correct for the rigid and
  // uniformly-scaled transforms level geometry actually uses, and the result is
  // renormalised anyway. A non-uniformly scaled triplanar mesh would skew the
  // blend weights slightly; use uniform scale on ground meshes.
  vNfWNrm = normalize( mat3( modelMatrix ) * nfObjNrm );
#endif
`;

/**
 * REPLACES `map_fragment`.
 *
 * Replacing rather than appending matters for the triplanar path: appending
 * would leave the original UV-space fetch in place, so a ground material would
 * pay four texture fetches instead of three and blend a garbage sample.
 *
 * `map_fragment` runs before `color_fragment`, so overwriting `diffuseColor`
 * here cannot clobber vertex colours — they are applied on the next line.
 */
const PATCH_ALBEDO = /* glsl */ `
// ── shared state, declared at main() scope ───────────────────────────────────
// The roughness / metalness / normal patches further down all need these, and
// map_fragment is the earliest hook three gives us, so they are computed once
// here rather than three times. nfUvOff in particular MUST be identical
// across every map or the albedo, the relief and the gloss would each land on
// a different part of the surface.
vec2  nfUvOff  = vec2( 0.0 );
vec3  nfMacroV = vec3( 0.5 );
float nfGrimeK = 0.0;
float nfDustK  = 0.0;
float nfGrainV = 0.5;

#if defined( NF_MACRO ) || defined( NF_UVWARP )
  // A triplanar surface is used precisely because its mesh UVs are unreliable,
  // so the macro field has to be projected from world space too - sampling it
  // with vMapUv would reintroduce exactly the stretching we are avoiding.
  #ifdef NF_TRIPLANAR
    vec2 nfFieldUv = vNfWPos.xz * nfTriScale;
  #else
    vec2 nfFieldUv = vMapUv;
  #endif
#endif

#ifdef NF_UVWARP
{
  /**
   * ANTI-TILING BY DOMAIN WARP.
   *
   * A tiled texture betrays itself through the ALIGNMENT of its repeats, not
   * through the repeats themselves. What the eye locks onto on a long ground
   * plane is the straight ruled rows of identical features marching to the
   * horizon; it is nowhere near as good at spotting the same feature twice
   * when the two copies are not collinear.
   *
   * So instead of trying to make the tile unique (hex tiling, stochastic
   * tiling — both triple the fetch count of EVERY map, on the ground, which
   * is the surface that already costs the most because it is triplanar and
   * fills the bottom half of the screen), we displace the sample point by a
   * smooth vector field whose period is several tiles. The rows bend into
   * wandering curves and the lattice stops being legible.
   *
   * Cost: one texture fetch, of the macro field we are sampling anyway.
   *
   * The warp is read from .rb rather than .rg because .r and .g are the same
   * spatial field family (brightness / roughness share a warp) whereas .b is
   * generated at a different frequency, so .rb decorrelates the two components
   * of the offset vector and the warp does not collapse onto a diagonal.
   *
   * Local stretch is d(offset)/d(uv) ~ nfWarp.x * nfWarp.y * 2, i.e. a few
   * percent at the values used here — invisible on a granular surface, and on
   * a poured one it reads as settlement.
   */
  vec2 wv = texture2D( nfMacro, nfFieldUv * nfWarp.y ).rb - 0.5;
  nfUvOff = wv * nfWarp.x;
}
#endif

#ifdef NF_MACRO
  nfMacroV = texture2D( nfMacro, ( nfFieldUv + nfUvOff ) * nfMacroScale ).rgb;
#endif

#ifdef NF_WEATHER
{
  /**
   * WORLD-SPACE WEATHERING.
   *
   * A tile cannot know where the ground is, so tile-space noise can never
   * produce the two most recognisable weathering patterns in a real street:
   * the filthy splash-back band in the bottom half-metre of every wall, and
   * the pale dust that settles on every up-facing ledge. Both are trivial in
   * world space and both are worth far more than another octave of noise.
   */
  vec3 gwn = normalize( vNfWNrm );

  // Splash-back and capillary rise only happen on something water runs DOWN,
  // so the band is gated on verticality. Without this gate a concrete floor
  // (world y ~ 0, and there is a lot of it) would go uniformly black.
  float vertical = smoothstep( 0.30, 0.72, 1.0 - abs( gwn.y ) );

  // Break the top of the band with the macro field so it is a ragged tide-mark
  // rather than a ruled line. +-0.4 of the band height is plenty.
  float ragged = ( nfMacroV.r - 0.5 ) * 0.55 + ( nfMacroV.b - 0.5 ) * 0.30;
  float band = 1.0 - smoothstep( 0.0, 1.0, vNfWPos.y / max( nfWeather.x, 1e-3 ) + ragged );
  // Squared: grime is not linear in height, it is concentrated in the last
  // 10-15 cm where the rain actually splashes.
  nfGrimeK = saturate( band ) * saturate( band ) * vertical * nfWeather.y;

  float up = smoothstep( 0.55, 0.92, gwn.y );
  nfDustK = up * nfWeather.z * ( 0.45 + 0.55 * nfMacroV.r );
}
#endif

// ── base colour ──────────────────────────────────────────────────────────────
#if defined( NF_TRIPLANAR ) && defined( USE_MAP )
{
  vec3 wn = normalize( vNfWNrm );
  vec3 tw = nfTriWeights( wn );
  vec3 sgn = sign( wn );
  vec4 tx = texture2D( map, nfTriUvX( vNfWPos, sgn, nfTriScale ) + nfUvOff );
  vec4 ty = texture2D( map, nfTriUvY( vNfWPos, sgn, nfTriScale ) + nfUvOff );
  vec4 tz = texture2D( map, nfTriUvZ( vNfWPos, sgn, nfTriScale ) + nfUvOff );
  diffuseColor = vec4( diffuse, opacity ) * ( tx * tw.x + ty * tw.y + tz * tw.z );
}
#elif defined( NF_UVWARP ) && defined( USE_MAP )
  // Same as three's map_fragment, but through the warp. (No video-texture
  // branch: nothing in this library is a video texture.)
  diffuseColor *= texture2D( map, vMapUv + nfUvOff );
#else
  #include <map_fragment>
#endif

#ifdef NF_DETAILALB
{
  // MICRO GRAIN. The detail map's alpha is its height field, so this is the
  // cavity signal of a surface sampled ~12x finer than the base map can
  // resolve. Multiplying albedo by it restores the band between the base map's
  // Nyquist limit (4 mm on sand, 1.2 mm on concrete) and the ~0.3 mm the eye
  // can still resolve at arm's length. It is also energy-neutral: the field
  // averages to 0.5 and the expression below is 1.0 at 0.5, so distant
  // geometry is unaffected and only near geometry gains grain.
  #ifdef NF_TRIPLANAR
    // One tap, not three. On the ground the Y projection carries essentially
    // all of the weight, and grain has no orientation to get wrong, so the
    // two vertical projections would cost 2 fetches to change nothing.
    vec2 dUv = vNfWPos.xz * nfTriScale + nfUvOff;
  #else
    vec2 dUv = vMapUv + nfUvOff;
  #endif
  nfGrainV = texture2D( nfDetailNormal, dUv * nfDetailScale + nfDetailOffset ).a;
  diffuseColor.rgb *= mix( 1.0, 0.34 + nfGrainV * 1.32, nfDetailAlbedo );
}
#endif

#ifdef NF_MACRO
{
  float k = nfMacroStrength;
  // 0.5 is neutral in the macro field, so this is energy-neutral on average:
  // the effect fades out with distance instead of shifting the level tone.
  float bright = mix( 1.0, 0.42 + nfMacroV.r * 1.16, k );
  vec3 warm = vec3( 1.13, 1.03, 0.86 );   // sun-bleached
  vec3 cool = vec3( 0.90, 0.96, 1.10 );   // shadowed / damp
  vec3 tint = mix( vec3( 1.0 ), mix( cool, warm, nfMacroV.b ), k * 0.70 );
  diffuseColor.rgb *= bright * tint;
}
#endif

#ifdef NF_WEATHER
{
  // Linear-light constants. Grime is a dark neutral earth; dust is the pale
  // sand that is airborne everywhere in this compound. Both are DESATURATING
  // as well as shifting, which is what stops them reading as a coloured wash.
  vec3 grimeCol = vec3( 0.147, 0.102, 0.058 );
  vec3 dustCol  = vec3( 0.434, 0.342, 0.209 );
  diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * 0.34 + grimeCol * 0.42, nfGrimeK );
  diffuseColor.rgb = mix( diffuseColor.rgb, mix( diffuseColor.rgb, dustCol, 0.72 ), nfDustK );
}
#endif
`;

/**
 * REPLACES `roughnessmap_fragment`.
 *
 * Replacing rather than appending, because the warp has to be applied to the
 * fetch itself — appending would sample the unwarped texel first and throw it
 * away, which is a wasted fetch on the most expensive material in the level.
 */
const PATCH_ROUGHNESS = /* glsl */ `
float roughnessFactor = roughness;

#if defined( NF_TRIPLANAR ) && defined( USE_ROUGHNESSMAP )
{
  vec3 wn = normalize( vNfWNrm );
  vec3 tw = nfTriWeights( wn );
  vec3 sgn = sign( wn );
  float rx = texture2D( roughnessMap, nfTriUvX( vNfWPos, sgn, nfTriScale ) + nfUvOff ).g;
  float ry = texture2D( roughnessMap, nfTriUvY( vNfWPos, sgn, nfTriScale ) + nfUvOff ).g;
  float rz = texture2D( roughnessMap, nfTriUvZ( vNfWPos, sgn, nfTriScale ) + nfUvOff ).g;
  roughnessFactor = roughness * ( rx * tw.x + ry * tw.y + rz * tw.z );
}
#elif defined( USE_ROUGHNESSMAP )
  roughnessFactor *= texture2D( roughnessMap, vRoughnessMapUv + nfUvOff ).g;
#endif

#ifdef NF_DETAILALB
  // Grain modulates gloss as well as colour: a proud granule presents a
  // smoother facet to the light than the packed interstice beside it, which is
  // what makes granular ground sparkle at grazing incidence instead of
  // reading as felt. Small on purpose — this is a texture-scale wobble, not
  // the macro-scale patchiness below.
  roughnessFactor = clamp( roughnessFactor - ( nfGrainV - 0.5 ) * 0.30 * nfDetailAlbedo, 0.025, 1.0 );
#endif

#ifdef NF_MACRO
  // Roughness variance at macro scale is the highest-value line in this file.
  // A surface with one roughness value always reads as CG no matter how good
  // its albedo is, because every part of it answers the sun identically.
  // nfMacroRough is an absolute offset at full macro contrast, so 0.35 means
  // the same concrete goes from 0.55 (rain-polished) to 0.95 (chalked) across
  // one macro period.
  roughnessFactor = clamp(
    roughnessFactor + ( nfMacroV.g - 0.5 ) * 2.0 * nfMacroRough * nfMacroStrength,
    0.025, 1.0 );
#endif

#ifdef NF_WEATHER
  // Both films are matte and both bury whatever gloss the substrate had.
  roughnessFactor = mix( roughnessFactor, 0.97, nfGrimeK * 0.80 );
  roughnessFactor = mix( roughnessFactor, 0.94, nfDustK * 0.70 );
#endif
`;

/**
 * REPLACES `metalnessmap_fragment`.
 * Kept consistent with albedo/roughness: mixing a world-projected albedo with
 * a UV-projected metalness would put the rust in the wrong place.
 */
const PATCH_METALNESS = /* glsl */ `
float metalnessFactor = metalness;

#if defined( NF_TRIPLANAR ) && defined( USE_METALNESSMAP )
{
  vec3 wn = normalize( vNfWNrm );
  vec3 tw = nfTriWeights( wn );
  vec3 sgn = sign( wn );
  float mx = texture2D( metalnessMap, nfTriUvX( vNfWPos, sgn, nfTriScale ) + nfUvOff ).b;
  float my = texture2D( metalnessMap, nfTriUvY( vNfWPos, sgn, nfTriScale ) + nfUvOff ).b;
  float mz = texture2D( metalnessMap, nfTriUvZ( vNfWPos, sgn, nfTriScale ) + nfUvOff ).b;
  metalnessFactor = metalness * ( mx * tw.x + my * tw.y + mz * tw.z );
}
#elif defined( USE_METALNESSMAP )
  metalnessFactor *= texture2D( metalnessMap, vMetalnessMapUv + nfUvOff ).b;
#endif

#ifdef NF_WEATHER
  // Grime and dust are dielectric FILMS. A rusty container roof under a
  // centimetre of dust is not a conductor any more, and forgetting that is why
  // weathered metal so often still reads as shiny metal wearing a brown coat.
  metalnessFactor *= ( 1.0 - nfGrimeK * 0.85 ) * ( 1.0 - nfDustK * 0.75 );
#endif
`;

/**
 * Appended after `normal_fragment_maps`.
 *
 * `mapN` and `tbn` are declared at function scope inside that chunk, so they
 * are still live here — which lets us perturb the tangent-space normal and
 * re-transform, rather than replacing the whole chunk and having to reproduce
 * its object-space and bump-map branches.
 */
const PATCH_NORMAL = /* glsl */ `
#if defined( NF_TRIPLANAR ) && defined( USE_NORMALMAP_TANGENTSPACE )
{
  vec3 wn = normalize( vNfWNrm );
  vec3 tw = nfTriWeights( wn );
  vec3 sgn = sign( wn );

  vec2 ux = nfTriUvX( vNfWPos, sgn, nfTriScale ) + nfUvOff;
  vec2 uy = nfTriUvY( vNfWPos, sgn, nfTriScale ) + nfUvOff;
  vec2 uz = nfTriUvZ( vNfWPos, sgn, nfTriScale ) + nfUvOff;

  vec3 nx = texture2D( normalMap, ux ).xyz * 2.0 - 1.0;
  vec3 ny = texture2D( normalMap, uy ).xyz * 2.0 - 1.0;
  vec3 nz = texture2D( normalMap, uz ).xyz * 2.0 - 1.0;
  nx.xy *= normalScale;
  ny.xy *= normalScale;
  nz.xy *= normalScale;

  #ifdef NF_DETAIL
    nx = nfApplyDetail( nx, ux );
    ny = nfApplyDetail( ny, uy );
    nz = nfApplyDetail( nz, uz );
  #endif

  // The U flip above MIRRORS the sampled pattern, so the tangent normal's x
  // derivative has to be negated to match. Without this the bumps on the -X,
  // -Y and +Z faces light from the wrong side: for a -X face the decoded world
  // normal comes out (-1, -Hv, -Hu) where the analytic answer is (-1, -Hv, +Hu)
  // — a ~35 degree error, and it is applied AFTER the detail RNM so the detail
  // layer (sampled with the same flipped UVs) is corrected along with it.
  nx.x *=  sgn.x;
  ny.x *=  sgn.y;
  nz.x *= -sgn.z;

  // Whiteout blend into world space (Golus): add the geometric normal's
  // matching pair into XY, keep |Z| aligned with the projection axis, then
  // swizzle each back to world orientation and weight-blend.
  nx = vec3( nx.xy + wn.zy, abs( nx.z ) * wn.x );
  ny = vec3( ny.xy + wn.xz, abs( ny.z ) * wn.y );
  nz = vec3( nz.xy + wn.xy, abs( nz.z ) * wn.z );

  vec3 worldN = normalize( nx.zyx * tw.x + ny.xzy * tw.y + nz.xyz * tw.z );
  // World -> view. 'viewMatrix' is part of three's standard fragment prefix.
  normal = normalize( ( viewMatrix * vec4( worldN, 0.0 ) ).xyz );
}
#elif ( defined( NF_DETAIL ) || defined( NF_UVWARP ) ) && defined( USE_NORMALMAP_TANGENTSPACE )
{
  #ifdef NF_UVWARP
    // Re-fetch through the warp so the relief lands on the same texel as the
    // albedo. Discarding the chunk's unwarped fetch costs one sample and is
    // the cheapest correct option: replacing normal_fragment_maps outright
    // would mean reproducing its object-space, bump-map and anisotropy
    // branches, which move between three releases.
    mapN = texture2D( normalMap, vNormalMapUv + nfUvOff ).xyz * 2.0 - 1.0;
    mapN.xy *= normalScale;
  #endif
  #ifdef NF_DETAIL
    mapN = nfApplyDetail( mapN, vNormalMapUv + nfUvOff );
  #endif
  normal = normalize( tbn * mapN );
}
#endif

#ifdef NF_WEATHER
{
  // A grime or dust film is a coating: it fills the substrate's micro-relief
  // and flattens it. Lerping the shaded normal toward the geometric one is a
  // cheap, monotone stand-in for that, and without it a wall reads as "clean
  // relief with a dark stripe painted on" instead of as caked dirt.
  vec3 flatN = normalize( ( viewMatrix * vec4( normalize( vNfWNrm ), 0.0 ) ).xyz );
  normal = normalize( mix( normal, flatN, saturate( nfGrimeK * 0.55 + nfDustK * 0.45 ) ) );
}
#endif

#ifdef NF_SPECAA
{
  /**
   * GEOMETRIC SPECULAR ANTI-ALIASING (Kaplanyan et al. 2016 / Tokuyoshi 2019).
   *
   * The whole point of this library is high-frequency normal detail, and the
   * price of high-frequency normals is that a smooth surface turns into a
   * field of crawling glints: within one pixel the normal sweeps through many
   * directions, the specular lobe is narrower than that sweep, and the
   * integral the GPU computes from a single sample is simply wrong.
   *
   * The fix is to fold the pixel's normal variance into roughness, which is
   * exactly what a wider lobe means physically. Two screen-space derivatives
   * and a hypot, and it converts sparkle into a soft sheen — which is what a
   * real granular surface looks like anyway.
   *
   * roughnessFactor is declared by the roughness chunk, which three emits
   * BEFORE normal_fragment_maps, so it is still live here and still upstream of
   * lights_physical_fragment. This is the only point in the pipeline where the
   * final shading normal and the roughness are both available.
   */
  vec3 dnx = dFdx( normal );
  vec3 dny = dFdy( normal );
  float variance = 0.25 * ( dot( dnx, dnx ) + dot( dny, dny ) );
  // Clamped: without a cap, a silhouette edge or an interpenetration would
  // spike the derivative and blow the surface out to fully rough.
  float kernel = min( 2.0 * variance, 0.25 );
  roughnessFactor = min( 1.0, sqrt( roughnessFactor * roughnessFactor + kernel ) );
}
#endif
`;

// ─────────────────────────────────────────────────────────────────────────────

export class MaterialLibrary implements IMaterialLibrary {
  readonly name = 'materials';

  private forge: TextureForge | null = null;
  private readonly materials = new Map<string, PatchableMaterial>();
  private readonly descriptors = new Map<string, MaterialDescriptor>();
  private readonly sets = new Map<string, PBRTextureSet>();
  private readonly patches = new Map<string, Patch>();

  private detailNormal: THREE.Texture | null = null;
  private macroField: THREE.Texture | null = null;

  /** Placeholder handed out before `init` resolves, so callers never see null. */
  private readonly fallback: THREE.MeshStandardMaterial;

  private elapsed = 0;
  private waterPatch: Patch | null = null;
  private waterMaterial: PatchableMaterial | null = null;

  /** True once every surface has been baked. */
  ready = false;

  private initStarted = false;
  private resolveReady!: () => void;
  private readyPromise = new Promise<void>((r) => { this.resolveReady = r; });

  /** Total baked texture memory in bytes, mip chains included. */
  vramBytes = 0;

  /**
   * Surfaces whose shader failed to compile, as `textureName: first error line`.
   *
   * MUST be empty. A failed bake is not an exception — the draw is dropped and
   * the render target keeps its clear colour — so the only in-game symptom is a
   * surface with black albedo and an ORM of (0, 0, 0): fully occluded, mirror
   * roughness, dielectric. `concrete` and `plaster` shipped exactly that way.
   */
  shaderFailures: readonly string[] = [];

  constructor() {
    // Neutral weathered concrete so anything rendered before init still reads
    // as part of the art direction rather than as magenta debug material.
    this.fallback = new THREE.MeshStandardMaterial({
      // NO convertSRGBToLinear() here: `Color.setHex` already defaults to
      // `SRGBColorSpace` and runs `ColorManagement.colorSpaceToWorking`, so
      // converting a second time would darken this by roughly 4.8x.
      color: new THREE.Color(0x8a8578),
      roughness: 0.9,
      metalness: 0.0,
      name: 'noisefield.fallback',
    });
    this.fallback.userData['id'] = 'fallback';
    this.fallback.userData['surface'] = 'concrete';
    this.fallback.userData['tileMeters'] = 2.0;
    this.fallback.userData['triplanar'] = false;
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  async init(ctx: EngineContext): Promise<void> {
    // The engine calls init() once, but a hot reload or a duplicate
    // registration would otherwise bake a second full texture set and leak the
    // first. Returning the same promise also makes whenReady() correct.
    if (this.initStarted) return this.readyPromise;
    this.initStarted = true;

    const t0 = performance.now();
    const forge = new TextureForge(ctx.renderer, ctx.settings.anisotropy);
    this.forge = forge;

    const tier = ctx.settings.tier;

    try {
      await this.bakeAll(forge, tier);
    } catch (err) {
      // Never leave whenReady() pending: a half-baked library still renders
      // (missing ids fall back), whereas a hung promise deadlocks every system
      // that gated its own init on ours.
      console.error('[materials] bake failed; unbaked surfaces will use the fallback', err);
    } finally {
      this.vramBytes = forge.bytesAllocated;
      this.ready = true;
      this.resolveReady();
    }

    const mb = this.vramBytes / (1024 * 1024);
    const ms = (performance.now() - t0).toFixed(0);
    console.info(
      `[materials] ${this.materials.size}/${ALL_RECIPES.length} surfaces, ${forge.textureCount} textures, ` +
      `${mb.toFixed(1)} MB VRAM, baked in ${ms} ms (tier=${tier}, aniso=${forge.anisotropy})`,
    );
    if (this.vramBytes > VRAM_BUDGET_BYTES) {
      console.warn(
        `[materials] texture budget exceeded: ${mb.toFixed(1)} MB > ${(VRAM_BUDGET_BYTES / 1048576).toFixed(0)} MB. ` +
        'Lower QualitySettings.tier to shrink the bake.',
      );
    }
    // A failed bake is not an exception — it returns the clear colour, so the
    // surface silently ships black with an ORM of 0/0/0. Report it as an error
    // and keep the list queryable so a test can assert that it is empty.
    this.shaderFailures = [...forge.shaderFailures];
    if (this.shaderFailures.length > 0) {
      console.error(
        `[materials] ${this.shaderFailures.length} shader(s) FAILED TO BAKE and contain the clear colour ` +
        `(black albedo, ORM 0/0/0):\n  ${this.shaderFailures.join('\n  ')}`,
      );
    }
  }

  /** The bake itself, split out so `init` can guarantee the readiness promise. */
  private async bakeAll(forge: TextureForge, tier: QualityTier): Promise<void> {
    // ── shared fields, baked first because every material references them ────
    const detailSize = this.resolveSize(1024, tier);
    const detail = forge.bakeNormalFromSurface(
      DETAIL_NORMAL_GLSL, detailSize, 7.0, NF_DETAIL_SLOPE, 'noisefield.detail',
    );
    this.detailNormal = detail.normal;

    this.macroField = forge.bakeSurfacePass(
      MACRO_VARIATION_GLSL, PASS_ALBEDO, this.resolveSize(512, tier), 3.0, {},
      // Linear, NOT sRGB: these are control values, not colour. Encoding them
      // would skew every threshold downstream.
      { colorSpace: THREE.NoColorSpace, name: 'noisefield.macro' },
    );
    await yieldToBrowser();

    // ── per-surface bakes ────────────────────────────────────────────────────
    for (const recipe of ALL_RECIPES) {
      this.bakeRecipe(forge, recipe, tier);
      // One yield per surface. A full ultra bake is ~70 GPU passes; without
      // yielding the browser would report the tab as unresponsive even though
      // the total wall time is well under a second.
      await yieldToBrowser();
    }
  }

  /**
   * Scrolls the two water normal layers in opposite directions.
   *
   * The base normal map moves one way and the detail layer (sampled through
   * the shared patch uniform) moves the other, so the two wave systems
   * interfere. Scrolling a single layer just looks like a sliding texture; two
   * counter-moving layers is the cheapest thing that actually reads as water.
   */
  update(_ctx: EngineContext, frame: FrameContext): void {
    if (!this.ready || !this.waterPatch || !this.waterMaterial) return;
    this.elapsed += frame.dt;
    const t = this.elapsed;

    const mat = this.waterMaterial;
    const baseX = 0.013;
    const baseY = 0.021;
    // Wrapped into [0,1): a RepeatWrapping offset is periodic mod 1, so this is
    // exact, invisible, and stops `elapsed * v` from shedding mantissa bits
    // after a long session.
    const w = (v: number): number => v - Math.floor(v);
    if (mat.normalMap) mat.normalMap.offset.set(w(t * baseX), w(t * baseY));
    if (mat.map) mat.map.offset.set(w(t * -0.008), w(t * 0.011));

    // The detail layer is sampled at `vNormalMapUv * detailScale + offset`, so
    // it INHERITS the base scroll multiplied by detailScale. A plain negative
    // offset is not enough — the inherited term dominates and both layers end
    // up travelling the same way, which reads as one sliding texture rather
    // than as interference. Cancel the inherited motion, then add the genuinely
    // opposite velocity.
    const ds = this.waterPatch.detailScale.value;
    this.waterPatch.detailOffset.value.set(
      w(t * (-baseX * ds - 0.021)),
      w(t * (-baseY * ds - 0.034)),
    );
  }

  dispose(): void {
    for (const material of this.materials.values()) material.dispose();
    this.materials.clear();
    this.descriptors.clear();
    this.sets.clear();
    this.patches.clear();
    this.fallback.dispose();
    this.forge?.dispose();
    this.forge = null;
    this.detailNormal = null;
    this.macroField = null;
    this.waterPatch = null;
    this.waterMaterial = null;
    this.ready = false;
    this.vramBytes = 0;
    // Re-arm, so dispose() -> init() actually re-bakes instead of silently
    // no-opping on the idempotency guard and handing out the fallback forever.
    this.initStarted = false;
    this.readyPromise = new Promise<void>((r) => { this.resolveReady = r; });
  }

  // ── IMaterialLibrary ───────────────────────────────────────────────────────

  get(id: string): THREE.Material {
    const found = this.materials.get(id);
    if (found) return found;
    if (this.ready) {
      console.warn(`[materials] unknown material id "${id}"; known ids: ${this.ids().join(', ')}`);
    } else {
      // Warning here rather than staying quiet: a material is assigned BY
      // VALUE, so a caller that grabs the fallback before the bake finishes
      // keeps the grey placeholder for the rest of the session. Await
      // whenReady() first.
      console.warn(`[materials] get("${id}") before init() completed — returning the fallback. Await whenReady().`);
    }
    return this.fallback;
  }

  forSurface(surface: SurfaceType): THREE.Material {
    return this.get(SURFACE_DEFAULT_ID[surface] ?? 'concrete');
  }

  /**
   * Resolve a mesh's surface class by walking up the hierarchy.
   *
   * Walking up matters because level geometry is usually tagged once on a
   * group ("this whole shed is wood") while the ray hit lands on a child mesh.
   * Defaults to concrete, which is the least surprising answer for untagged
   * static geometry in this level.
   */
  surfaceOf(object: THREE.Object3D): SurfaceType {
    let node: THREE.Object3D | null = object;
    while (node) {
      const raw: unknown = node.userData['surface'];
      if (typeof raw === 'string' && SURFACE_TYPES.has(raw)) {
        return raw as SurfaceType;
      }
      node = node.parent;
    }
    return 'concrete';
  }

  /** Tag for `surfaceOf`. Stored in `userData` so any system can read it. */
  tag(object: THREE.Object3D, surface: SurfaceType): void {
    object.userData['surface'] = surface;
  }

  // ── extended API ───────────────────────────────────────────────────────────

  /**
   * Resolves once every surface has been baked.
   *
   * `init()` is async, so anything that asks for a material during its own
   * `init()` gets the grey fallback and keeps it forever — the material is
   * assigned by value, not by reference. Await this first.
   */
  whenReady(): Promise<void> {
    return this.readyPromise;
  }

  /** All registered material ids (17 — more than there are SurfaceTypes). */
  ids(): string[] {
    return [...this.materials.keys()];
  }

  /**
   * Metres-to-UV factor for a surface: author geometry UVs as
   * `worldPosition * uvScale(id)` so one texture tile covers `tileMeters`.
   */
  uvScale(id: string): number {
    return 1 / this.tileMeters(id);
  }

  /** The material id a mesh is using, if it came from this library. */
  materialIdOf(object: THREE.Object3D): string | undefined {
    const mesh = object as Partial<THREE.Mesh>;
    const mat = mesh.material;
    if (!mat || Array.isArray(mat)) return undefined;
    const id: unknown = mat.userData['id'];
    return typeof id === 'string' ? id : undefined;
  }

  /**
   * A per-instance copy that KEEPS the shader patch.
   *
   * `material.clone()` does not work on anything from this library: three's
   * clone resets `defines` and copies neither `onBeforeCompile` nor
   * `customProgramCacheKey`, so the clone silently loses the detail normal,
   * the macro variation and the triplanar projection. This rebuilds and
   * re-patches instead. Use it sparingly — every variant is another material,
   * and the whole point of the library is that meshes share one.
   *
   * Always returns a `MeshPhysicalMaterial` (a superset of Standard) so that
   * any override in `MeshPhysicalMaterialParameters` is actually applicable,
   * even when the base recipe is a Standard material.
   */
  variant(id: string, overrides: THREE.MeshPhysicalMaterialParameters): THREE.Material {
    const recipe = ALL_RECIPES.find((r) => r.id === id);
    const set = this.sets.get(id);
    if (!recipe || !set) {
      console.warn(`[materials] cannot build a variant of unknown id "${id}"`);
      return this.get(id);
    }
    const tuning: RecipeTuning = { ...DEFAULT_TUNING, ...recipe.tuning };
    const merged: SurfaceRecipe = recipe.physical === true
      ? { ...recipe, params: { ...recipe.params, ...overrides } }
      : { ...recipe, physical: true, params: { ...recipe.params, ...overrides } };
    const material = this.buildMaterial(merged, set);
    this.patchMaterial(material, merged, tuning, false);
    material.userData['id'] = id;
    material.userData['surface'] = recipe.surface;
    material.userData['tileMeters'] = recipe.tileMeters;
    material.userData['triplanar'] = tuning.triplanar;
    return material;
  }

  /** Full descriptor (surface class + world tile size) for an id. */
  descriptor(id: string): MaterialDescriptor | undefined {
    return this.descriptors.get(id);
  }

  /** The baked texture set for an id, e.g. to build a custom material. */
  textures(id: string): PBRTextureSet | undefined {
    return this.sets.get(id);
  }

  /**
   * World metres covered by one texture tile. Materials are shared, so
   * `texture.repeat` must stay at (1,1); world geometry is expected to author
   * UVs as `worldPosition / tileMeters`.
   */
  tileMeters(id: string): number {
    return this.descriptors.get(id)?.tileMeters ?? 2.0;
  }

  /** Assign a material to a mesh and tag it in one call. */
  apply(object: THREE.Object3D, id: string): void {
    const descriptor = this.descriptors.get(id);
    const material = this.get(id);
    object.traverse((node) => {
      const mesh = node as Partial<THREE.Mesh>;
      if (mesh.isMesh === true) {
        (node as THREE.Mesh).material = material;
      }
    });
    this.tag(object, descriptor?.surface ?? 'concrete');
  }

  /** Runtime override of the detail/macro blend, e.g. from a debug UI. */
  setTuning(id: string, tuning: Partial<Pick<RecipeTuning, 'detailScale' | 'detailStrength' | 'macroScale' | 'macroStrength'>>): void {
    const patch = this.patches.get(id);
    if (!patch) return;
    if (tuning.detailScale !== undefined) patch.detailScale.value = tuning.detailScale;
    if (tuning.detailStrength !== undefined) patch.detailStrength.value = tuning.detailStrength;
    if (tuning.macroScale !== undefined) patch.macroScale.value = tuning.macroScale;
    if (tuning.macroStrength !== undefined) patch.macroStrength.value = tuning.macroStrength;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  /**
   * Resolution policy.
   *
   * 2048^2 hero maps only exist at `ultra`; everything else caps at 1024, and
   * only the lower two tiers apply a linear scale. Measured resident totals:
   *
   *   ultra  210.7 MB   (2048 concrete; 53 textures)
   *   high   178.7 MB   (concrete demoted to 1024 — the ONLY difference from
   *                      ultra, so this is 85% of it, not half)
   *   medium  44.7 MB   (0.5x linear)
   *   low     11.2 MB   (0.25x linear)
   *
   * Peak during the bake is one height map higher (<= 16 MB), because each
   * height field is released as soon as its normal map has been derived.
   */
  private resolveSize(base: number, tier: QualityTier): number {
    const scale = tier === 'low' ? 0.25 : tier === 'medium' ? 0.5 : 1.0;
    const cap = tier === 'ultra' ? 2048 : 1024;
    const raw = Math.min(base, cap) * scale;
    // Snap to a power of two: non-POT is legal in WebGL2 but POT keeps the mip
    // chain exact, which matters for a texture that must tile.
    const pot = 2 ** Math.round(Math.log2(Math.max(raw, 64)));
    return Math.max(64, Math.min(2048, pot));
  }

  private bakeRecipe(forge: TextureForge, recipe: SurfaceRecipe, tier: QualityTier): void {
    const tuning: RecipeTuning = { ...DEFAULT_TUNING, ...recipe.tuning };
    const size = this.resolveSize(recipe.size, tier);
    const ormSize = this.resolveSize(Math.max(64, recipe.size / (recipe.ormDivisor ?? 2)), tier);

    // Height -> Sobel -> normal, then the height is released unless the recipe
    // asked to keep it for displacement.
    const derived = forge.bakeNormalFromSurface(
      recipe.glsl, size, recipe.seed, tuning.normalStrength, `noisefield.${recipe.id}`, {}, recipe.keepHeight === true,
    );

    const map = forge.bakeSurfacePass(recipe.glsl, PASS_ALBEDO, size, recipe.seed, {}, {
      colorSpace: THREE.SRGBColorSpace,
      name: `noisefield.${recipe.id}.albedo`,
    });

    const orm = forge.bakeSurfacePass(recipe.glsl, PASS_ORM, ormSize, recipe.seed, {}, {
      colorSpace: THREE.NoColorSpace,
      name: `noisefield.${recipe.id}.orm`,
    });

    const set: PBRTextureSet = {
      map,
      normalMap: derived.normal,
      // One ORM texture in three slots — see the header note on glTF packing.
      roughnessMap: orm,
      metalnessMap: orm,
    };
    // Triplanar surfaces deliberately get no aoMap. three's `aomap_fragment`
    // inlines the clearcoat/sheen/envmap branches, so world-projecting it would
    // mean duplicating chunk internals that shift between releases — while a
    // UV-projected AO on a surface whose UVs we do not trust would smear. Ground
    // AO is low-contrast and SSAO covers it, so dropping it is the robust call.
    if (!tuning.triplanar) set.aoMap = orm;
    if (derived.height) set.displacementMap = derived.height;

    const material = this.buildMaterial(recipe, set);
    this.patchMaterial(material, recipe, tuning);

    // Published on the material itself because `IMaterialLibrary` hands back a
    // bare `THREE.Material`: code that only has the material (a decal builder,
    // a UV helper, a debug overlay) would otherwise have no way back to the
    // tile size, and guessing it is the most common integration mistake.
    material.userData['id'] = recipe.id;
    material.userData['surface'] = recipe.surface;
    material.userData['tileMeters'] = recipe.tileMeters;
    material.userData['triplanar'] = tuning.triplanar;

    this.materials.set(recipe.id, material);
    this.sets.set(recipe.id, set);
    this.descriptors.set(recipe.id, {
      id: recipe.id,
      surface: recipe.surface,
      tileMeters: recipe.tileMeters,
      material,
    });

    if (recipe.id === 'water') {
      this.waterMaterial = material;
      this.waterPatch = this.patches.get('water') ?? null;
    }
  }

  private buildMaterial(recipe: SurfaceRecipe, set: PBRTextureSet): PatchableMaterial {
    // roughness/metalness stay at 1 so they act as pure multipliers on the ORM
    // texture's G and B channels rather than overriding them.
    const common = {
      name: `noisefield.${recipe.id}`,
      map: set.map,
      normalMap: set.normalMap,
      normalScale: new THREE.Vector2(1, 1),
      roughnessMap: set.roughnessMap,
      metalnessMap: set.metalnessMap,
      roughness: 1.0,
      metalness: 1.0,
      aoMapIntensity: 1.0,
      envMapIntensity: 1.0,
      // 8-bit output plus a low-frequency macro gradient is exactly the case
      // that bands; dithering costs nothing and removes it.
      dithering: true,
    } satisfies THREE.MeshStandardMaterialParameters;

    // `aoMap` is spread in only when present: three's `Material.setValues`
    // warns "parameter 'aoMap' has value of undefined" for an explicit
    // undefined, and triplanar surfaces deliberately have no aoMap.
    const ao = set.aoMap ? { aoMap: set.aoMap } : {};

    if (recipe.physical === true) {
      return new THREE.MeshPhysicalMaterial({ ...common, ...ao, ...recipe.params });
    }
    return new THREE.MeshStandardMaterial({ ...common, ...ao, ...recipe.params });
  }

  /**
   * Inject the detail-normal / macro-variation / triplanar code.
   *
   * `customProgramCacheKey` is mandatory here: three caches compiled programs
   * by material type plus defines, and would happily reuse one patched
   * material's program for another with different injected source.
   */
  private patchMaterial(material: PatchableMaterial, recipe: SurfaceRecipe, tuning: RecipeTuning, register = true): void {
    const useDetail = tuning.detailScale > 0 && tuning.detailStrength > 0 && this.detailNormal !== null;
    const useDetail2 = useDetail && tuning.detailScale2 > 0 && tuning.detailStrength2 > 0;
    const useDetailAlb = useDetail && tuning.detailAlbedo > 0;
    const useTri = tuning.triplanar;
    const useWeather = tuning.grimeStrength * tuning.grimeHeight > 0 || tuning.dustStrength > 0;
    // The warp vector is read out of the macro field, and the weather band is
    // broken up by it, so both imply it. Forcing it here rather than silently
    // producing an unwarped surface is deliberate: a missing warp is invisible
    // in code review and glaringly obvious on a long ground plane.
    const wantWarp = tuning.uvWarp > 0;
    const useMacro = (tuning.macroScale > 0 && tuning.macroStrength > 0 && this.macroField !== null)
      || ((wantWarp || useWeather) && this.macroField !== null);
    const useWarp = wantWarp && useMacro;
    if (!useDetail && !useMacro && !useTri && !useWeather) return;

    const patch: Patch = {
      detailScale: { value: tuning.detailScale },
      detailStrength: { value: tuning.detailStrength },
      detailScale2: { value: tuning.detailScale2 },
      detailStrength2: { value: tuning.detailStrength2 },
      detailAlbedo: { value: tuning.detailAlbedo },
      macroScale: { value: tuning.macroScale },
      macroStrength: { value: tuning.macroStrength },
      macroRough: { value: tuning.macroRough },
      warp: { value: new THREE.Vector2(tuning.uvWarp, tuning.uvWarpScale) },
      weather: { value: new THREE.Vector3(tuning.grimeHeight, tuning.grimeStrength, tuning.dustStrength) },
      detailOffset: { value: new THREE.Vector2(0, 0) },
      triScale: { value: 1 / Math.max(tuning.triplanarScale, 1e-3) },
    };
    // Variants deliberately do not register: otherwise setTuning('water')
    // would retune the variant and the shared water material would stop
    // animating, because update() holds whichever patch was registered last.
    if (register) this.patches.set(recipe.id, patch);

    // MERGE, never overwrite: MeshPhysicalMaterial relies on its own STANDARD
    // and PHYSICAL defines being present.
    const defines: Record<string, unknown> = { ...(material.defines ?? {}) };
    if (useDetail) defines['NF_DETAIL'] = '';
    if (useDetail2) defines['NF_DETAIL2'] = '';
    if (useDetailAlb) defines['NF_DETAILALB'] = '';
    if (useMacro) defines['NF_MACRO'] = '';
    if (useTri) defines['NF_TRIPLANAR'] = '';
    if (useWarp) defines['NF_UVWARP'] = '';
    if (useWeather) defines['NF_WEATHER'] = '';
    // One flag for "this shader needs the world position and normal", so the
    // vertex patch does not have to know why.
    if (useTri || useWeather) defines['NF_WORLDSPACE'] = '';
    // Specular AA is only worth its two derivatives where we have deliberately
    // put high-frequency normals, which is exactly where NF_DETAIL is on.
    if (useDetail) defines['NF_SPECAA'] = '';
    material.defines = defines;

    const detailNormal = this.detailNormal;
    const macroField = this.macroField;

    material.onBeforeCompile = (shader) => {
      if (useDetail && detailNormal) {
        shader.uniforms['nfDetailNormal'] = { value: detailNormal };
        shader.uniforms['nfDetailScale'] = patch.detailScale;
        shader.uniforms['nfDetailStrength'] = patch.detailStrength;
        shader.uniforms['nfDetailOffset'] = patch.detailOffset;
        if (useDetail2) {
          shader.uniforms['nfDetailScale2'] = patch.detailScale2;
          shader.uniforms['nfDetailStrength2'] = patch.detailStrength2;
        }
        if (useDetailAlb) shader.uniforms['nfDetailAlbedo'] = patch.detailAlbedo;
      }
      if (useMacro && macroField) {
        shader.uniforms['nfMacro'] = { value: macroField };
        shader.uniforms['nfMacroScale'] = patch.macroScale;
        shader.uniforms['nfMacroStrength'] = patch.macroStrength;
        shader.uniforms['nfMacroRough'] = patch.macroRough;
      }
      if (useWarp) shader.uniforms['nfWarp'] = patch.warp;
      if (useWeather) shader.uniforms['nfWeather'] = patch.weather;
      if (useTri) shader.uniforms['nfTriScale'] = patch.triScale;

      if (useTri || useWeather) {
        shader.vertexShader = shader.vertexShader
          .replace('#include <common>', `#include <common>\n${PATCH_VERT_PARS}`)
          .replace('#include <project_vertex>', `${PATCH_VERT_BODY}\n#include <project_vertex>`);
      }

      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>\n${PATCH_FRAG_PARS}`)
        // The first three REPLACE their chunk (and re-include or reproduce it
        // on the paths that need the stock behaviour), because the UV warp has
        // to be inside the fetch. PATCH_NORMAL still appends, so that three's
        // object-space / bump / anisotropy branches keep working untouched.
        .replace('#include <map_fragment>', PATCH_ALBEDO)
        .replace('#include <roughnessmap_fragment>', PATCH_ROUGHNESS)
        .replace('#include <metalnessmap_fragment>', PATCH_METALNESS)
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${PATCH_NORMAL}`);
    };

    // Keyed on the FEATURE FLAGS, not the id. three appends this to a key that
    // already contains the material type, every define and every parameter, and
    // the injected source depends only on these flags — all the tuning lives in
    // uniforms. Keying on `recipe.id` would force ~17 program compiles where a
    // handful will do.
    const key = 'noisefield:'
      + (useDetail ? 'd' : '') + (useDetail2 ? '2' : '') + (useDetailAlb ? 'a' : '')
      + (useMacro ? 'm' : '') + (useTri ? 't' : '') + (useWarp ? 'w' : '') + (useWeather ? 'g' : '')
      + (useDetail ? 's' : '');
    material.customProgramCacheKey = () => key;
    material.needsUpdate = true;
  }
}

/**
 * Hand control back to the browser between bakes.
 *
 * `setTimeout(0)` rather than `requestAnimationFrame` on purpose: rAF would tie
 * the bake rate to the display refresh and stretch a ~0.5 s job over a second
 * or more on a 60 Hz panel, whereas a macrotask yield lets the event loop
 * drain and immediately continues.
 */
function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
