/**
 * NOISEFIELD — shared detail + macro-variation fields.
 *
 * These two textures are baked once and referenced by every material. They are
 * the cheapest, highest-impact anti-tiling technique available:
 *
 *   DETAIL NORMAL  sampled at ~8-16x the base UV rate (and a second time at
 *                  ~1/5 of that). Supplies the micro-scale bumpiness that a
 *                  1024^2 base map physically cannot hold, so surfaces keep
 *                  specular breakup when the camera is 30 cm away. Blended
 *                  with reoriented normal mapping (RNM) rather than a lerp, so
 *                  the base map's shape is preserved exactly and only
 *                  perturbed.
 *
 *   MACRO FIELD    sampled at ~1/8 the base UV rate, i.e. one macro feature
 *                  spans eight texture tiles. Multiplies albedo, offsets
 *                  roughness, and (when a surface opts in) drives the domain
 *                  warp that bends the tile lattice out of alignment.
 *
 * Both are grey/neutral by construction so they work on every surface.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY THE AMPLITUDE SPECTRUM IS THE WHOLE BALL GAME
 * ─────────────────────────────────────────────────────────────────────────────
 * A detail normal map is only useful at the mip level the hardware actually
 * selects. At 2 m from a wall, with a 1024^2 detail map covering ~0.2 m, one
 * detail texel is ~0.2 mm while one screen pixel covers ~1.8 mm — so the GPU
 * reads mip 3, an 8x8 texel average.
 *
 * That makes the amplitude spectrum, not the peak slope, the thing that
 * decides whether detail survives. If the field is amplitude-FLAT (every
 * octave the same height), the slope is dominated by the top octave, the
 * derived normals sit at ~85-88 degrees off vertical, and averaging eight of
 * those near-random directions returns (0,0,1) — dead flat. The surface then
 * has violent noise at 20 cm and nothing at all at 2 m, which is precisely the
 * "smooth plastic at combat range" failure detail maps exist to prevent.
 *
 * A 1/f ("pink") spectrum — amplitude halved every time frequency doubles —
 * gives every octave the SAME slope contribution. Each mip drops the top
 * octave and keeps the rest, so the surface degrades gracefully from micro
 * grain at 30 cm to centimetre relief at 3 m instead of falling off a cliff.
 * Natural weathered surfaces are close to 1/f for exactly this reason.
 */

/**
 * Micro-surface height: a 1/f octave stack plus clustered grit and sparse
 * pits. Baked as a height pass and run through the Sobel derivation like any
 * other surface, which keeps the encoding identical everywhere.
 *
 * Total peak-to-peak amplitude is ~1.0 so the 16-bit height target is fully
 * used; the *slope* is then set once, globally, by the bake strength in
 * MaterialLibrary (see NF_DETAIL_SLOPE there).
 */
export const DETAIL_NORMAL_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {
  // ── 1/f octave stack ──────────────────────────────────────────────────────
  // Amplitude halves as frequency doubles, so slope_i = k * A_i * f_i is the
  // same for every i. See the file header for why this is load-bearing.
  //
  // The lowest octave (period 5) lands at ~4 cm of surface once a wall samples
  // this at 12x its base rate — that is the band that reads as "rendered
  // masonry" at three metres. The top octave (period 160) lands at ~1.2 mm,
  // which is the band that reads as grain when the player's face is on it.
  float h = 0.5;
  h += ( apxGradient( uv *   5.0,   5.0 ) - 0.5 ) * 0.520;
  h += ( apxValue(    uv *  10.0,  10.0 ) - 0.5 ) * 0.260;
  h += ( apxValue(    uv *  20.0,  20.0 ) - 0.5 ) * 0.130;
  h += ( apxValue(    uv *  40.0,  40.0 ) - 0.5 ) * 0.065;
  h += ( apxValue(    uv *  80.0,  80.0 ) - 0.5 ) * 0.033;
  h += ( apxValue(    uv * 160.0, 160.0 ) - 0.5 ) * 0.017;

  // ── clustered grit ────────────────────────────────────────────────────────
  // Worley supplies the CLUMPED granule silhouette that a gradient-noise stack
  // never produces: real grit sits in drifts, not in an even dither. Held to
  // the same slope budget as the octave it replaces.
  float grit = 1.0 - apxFbmWorley( uv * 26.0, 26.0, 2, 0.5, 1.0 );
  h += ( grit - 0.5 ) * 0.105;

  // ── sparse pits ───────────────────────────────────────────────────────────
  // Only the deepest few percent of a cellular field clear the threshold, so
  // these are isolated pinholes rather than a regular dimple pattern. Being
  // sparse, they contribute steep local slope without polluting the average —
  // they survive as glints under a low sun and vanish cleanly in the mips.
  vec3  cw  = apxWorley( uv * 18.0, 18.0, 1.0 );
  float pit = apxSat( 1.0 - cw.x * 5.5 ) * step( 0.875, cw.z );
  h -= pit * 0.150;

  // ── hairline scratch film ─────────────────────────────────────────────────
  // Two crossed anisotropic scratch sets. Directional micro-structure is what
  // makes a specular highlight streak instead of forming a clean disc, and a
  // streaked highlight is the single most reliable "this is a real material"
  // cue at grazing angles.
  float sc = apxScratches( uv, 30.0, 4.0, 6.0 )
           + apxScratches( uv * 2.0 + 19.0, 7.0, 26.0, 6.0 ) * 0.8;
  h -= apxSat( sc ) * 0.055;

  s.height = clamp( h, 0.0, 1.0 );
}
`;

/**
 * Macro variation, packed as:
 *   R = albedo brightness multiplier control (0.5 is neutral)
 *   G = roughness offset control (0.5 is neutral)
 *   B = warm/cool tint control (0.5 is neutral)
 *
 * Two decorrelated scales per channel. A single-scale macro field only trades
 * a small repeat for a large one — the eye still locks onto it on a long
 * ground plane. Beating two scales against each other (periods 3 and 9, 5 and
 * 13, so their least common multiple is the whole texture) produces a field
 * with no visually obvious period.
 *
 * The contrast expansion at the end is not cosmetic. Fractal noise spends most
 * of its time near its mean, so an un-expanded field delivers roughly a third
 * of the swing the strength uniform claims — which is exactly how a material
 * system ends up "having macro variation" that nobody can see. Expanding about
 * the neutral midpoint clips symmetrically, so the mip average stays at 0.5
 * and the effect still fades out with distance instead of shifting the tone of
 * the level. The clipped plateaus are a feature: they read as discrete
 * weathering patches rather than as smooth gradients.
 */
export const MACRO_VARIATION_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {
  // ── brightness ────────────────────────────────────────────────────────────
  vec2  w  = apxWarp( uv * 3.0, 3.0, 0.70, 3 );
  float b0 = apxFbm( w, 3.0, 4, 2.0, 0.58 );
  float b1 = apxFbmValue( uv * 9.0 + 5.0, 9.0, 3, 0.55 );
  float bright = mix( b0, b1, 0.34 );
  bright = 0.5 + ( bright - 0.5 ) * 2.70;

  // ── roughness ─────────────────────────────────────────────────────────────
  // A ridged component is mixed in deliberately: roughness in the real world
  // is not symmetric noise. Surfaces have *patches* that are markedly smoother
  // than their surroundings — a polished traffic path, a damp corner, a grease
  // film — separated by broad areas at the base value. Ridged noise has that
  // sparse-extreme shape; plain fbm does not.
  float r0 = apxFbmValue( uv * 5.0 + 31.0, 5.0, 3, 0.55 );
  float r1 = apxRidged( uv * 13.0 + 61.0, 13.0, 3, 0.50 );
  float rough = mix( r0, r1, 0.42 );
  rough = 0.5 + ( rough - 0.5 ) * 3.00;

  // ── tint ──────────────────────────────────────────────────────────────────
  // Lowest frequency of the three: sun-bleached vs shadow-damp is a
  // building-scale property, not a per-panel one.
  float t0 = apxFbm( uv * 2.0 + 77.0, 2.0, 3, 2.0, 0.62 );
  float t1 = apxFbmValue( uv * 7.0 + 113.0, 7.0, 2, 0.5 );
  float tint = mix( t0, t1, 0.28 );
  tint = 0.5 + ( tint - 0.5 ) * 2.30;

  s.albedo = clamp( vec3( bright, rough, tint ), vec3( 0.0 ), vec3( 1.0 ) );
}
`;
