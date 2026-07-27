/**
 * NOISEFIELD — ground surfaces.
 *
 * sand · dirt · asphalt
 *
 * Ground is seen at extreme grazing angles more than any other surface, and it
 * is the surface the player is closest to at all times — the bottom third of
 * every frame is floor at one to three metres. Two things therefore dominate:
 *
 *   (a) A COMPLETE SCALE LADDER. Every octave from metre-scale undulation down
 *       to the finest thing the map can hold must be occupied, with nothing
 *       missing in between. A ladder with a gap reads as "smeared clay": the
 *       eye sees big soft shapes, sees fine noise, and finds nothing at the
 *       10-100 mm scale it uses to judge what a material actually is.
 *
 *   (b) HEIGHT / ALBEDO CORRELATION. Granular ground is darker in its hollows,
 *       because fine dark material washes into them and the crests are
 *       wind-sorted to the palest, driest grains. This is what couples the lit
 *       relief to the colour; without it the normal map and the albedo read as
 *       two unrelated layers stacked on the same polygon, which is the single
 *       most common reason procedural ground looks fake.
 *
 * Where the ladder stops: a 1024^2 map over a 4 m tile has 3.9 mm texels, so
 * its Nyquist limit is ~8 mm and its usable limit ~20 mm. Everything finer is
 * supplied by the shared detail field, whose normal AND height (packed in
 * alpha) are sampled at 14-18x the base rate — see `detail.ts`. The two bands
 * are chosen to meet with overlap, not with a gap.
 *
 * sand and dirt additionally opt into triplanar projection so they cannot
 * stretch on sloped terrain regardless of how the world's UVs were authored,
 * and into the macro-driven UV warp, which bends the tile lattice out of
 * alignment on the long sightlines where a repeat would otherwise be obvious.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * (c) ALBEDO IS MEASURED, NOT EYEBALLED — AND IT IS MEASURED IN LINEAR SPACE
 * ─────────────────────────────────────────────────────────────────────────────
 * Sun-baked desert ground has a diffuse reflectance of roughly 0.28-0.45 in
 * LINEAR light (sRGB ~144-179). Every number in this file is authored as an
 * sRGB triple through `apxHex`, which converts once, so the constants read like
 * a colour picker while the shader works in linear.
 *
 * The trap is that this material is a stack of ~8 `mix()` layers, and every
 * layer that multiplies by a one-sided mask quietly removes a fraction of a
 * stop. Three of those in series turn a plausible palette into wet mud: `dirt`
 * measured (0.169, 0.112, 0.062) linear — sRGB (115, 95, 72), a dark damp-soil
 * brown — from a palette whose own midpoint was already only 0.155. The MEAN of
 * the baked texture is the acceptance criterion, not the palette constants, and
 * it is verified by rendering the map into a linear render target and averaging
 * the readback. Damp hollows and crack interiors are allowed to go far darker;
 * the mean is what must land in range.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * (d) ROUGHNESS HAS A FLOOR, BECAUSE THE NORMAL FIELD IS DELIBERATELY LOUD
 * ─────────────────────────────────────────────────────────────────────────────
 * Ground carries the highest-frequency normal detail of any surface here (the
 * base map plus a shared detail tap at 15-18x the base UV rate). Specular
 * aliasing scales as normal-variance-per-pixel over roughness squared, so a
 * low-roughness texel under that field does not read as "polished" — it reads
 * as a white sparkle that crawls when the camera moves, i.e. wet glitter on
 * what is supposed to be dry dust.
 *
 * Two things therefore have hard limits here rather than artistic ones:
 *
 *   • the `clamp()` floor on `s.rough`, and the smoothest target any mask is
 *     allowed to reach. Dry granular ground is never below ~0.55 perceptual
 *     roughness; even damp sand is ~0.55-0.60, not 0.30.
 *   • `macroRough` in the recipe tuning, which is an ABSOLUTE offset applied on
 *     top of this texture at shading time. macroRough 0.28 at macroStrength
 *     0.58 subtracts up to 0.16 from whatever is baked here, so the floor in
 *     this file is not the floor the shader sees — the two have to be budgeted
 *     together. Effective minimum = floor - macroRough * macroStrength
 *                                         - 0.15 * detailAlbedo.
 */

import type { SurfaceRecipe } from './Recipe';

// ─────────────────────────────────────────────────────────────────────────────
// Sand
// ─────────────────────────────────────────────────────────────────────────────

const SAND_GLSL = /* glsl */ `
/**
 * Wind ripples.
 *
 * A real ripple field has LONG, COHERENT crest lines that wander only slowly
 * across their own direction. The phase warp must therefore be strongly
 * ANISOTROPIC — high period across the ripple, low along it. An isotropic warp
 * turns the field into the brain-coral maze that gives away every procedural
 * sand shader, which is exactly what the previous revision of this file did.
 *
 * TILING: 'waves' must be an integer AND a multiple of 4, because the crest
 * direction is sheared by uv.x * 0.25 and the phase must still advance by a
 * whole number of cycles across the tile in BOTH axes.
 */
float nfRipples( vec2 uv, float waves, float bend ) {
  vec2 per = vec2( 5.0, 2.0 );
  float phase = ( apxFbm( uv * per, per, 3, 2.0, 0.55 ) - 0.5 ) * bend;
  float t = uv.y + uv.x * 0.25 + phase;
  float w = sin( t * waves * APX_TAU );
  // Asymmetric profile: windward slope long and shallow, lee slope short and
  // steep. Raising a shifted sine to a power is the cheap way to get that.
  return pow( apxSat( w * 0.5 + 0.5 ), 1.7 );
}

void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. metre-scale undulation (4 m tile: ~2 m and ~57 cm) ─────────────────
  float dune  = apxFbm( apxWarp( uv * 2.0, 2.0, 0.45, 3 ), 2.0, 4, 2.0, 0.55 );
  float scuff = apxFbmValue( uv * 7.0, 7.0, 4, 0.55 );   // footprint / churn

  // ── 2. wind ripples: two crossed generations (~11 cm and ~4.5 cm) ─────────
  // One ripple scale reads as corrugated iron. Two, crossed and unequally
  // weighted, read as sand — the fine set breaks the coarse set's crests into
  // the scalloped edge that a real ripple has.
  float rip1 = nfRipples( uv, 36.0, 0.30 );
  float rip2 = nfRipples( uv.yx + 7.0, 88.0, 0.18 );
  // Ripples only survive on undisturbed sand. Churned ground is smooth in the
  // ripple sense and rough in the grain sense, which is a real and very
  // readable distinction.
  float ripMask = smoothstep( 0.26, 0.74, apxFbmValue( uv * 3.0 + 13.0, 3.0, 3, 0.6 ) );
  float calm = ripMask * ( 1.0 - apxSat( ( scuff - 0.55 ) * 2.4 ) );
  float rip = ( rip1 * 0.70 + rip2 * 0.30 ) * calm;

  // ── 3. small stones and gravel lag (~3 cm and ~1.6 cm) ────────────────────
  // Wind removes the fines and leaves a "desert pavement" of stones behind, so
  // stones cluster in the DEFLATED (low) areas rather than scattering evenly.
  vec3  st    = apxWorley( uv * 130.0, 130.0, 1.0 );
  float lag   = smoothstep( 0.40, 0.75, 1.0 - dune );
  float stone = apxSat( 1.0 - st.x * 4.2 ) * step( 0.80, st.z ) * ( 0.35 + 0.65 * lag );
  vec3  stoneCol = mix( apxHex( 154.0, 140.0, 116.0 ), apxHex( 92.0, 82.0, 70.0 ), fract( st.z * 6.1 ) );

  // Coarse grain clusters at the very limit of what this map can hold. Below
  // this the detail field takes over; the two bands deliberately overlap.
  float grainMid = 1.0 - apxFbmWorley( uv * 150.0, 150.0, 2, 0.5, 1.0 );

  // ── 4. damp patches ───────────────────────────────────────────────────────
  // Sand that still holds moisture is darker, MUCH smoother and slightly
  // cooler. The roughness delta (0.95 -> 0.42) is the point: it is the only
  // thing in this material that produces a specular highlight, and one damp
  // hollow catching the sun sells the whole surface as a real substance.
  float damp = apxSat( ( apxFbm( uv * 3.0 + 44.0, 3.0, 4, 2.0, 0.6 ) - 0.60 ) * 3.4 );
  // Water pools in hollows, so bias the damp mask by the undulation.
  damp *= smoothstep( 0.62, 0.25, dune );

  // Dry crust: sun-baked sand forms a pale, slightly harder skin that cracks
  // into plates on the high, exposed drifts. The opposite end of the same axis.
  float crust = apxSat( ( apxFbmValue( uv * 5.0 + 91.0, 5.0, 3, 0.55 ) - 0.48 ) * 3.0 )
              * smoothstep( 0.45, 0.85, dune );

  // ── albedo ────────────────────────────────────────────────────────────────
  // Two sand tones rather than one flat palette entry: quartz-rich pale buff
  // and an iron-stained warmer ochre, mixed at the drift scale. A single hue
  // with only value variation is what makes procedural sand read as grey.
  vec3 pale  = apxHex( 214.0, 194.0, 152.0 );
  vec3 ochre = apxHex( 188.0, 156.0, 108.0 );
  vec3 dry = mix( ochre, pale, apxSat( dune * 1.25 - 0.12 ) );
  dry = mix( dry, dry * vec3( 1.04, 1.00, 0.92 ), scuff * 0.45 );
  // Ripple crests are wind-sorted: lighter, finer material sits on top.
  //
  // Centred on the ripple field's ACTUAL mean (~0.18 after the calm mask), not
  // on 0.5. mix(a*0.9, a*1.1, rip) looks symmetric but is not: rip spends most
  // of its time near zero, so it applied a flat 10% darkening to the whole
  // surface and only the crests came back. Every one-sided mask multiplied
  // into albedo has to be re-centred like this or the material quietly loses
  // a stop of brightness per layer.
  dry *= 1.0 + ( rip - 0.18 ) * 0.42;
  dry = mix( dry, dry * 1.09 + apxHex( 18.0, 16.0, 12.0 ), crust * 0.55 );

  vec3 albedo = mix( dry, stoneCol, stone * 0.88 );
  albedo = mix( albedo, albedo * vec3( 0.64, 0.63, 0.66 ), damp * 0.85 );

  // ── height ────────────────────────────────────────────────────────────────
  //
  // AMPLITUDES ARE PHYSICALLY PROPORTIONED, and that is not a nicety.
  //
  // One normalStrength converts this whole field to slope, so the octaves have
  // to be in the ratio real sand has, or whichever one is over-amplitude
  // saturates the derived normal and drowns the rest. The previous revision
  // gave the ripples 0.20 of the range at an 11 cm period — six times their
  // real relief — which drove them past 80 degrees, turned them into hard
  // black/white zebra stripes at mid distance, and left the dune undulation
  // invisible underneath.
  //
  // Taking the full 0..1 range as ~30 cm of relief over a 4 m tile:
  //   dune      +-60 mm over 2 m      -> 0.40
  //   scuff     +-25 mm over 57 cm    -> 0.16
  //   ripple    +-5  mm over 11 cm    -> 0.075  (still ~2x exaggerated)
  //   grain     +-1.5 mm over 2.7 cm  -> 0.028
  //   stone      +4  mm proud         -> 0.016
  //
  // The stone term was 0.030, and it is the steepest thing in this stack by a
  // wide margin: apxSat( 1 - F1 * 4.2 ) at period 130 has |dh/duv| = 0.030 x
  // 4.2 x 130 = 16.4, which at normalStrength 0.30 is atan(4.9) = 78 degrees on
  // a mask that covers a fifth of the surface. Steep AND dense is what aliases;
  // steep and sparse (a bughole) does not. Halved.
  float h = 0.50
          + ( dune - 0.5 ) * 0.40
          + ( scuff - 0.5 ) * 0.16
          + ( rip - 0.18 ) * 0.075
          + ( grainMid - 0.5 ) * 0.028
          + stone * 0.016
          - crust * 0.012;
  s.height = clamp( h, 0.0, 1.0 );

  // ── height / albedo correlation ───────────────────────────────────────────
  // See the file header. Applied AFTER the height is final so it tracks every
  // contributing octave, including the stones and the ripple crests.
  // Range chosen so the MEAN is ~1.0: an asymmetric range here is another
  // quiet stop of brightness loss, and three of those in series is what turns
  // pale desert sand into wet mud.
  albedo *= mix( 0.80, 1.16, smoothstep( 0.20, 0.82, s.height ) );

  // ── roughness / metalness ─────────────────────────────────────────────────
  float rough = 0.95 + ( grainMid - 0.5 ) * 0.10;
  // 0.60, not 0.54. Damp sand is still the specular story — it is the only
  // thing here that forms a highlight at all — but a broad sheen is what damp
  // sand actually looks like, and 0.54 under an 18x detail normal produced a
  // field of crawling white glints instead. See note (d) in the file header.
  rough = mix( rough, 0.60, damp );              // the specular story
  rough = mix( rough, 0.68, stone * 0.85 );      // polished stone faces
  rough = mix( rough, 0.80, crust * 0.7 );       // crust is slightly burnished
  rough = mix( rough, 0.99, apxSat( ( scuff - 0.6 ) * 2.5 ) * 0.6 );
  // FLOOR 0.52, was 0.30. With macroRough 0.15 x macroStrength 0.58 and the
  // grain wobble on top, the shader sees a minimum of ~0.39 — comfortably out
  // of the glitter regime (< ~0.25) and still soft enough to catch the sun.
  s.rough = clamp( rough, 0.52, 1.0 );
  s.metal = 0.0;

  float ao = smoothstep( 0.16, 0.72, s.height );
  s.ao = clamp( mix( 0.55, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Dirt / dry earth
// ─────────────────────────────────────────────────────────────────────────────

const DIRT_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. clod structure (3 m tile: ~33 cm and ~9 cm) ────────────────────────
  // Compacted earth breaks into clods, not smooth waves, so the base is a
  // warped worley field rather than fbm. Two generations: the big turned clods
  // and the crumb they break down into.
  vec2  cw     = apxWarp( uv * 9.0, 9.0, 0.45, 3 );
  float clod   = 1.0 - apxWorley( cw, 9.0, 1.0 ).x;
  float crumb  = 1.0 - apxWorley( apxWarp( uv * 34.0, 34.0, 0.30, 2 ), 34.0, 1.0 ).x;
  float macro  = apxFbm( uv * 3.0, 3.0, 4, 2.0, 0.55 );
  float mid    = apxFbmValue( uv * 14.0, 14.0, 4, 0.55 );

  // ── 2. dried-mud polygon cracks ───────────────────────────────────────────
  // Constant-width Voronoi edges. Held to a MINORITY of the surface: a crack
  // network that covers the whole tile is the loudest possible repeat, because
  // its cells are all the same size and the eye counts them.
  float dry   = apxVoronoiEdge( apxWarp( uv * 6.0, 6.0, 0.36, 3 ), 6.0, 0.9 );
  // The transition width is a SLOPE BUDGET, not a look control. At (0.0, 0.042)
  // the 0.24 height drop below fell across 0.007 uv, i.e. |dh/duv| = 34, which
  // at normalStrength 0.45 is atan(15) = 86 degrees: a saturated normal, and
  // the reason the derived map measured 0.063-0.957 in X even after a 16:1 mip
  // average. Widened and started off zero so the crack still has a hard read
  // without a vertical wall on either side of it.
  float crack = 1.0 - smoothstep( 0.010, 0.070, dry );
  crack *= smoothstep( 0.44, 0.74, apxFbm( uv * 3.0 + 88.0, 3.0, 3, 2.0, 0.6 ) );

  // ── 3. pebbles, gravel and organic debris ─────────────────────────────────
  // Three grades. A single pebble population reads as a polka dot; a graded
  // one reads as ground.
  vec3  pb     = apxWorley( uv * 46.0, 46.0, 1.0 );
  float pebble = apxSat( 1.0 - pb.x * 4.2 ) * step( 0.76, pb.z );
  // Lifted with the earth palette: desert-pavement gravel is sun-bleached, and
  // leaving these at their old values would have made the stones read as wet
  // pebbles scattered over dry dust. Still DARKER than the matrix — stones
  // genuinely are — just not by a factor of four.
  vec3  pebCol = mix( apxHex( 166.0, 155.0, 136.0 ), apxHex( 112.0, 102.0, 90.0 ), fract( pb.z * 4.7 ) );

  vec3  gv     = apxWorley( uv * 118.0, 118.0, 1.0 );
  float grit   = apxSat( 1.0 - gv.x * 5.0 ) * step( 0.58, gv.z );

  vec3  db     = apxWorleyAniso( uv, 46.0, 14.0, 1.0 );
  float debris = apxSat( 1.0 - db.x * 7.0 ) * step( 0.90, db.z );

  // ── 4. traffic polish ─────────────────────────────────────────────────────
  // Feet and tyres compact a path: the clods are crushed flat, the surface
  // goes darker and markedly smoother. This is the roughness story for dirt.
  float path = apxSat( ( apxFbm( apxWarp( uv * 2.0, 2.0, 0.6, 3 ), 2.0, 3, 2.0, 0.6 ) - 0.54 ) * 3.6 );

  // ── albedo ────────────────────────────────────────────────────────────────
  // SUN-BAKED, NOT WET. Target: a baked-texture MEAN of 0.28-0.45 in LINEAR
  // light (sRGB ~144-179) — see note (c) in the file header.
  //
  // The old pair was sRGB (128,104,76) / (86,68,50), i.e. linear (0.216,0.138,
  // 0.072) and (0.093,0.058,0.032): a midpoint of 0.155 BEFORE the eight mix()
  // layers below took their cut, which is how the map measured (0.169, 0.112,
  // 0.062) — dark wet-soil brown. These two are (0.527,0.440,0.314) and
  // (0.220,0.181,0.128) linear, a midpoint of 0.373, which survives the stack
  // as a measured mean of ~0.36 / 0.30 / 0.21.
  //
  // Hue: R/B falls from 2.7 to 1.7. Dry desert ground is a DESATURATED warm
  // tan-grey — the saturated orange-brown of the old palette is what wet iron-
  // rich soil looks like, not what sun-bleached dust looks like.
  vec3 earth = mix( apxHex( 192.0, 177.0, 152.0 ), apxHex( 129.0, 118.0, 100.0 ), mid );
  earth *= mix( 0.80, 1.20, macro );
  earth = mix( earth, apxPalDust(), apxSat( ( clod - 0.55 ) * 2.0 ) * 0.32 );

  vec3 albedo = mix( earth, pebCol, pebble * 0.85 );
  albedo = mix( albedo, pebCol * 1.06, grit * 0.35 );
  albedo = mix( albedo, apxHex( 124.0, 110.0, 74.0 ), debris * 0.6 );
  albedo = mix( albedo, earth * 0.76, path * 0.55 );
  // Crack interiors are in shadow and hold darker, damper material.
  albedo = mix( albedo, earth * 0.38, crack * 0.85 );

  // ── height ────────────────────────────────────────────────────────────────
  // AMPLITUDES ARE A SLOPE BUDGET. The three cellular terms are the dangerous
  // ones because they are simultaneously steep and area-dense: 'grit' is
  // sat(1 - 5*F1) at period 118, so its slope is amplitude x 5 x 118 — at the
  // old 0.014 that is |dh/duv| = 8.3, atan(3.7) = 75 degrees, over 42% of the
  // surface. 'pebble' was 74 degrees over 24%. That, plus the crack wall above,
  // is the high-frequency normal energy that was aliasing into white sparkle.
  // Each is cut to roughly the relief it actually has on a 3 m tile.
  float h = 0.55
          + ( macro - 0.5 ) * 0.20
          + ( clod - 0.5 ) * 0.17 * ( 1.0 - path * 0.75 )
          + ( crumb - 0.5 ) * 0.040 * ( 1.0 - path * 0.6 )
          + ( mid - 0.5 ) * 0.07
          + pebble * 0.024
          + grit * 0.006
          + debris * 0.008
          - path * 0.030
          - crack * 0.11;
  s.height = clamp( h, 0.0, 1.0 );

  // Hollows hold the fines and the damp; the crushed high spots are dusty.
  albedo *= mix( 0.80, 1.14, smoothstep( 0.22, 0.80, s.height ) );

  float rough = 0.96 + ( mid - 0.5 ) * 0.06;
  rough = mix( rough, 0.70, pebble * 0.85 );     // pebbles are smoother
  rough = mix( rough, 0.64, path * 0.8 );        // compacted path is polished
  rough = mix( rough, 0.99, crack );
  // FLOOR 0.55, was 0.34. Effective shader minimum after macroRough 0.16 x
  // macroStrength 0.58 and the grain wobble is ~0.42. See note (d).
  s.rough = clamp( rough, 0.55, 1.0 );
  s.metal = 0.0;

  float ao = smoothstep( 0.20, 0.68, s.height );
  ao = mix( ao, ao * 0.42, crack );
  s.ao = clamp( mix( 0.45, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Asphalt / road
// ─────────────────────────────────────────────────────────────────────────────

const ASPHALT_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. graded aggregate in bitumen ────────────────────────────────────────
  // Asphalt is exposed stone glued together. The chips are 5-20 mm; on a 3 m
  // tile that is periods of 150-600, NOT the 34-110 an earlier revision used
  // (which produced 30-90 mm boulders and read as crazy paving). 1024 texels
  // over 3 m is 2.9 mm, so 100/190/330 is as fine as this map can go before
  // the top grade starts to alias.
  vec3  a1 = apxWorley( uv * 100.0, 100.0, 1.0 );
  vec3  a2 = apxWorley( uv * 190.0, 190.0, 1.0 );
  vec3  a3 = apxWorley( uv * 330.0, 330.0, 1.0 );

  float chip1 = apxSat( 1.0 - a1.x * 2.6 ) * step( 0.42, a1.z );
  float chip2 = apxSat( 1.0 - a2.x * 3.0 ) * step( 0.50, a2.z );
  float chip3 = apxSat( 1.0 - a3.x * 3.4 ) * step( 0.55, a3.z );
  float chips = apxSat( chip1 * 0.52 + chip2 * 0.30 + chip3 * 0.22 );

  // Aged, not fresh-laid. A new wearing course is ~0.05 linear and this map
  // measured 0.057; bitumen in permanent desert sun oxidises grey and holds a
  // permanent film of blown sand, which puts a real road nearer 0.10-0.13
  // linear. Still by far the darkest surface in the compound — asphalt is
  // deliberately NOT brought into the 0.28-0.45 dry-earth band, because it is
  // not dry earth. (It is also currently unreferenced by the level's material
  // buckets; this is correctness, not a visible fix.)
  vec3 binder = apxHex( 68.0, 66.0, 64.0 );
  vec3 stone  = mix( apxHex( 152.0, 147.0, 138.0 ), apxHex( 94.0, 90.0, 85.0 ), fract( a1.z * 7.7 ) );

  // Ravelling: where the binder has aged out, more stone is exposed and the
  // surface goes pale and open. Where it is fresh, the chips are drowned.
  float wear = apxFbm( apxWarp( uv * 3.0, 3.0, 0.5, 3 ), 3.0, 4, 2.0, 0.55 );
  vec3 albedo = mix( binder, stone, chips * mix( 0.30, 1.00, wear ) );

  // ── 2. tar patches ────────────────────────────────────────────────────────
  // Crack-sealing tar: near-black, much smoother, and it drowns the aggregate.
  vec2  tw  = apxWarp( uv * 2.0, 2.0, 0.7, 3 );
  float tar = smoothstep( 0.60, 0.70, apxFbm( tw, 2.0, 4, 2.0, 0.6 ) );
  albedo = mix( albedo, apxHex( 44.0, 43.0, 42.0 ), tar * 0.9 );

  // ── 3. cracks ─────────────────────────────────────────────────────────────
  float e = apxVoronoiEdge( apxWarp( uv * 5.0, 5.0, 0.35, 3 ), 5.0, 0.9 );
  float crack = ( 1.0 - smoothstep( 0.0, 0.026, e ) )
              * smoothstep( 0.42, 0.68, apxFbm( uv * 3.0 + 17.0, 3.0, 3, 2.0, 0.6 ) )
              * ( 1.0 - tar * 0.7 );
  albedo = mix( albedo, vec3( 0.010 ), crack * 0.9 );

  // ── 4. polished wheel path and oil ────────────────────────────────────────
  // The two wheel tracks are burnished almost to a gloss while the crown and
  // the gutter stay open and matte. This is by far the strongest roughness
  // signal on a real road, and it needs no colour change at all to read.
  float polish = apxSat( ( apxFbm( apxWarp( uv * vec2( 3.0, 1.0 ), vec2( 3.0, 1.0 ), 0.35, 3 ), vec2( 3.0, 1.0 ), 3, 2.0, 0.6 ) - 0.52 ) * 4.0 );
  float oil = apxSat( ( apxFbmValue( uv * 4.0 + 200.0, 4.0, 4, 0.6 ) - 0.62 ) * 4.5 );
  albedo = mix( albedo, albedo * 0.32, oil * 0.85 );

  // ── 5. faded lane marking ─────────────────────────────────────────────────
  float stripe = apxBand( uv.x, 0.44, 0.56, 0.012 );
  float paintWear = smoothstep( 0.30, 0.72, apxFbmValue( uv * 12.0 + 5.0, 12.0, 4, 0.55 ) );
  float paint = stripe * paintWear * ( 1.0 - tar ) * ( 1.0 - crack );
  albedo = mix( albedo, apxHex( 200.0, 190.0, 162.0 ), paint * 0.80 );

  // ── 6. sand blown across the road ─────────────────────────────────────────
  float sandFilm = apxSat( ( apxFbmValue( uv * 4.0 + 300.0, 4.0, 3, 0.6 ) - 0.5 ) * 2.2 );
  albedo = mix( albedo, apxPalSand() * 0.82, sandFilm * 0.26 );

  float h = 0.55
          + chips * 0.075
          - ( 1.0 - chips ) * 0.018
          - crack * 0.20
          + tar * 0.03
          - polish * 0.020
          + paint * 0.012;
  s.height = clamp( h, 0.0, 1.0 );

  // Grit and dust collect between the chips, so the interstices are darker
  // than the exposed stone faces by more than the aggregate mask alone says.
  albedo *= mix( 0.84, 1.12, smoothstep( 0.34, 0.74, s.height ) );

  float rough = 0.93;
  rough = mix( rough, 0.70, chips * 0.60 );     // polished stone faces
  rough = mix( rough, 0.40, tar );              // fresh tar is glossy
  rough = mix( rough, 0.34, polish * 0.85 );    // burnished wheel path
  rough = mix( rough, 0.28, oil );              // oil is the glossiest thing here
  rough = mix( rough, 0.99, crack );
  rough = mix( rough, 0.85, paint );
  rough = mix( rough, 0.96, sandFilm * 0.65 );
  // FLOOR 0.22, was 0.14. Asphalt genuinely does polish to a near-gloss in the
  // wheel paths, but 0.14 sits under a detail normal running at 16x the base UV
  // rate, and that combination aliases into sparkle rather than reading as
  // polish. See note (d) in the file header.
  s.rough = clamp( rough, 0.22, 1.0 );
  s.metal = 0.0;

  float ao = smoothstep( 0.28, 0.68, s.height );
  ao = mix( ao, ao * 0.40, crack );
  s.ao = clamp( mix( 0.42, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Recipes
// ─────────────────────────────────────────────────────────────────────────────

export const GROUND_RECIPES: SurfaceRecipe[] = [
  {
    id: 'sand',
    surface: 'sand',
    tileMeters: 4.0,
    size: 1024,
    ormDivisor: 2,
    seed: 71.0,
    glsl: SAND_GLSL,
    note: 'crossed wind ripples, desert-pavement stones, damp hollows; triplanar + UV warp',
    // Triplanar: terrain UVs are the classic unreliable case, and a homogeneous
    // granular material is exactly where world-space projection is free of
    // downsides. 4 m per tile keeps the ripple spacing at a believable 11 cm.
    tuning: {
      // 0.30, not 0.55: with the amplitudes now physically proportioned the
      // same strength would over-drive the dunes. See the height note in
      // SAND_GLSL.
      normalStrength: 0.30,
      // 18x on a 4 m tile puts the detail field's own 5-period octave at 11 cm
      // and its top octave at 1.4 mm: the whole band the base map cannot hold.
      detailScale: 18.0,
      // 0.40, was 0.62. The detail tap is the HIGHEST-frequency normal energy
      // on the surface the player is closest to and sees most obliquely, and at
      // 0.62 it was the dominant term in the per-pixel normal variance that
      // drives specular aliasing. Cutting the tangent XY by a third drops that
      // variance by ~57% (it goes as the square) while the grain is still
      // clearly present at 30 cm. See note (d) in the file header.
      detailStrength: 0.40,
      // No second tap. The base map already occupies 20 mm - 4 m, so a coarse
      // detail tap would only duplicate it, and on the triplanar path it would
      // cost three extra fetches on the single most expensive material here.
      detailScale2: 0.0,
      detailStrength2: 0.0,
      detailAlbedo: 0.26,
      macroScale: 0.075,
      macroStrength: 0.58,
      // 0.15, was 0.26. macroRough is an ABSOLUTE roughness offset applied on
      // top of the baked map, so it sets how far below this material's floor
      // the shader can actually go: 0.26 x 0.58 = 0.151, which took the old
      // 0.30 floor down to 0.11 — a mirror, on granular sand. Now 0.087, for an
      // effective minimum of ~0.39.
      macroRough: 0.15,
      uvWarp: 0.55,
      uvWarpScale: 0.13,
      triplanar: true,
      triplanarScale: 4.0,
    },
  },
  {
    id: 'dirt',
    surface: 'dirt',
    tileMeters: 3.0,
    size: 1024,
    ormDivisor: 2,
    seed: 89.0,
    glsl: DIRT_GLSL,
    note: 'clods and crumb, graded pebbles, compacted traffic path; triplanar + UV warp',
    tuning: {
      // 0.34, was 0.45. The base height field is a stack of cellular terms with
      // hard edges; at 0.45 the derived normal saturated (X ranged 0.06-0.96
      // even after a 16:1 mip average, i.e. facets past 80 degrees). The height
      // amplitudes above were cut at the same time — both were needed, because
      // slope is amplitude x frequency x strength and only the product matters.
      normalStrength: 0.34,
      detailScale: 15.0,
      // 0.40, was 0.62 — same reasoning as sand.
      detailStrength: 0.40,
      detailScale2: 0.0,
      detailStrength2: 0.0,
      detailAlbedo: 0.26,
      macroScale: 0.085,
      macroStrength: 0.58,
      // 0.16, was 0.28: 0.28 x 0.58 = 0.162 took the old 0.34 floor to 0.14.
      macroRough: 0.16,
      uvWarp: 0.50,
      uvWarpScale: 0.14,
      triplanar: true,
      triplanarScale: 3.0,
    },
  },
  {
    id: 'asphalt',
    surface: 'concrete',
    tileMeters: 3.0,
    size: 1024,
    ormDivisor: 2,
    seed: 103.0,
    glsl: ASPHALT_GLSL,
    note: 'graded 5-20 mm aggregate, burnished wheel path, tar, oil, faded marking',
    tuning: {
      normalStrength: 0.52,
      detailScale: 16.0,
      detailStrength: 0.75,
      detailScale2: 3.2,
      detailStrength2: 0.35,
      detailAlbedo: 0.24,
      macroScale: 0.085,
      macroStrength: 0.58,
      // 0.22, was 0.32. Budgeted against the 0.22 roughness floor: 0.32 x 0.58
      // took the shaded minimum to 0.14, which is a mirror under a 16x detail
      // normal. Now ~0.20.
      macroRough: 0.22,
      uvWarp: 0.40,
      uvWarpScale: 0.15,
    },
  },
];
