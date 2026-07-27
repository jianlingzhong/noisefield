/**
 * NOISEFIELD — masonry surfaces.
 *
 * concrete · plaster (stucco) · brick · cinderblock
 *
 * These four carry most of the compound's screen area, so they get the most
 * attention. The shared theme is: a base matrix, a discrete aggregate/unit
 * population on top, a joint/seam system that recesses the height field, and
 * then *damage and staining driven by that joint system* — which is the part
 * that separates a real material from coloured noise. Dirt does not appear at
 * random; it accumulates in cavities and runs downhill from horizontal edges.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * DO NOT NAME A LOCAL `patch`.
 * ─────────────────────────────────────────────────────────────────────────────
 * `patch` is a RESERVED WORD in GLSL ES 3.00 (it belongs to the tessellation
 * stage). three.js compiles every non-raw ShaderMaterial as `#version 300 es`,
 * so a `float patch = ...` here is a hard compile error:
 *
 *   ERROR: 'patch' : Illegal use of reserved word
 *
 * and the failure is SILENT from the bake's point of view — the draw is
 * skipped, the render target keeps its clear colour, and concrete and plaster
 * both shipped as pure black albedo AND pure black ORM (occlusion 0,
 * roughness 0, metalness 0) with a dead-flat derived normal. The repair mask
 * is therefore called `repair`. `TextureForge` now lints every baked shader
 * against the full ESSL3 reserved list so this class of bug cannot recur
 * quietly; see `RESERVED_ESSL3` there.
 */

import type { SurfaceRecipe } from './Recipe';

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

const GLSL_MASONRY_HELPERS = /* glsl */ `
/**
 * Horizontal form-board / course joint system.
 * @return x = joint mask (1 at the joint centre), already per-course randomised
 *         y = 0..1 "distance below the joint", used to drive runoff staining
 *
 * Tiles because 'courses' is an integer and the wobble depends only on u.
 *
 * THE PER-COURSE HASH IS NOT DECORATION. An unmodulated joint system is a
 * perfectly regular set of parallel lines, and on a floor — where this same
 * concrete is used, because the level has one concrete — a regular grid of
 * lines marching to the horizon is the single loudest tiling artefact a
 * material can produce. It was the most obvious defect in the shipped build.
 * Hashing presence and depth per course means roughly half the lines nearly
 * vanish and no two neighbours match, which reads as a real pour and gives the
 * eye nothing periodic to lock onto.
 */
vec2 nfJoints( vec2 uv, float courses, float width ) {
  float row = floor( uv.y * courses );
  float fy  = fract( uv.y * courses );
  // mod(row, courses) so the hash itself is periodic across the tile seam.
  float rnd = apxHash21( vec2( mod( row, courses ), 0.0 ) + 6.1 );
  float wob = ( apxFbm( vec2( uv.x * 10.0, 3.0 ), 10.0, 3, 2.0, 0.5 ) - 0.5 ) * 0.09;
  float c   = 0.5 + wob;
  // Per-course width and strength. step() keeps ~35% of courses effectively
  // closed (a well-made joint leaves almost no mark).
  float w = width * mix( 0.45, 1.55, fract( rnd * 7.7 ) );
  float strength = mix( 0.15, 1.0, smoothstep( 0.35, 0.85, fract( rnd * 3.3 ) ) );
  float joint = apxSat( 1.0 - abs( fy - c ) / max( w, 1e-4 ) ) * strength;
  // Runoff PEAKS at the joint and fades downhill, wrapping round to the next
  // joint up. fract() of a periodic quantity stays seamless, and its only
  // discontinuity sits exactly on the joint where the joint mask hides it.
  // (Sign matters: (c - fy)/c is the same curve upside down, i.e. staining
  //  strongest where the water has already run out.)
  float below = 1.0 - fract( c - fy );
  return vec2( joint, below );
}

/**
 * Crack network. Voronoi EDGES (constant width, unlike F2-F1) gated by a
 * low-frequency mask so only part of the wall has cracked, and thinned by a
 * second high-frequency term so the lines taper out instead of ending abruptly.
 */
float nfCracks( vec2 uv, float scale, float width, float coverage ) {
  vec2 q = apxWarp( uv * scale, scale, 0.35, 3 );
  float e = apxVoronoiEdge( q, scale, 0.85 );
  float line = 1.0 - smoothstep( 0.0, width, e );
  float gate = smoothstep( coverage - 0.18, coverage + 0.18, apxFbm( uv * 4.0 + 61.0, 4.0, 3, 2.0, 0.6 ) );
  float taper = smoothstep( 0.30, 0.70, apxFbmValue( uv * 17.0, 17.0, 3, 0.5 ) );
  return line * gate * taper;
}

/**
 * Brick lattice shared by the brick recipe and by plaster's fall-away patches.
 * NOTE: 'count.y' must be EVEN, otherwise the half-brick row stagger does not
 * realign across the tile seam and you get a visible discontinuity.
 * @return xyz = brick albedo, w = surface height contribution (0..1)
 */
vec4 nfBrickLayer( vec2 uv, vec2 count, float mortarWidth ) {
  vec4 m = apxMasonry( uv, count, 0.5 );
  vec2 local = m.xy;
  float rnd = m.z;

  // Distance to the joint, in brick-local units, corrected for the cell's
  // aspect ratio so a 3:1 brick still gets an even-width mortar line.
  vec2 aspect = count / min( count.x, count.y );
  vec2 d = min( local, 1.0 - local ) * aspect.yx;
  float edge = min( d.x, d.y );
  float mortar = 1.0 - smoothstep( mortarWidth, mortarWidth * 2.2, edge );

  // Per-brick colour: fired clay varies from sandy buff to deep oxide red.
  vec3 clayA = apxHex( 168.0,  92.0,  62.0 );
  vec3 clayB = apxHex( 122.0,  66.0,  48.0 );
  vec3 clayC = apxHex( 186.0, 146.0, 104.0 );
  vec3 brick = mix( clayA, clayB, fract( rnd * 7.13 ) );
  brick = mix( brick, clayC, smoothstep( 0.72, 1.0, fract( rnd * 3.71 ) ) );

  // Within-brick mottle, offset per brick so neighbours are not clones.
  float mott = apxFbmValue( local * 5.0 + rnd * 40.0, 5.0, 3, 0.55 );
  brick *= mix( 0.84, 1.14, mott );

  // Spalling: a corner or face of the brick has flaked off, exposing a lighter,
  // rougher, unfired-looking interior. Gated hard so only a few bricks show it.
  float spallGate = step( 0.80, fract( rnd * 11.9 ) );
  float spall = apxSat( apxFbmValue( local * 3.0 + rnd * 17.0, 3.0, 3, 0.5 ) * 1.6 - 0.85 ) * spallGate;
  spall *= smoothstep( 0.55, 0.15, edge );  // strongest at the brick's edges
  brick = mix( brick, apxHex( 196.0, 168.0, 138.0 ), spall );

  vec3 mortarCol = apxHex( 176.0, 170.0, 156.0 ) * mix( 0.82, 1.08, apxFbmValue( uv * 40.0, 40.0, 3, 0.5 ) );
  vec3 col = mix( brick, mortarCol, mortar );

  // Efflorescence: soluble salts wick out through the mortar and bloom white.
  float efflo = apxSat( apxFbm( uv * 7.0 + 23.0, 7.0, 4, 2.0, 0.55 ) * 1.7 - 0.75 ) * ( 0.35 + 0.65 * mortar );
  col = mix( col, vec3( 0.78, 0.77, 0.73 ), efflo * 0.55 );

  float h = 0.62 - mortar * 0.30 - spall * 0.18 + ( mott - 0.5 ) * 0.03;
  return vec4( col, h );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Concrete — hero surface
// ─────────────────────────────────────────────────────────────────────────────

const CONCRETE_GLSL = /* glsl */ `
${GLSL_MASONRY_HELPERS}

void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. cement matrix ──────────────────────────────────────────────────────
  // Warped fbm rather than plain fbm: real pour discoloration has soft,
  // interpenetrating lobes, not the isotropic blobs raw fbm produces.
  float blotch = apxFbm( apxWarp( uv * 3.0, 3.0, 0.45, 3 ), 3.0, 4, 2.0, 0.55 );
  float mid    = apxFbmValue( uv * 11.0, 11.0, 4, 0.5 );
  float fines  = apxFbmValue( uv * 46.0, 46.0, 3, 0.5 );   // cement laitance

  // SUN-BLEACHED, because this material is the courtyard apron, the market
  // street and the north hardstanding as well as the walls — i.e. it is the
  // outdoor ground the player stands on, and outdoor ground in this compound
  // must measure 0.28-0.45 in LINEAR light (sRGB ~144-179).
  //
  // apxPalConcrete() is (0.252, 0.231, 0.185) linear: correct for shaded
  // structural concrete, and it baked to a measured mean of 0.268 — sRGB 141,
  // a mid grey that read as wet screed under a desert sun. Years of UV and
  // airborne lime carbonate a poured surface to a noticeably paler, chalkier
  // grey than the parent mix, so the base is lifted 1.20x and warmed very
  // slightly. Measured result: ~0.31 linear (sRGB ~150).
  vec3 cement = apxPalConcrete() * vec3( 1.22, 1.20, 1.16 );
  cement *= mix( 0.76, 1.20, blotch );
  cement = mix( cement, cement * vec3( 1.07, 1.00, 0.87 ), mid * 0.55 );
  cement *= mix( 0.94, 1.06, fines );

  // ── 2. graded aggregate ───────────────────────────────────────────────────
  // THREE populations at 22 / 12 / 6 mm on a 2.4 m tile. The previous revision
  // used 24 and 68 periods, i.e. 100 mm and 35 mm "aggregate" — boulders, not
  // concrete, and the reason the surface read as low-frequency mush. 2048
  // texels over 2.4 m is 1.17 mm, so 380 periods (6 mm) is the practical floor
  // before the top grade starts to alias; anything finer is the detail field's
  // job.
  vec3  agg     = apxWorley( uv * 110.0, 110.0, 1.0 );
  float exposed = smoothstep( 0.56, 0.72, agg.z );   // weathered proud
  float dome    = apxSat( 1.0 - agg.x * 2.4 );
  float proud   = dome * dome * exposed;
  float buried  = dome * ( 1.0 - exposed );

  vec3 stone = mix( apxHex( 206.0, 200.0, 186.0 ), apxHex( 104.0, 96.0, 88.0 ), fract( agg.z * 9.7 ) );

  vec3  agg2 = apxWorley( uv * 200.0, 200.0, 1.0 );
  float fine = apxSat( 1.0 - agg2.x * 3.2 ) * step( 0.45, agg2.z );
  vec3  agg3 = apxWorley( uv * 380.0, 380.0, 1.0 );
  float sandy = apxSat( 1.0 - agg3.x * 3.8 ) * step( 0.40, agg3.z );

  vec3 albedo = mix( cement, stone, proud * 0.88 + buried * 0.22 );
  albedo = mix( albedo, stone * 0.96, fine * 0.34 );
  albedo = mix( albedo, stone * 1.04, sandy * 0.20 );

  // ── 3. bugholes ───────────────────────────────────────────────────────────
  // Air trapped against the shutter face leaves small round voids. They are
  // THE signature of formed concrete at reading distance, they are absent from
  // every "concrete" noise preset, and they cost one worley lookup.
  vec3  bh   = apxWorley( uv * 150.0, 150.0, 1.0 );
  float hole = apxSat( 1.0 - bh.x * 6.5 ) * step( 0.90, bh.z );
  albedo = mix( albedo, albedo * 0.38, hole * 0.85 );

  // ── 4. form-board seams ───────────────────────────────────────────────────
  // Concrete poured against timber shuttering keeps the horizontal joint of
  // each board, plus a faint imprint of the board's own grain. 4 courses on a
  // 2.4 m tile = 600 mm boards. nfJoints randomises presence per course —
  // see the long note on that function; it is why this no longer reads as a
  // ruled grid when the same material is used as a floor.
  vec2  jn    = nfJoints( uv, 4.0, 0.030 );
  float joint = jn.x;
  float below = jn.y;
  float boardGrain = apxFbmValue( vec2( uv.x * 3.0, uv.y * 90.0 ), vec2( 3.0, 90.0 ), 3, 0.5 );
  albedo *= mix( 1.0, 0.97 + boardGrain * 0.06, 0.5 );
  albedo = mix( albedo, albedo * 0.84, joint * 0.7 );

  // ── 5. cracks with dirt accumulation ──────────────────────────────────────
  // Thinner and less widespread than before. A crack network that covers the
  // whole surface has uniform cell size, and uniform cell size is a repeat.
  float crack = nfCracks( uv, 9.0, 0.026, 0.30 );
  // Hairline second network at a finer scale for crazing.
  float craze = nfCracks( uv * 2.0 + 9.0, 19.0, 0.018, 0.50 ) * 0.45;
  float allCrack = apxSat( crack + craze );
  vec3 crackDirt = mix( apxPalShadow() * 2.2, apxPalRust() * 0.55, 0.35 );
  albedo = mix( albedo, crackDirt, allCrack * 0.80 );

  // ── 6. water staining ─────────────────────────────────────────────────────
  // Rain sheets off each board joint and runs down, depositing airborne dust.
  // The streak field is anisotropic (dense across, sparse along) so it reads as
  // flow rather than as noise.
  float streak = apxStreaks( uv, 38.0, 5.0, 4 );
  float stain  = apxSat( ( streak - 0.40 ) * 2.4 ) * below * ( 0.35 + 0.65 * joint );
  stain += apxSat( ( streak - 0.52 ) * 2.1 ) * below * 0.40;
  stain = apxSat( stain );
  albedo = mix( albedo, albedo * vec3( 0.58, 0.56, 0.55 ), stain * 0.78 );

  // ── 7. patch repairs ──────────────────────────────────────────────────────
  // A later, greyer, smoother mortar patch over spalled reinforcement, with a
  // hard edge where it was floated off. Deliberately a MINORITY of the surface
  // and deliberately hard-edged: the eye reads soft blends as lighting and
  // hard ones as history, and history is what makes a wall look built.
  float pf = apxFbm( apxWarp( uv * 2.0, 2.0, 0.55, 3 ), 2.0, 4, 2.0, 0.6 );
  // NOT named 'patch': that identifier is RESERVED in GLSL ES 3.00. See file header.
  float repair = smoothstep( 0.615, 0.640, pf );
  albedo = mix( albedo, apxHex( 150.0, 148.0, 142.0 ) * mix( 0.9, 1.1, mid ), repair * 0.88 );

  // ── 8. chipped edges ──────────────────────────────────────────────────────
  // Impact damage exposes lighter, un-weathered interior. Concentrated near the
  // board joints because that is where the arris is.
  float chipField = apxFbmValue( uv * 9.0 + 133.0, 9.0, 3, 0.5 );
  float chip = apxSat( ( chipField - 0.60 ) * 4.0 ) * smoothstep( 0.0, 0.35, joint );
  chip += apxSat( ( apxFbmValue( uv * 15.0 + 71.0, 15.0, 3, 0.5 ) - 0.74 ) * 5.0 ) * 0.6;
  chip = apxSat( chip );
  albedo = mix( albedo, apxHex( 182.0, 176.0, 164.0 ), chip * 0.8 );

  // ── 9. sun bleaching / dust film ──────────────────────────────────────────
  float dust = apxFbmValue( uv * 5.0 + 200.0, 5.0, 3, 0.6 );
  albedo = mix( albedo, apxPalDust() * 1.05, dust * 0.20 );

  // ── height ────────────────────────────────────────────────────────────────
  // Amplitudes proportioned to real relief on a 2.4 m tile, taking the 0..1
  // range as ~40 mm. A 22 mm aggregate dome stands ~2 mm proud (0.050); a
  // bughole is ~4 mm deep (0.090); a form-board step is ~2 mm (0.045). Getting
  // this ratio wrong is what saturates the derived normal and collapses the
  // whole surface into one screaming octave — see the same note in ground.ts.
  float h = 0.52
          + proud * 0.050
          + fine * 0.018
          + sandy * 0.007
          + ( blotch - 0.5 ) * 0.012
          + ( fines - 0.5 ) * 0.016
          - hole * 0.060
          - joint * 0.045
          - allCrack * 0.100
          - chip * 0.055
          + repair * 0.016;
  s.height = clamp( h, 0.0, 1.0 );

  // Airborne grit settles into every cavity, so the low ground is darker than
  // the cavity mask alone accounts for. Weak on concrete (it is not granular)
  // but it is what stops the aggregate reading as a printed pattern.
  albedo *= mix( 0.88, 1.08, smoothstep( 0.32, 0.72, s.height ) );

  // ── roughness ─────────────────────────────────────────────────────────────
  // Roughness variance is what sells concrete under a low sun. Polished
  // aggregate catches highlights; dirt in the cracks kills them dead. The
  // spread here is 0.30 -> 1.00 within one tile, and the macro field adds
  // another +-0.34 on top of that at building scale.
  float rough = 0.92;
  rough = mix( rough, 0.60, proud * 0.95 );       // wet-polished stone faces
  rough = mix( rough, 0.70, fine * 0.5 );
  rough = mix( rough, 0.99, allCrack );           // dirt-packed cracks
  rough = mix( rough, 0.56, stain * 0.65 );       // water-polished runoff
  rough = mix( rough, 0.96, chip );               // fresh fracture is matte
  rough = mix( rough, 0.99, hole * 0.8 );         // open pore
  rough = mix( rough, 0.72, repair * 0.7 );       // steel-floated repair
  rough += ( mid - 0.5 ) * 0.14;
  // FLOOR 0.45, was 0.26 — AND macroRough drops 0.32 -> 0.20 in the recipe.
  //
  // This material is not only walls: it is the courtyard apron, the market
  // street and the north hardstanding, i.e. the outdoor ground the player
  // stands on and views at grazing angles. It also carries the strongest detail
  // normal in the library (two taps at 13x and 2.6x). macroRough is an absolute
  // offset applied on top of what is baked here, so the OLD numbers gave the
  // shader a minimum roughness of 0.26 - 0.32*0.62 - 0.03 = 0.03: a mirror,
  // under the loudest normal field we generate, on a horizontal surface in
  // direct sun. That is a specular-aliasing generator, not a polished slab.
  // Now 0.45 - 0.20*0.62 - 0.03 = 0.30, which still reads as rain-polished
  // aggregate without resolving into crawling glints.
  s.rough = clamp( rough, 0.45, 1.0 );

  s.metal = 0.0;

  // ── ambient occlusion ─────────────────────────────────────────────────────
  // Cavity occlusion derived from the height field, plus explicit darkening in
  // the joints where a screen-space AO pass cannot reach at this scale.
  float ao = smoothstep( 0.24, 0.62, s.height );
  ao = mix( ao, ao * 0.55, allCrack );
  ao = mix( ao, ao * 0.72, joint );
  ao = mix( ao, ao * 0.30, hole );
  s.ao = clamp( mix( 0.42, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Plaster / stucco — hero surface, the dominant architectural finish
// ─────────────────────────────────────────────────────────────────────────────

const PLASTER_GLSL = /* glsl */ `
${GLSL_MASONRY_HELPERS}

void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. trowel-worked render ───────────────────────────────────────────────
  // A hand-floated finish is made of overlapping arcs. Two-level domain warping
  // on a ridged field reproduces the swept, directional look; plain fbm gives
  // the "orange peel" that reads as a bump-map preset.
  //
  // Two arc scales, because a plasterer works with two motions: long sweeps of
  // the float (period 4, ~75 cm) and short finishing strokes (period 15,
  // ~20 cm). One scale alone reads as marbling.
  vec2  sw     = apxWarp2( uv * 4.0, 4.0, 0.90, 4 );
  float swirl  = apxRidged( sw, 4.0, 4, 0.55 );
  vec2  sw2    = apxWarp2( uv * 15.0 + 3.0, 15.0, 0.55, 3 );
  float stroke = apxRidged( sw2, 15.0, 3, 0.50 );
  float coarse = apxFbmValue( uv * 8.0, 8.0, 4, 0.55 );

  // Sand aggregate in the render itself - stucco is a sand/lime mix. 90 and
  // 260 periods on a 3 m tile are 33 mm and 12 mm: the grade you actually see
  // in a rendered wall from two metres.
  float sand = apxFbmWorley( uv * 90.0, 90.0, 2, 0.5, 1.0 );
  float grit = 1.0 - sand;
  vec3  gv   = apxWorley( uv * 260.0, 260.0, 1.0 );
  float sharp = apxSat( 1.0 - gv.x * 4.4 ) * step( 0.62, gv.z );

  vec3 base = apxPalSand();
  base *= mix( 0.84, 1.14, coarse );
  base = mix( base, base * vec3( 1.04, 0.99, 0.91 ), swirl * 0.55 );
  base = mix( base, base * 1.05, stroke * 0.35 );
  base = mix( base, base * 1.10, sharp * 0.30 );

  // ── 2. blistering ─────────────────────────────────────────────────────────
  // Trapped moisture lifts the skin into small domes that later burst.
  vec3  bl      = apxWorley( uv * 18.0, 18.0, 1.0 );
  float blister = apxSat( 1.0 - bl.x * 3.2 ) * step( 0.78, bl.z );
  float burst   = blister * step( 0.90, fract( bl.z * 13.0 ) );

  // ── 3. fall-away patches exposing the brick behind ────────────────────────
  // This is the signature detail of a sun-baked compound wall. The patch mask
  // is warped so its border is ragged, and a slightly larger mask gives the
  // feathered "thin render" halo around the hole.
  vec2  pw     = apxWarp( uv * 3.0, 3.0, 0.60, 4 );
  float patchF = apxFbm( pw, 3.0, 4, 2.0, 0.55 );
  float hole   = smoothstep( 0.615, 0.680, patchF );
  float halo   = smoothstep( 0.560, 0.640, patchF ) - hole;

  // 13 stretchers x 40 courses on a 3 m tile = 231 x 75 mm, i.e. a real metric
  // brick plus its 10 mm joint. count.y is even, so the half-brick stagger
  // still realigns across the seam.
  vec4 brick = nfBrickLayer( uv, vec2( 13.0, 40.0 ), 0.085 );

  vec3 albedo = base;
  albedo = mix( albedo, albedo * mix( 0.95, 1.06, blister ), 0.6 );
  albedo = mix( albedo, brick.rgb, hole );
  // The halo is render that has thinned but not detached: the brick colour
  // bleeds through slightly and the edge is chalky where it fractured.
  albedo = mix( albedo, mix( base, brick.rgb, 0.30 ) * 1.06, halo );

  // ── 4. patch repairs ──────────────────────────────────────────────────────
  // Somebody has re-rendered part of this wall in a slightly greyer, slightly
  // smoother mix and not bothered to match it. Hard-edged on purpose: soft
  // blends read as lighting, hard ones read as history.
  float pf = apxFbm( apxWarp( uv * 2.0 + 41.0, 2.0, 0.60, 3 ), 2.0, 4, 2.0, 0.6 );
  // NOT named 'patch': that identifier is RESERVED in GLSL ES 3.00. See file header.
  float repair = smoothstep( 0.600, 0.628, pf ) * ( 1.0 - hole );
  vec3 repairCol = apxDesaturate( base, 0.45 ) * 0.94 + apxHex( 12.0, 12.0, 14.0 );
  albedo = mix( albedo, repairCol, repair * 0.85 );

  // ── 5. crazing + hairline cracks + dirt ───────────────────────────────────
  // Two crack generations: the shrinkage crazing that covers everything, and a
  // sparser set of real structural hairlines that run further and hold dirt.
  float craze = nfCracks( uv, 16.0, 0.020, 0.52 ) * ( 1.0 - hole );
  float hair  = nfCracks( uv * 0.5 + 5.0, 4.0, 0.012, 0.34 ) * ( 1.0 - hole );
  float cracks = apxSat( craze * 0.75 + hair );
  albedo = mix( albedo, albedo * 0.58, cracks * 0.72 );

  // Runoff below the wall's string course, and generic airborne dust that
  // settles more heavily where the surface is rough.
  vec2  jn     = nfJoints( uv, 2.0, 0.030 );
  float streak = apxStreaks( uv, 30.0, 4.0, 4 );
  float stain  = apxSat( ( streak - 0.42 ) * 2.5 ) * jn.y;
  // A second, longer-range streak set independent of the joints: rain running
  // off a parapet does not care where the string courses are.
  float streak2 = apxStreaks( uv, 17.0, 2.0, 4 );
  stain = apxSat( stain + apxSat( ( streak2 - 0.56 ) * 3.0 ) * 0.7 );
  albedo = mix( albedo, albedo * vec3( 0.60, 0.57, 0.53 ), stain * 0.65 );

  // Airborne dust settles into the open, porous areas of the render.
  //
  // It is tempting to put a splash-back band at the base of the wall with
  // fract(v * 1.0), but a period-1 fract has no interior counterpart: its step
  // is the only discontinuity in the whole texture and it lands EXACTLY on the
  // v seam, drawing a hard line across the hero surface every tile. A tile has
  // no "base" - height-dependent grime has to come from world space, not here.
  float dust = apxFbmValue( uv * 6.0 + 91.0, 6.0, 3, 0.6 );
  float dustMask = apxSat( ( dust - 0.34 ) * 1.7 ) * ( 0.40 + 0.60 * ( 1.0 - swirl ) );
  albedo = mix( albedo, apxPalDust() * 0.92, dustMask * 0.34 );

  // Sun bleaching: the whole surface drifts toward a chalky, desaturated buff.
  albedo = mix( albedo, apxDesaturate( albedo, 0.35 ) * 1.06, 0.22 );

  // ── height ────────────────────────────────────────────────────────────────
  float h = 0.60
          + ( swirl - 0.5 ) * 0.085
          + ( stroke - 0.5 ) * 0.035
          + ( coarse - 0.5 ) * 0.045
          + grit * 0.024
          + sharp * 0.018
          + blister * 0.050
          - burst * 0.065
          - cracks * 0.085
          - jn.x * 0.030
          + repair * 0.025;
  // Inside a hole we are one render-thickness lower and following the brick.
  h = mix( h, 0.30 + brick.w * 0.22, hole );
  h = mix( h, h - 0.035, halo );
  s.height = clamp( h, 0.0, 1.0 );

  // ── roughness ─────────────────────────────────────────────────────────────
  // Range 0.36 -> 1.00 within the tile. The burnished trowel arcs are the
  // important half of it: a float leaves the high points of each stroke
  // markedly glossier than the hollows, so raking sun picks the arcs out as
  // sheen even where there is no colour difference at all. That effect is why
  // a real stucco wall never looks like a flat plane, and it is exactly what a
  // constant-roughness material throws away.
  float rough = 0.90 + ( coarse - 0.5 ) * 0.14;
  rough = mix( rough, 0.52, swirl * swirl * 0.85 );
  rough = mix( rough, 0.62, stroke * 0.45 );
  rough = mix( rough, 0.95, hole );              // exposed brick is matte
  rough = mix( rough, 0.99, cracks );
  rough = mix( rough, 0.44, stain * 0.70 );      // rain-washed and polished
  rough = mix( rough, 0.74, repair * 0.8 );      // steel-floated repair
  rough = mix( rough, 0.97, grit * 0.35 );
  s.rough = clamp( rough, 0.36, 1.0 );

  s.metal = 0.0;

  float ao = smoothstep( 0.20, 0.66, s.height );
  ao = mix( ao, ao * 0.50, cracks );
  ao = mix( ao, ao * 0.62, hole );
  s.ao = clamp( mix( 0.40, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Brick
// ─────────────────────────────────────────────────────────────────────────────

const BRICK_GLSL = /* glsl */ `
${GLSL_MASONRY_HELPERS}

void nfSurface( vec2 uv, inout NfSurface s ) {
  // 8 stretchers across x 24 courses over a 1.8 m tile = 225 x 75 mm units,
  // i.e. a 215 x 65 mm metric brick plus its 10 mm joint.
  // The course count is even so the half-brick stagger realigns at the seam.
  vec4 b = nfBrickLayer( uv, vec2( 8.0, 24.0 ), 0.075 );
  vec3 albedo = b.rgb;
  float h = b.w;

  vec4 m = apxMasonry( uv, vec2( 8.0, 24.0 ), 0.5 );
  vec2 aspect = vec2( 8.0, 24.0 ) / 8.0;
  vec2 d = min( m.xy, 1.0 - m.xy ) * aspect.yx;
  float edge = min( d.x, d.y );
  float mortar = 1.0 - smoothstep( 0.075, 0.165, edge );

  // Pitted face texture: fired clay is full of blowholes.
  vec3  pit  = apxWorley( uv * 110.0, 110.0, 1.0 );
  float pits = apxSat( 1.0 - pit.x * 4.0 ) * step( 0.72, pit.z );
  h -= pits * 0.05;
  albedo *= mix( 1.0, 0.86, pits );

  // Soot / weathering wash running down from every bed joint.
  float streak = apxStreaks( uv, 44.0, 8.0, 4 );
  // A brick spans local.y 0..1 with its bed joint at the TOP, so the wash
  // from that joint is strongest at local.y -> 1.
  float below  = fract( uv.y * 24.0 );
  float wash   = apxSat( ( streak - 0.48 ) * 2.6 ) * below;
  albedo = mix( albedo, albedo * vec3( 0.68, 0.66, 0.64 ), wash * 0.45 );

  // Sand-laden wind polishes and lightens the exposed faces.
  float bleach = apxFbmValue( uv * 4.0 + 51.0, 4.0, 3, 0.6 );
  albedo = mix( albedo, apxPalDust() * 0.95, bleach * 0.16 * ( 1.0 - mortar ) );

  s.albedo = albedo;
  s.height = clamp( h, 0.0, 1.0 );

  // A fired brick face is markedly glossier than the mortar around it — the
  // vitrified skin from the kiln survives decades of weather. That contrast,
  // not the colour difference, is what makes brickwork read as brickwork at
  // a distance where the individual units are only a few pixels tall.
  float rough = 0.72;
  rough = mix( rough, 0.97, mortar );       // mortar is chalky and open
  rough = mix( rough, 0.94, pits );
  rough = mix( rough, 0.46, wash * 0.6 );   // rain-washed brick is polished
  s.rough = clamp( rough + ( bleach - 0.5 ) * 0.14, 0.30, 1.0 );
  s.metal = 0.0;

  float ao = smoothstep( 0.22, 0.66, h );
  ao = mix( ao, ao * 0.45, mortar );        // deep shadow in the raked joint
  s.ao = clamp( mix( 0.38, 1.0, ao ), 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Cinderblock / CMU
// ─────────────────────────────────────────────────────────────────────────────

const CINDERBLOCK_GLSL = /* glsl */ `
${GLSL_MASONRY_HELPERS}

void nfSurface( vec2 uv, inout NfSurface s ) {
  // 400 x 200 mm units on a 2.4 m tile => 6 across, 12 courses (even, tiles).
  vec2 count = vec2( 6.0, 12.0 );
  vec4 m = apxMasonry( uv, count, 0.5 );
  vec2 local = m.xy;
  float rnd = m.z;

  vec2 aspect = count / min( count.x, count.y );
  vec2 d = min( local, 1.0 - local ) * aspect.yx;
  float edge = min( d.x, d.y );
  float mortar = 1.0 - smoothstep( 0.055, 0.135, edge );

  // CMU is a coarse, very porous cast product: big open voids, visible cinder
  // aggregate, and a grey that varies noticeably block to block.
  vec3 grey = mix( apxHex( 158.0, 154.0, 143.0 ), apxHex( 126.0, 124.0, 116.0 ), fract( rnd * 5.3 ) );
  float mott = apxFbmValue( local * 4.0 + rnd * 27.0, 4.0, 3, 0.55 );
  grey *= mix( 0.88, 1.10, mott );

  vec3  vo   = apxWorley( uv * 46.0, 46.0, 1.0 );
  float void_ = apxSat( 1.0 - vo.x * 3.6 ) * step( 0.62, vo.z );
  vec3  ag   = apxWorley( uv * 88.0, 88.0, 1.0 );
  float cind = apxSat( 1.0 - ag.x * 4.2 ) * step( 0.5, ag.z );

  vec3 albedo = grey;
  albedo = mix( albedo, albedo * 0.55, void_ * 0.8 );
  albedo = mix( albedo, mix( apxHex( 92.0, 88.0, 84.0 ), apxHex( 176.0, 170.0, 158.0 ), fract( ag.z * 8.1 ) ), cind * 0.35 );

  vec3 mortarCol = apxHex( 172.0, 167.0, 154.0 ) * mix( 0.85, 1.06, apxFbmValue( uv * 36.0, 36.0, 3, 0.5 ) );
  albedo = mix( albedo, mortarCol, mortar );

  // Cracks tend to run through the mortar, so bias the crack gate by it.
  float crack = nfCracks( uv, 7.0, 0.035, 0.45 ) * ( 0.45 + 0.55 * mortar );
  albedo = mix( albedo, apxPalShadow() * 2.0, crack * 0.75 );

  float streak = apxStreaks( uv, 36.0, 6.0, 4 );
  float below  = fract( uv.y * count.y );
  float wash   = apxSat( ( streak - 0.5 ) * 2.4 ) * below;
  albedo = mix( albedo, albedo * vec3( 0.70, 0.68, 0.65 ), wash * 0.4 );

  float dust = apxFbmValue( uv * 5.0 + 17.0, 5.0, 3, 0.6 );
  albedo = mix( albedo, apxPalDust(), dust * 0.13 );

  float h = 0.62 + ( mott - 0.5 ) * 0.05 - void_ * 0.12 - cind * 0.02 - mortar * 0.24 - crack * 0.12;
  s.height = clamp( h, 0.0, 1.0 );
  s.albedo = albedo;

  float rough = 0.93 + ( mott - 0.5 ) * 0.14;
  rough = mix( rough, 0.99, void_ );
  rough = mix( rough, 0.96, mortar );
  rough = mix( rough, 0.58, wash * 0.65 );  // runoff burnishes the block face
  rough = mix( rough, 0.70, cind * 0.45 );  // vitreous cinder catches light
  s.rough = clamp( rough, 0.38, 1.0 );
  s.metal = 0.0;

  float ao = smoothstep( 0.24, 0.68, h );
  ao = mix( ao, ao * 0.35, void_ );
  ao = mix( ao, ao * 0.55, mortar );
  s.ao = clamp( mix( 0.36, 1.0, ao ), 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Recipes
// ─────────────────────────────────────────────────────────────────────────────

export const MASONRY_RECIPES: SurfaceRecipe[] = [
  {
    id: 'concrete',
    surface: 'concrete',
    // 2.4 m so the four form-board courses land at 600 mm, which is a real
    // shuttering board. See the tileMeters audit in the library header.
    tileMeters: 2.4,
    size: 2048,
    ormDivisor: 2,
    seed: 11.0,
    glsl: CONCRETE_GLSL,
    note: 'hero: 6-22 mm graded aggregate, bugholes, randomised form seams, runoff, repairs',
    tuning: {
      normalStrength: 0.52,
      // 13x on a 2.4 m tile puts the detail field's low octave at 37 mm and its
      // top octave at 1.2 mm. The coarse tap at 2.6x covers 185 mm - 6 mm, so
      // the two together span everything from "reads at 4 m" to "reads at
      // 30 cm" with overlap and no gap.
      detailScale: 13.0,
      // 0.62 / 0.38, were 0.80 / 0.45. Still the strongest detail pair in the
      // library — this is a hero surface at 30 cm — but 0.80 at 13x on a
      // horizontal apron in direct sun was the largest single contributor to
      // per-pixel normal variance anywhere in the level.
      detailStrength: 0.62,
      detailScale2: 2.6,
      detailStrength2: 0.38,
      detailAlbedo: 0.20,
      macroScale: 0.075,
      macroStrength: 0.62,
      // 0.20, was 0.32. Budgeted against the 0.45 roughness floor: see the
      // long note beside that clamp in CONCRETE_GLSL.
      macroRough: 0.20,
      // Modest warp: concrete is used on walls too, where a strong warp would
      // visibly bow the form-board lines. Enough to break floor alignment.
      uvWarp: 0.22,
      uvWarpScale: 0.16,
      // Splash-back band and ledge dust. This is the fix for "no dirt
      // accumulation at the base" — it cannot come from tile space.
      grimeHeight: 0.55,
      grimeStrength: 0.80,
      dustStrength: 0.45,
    },
  },
  {
    id: 'plaster',
    surface: 'concrete',
    tileMeters: 3.0,
    size: 1024,
    ormDivisor: 2,
    seed: 27.0,
    glsl: PLASTER_GLSL,
    note: 'two-scale trowel arcs, blistering, hairline cracks, repairs, fall-away patches',
    tuning: {
      normalStrength: 0.48,
      detailScale: 14.0,
      detailStrength: 0.82,
      detailScale2: 2.8,
      detailStrength2: 0.48,
      detailAlbedo: 0.20,
      macroScale: 0.07,
      macroStrength: 0.68,
      macroRough: 0.34,
      uvWarp: 0.26,
      uvWarpScale: 0.15,
      grimeHeight: 0.60,
      grimeStrength: 0.88,
      dustStrength: 0.50,
    },
  },
  {
    id: 'brick',
    surface: 'concrete',
    // 8 stretchers x 24 courses on 1.8 m = 225 x 75 mm: a metric brick
    // (215 x 65) plus its 10 mm joint. The old 2.0 m gave 250 x 83 mm bricks.
    tileMeters: 1.8,
    size: 1024,
    ormDivisor: 2,
    seed: 43.0,
    glsl: BRICK_GLSL,
    note: 'running bond at true 225 x 75 mm, per-brick hash, efflorescence, spalling',
    tuning: {
      normalStrength: 0.80,
      detailScale: 11.0,
      detailStrength: 0.70,
      // No coarse tap and no UV warp on brick: the coarse tap would fight the
      // bond pattern's own relief, and warping a rectilinear lattice bends the
      // courses into visible waves. Structured masonry has to stay straight.
      detailScale2: 0.0,
      detailStrength2: 0.0,
      detailAlbedo: 0.22,
      macroScale: 0.09,
      macroStrength: 0.65,
      macroRough: 0.30,
      grimeHeight: 0.50,
      grimeStrength: 0.80,
      dustStrength: 0.40,
    },
  },
  {
    id: 'cinderblock',
    surface: 'concrete',
    // 6 x 12 units on 2.4 m = 400 x 200 mm, the standard CMU. Unchanged.
    tileMeters: 2.4,
    size: 1024,
    ormDivisor: 2,
    seed: 59.0,
    glsl: CINDERBLOCK_GLSL,
    note: 'CMU: open voids, cinder aggregate, raked joints',
    tuning: {
      normalStrength: 0.75,
      detailScale: 12.0,
      detailStrength: 0.78,
      detailScale2: 0.0,
      detailStrength2: 0.0,
      detailAlbedo: 0.26,
      macroScale: 0.09,
      macroStrength: 0.68,
      macroRough: 0.32,
      grimeHeight: 0.55,
      grimeStrength: 0.85,
      dustStrength: 0.45,
    },
  },
];
