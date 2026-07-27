/**
 * NOISEFIELD — metal surfaces.
 *
 * steel (bare/rusted) · painted steel (shipping-container style)
 *
 * The single most important thing about a weathered metal is that RUST IS NOT
 * METAL. Iron oxide is a rough dielectric; the parent steel is a smooth
 * conductor. If both are shaded with the same metalness you get "brown metal",
 * which is the classic tell of a hand-tuned material. Here the rust mask drives
 * metalness from 1 to 0 and roughness from 0.25 to 0.95 simultaneously, and
 * that *contrast* is what the eye reads as corrosion, far more than the colour.
 *
 * Corrosion is also not random: it starts at welds, fastener heads, panel
 * edges and scratches, then creeps outward and bleeds downward. Every rust
 * mask here is seeded from one of those features.
 */

import type { SurfaceRecipe } from './Recipe';

const GLSL_METAL_HELPERS = /* glsl */ `
/**
 * Brushed-metal anisotropy: extremely stretched noise along U.
 * Wrapping both axes on integer periods keeps it tileable even though the
 * aspect ratio is ~100:1.
 */
float nfBrushed( vec2 uv, float across, float along ) {
  // Per-axis periods: a single max() period would leave the short axis
  // unwrapped and put a seam straight down the middle of the panel.
  vec2 p1 = vec2( along, across );
  vec2 p2 = p1 * 2.0;
  return apxValue( uv * p1, p1 ) * 0.65 + apxValue( uv * p2, p2 ) * 0.35;
}

/**
 * Rust growth field.
 * @param seedMask  where corrosion is allowed to start (welds, edges, chips)
 * @param spread    how far the bloom creeps out from the seed
 *
 * The bloom is a turbulence field biased by the seed mask, then hard-shouldered
 * with smoothstep so it has a definite boundary. Real rust has a crusty,
 * well-defined margin - a soft lerp reads as a stain, not as corrosion.
 */
float nfRustBloom( vec2 uv, float seedMask, float spread, float scale ) {
  float turb = apxTurbulence( apxWarp( uv * scale, scale, 0.40, 3 ), scale, 4, 0.55 );
  float field = seedMask * spread + ( turb - 0.35 ) * 0.9;
  // (0.24, 0.46), was (0.10, 0.55). The shoulder width is what decides whether
  // this reads as CORROSION or as a brown wash, because the mask drives
  // metalness: a texel at rust = 0.5 is shaded as a half-conductor, which is a
  // material that does not exist. It is either steel or it is oxide.
  //
  // Measured at native resolution, the wide shoulder put 43% of the metalness
  // map between 0.4 and 0.9 — the "cheap dirty chrome" band — against 44% of
  // genuinely clean conductor and only 6% of genuinely dielectric oxide.
  // Narrowing the ramp to less than half its width collapses most of that
  // middle onto one end or the other while keeping the margin soft enough that
  // it does not alias into a hard black line at distance.
  return apxSat( smoothstep( 0.24, 0.46, field ) );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Bare / rusted steel
// ─────────────────────────────────────────────────────────────────────────────

const METAL_GLSL = /* glsl */ `
${GLSL_METAL_HELPERS}

void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. mill finish ────────────────────────────────────────────────────────
  float brush  = nfBrushed( uv, 340.0, 5.0 );
  float dents  = apxFbm( uv * 6.0, 6.0, 4, 2.0, 0.5 );
  float pit    = apxSat( 1.0 - apxWorley( uv * 90.0, 90.0, 1.0 ).x * 4.5 );

  vec3 steel = apxHex( 152.0, 154.0, 158.0 );
  steel *= mix( 0.86, 1.10, brush );
  steel *= mix( 0.94, 1.05, dents );

  // ── 2. weld seams ─────────────────────────────────────────────────────────
  // Two horizontal beads per tile with the characteristic stacked ripple, plus
  // a heat-affected zone either side that discolours to straw/blue.
  float bead = apxWeldBead( uv, 2.0, 0.030, 9.0 );
  float haz  = apxWeldBead( uv, 2.0, 0.105, 9.0 ) - bead;
  vec3 hazCol = mix( apxHex( 150.0, 120.0, 70.0 ), apxHex( 80.0, 96.0, 130.0 ), apxTri( uv.y * 6.0 ) );

  // ── 3. rivets ─────────────────────────────────────────────────────────────
  vec2  rv    = apxRivets( uv, vec2( 8.0, 8.0 ), 0.22 );
  float rivet = rv.x * step( 0.30, rv.y );   // not every station has a fastener

  // ── 4. scratches down to bright base metal ────────────────────────────────
  float scratch = apxScratches( uv, 26.0, 3.0, 5.5 );
  float scratch2 = apxScratches( uv * 2.0 + 31.0, 55.0, 7.0, 8.0 ) * 0.6;
  float scratches = apxSat( scratch + scratch2 );

  // ── 4b. edge wear ─────────────────────────────────────────────────────────
  // Anything handled, dragged or brushed past is polished BRIGHT along its
  // arrises while the field stays dull. The mesh's corners are geometry we
  // cannot see from in here, so the stand-in is a high-contrast, very
  // low-frequency mask biased to its own extremes — the same statistical
  // shape, and it lands somewhere plausible on any UV layout.
  float wearF = apxFbm( apxWarp( uv * 3.0, 3.0, 0.65, 3 ), 3.0, 4, 2.0, 0.55 );
  float polish = apxSat( ( wearF - 0.56 ) * 4.2 );

  // ── 5. rust ───────────────────────────────────────────────────────────────
  // Seeded from welds, fasteners and scratch damage: exactly where the
  // protective mill scale is broken. Polished areas resist it — they are the
  // parts still being handled.
  float seed = apxSat( bead * 1.0 + haz * 0.45 + rivet * 0.8 + scratches * 0.5 );
  float rust = nfRustBloom( uv, seed, 0.85, 4.0 );
  // A second, older and larger generation of corrosion, independent of seeds.
  float rustOld = nfRustBloom( uv + 71.0, 0.15, 1.0, 2.0 ) * 0.7;
  rust = apxSat( max( rust, rustOld ) ) * ( 1.0 - polish * 0.85 );

  // Rust runoff: dissolved oxide bleeds downhill and stains the clean steel.
  float streak = apxStreaks( uv, 42.0, 5.0, 4 );
  // Beads sit where uv.y*2 + 0.5 is integral; the bleed is strongest just
  // UNDER a bead and fades to the next bead down.
  float below  = fract( uv.y * 2.0 + 0.5 );
  float bleed  = apxSat( ( streak - 0.50 ) * 2.6 ) * below * apxSat( seed * 2.0 ) * ( 1.0 - rust );

  // Rust colour is layered: dark scale in the core, bright orange at the
  // advancing margin, pale ochre in the thinnest film.
  float rustDepth = apxFbmValue( uv * 12.0 + 5.0, 12.0, 3, 0.55 );
  vec3 rustCore = apxHex( 88.0, 46.0, 28.0 );
  vec3 rustMid  = apxPalRust();
  vec3 rustEdge = apxHex( 166.0, 104.0, 58.0 );
  vec3 rustCol  = mix( rustCore, rustMid, rustDepth );
  rustCol = mix( rustCol, rustEdge, apxSat( ( rust - 0.25 ) * ( 1.0 - rust ) * 3.0 ) );

  // ── composite ─────────────────────────────────────────────────────────────
  vec3 albedo = steel;
  albedo = mix( albedo, albedo * 1.14, bead * 0.5 );
  albedo = mix( albedo, hazCol, haz * 0.55 );
  albedo = mix( albedo, albedo * 1.18, rivet * 0.4 );
  albedo = mix( albedo, apxHex( 214.0, 216.0, 220.0 ), scratches * 0.7 );
  albedo = mix( albedo, apxHex( 196.0, 199.0, 205.0 ), polish * 0.55 );
  albedo = mix( albedo, rustCol * 0.85, bleed * 0.55 );
  albedo = mix( albedo, rustCol, rust );
  // Everything in this compound is under a film of sand.
  float dust = apxFbmValue( uv * 5.0 + 121.0, 5.0, 3, 0.6 );
  albedo = mix( albedo, apxPalDust(), dust * 0.10 );

  // ── height ────────────────────────────────────────────────────────────────
  float rustCrust = rust * ( 0.35 + 0.65 * apxFbmWorley( uv * 40.0, 40.0, 2, 0.5, 1.0 ) );
  float h = 0.55
          + ( dents - 0.5 ) * 0.10
          + ( brush - 0.5 ) * 0.012
          + bead * 0.070
          + rivet * 0.090
          - scratches * 0.030
          + rustCrust * 0.055
          - pit * rust * 0.070;      // deep corrosion pits, only inside rust
  s.height = clamp( h, 0.0, 1.0 );

  // ── the money shot: rough/metal contrast ──────────────────────────────────
  // The full span here is 0.06 (mirror-polished arris) to 0.98 (crusted
  // oxide). If a weathered-metal material only reads as "brown", it is almost
  // always because this span was collapsed: brown paint and brown rust look
  // identical until one of them is rough AND dielectric and the other is
  // smooth AND conductive.
  float rough = 0.34 + ( brush - 0.5 ) * 0.30;   // anisotropic mill finish
  rough = mix( rough, 0.55, haz * 0.6 );
  rough = mix( rough, 0.62, bead );
  rough = mix( rough, 0.12, scratches * 0.85 );  // fresh scratch is bright
  rough = mix( rough, 0.09, polish * 0.75 );     // handled arris
  rough = mix( rough, 0.98, rust );              // oxide is matte
  rough = mix( rough, 0.76, bleed * 0.6 );
  rough = mix( rough, rough * 1.10, dust * 0.5 );
  s.rough = clamp( rough, 0.06, 1.0 );

  // Iron oxide is a DIELECTRIC. This single line does more for the look than
  // all of the albedo work above.
  float metal = 1.0;
  metal = mix( metal, 0.0, rust );
  // 0.12, was 0.30. Rust runoff is a deposited OXIDE film — it is dielectric,
  // just thinner and more transparent than the crust it ran off. 0.30 parked a
  // broad, gravity-ramped band of the panel at metalness ~0.6, which is the
  // single largest source of the mid-range that makes weathered steel read as
  // dirty chrome.
  metal = mix( metal, 0.12, bleed * 0.55 );
  metal = mix( metal, metal * 0.85, dust * 0.4 );
  s.metal = clamp( metal, 0.0, 1.0 );

  float ao = smoothstep( 0.30, 0.72, s.height );
  ao = mix( ao, ao * 0.72, rust * 0.5 );
  s.ao = clamp( mix( 0.55, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Painted steel — shipping container / vehicle panels
// ─────────────────────────────────────────────────────────────────────────────

const PAINTED_STEEL_GLSL = /* glsl */ `
${GLSL_METAL_HELPERS}

void nfSurface( vec2 uv, inout NfSurface s ) {

  // Corrugation is GEOMETRY, not texture - baking it in would fight the mesh's
  // silhouette and break at every corner. This map is the flat panel finish.

  // ── 1. paint ──────────────────────────────────────────────────────────────
  // A desert-weathered industrial teal. Sun fade is strongly directional in
  // reality (whichever face points at the sun) but we approximate with a
  // low-frequency bleach field so any orientation reads plausibly.
  vec3 paintFresh = apxHex( 62.0, 96.0, 92.0 );
  vec3 paintFaded = apxHex( 128.0, 148.0, 140.0 );
  float fade = apxFbm( apxWarp( uv * 2.0, 2.0, 0.5, 3 ), 2.0, 4, 2.0, 0.6 );
  vec3 paint = mix( paintFresh, paintFaded, apxSat( fade * 1.25 - 0.1 ) );

  // Roller/spray mottle plus the faint orange-peel of an industrial finish.
  float peel = apxFbmValue( uv * 40.0, 40.0, 3, 0.5 );
  paint *= mix( 0.95, 1.06, peel );

  // ── 2. stencil grime ──────────────────────────────────────────────────────
  // Rectangular ghost of a removed placard: slightly less faded because it was
  // shielded from UV. Reads as "something used to be bolted here".
  float placard = apxBand( uv.x, 0.18, 0.52, 0.006 ) * apxBand( uv.y, 0.58, 0.82, 0.006 );
  paint = mix( paint, mix( paintFresh, paint, 0.35 ), placard * 0.55 );

  // ── 3. paint chipping ─────────────────────────────────────────────────────
  // Chips have a hard edge (paint fractures, it does not fade away) and a
  // slightly lighter rim of lifted primer.
  vec2  cwv   = apxWarp( uv * 7.0, 7.0, 0.5, 3 );
  float chipF = apxFbm( cwv, 7.0, 5, 2.0, 0.55 );
  float chip  = smoothstep( 0.615, 0.645, chipF );
  float rim   = smoothstep( 0.590, 0.618, chipF ) - chip;

  float scratches = apxSat( apxScratches( uv, 22.0, 3.0, 5.0 ) + apxScratches( uv * 2.0 + 7.0, 48.0, 6.0, 7.0 ) * 0.5 );

  // ── 3b. edge wear down to bare steel ──────────────────────────────────────
  // A container is stacked, dragged and hit by forklifts. Its arrises lose the
  // paint entirely and go BRIGHT — a hard specular line along every corner is
  // one of the most recognisable things about a used shipping container, and
  // it is pure metalness/roughness contrast, not colour.
  float wearF = apxFbm( apxWarp( uv * 4.0 + 13.0, 4.0, 0.6, 3 ), 4.0, 4, 2.0, 0.55 );
  float bright = apxSat( ( wearF - 0.615 ) * 9.0 );

  // ── 4. rust through the breaks ────────────────────────────────────────────
  float panelEdge = apxSat( 1.0 - abs( fract( uv.y * 2.0 ) - 0.5 ) * 8.0 );
  float seed = apxSat( chip * 1.0 + scratches * 0.6 + panelEdge * 0.55 );
  float rust = nfRustBloom( uv, seed, 0.95, 4.0 ) * ( 1.0 - bright * 0.9 );

  // Runoff streaks below every chip and panel joint. This is THE detail that
  // sells a container; without it the panel looks like painted plastic.
  float streak = apxStreaks( uv, 46.0, 4.0, 4 );
  // Panel joints sit at fract(uv.y*2) == 0.5; drips peak just under a joint.
  float below  = 1.0 - fract( 0.5 - uv.y * 2.0 );
  float drip   = apxSat( ( streak - 0.46 ) * 2.8 ) * below * apxSat( seed * 1.6 + rust * 0.6 );

  vec3 rustCol = mix( apxHex( 96.0, 52.0, 30.0 ), apxHex( 156.0, 96.0, 54.0 ), apxFbmValue( uv * 14.0, 14.0, 3, 0.55 ) );
  vec3 primer  = apxHex( 168.0, 132.0, 96.0 );
  vec3 bare    = apxHex( 186.0, 188.0, 192.0 );

  vec3 albedo = paint;
  albedo = mix( albedo, primer, rim * 0.8 );
  albedo = mix( albedo, primer * 0.9, chip * 0.9 );
  albedo = mix( albedo, bare, scratches * 0.55 * ( 1.0 - rust ) );
  albedo = mix( albedo, bare * 1.04, bright * 0.85 );
  albedo = mix( albedo, rustCol * 0.9, drip * 0.65 );
  albedo = mix( albedo, rustCol, rust );

  float dust = apxFbmValue( uv * 6.0 + 210.0, 6.0, 3, 0.6 );
  albedo = mix( albedo, apxPalDust(), dust * 0.12 );

  // ── height ────────────────────────────────────────────────────────────────
  float rustCrust = rust * ( 0.4 + 0.6 * apxFbmWorley( uv * 44.0, 44.0, 2, 0.5, 1.0 ) );
  float h = 0.58
          + ( peel - 0.5 ) * 0.020
          - chip * 0.035
          + rim * 0.012
          - scratches * 0.020
          + rustCrust * 0.050
          - panelEdge * 0.030
          - bright * 0.014;
  s.height = clamp( h, 0.0, 1.0 );

  // Intact paint is a smooth dielectric with a clearcoat (set on the material);
  // where it has failed we are back to the rough-oxide / smooth-steel split.
  // Span 0.07 -> 0.96, and — critically — metalness swings 0 -> 1 over the
  // SAME masks, in the same direction. Rust that is merely brown reads as
  // painted metal; rust that is brown AND rough AND dielectric reads as rust.
  float rough = 0.42 + ( peel - 0.5 ) * 0.16;
  rough = mix( rough, 0.58, apxSat( fade * 1.2 ) * 0.50 );   // chalked by UV
  rough = mix( rough, 0.72, chip );
  rough = mix( rough, 0.16, scratches * 0.75 );
  rough = mix( rough, 0.11, bright * 0.85 );                 // polished arris
  rough = mix( rough, 0.96, rust );
  rough = mix( rough, 0.78, drip * 0.6 );
  s.rough = clamp( rough, 0.07, 1.0 );

  float metal = 0.0;                                  // paint film
  metal = mix( metal, 1.0, scratches * 0.8 );         // bare steel in the gouge
  metal = mix( metal, 1.0, bright * 0.92 );           // worn back to substrate
  metal = mix( metal, 0.0, rust );                    // oxide is not a metal
  s.metal = clamp( metal, 0.0, 1.0 );

  float ao = smoothstep( 0.32, 0.72, s.height );
  ao = mix( ao, ao * 0.70, rust * 0.6 );
  ao = mix( ao, ao * 0.75, chip );
  s.ao = clamp( mix( 0.55, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Recipes
// ─────────────────────────────────────────────────────────────────────────────

export const METAL_RECIPES: SurfaceRecipe[] = [
  {
    id: 'metal',
    surface: 'metal',
    tileMeters: 1.6,
    size: 1024,
    ormDivisor: 2,
    seed: 131.0,
    glsl: METAL_GLSL,
    note: 'brushed steel, welds, rivets, seeded rust blooms, polished arrises',
    tuning: {
      normalStrength: 0.48,
      detailScale: 12.0,
      detailStrength: 0.55,
      detailScale2: 2.4,
      detailStrength2: 0.30,
      // Low on metal: micro colour grain on a conductor reads as dirt, and the
      // interesting variation on steel is specular, not diffuse.
      detailAlbedo: 0.10,
      macroScale: 0.10,
      macroStrength: 0.55,
      // Metal lives or dies on gloss variation, so the macro roughness swing
      // is the largest of any surface here.
      macroRough: 0.40,
      grimeHeight: 0.35,
      grimeStrength: 0.55,
      dustStrength: 0.60,
    },
  },
  {
    id: 'steel_painted',
    surface: 'metal',
    tileMeters: 2.2,
    size: 1024,
    ormDivisor: 2,
    seed: 149.0,
    physical: true,
    glsl: PAINTED_STEEL_GLSL,
    note: 'container panel: UV fade, chipping, rust runoff, clearcoat',
    // Clearcoat only pays off where a lacquer film genuinely exists. Its own
    // roughness is kept high so the panel reads as chalked, not showroom.
    params: { clearcoat: 0.45, clearcoatRoughness: 0.55, sheen: 0.0 },
    tuning: {
      normalStrength: 0.46,
      detailScale: 12.0,
      detailStrength: 0.52,
      detailScale2: 2.4,
      detailStrength2: 0.28,
      detailAlbedo: 0.12,
      macroScale: 0.09,
      macroStrength: 0.60,
      macroRough: 0.38,
      grimeHeight: 0.40,
      grimeStrength: 0.60,
      dustStrength: 0.65,
    },
  },
];
