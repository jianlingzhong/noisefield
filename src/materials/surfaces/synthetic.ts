/**
 * NOISEFIELD — synthetic + transmissive surfaces.
 *
 * fabric (canvas/tarp) · rubber (tyre) · glass · broken glass · water
 */

import * as THREE from 'three';

import type { SurfaceRecipe } from './Recipe';

// ─────────────────────────────────────────────────────────────────────────────
// Canvas / tarpaulin
// ─────────────────────────────────────────────────────────────────────────────

const FABRIC_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. plain weave ────────────────────────────────────────────────────────
  // A real weave is not a flat checkerboard: the thread that passes OVER is
  // lit and raised, the one passing UNDER is shadowed and recessed. apxWeave
  // returns both the combined height and which thread won this cell.
  vec2  wv    = apxWeave( uv, 96.0, 0.10 );
  float weave = wv.x;
  float over  = wv.y;

  // Slub: hand-spun thread varies in diameter along its length.
  float slub = apxFbmValue( vec2( uv.x * 96.0, uv.y * 6.0 ), vec2( 96.0, 6.0 ), 3, 0.5 );

  // ── 2. sag folds ──────────────────────────────────────────────────────────
  // A tarp is never taut. Low-frequency, strongly anisotropic creases with a
  // sharp crest (ridged, not smooth) where the fabric has been folded.
  float fold = apxRidged( vec2( uv.x * 3.0, uv.y * 7.0 ), vec2( 3.0, 7.0 ), 3, 0.55 );
  float sag  = apxFbm( uv * 2.0, 2.0, 3, 2.0, 0.6 );

  // ── 3. wear ───────────────────────────────────────────────────────────────
  // Abrasion happens on the fold crests, where the fabric rubs. Threads there
  // break first, so the fray mask is gated by the fold field.
  float abrade = apxSat( ( fold - 0.55 ) * 2.4 );
  float fray = apxSat( apxWorleyAniso( uv, 140.0, 26.0, 1.0 ).x * -5.0 + 1.0 ) * abrade;
  float hole = smoothstep( 0.80, 0.90, apxFbmValue( uv * 5.0 + 44.0, 5.0, 4, 0.55 ) ) * abrade;

  // ── 4. colour ─────────────────────────────────────────────────────────────
  // Olive-drab canvas bleached toward buff by years of sun.
  vec3 dyed    = apxHex( 104.0, 100.0,  68.0 );
  vec3 bleached = apxHex( 172.0, 162.0, 128.0 );
  float sun = apxSat( apxFbm( uv * 3.0 + 13.0, 3.0, 4, 2.0, 0.6 ) * 1.35 - 0.12 );
  // Crests get the most UV, so bias bleaching by the fold field.
  sun = apxSat( sun * 0.7 + abrade * 0.6 );
  vec3 cloth = mix( dyed, bleached, sun );

  // Thread-level shading: the over-thread catches light, the under one does not.
  cloth *= mix( 0.80, 1.10, weave );
  cloth *= mix( 0.97, 1.03, over );
  cloth *= mix( 0.92, 1.06, slub );

  // Dirt collects in the fold valleys.
  float valley = apxSat( 1.0 - fold * 1.6 );
  cloth = mix( cloth, cloth * vec3( 0.62, 0.60, 0.56 ), valley * 0.35 );
  cloth = mix( cloth, apxPalDust(), apxSat( ( sag - 0.5 ) * 1.5 ) * 0.12 );
  cloth = mix( cloth, cloth * 1.18, fray * 0.6 );

  s.albedo = cloth;
  // Threadbare holes cut all the way through.
  s.alpha = 1.0 - smoothstep( 0.35, 0.75, hole );

  s.height = clamp( 0.52 + ( weave - 0.5 ) * 0.30 + ( fold - 0.5 ) * 0.22 + ( sag - 0.5 ) * 0.10 + fray * 0.06, 0.0, 1.0 );

  // Cloth is uniformly rough; the interesting response comes from the sheen
  // term on the material, which is what gives fabric its rim brightening.
  float rough = 0.90 - ( weave - 0.5 ) * 0.08;
  rough = mix( rough, 0.96, fray );
  rough = mix( rough, 0.84, sun * 0.3 );
  s.rough = clamp( rough, 0.55, 1.0 );
  s.metal = 0.0;
  s.ao = clamp( mix( 0.55, 1.0, smoothstep( 0.18, 0.72, s.height ) ), 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Rubber / tyre
// ─────────────────────────────────────────────────────────────────────────────

const RUBBER_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. tread ──────────────────────────────────────────────────────────────
  // Blocky all-terrain pattern: a circumferential rib plus staggered lugs.
  // Integer counts, and the stagger uses mod(row,2) with an even row count.
  vec4 m = apxMasonry( uv, vec2( 6.0, 10.0 ), 0.5 );
  vec2 local = m.xy;
  float rnd = m.z;
  vec2 d = min( local, 1.0 - local );
  float groove = 1.0 - smoothstep( 0.10, 0.20, min( d.x * 1.0, d.y * 1.6 ) );

  // Sipes: fine slits cut across each lug for wet grip.
  float sipe = apxSat( 1.0 - abs( fract( local.y * 4.0 ) - 0.5 ) * 9.0 ) * ( 1.0 - groove );

  // ── 2. compound ───────────────────────────────────────────────────────────
  // Carbon-black rubber: nearly no colour, all of the read comes from the
  // roughness structure and the moulding texture.
  float mould = apxFbmValue( uv * 70.0, 70.0, 3, 0.5 );
  float crack = 1.0 - smoothstep( 0.0, 0.05, apxVoronoiEdge( uv * 40.0, 40.0, 0.9 ) );
  float ozone = crack * smoothstep( 0.40, 0.75, apxFbmValue( uv * 6.0 + 3.0, 6.0, 3, 0.6 ) );

  vec3 rubber = apxHex( 30.0, 30.0, 32.0 );
  rubber *= mix( 0.85, 1.18, mould );
  rubber = mix( rubber, rubber * 0.6, ozone * 0.7 );
  // Wear polishes the crown of each lug to a faint grey sheen.
  float wear = apxSat( ( 1.0 - groove ) * mix( 0.4, 1.0, fract( rnd * 5.7 ) ) );
  rubber = mix( rubber, apxHex( 62.0, 60.0, 60.0 ), wear * 0.25 );

  float dust = apxFbmValue( uv * 5.0 + 88.0, 5.0, 3, 0.6 );
  rubber = mix( rubber, apxPalDust() * 0.55, dust * 0.16 );

  s.albedo = rubber;
  s.height = clamp( 0.68 - groove * 0.42 - sipe * 0.16 - ozone * 0.10 + ( mould - 0.5 ) * 0.04, 0.0, 1.0 );

  float rough = 0.88 + ( mould - 0.5 ) * 0.10;
  rough = mix( rough, 0.66, wear * 0.7 );        // polished crown
  rough = mix( rough, 0.97, ozone );
  s.rough = clamp( rough, 0.45, 1.0 );
  s.metal = 0.0;
  s.ao = clamp( mix( 0.42, 1.0, smoothstep( 0.20, 0.72, s.height ) ), 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Glass — intact
// ─────────────────────────────────────────────────────────────────────────────

const GLASS_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {
  // Clean glass has essentially no albedo detail; everything interesting comes
  // from transmission and from the DIRT sitting on it. Modelling only the dirt
  // is both cheaper and more convincing than inventing surface colour.
  float film   = apxFbmValue( uv * 4.0, 4.0, 4, 0.6 );
  float smear  = apxStreaks( uv, 24.0, 3.0, 4 );
  float grime  = apxSat( ( apxFbmValue( uv * 9.0 + 31.0, 9.0, 4, 0.55 ) - 0.55 ) * 2.6 );
  float spots  = apxSat( 1.0 - apxWorley( uv * 40.0, 40.0, 1.0 ).x * 7.0 ) * step( 0.80, apxWorley( uv * 40.0, 40.0, 1.0 ).z );

  float dirt = apxSat( grime * 0.7 + apxSat( ( smear - 0.55 ) * 2.2 ) * 0.5 + spots * 0.6 );

  // Tinted very slightly green - float glass really is, from the iron content.
  vec3 glass = apxHex( 226.0, 236.0, 232.0 );
  s.albedo = mix( glass, apxPalDust() * 0.75, dirt * 0.55 );

  s.height = clamp( 0.5 + ( film - 0.5 ) * 0.02 + dirt * 0.01, 0.0, 1.0 );
  // Near-mirror where clean, matte where the dust film sits. That contrast is
  // what makes a window look like a window rather than a blue rectangle.
  s.rough = clamp( mix( 0.035, 0.55, dirt ), 0.02, 1.0 );
  s.metal = 0.0;
  s.ao = 1.0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Glass — shattered
// ─────────────────────────────────────────────────────────────────────────────

const GLASS_BROKEN_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {
  // Three impact hubs, each throwing radial arms and concentric rings.
  // apxSpiderCracks wraps its delta into [-0.5,0.5] so a fracture whose hub is
  // near an edge continues correctly onto the neighbouring tile.
  vec3 c0 = apxSpiderCracks( uv, vec2( 0.31, 0.62 ), 13.0, 9.0,  0.020 );
  vec3 c1 = apxSpiderCracks( uv, vec2( 0.74, 0.28 ), 17.0, 12.0, 0.016 );
  vec3 c2 = apxSpiderCracks( uv, vec2( 0.55, 0.86 ),  9.0, 7.0,  0.024 );

  float radial = apxSat( c0.x + c1.x + c2.x );
  float rings  = apxSat( c0.y + c1.y + c2.y );

  // A background shatter network so the pane is fragmented between the hubs.
  float net = 1.0 - smoothstep( 0.0, 0.022, apxVoronoiEdge( apxWarp( uv * 9.0, 9.0, 0.25, 3 ), 9.0, 0.95 ) );
  // The network only survives where an impact has weakened the pane.
  float energy = apxSat( 1.0 - min( min( c0.z, c1.z ), c2.z ) * 2.4 );
  net *= energy;

  float crack = apxSat( radial * 1.1 + rings * 0.8 + net * 0.9 );

  // Missing shards: fully-enclosed fragments have fallen out.
  vec3  frag = apxWorley( apxWarp( uv * 9.0, 9.0, 0.25, 3 ), 9.0, 0.95 );
  float gone = step( 0.90, frag.z ) * energy * ( 1.0 - crack );

  float film = apxFbmValue( uv * 6.0, 6.0, 3, 0.6 );
  float dust = apxSat( ( apxFbmValue( uv * 12.0 + 5.0, 12.0, 4, 0.55 ) - 0.5 ) * 2.0 );

  vec3 glass = apxHex( 224.0, 234.0, 230.0 );
  // A fracture face scatters light and looks white/frosted, not dark.
  vec3 albedo = mix( glass, vec3( 0.92, 0.95, 0.96 ), crack * 0.85 );
  albedo = mix( albedo, apxPalDust() * 0.8, dust * 0.30 );

  s.albedo = albedo;
  s.alpha = 1.0 - gone;
  // Each fragment is tilted slightly, which is why a broken pane sparkles: the
  // per-fragment height offset becomes a per-fragment normal after the Sobel
  // derivation.
  s.height = clamp( 0.55 + ( fract( frag.z * 7.3 ) - 0.5 ) * 0.05 * energy - crack * 0.16, 0.0, 1.0 );
  s.rough = clamp( mix( 0.04, 0.62, apxSat( crack * 0.9 + dust * 0.4 ) ), 0.02, 1.0 );
  s.metal = 0.0;
  s.ao = clamp( 1.0 - crack * 0.30, 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Water
// ─────────────────────────────────────────────────────────────────────────────

const WATER_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {
  // Two counter-travelling gravity-wave trains at different scales plus a
  // capillary ripple layer. Animation is done by scrolling the map offsets at
  // different rates in MaterialLibrary.update(), which is why two independent
  // wave systems matter: scrolled together they would look like a sliding
  // texture, scrolled apart they interfere and read as moving water.
  float w1 = apxFbm( apxWarp( uv * 5.0, 5.0, 0.35, 3 ), 5.0, 4, 2.0, 0.55 );
  float w2 = apxFbm( apxWarp( uv * 11.0 + 37.0, 11.0, 0.25, 3 ), 11.0, 3, 2.0, 0.5 );
  float cap = apxFbmValue( uv * 46.0, 46.0, 3, 0.5 );

  float h = 0.5 + ( w1 - 0.5 ) * 0.55 + ( w2 - 0.5 ) * 0.30 + ( cap - 0.5 ) * 0.10;
  s.height = clamp( h, 0.0, 1.0 );

  // Stagnant compound water: silty, greenish, with a scum film in the calmer
  // low-lying areas.
  float scum = apxSat( ( apxFbmValue( uv * 3.0 + 71.0, 3.0, 4, 0.6 ) - 0.55 ) * 2.4 );
  vec3 deep = apxHex( 32.0, 44.0, 40.0 );
  vec3 silt = apxHex( 74.0, 78.0, 58.0 );
  vec3 albedo = mix( deep, silt, apxSat( ( w1 - 0.4 ) * 1.2 ) * 0.5 );
  albedo = mix( albedo, apxHex( 96.0, 104.0, 72.0 ), scum * 0.6 );
  s.albedo = albedo;

  // Water is the smoothest thing in the level; the scum film is the exception
  // and that contrast is what makes the surface read as liquid.
  s.rough = clamp( mix( 0.045, 0.42, scum ), 0.02, 1.0 );
  s.metal = 0.0;
  s.ao = 1.0;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Recipes
// ─────────────────────────────────────────────────────────────────────────────

export const SYNTHETIC_RECIPES: SurfaceRecipe[] = [
  {
    id: 'fabric',
    surface: 'fabric',
    tileMeters: 1.2,
    size: 1024,
    ormDivisor: 2,
    seed: 211.0,
    physical: true,
    glsl: FABRIC_GLSL,
    note: 'plain weave, sag folds, abrasion, sun bleach, threadbare holes',
    params: {
      // Sheen is the defining BRDF feature of cloth: a retro-reflective lobe at
      // grazing angles. Without it canvas shades exactly like rough plastic.
      sheen: 1.0,
      sheenRoughness: 0.85,
      sheenColor: new THREE.Color(0.86, 0.82, 0.70),
      side: THREE.DoubleSide,
      alphaTest: 0.5,
    },
    tuning: {
      normalStrength: 0.42, detailScale: 8.0, detailStrength: 0.55,
      // Cloth already carries its weave in the base map; a coarse detail tap
      // would read as a second, larger weave printed on top of it.
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.18,
      macroScale: 0.16, macroStrength: 0.55, macroRough: 0.26,
      dustStrength: 0.55,
    },
  },
  {
    id: 'rubber',
    surface: 'rubber',
    tileMeters: 0.8,
    size: 512,
    ormDivisor: 2,
    seed: 223.0,
    glsl: RUBBER_GLSL,
    note: 'tyre tread, sipes, ozone crazing, polished crown',
    tuning: {
      normalStrength: 0.90, detailScale: 9.0, detailStrength: 0.60,
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.16,
      macroScale: 0.22, macroStrength: 0.45, macroRough: 0.28,
      dustStrength: 0.40,
    },
  },
  {
    id: 'glass',
    surface: 'glass',
    tileMeters: 1.5,
    size: 512,
    ormDivisor: 2,
    seed: 239.0,
    physical: true,
    glsl: GLASS_GLSL,
    note: 'float glass with dirt film; real transmission',
    params: {
      transmission: 0.92,
      thickness: 0.012,
      ior: 1.52,
      // Transmission ignores `opacity`, so we stay non-transparent and let the
      // transmission pass do the work. This also keeps glass out of the sorted
      // blend queue, which is where most WebGL glass goes wrong.
      transparent: false,
      // 1.0, not 0.0. `metalness` is a MULTIPLIER on the ORM texture's blue
      // channel, and this recipe writes s.metal = 0.0, so the shaded
      // metalnessFactor is 0 either way — the two are bit-identical. Keeping
      // every material in the library at exactly 1.0 makes "the scalars are
      // pure ORM multipliers, all dielectric behaviour lives in the texture"
      // an invariant that can be asserted, instead of a convention with three
      // exceptions that each have to be reasoned about.
      metalness: 1.0,
      specularIntensity: 1.0,
      side: THREE.DoubleSide,
    },
    tuning: {
      normalStrength: 0.10, detailScale: 4.0, detailStrength: 0.06,
      // Glass gets NO detail albedo and no coarse tap. Micro grain on a
      // transmissive surface reads as frosting, which is the opposite of the
      // intent, and the dirt film in the base map is already doing that job.
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.0,
      macroScale: 0.25, macroStrength: 0.25, macroRough: 0.10,
    },
  },
  {
    id: 'glass_broken',
    surface: 'glass',
    tileMeters: 1.5,
    size: 1024,
    ormDivisor: 2,
    seed: 251.0,
    physical: true,
    glsl: GLASS_BROKEN_GLSL,
    note: 'three impact hubs, radial + concentric fracture, missing shards',
    params: {
      transmission: 0.80,
      thickness: 0.012,
      ior: 1.52,
      transparent: false,
      alphaTest: 0.5,
      // 1.0, not 0.0. `metalness` is a MULTIPLIER on the ORM texture's blue
      // channel, and this recipe writes s.metal = 0.0, so the shaded
      // metalnessFactor is 0 either way — the two are bit-identical. Keeping
      // every material in the library at exactly 1.0 makes "the scalars are
      // pure ORM multipliers, all dielectric behaviour lives in the texture"
      // an invariant that can be asserted, instead of a convention with three
      // exceptions that each have to be reasoned about.
      metalness: 1.0,
      side: THREE.DoubleSide,
    },
    tuning: {
      normalStrength: 0.55, detailScale: 5.0, detailStrength: 0.14,
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.0,
      macroScale: 0.25, macroStrength: 0.20, macroRough: 0.10,
    },
  },
  {
    id: 'water',
    surface: 'water',
    tileMeters: 3.0,
    size: 512,
    ormDivisor: 2,
    seed: 263.0,
    physical: true,
    glsl: WATER_GLSL,
    note: 'two interfering wave trains + capillary ripple, animated in update()',
    params: {
      // 1.0, not 0.0. `metalness` is a MULTIPLIER on the ORM texture's blue
      // channel, and this recipe writes s.metal = 0.0, so the shaded
      // metalnessFactor is 0 either way — the two are bit-identical. Keeping
      // every material in the library at exactly 1.0 makes "the scalars are
      // pure ORM multipliers, all dielectric behaviour lives in the texture"
      // an invariant that can be asserted, instead of a convention with three
      // exceptions that each have to be reasoned about.
      metalness: 1.0,
      clearcoat: 1.0,
      clearcoatRoughness: 0.06,
      ior: 1.33,
      transmission: 0.35,
      thickness: 0.4,
      transparent: false,
    },
    tuning: {
      normalStrength: 0.30, detailScale: 4.0, detailStrength: 0.25,
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.0,
      macroScale: 0.20, macroStrength: 0.20, macroRough: 0.08,
    },
  },
];
