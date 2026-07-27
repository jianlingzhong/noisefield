/**
 * NOISEFIELD — organic surfaces.
 *
 * wood · foliage · flesh
 */

import * as THREE from 'three';

import type { SurfaceRecipe } from './Recipe';

// ─────────────────────────────────────────────────────────────────────────────
// Weathered timber planking
// ─────────────────────────────────────────────────────────────────────────────

const WOOD_GLSL = /* glsl */ `
/**
 * Growth rings.
 *
 * A saw cut through a log produces rings whose spacing is uneven (fast spring
 * growth, slow summer growth) and whose path is bent by every knot. We build
 * that by taking a coordinate that runs ACROSS the plank, warping it with a
 * low-frequency field, and then applying a sharp periodic ramp. Squaring the
 * triangle wave produces the hard latewood band next to a soft earlywood one.
 */
float nfGrain( vec2 g, float rings ) {
  // g.x is plank-local (it resets at the tile edge, so it is exactly periodic
  // and safe at any scale). g.y advances exactly 1.0 per tile, so scaling it
  // by PER and wrapping on PER satisfies obligation T2.
  const float PER = 12.0;
  vec2 q = g * PER;
  float bend = ( apxFbm( q, PER, 4, 2.0, 0.55 ) - 0.5 ) * 0.55;
  float fine = ( apxFbmValue( q * vec2( 6.0, 1.0 ), vec2( PER * 6.0, PER ), 3, 0.5 ) - 0.5 ) * 0.12;
  float ring = apxTri( ( g.x + bend + fine ) * rings );
  // Bias toward the latewood band and sharpen it.
  return pow( ring, 2.2 );
}

void nfSurface( vec2 uv, inout NfSurface s ) {

  // ── 1. plank layout ───────────────────────────────────────────────────────
  // 5 boards running along V. Integer count => tiles.
  float boards = 5.0;
  float bx = uv.x * boards;
  float bi = floor( bx );
  float bl = fract( bx );
  float rnd = apxHash21( vec2( mod( bi, boards ), 0.0 ) + 4.7 );

  // Each board is a different cut from a different log: offset and flip the
  // grain coordinate per board so neighbours are never identical.
  float flip = step( 0.5, fract( rnd * 3.3 ) );
  float gx = mix( bl, 1.0 - bl, flip ) + rnd * 3.0;
  float gy = uv.y + rnd * 7.0;

  // ── 2. grain + rays ───────────────────────────────────────────────────────
  float grain = nfGrain( vec2( gx, gy ), 9.0 + floor( rnd * 5.0 ) );
  // Medullary rays: short bright flecks running across the grain.
  float rays = apxSat( apxWorleyAniso( vec2( gx, gy ), 8.0, 46.0, 1.0 ).x * -3.0 + 1.0 ) * 0.5;

  // ── 3. knots ──────────────────────────────────────────────────────────────
  // A knot is a former branch: dark, dense, and it deflects the surrounding
  // grain into concentric ovals. We add its own ring system on top.
  vec3  kw   = apxWorleyAniso( vec2( gx * 0.35, gy ), 3.0, 6.0, 1.0 );
  float knotGate = step( 0.88, kw.z );
  float knotD = kw.x;
  float knot  = apxSat( 1.0 - knotD * 5.0 ) * knotGate;
  float knotRings = pow( apxTri( knotD * 14.0 ), 2.0 ) * knot;

  // ── 4. weathering ─────────────────────────────────────────────────────────
  // Exposed timber silvers: lignin breaks down, the surface goes grey and the
  // softer earlywood erodes away leaving the latewood proud (raised grain).
  float silver = apxSat( apxFbm( uv * 3.0 + 19.0, 3.0, 4, 2.0, 0.6 ) * 1.3 - 0.15 );
  float split  = apxSat( apxWorleyAniso( uv, 90.0, 2.0, 1.0 ).x * -6.0 + 1.0 );  // long checks
  float splinter = apxSat( apxWorleyAniso( uv * 2.0 + 3.0, 140.0, 3.0, 1.0 ).x * -8.0 + 1.0 ) * 0.6;

  // ── 5. joints, end grain and nails ────────────────────────────────────────
  float joint = apxSat( 1.0 - min( bl, 1.0 - bl ) * boards * 3.2 );
  // Two cross-cuts per tile: the end grain drinks water and goes much darker.
  float endBand = apxSat( 1.0 - abs( fract( uv.y * 2.0 ) - 0.5 ) * 12.0 );
  vec2  nv    = apxRivets( vec2( bl, uv.y ), vec2( 1.0, 8.0 ), 0.10 );
  float nail  = nv.x * step( 0.62, nv.y ) * endBand;

  // ── albedo ────────────────────────────────────────────────────────────────
  vec3 early = mix( apxHex( 168.0, 132.0,  88.0 ), apxHex( 146.0, 110.0,  70.0 ), fract( rnd * 5.1 ) );
  vec3 late  = mix( apxHex(  96.0,  66.0,  40.0 ), apxHex(  74.0,  50.0,  32.0 ), fract( rnd * 2.7 ) );
  vec3 wood  = mix( early, late, grain );
  wood = mix( wood, wood * 1.16, rays );
  wood = mix( wood, apxHex( 52.0, 36.0, 24.0 ), knot * 0.85 );
  wood = mix( wood, apxHex( 38.0, 26.0, 18.0 ), knotRings * 0.5 );
  wood = mix( wood, wood * 0.72, endBand * 0.45 );

  // Silvering desaturates toward a cool grey and lifts the value.
  vec3 weathered = mix( apxDesaturate( wood, 0.85 ) * 1.25, apxHex( 148.0, 144.0, 138.0 ), 0.45 );
  vec3 albedo = mix( wood, weathered, silver * 0.80 );

  albedo = mix( albedo, albedo * 0.45, joint * 0.85 );
  albedo = mix( albedo, albedo * 0.55, split * 0.7 );
  albedo = mix( albedo, apxHex( 96.0, 92.0, 88.0 ), nail * 0.8 );

  float dust = apxFbmValue( uv * 5.0 + 55.0, 5.0, 3, 0.6 );
  albedo = mix( albedo, apxPalDust(), dust * 0.10 );

  // ── height ────────────────────────────────────────────────────────────────
  // Raised grain: erosion removes earlywood, so height tracks the grain value
  // and the effect is scaled by how weathered the board is.
  float h = 0.60
          + ( grain - 0.5 ) * 0.06 * ( 0.4 + silver * 1.6 )
          + knot * 0.030
          - knotRings * 0.020
          - joint * 0.170
          - split * 0.120
          - splinter * 0.060
          - nail * 0.090
          + ( dust - 0.5 ) * 0.01;
  s.height = clamp( h, 0.0, 1.0 );

  float rough = 0.78 + ( grain - 0.5 ) * 0.10;
  rough = mix( rough, 0.94, silver );        // silvered timber is dead matte
  rough = mix( rough, 0.62, knot * 0.8 );    // resinous knots stay slick
  rough = mix( rough, 0.97, split );
  rough = mix( rough, 0.55, nail * 0.8 );
  s.rough = clamp( rough, 0.35, 1.0 );

  s.metal = nail * 0.8;                      // the nail head really is steel

  float ao = smoothstep( 0.24, 0.70, s.height );
  ao = mix( ao, ao * 0.40, joint );
  ao = mix( ao, ao * 0.55, split );
  s.ao = clamp( mix( 0.42, 1.0, ao ), 0.0, 1.0 );

  s.albedo = albedo;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Foliage — leaf card atlas with alpha cut-out
// ─────────────────────────────────────────────────────────────────────────────

const FOLIAGE_GLSL = /* glsl */ `
/**
 * A single leaf in cell-local space, centred on (0.5, 0.5).
 * @return x = coverage (0..1), y = midrib/vein mask, z = 0..1 tip-to-base
 *
 * The blade is an intersection of two circular arcs (a "vesica"), which is a
 * much better leaf silhouette than an ellipse - it has the pointed tip and
 * apex that make the shape read instantly.
 */
vec3 nfLeaf( vec2 p, float aspect, float wobble ) {
  vec2 q = p - 0.5;
  q.x *= aspect;
  // Ragged margin: perturb the radius with a periodic-in-angle noise.
  float ang = atan( q.y, q.x );
  float edge = ( apxValue( vec2( ang * 3.0, length( q ) * 6.0 ) * 4.0, 24.0 ) - 0.5 ) * wobble;
  float r = length( q ) + edge;

  // Vesica: two offset discs. |y| term pinches the ends into points.
  float blade = 1.0 - smoothstep( 0.30, 0.34, r + abs( q.y ) * 0.55 );

  // Midrib plus pinnate side veins angled toward the tip.
  float mid = apxSat( 1.0 - abs( q.x ) * 42.0 );
  float side = apxSat( 1.0 - abs( fract( ( q.y * 9.0 + abs( q.x ) * 14.0 ) ) - 0.5 ) * 9.0 );
  float vein = apxSat( mid + side * 0.55 * smoothstep( 0.0, 0.06, abs( q.x ) ) ) * blade;

  float tip = apxSat( q.y * 1.6 + 0.5 );
  return vec3( blade, vein, tip );
}

void nfSurface( vec2 uv, inout NfSurface s ) {
  // 3x3 leaves per tile, each rotated and scaled by its cell hash so the atlas
  // does not read as a grid. Rotation happens in CELL-LOCAL space only, which
  // is safe for tiling (see the note on apxRot in noise.ts).
  float n = 3.0;
  vec2 p = uv * n;
  vec2 cell = floor( p );
  vec2 local = fract( p );

  float cover = 0.0;
  float vein = 0.0;
  float tip = 0.0;
  float idAcc = 0.0;

  // Sample the 3x3 neighbourhood so leaves overlap across cell borders - a
  // single-cell lookup leaves obvious gaps at every cell boundary.
  for ( int y = -1; y <= 1; y++ ) {
    for ( int x = -1; x <= 1; x++ ) {
      vec2 g = vec2( float( x ), float( y ) );
      vec2 c = mod( cell + g, vec2( n ) );
      vec3 rnd = apxHash23( c + 2.3 );
      // Jittered centre inside the neighbour cell.
      vec2 centre = g + 0.5 + ( rnd.xy - 0.5 ) * 0.7;
      vec2 lp = ( local - centre ) / mix( 0.85, 1.35, rnd.z ) + 0.5;
      lp = apxRot( lp - 0.5, rnd.x * APX_TAU ) + 0.5;
      vec3 leaf = nfLeaf( lp, 2.3, 0.055 );
      if ( leaf.x > cover ) {
        cover = leaf.x;
        vein = leaf.y;
        tip = leaf.z;
        idAcc = rnd.y;
      }
    }
  }

  // ── colour ────────────────────────────────────────────────────────────────
  // Dry-climate foliage: olive/sage rather than a lush temperate green, with
  // sun-scorched tips and dust on every upper surface.
  vec3 deep  = apxHex(  58.0,  74.0,  42.0 );
  vec3 light = apxHex( 118.0, 132.0,  74.0 );
  vec3 dryTip = apxHex( 142.0, 116.0,  62.0 );

  float mott = apxFbmValue( uv * 22.0 + idAcc * 30.0, 22.0, 3, 0.55 );
  vec3 leafCol = mix( deep, light, mix( 0.25, 0.85, fract( idAcc * 6.7 ) ) );
  leafCol *= mix( 0.86, 1.14, mott );
  leafCol = mix( leafCol, dryTip, apxSat( ( tip - 0.55 ) * 2.2 ) * mix( 0.1, 0.8, fract( idAcc * 11.1 ) ) );
  leafCol = mix( leafCol, leafCol * 1.25, vein * 0.5 );

  float dust = apxFbmValue( uv * 8.0 + 9.0, 8.0, 3, 0.6 );
  leafCol = mix( leafCol, apxPalDust() * 0.8, dust * 0.18 );

  s.albedo = leafCol;
  // Hard cut-out. alphaTest on the material turns this into a crisp silhouette
  // that still writes depth, which is what lets foliage self-shadow properly.
  s.alpha = smoothstep( 0.35, 0.62, cover );

  s.height = clamp( 0.5 + vein * 0.22 + ( mott - 0.5 ) * 0.10, 0.0, 1.0 );
  // Leaf cuticle is waxy: noticeably smoother than anything else in the scene,
  // which is what produces the sun glint through a canopy.
  s.rough = clamp( mix( 0.52, 0.78, mott ) - vein * 0.10 + dust * 0.20, 0.25, 1.0 );
  s.metal = 0.0;
  s.ao = clamp( 0.72 + vein * 0.20, 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Flesh — enemy skin
// ─────────────────────────────────────────────────────────────────────────────

const FLESH_GLSL = /* glsl */ `
void nfSurface( vec2 uv, inout NfSurface s ) {
  // ── pores and micro-relief ────────────────────────────────────────────────
  // Skin's micro-structure is a network of shallow creases (the "skin lines")
  // with pores at the junctions. Voronoi edges give the creases directly.
  float lines = 1.0 - smoothstep( 0.0, 0.055, apxVoronoiEdge( uv * 60.0, 60.0, 0.9 ) );
  vec3  pw    = apxWorley( uv * 90.0, 90.0, 1.0 );
  float pore  = apxSat( 1.0 - pw.x * 6.0 ) * step( 0.55, pw.z );
  float micro = apxFbmValue( uv * 150.0, 150.0, 2, 0.5 );

  // ── pigmentation ──────────────────────────────────────────────────────────
  float melanin = apxFbm( uv * 3.0, 3.0, 4, 2.0, 0.6 );
  float blush   = apxFbmValue( uv * 5.0 + 21.0, 5.0, 3, 0.6 );

  vec3 baseSkin = mix( apxHex( 176.0, 132.0, 104.0 ), apxHex( 128.0,  90.0,  66.0 ), melanin );
  // Sub-dermal capillary flush: the reddening that makes skin read as alive.
  vec3 flush    = apxHex( 178.0,  96.0,  80.0 );
  vec3 albedo   = mix( baseSkin, flush, apxSat( ( blush - 0.45 ) * 1.6 ) * 0.35 );

  // Veins: darker, cooler, low-frequency and directional.
  float vein = apxSat( 1.0 - apxWorleyAniso( uv, 5.0, 14.0, 1.0 ).x * 6.0 );
  albedo = mix( albedo, albedo * vec3( 0.80, 0.84, 0.94 ), vein * 0.22 );

  // Creases are darker (occlusion plus thicker stratum corneum).
  albedo = mix( albedo, albedo * 0.78, lines * 0.55 );
  albedo = mix( albedo, albedo * 0.70, pore * 0.4 );

  // Grime: nobody in a compound firefight has clean skin.
  float grime = apxSat( ( apxFbmValue( uv * 4.0 + 61.0, 4.0, 4, 0.6 ) - 0.52 ) * 2.6 );
  albedo = mix( albedo, apxHex( 78.0, 66.0, 56.0 ), grime * 0.30 );

  s.albedo = albedo;
  s.height = clamp( 0.62 - lines * 0.10 - pore * 0.16 + ( micro - 0.5 ) * 0.06 + vein * 0.03, 0.0, 1.0 );

  // Skin is oily and never uniformly rough: the T-zone shine is what stops a
  // character looking like painted clay.
  float oil = apxSat( ( apxFbmValue( uv * 6.0 + 200.0, 6.0, 3, 0.6 ) - 0.42 ) * 2.0 );
  float rough = mix( 0.62, 0.42, oil );
  rough = mix( rough, 0.78, lines * 0.5 );
  rough = mix( rough, 0.84, grime * 0.6 );
  s.rough = clamp( rough, 0.28, 1.0 );
  s.metal = 0.0;
  s.ao = clamp( 0.80 - lines * 0.22 - pore * 0.15, 0.0, 1.0 );
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Recipes
// ─────────────────────────────────────────────────────────────────────────────

export const ORGANIC_RECIPES: SurfaceRecipe[] = [
  {
    id: 'wood',
    surface: 'wood',
    // 5 boards on a 1.2 m tile = 240 mm, a real scaffold plank. The previous
    // 1.8 m gave 360 mm boards, which is wider than any timber in this
    // compound would be, and it also disagreed with the level's hardcoded 1.2.
    tileMeters: 1.2,
    size: 1024,
    ormDivisor: 2,
    seed: 167.0,
    glsl: WOOD_GLSL,
    note: 'planks at 240 mm, rings + rays, knots, raised grain silvering, nail heads',
    tuning: {
      normalStrength: 0.68,
      detailScale: 10.0,
      detailStrength: 0.65,
      detailScale2: 2.0,
      detailStrength2: 0.35,
      detailAlbedo: 0.20,
      macroScale: 0.09,
      macroStrength: 0.60,
      macroRough: 0.30,
      grimeHeight: 0.35,
      grimeStrength: 0.60,
      dustStrength: 0.55,
    },
  },
  {
    id: 'foliage',
    surface: 'foliage',
    tileMeters: 0.9,
    size: 1024,
    ormDivisor: 2,
    seed: 181.0,
    physical: true,
    glsl: FOLIAGE_GLSL,
    note: 'leaf cut-out atlas with veins, scorched tips, waxy cuticle',
    params: {
      // Cut-out rather than blending: alphaTest keeps foliage in the opaque
      // pass so it depth-sorts and shadow-maps correctly, which alpha blending
      // never manages for overlapping cards.
      transparent: false,
      alphaTest: 0.42,
      side: THREE.DoubleSide,
      // Sheen approximates the forward-scatter rim you see when a leaf is
      // backlit. Real transmission would force every leaf through the
      // transmission render pass, which is far too expensive for a canopy.
      sheen: 0.55,
      sheenRoughness: 0.75,
      shadowSide: THREE.DoubleSide,
    },
    tuning: {
      normalStrength: 0.38, detailScale: 6.0, detailStrength: 0.30,
      // A leaf is a thin card: a coarse detail tap would give it relief it
      // does not have, and micro grain would read as blight.
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.10,
      macroScale: 0.20, macroStrength: 0.40, macroRough: 0.18,
      dustStrength: 0.45,
    },
  },
  {
    id: 'flesh',
    surface: 'flesh',
    tileMeters: 0.6,
    size: 512,
    ormDivisor: 2,
    seed: 193.0,
    physical: true,
    glsl: FLESH_GLSL,
    note: 'pores, skin lines, capillary flush, oily T-zone',
    params: {
      // A warm sheen is the cheapest convincing stand-in for subsurface
      // scattering: it adds a soft reddish falloff at grazing angles, which is
      // the dominant visual cue of translucent skin at gameplay distance.
      sheen: 0.65,
      sheenRoughness: 0.85,
      clearcoat: 0.10,
      clearcoatRoughness: 0.7,
    },
    tuning: {
      normalStrength: 0.32, detailScale: 5.0, detailStrength: 0.28,
      detailScale2: 0.0, detailStrength2: 0.0, detailAlbedo: 0.08,
      macroScale: 0.40, macroStrength: 0.25, macroRough: 0.12,
    },
  },
];
