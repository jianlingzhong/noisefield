/**
 * NOISEFIELD — GLSL procedural-noise library.
 *
 * Source is written in ESSL1 style. three.js r185 transparently recompiles every
 * non-raw ShaderMaterial as '#version 300 es' with compatibility defines
 * (texture2D->texture, varying->in/out, gl_FragColor->pc_fragColor), so this
 * actually lands as ESSL3. The loops are nonetheless written in the stricter
 * ESSL1 Appendix-A form "for (int i = 0; i < CONST; ++i)" with an early break,
 * which keeps the source valid under either dialect.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * THE TILING CONTRACT — read this before adding a function or a call site.
 * ─────────────────────────────────────────────────────────────────────────────
 * Every texture we bake is sampled with RepeatWrapping, so one visible seam
 * ruins the whole surface. The rule that makes seamlessness *structural*
 * rather than something we eyeball is:
 *
 *   Every hash is taken on an INTEGER LATTICE COORDINATE THAT HAS BEEN WRAPPED
 *   WITH mod(cell, period) BEFORE HASHING.
 *
 * That gives two obligations at every call site:
 *
 *   (T1) PERIODS MUST BE POSITIVE INTEGERS. A period of 2.5 shifts the lattice
 *        alignment itself across the seam and tiles nothing.
 *
 *   (T2) THE DOMAIN DELTA PER UV TILE MUST BE A MULTIPLE OF THE PERIOD, PER
 *        AXIS. If q(uv + e) == q(uv) + d, seamlessness needs d ≡ 0 (mod P).
 *        In practice: pass q = uv * P and period = P.
 *        A coordinate that is *exactly periodic* (e.g. a cell-local fract, or
 *        a plank-local coordinate that resets at the tile edge) has d = 0 and
 *        so is safe at any scale.
 *
 * Because (T2) is per-axis, every core function takes a `vec2` period. The
 * scalar overloads are sugar for the isotropic case. Anisotropic fields —
 * brushed metal, runoff streaks, wood grain, weave — MUST use the vec2 form;
 * passing max(px, py) silently breaks the short axis.
 *
 * Corollaries:
 *   • Octave lacunarity must be an INTEGER (we use 2) and the period is scaled
 *     by the same factor each octave, so every octave stays wrapped.
 *   • Adding a CONSTANT offset to the domain is safe (it shifts the point and
 *     its lattice index equally, so the residue is unchanged).
 *   • ROTATING or NON-INTEGER-SCALING the global domain is NOT safe.
 *   • Domain warping is safe as long as the warp field is periodic with the
 *     same period — which it is, being built from these same functions.
 *
 * Deliberate omission: true simplex noise. Its skew factor (sqrt(3)-1)/2 is
 * irrational, so the skewed lattice can never be wrapped to an integer period.
 * apxGradient (Perlin) plus apxGradientRot (a second lattice rotated by 90
 * degrees, which IS integer-preserving) removes the axis-aligned artefacts
 * that make people reach for simplex, and still tiles exactly.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Hashing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Bit-mixing hashes derived from Dave Hoskins' "Hash without Sine"
 * (https://www.shadertoy.com/view/4djSRW). The magic constants and the
 * swizzle pattern are his; this is a direct adaptation rather than an
 * independent implementation in the same spirit, and is credited as such.
 *
 * Chosen over the classic
 * fract(sin(dot(p,k)) * 43758.5453) because sin() precision is
 * driver-dependent: the same shader produces visibly different textures on
 * mobile vs desktop. These use only multiply/fract/dot and are stable.
 *
 * Inputs are always small non-negative integers (wrapped lattice coordinates,
 * < 512), so we stay far away from float32 mantissa exhaustion.
 *
 * Every hash is salted with uSeed, a per-surface constant the host declares
 * (TextureForge does). Salting the INPUT rather than the mixing constants is
 * deliberate: folding a value of a few hundred into the 33.33 term would push
 * the intermediate products past float32's usable mantissa and visibly quantise
 * the result. Because the salt is constant across the whole texture it cannot
 * affect periodicity - identical wrapped cells still hash identically.
 *
 * REQUIRES the host to declare: uniform float uSeed;
 */
export const GLSL_HASH = /* glsl */ `
#ifndef APX_HASH_INCLUDED
#define APX_HASH_INCLUDED

#define APX_TAU 6.283185307179586
#define APX_PI  3.141592653589793

float apxHash11( float p ) {
  vec3 p3 = fract( vec3( p + uSeed ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}

float apxHash21( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx + uSeed ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}

vec2 apxHash22( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx + uSeed ) * vec3( 0.1031, 0.1030, 0.0973 ) );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.xx + p3.yz ) * p3.zy );
}

vec3 apxHash23( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx + uSeed ) * vec3( 0.1031, 0.1030, 0.0973 ) );
  p3 += dot( p3, p3.yxz + 33.33 );
  return fract( ( p3.xxy + p3.yzz ) * p3.zyx );
}

/** Wrap a lattice coordinate into [0, period). THE tiling primitive. */
vec2 apxWrap( vec2 cell, vec2 period ) { return mod( cell, period ); }
vec2 apxWrap( vec2 cell, float period ) { return mod( cell, vec2( period ) ); }

#endif
`;

// ─────────────────────────────────────────────────────────────────────────────
// Small numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

export const GLSL_UTIL = /* glsl */ `
#ifndef APX_UTIL_INCLUDED
#define APX_UTIL_INCLUDED

float apxSat( float x ) { return clamp( x, 0.0, 1.0 ); }
vec3  apxSat3( vec3 x ) { return clamp( x, vec3( 0.0 ), vec3( 1.0 ) ); }

/** Linear remap of [a,b] onto [0,1], clamped. */
float apxRemap( float x, float a, float b ) { return apxSat( ( x - a ) / max( b - a, 1e-5 ) ); }

/** Band pass: 1 inside [a,b], smoothly falling off over width w. */
float apxBand( float x, float a, float b, float w ) {
  return smoothstep( a - w, a, x ) * ( 1.0 - smoothstep( b, b + w, x ) );
}

/** Polynomial smooth minimum — used to fuse overlapping height blobs. */
float apxSmin( float a, float b, float k ) {
  float h = apxSat( 0.5 + 0.5 * ( b - a ) / k );
  return mix( b, a, h ) - k * h * ( 1.0 - h );
}

float apxLuma( vec3 c ) { return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) ); }

vec3 apxDesaturate( vec3 c, float t ) { return mix( c, vec3( apxLuma( c ) ), t ); }

/**
 * sRGB to linear. Palette constants are authored as sRGB hex (what an artist
 * reads off a colour picker); the bake target is linear-light, so every
 * literal colour passes through this exactly once.
 */
vec3 apxSRGBToLinear( vec3 c ) {
  return mix( c / 12.92, pow( ( c + 0.055 ) / 1.055, vec3( 2.4 ) ), step( vec3( 0.04045 ), c ) );
}

/** An 8-bit sRGB triple, converted to linear. */
vec3 apxHex( float r, float g, float b ) {
  return apxSRGBToLinear( vec3( r, g, b ) / 255.0 );
}

/** Triangle wave with period 1 — tiles trivially. */
float apxTri( float x ) { return abs( fract( x ) * 2.0 - 1.0 ); }

/**
 * Rotate a 2D coordinate. NOTE: this breaks lattice periodicity if applied to
 * the GLOBAL domain. Only ever use it on cell-local coordinates, which are
 * confined to one cell and therefore cannot leak across a tile boundary.
 */
vec2 apxRot( vec2 p, float a ) {
  float c = cos( a ), s = sin( a );
  return vec2( c * p.x - s * p.y, s * p.x + c * p.y );
}

#endif
`;

// ─────────────────────────────────────────────────────────────────────────────
// Core noise. Every primitive takes a vec2 period (see obligation T2).
// ─────────────────────────────────────────────────────────────────────────────

export const GLSL_NOISE = /* glsl */ `
#ifndef APX_NOISE_INCLUDED
#define APX_NOISE_INCLUDED

// ── value noise ──────────────────────────────────────────────────────────────

/** Bilinear value noise on a wrapped lattice. Cheapest tileable field we have. */
float apxValue( vec2 p, vec2 period ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  // Quintic fade: C2 continuous, so derived normal maps show no faceting.
  vec2 u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );
  float a = apxHash21( apxWrap( i + vec2( 0.0, 0.0 ), period ) );
  float b = apxHash21( apxWrap( i + vec2( 1.0, 0.0 ), period ) );
  float c = apxHash21( apxWrap( i + vec2( 0.0, 1.0 ), period ) );
  float d = apxHash21( apxWrap( i + vec2( 1.0, 1.0 ), period ) );
  return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );
}
float apxValue( vec2 p, float period ) { return apxValue( p, vec2( period ) ); }

// ── gradient (Perlin) noise ──────────────────────────────────────────────────

vec2 apxGradVec( vec2 cell, vec2 period ) {
  float h = apxHash21( apxWrap( cell, period ) ) * APX_TAU;
  return vec2( cos( h ), sin( h ) );
}

/** Perlin gradient noise, remapped to [0,1]. Exactly periodic on 'period'. */
float apxGradient( vec2 p, vec2 period ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  vec2 u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );
  float a = dot( apxGradVec( i + vec2( 0.0, 0.0 ), period ), f - vec2( 0.0, 0.0 ) );
  float b = dot( apxGradVec( i + vec2( 1.0, 0.0 ), period ), f - vec2( 1.0, 0.0 ) );
  float c = dot( apxGradVec( i + vec2( 0.0, 1.0 ), period ), f - vec2( 0.0, 1.0 ) );
  float d = dot( apxGradVec( i + vec2( 1.0, 1.0 ), period ), f - vec2( 1.0, 1.0 ) );
  // Perlin's 2D range is +/- sqrt(2)/2; scale to fill [0,1].
  return 0.5 + mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );
}
float apxGradient( vec2 p, float period ) { return apxGradient( p, vec2( period ) ); }

/**
 * Gradient noise averaged with a copy sampled on a 90-degree-rotated lattice.
 * A 90-degree rotation maps integers to integers, so periodicity survives —
 * this is the tileable substitute for simplex noise (see the file header).
 */
float apxGradientRot( vec2 p, float period ) {
  float n0 = apxGradient( p, period );
  float n1 = apxGradient( vec2( p.y, period - p.x ), period );
  return 0.5 * ( n0 + n1 );
}

// ── worley / cellular ────────────────────────────────────────────────────────

/**
 * Worley cellular noise on a wrapped lattice.
 * Returns vec3( F1, F2, cellRandom ) where F1/F2 are Euclidean distances to
 * the nearest and second-nearest feature point and cellRandom is a stable
 * random value for the owning cell (per-pebble / per-brick variation).
 *
 * 'jitter' in [0,1] moves the feature point off the cell centre; 1 is fully
 * random (organic), 0 is a perfect grid (man-made).
 */
vec3 apxWorley( vec2 p, vec2 period, float jitter ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  float f1 = 8.0;
  float f2 = 8.0;
  float id = 0.0;
  for ( int y = -1; y <= 1; y++ ) {
    for ( int x = -1; x <= 1; x++ ) {
      vec2 g = vec2( float( x ), float( y ) );
      vec2 cell = apxWrap( i + g, period );
      vec2 o = 0.5 + ( apxHash22( cell ) - 0.5 ) * jitter;
      vec2 r = g + o - f;
      float d = dot( r, r );
      if ( d < f1 ) {
        f2 = f1;
        f1 = d;
        id = apxHash21( cell + 17.31 );
      } else if ( d < f2 ) {
        f2 = d;
      }
    }
  }
  return vec3( sqrt( f1 ), sqrt( f2 ), id );
}
vec3 apxWorley( vec2 p, float period, float jitter ) { return apxWorley( p, vec2( period ), jitter ); }

/**
 * Anisotropic worley: cells stretched into streaks.
 * 'uv' must advance exactly 1.0 per tile (a true surface UV); the periods are
 * applied per axis, which is what keeps the short axis seamless.
 */
vec3 apxWorleyAniso( vec2 uv, float periodX, float periodY, float jitter ) {
  vec2 period = vec2( periodX, periodY );
  return apxWorley( uv * period, period, jitter );
}

/**
 * Distance to the nearest Voronoi EDGE (Inigo Quilez's two-pass formulation).
 *
 * Why not F2 - F1? Because that produces lines whose width varies with the
 * local cell geometry — thick at corners, thin along long edges. Real cracks
 * have roughly constant width. Pass 1 locates the owning feature point; pass 2
 * measures the perpendicular distance to the bisector between it and each
 * neighbour, which is a true distance-to-edge and therefore uniform.
 */
float apxVoronoiEdge( vec2 p, vec2 period, float jitter ) {
  vec2 i = floor( p );
  vec2 f = fract( p );

  vec2 mg = vec2( 0.0 );
  vec2 mo = vec2( 0.0 );
  float md = 8.0;
  for ( int y = -1; y <= 1; y++ ) {
    for ( int x = -1; x <= 1; x++ ) {
      vec2 g = vec2( float( x ), float( y ) );
      vec2 o = 0.5 + ( apxHash22( apxWrap( i + g, period ) ) - 0.5 ) * jitter;
      vec2 r = g + o - f;
      float d = dot( r, r );
      if ( d < md ) { md = d; mg = g; mo = o; }
    }
  }

  vec2 mr = mg + mo - f;
  md = 8.0;
  for ( int y = -2; y <= 2; y++ ) {
    for ( int x = -2; x <= 2; x++ ) {
      vec2 g = mg + vec2( float( x ), float( y ) );
      vec2 o = 0.5 + ( apxHash22( apxWrap( i + g, period ) ) - 0.5 ) * jitter;
      vec2 r = g + o - f;
      vec2 diff = r - mr;
      if ( dot( diff, diff ) > 1e-5 ) {
        md = min( md, dot( 0.5 * ( mr + r ), normalize( diff ) ) );
      }
    }
  }
  return md;
}
float apxVoronoiEdge( vec2 p, float period, float jitter ) { return apxVoronoiEdge( p, vec2( period ), jitter ); }

// ── fractal sums ─────────────────────────────────────────────────────────────

// ESSL1 requires a compile-time loop bound; the runtime 'octaves' argument is
// honoured via an early break, which Appendix A explicitly permits.
#define APX_MAX_OCTAVES 9

/**
 * Fractional Brownian motion.
 * 'lacunarity' MUST be an integer (2 or 3) — see the tiling contract. The
 * period is scaled by the same factor per octave so every octave stays
 * wrapped.
 */
float apxFbm( vec2 p, vec2 period, int octaves, float lacunarity, float gain ) {
  float amp = 0.5;
  float sum = 0.0;
  float norm = 0.0;
  vec2 q = p;
  vec2 per = period;
  for ( int i = 0; i < APX_MAX_OCTAVES; i++ ) {
    if ( i >= octaves ) break;
    sum += amp * apxGradient( q, per );
    norm += amp;
    q *= lacunarity;
    per *= lacunarity;
    amp *= gain;
  }
  return sum / max( norm, 1e-5 );
}
float apxFbm( vec2 p, float period, int octaves, float lacunarity, float gain ) {
  return apxFbm( p, vec2( period ), octaves, lacunarity, gain );
}

/** fBm over value noise — softer, blobbier, cheaper. Good for stains/damp. */
float apxFbmValue( vec2 p, vec2 period, int octaves, float gain ) {
  float amp = 0.5, sum = 0.0, norm = 0.0;
  vec2 q = p;
  vec2 per = period;
  for ( int i = 0; i < APX_MAX_OCTAVES; i++ ) {
    if ( i >= octaves ) break;
    sum += amp * apxValue( q, per );
    norm += amp;
    q *= 2.0; per *= 2.0; amp *= gain;
  }
  return sum / max( norm, 1e-5 );
}
float apxFbmValue( vec2 p, float period, int octaves, float gain ) {
  return apxFbmValue( p, vec2( period ), octaves, gain );
}

/**
 * Ridged multifractal. 1 - |2n-1| folds the noise about its midline giving
 * sharp creases; squaring sharpens them further. This produces rock fissures,
 * wood grain and wind-carved sand.
 */
float apxRidged( vec2 p, vec2 period, int octaves, float gain ) {
  float amp = 0.5, sum = 0.0, norm = 0.0;
  vec2 q = p;
  vec2 per = period;
  for ( int i = 0; i < APX_MAX_OCTAVES; i++ ) {
    if ( i >= octaves ) break;
    float n = 1.0 - abs( apxGradient( q, per ) * 2.0 - 1.0 );
    sum += amp * n * n;
    norm += amp;
    q *= 2.0; per *= 2.0; amp *= gain;
  }
  return sum / max( norm, 1e-5 );
}
float apxRidged( vec2 p, float period, int octaves, float gain ) {
  return apxRidged( p, vec2( period ), octaves, gain );
}

/** Turbulence: absolute-value fBm. Billowy; used for rust and smoke stains. */
float apxTurbulence( vec2 p, vec2 period, int octaves, float gain ) {
  float amp = 0.5, sum = 0.0, norm = 0.0;
  vec2 q = p;
  vec2 per = period;
  for ( int i = 0; i < APX_MAX_OCTAVES; i++ ) {
    if ( i >= octaves ) break;
    sum += amp * abs( apxGradient( q, per ) * 2.0 - 1.0 );
    norm += amp;
    q *= 2.0; per *= 2.0; amp *= gain;
  }
  return sum / max( norm, 1e-5 );
}
float apxTurbulence( vec2 p, float period, int octaves, float gain ) {
  return apxTurbulence( p, vec2( period ), octaves, gain );
}

/**
 * Worley fBm — stacked cellular octaves. Produces the clustered-granule look
 * that reads as aggregate, gravel or coarse render.
 */
float apxFbmWorley( vec2 p, vec2 period, int octaves, float gain, float jitter ) {
  float amp = 0.5, sum = 0.0, norm = 0.0;
  vec2 q = p;
  vec2 per = period;
  for ( int i = 0; i < APX_MAX_OCTAVES; i++ ) {
    if ( i >= octaves ) break;
    sum += amp * apxWorley( q, per, jitter ).x;
    norm += amp;
    q *= 2.0; per *= 2.0; amp *= gain;
  }
  return sum / max( norm, 1e-5 );
}
float apxFbmWorley( vec2 p, float period, int octaves, float gain, float jitter ) {
  return apxFbmWorley( p, vec2( period ), octaves, gain, jitter );
}

// ── domain warping ───────────────────────────────────────────────────────────

/**
 * Displace the sample point by a vector field built from the same periodic
 * noise. The constant offsets (13.7 / 47.3) decorrelate the two channels
 * without breaking periodicity: a constant shift moves the point and its
 * lattice index equally, so the wrapped residue is unchanged.
 */
vec2 apxWarp( vec2 p, vec2 period, float amount, int octaves ) {
  float wx = apxFbm( p + 13.7, period, octaves, 2.0, 0.5 ) * 2.0 - 1.0;
  float wy = apxFbm( p + 47.3, period, octaves, 2.0, 0.5 ) * 2.0 - 1.0;
  return p + amount * vec2( wx, wy );
}
vec2 apxWarp( vec2 p, float period, float amount, int octaves ) {
  return apxWarp( p, vec2( period ), amount, octaves );
}

/** Two-level warp (warp the warp). Gives the swirling, organic look. */
vec2 apxWarp2( vec2 p, float period, float amount, int octaves ) {
  vec2 q = apxWarp( p, period, amount, octaves );
  return apxWarp( q, period, amount * 0.5, octaves );
}

#endif
`;

// ─────────────────────────────────────────────────────────────────────────────
// Higher-level pattern primitives shared by the surface shaders
// ─────────────────────────────────────────────────────────────────────────────

export const GLSL_PATTERNS = /* glsl */ `
#ifndef APX_PATTERNS_INCLUDED
#define APX_PATTERNS_INCLUDED

/**
 * Running-bond masonry lattice.
 *
 * @param uv      surface UV in [0,1]
 * @param count   units across (x) and courses up (y) — MUST be integers, and
 *                count.y MUST BE EVEN or the half-unit row stagger does not
 *                realign across the tile seam
 * @param stagger row offset as a fraction of a unit (0.5 = running bond)
 * @return  xy = unit-local UV in [0,1]
 *          z  = per-unit random in [0,1]
 *          w  = raw course index
 *
 * WARNING: only xyz are periodic. .w is floor(v * count.y), which grows
 * monotonically and therefore differs by count.y across the tile seam. It is
 * informational only — using it to drive any visible term introduces a seam.
 */
vec4 apxMasonry( vec2 uv, vec2 count, float stagger ) {
  float row = floor( uv.y * count.y );
  float shift = stagger * mod( row, 2.0 );
  vec2 p = vec2( uv.x * count.x + shift, uv.y * count.y );
  vec2 cell = floor( p );
  vec2 local = fract( p );
  float rnd = apxHash21( mod( cell, count ) + 3.19 );
  return vec4( local, rnd, row );
}

/**
 * Distance from a unit-local UV to the unit border, in unit-local units.
 * Feed to smoothstep for mortar joints, or bias negative for spalled edges.
 */
float apxCellBorder( vec2 local ) {
  vec2 d = min( local, 1.0 - local );
  return min( d.x, d.y );
}

/**
 * Directional grime / water-runoff streaks.
 *
 * Runoff is coherent along gravity and incoherent across it, so the field is
 * heavily anisotropic: high frequency across the flow, low frequency along it.
 * The per-axis period is what keeps the short axis seamless — using
 * max(across, along) as a single scalar period silently breaks the other one.
 *
 * @param acrossPeriod integer lattice period across the flow (streak density)
 * @param alongPeriod  integer lattice period along the flow (streak length)
 */
float apxStreaks( vec2 uv, float acrossPeriod, float alongPeriod, int octaves ) {
  vec2 period = vec2( acrossPeriod, alongPeriod );
  return apxFbm( uv * period, period, octaves, 2.0, 0.55 );
}

/**
 * Fine scratch field: very high frequency across the scratch, very low along
 * it, thresholded so only the deepest valleys survive as thin lines.
 *
 * @param density    integer period across the scratch direction
 * @param extent     integer period along it (smaller = longer scratches)
 * @param sharpness  larger = fewer, thinner scratches
 */
float apxScratches( vec2 uv, float density, float extent, float sharpness ) {
  vec3 w = apxWorleyAniso( uv, density, extent, 1.0 );
  // F2-F1 gives the ridge line between cells; invert and sharpen to hairlines.
  float line = 1.0 - apxSat( ( w.y - w.x ) * sharpness );
  // Per-scratch visibility so they do not all have identical depth.
  float vis = step( 0.55, w.z );
  return line * line * vis;
}

/**
 * Regular grid of rounded studs / rivets / bolts.
 * @return x = 0..1 dome height, y = per-rivet random
 */
vec2 apxRivets( vec2 uv, vec2 count, float radius ) {
  vec2 p = uv * count;
  vec2 cell = floor( p );
  vec2 local = fract( p ) - 0.5;
  float d = length( local );
  float dome = apxSat( 1.0 - d / max( radius, 1e-4 ) );
  // sqrt profile gives a hemispherical cap rather than a cone.
  dome = sqrt( max( dome, 0.0 ) );
  return vec2( dome, apxHash21( mod( cell, count ) + 91.7 ) );
}

/**
 * Horizontal weld bead: a raised, rippled ridge at a set of V positions.
 * 'rows' and 'rippleFreq' must be integers so the bead closes at the seam.
 */
float apxWeldBead( vec2 uv, float rows, float width, float rippleFreq ) {
  float p = uv.y * rows;
  float local = abs( fract( p ) - 0.5 ) * 2.0;   // 0 at bead centre, 1 between
  // The ripple modulates bead width so it reads as stacked molten passes.
  float ripple = 0.5 + 0.5 * sin( uv.x * rippleFreq * APX_TAU );
  float w = width * ( 0.75 + 0.5 * ripple );
  return apxSat( 1.0 - local / max( w, 1e-4 ) );
}

/**
 * Radial spider-web fracture used for broken glass.
 * The delta to the hub is wrapped into [-0.5, 0.5] so a fracture whose hub is
 * near an edge continues correctly onto the neighbouring tile.
 * @return x = radial crack mask, y = concentric ring mask, z = distance to hub
 */
vec3 apxSpiderCracks( vec2 uv, vec2 hub, float arms, float rings, float width ) {
  vec2 d = uv - hub;
  d = fract( d + 0.5 ) - 0.5;
  float r = length( d );
  float a = atan( d.y, d.x );
  // Jitter the arm angle so the arms are not a perfect fan.
  float wob = ( apxValue( vec2( a * 2.0, r * 8.0 ) * 3.0, 24.0 ) - 0.5 ) * 0.35;
  float arm = abs( fract( ( a / APX_TAU ) * arms + wob ) - 0.5 ) * 2.0;
  float radial = apxSat( 1.0 - arm / max( width / max( r, 0.02 ), 1e-4 ) );
  float ring = apxSat( 1.0 - abs( fract( r * rings ) - 0.5 ) * 2.0 / max( width * 6.0, 1e-4 ) );
  // Cracks fade out with distance from the impact hub.
  float fade = apxSat( 1.0 - r * 2.2 );
  return vec3( radial * fade, ring * fade * apxSat( 1.0 - r * 3.0 ), r );
}

/**
 * Plain-weave fabric: two orthogonal thread sets where the over/under
 * alternates per cell. 'count' must be an integer for tiling.
 */
vec2 apxWeave( vec2 uv, float count, float threadGap ) {
  vec2 p = uv * count;
  vec2 cell = floor( p );
  vec2 f = fract( p );
  // Rounded thread cross-section: cosine profile across the thread.
  float warp = pow( apxSat( sin( f.x * APX_PI ) ), 0.55 );
  float weft = pow( apxSat( sin( f.y * APX_PI ) ), 0.55 );
  // A checkerboard decides which thread is on top in this cell.
  float over = mod( cell.x + cell.y, 2.0 );
  float top = mix( warp, weft, over );
  float bot = mix( weft, warp, over ) * 0.45;
  float gap = apxSat( 1.0 - threadGap );
  return vec2( max( top, bot ) * gap, over );
}

#endif
`;

/**
 * The complete library, in dependency order. Prepended to every surface
 * shader; the include guards make double-inclusion harmless.
 */
export const NOISE_LIB: string = GLSL_HASH + GLSL_UTIL + GLSL_NOISE + GLSL_PATTERNS;

/**
 * Shared art-direction palette as GLSL constants, so surface shaders stay
 * consistent with each other. Values are sRGB 0-255 triples converted to
 * linear by apxHex.
 *
 * Sun-baked North-African compound at golden hour:
 *   sand      #C2A878   plaster, dunes, dust
 *   concrete  #8A8578   weathered structural concrete
 *   rust      #7A4A32   oxidised steel, iron staining
 *   shadow    #1A2028   blue-shifted ambient occlusion tint
 */
export const GLSL_PALETTE = /* glsl */ `
#ifndef APX_PALETTE_INCLUDED
#define APX_PALETTE_INCLUDED
vec3 apxPalSand()     { return apxHex( 194.0, 168.0, 120.0 ); }
vec3 apxPalConcrete() { return apxHex( 138.0, 133.0, 120.0 ); }
vec3 apxPalRust()     { return apxHex( 122.0,  74.0,  50.0 ); }
vec3 apxPalShadow()   { return apxHex(  26.0,  32.0,  40.0 ); }
vec3 apxPalDust()     { return apxHex( 176.0, 158.0, 126.0 ); }
#endif
`;
