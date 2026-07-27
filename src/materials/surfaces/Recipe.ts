/**
 * NOISEFIELD — surface recipe contract.
 *
 * A recipe is a pure description: a GLSL body that implements
 * `void nfSurface( vec2 uv, inout NfSurface s )` plus the metadata the
 * MaterialLibrary needs to bake it and wire up a three.js material. Recipes
 * never touch the renderer, which keeps them trivially testable and lets the
 * library re-bake everything at a different resolution without edits.
 */

import type * as THREE from 'three';
import type { SurfaceType } from '../../core/Types';

/**
 * Shader-patch parameters. These drive the `onBeforeCompile` injection that
 * turns a tiled texture into a *surface*: a high-frequency detail normal at a
 * much smaller UV scale, plus a low-frequency macro tint at a much larger one.
 * Between them they break up the two things that give tiling away — repeated
 * micro-detail and repeated large blotches.
 */
export interface RecipeTuning {
  /** Slope multiplier for the height → normal derivation. */
  normalStrength: number;
  /** UV multiplier for the shared detail normal. Typically 8–16. 0 disables. */
  detailScale: number;
  /** Blend weight of the detail normal, 0..1. */
  detailStrength: number;
  /**
   * UV multiplier for a SECOND, much coarser tap of the same detail normal.
   * Typically `detailScale * 0.15..0.3`. 0 disables the second tap.
   *
   * One detail tap only covers one octave band, so it fixes the surface at
   * exactly one viewing distance and nowhere else. Two taps five-ish times
   * apart cover ~4 mm to ~15 cm simultaneously, which spans the whole range a
   * player moves through, and — because the coarse tap's period is unrelated
   * to the base tile's — it also perturbs the base map's own repeat.
   */
  detailScale2: number;
  /** Blend weight of the coarse detail tap, 0..1. */
  detailStrength2: number;
  /**
   * How strongly the detail field's HEIGHT (carried in the detail normal
   * map's alpha) multiplies albedo. 0 disables.
   *
   * This is the only source of sub-texel colour grain available. A 1024^2 sand
   * map covering 4 m has 3.9 mm texels, so grain, grit and the shadowed side
   * of every pebble are below its Nyquist limit and simply cannot be baked —
   * which is why a technically-correct granular albedo still reads as a smooth
   * wash. Sampling the detail map's alpha at 12-16x the base rate puts that
   * missing band back for the cost of one channel we already paid for.
   */
  detailAlbedo: number;
  /** UV multiplier for the macro variation. Typically 1/8. 0 disables. */
  macroScale: number;
  /** Blend weight of the macro tint/roughness modulation, 0..1. */
  macroStrength: number;
  /**
   * How hard the macro field drives ROUGHNESS, as an absolute offset at full
   * macro contrast. 0.35 means the surface roughness wanders by ±0.35 over the
   * macro period. This is deliberately large: spatial roughness variance is
   * the single strongest cue that a surface is a real substance rather than a
   * coloured plane, and it is the first thing a procedural material loses.
   */
  macroRough: number;
  /**
   * Anti-tiling domain warp, in base-UV units (so 0.25 = a quarter of a tile).
   * 0 disables. Requires the macro field, which supplies the warp vector.
   */
  uvWarp: number;
  /** UV multiplier for the warp field. Must be much lower than `macroScale`. */
  uvWarpScale: number;
  /**
   * World-space weathering: metres of splash-back / dust grime rising from
   * y = 0 up a vertical surface. 0 disables. Real walls are filthy for the
   * first half-metre and comparatively clean above it, and no amount of
   * tile-space noise can produce that because a tile does not know where the
   * ground is.
   */
  grimeHeight: number;
  /** Strength of the base grime band, 0..1. */
  grimeStrength: number;
  /**
   * Strength of top-face dust accumulation, 0..1. Applied to up-facing
   * geometry (ledges, parapets, container roofs, crate tops) in world space.
   */
  dustStrength: number;
  /** Project in world space instead of using mesh UVs. */
  triplanar: boolean;
  /** World metres per triplanar tile (only used when `triplanar`). */
  triplanarScale: number;
}

export const DEFAULT_TUNING: RecipeTuning = {
  normalStrength: 1.0,
  detailScale: 11.0,
  detailStrength: 0.70,
  detailScale2: 2.2,
  detailStrength2: 0.45,
  detailAlbedo: 0.22,
  macroScale: 0.125,
  macroStrength: 0.6,
  macroRough: 0.30,
  uvWarp: 0.0,
  uvWarpScale: 0.045,
  grimeHeight: 0.0,
  grimeStrength: 0.0,
  dustStrength: 0.0,
  triplanar: false,
  triplanarScale: 2.0,
};

interface RecipeBase {
  /** Stable lookup key, e.g. `concrete`, `steel_painted`. */
  id: string;
  /** Gameplay classification: drives impacts, footsteps, penetration. */
  surface: SurfaceType;
  /** World-space metres covered by one texture tile. */
  tileMeters: number;
  /** Base albedo/normal resolution before the quality scale is applied. */
  size: number;
  /** ORM resolution divisor. 2 = half of `size`; roughness rarely needs parity. */
  ormDivisor?: number;
  /** Decorrelation constant fed to the shader as `uSeed`. */
  seed: number;
  /** GLSL defining `void nfSurface( vec2 uv, inout NfSurface s )`. */
  glsl: string;
  /** Keep the height field around and expose it as `displacementMap`. */
  keepHeight?: boolean;
  tuning?: Partial<RecipeTuning>;
  /** Human-readable note surfaced in the init log. */
  note?: string;
}

/** Recipes that only need the metal/rough workflow — cheaper to shade. */
interface StandardRecipe extends RecipeBase {
  physical?: false;
  params?: THREE.MeshStandardMaterialParameters;
}

/** Recipes that buy real quality from clearcoat / transmission / sheen. */
interface PhysicalRecipe extends RecipeBase {
  physical: true;
  params?: THREE.MeshPhysicalMaterialParameters;
}

export type SurfaceRecipe = StandardRecipe | PhysicalRecipe;
