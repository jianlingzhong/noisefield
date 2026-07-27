/**
 * NOISEFIELD — surface registry.
 *
 * The full recipe set plus the fallback mapping from a gameplay `SurfaceType`
 * to the material id that best represents it. Several ids share a SurfaceType
 * (plaster, brick, cinderblock and asphalt are all `concrete` for ballistics
 * and footstep purposes) — the mapping picks the canonical one.
 */

import type { SurfaceType } from '../../core/Types';
import { GROUND_RECIPES } from './ground';
import { MASONRY_RECIPES } from './masonry';
import { METAL_RECIPES } from './metals';
import { ORGANIC_RECIPES } from './organic';
import type { SurfaceRecipe } from './Recipe';
import { SYNTHETIC_RECIPES } from './synthetic';

export const ALL_RECIPES: readonly SurfaceRecipe[] = [
  ...MASONRY_RECIPES,
  ...GROUND_RECIPES,
  ...METAL_RECIPES,
  ...ORGANIC_RECIPES,
  ...SYNTHETIC_RECIPES,
];

/** Canonical material id for each gameplay surface class. */
export const SURFACE_DEFAULT_ID: Readonly<Record<SurfaceType, string>> = {
  concrete: 'concrete',
  metal: 'metal',
  wood: 'wood',
  sand: 'sand',
  dirt: 'dirt',
  glass: 'glass',
  water: 'water',
  flesh: 'flesh',
  foliage: 'foliage',
  fabric: 'fabric',
  rubber: 'rubber',
};

export type { SurfaceRecipe } from './Recipe';
export { DEFAULT_TUNING } from './Recipe';
export type { RecipeTuning } from './Recipe';
