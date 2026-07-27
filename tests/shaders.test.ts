import { describe, expect, it } from 'vitest';
import { ALL_RECIPES } from '../src/materials/surfaces';
import { GLSL_HASH, GLSL_NOISE, GLSL_PALETTE, GLSL_PATTERNS, GLSL_UTIL } from '../src/materials/noise';

/*
 * Guard against GLSL that will not compile.
 *
 * This exists because of a real and expensive bug. Two surface recipes declared
 * a local named `patch`, which is a RESERVED WORD in GLSL ES 3.00. The shader
 * failed to compile, the bake silently returned the renderer's clear colour,
 * and two materials shipped as pure black — albedo 0, ORM 0. Nothing threw.
 * The only symptom was that a wall looked wrong, and it took a pixel readback
 * to find it.
 *
 * A compile error in a procedural texture pipeline is not loud: the failure
 * mode is a plausible-looking dark surface. So it is worth catching statically.
 */

/**
 * Reserved and future-reserved words in GLSL ES 3.00 that are ordinary enough
 * in English to be picked as a variable name. This is deliberately not the full
 * keyword list — `for`, `if` and friends are not realistic mistakes, whereas
 * `sample`, `filter`, `buffer` and `patch` very much are.
 */
const TEMPTING_RESERVED_WORDS = [
  'patch',
  'sample',
  'filter',
  'buffer',
  'shared',
  'input',
  'output',
  'active',
  'common',
  'partition',
  'resource',
  'subroutine',
  'precise',
  'superp',
  'template',
  'this',
  'packed',
  'namespace',
  'using',
  'union',
  'interface',
  'public',
  'static',
  'extern',
  'external',
  'inline',
  'noinline',
  'long',
  'short',
  'half',
  'fixed',
  'unsigned',
  'volatile',
  'restrict',
  'readonly',
  'writeonly',
  'coherent',
  'goto',
  'cast',
  'sizeof',
  'typedef',
] as const;

/** Types that can begin a declaration, i.e. `<type> <name>`. */
const DECL_TYPES =
  '(?:float|int|uint|bool|void|vec2|vec3|vec4|ivec2|ivec3|ivec4|uvec2|uvec3|uvec4|bvec2|bvec3|bvec4|mat2|mat3|mat4|mat2x2|mat2x3|mat2x4|mat3x2|mat3x3|mat3x4|mat4x2|mat4x3|mat4x4)';

/**
 * Strip comments before scanning.
 *
 * Without this the check trips on prose — several recipes legitimately discuss
 * "sampling" or "a patch of rust" in their comments, and flagging those would
 * make the test noise that people learn to ignore.
 */
function stripComments(glsl: string): string {
  return glsl.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
}

/** Every declaration in `glsl` whose declared name is a reserved word. */
function findReservedDeclarations(glsl: string): string[] {
  const body = stripComments(glsl);
  const hits: string[] = [];
  for (const word of TEMPTING_RESERVED_WORDS) {
    // `<type> <reserved>` followed by an assignment, semicolon, comma or paren
    // — i.e. a declaration rather than an incidental substring.
    const decl = new RegExp(`\\b${DECL_TYPES}\\s+${word}\\b\\s*[;=,)\\[]`, 'g');
    if (decl.test(body)) hits.push(`declares "${word}"`);

    // Function parameters: `( ... float patch ...)` is covered above, but a
    // bare `out`/`in`/`inout` qualifier is not.
    const param = new RegExp(`\\b(?:in|out|inout)\\s+${DECL_TYPES}\\s+${word}\\b`, 'g');
    if (param.test(body)) hits.push(`uses "${word}" as a parameter name`);
  }
  return hits;
}

describe('GLSL reserved words', () => {
  it('has recipes to check', () => {
    // A regression where ALL_RECIPES silently became empty would make every
    // other assertion in this file pass vacuously.
    expect(ALL_RECIPES.length).toBeGreaterThan(10);
  });

  it.each(ALL_RECIPES.map((r) => [r.id, r.glsl] as const))(
    'recipe "%s" declares no GLSL ES 3.00 reserved word',
    (_id, glsl) => {
      expect(findReservedDeclarations(glsl)).toEqual([]);
    },
  );

  it.each([
    ['GLSL_HASH', GLSL_HASH],
    ['GLSL_UTIL', GLSL_UTIL],
    ['GLSL_NOISE', GLSL_NOISE],
    ['GLSL_PATTERNS', GLSL_PATTERNS],
    ['GLSL_PALETTE', GLSL_PALETTE],
  ])('shared library %s declares no reserved word', (_name, glsl) => {
    expect(findReservedDeclarations(glsl)).toEqual([]);
  });

  it('detects the historical `float patch` regression', () => {
    // Proves the check above can actually fail. A guard test that cannot fail
    // is worse than no test, because it produces false confidence — and this
    // is the exact line that shipped two black materials.
    const offending = 'void nfSurface( vec2 uv ) { float patch = 0.5; }';
    expect(findReservedDeclarations(offending)).not.toEqual([]);
  });

  it('does not flag reserved words appearing only in comments', () => {
    const benign = '// a patch of rust, sampled from a buffer\nfloat rust = 0.5;';
    expect(findReservedDeclarations(benign)).toEqual([]);
  });
});

describe('GLSL recipe structure', () => {
  it.each(ALL_RECIPES.map((r) => [r.id, r.glsl] as const))(
    'recipe "%s" defines the surface entry point',
    (_id, glsl) => {
      // Every recipe is compiled into a wrapper that calls this function. If a
      // recipe does not define it, the bake fails at runtime rather than here.
      expect(glsl).toContain('Surface(');
    },
  );

  it.each(ALL_RECIPES.map((r) => [r.id, r.glsl] as const))(
    'recipe "%s" has balanced braces',
    (_id, glsl) => {
      const body = stripComments(glsl);
      const open = (body.match(/\{/g) ?? []).length;
      const close = (body.match(/\}/g) ?? []).length;
      expect(open).toBe(close);
    },
  );

  it('recipe ids are unique', () => {
    const ids = ALL_RECIPES.map((r) => r.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
