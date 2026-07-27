/**
 * NOISEFIELD — TextureForge.
 *
 * A GPU texture baker: renders a fragment shader over a fullscreen quad into a
 * `WebGLRenderTarget` and hands back the target's `THREE.Texture`. No CPU
 * readback ever happens — the pixels are produced on the GPU and stay there.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * COLOUR SPACE — the part everybody gets wrong
 * ─────────────────────────────────────────────────────────────────────────────
 * three.js r185 renders into a render target using the *working* colour space
 * (linear-sRGB); it does NOT apply `renderer.outputColorSpace`, and a
 * `ShaderMaterial` never includes `<colorspace_fragment>`, so nothing in the
 * software path touches our fragment output.
 *
 * However, `WebGLTextures.getInternalFormat` allocates `SRGB8_ALPHA8` (instead
 * of `RGBA8`) whenever `texture.colorSpace` has an sRGB transfer function. On a
 * WebGL2 framebuffer with an sRGB-encoded colour attachment, the *hardware*
 * performs linear → sRGB on write and sRGB → linear on sample.
 *
 * So the rule is:
 *   • albedo  → `colorSpace: SRGBColorSpace`, shader writes LINEAR values.
 *               The hardware round-trips them, and we get sRGB's extra
 *               precision in the darks for free.
 *   • normal / ORM / height → `colorSpace: NoColorSpace` (RGBA8), values are
 *               raw data and must never be transfer-encoded.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * HEIGHT PRECISION
 * ─────────────────────────────────────────────────────────────────────────────
 * Height fields are baked into the RG channels of an RGBA8 target as a
 * fixed-point 16-bit value (`apxPackUnit` / `apxUnpackUnit`). Deriving a normal
 * map from an 8-bit height field produces very visible stair-stepping on
 * low-slope areas; 16 bits removes it. We deliberately do not use a
 * half-float target because R16F/RGBA16F colour-renderability depends on
 * `EXT_color_buffer_float`, whereas RGBA8 is guaranteed by the ES3 spec.
 */

import * as THREE from 'three';
import { GLSL_PALETTE, NOISE_LIB } from './noise';

/** Uniform bag accepted by `bake`. */
export type ForgeUniforms = Record<string, THREE.IUniform>;

/** Which channel set a surface shader should emit on this pass. */
export const PASS_HEIGHT = 0;
export const PASS_ALBEDO = 1;
export const PASS_ORM = 2;

export interface BakeOptions {
  /** sRGB for albedo, NoColorSpace for every data map. Default: NoColorSpace. */
  colorSpace: THREE.ColorSpace;
  /** Max anisotropic taps. Clamped to the device limit. */
  anisotropy: number;
  /** Default true; disable only for one-shot intermediates. */
  generateMipmaps: boolean;
  /** RGBA (default) or Red for single-channel data. */
  format: THREE.PixelFormat;
  /** Debug name, shows up in Spector/renderdoc captures. */
  name: string;
  /** `#define` lines injected above the shader body. */
  defines: Record<string, string | number>;
}

interface Allocation {
  target: THREE.WebGLRenderTarget;
  bytes: number;
}

/**
 * Every RESERVED word of GLSL ES 3.00 (spec §3.6, "keywords reserved for
 * future use") that is NOT also a keyword in GLSL ES 1.00.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY THIS LIST EXISTS AND WHY IT IS CHECKED ON EVERY BAKE
 * ─────────────────────────────────────────────────────────────────────────────
 * The surface shaders are written in ESSL1 style, but three.js recompiles every
 * non-raw `ShaderMaterial` as `#version 300 es` with compatibility defines. So
 * the dialect that actually reaches the driver is ESSL3, and ESSL3 reserves a
 * batch of ordinary-looking English words that ESSL1 happily accepts as
 * identifiers — `patch`, `sample`, `filter`, `active`, `common`, `resource`.
 *
 * A collision is a HARD COMPILE ERROR, and the resulting failure is invisible
 * from here: three logs to the console, the draw is skipped, and the render
 * target simply keeps its clear colour. The bake "succeeds", the material is
 * registered, and the surface ships as pure black — black albedo, and an ORM of
 * (0,0,0), i.e. occlusion 0 / roughness 0 / metalness 0, which is the single
 * most destructive value that texture can hold. That is exactly how `concrete`
 * and `plaster` — the two highest-screen-area surfaces in the level — shipped
 * black: both declared `float patch`.
 *
 * A reserved word can never legally appear in the source, so a whole-word match
 * outside comments is unambiguous. The scan is a regex over ~30 KB, ~50 times,
 * once at init: microseconds, against a class of bug that costs a day.
 */
const RESERVED_ESSL3: readonly string[] = [
  'active', 'asm', 'cast', 'class', 'coherent', 'common', 'enum', 'extern',
  'external', 'filter', 'fixed', 'goto', 'half', 'hvec2', 'hvec3', 'hvec4',
  'fvec2', 'fvec3', 'fvec4', 'inline', 'input', 'interface', 'long',
  'namespace', 'noinline', 'noperspective', 'output', 'partition', 'patch',
  'public', 'readonly', 'resource', 'restrict', 'row_major', 'sample',
  'short', 'sizeof', 'static', 'subroutine', 'superp', 'template', 'this',
  'typedef', 'union', 'unsigned', 'using', 'volatile', 'writeonly',
];

const RESERVED_RE = new RegExp(`\\b(?:${RESERVED_ESSL3.join('|')})\\b`, 'g');

/** Strip `//` and block comments so the reserved-word scan cannot false-positive on prose. */
function stripGlslComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
}

/**
 * Whole-word ESSL3 reserved identifiers used in `source`, deduplicated.
 * Preprocessor lines are skipped: `#include <common>` is three's own chunk
 * name, not an identifier, and never reaches the compiler as one.
 */
function reservedIdentifiersIn(source: string): string[] {
  const body = stripGlslComments(source)
    .split('\n')
    .filter((line) => !line.trimStart().startsWith('#'))
    .join('\n');
  const hits = new Set<string>();
  RESERVED_RE.lastIndex = 0;
  for (let m = RESERVED_RE.exec(body); m !== null; m = RESERVED_RE.exec(body)) hits.add(m[0]);
  return [...hits];
}

/** GLSL helpers that only the forge needs (packing + the surface struct). */
const FORGE_PRELUDE = /* glsl */ `
precision highp float;
precision highp int;

varying vec2 vUv;

uniform vec2  uTexel;   // 1 / resolution
uniform float uSize;    // resolution in texels
uniform float uSeed;    // per-surface decorrelation constant

/**
 * Fixed-point 16-bit packing across two 8-bit channels.
 * hi holds the top 8 bits, lo the remainder scaled back into [0,1).
 */
vec2 apxPackUnit( float v ) {
  v = clamp( v, 0.0, 0.9999 );
  float s = v * 255.0;
  return vec2( floor( s ) / 255.0, fract( s ) );
}
float apxUnpackUnit( vec2 p ) { return p.x + p.y * ( 1.0 / 255.0 ); }
`;

/**
 * The contract every surface shader implements. Fields are pre-seeded with
 * sane defaults by the generated `main()` so a recipe only has to write the
 * channels it actually cares about.
 */
const SURFACE_STRUCT = /* glsl */ `
struct NfSurface {
  vec3  albedo;   // LINEAR light values, not sRGB
  float rough;    // perceptual roughness, 0 = mirror
  float metal;    // 0 dielectric, 1 conductor
  float ao;       // baked cavity occlusion, 1 = unoccluded
  float height;   // 0..1, drives the derived normal map
  float alpha;    // albedo alpha, used for foliage cut-outs
};
`;

const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  // PlaneGeometry(2,2) already spans clip space, so no matrices are needed.
  gl_Position = vec4( position.xy, 0.0, 1.0 );
}
`;

/** Sobel-based height → tangent-space normal derivation. */
const NORMAL_FROM_HEIGHT_FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D uHeight;
uniform vec2 uTexel;
uniform float uSlope;

float apxUnpackUnit( vec2 p ) { return p.x + p.y * ( 1.0 / 255.0 ); }

float H( vec2 o ) {
  // The height texture is RepeatWrapping, so taps that fall outside [0,1] wrap
  // to the opposite edge. That is what makes the DERIVED normal map seamless
  // too — a naive clamped fetch would leave a one-texel ridge at every seam.
  return apxUnpackUnit( texture2D( uHeight, vUv + o * uTexel ).rg );
}

void main() {
  // 3x3 Sobel rather than a 2-tap central difference: the implicit 1-2-1
  // low-pass across the perpendicular axis suppresses the per-texel noise that
  // granular height fields are full of, which would otherwise alias badly once
  // the map is minified.
  float tl = H( vec2( -1.0,  1.0 ) ), tc = H( vec2( 0.0,  1.0 ) ), tr = H( vec2( 1.0,  1.0 ) );
  float ml = H( vec2( -1.0,  0.0 ) ),                              mr = H( vec2( 1.0,  0.0 ) );
  float bl = H( vec2( -1.0, -1.0 ) ), bc = H( vec2( 0.0, -1.0 ) ), br = H( vec2( 1.0, -1.0 ) );

  // Sobel sums to 8x the per-texel derivative for a linear ramp, so /8 gives
  // dh/dTexel; uSlope folds in (resolution * user strength) to convert that to
  // dh/dUV, making the result independent of the bake resolution.
  float dX = ( ( tr + 2.0 * mr + br ) - ( tl + 2.0 * ml + bl ) ) * 0.125;
  float dY = ( ( tl + 2.0 * tc + tr ) - ( bl + 2.0 * bc + br ) ) * 0.125;

  // Surface P(u,v) = (u, v, h). Tu = (1,0,dh/du), Tv = (0,1,dh/dv),
  // N = Tu x Tv = (-dh/du, -dh/dv, 1)  -> green points along +V, i.e. the
  // OpenGL convention that MeshStandardMaterial's tangent frame expects.
  vec3 n = normalize( vec3( -dX * uSlope, -dY * uSlope, 1.0 ) );

  // ALPHA CARRIES THE HEIGHT, at 8-bit.
  //
  // The channel is allocated whether we use it or not (RGBA8 is the only
  // guaranteed colour-renderable byte format), so this is genuinely free — and
  // it turns the shared detail normal into a shared detail *cavity* map as
  // well. Multiplying albedo by it gives every surface high-frequency colour
  // grain at the detail rate, which no base map can hold: sand at a 4 m tile
  // has 3.9 mm texels, so grain and grit are physically below its Nyquist
  // limit, and without this the albedo stays a smooth wash however good the
  // normals are. 8 bits is plenty here; only the Sobel needed 16, and it reads
  // the separate RG-packed source above.
  gl_FragColor = vec4( n * 0.5 + 0.5, H( vec2( 0.0 ) ) );
}
`;

export class TextureForge {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly quad: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  private readonly geometry = new THREE.PlaneGeometry(2, 2);
  private readonly allocations: Allocation[] = [];
  private readonly maxAnisotropy: number;

  /** Default anisotropy applied when a bake does not override it. */
  anisotropy: number;

  /** Running total of GPU bytes handed out by this forge, mipmaps included. */
  private bytes = 0;

  /**
   * Bakes whose shader did not compile or link, as `name: reason`.
   *
   * A failed bake still returns a texture — it is simply the clear colour — so
   * without this list the only symptom is a black surface a hundred draw calls
   * later. `MaterialLibrary.init` reports it as an error, and a test or debug
   * overlay can assert that it is empty.
   */
  readonly shaderFailures: string[] = [];

  constructor(renderer: THREE.WebGLRenderer, anisotropy = 8) {
    this.renderer = renderer;
    this.maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
    this.anisotropy = Math.min(anisotropy, this.maxAnisotropy);

    // A placeholder material; every bake swaps in its own and disposes it after.
    const placeholder = new THREE.ShaderMaterial({ vertexShader: FULLSCREEN_VERT, fragmentShader: 'void main(){ gl_FragColor = vec4(0.0); }' });
    this.quad = new THREE.Mesh(this.geometry, placeholder);
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
  }

  /** Total VRAM allocated by this forge, in bytes (includes the mip chain). */
  get bytesAllocated(): number {
    return this.bytes;
  }

  get textureCount(): number {
    return this.allocations.length;
  }

  // ── core ───────────────────────────────────────────────────────────────────

  /**
   * Render `fragmentShader` into a `size x size` texture and return it.
   *
   * The shader may use `vUv` (0..1), `uTexel`, `uSize`, `uSeed` and anything
   * passed in `uniforms`. The full noise library and palette are prepended.
   */
  bake(fragmentShader: string, size: number, uniforms: ForgeUniforms = {}, options: Partial<BakeOptions> = {}): THREE.Texture {
    const colorSpace = options.colorSpace ?? THREE.NoColorSpace;
    const generateMipmaps = options.generateMipmaps ?? true;
    const format = options.format ?? THREE.RGBAFormat;
    const aniso = Math.min(options.anisotropy ?? this.anisotropy, this.maxAnisotropy);

    const target = new THREE.WebGLRenderTarget(size, size, {
      format,
      type: THREE.UnsignedByteType,
      colorSpace,
      // Trilinear + anisotropic. Without mipmaps a tiled surface viewed at a
      // grazing angle turns into shimmering noise, which is the single most
      // obvious "this is a hobby project" tell.
      minFilter: generateMipmaps ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      anisotropy: aniso,
      generateMipmaps,
      depthBuffer: false,
      stencilBuffer: false,
    });
    const name = options.name ?? 'noisefield.bake';
    target.texture.name = name;

    const defineBlock = Object.entries(options.defines ?? {})
      .map(([k, v]) => `#define ${k} ${v}`)
      .join('\n');

    const source = `${defineBlock}\n${FORGE_PRELUDE}\n${NOISE_LIB}\n${GLSL_PALETTE}\n${SURFACE_STRUCT}\n${fragmentShader}`;

    // Authoring-time check, before a single GPU cycle is spent. See RESERVED_ESSL3.
    const reserved = reservedIdentifiersIn(source);
    if (reserved.length > 0) {
      const why = `${name}: GLSL ES 3.00 reserved word(s) used as identifiers: ${reserved.join(', ')}`;
      this.shaderFailures.push(why);
      console.error(
        `[forge] ${why}. three.js compiles this as "#version 300 es", so the shader will NOT compile and ` +
        'the texture will bake to the clear colour (black albedo, ORM = 0/0/0). Rename the identifier.',
      );
    }

    const material = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: source,
      uniforms: {
        uTexel: { value: new THREE.Vector2(1 / size, 1 / size) },
        uSize: { value: size },
        uSeed: { value: 0 },
        ...uniforms,
      },
      depthTest: false,
      depthWrite: false,
      // NoBlending is mandatory: with the default blending the alpha channel we
      // write for foliage cut-outs would be composited against the clear colour
      // instead of stored verbatim.
      blending: THREE.NoBlending,
    });

    this.renderInto(target, material, name);
    material.dispose();

    this.track(target, size, format, generateMipmaps);
    return target.texture;
  }

  /**
   * Convenience wrapper for a surface recipe: wraps the recipe's
   * `nfSurface()` body in a `main()` that emits the requested channel set.
   *
   * Running three separate passes over the same body (rather than one MRT pass)
   * costs a few extra milliseconds *once*, at init, and buys a shader that
   * compiles on every driver without needing `gl_FragData`/GLSL3 plumbing. The
   * unused outputs are dead-code eliminated because `pass` is a `#define`.
   */
  bakeSurfacePass(
    surfaceGlsl: string,
    pass: number,
    size: number,
    seed: number,
    uniforms: ForgeUniforms = {},
    options: Partial<BakeOptions> = {},
  ): THREE.Texture {
    const body = /* glsl */ `
${surfaceGlsl}

void main() {
  NfSurface s;
  s.albedo = vec3( 0.5 );
  s.rough  = 0.85;
  s.metal  = 0.0;
  s.ao     = 1.0;
  s.height = 0.5;
  s.alpha  = 1.0;

  nfSurface( vUv, s );

#if NF_PASS == 0
  // 16-bit fixed point across RG so the Sobel pass has enough precision.
  gl_FragColor = vec4( apxPackUnit( s.height ), 0.0, 1.0 );
#elif NF_PASS == 1
  // LINEAR albedo; the SRGB8_ALPHA8 attachment encodes it in hardware.
  gl_FragColor = vec4( max( s.albedo, vec3( 0.0 ) ), clamp( s.alpha, 0.0, 1.0 ) );
#else
  // glTF ORM packing: R = occlusion, G = roughness, B = metalness.
  // three.js reads exactly these channels for aoMap/roughnessMap/metalnessMap,
  // so one texture drives all three and we save 2/3 of the bandwidth.
  gl_FragColor = vec4( clamp( s.ao, 0.0, 1.0 ), clamp( s.rough, 0.0, 1.0 ), clamp( s.metal, 0.0, 1.0 ), 1.0 );
#endif
}
`;
    return this.bake(body, size, { uSeed: { value: seed }, ...uniforms }, {
      ...options,
      defines: { ...(options.defines ?? {}), NF_PASS: pass },
    });
  }

  /**
   * Derive a tangent-space normal map from a packed height field.
   *
   * `strength` is resolution-independent: 1.0 means "a height delta of 1.0
   * across one UV unit produces a 45 degree slope".
   */
  bakeNormalFromHeight(heightTex: THREE.Texture, strength: number, size: number, name = 'noisefield.normal'): THREE.Texture {
    const target = new THREE.WebGLRenderTarget(size, size, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      colorSpace: THREE.NoColorSpace,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      anisotropy: this.anisotropy,
      generateMipmaps: true,
      depthBuffer: false,
      stencilBuffer: false,
    });
    target.texture.name = name;

    const material = new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VERT,
      fragmentShader: NORMAL_FROM_HEIGHT_FRAG,
      uniforms: {
        uHeight: { value: heightTex },
        uTexel: { value: new THREE.Vector2(1 / size, 1 / size) },
        uSlope: { value: strength * size },
      },
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });

    this.renderInto(target, material, name);
    material.dispose();

    this.track(target, size, THREE.RGBAFormat, true);
    return target.texture;
  }

  /**
   * Bake a height field, derive its normal map, then throw the height away.
   *
   * Keeping every height map alive would cost a third of our entire texture
   * budget for data that nothing samples (we do not tessellate). Callers that
   * genuinely want displacement ask for it explicitly.
   */
  bakeNormalFromSurface(
    surfaceGlsl: string,
    size: number,
    seed: number,
    strength: number,
    name: string,
    uniforms: ForgeUniforms = {},
    keepHeight = false,
  ): { normal: THREE.Texture; height: THREE.Texture | null } {
    const height = this.bakeSurfacePass(surfaceGlsl, PASS_HEIGHT, size, seed, uniforms, {
      generateMipmaps: keepHeight,
      name: `${name}.height`,
    });
    const normal = this.bakeNormalFromHeight(height, strength, size, `${name}.normal`);
    if (!keepHeight) {
      this.release(height);
      return { normal, height: null };
    }
    return { normal, height };
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private renderInto(target: THREE.WebGLRenderTarget, material: THREE.ShaderMaterial, name: string): void {
    const prevTarget = this.renderer.getRenderTarget();
    // getRenderTarget() alone loses the bound cube face and mip level, so a
    // bake invoked mid-frame (a hot-reload re-bake, say) would silently rebind
    // face 0 / level 0 of whatever was previously active.
    const prevCubeFace = this.renderer.getActiveCubeFace();
    const prevMipLevel = this.renderer.getActiveMipmapLevel();
    const prevAutoClear = this.renderer.autoClear;

    // Catch a compile/link failure AT THE BAKE, attributed to this texture.
    //
    // Without this hook three prints a wall of numbered GLSL to the console and
    // the bake reports success, because a broken program just means the draw is
    // dropped and the target keeps its clear colour. Every downstream symptom
    // (black albedo, ORM = 0/0/0, dead-flat normal) then points at the material
    // system rather than at the one shader that actually failed. See
    // RESERVED_ESSL3 for the instance of this that shipped.
    const prevOnShaderError = this.renderer.debug.onShaderError;
    this.renderer.debug.onShaderError = (gl, program, vs, fs): void => {
      const vLog = gl.getShaderInfoLog(vs) ?? '';
      const fLog = gl.getShaderInfoLog(fs) ?? '';
      const pLog = gl.getProgramInfoLog(program) ?? '';
      const reason = `${name}: ${(fLog || vLog || pLog).trim().split('\n')[0]}`;
      this.shaderFailures.push(reason);
      console.error(
        `[forge] shader error while baking "${name}" — this texture will contain the clear colour ` +
        `(black albedo / ORM 0,0,0), NOT the recipe.\nfragment: ${fLog}\nvertex: ${vLog}\nprogram: ${pLog}`,
      );
    };

    this.quad.material = material;
    this.renderer.autoClear = true;
    this.renderer.setRenderTarget(target);
    // `render()` itself calls `textures.updateRenderTargetMipmap()` on the bound
    // target before it returns, so the mip chain already exists by the time we
    // unbind. That is why `generateMipmaps` and `minFilter` have to be set at
    // construction time rather than afterwards.
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(prevTarget, prevCubeFace, prevMipLevel);
    this.renderer.autoClear = prevAutoClear;
    this.renderer.debug.onShaderError = prevOnShaderError;
  }

  private track(target: THREE.WebGLRenderTarget, size: number, format: THREE.PixelFormat, mips: boolean): void {
    const channels = format === THREE.RedFormat ? 1 : format === THREE.RGFormat ? 2 : 4;
    // A full mip chain adds sum(1/4^n) = 1/3 on top of the base level.
    const bytes = Math.round(size * size * channels * (mips ? 4 / 3 : 1));
    this.allocations.push({ target, bytes });
    this.bytes += bytes;
  }

  /** Free a texture produced by this forge (used for scratch height fields). */
  release(texture: THREE.Texture): void {
    const idx = this.allocations.findIndex((a) => a.target.texture === texture);
    if (idx < 0) return;
    const alloc = this.allocations[idx];
    this.bytes -= alloc.bytes;
    alloc.target.dispose();
    this.allocations.splice(idx, 1);
  }

  dispose(): void {
    for (const alloc of this.allocations) alloc.target.dispose();
    this.allocations.length = 0;
    this.bytes = 0;
    this.geometry.dispose();
    this.quad.material.dispose();
  }
}
