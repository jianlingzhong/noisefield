/**
 * Ambient type declarations for the `n8ao` package.
 *
 * n8ao 2.0.0 ships a Parcel bundle (`dist/N8AO.js`) with no `types` entry and
 * no bundled `.d.ts`, so importing it under `strict` fails with TS7016. These
 * declarations were transcribed directly from `node_modules/n8ao/dist/N8AO.js`
 * (the `N8AOPostPass` class and its `configuration` proxy) rather than from
 * memory, and cover exactly the surface this subsystem uses.
 */
declare module 'n8ao' {
  import type { Camera, Color, Scene, Texture, WebGLRenderer, WebGLRenderTarget } from 'three';
  import { Pass } from 'postprocessing';

  /** Depth encodings n8ao can consume. Auto-detected from the renderer. */
  export const DepthType: {
    readonly Default: number;
    readonly Log: number;
    readonly Reverse: number;
  };

  /**
   * Live-tunable AO settings. Assignment goes through a Proxy that reconfigures
   * the internal passes, so these must be set as properties (never replaced
   * wholesale by assigning a new object to `configuration`).
   */
  export interface N8AOConfiguration {
    /** Samples per pixel for the AO estimate. 8 / 16 / 32 are the useful steps. */
    aoSamples: number;
    /** World-space occlusion radius in metres. */
    aoRadius: number;
    aoTones: number;
    denoiseSamples: number;
    denoiseRadius: number;
    /** Multiplier on aoRadius past which occluders stop contributing. */
    distanceFalloff: number;
    intensity: number;
    denoiseIterations: number;
    /** 0 = combined, 1 = AO only, 2 = no AO, 3 = split, 4 = split AO. */
    renderMode: 0 | 1 | 2 | 3 | 4;
    biasOffset: number;
    biasMultiplier: number;
    /** Occlusion tint. Non-black gives colour-shifted contact shadows. */
    color: Color;
    gammaCorrection: boolean;
    depthBufferType: number;
    /** Scale the radius in screen space instead of world space. */
    screenSpaceRadius: boolean;
    /** Compute AO at half resolution and depth-aware upsample. */
    halfRes: boolean;
    depthAwareUpsampling: boolean;
    colorMultiply: boolean;
    transparencyAware: boolean;
    accumulate: boolean;
    neuralDenoise: boolean;
  }

  /**
   * Ground-truth-style ambient occlusion as a `postprocessing` Pass. Consumes
   * the composer's shared depth texture; `needsDepthTexture` and `needsSwap`
   * are both true.
   */
  export class N8AOPostPass extends Pass {
    constructor(scene: Scene, camera: Camera, width?: number, height?: number);

    configuration: N8AOConfiguration;
    scene: Scene;
    camera: Camera;
    width: number;
    height: number;
    /** When true, `gammaCorrection` tracks `renderToScreen` automatically. */
    autosetGamma: boolean;
    debugMode: boolean;

    setSize(width: number, height: number): void;
    setDepthTexture(depthTexture: Texture | null): void;
    /** Invalidates any temporal accumulation. */
    firstFrame(): void;
    enableDebugMode(): void;
    disableDebugMode(): void;

    render(
      renderer: WebGLRenderer,
      inputBuffer: WebGLRenderTarget | null,
      outputBuffer: WebGLRenderTarget | null,
      deltaTime?: number,
      stencilTest?: boolean,
    ): void;
  }

  /** Standalone (non-postprocessing) variant. Unused by this project. */
  export class N8AOPass extends Pass {
    constructor(scene: Scene, camera: Camera, width?: number, height?: number);
    configuration: N8AOConfiguration;
  }
}
