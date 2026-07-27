/**
 * NOISEFIELD — RenderPipeline.
 *
 * The whole frame, in order:
 *
 *   1. RenderPass          world scene → HDR half-float buffer (+ depth)
 *   2. N8AOPostPass        ground-truth ambient occlusion
 *   3. MotionBlurPass      depth-reprojected camera motion blur
 *   4. ViewmodelPass       first-person weapon, composited with a cleared depth
 *   5. EffectPass          bloom → exposure → ACES tone mapping
 *   6. EffectPass          SMAA (or FXAA)
 *   7. EffectPass          CA / sharpen / grade / vignette / grain → screen
 *
 * WHY THIS ORDER
 * --------------
 * AO before the viewmodel: N8AO reconstructs geometry from the *world* depth
 * buffer, and the viewmodel is rendered with a completely different camera
 * (near 0.008, far 12). Feeding its depth into the AO solver would produce a
 * black halo around the weapon.
 *
 * Motion blur before the viewmodel, for the same reason inverted: the depth
 * buffer's velocity field belongs to the world, and applying it to the weapon —
 * which is nearly static in screen space — would smear it every time the player
 * turns. Running the blur first leaves the weapon perfectly crisp with no
 * masking machinery.
 *
 * Viewmodel *inside* the composer rather than after it: a naive
 * `composer.render(); renderer.render(viewScene)` skips tone mapping, bloom and
 * the grade for the weapon, so it floats over the frame in a different colour
 * space — washed out, unbloomed, ungraded. Compositing it as a pass means the
 * muzzle flash blooms and the weapon is tone mapped and graded exactly like the
 * world it is held in. The pass clears only depth, so the weapon can never clip
 * through a wall while still self-occluding correctly.
 *
 * SMAA after tone mapping (edge detection is meaningless on unbounded HDR) and
 * before grain (grain would swamp the edge detector). Chromatic aberration
 * after SMAA so the fringe is not smeared away.
 *
 * FAILURE POLICY
 * --------------
 * Every optional stage is constructed inside its own try/catch. A GPU that
 * cannot compile N8AO, or a browser that chokes on the SMAA lookup textures,
 * degrades the chain rather than producing a black screen; and if the composer
 * itself cannot be created we fall all the way back to two direct
 * `renderer.render` calls.
 */

import * as THREE from 'three';
import {
  BlendFunction,
  BloomEffect,
  EdgeDetectionMode,
  Effect,
  EffectAttribute,
  EffectComposer,
  EffectPass,
  FXAAEffect,
  Pass,
  RenderPass,
  SMAAEffect,
  SMAAPreset,
  ToneMappingEffect,
  ToneMappingMode,
} from 'postprocessing';
import { N8AOPostPass } from 'n8ao';
import type { IRenderPipeline } from '../core/Engine';
import type { EngineContext, FrameContext, QualitySettings } from '../core/Types';
import {
  FILM_GRADE_FRAGMENT,
  MOTION_BLUR_FRAGMENT,
  MOTION_BLUR_SAMPLES_HIGH,
  MOTION_BLUR_SAMPLES_LOW,
} from './shaders/PostShaders';

// ─────────────────────────────────────────────────────────────────────────────
// Custom passes / effects
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Draws the first-person viewmodel scene on top of the current buffer with a
 * freshly cleared depth attachment.
 *
 * `needsSwap = false`: we render *into* the pass input rather than producing a
 * new image, so the composer must keep using the same buffer afterwards.
 * `needsDepthBlit` is deliberately left false — the composer's shared depth
 * texture must keep holding *world* depth for any later depth-aware effect.
 */
class ViewmodelPass extends Pass {
  private readonly viewScene: THREE.Scene;
  private readonly viewCamera: THREE.Camera;

  constructor(viewScene: THREE.Scene, viewCamera: THREE.Camera) {
    super('NoisefieldViewmodelPass');
    this.viewScene = viewScene;
    this.viewCamera = viewCamera;
    this.needsSwap = false;
  }

  override render(
    renderer: THREE.WebGLRenderer,
    inputBuffer: THREE.WebGLRenderTarget | null,
    _outputBuffer: THREE.WebGLRenderTarget | null,
    _deltaTime?: number,
    _stencilTest?: boolean,
  ): void {
    if (this.viewScene.children.length === 0) return;

    renderer.setRenderTarget(this.renderToScreen ? null : inputBuffer);
    // Depth only. The world colour underneath must survive untouched; the
    // composer has already disabled autoClear so nothing else will wipe it.
    renderer.clearDepth();
    renderer.render(this.viewScene, this.viewCamera);
  }
}

/** Depth-reprojected camera motion blur. See `shaders/PostShaders.ts`. */
class MotionBlurEffect extends Effect {
  private readonly uInvViewProj: THREE.Uniform<THREE.Matrix4>;
  private readonly uPrevViewProj: THREE.Uniform<THREE.Matrix4>;
  private readonly uShutter: THREE.Uniform<number>;
  private readonly uMaxBlur: THREE.Uniform<number>;

  constructor(samples: number, shutter: number, maxBlur: number) {
    const invViewProj = new THREE.Uniform<THREE.Matrix4>(new THREE.Matrix4());
    const prevViewProj = new THREE.Uniform<THREE.Matrix4>(new THREE.Matrix4());
    const shutterU = new THREE.Uniform<number>(shutter);
    const maxBlurU = new THREE.Uniform<number>(maxBlur);

    super('NoisefieldMotionBlur', MOTION_BLUR_FRAGMENT, {
      // DEPTH gets us the composer's stable depth texture; CONVOLUTION declares
      // that we sample the input buffer at offsets (only one such effect is
      // allowed per EffectPass, which is why this one lives alone).
      attributes: EffectAttribute.DEPTH | EffectAttribute.CONVOLUTION,
      blendFunction: BlendFunction.SRC,
      defines: new Map<string, string>([['MB_SAMPLES', samples.toFixed(0)]]),
      uniforms: new Map<string, THREE.Uniform>([
        ['invViewProj', invViewProj],
        ['prevViewProj', prevViewProj],
        ['shutter', shutterU],
        ['maxBlur', maxBlurU],
      ]),
    });

    this.uInvViewProj = invViewProj;
    this.uPrevViewProj = prevViewProj;
    this.uShutter = shutterU;
    this.uMaxBlur = maxBlurU;
  }

  setMatrices(invViewProj: THREE.Matrix4, prevViewProj: THREE.Matrix4): void {
    this.uInvViewProj.value.copy(invViewProj);
    this.uPrevViewProj.value.copy(prevViewProj);
  }

  /** Fraction of a frame's motion the virtual shutter integrates. */
  set shutter(v: number) {
    this.uShutter.value = v;
  }
  get shutter(): number {
    return this.uShutter.value;
  }

  set maxBlur(v: number) {
    this.uMaxBlur.value = v;
  }
  get maxBlur(): number {
    return this.uMaxBlur.value;
  }
}

/** Trivial pre-tone-map exposure multiply. Merges into the tone mapping pass. */
class ExposureEffect extends Effect {
  private readonly uExposure: THREE.Uniform<number>;

  constructor(exposure: number) {
    const u = new THREE.Uniform<number>(exposure);
    super(
      'NoisefieldExposure',
      /* glsl */ `
uniform float exposure;
void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
  outputColor = vec4(inputColor.rgb * exposure, inputColor.a);
}
`,
      {
        blendFunction: BlendFunction.SRC,
        uniforms: new Map<string, THREE.Uniform>([['exposure', u]]),
      },
    );
    this.uExposure = u;
  }

  set exposure(v: number) {
    this.uExposure.value = v;
  }
  get exposure(): number {
    return this.uExposure.value;
  }
}

/** Final cinematic finishing. See `shaders/PostShaders.ts` for the shader. */
class FilmGradeEffect extends Effect {
  private readonly uGrainTime: THREE.Uniform<number>;
  private readonly uGrainAmount: THREE.Uniform<number>;
  private readonly uCaStrength: THREE.Uniform<number>;

  constructor() {
    const grainTime = new THREE.Uniform<number>(0);
    // DISPLAY-referred peak-to-peak grain amplitude, not linear. See the film
    // grain block in PostShaders.ts: the value is divided by the sRGB transfer
    // slope at each pixel, so it means "this many units of 0..1 output", the
    // same at every brightness. 0.026 reproduces the previous grain exactly in
    // the midtones (0.022 linear at linear 0.18 is 0.026 of display range) and
    // differs only where the old formulation was broken — in the shadows,
    // where a fixed linear offset was up to twenty times too strong.
    const grainAmount = new THREE.Uniform<number>(0.026);
    // 0.0005 — about half a pixel of fringe at 1080p, only in the outer third
    // of the frame. The sand's specular detail aliases into isolated bright
    // pixels at grazing angles, and any larger offset splits each of those into
    // a visible red/blue pair, turning honest aliasing into rainbow confetti.
    const caStrength = new THREE.Uniform<number>(0.0005);

    super('NoisefieldFilmGrade', FILM_GRADE_FRAGMENT, {
      attributes: EffectAttribute.CONVOLUTION,
      blendFunction: BlendFunction.SRC,
      uniforms: new Map<string, THREE.Uniform>([
        ['caStrength', caStrength],
        ['caInner', new THREE.Uniform<number>(0.42)],
        // 0.11, down from 0.16. With the contrast curve now doing real work, a
        // 0.16 unsharp overshoots on the high-contrast edge between a sunlit
        // wall and the sky and rings visibly — a bright halo plus a dark one,
        // which on a coloured edge separates into red/cyan and reads as far
        // worse aberration than the CA it sits next to.
        ['sharpness', new THREE.Uniform<number>(0.11)],
        // Small residual channel balance only; the S-curve and the split tone
        // below carry the look. Black is pulled down a hair (a true 0 floor
        // somewhere in frame is most of what reads as "contrast") with a touch
        // of blue left in it so the crush stays cool rather than muddy.
        // A small POSITIVE lift, cool-tinted. Printed film never reaches a true
        // zero; the densest black on a release print still passes a little
        // light, and that milky, slightly blue floor is a large part of why an
        // image reads as photographed rather than rendered. It also buys back
        // shadow separation the S-curve's toe would otherwise flatten.
        // Weighted toward blue rather than blue-green. Lift is an ADDITIVE
        // offset, so it is the only part of the grade that still has authority
        // at near-zero output — and on a large deeply shadowed wall it becomes
        // the wall's entire hue. At (0.004, 0.009, 0.019) the green sat at 2.2x
        // the red and the shaded side of the interior shot came out frankly
        // teal. Keeping green close to red makes the same lift read as sky.
        ['gradeLift', new THREE.Uniform<THREE.Vector3>(new THREE.Vector3(0.006, 0.007, 0.016))],
        ['gradeGamma', new THREE.Uniform<THREE.Vector3>(new THREE.Vector3(1.0, 1.005, 1.04))],
        ['gradeGain', new THREE.Uniform<THREE.Vector3>(new THREE.Vector3(1.03, 1.0, 0.96))],
        // 1.14: enough to open a real gap between lit sand and shadowed sand
        // without posterising the dust texture. 1.22 was tried and, stacked on
        // the corrected light ratio, drove a quarter of the market shot below
        // 0.10 display luma. Contrast is cheap to add and expensive to undo.
        ['contrast', new THREE.Uniform<number>(1.14)],
        // Pivot just below display middle grey. Most of a golden-hour frame —
        // raked ground, shadowed walls — sits in the lower midtones; pivoting at
        // 0.5 would push all of it down and darken the image instead of
        // steepening it. 0.44 puts the steepest part of the curve where the
        // pixels actually are while leaving the toe shallow enough to keep
        // detail in a shaded alley.
        ['gradePivot', new THREE.Uniform<number>(0.44)],
        // Sky-blue in the toe, low-sun amber in the shoulder. Deliberately
        // gentle per-channel numbers: the eye reads a 6% channel imbalance as a
        // strong colour cast, and anything stronger tips into a teal-and-orange
        // parody of the look rather than the look itself.
        // Blue, not cyan. Pulling red down hard while leaving green near unity
        // makes a neutral shadow swing toward teal, which reads as a colour
        // error rather than as skylight; skylight is blue-violet, so red and
        // green come down together and only blue goes up.
        ['shadowTint', new THREE.Uniform<THREE.Vector3>(new THREE.Vector3(0.94, 0.955, 1.12))],
        // Green sits slightly under unity on purpose. ACES pulls bright warm
        // tones toward yellow as it desaturates them into the shoulder, and on
        // a frame full of sunlit cream stucco that measures as a real green
        // excess (G above the R-B midpoint by 0.030 on the alley shot). Taking
        // 2% off the green highlight cancels it without touching the midtones,
        // where ACES behaves.
        ['highlightTint', new THREE.Uniform<THREE.Vector3>(new THREE.Vector3(1.07, 0.98, 0.9))],
        ['splitStrength', new THREE.Uniform<number>(0.85)],
        // 1.16 with the low-end rolloff in the shader. Sand at this sun angle is
        // genuinely a saturated ochre; the previous 1.04 left it grey-beige.
        ['saturation', new THREE.Uniform<number>(1.16)],
        ['vignetteDarkness', new THREE.Uniform<number>(0.3)],
        ['vignetteOffset', new THREE.Uniform<number>(0.52)],
        ['grainAmount', grainAmount],
        ['grainTime', grainTime],
      ]),
    });

    this.uGrainTime = grainTime;
    this.uGrainAmount = grainAmount;
    this.uCaStrength = caStrength;
  }

  /**
   * Advances the grain animation. Wrapped well below the point where float32
   * loses sub-integer precision, otherwise the grain freezes after a few
   * minutes of play.
   */
  advance(dt: number): void {
    this.uGrainTime.value = (this.uGrainTime.value + dt * 61.7) % 2048;
  }

  set grain(v: number) {
    this.uGrainAmount.value = v;
  }
  get grain(): number {
    return this.uGrainAmount.value;
  }

  set chromaticAberration(v: number) {
    this.uCaStrength.value = v;
  }
  get chromaticAberration(): number {
    return this.uCaStrength.value;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

export class RenderPipeline implements IRenderPipeline {
  readonly name = 'pipeline';

  /** Null only if EffectComposer construction failed; see `renderFallback`. */
  private composer: EffectComposer | null = null;

  private renderPass: RenderPass | null = null;
  private aoPass: N8AOPostPass | null = null;
  private motionPass: EffectPass | null = null;
  private viewmodelPass: ViewmodelPass | null = null;

  private bloomEffect: BloomEffect | null = null;
  private toneMappingEffect: ToneMappingEffect | null = null;
  private exposureEffect: ExposureEffect | null = null;
  private motionBlurEffect: MotionBlurEffect | null = null;
  private filmGradeEffect: FilmGradeEffect | null = null;

  private ctx: EngineContext | null = null;
  private pendingSize: { w: number; h: number } | null = null;

  // Scratch matrices — `render` runs 60+ times a second and must not allocate.
  private readonly viewProj = new THREE.Matrix4();
  private readonly invViewProj = new THREE.Matrix4();
  private readonly prevViewProj = new THREE.Matrix4();
  private hasPreviousFrame = false;

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;
    const settings = ctx.settings;

    try {
      // Half-float is not optional: bloom, exposure and the tone mapper all
      // need real HDR headroom, and an 8-bit intermediate visibly bands the
      // sky gradient long before the grain can dither it away.
      this.composer = new EffectComposer(ctx.renderer, {
        frameBufferType: THREE.HalfFloatType,
        depthBuffer: true,
        stencilBuffer: false,
        multisampling: 0, // SMAA handles edges; MSAA would double the bandwidth
      });
    } catch (err) {
      console.error('[RenderPipeline] EffectComposer unavailable; using direct rendering.', err);
      this.composer = null;
      return;
    }

    const composer = this.composer;

    // 1 ── world
    this.renderPass = new RenderPass(ctx.scene, ctx.camera);
    composer.addPass(this.renderPass);

    // 2 ── ambient occlusion
    if (settings.ssao) this.addAmbientOcclusion(composer, ctx, settings);

    // 3 ── motion blur (before the viewmodel: see the file header)
    if (settings.motionBlur) this.addMotionBlur(composer, ctx, settings);

    // 4 ── viewmodel
    try {
      this.viewmodelPass = new ViewmodelPass(ctx.viewScene, ctx.viewCamera);
      composer.addPass(this.viewmodelPass);
    } catch (err) {
      console.error('[RenderPipeline] Viewmodel pass failed; the weapon may clip into walls.', err);
      this.viewmodelPass = null;
    }

    // 5 ── bloom + exposure + tone mapping (one merged fullscreen pass)
    this.addToneMapping(composer, ctx, settings);

    // 6 ── anti-aliasing
    this.addAntiAliasing(composer, ctx, settings);

    // 7 ── cinematic finishing, always last so it owns the sRGB encode
    try {
      this.filmGradeEffect = new FilmGradeEffect();
      composer.addPass(new EffectPass(ctx.camera, this.filmGradeEffect));
    } catch (err) {
      console.error('[RenderPipeline] Film grade unavailable.', err);
      this.filmGradeEffect = null;
    }

    // Engine::resize may have fired before we existed.
    if (this.pendingSize) {
      composer.setSize(this.pendingSize.w, this.pendingSize.h);
      this.pendingSize = null;
    } else {
      composer.setSize(ctx.width, ctx.height);
    }
  }

  render(ctx: EngineContext, frame: FrameContext): void {
    const composer = this.composer;
    if (composer === null) {
      this.renderFallback(ctx);
      return;
    }

    const mb = this.motionBlurEffect;
    if (mb !== null) {
      // Camera matrices are already settled: systems have run their `update`
      // and the engine calls us afterwards.
      ctx.camera.updateMatrixWorld();
      this.viewProj.multiplyMatrices(ctx.camera.projectionMatrix, ctx.camera.matrixWorldInverse);
      if (!this.hasPreviousFrame) {
        // Frame one has no history; an identity "previous" matrix would smear
        // the entire image across the screen.
        this.prevViewProj.copy(this.viewProj);
        this.hasPreviousFrame = true;
      }
      this.invViewProj.copy(this.viewProj).invert();
      mb.setMatrices(this.invViewProj, this.prevViewProj);
    }

    this.filmGradeEffect?.advance(frame.dt);

    composer.render(frame.dt);

    if (mb !== null) this.prevViewProj.copy(this.viewProj);
  }

  resize(width: number, height: number): void {
    if (this.composer === null) {
      this.pendingSize = { w: width, h: height };
      return;
    }
    this.composer.setSize(width, height);
  }

  dispose(): void {
    // EffectComposer::dispose walks its passes (and their effects) for us.
    this.composer?.dispose();
    this.composer = null;
    this.renderPass = null;
    this.aoPass = null;
    this.motionPass = null;
    this.viewmodelPass = null;
    this.bloomEffect = null;
    this.toneMappingEffect = null;
    this.exposureEffect = null;
    this.motionBlurEffect = null;
    this.filmGradeEffect = null;
  }

  // ── public tuning surface ──────────────────────────────────────────────────

  /** Linear exposure multiplier applied before tone mapping. */
  set exposure(v: number) {
    if (this.exposureEffect) this.exposureEffect.exposure = v;
  }
  get exposure(): number {
    return this.exposureEffect?.exposure ?? 1;
  }

  /** Film grain strength, 0..0.1. Above ~0.05 it reads as video noise. */
  set grain(v: number) {
    if (this.filmGradeEffect) this.filmGradeEffect.grain = v;
  }
  get grain(): number {
    return this.filmGradeEffect?.grain ?? 0;
  }

  /** Peak chromatic aberration offset in UV units at the frame corner. */
  set chromaticAberration(v: number) {
    if (this.filmGradeEffect) this.filmGradeEffect.chromaticAberration = v;
  }
  get chromaticAberration(): number {
    return this.filmGradeEffect?.chromaticAberration ?? 0;
  }

  /** Virtual shutter fraction for motion blur; 0.4 ~ a 144° shutter angle. */
  set motionBlurShutter(v: number) {
    if (this.motionBlurEffect) this.motionBlurEffect.shutter = v;
  }
  get motionBlurShutter(): number {
    return this.motionBlurEffect?.shutter ?? 0;
  }

  get bloom(): BloomEffect | null {
    return this.bloomEffect;
  }

  get ambientOcclusion(): N8AOPostPass | null {
    return this.aoPass;
  }

  // ── stage construction ─────────────────────────────────────────────────────

  private addAmbientOcclusion(
    composer: EffectComposer,
    ctx: EngineContext,
    settings: QualitySettings,
  ): void {
    try {
      const size = ctx.renderer.getDrawingBufferSize(new THREE.Vector2());
      const pass = new N8AOPostPass(ctx.scene, ctx.camera, size.width, size.height);
      const cfg = pass.configuration;

      // Scene scale: the arena is ~90 m across, a player is 1.8 m tall, a crate
      // is 0.6-1.2 m, a doorway 2 m. AO's job here is the contact darkening the
      // sun cannot resolve — where a crate meets the floor, the inside corner
      // of a stairwell, under a parked truck. 1.3 m is a little under a
      // crate-height and reliably grounds props without reaching far enough to
      // shade whole wall panels, which reads as dirt rather than as contact.
      cfg.aoRadius = 1.3;
      // Occluders keep contributing out to one full radius. The previous 0.7
      // truncated the falloff well inside the sample radius, so the outer two
      // thirds of every sample kernel were thrown away and the effect was much
      // weaker than `intensity` suggested.
      cfg.distanceFalloff = 1.0;
      // With the ambient now a fifth of what it was, AO has a fifth as much
      // light to remove, so the exponent has to come up to keep contact
      // shadows readable. 5.0 is firm but still short of the sooty look.
      cfg.intensity = 5.0;
      cfg.screenSpaceRadius = false;
      // Occlusion is tinted with the shadow blue of the art direction rather
      // than pure black, matching the hemisphere fill so AO and shadow agree.
      cfg.color.setRGB(0.055, 0.072, 0.098, THREE.LinearSRGBColorSpace);
      cfg.colorMultiply = true;
      // 6, not 12. The denoiser is a joint bilateral blur of the AO buffer; at
      // radius 12 it smears a 1.3 m contact shadow across ~24 px and the thing
      // it is denoising stops being visible at all.
      cfg.denoiseRadius = 6;

      switch (settings.ssaoQuality) {
        case 'low':
          cfg.aoSamples = 8;
          cfg.denoiseSamples = 4;
          cfg.denoiseIterations = 1;
          cfg.halfRes = true;
          break;
        case 'medium':
          cfg.aoSamples = 16;
          cfg.denoiseSamples = 8;
          cfg.denoiseIterations = 2;
          cfg.halfRes = true;
          break;
        default:
          cfg.aoSamples = settings.tier === 'ultra' ? 32 : 16;
          cfg.denoiseSamples = 8;
          cfg.denoiseIterations = 2;
          cfg.halfRes = false;
          break;
      }
      cfg.depthAwareUpsampling = true;
      // `autosetGamma` ties gamma correction to renderToScreen, which is
      // exactly right here: we always render into a linear HDR buffer.

      composer.addPass(pass);
      this.aoPass = pass;
    } catch (err) {
      console.error('[RenderPipeline] N8AO unavailable; continuing without AO.', err);
      this.aoPass = null;
    }
  }

  private addMotionBlur(
    composer: EffectComposer,
    ctx: EngineContext,
    settings: QualitySettings,
  ): void {
    try {
      const samples =
        settings.tier === 'low' || settings.tier === 'medium'
          ? MOTION_BLUR_SAMPLES_LOW
          : MOTION_BLUR_SAMPLES_HIGH;
      // 0.4 keeps the blur perceptible on a fast turn but invisible on the
      // small camera drift of idle sway — anything higher smears the crosshair.
      this.motionBlurEffect = new MotionBlurEffect(samples, 0.4, 0.03);
      this.motionPass = new EffectPass(ctx.camera, this.motionBlurEffect);
      composer.addPass(this.motionPass);
    } catch (err) {
      console.error('[RenderPipeline] Motion blur unavailable.', err);
      this.motionBlurEffect = null;
      this.motionPass = null;
    }
  }

  private addToneMapping(
    composer: EffectComposer,
    ctx: EngineContext,
    settings: QualitySettings,
  ): void {
    const effects: Effect[] = [];

    if (settings.bloom) {
      try {
        this.bloomEffect = new BloomEffect({
          // Mip-chain blur: a proper wide, soft falloff instead of the boxy
          // Kawase chain, and it costs less at high radii.
          mipmapBlur: true,
          // Threshold is measured on the raw HDR buffer, BEFORE the exposure
          // multiply — so it is a statement about scene radiance, not about
          // output level. Sunlit sand seen at this grazing angle sits near 0.18
          // and a sun-facing white wall near 0.55, so 0.62 lets the brightest
          // sun-struck edges just tip over and glow while the ground never
          // does. Dropping it to catch more would haze the whole frame, which
          // is the single fastest way to undo the contrast this pass exists to
          // preserve.
          luminanceThreshold: 0.62,
          // Wide smoothing so the bloom fades in over most of a stop instead of
          // switching on at a hard radiance boundary that crawls as you walk.
          luminanceSmoothing: 0.4,
          intensity: 0.9,
          radius: 0.76,
          levels: settings.tier === 'low' ? 6 : 8,
          blendFunction: BlendFunction.ADD,
        });
        effects.push(this.bloomEffect);
      } catch (err) {
        console.error('[RenderPipeline] Bloom unavailable.', err);
        this.bloomEffect = null;
      }
    }

    // Exposure sits between bloom and the tone mapper so it scales the total
    // light reaching the sensor, which is what a real exposure control does.
    try {
      // 1.42, not 1.0. Cutting the ambient from "the whole sky, twice" down to
      // roughly a quarter of the direct sun (see AMBIENT_TO_DIRECT in
      // Lighting.ts) removes about 40% of the total light in the frame. That is
      // the entire point — it is what creates shadows — but the *lit* half must
      // not come down with it, so the sensor is opened back up to put sunlit
      // sand near the top of the curve. Net result: lit surfaces land close to
      // where they were, shadowed surfaces at a fifth of them.
      //
      // 1.75 was tried first and clipped: 9% of the market frame sat above 0.85
      // display luma with the sunlit stucco reading as paper white.
      this.exposureEffect = new ExposureEffect(1.42);
      effects.push(this.exposureEffect);
    } catch (err) {
      console.error('[RenderPipeline] Exposure effect unavailable.', err);
      this.exposureEffect = null;
    }

    try {
      this.toneMappingEffect = new ToneMappingEffect({
        // ACES: the highlight rolloff and hue shift that reads as "film".
        // Adaptive/auto exposure is deliberately off — an auto-exposing FPS
        // pumps brightness every time the player looks at the sky.
        mode: ToneMappingMode.ACES_FILMIC,
        blendFunction: BlendFunction.SRC,
      });
      effects.push(this.toneMappingEffect);
    } catch (err) {
      console.error('[RenderPipeline] Tone mapping unavailable; the image will be clipped.', err);
      this.toneMappingEffect = null;
    }

    if (effects.length > 0) composer.addPass(new EffectPass(ctx.camera, ...effects));
  }

  private addAntiAliasing(
    composer: EffectComposer,
    ctx: EngineContext,
    settings: QualitySettings,
  ): void {
    if (settings.antialias === 'none') return;

    try {
      if (settings.antialias === 'fxaa') {
        composer.addPass(new EffectPass(ctx.camera, new FXAAEffect()));
        return;
      }

      // 'taa' is accepted but resolved to SMAA: temporal AA without per-object
      // motion vectors ghosts badly on every moving character and tracer, which
      // is far more objectionable than the residual aliasing SMAA leaves. The
      // mild unsharp in the grade pass makes up the lost acutance.
      const smaa = new SMAAEffect({
        preset:
          settings.tier === 'low' || settings.tier === 'medium' ? SMAAPreset.MEDIUM : SMAAPreset.ULTRA,
        edgeDetectionMode: EdgeDetectionMode.COLOR,
      });
      // The buffer is tone mapped but still *linear*, so perceptual contrast at
      // the low end is compressed relative to sRGB; the default 0.1 threshold
      // would miss edges in shadow.
      smaa.edgeDetectionMaterial.edgeDetectionThreshold = 0.045;
      composer.addPass(new EffectPass(ctx.camera, smaa));
    } catch (err) {
      console.error('[RenderPipeline] Anti-aliasing unavailable; the image will alias.', err);
    }
  }

  // ── fallback ───────────────────────────────────────────────────────────────

  /** Last-resort path used only if the composer could not be constructed. */
  private renderFallback(ctx: EngineContext): void {
    const renderer = ctx.renderer;
    renderer.setRenderTarget(null);
    renderer.autoClear = true;
    renderer.render(ctx.scene, ctx.camera);
    renderer.autoClear = false;
    renderer.clearDepth();
    renderer.render(ctx.viewScene, ctx.viewCamera);
    renderer.autoClear = true;
  }
}
