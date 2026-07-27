/**
 * NOISEFIELD — Sky.
 *
 * Renders the physically based atmosphere (see `shaders/Atmosphere.ts`) two
 * ways from one shader:
 *
 *   1. VISIBLE DOME — a single full-screen triangle drawn last among the opaque
 *      objects with `depthTest: true, depthWrite: false` and `gl_Position.z = w`.
 *      Because it draws after the world, early-Z rejects every pixel already
 *      covered by geometry, so on a typical frame the scattering march runs on
 *      maybe a third of the screen. It is a *screen* pass rather than a dome
 *      mesh so there is no sphere to fall outside `camera.far` and no seam.
 *
 *   2. IMAGE-BASED LIGHT — the same shader on an inverted unit box, pushed
 *      through `PMREMGenerator` into a roughness-prefiltered cube map and
 *      assigned to `scene.environment`. This is not optional polish: without a
 *      real sky in the environment slot, every metal in the game reflects a
 *      flat grey studio and instantly reads as a tech demo. With it, a rusted
 *      barrel picks up the warm horizon on one side and the blue zenith on the
 *      other, for free, on every surface.
 *
 * The IBL is regenerated only when `Lighting.atmosphereVersion` changes, so a
 * static sun costs literally zero per-frame work beyond the visible dome.
 *
 * The sun disc is excluded from the IBL (`uSunDiscIntensity = 0` in cube mode):
 * a 0.5°-wide, 100×-brighter-than-sky disc smeared across a 256² cube face
 * turns into an enormous white blob, and the direct specular highlight is
 * already handled correctly by the DirectionalLight.
 */

import * as THREE from 'three';
import type { EngineContext, FrameContext, QualityTier, System } from '../core/Types';
import {
  DEFAULT_SUN_AZIMUTH_DEG,
  DEFAULT_SUN_ELEVATION_DEG,
  GROUND_ALBEDO,
  LIGHT_STEPS_HIGH,
  LIGHT_STEPS_LOW,
  MIE_G,
  SUN_INTENSITY,
  sunDirectionFromAngles,
  VIEW_STEPS_HIGH,
  VIEW_STEPS_LOW,
} from './shaders/Atmosphere';
import { SKY_FRAGMENT, SKY_VERTEX_CUBE, SKY_VERTEX_SCREEN } from './shaders/SkyShader';
import type { Lighting } from './Lighting';

/** Cube face resolution handed to PMREMGenerator. 256 is its sweet spot. */
const IBL_SIZE = 256;
/** Lower bound between IBL rebuilds, seconds. Guards a sun animated per frame. */
const IBL_MIN_INTERVAL = 0.25;

/**
 * Radiance multiplier for the solar disc, on top of the scattering intensity.
 * Tuned so the disc sits well above the bloom threshold (producing a real
 * aureole) without saturating a half-float buffer.
 */
const SUN_DISC_INTENSITY = 8.0;

export class Sky implements System {
  readonly name = 'sky';

  /** Shared uniform block: the screen and cube materials point at the same object. */
  readonly uniforms = {
    uSunDirection: new THREE.Uniform<THREE.Vector3>(new THREE.Vector3(0, 1, 0)),
    uSunIntensity: new THREE.Uniform<number>(SUN_INTENSITY),
    uSunDiscIntensity: new THREE.Uniform<number>(SUN_DISC_INTENSITY),
    uMieG: new THREE.Uniform<number>(MIE_G),
    uEyeAltitude: new THREE.Uniform<number>(2),
    uTime: new THREE.Uniform<number>(0),
    uCloudAmount: new THREE.Uniform<number>(0.38),
    uCloudCoverage: new THREE.Uniform<number>(0.3),
    uGroundAlbedo: new THREE.Uniform<THREE.Color>(
      new THREE.Color().setRGB(GROUND_ALBEDO[0], GROUND_ALBEDO[1], GROUND_ALBEDO[2], THREE.LinearSRGBColorSpace),
    ),
    uInvProjection: new THREE.Uniform<THREE.Matrix4>(new THREE.Matrix4()),
    uCameraWorld: new THREE.Uniform<THREE.Matrix4>(new THREE.Matrix4()),
  };

  /** The prefiltered environment currently assigned to the scene, if any. */
  environment: THREE.Texture | null = null;

  private screenMaterial: THREE.ShaderMaterial | null = null;
  private cubeMaterial: THREE.ShaderMaterial | null = null;
  private mesh: THREE.Mesh | null = null;
  private cubeScene: THREE.Scene | null = null;
  private pmrem: THREE.PMREMGenerator | null = null;
  private envTarget: THREE.WebGLRenderTarget | null = null;

  private ctx: EngineContext | null = null;
  private lighting: Lighting | null = null;
  private lastAtmosphereVersion = -1;
  private iblCooldown = 0;
  private iblDirty = true;
  private failed = false;

  /** Fallback sun used when no Lighting system is registered. */
  private readonly fallbackSun = new THREE.Vector3();

  constructor() {
    sunDirectionFromAngles(DEFAULT_SUN_ELEVATION_DEG, DEFAULT_SUN_AZIMUTH_DEG, this.fallbackSun);
    this.uniforms.uSunDirection.value.copy(this.fallbackSun).negate();
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctx = ctx;

    try {
      const defines = Sky.qualityDefines(ctx.settings.tier);

      this.screenMaterial = new THREE.ShaderMaterial({
        name: 'noisefield.sky.screen',
        vertexShader: SKY_VERTEX_SCREEN,
        fragmentShader: SKY_FRAGMENT,
        uniforms: this.uniforms,
        defines: { ...defines },
        depthWrite: false,
        depthTest: true,
        // The projected full-screen triangle's winding is not worth reasoning
        // about; disabling culling is free for three vertices.
        side: THREE.DoubleSide,
        fog: false,
      });

      // A single triangle covering the viewport beats a quad: no diagonal seam
      // and one fewer vertex worth of overlapping fragment invocations.
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
      );

      const mesh = new THREE.Mesh(geometry, this.screenMaterial);
      mesh.name = 'noisefield.sky';
      mesh.frustumCulled = false;
      // Drawn after every other opaque object so early-Z can reject it.
      mesh.renderOrder = 1000;
      mesh.matrixAutoUpdate = false;
      mesh.onBeforeRender = (_renderer, _scene, camera): void => {
        // Rebuild the view ray basis for whatever camera is drawing us; this
        // keeps the sky exact through FOV changes (ADS) with no extra plumbing.
        this.uniforms.uInvProjection.value.copy(camera.projectionMatrixInverse);
        this.uniforms.uCameraWorld.value.copy(camera.matrixWorld);
      };
      this.mesh = mesh;
      ctx.scene.add(mesh);

      // Nothing else may paint over the sky.
      ctx.scene.background = null;

      // ── IBL capture rig ─────────────────────────────────────────────────────
      this.cubeMaterial = new THREE.ShaderMaterial({
        name: 'noisefield.sky.cube',
        vertexShader: SKY_VERTEX_CUBE,
        fragmentShader: SKY_FRAGMENT,
        uniforms: this.uniforms,
        // Cheaper march: the result is blurred into low-order spherical
        // harmonics anyway, so extra samples buy nothing visible.
        defines: { ATM_VIEW_STEPS: VIEW_STEPS_LOW, ATM_LIGHT_STEPS: LIGHT_STEPS_LOW },
        side: THREE.BackSide,
        depthWrite: false,
        depthTest: false,
        fog: false,
      });

      const cubeScene = new THREE.Scene();
      const box = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), this.cubeMaterial);
      box.frustumCulled = false;
      cubeScene.add(box);
      this.cubeScene = cubeScene;

      this.pmrem = new THREE.PMREMGenerator(ctx.renderer);
      this.pmrem.compileCubemapShader();
    } catch (err) {
      // A broken sky must not take the whole renderer down with it.
      this.failed = true;
      console.error('[Sky] initialisation failed; falling back to a flat background.', err);
      ctx.scene.background = new THREE.Color(0x7d9bbd);
    }

    this.lighting = ctx.get<Lighting>('lighting') ?? null;
    this.syncSun();
    this.regenerateEnvironment();
  }

  update(ctx: EngineContext, frame: FrameContext): void {
    if (this.failed) return;

    this.uniforms.uTime.value = frame.elapsed;
    // Eye altitude feeds the scattering origin; it matters at the horizon,
    // where a couple of metres visibly changes how far the ground curves away.
    this.uniforms.uEyeAltitude.value = Math.max(0.5, ctx.camera.position.y);

    if (this.lighting === null) this.lighting = ctx.get<Lighting>('lighting') ?? null;

    if (this.syncSun()) this.iblDirty = true;

    // rawDt, not dt: `dt` is zeroed while the game is paused, and an IBL
    // refresh requested at the moment of pausing would otherwise never fire.
    this.iblCooldown -= frame.rawDt;
    if (this.iblDirty && this.iblCooldown <= 0) {
      this.regenerateEnvironment();
    }
  }

  dispose(): void {
    this.mesh?.geometry.dispose();
    this.mesh?.removeFromParent();
    this.screenMaterial?.dispose();
    this.cubeMaterial?.dispose();
    if (this.cubeScene) {
      for (const child of this.cubeScene.children) {
        if (child instanceof THREE.Mesh) child.geometry.dispose();
      }
    }
    this.envTarget?.dispose();
    this.pmrem?.dispose();
    if (this.ctx) {
      if (this.ctx.scene.environment === this.environment) this.ctx.scene.environment = null;
      if (this.ctx.viewScene.environment === this.environment) this.ctx.viewScene.environment = null;
    }
    this.environment = null;
  }

  // ── public API ─────────────────────────────────────────────────────────────

  /** Radiance multiplier of the solar disc in the visible sky. */
  set sunDiscIntensity(v: number) {
    this.uniforms.uSunDiscIntensity.value = v;
  }
  get sunDiscIntensity(): number {
    return this.uniforms.uSunDiscIntensity.value;
  }

  /** 0 removes the cloud deck entirely. */
  set cloudAmount(v: number) {
    this.uniforms.uCloudAmount.value = THREE.MathUtils.clamp(v, 0, 1);
    this.iblDirty = true;
  }
  get cloudAmount(): number {
    return this.uniforms.uCloudAmount.value;
  }

  /** Fraction of the sky the cloud deck covers, 0..1. */
  set cloudCoverage(v: number) {
    this.uniforms.uCloudCoverage.value = THREE.MathUtils.clamp(v, 0, 1);
    this.iblDirty = true;
  }
  get cloudCoverage(): number {
    return this.uniforms.uCloudCoverage.value;
  }

  /**
   * Re-renders the sky into a prefiltered environment map and assigns it to
   * both the world and the viewmodel scenes.
   */
  regenerateEnvironment(): void {
    const ctx = this.ctx;
    const pmrem = this.pmrem;
    const cubeScene = this.cubeScene;
    if (this.failed || !ctx || !pmrem || !cubeScene) return;

    const previous = this.envTarget;
    const disc = this.uniforms.uSunDiscIntensity.value;

    try {
      // Drop the disc for the capture: prefiltering a point-bright source into
      // 256² faces produces a huge, obviously-wrong white smear, and the
      // DirectionalLight already provides the correct direct specular.
      this.uniforms.uSunDiscIntensity.value = 0;
      // Clouds do belong in the IBL — they are a large, diffuse light source.
      const target = pmrem.fromScene(cubeScene, 0, 0.1, 12);
      this.envTarget = target;
      this.environment = target.texture;

      ctx.scene.environment = target.texture;
      // NOT 1.0. At full strength the PMREM sky delivers the entire
      // hemispherical sky irradiance, which — on top of the hemisphere fill —
      // is more ambient than the sun is key, and every shadow in the game
      // washes out. `Lighting` measures the sky's actual irradiance and solves
      // for the intensity that lands the sun-to-sky ratio on its target; see
      // AMBIENT_TO_DIRECT in Lighting.ts. Falls back to 1.0 only if no Lighting
      // system is registered, in which case nothing else is balanced either.
      ctx.scene.environmentIntensity = this.lighting?.environmentIntensity ?? 1.0;
      // The viewmodel lives in its own scene and would otherwise render with no
      // ambient at all, making the weapon look pasted on. It runs richer than
      // the world for the same reason `VIEWMODEL_FILL_BOOST` exists.
      ctx.viewScene.environment = target.texture;
      ctx.viewScene.environmentIntensity = (this.lighting?.environmentIntensity ?? 1.0) * 2.2;

      previous?.dispose();
    } catch (err) {
      console.error('[Sky] PMREM generation failed; keeping the previous environment.', err);
      this.envTarget = previous;
    } finally {
      this.uniforms.uSunDiscIntensity.value = disc;
    }

    this.iblDirty = false;
    this.iblCooldown = IBL_MIN_INTERVAL;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  /**
   * Pulls the sun from `Lighting` (the owner of the canonical direction).
   *
   * @returns true when the sun actually moved, i.e. the IBL is now stale.
   */
  private syncSun(): boolean {
    const lighting = this.lighting;
    if (!lighting) {
      // No Lighting registered: drive ourselves so the sky is still correct.
      this.uniforms.uSunDirection.value.copy(this.fallbackSun).negate();
      return false;
    }

    if (lighting.atmosphereVersion === this.lastAtmosphereVersion) return false;
    this.lastAtmosphereVersion = lighting.atmosphereVersion;
    // The shader wants the vector pointing AT the sun; the engine convention is
    // the opposite, so negate exactly once, here.
    this.uniforms.uSunDirection.value.copy(lighting.sunDirection).negate().normalize();
    return true;
  }

  /** March quality per tier. Sky pixels are the only ones that pay for this. */
  private static qualityDefines(tier: QualityTier): Record<string, number> {
    switch (tier) {
      case 'low':
        return { ATM_VIEW_STEPS: VIEW_STEPS_LOW, ATM_LIGHT_STEPS: LIGHT_STEPS_LOW };
      case 'medium':
        return { ATM_VIEW_STEPS: 10, ATM_LIGHT_STEPS: 4 };
      default:
        return { ATM_VIEW_STEPS: VIEW_STEPS_HIGH, ATM_LIGHT_STEPS: LIGHT_STEPS_HIGH };
    }
  }
}
