import * as THREE from 'three';
import { EventBus } from './EventBus';
import { Input } from './Input';
import { DEFAULT_SETTINGS, type EngineContext, type FrameContext, type QualitySettings, type System } from './Types';

/** Anything capable of producing the final image for a frame. */
export interface IRenderPipeline extends System {
  render(ctx: EngineContext, frame: FrameContext): void;
}

const FIXED_STEP = 1 / 120;
const MAX_STEPS_PER_FRAME = 8;
/** Clamp for `dt`. A tab that was backgrounded must not teleport the sim. */
const MAX_FRAME_DT = 0.1;

/**
 * Owns the canvas, the render loop and subsystem lifecycle.
 *
 * Frame order is fixed and deliberate:
 *   input.beginFrame → fixedUpdate×N → update → render → lateUpdate → input.endFrame
 *
 * `update` runs after the physics steps so systems observe a settled world, and
 * `lateUpdate` runs after render so post-render readbacks (e.g. TAA history,
 * motion vectors) see the frame that was actually presented.
 */
export class Engine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  /** Viewmodel scene, drawn over the world with its own depth range. */
  readonly viewScene: THREE.Scene;
  readonly viewCamera: THREE.PerspectiveCamera;
  readonly bus = new EventBus();
  readonly input: Input;
  readonly settings: QualitySettings;
  readonly canvas: HTMLCanvasElement;

  private readonly systems: System[] = [];
  private readonly byName = new Map<string, System>();
  private pipeline: IRenderPipeline | null = null;

  private accumulator = 0;
  private lastTime = 0;
  private frameIndex = 0;
  private elapsed = 0;
  private running = false;
  private rafHandle = 0;
  private paused = false;

  private width = 1;
  private height = 1;

  /** Rolling average frame time in ms, exposed for the HUD/perf overlay. */
  frameTimeMs = 16.7;

  constructor(canvas: HTMLCanvasElement, settings: Partial<QualitySettings> = {}) {
    this.canvas = canvas;
    this.settings = { ...DEFAULT_SETTINGS, ...settings };

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false, // handled by the post stack (SMAA/TAA)
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
      alpha: false,
      // Required so the headless capture harness can read the framebuffer.
      preserveDrawingBuffer: true,
    });

    // Colour management: everything authored in linear, tonemapped to sRGB at
    // the end of the post chain. ACES gives the filmic highlight rolloff that
    // reads as "cinematic" rather than the blown-out look of linear clamping.
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = this.settings.shadows;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = true;
    this.renderer.info.autoReset = false;

    this.scene = new THREE.Scene();
    this.viewScene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(this.settings.fov, 1, 0.05, 800);
    this.camera.position.set(0, 1.7, 0);
    // A tight near/far pair for the viewmodel maximises depth precision and
    // guarantees the weapon never z-fights with or clips into world geometry.
    this.viewCamera = new THREE.PerspectiveCamera(60, 1, 0.008, 12);

    this.input = new Input(canvas, this.settings);

    this.resize();
    window.addEventListener('resize', this.resize);
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  // ── system registry ────────────────────────────────────────────────────────

  add<T extends System>(system: T): T {
    if (this.byName.has(system.name)) {
      throw new Error(`Duplicate system name: ${system.name}`);
    }
    this.systems.push(system);
    this.byName.set(system.name, system);
    return system;
  }

  setPipeline(pipeline: IRenderPipeline): void {
    this.pipeline = pipeline;
  }

  get<T extends System>(name: string): T | undefined {
    return this.byName.get(name) as T | undefined;
  }

  get context(): EngineContext {
    return {
      renderer: this.renderer,
      scene: this.scene,
      camera: this.camera,
      viewScene: this.viewScene,
      viewCamera: this.viewCamera,
      bus: this.bus,
      input: this.input,
      settings: this.settings,
      width: this.width,
      height: this.height,
      get: <T extends System>(n: string) => this.byName.get(n) as T | undefined,
    };
  }

  async init(): Promise<void> {
    const ctx = this.context;
    for (const s of this.systems) {
      if (!s.init) continue;
      try {
        await s.init(ctx);
      } catch (err) {
        // One broken subsystem must not prevent the rest of the game booting.
        console.error(`[Engine] init failed for "${s.name}":`, err);
      }
    }
    this.resize();
  }

  // ── loop ───────────────────────────────────────────────────────────────────

  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now();
    this.rafHandle = requestAnimationFrame(this.tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafHandle);
  }

  setPaused(paused: boolean): void {
    if (this.paused === paused) return;
    this.paused = paused;
    this.bus.emit('game:pause', { paused });
    // Reset the clock so unpausing does not deliver one enormous dt.
    this.lastTime = performance.now();
    this.accumulator = 0;
  }

  get isPaused(): boolean {
    return this.paused;
  }

  private tick = (now: number): void => {
    if (!this.running) return;
    this.rafHandle = requestAnimationFrame(this.tick);

    const rawDt = (now - this.lastTime) / 1000;
    this.lastTime = now;
    this.frameTimeMs += ((rawDt * 1000) - this.frameTimeMs) * 0.1;

    const dt = Math.min(rawDt, MAX_FRAME_DT);
    this.elapsed += dt;
    this.frameIndex++;

    const frame: FrameContext = {
      dt: this.paused ? 0 : dt,
      rawDt,
      elapsed: this.elapsed,
      frame: this.frameIndex,
    };
    const ctx = this.context;

    this.input.beginFrame();

    if (!this.paused) {
      this.accumulator += dt;
      let steps = 0;
      while (this.accumulator >= FIXED_STEP && steps < MAX_STEPS_PER_FRAME) {
        for (const s of this.systems) s.fixedUpdate?.(ctx, FIXED_STEP);
        this.accumulator -= FIXED_STEP;
        steps++;
      }
      // If we blew the step budget, drop the backlog rather than accumulating
      // debt that would make the next frames progressively worse.
      if (steps >= MAX_STEPS_PER_FRAME) this.accumulator = 0;
    }

    for (const s of this.systems) s.update?.(ctx, frame);

    this.renderer.info.reset();
    if (this.pipeline) {
      this.pipeline.render(ctx, frame);
    } else {
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);
      this.renderer.autoClear = false;
      this.renderer.clearDepth();
      this.renderer.render(this.viewScene, this.viewCamera);
      this.renderer.autoClear = true;
    }

    for (const s of this.systems) s.lateUpdate?.(ctx, frame);

    this.input.endFrame();
  };

  /** Advance exactly one frame. Used by the headless capture harness. */
  stepOnce(dtMs = 16.667): void {
    this.tickManual(dtMs);
  }

  private tickManual(dtMs: number): void {
    const dt = dtMs / 1000;
    this.elapsed += dt;
    this.frameIndex++;
    const frame: FrameContext = { dt, rawDt: dt, elapsed: this.elapsed, frame: this.frameIndex };
    const ctx = this.context;
    this.input.beginFrame();
    this.accumulator += dt;
    let steps = 0;
    while (this.accumulator >= FIXED_STEP && steps < MAX_STEPS_PER_FRAME) {
      for (const s of this.systems) s.fixedUpdate?.(ctx, FIXED_STEP);
      this.accumulator -= FIXED_STEP;
      steps++;
    }
    for (const s of this.systems) s.update?.(ctx, frame);
    // `info.autoReset` is off, so without this the capture harness reports the
    // sum over every stepped frame rather than the cost of one.
    this.renderer.info.reset();
    if (this.pipeline) this.pipeline.render(ctx, frame);
    for (const s of this.systems) s.lateUpdate?.(ctx, frame);
    this.input.endFrame();
  }

  // ── viewport ───────────────────────────────────────────────────────────────

  private resize = (): void => {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    this.width = w;
    this.height = h;

    const dpr = Math.min(window.devicePixelRatio || 1, 2) * this.settings.renderScale;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);

    const aspect = w / h;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.viewCamera.aspect = aspect;
    this.viewCamera.updateProjectionMatrix();

    for (const s of this.systems) s.resize?.(w, h);
    this.pipeline?.resize?.(w, h);
  };

  private onVisibility = (): void => {
    // Reset timing when returning to a backgrounded tab.
    if (!document.hidden) {
      this.lastTime = performance.now();
      this.accumulator = 0;
    }
  };

  dispose(): void {
    this.stop();
    window.removeEventListener('resize', this.resize);
    document.removeEventListener('visibilitychange', this.onVisibility);
    for (const s of this.systems) s.dispose?.();
    this.input.dispose();
    this.renderer.dispose();
  }
}
