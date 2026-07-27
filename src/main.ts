/**
 * NOISEFIELD — boot and integration.
 *
 * Registration order is the initialisation order, and it is load-bearing:
 *
 *   materials  first — every other subsystem asks it for shared PBR materials.
 *   physics    early — the level, player, AI and ballistics all query it.
 *   lighting   before sky — Sky reads the sun direction from Lighting.
 *   level      before enemies — EnemyManager samples its bounds for the navgrid.
 *   player     before enemies — EnemyManager binds it as the AI target.
 *   bridge     after the above — it introduces subsystems to each other.
 *   pipeline   last — N8AO and the composer want a populated scene.
 */

import * as THREE from 'three';
import { Engine } from './core/Engine';
import { Bridge } from './core/Bridge';
import { DEFAULT_SETTINGS, type QualitySettings } from './core/Types';

import { MaterialLibrary } from './materials/MaterialLibrary';
import { Physics } from './physics/Physics';
import { Lighting } from './render/Lighting';
import { Sky } from './render/Sky';
import { Level } from './world/Level';
import { Decals } from './world/Decals';
import { Player } from './entities/Player';
import { WeaponSystem } from './weapons/WeaponSystem';
import { EnemyManager } from './entities/EnemyManager';
import { Particles } from './fx/Particles';
import { Impacts } from './fx/Impacts';
import { Tracers } from './fx/Tracers';
import { Volumetrics } from './fx/Volumetrics';
import { AudioEngine } from './audio/AudioEngine';
import { RenderPipeline } from './render/RenderPipeline';
import { applyCaptureSettings, captureRequested, runCapture } from './core/Capture';
import { HUD } from './ui/HUD';
import { Menu } from './ui/Menu';

// ─────────────────────────────────────────────────────────────────────────────
// Boot UI
// ─────────────────────────────────────────────────────────────────────────────

const bootEl = document.getElementById('boot');
const barEl = document.getElementById('boot-bar');
const statusEl = document.getElementById('boot-status');
const enterEl = document.getElementById('boot-enter') as HTMLButtonElement | null;
const fatalEl = document.getElementById('fatal');

function progress(pct: number, label: string): void {
  if (barEl) barEl.style.width = `${Math.round(pct * 100)}%`;
  if (statusEl) statusEl.textContent = label;
}

function fatal(err: unknown): void {
  console.error(err);
  if (!fatalEl) return;
  fatalEl.style.display = 'block';
  const msg = err instanceof Error ? `${err.message}\n\n${err.stack ?? ''}` : String(err);
  fatalEl.textContent = `NOISEFIELD failed to start.\n\n${msg}`;
}

/** Yield to the browser so the boot bar actually paints between stages. */
const nextFrame = (): Promise<void> =>
  new Promise((resolve) => requestAnimationFrame(() => resolve()));

// ─────────────────────────────────────────────────────────────────────────────
// Quality auto-detection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Pick a starting tier from the GPU string and the device pixel ratio. This is
 * a heuristic, not a benchmark — the adaptive scaler below corrects it from
 * measured frame time within a couple of seconds either way.
 */
function detectSettings(): Partial<QualitySettings> {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) return { tier: 'low', shadows: false, ssao: false, volumetrics: false, antialias: 'fxaa' };

  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : '';
  const maxAniso = gl.getExtension('EXT_texture_filter_anisotropic');
  const anisotropy = maxAniso ? gl.getParameter(maxAniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) : 1;

  const lowered = renderer.toLowerCase();
  const softwareRasterised = /swiftshader|llvmpipe|software|basic render/.test(lowered);
  const integrated = /intel|uhd|iris|apple m1|adreno|mali|powervr/.test(lowered);

  if (softwareRasterised) {
    // Headless capture and CPU fallback: keep everything on so screenshots are
    // representative, but shrink the buffer so a frame still completes.
    return { tier: 'high', renderScale: 0.75, shadowMapSize: 1024, anisotropy: Math.min(anisotropy, 8) };
  }
  if (integrated) {
    return { tier: 'medium', renderScale: 0.85, shadowMapSize: 1536, shadowCascades: 3, anisotropy: Math.min(anisotropy, 8) };
  }
  return { tier: 'ultra', anisotropy: Math.min(anisotropy, 16) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

async function boot(): Promise<void> {
  const canvas = document.getElementById('viewport') as HTMLCanvasElement;
  if (!canvas) throw new Error('#viewport canvas missing');

  progress(0.04, 'Detecting hardware');
  const engine = new Engine(canvas, detectSettings());
  // Review captures must not inherit the headless auto-detect downgrade.
  applyCaptureSettings(engine.settings);
  await nextFrame();

  progress(0.1, 'Forging materials');
  engine.add(new MaterialLibrary());
  engine.add(new Physics());
  engine.add(new Lighting());
  engine.add(new Sky());
  engine.add(new Decals());
  engine.add(new Level());
  engine.add(new Player());
  engine.add(new WeaponSystem());
  engine.add(new EnemyManager());
  engine.add(new Particles());
  engine.add(new Impacts());
  engine.add(new Tracers());
  engine.add(new Volumetrics());
  engine.add(new AudioEngine());
  engine.add(new HUD());
  engine.add(new Menu());
  const bridge = engine.add(new Bridge());
  const pipeline = engine.add(new RenderPipeline());
  engine.setPipeline(pipeline);

  /*
   * Make the pause menu actually pause the simulation.
   *
   * `EngineContext` deliberately does not expose the engine, so a UI module
   * cannot stop the world by itself — it can only announce intent on the bus.
   * That keeps a UI bug from being able to wedge the game, but it means the
   * last connection has to be made here in the composition root.
   *
   * Without this line the menu opens and its settings apply, but `setPaused`
   * is never called from anywhere: `Engine.paused` stays false forever and the
   * world keeps running behind the menu. That was the shipped state, and it
   * made the pause gating inside the engine unreachable code.
   *
   * `setPaused` re-emits `game:pause`, and the menu accepts its own echo
   * idempotently, so this does not loop.
   */
  engine.bus.on('game:pause', (p) => engine.setPaused(p.paused));

  progress(0.25, 'Building world');
  await nextFrame();

  await engine.init();
  bridge.attachCamera(engine.camera);

  progress(0.85, 'Compiling shaders');
  await nextFrame();

  // Warm the shader cache before the first presented frame. Without this the
  // first second of play is a slideshow of compile hitches as each material is
  // encountered for the first time.
  try {
    engine.renderer.compile(engine.scene, engine.camera);
    engine.renderer.compile(engine.viewScene, engine.viewCamera);
  } catch (err) {
    console.warn('[boot] shader precompile skipped:', err);
  }

  progress(1, 'Ready');

  // Expose for the headless capture harness and for debugging.
  (window as unknown as Record<string, unknown>).__NOISEFIELD = {
    engine,
    THREE,
    settings: engine.settings,
    get stats() {
      return {
        frameTimeMs: engine.frameTimeMs,
        calls: engine.renderer.info.render.calls,
        triangles: engine.renderer.info.render.triangles,
        programs: engine.renderer.info.programs?.length ?? 0,
      };
    },
  };


  // Headless visual-QA path: compose one deterministic frame and stop. Must run
  // before `engine.start()` so the rAF loop never competes with the fixed-step
  // capture, which would make the output non-reproducible.
  if (captureRequested()) {
    bootEl?.style.setProperty('display', 'none');
    const capAudio = engine.get<AudioEngine>('audio');
    void capAudio?.resume();
    await runCapture(engine);
    return;
  }

  engine.start();

  // ── deploy ────────────────────────────────────────────────────────────────
  const deploy = (): void => {
    bootEl?.classList.add('hidden');
    engine.input.requestPointerLock();
    // Audio contexts may only start from a user gesture.
    const audio = engine.get<AudioEngine>('audio');
    void audio?.resume();
    window.setTimeout(() => bootEl?.style.setProperty('display', 'none'), 700);
  };

  if (enterEl) {
    enterEl.classList.add('show');
    enterEl.addEventListener('click', deploy, { once: false });
  }
  // Clicking anywhere on the boot screen also deploys.
  bootEl?.addEventListener('click', (e) => {
    if (e.target === enterEl) return;
    deploy();
  });
  // Re-acquire pointer lock by clicking the canvas after Esc.
  canvas.addEventListener('click', () => {
    if (!engine.input.pointerLocked && !engine.isPaused) engine.input.requestPointerLock();
  });

  startAdaptiveScaler(engine);
}

/**
 * Keep the frame time near 16.7ms by trading render scale, then effects.
 *
 * Measured over a 90-frame window so a single GC pause or an alt-tab cannot
 * trigger a downgrade, and with a wide dead-band plus a cooldown so the scaler
 * cannot oscillate — a resolution that visibly pumps up and down is worse than
 * one that is simply a bit low.
 */
function startAdaptiveScaler(engine: Engine): void {
  const MIN_SCALE = 0.6;
  const MAX_SCALE = 1.0;
  let cooldown = 3;

  window.setInterval(() => {
    if (engine.isPaused) return;
    cooldown -= 0.5;
    if (cooldown > 0) return;

    const ms = engine.frameTimeMs;
    const s = engine.settings;

    if (ms > 24 && s.renderScale > MIN_SCALE) {
      s.renderScale = Math.max(MIN_SCALE, s.renderScale - 0.1);
      window.dispatchEvent(new Event('resize'));
      cooldown = 3;
    } else if (ms < 13 && s.renderScale < MAX_SCALE) {
      s.renderScale = Math.min(MAX_SCALE, s.renderScale + 0.05);
      window.dispatchEvent(new Event('resize'));
      cooldown = 4;
    }
  }, 500);
}

boot().catch(fatal);
