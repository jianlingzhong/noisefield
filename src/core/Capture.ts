import * as THREE from 'three';
import type { Engine } from './Engine';
import type { Player } from '../entities/Player';
import type { WeaponSystem } from '../weapons/WeaponSystem';
import type { QualitySettings, WeaponId } from './Types';

/**
 * Deterministic headless capture harness.
 *
 * Visual review needs frames that are reproducible: the same URL must produce
 * the same pixels every time, or a critic cannot tell a real regression from
 * particle jitter. This drives the engine with a fixed timestep from a fixed
 * camera pose, disables the adaptive resolution scaler, and only signals ready
 * once the frame is fully settled (shadows converged, IBL baked, particles at a
 * known age).
 *
 * Activated by `?shot=<preset>`; ignored entirely in normal play.
 *
 * Query parameters:
 *   shot=<name>       named preset (see PRESETS)
 *   cam=x,y,z         explicit camera position, overrides the preset
 *   look=x,y,z        explicit look-at target, overrides the preset
 *   frames=<n>        simulation frames to advance before capturing
 *   weapon=<id>       weapon to equip
 *   ads=1             aim down sights
 *   fire=<n>          fire n rounds before settling
 *   hud=0             hide the HUD (for pure-render evaluation)
 *   isolate=1         hide the world, leaving the viewmodel on a flat backdrop
 *   bg=RRGGBB         backdrop colour for isolate mode (default mid grey)
 */

export interface ShotPreset {
  readonly name: string;
  readonly pos: [number, number, number];
  readonly look: [number, number, number];
  readonly frames?: number;
  readonly weapon?: WeaponId;
  readonly ads?: boolean;
  readonly fire?: number;
  readonly description: string;
}

/**
 * Camera poses chosen to expose different failure modes: silhouette and
 * lighting at range, material detail up close, shadow contact, interior
 * bounce, and the viewmodel against a bright background.
 */
export const PRESETS: Record<string, ShotPreset> = {
  establishing: {
    name: 'establishing',
    pos: [-34, 6.2, -34],
    look: [4, 2.5, 4],
    frames: 150,
    description: 'Wide golden-hour establishing shot across the compound.',
  },
  courtyard: {
    name: 'courtyard',
    pos: [-14, 1.7, -16],
    look: [2, 3.0, 2],
    frames: 150,
    description: 'Eye-level in the central courtyard facing the hero landmark.',
  },
  alley: {
    name: 'alley',
    pos: [-27, 1.7, 12],
    look: [-27, 1.9, -20],
    frames: 150,
    description: 'Tight west alley — contact shadows, occlusion, close materials.',
  },
  market: {
    name: 'market',
    pos: [26, 1.7, 20],
    look: [26, 2.2, -22],
    frames: 150,
    description: 'East market street with awnings and stalls, strong shadow play.',
  },
  rooftop: {
    name: 'rooftop',
    pos: [-18, 8.4, -8],
    look: [14, 3.0, 10],
    frames: 150,
    description: 'Rooftop overlook — long sightline, atmospheric perspective.',
  },
  interior: {
    name: 'interior',
    pos: [-8, 1.7, -26],
    look: [6, 1.6, -20],
    frames: 150,
    description: 'Building interior — bounce light, ambient occlusion, contrast.',
  },
  weapon: {
    name: 'weapon',
    pos: [-14, 1.7, -14],
    look: [6, 2.2, 4],
    frames: 150,
    weapon: 'ar_vector',
    description: 'Hipfire viewmodel against the courtyard.',
  },
  ads: {
    name: 'ads',
    pos: [-14, 1.7, -14],
    look: [6, 2.0, 4],
    frames: 200,
    weapon: 'ar_vector',
    ads: true,
    description: 'Aiming down sights — optic alignment and reticle.',
  },
  combat: {
    name: 'combat',
    pos: [-12, 1.7, -12],
    look: [8, 1.8, 6],
    frames: 210,
    weapon: 'ar_vector',
    fire: 6,
    description: 'Mid-burst: muzzle flash, tracers, impacts, shells, smoke.',
  },
  arms: {
    name: 'arms',
    pos: [-14, 1.7, -14],
    look: [6, 2.2, 4],
    frames: 90,
    weapon: 'ar_vector',
    description: 'Viewmodel hands and sleeve, intended for use with isolate=1.',
  },
  sun: {
    name: 'sun',
    pos: [10, 2.4, 22],
    look: [-30, 12, -30],
    frames: 150,
    description: 'Facing into the sun — sky, god rays, bloom, flare behaviour.',
  },
};

/** Every valid `WeaponId`, used to validate the `weapon` query parameter. */
export const WEAPON_IDS = [
  'ar_vector',
  'smg_wasp',
  'sniper_longbow',
  'shotgun_breach',
  'pistol_sidearm',
] as const satisfies readonly WeaponId[];

interface CaptureOptions {
  preset: ShotPreset;
  frames: number;
  hud: boolean;
  isolate: boolean;
  bg: number;
}

/**
 * Hide the world so only the first-person viewmodel remains.
 *
 * The viewmodel lives in its own scene (`engine.viewScene`) with its own camera
 * and its own light rig, and is composited over the world buffer by a dedicated
 * pass. That separation means the world can be blanked without touching the
 * weapon's lighting at all: the sun key, the fill and the baked environment map
 * that light the hands are properties of `viewScene`, not children of the world
 * scene, so they survive this untouched.
 *
 * Deliberately does NOT change the viewmodel field of view. Narrowing the FOV
 * would be the obvious way to get a bigger picture of the hands, but it would
 * also change their perspective and foreshortening — the reviewer would then be
 * judging a lens the player never looks through. Magnification is done by
 * cropping the captured image instead, so every pixel under review is a pixel
 * the player actually sees.
 */
function isolateViewmodel(engine: Engine, bg: number): void {
  for (const child of engine.scene.children) child.visible = false;
  engine.scene.background = new THREE.Color(bg);
  // The world's own fog would otherwise tint the flat backdrop.
  engine.scene.fog = null;
}

export function parseVec(raw: string | null): [number, number, number] | null {
  if (!raw) return null;
  const parts = raw.split(',').map((s) => Number(s.trim()));
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) return null;
  return [parts[0], parts[1], parts[2]];
}

export function captureRequested(): boolean {
  const q = new URLSearchParams(window.location.search);
  return q.has('shot') || q.has('cam');
}

/**
 * Apply quality overrides from the query string BEFORE systems initialise.
 *
 * Auto-detection sees SwiftShader in headless Chrome and sensibly drops render
 * scale and shadow resolution — sensible for play, wrong for review, because it
 * makes the renderer look worse than it is on real hardware. Review shots must
 * be captured at the settings a player with a GPU would actually get.
 */
export function applyCaptureSettings(settings: QualitySettings): void {
  const q = new URLSearchParams(window.location.search);
  if (!q.has('shot') && !q.has('cam')) return;

  const tier = q.get('tier');
  if (tier === 'low' || tier === 'medium' || tier === 'high' || tier === 'ultra') settings.tier = tier;

  const num = (key: string, min: number, max: number): number | null => {
    const raw = q.get(key);
    if (raw === null) return null;
    const v = Number(raw);
    return Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : null;
  };
  const bool = (key: string): boolean | null => {
    const raw = q.get(key);
    return raw === null ? null : raw !== '0' && raw !== 'false';
  };

  const scale = num('scale', 0.25, 2);
  if (scale !== null) settings.renderScale = scale;
  const shadowMap = num('shadowmap', 256, 4096);
  if (shadowMap !== null) settings.shadowMapSize = shadowMap;
  const cascades = num('cascades', 1, 4);
  if (cascades !== null) settings.shadowCascades = Math.round(cascades);
  const aniso = num('aniso', 1, 16);
  if (aniso !== null) settings.anisotropy = Math.round(aniso);
  const fov = num('fov', 40, 120);
  if (fov !== null) settings.fov = fov;

  const shadows = bool('shadows');
  if (shadows !== null) settings.shadows = shadows;
  const ssao = bool('ssao');
  if (ssao !== null) settings.ssao = ssao;
  const bloom = bool('bloom');
  if (bloom !== null) settings.bloom = bloom;
  const volumetrics = bool('volumetrics');
  if (volumetrics !== null) settings.volumetrics = volumetrics;
  const motionBlur = bool('motionblur');
  if (motionBlur !== null) settings.motionBlur = motionBlur;

  const aa = q.get('aa');
  if (aa === 'none' || aa === 'fxaa' || aa === 'smaa' || aa === 'taa') settings.antialias = aa;

  // Review defaults: full resolution and full-quality shadows unless the caller
  // deliberately asked for something else.
  if (scale === null) settings.renderScale = 1;
  if (shadowMap === null) settings.shadowMapSize = 2048;
  if (aniso === null) settings.anisotropy = 16;
}

function resolveOptions(): CaptureOptions {
  const q = new URLSearchParams(window.location.search);
  const name = q.get('shot') ?? 'establishing';
  const base = PRESETS[name] ?? PRESETS.establishing;

  const pos = parseVec(q.get('cam')) ?? base.pos;
  const look = parseVec(q.get('look')) ?? base.look;
  // Validate against the real roster rather than casting. An unchecked cast
  // let any string through as a WeaponId, so `?weapon=nonsense` reached
  // `weapons.switchTo()` as a bad key; it also made the claim that every query
  // parameter is allowlisted untrue.
  const weaponRaw = q.get('weapon');
  const weapon: WeaponId | undefined =
    weaponRaw !== null && (WEAPON_IDS as readonly string[]).includes(weaponRaw)
      ? (weaponRaw as WeaponId)
      : base.weapon;
  const ads = q.has('ads') ? q.get('ads') !== '0' : base.ads;
  const fire = q.has('fire') ? Number(q.get('fire')) : base.fire;

  const bgRaw = q.get('bg');
  const bgParsed = bgRaw !== null ? Number.parseInt(bgRaw, 16) : NaN;

  return {
    preset: { ...base, pos, look, weapon, ads, fire },
    frames: Number(q.get('frames') ?? base.frames ?? 150),
    hud: q.get('hud') !== '0',
    isolate: q.get('isolate') === '1',
    // 0x3a3a3a is a neutral mid grey: dark enough that the olive sleeve does not
    // disappear into it, light enough to show the hand's shadowed silhouette.
    bg: Number.isFinite(bgParsed) ? bgParsed : 0x3a3a3a,
  };
}

/**
 * Run the capture. Resolves once the frame is composed and a `#noisefield-ready`
 * marker has been added to the DOM for a headless capture tool to wait on.
 */
export async function runCapture(engine: Engine): Promise<void> {
  const opts = resolveOptions();
  const { preset } = opts;

  // Determinism: a fixed step means particles, springs and animation phases
  // land identically on every run, so two screenshots of the same URL differ
  // only when the code changed.
  const STEP_MS = 1000 / 60;

  const player = engine.get<Player>('player');
  const weapons = engine.get<WeaponSystem>('weapons');

  const camPos = new THREE.Vector3(...preset.pos);
  const lookAt = new THREE.Vector3(...preset.look);

  // Derive yaw/pitch from the requested look vector and push them into the
  // player, which owns the camera transform — writing camera.rotation directly
  // would be overwritten on the next player update.
  const dir = lookAt.clone().sub(camPos).normalize();
  const yaw = Math.atan2(-dir.x, -dir.z);
  const pitch = Math.asin(THREE.MathUtils.clamp(dir.y, -1, 1));

  /**
   * Pin the player every step.
   *
   * The player owns the camera and is subject to gravity and collision, so a
   * one-shot `setPosition` is undone within a few frames — the first capture
   * attempt asked for a rooftop and got a basement floor. Re-asserting the pose
   * before each step is effectively noclip without needing a flag on the player
   * itself: `setPosition` zeroes velocity, so the only drift per frame is one
   * step of gravity (≈2.7 mm at 60 Hz), which never accumulates.
   *
   * `yaw`/`pitch` are TypeScript-private, which is a compile-time fiction; the
   * fields genuinely exist at runtime and this is the capture harness, not
   * production code, so writing them is the pragmatic choice over widening the
   * player's public surface for a test hook.
   */
  const pinnedFoot = new THREE.Vector3(camPos.x, camPos.y - 1.7, camPos.z);
  const pin = (): void => {
    if (!player) return;
    player.setPosition(pinnedFoot);
    const p = player as unknown as { yaw: number; pitch: number };
    p.yaw = yaw;
    p.pitch = pitch;
  };
  const step = (n: number): void => {
    for (let i = 0; i < n; i++) {
      pin();
      engine.stepOnce(STEP_MS);
    }
  };

  pin();
  engine.camera.position.copy(camPos);
  engine.camera.rotation.order = 'YXZ';
  engine.camera.rotation.set(pitch, yaw, 0);

  if (weapons && preset.weapon) weapons.switchTo(preset.weapon);

  if (!opts.hud) {
    const uiRoot = document.getElementById('ui-root');
    if (uiRoot) uiRoot.style.display = 'none';
  }

  // Settle: let IBL bake, shadows converge, materials compile, AI place itself.
  const settle = Math.max(30, Math.floor(opts.frames * 0.55));
  step(settle);

  if (preset.ads) {
    engine.input.__setAction('ads', true);
    // ADS transitions take up to ~420ms; 45 frames covers the slowest weapon.
    step(45);
  }

  // Settle the world BEFORE firing, not after.
  //
  // This block used to run after the burst, which meant the capture spent
  // another ~95 frames — over a second and a half — sitting still while every
  // transient the burst produced quietly expired. The `combat` preset
  // was reliably photographing the moment after the smoke cleared: the ammo
  // counter had gone down, and there was nothing else to show for it. Muzzle
  // flash, tracers, impact sparks and shell ejection all live for a few frames
  // at most, so the burst has to be the LAST thing that happens.
  const remaining = Math.max(4, opts.frames - settle);
  step(remaining);

  if (preset.fire && preset.fire > 0) {
    engine.input.__setAction('fire', true);
    // Hold the trigger long enough to emit `fire` rounds at the weapon's RPM,
    // then release and let the last flash bloom for a couple of frames so the
    // shot captures a live muzzle flash rather than the gap between rounds.
    // A muzzle flash lives for one or two frames. Releasing the trigger and
    // then stepping on still lands the capture in the gap between rounds, which
    // is why the burst was invisible even once it was moved to the end. Hold the
    // trigger down THROUGH the captured frame and stop one frame after a round
    // leaves the barrel, so the flash, the ejecting case and the first tracer
    // segment are all still alive when the frame is composed.
    const rounds = preset.fire;
    step(rounds * 6 - 1);
    step(1);
    // Released only after the final frame is composed, so the flash survives
    // into the capture but the weapon is not left firing behind it.
    engine.input.__setAction('fire', false);
  }

  // Blank the world only at the very end. The sky has to be visible while the
  // scene settles or the environment probe bakes against a black sky and the
  // hands come out unlit — which would look like a shading bug in the weapon
  // rather than what it is, a broken capture.
  if (opts.isolate) isolateViewmodel(engine, opts.bg);

  // A couple of extra frames so any temporal effect (SMAA history, bloom
  // feedback) has resolved against the final composition.
  step(2);

  const marker = document.createElement('div');
  marker.id = 'noisefield-ready';
  marker.setAttribute('data-shot', preset.name);
  marker.style.cssText = 'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;';
  document.body.appendChild(marker);

  (window as unknown as Record<string, unknown>).__NOISEFIELD_SHOT = {
    preset: preset.name,
    description: preset.description,
    camera: preset.pos,
    stats: {
      frameTimeMs: engine.frameTimeMs,
      calls: engine.renderer.info.render.calls,
      triangles: engine.renderer.info.render.triangles,
    },
  };
}
