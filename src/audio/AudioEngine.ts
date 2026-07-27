/**
 * NOISEFIELD — audio engine.
 *
 * Fully event-driven: the engine subscribes to the game's event bus and never
 * reaches into another subsystem except (optionally) physics, for enclosure and
 * occlusion probes. It owns the Web Audio graph, the voice pool, the mix bus
 * architecture and the spatialisation model.
 *
 * ── INTEGRATION: what the host must do ───────────────────────────────────────
 * Register it (`engine.add(new AudioEngine())`) and then, EVERY FRAME, call:
 *
 *   audio.setListener(camPosition, camForward, camUp, playerVelocity);
 *
 * This is the one thing that is not automatic, and omitting it fails silently:
 * the listener stays at the world origin facing −Z, so every sound is panned
 * and attenuated relative to a point the player is not standing at. `forward`
 * and `up` must both be unit length and mutually orthogonal — note that a
 * three.js camera's `.up` is NOT orthogonal to its forward vector once pitched,
 * so derive them from the camera's world matrix rather than passing `camera.up`.
 *
 * Everything else is optional:
 *   • `resume()` is called automatically on the first pointer/key/touch event,
 *     so audio unlocks itself; call it explicitly if you have your own gesture.
 *   • A system registered as `'physics'` enables reverb-zone estimation and
 *     occlusion. Without it the engine runs with an open-air reverb and no
 *     occlusion, which is a graceful degradation rather than a failure.
 *   • Emit `game:pause` to stop unprompted cues while paused.
 *
 * ── Mix architecture ─────────────────────────────────────────────────────────
 *
 *   voice ─┬─ [air-absorption LP] ─┬─ HRTF panner ─┬─→ weapons ─┐
 *          │                       └─ stereo pan ──┘            │
 *          └─ send ──→ reverbSend ──→ 4× convolver ──→ returns ──┤
 *                                                     world ─────┼→ fatigue LP → master → limiter → out
 *                                                     music ─────┘                ↑
 *                                                     ui ─────────────────────────┘  (bypasses fatigue)
 *
 * Why the UI bus bypasses the ear-fatigue filter: hitmarkers and kill confirms
 * are *information*, not diegetic sound. If a hitmarker gets muffled by the
 * same shot that produced the hit, the player loses the feedback loop that
 * makes shooting feel good. Everything diegetic goes through the filter.
 *
 * ── Distance model ───────────────────────────────────────────────────────────
 * Three independent effects, all of which real audio has and most games skip:
 *   1. ATTENUATION — PannerNode inverse distance model, refDistance 2m.
 *   2. AIR ABSORPTION — HF loss grows with distance, so the cutoff falls
 *      exponentially: 20kHz·e^(−d/55). At 200m that is ~530Hz, i.e. a dull
 *      thud. Paired with a crossfade to a separately synthesised distant
 *      report, because filtering a close-mic'd crack never sounds far away.
 *   3. PROPAGATION DELAY — sound travels at 343 m/s, so a shot 200m away is
 *      heard 0.58s after the muzzle flash. Implemented by scheduling the source
 *      start time, which is sample-accurate and costs nothing.
 *
 * ── Ducking rationale ────────────────────────────────────────────────────────
 * The acoustic reflex genuinely attenuates and lowpasses your hearing for a
 * few hundred ms after a loud impulse. Simulating it is not just realism: it
 * is the single cheapest way to make a weapon feel powerful, because the mix
 * *gets out of the way* of the shot and then swells back. `duck()` drops the
 * world bus, trims the weapons bus less (your own gun should stay present) and
 * sweeps a master lowpass down toward 850Hz, all recovering exponentially.
 */

import { Vector3 } from 'three';
import type {
  EngineContext,
  FrameContext,
  GameEvents,
  IAudio,
  IPhysics,
  SoundId,
  SurfaceType,
  WeaponId,
} from '../core/Types';
import { renderSound, SOUND_SPECS, WARMUP_ORDER } from './Synthesis';
import {
  classifyEnvironment,
  DEFAULT_BLEND,
  indoorness,
  renderImpulseResponse,
  REVERB_SPECS,
  type ReverbBlend,
  type ReverbId,
} from './Reverb';

// ─────────────────────────────────────────────────────────────────────────────
// Tuning constants
// ─────────────────────────────────────────────────────────────────────────────

const SPEED_OF_SOUND = 343;
/** Beyond this many concurrent voices we steal; well under any browser limit. */
const MAX_VOICES = 48;
/**
 * Environmental tails are long (up to 1.75s) and a full-auto weapon fires
 * ~13 rounds a second, so without a cap they alone would exhaust the voice
 * pool and sum into a wall of noise.
 */
const MAX_TAIL_VOICES = 8;
/** Air-absorption e-folding distance in metres. Tuned by ear against real rifle recordings. */
const AIR_ABSORPTION_SCALE = 55;
const MIN_CUTOFF = 350;
/** Below this distance the propagation delay (<15ms) is not worth desyncing VFX. */
const MIN_DELAY_DISTANCE = 5;
const MAX_TRAVEL_DELAY = 1.2;
/** A muzzle closer than this is the local player's own weapon. */
const FIRST_PERSON_DISTANCE = 2.5;
/** Crossfade window from the close report to the synthesised distant report. */
const DISTANT_FADE_START = 35;
const DISTANT_FADE_END = 130;

const REF_DISTANCE = 2;
const ROLLOFF = 0.9;
const MAX_DISTANCE = 500;

const PROBE_RANGE = 22;
const PROBE_INTERVAL = 0.25;
/** Hard cap on occlusion raycasts per frame; the rest reuse the cache. */
const OCCLUSION_RAY_BUDGET = 6;
const OCCLUSION_CACHE_TTL = 0.25;
const OCCLUSION_CELL = 1.5;
/**
 * Probe rays start this far from the listener. We are not given a handle on
 * the player's own collider to exclude via PhysicsFilter, so we simply begin
 * outside its plausible radius; without this the player's own capsule can
 * report every direction as blocked.
 */
const LISTENER_CLEARANCE = 0.6;

/** Per-frame time budget for lazily rendering the sound bank. */
const WARMUP_BUDGET_MS = 2.5;

const WHIZ_RADIUS = 3.4;
const WHIZ_MIN_INTERVAL = 0.045;

const LOW_HEALTH_FRACTION = 0.32;

type BusName = 'weapons' | 'world' | 'ui' | 'music';

/**
 * Static bus trims. Headroom is deliberate: the limiter should be shaping
 * peaks, not permanently gain-reducing, or the whole mix breathes.
 */
const BUS_LEVELS: Readonly<Record<BusName, number>> = {
  weapons: 0.90,
  world: 0.85,
  ui: 0.70,
  music: 0.50,
};

/** Voice-stealing priority. Higher survives. */
const PRIORITY = {
  explosion: 10,
  localWeapon: 9,
  ui: 8,
  remoteWeapon: 7,
  tail: 6,
  whizby: 6,
  impact: 5,
  mechanical: 5,
  player: 4,
  footstep: 3,
  shell: 2,
} as const;

const WEAPON_SOUND: Readonly<Record<WeaponId, SoundId>> = {
  ar_vector: 'fire_ar',
  smg_wasp: 'fire_smg',
  sniper_longbow: 'fire_sniper',
  shotgun_breach: 'fire_shotgun',
  pistol_sidearm: 'fire_pistol',
};

/** [duck amount 0..1, recovery seconds] — bigger charge, longer deafening. */
const WEAPON_DUCK: Readonly<Record<WeaponId, readonly [number, number]>> = {
  ar_vector: [0.50, 0.15],
  smg_wasp: [0.42, 0.11],
  sniper_longbow: [0.90, 0.85],
  shotgun_breach: [0.78, 0.42],
  pistol_sidearm: [0.45, 0.13],
};

/** How much environmental tail each weapon excites. */
const WEAPON_TAIL: Readonly<Record<WeaponId, number>> = {
  ar_vector: 0.55,
  smg_wasp: 0.40,
  sniper_longbow: 1.00,
  shotgun_breach: 0.85,
  pistol_sidearm: 0.42,
};

/** Playback-rate shift applied to the shared distant report per weapon. */
const WEAPON_DISTANT_RATE: Readonly<Record<WeaponId, number>> = {
  ar_vector: 1.00,
  smg_wasp: 1.14,
  sniper_longbow: 0.78,
  shotgun_breach: 0.88,
  pistol_sidearm: 1.06,
};

/**
 * Surface → impact sound, with rate/gain trims for the surfaces that have no
 * dedicated SoundId. Repitching a related material is far more convincing than
 * reusing it untouched: foliage is sand played fast and quiet, rubber is a
 * wood thock dropped an octave and damped.
 */
const IMPACT_MAP: Readonly<Record<SurfaceType, readonly [SoundId, number, number]>> = {
  concrete: ['impact_concrete', 1.0, 1.0],
  metal: ['impact_metal', 1.0, 1.0],
  wood: ['impact_wood', 1.0, 1.0],
  sand: ['impact_sand', 1.0, 1.0],
  dirt: ['impact_sand', 0.85, 0.9],
  glass: ['impact_glass', 1.0, 1.0],
  water: ['impact_water', 1.0, 1.0],
  flesh: ['impact_flesh', 1.0, 1.0],
  foliage: ['impact_sand', 1.45, 0.55],
  fabric: ['impact_sand', 0.70, 0.5],
  rubber: ['impact_wood', 0.62, 0.6],
};

const FOOTSTEP_MAP: Readonly<Record<SurfaceType, readonly [SoundId, number]>> = {
  concrete: ['footstep_concrete', 1.0],
  metal: ['footstep_metal', 1.0],
  wood: ['footstep_wood', 1.0],
  sand: ['footstep_sand', 1.0],
  dirt: ['footstep_sand', 0.88],
  glass: ['footstep_concrete', 1.25],
  water: ['footstep_sand', 0.78],
  flesh: ['footstep_sand', 0.70],
  foliage: ['footstep_sand', 1.30],
  fabric: ['footstep_sand', 0.85],
  rubber: ['footstep_wood', 0.75],
};

const CALIBER_RATE: Readonly<Record<string, number>> = {
  '5.56': 1.0,
  '9mm': 1.18,
  '.338': 0.82,
  '12ga': 0.72,
};

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Structural 3D point. `THREE.Vector3` satisfies this without importing it. */
interface Vec3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(a: number, b: number, x: number): number {
  if (b === a) return x < a ? 0 : 1;
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
}

function distance(a: Vec3, b: Vec3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Matches the PannerNode inverse model, so 2D-panned voices attenuate alike.
 *
 * Note there is deliberately no `maxDistance` clamp: the W3C inverse model
 * clamps distance to [refDistance, ∞) only, and `maxDistance` applies to the
 * `linear` model. Clamping here would make this helper diverge from the
 * PannerNode it is supposed to mirror at long range.
 */
function inverseGain(d: number): number {
  if (!(d > REF_DISTANCE)) return 1; // also catches NaN
  return REF_DISTANCE / (REF_DISTANCE + ROLLOFF * (d - REF_DISTANCE));
}

/** Interpolate in log-frequency, which is how pitch and filtering are heard. */
function lerpHz(from: number, to: number, t: number): number {
  return from * Math.pow(to / from, clamp(t, 0, 1));
}

/**
 * Derive three independent 0..1 values from a shot's seed. Using the event's
 * own seed (rather than Math.random) means the audio variation for a given
 * shot matches whatever the ballistics and VFX systems did with it — the same
 * shot always sounds the same, which matters for replays and for QA.
 *
 * Two details that matter more than they look:
 *  • the seed is scaled before truncation, because a caller is entitled to
 *    hand us a fractional seed and `x ^ y` would truncate every value in
 *    (-1, 1) to the same integer 0;
 *  • *every* output gets the full avalanche finaliser. Returning an
 *    intermediate state as one of the values leaves it a bare Weyl sequence,
 *    and sequential seeds (a shot counter is the obvious implementation) would
 *    then alternate high/low every shot — a fixed two-shot pitch wobble, which
 *    is precisely the audible pattern this whole scheme exists to avoid.
 */
function seedRandoms(seed: number): [number, number, number] {
  let h = (Math.round(seed * 65536) ^ 0x9e3779b9) >>> 0;
  const mix = (): number => {
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return h / 4294967296;
  };
  return [mix(), mix(), mix()];
}

/** Rate-limited console warning, so a per-shot failure cannot flood the log. */
const warnedOnce = new Set<string>();
function warnOnce(key: string, ...args: unknown[]): void {
  if (warnedOnce.has(key)) return;
  warnedOnce.add(key);
  console.warn(...args);
}

function busFor(id: SoundId): BusName {
  if (
    id === 'hitmarker' || id === 'headshot' || id === 'kill' ||
    id === 'lowhealth' || id === 'heartbeat'
  ) {
    return 'ui';
  }
  if (
    id.startsWith('fire_') || id.startsWith('reload_') || id.startsWith('tail_') ||
    id.startsWith('ads_') || id === 'dryfire'
  ) {
    // Tails ride the weapons bus on purpose: they are part of the shot, and
    // ducking them under the very shot that caused them would swallow them.
    return 'weapons';
  }
  return 'world';
}

/**
 * Position an AudioParam-based spatial node, falling back to the deprecated
 * setter on older Safari. `Partial<>` keeps the feature test honest instead of
 * lying to the type system about a non-optional member.
 */
function setPannerPosition(p: PannerNode, x: number, y: number, z: number, when: number): void {
  const t = p as Partial<PannerNode>;
  if (t.positionX && t.positionY && t.positionZ) {
    t.positionX.setValueAtTime(x, when);
    t.positionY.setValueAtTime(y, when);
    t.positionZ.setValueAtTime(z, when);
  } else if (typeof t.setPosition === 'function') {
    t.setPosition(x, y, z);
  }
}

function setPannerOrientation(p: PannerNode, x: number, y: number, z: number, when: number): void {
  const t = p as Partial<PannerNode>;
  if (t.orientationX && t.orientationY && t.orientationZ) {
    t.orientationX.setValueAtTime(x, when);
    t.orientationY.setValueAtTime(y, when);
    t.orientationZ.setValueAtTime(z, when);
  } else if (typeof t.setOrientation === 'function') {
    t.setOrientation(x, y, z);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal state types
// ─────────────────────────────────────────────────────────────────────────────

interface AudioGraph {
  ctx: AudioContext;
  master: GainNode;
  limiter: DynamicsCompressorNode;
  fatigue: BiquadFilterNode;
  buses: Record<BusName, GainNode>;
  reverbSend: GainNode;
  convolvers: Record<ReverbId, ConvolverNode>;
  returns: Record<ReverbId, GainNode>;
}

interface Voice {
  src: AudioBufferSourceNode;
  head: GainNode;
  chain: AudioNode[];
  startedAt: number;
  endsAt: number;
  priority: number;
  /** Estimated audible level, used to pick the least damaging steal. */
  audible: number;
  tag: string;
  dead: boolean;
}

interface PlayOptions {
  gain?: number;
  rate?: number;
  bus?: BusName;
  /** World position; omit for a non-diegetic 2D sound. */
  position?: Vec3;
  /** Emission axis; enables a directional cone (muzzle blasts). */
  orientation?: Vec3;
  /** Extra scheduling offset in seconds, on top of propagation delay. */
  delay?: number;
  /** 0..1 reverb send. */
  send?: number;
  priority?: number;
  /** Force a stereo pan instead of HRTF, in [-1, 1]. */
  pan?: number;
  /** Force a specific waveform variant. */
  variant?: number;
  /** 0..1 used to choose the variant deterministically from an event seed. */
  pick?: number;
  tag?: string;
  /** Skip the speed-of-sound arrival delay. */
  instant?: boolean;
  /** Skip the occlusion raycast (for cheap, high-rate sounds). */
  noOcclude?: boolean;
}

type WarmupTask =
  | { kind: 'sound'; id: SoundId; variant: number }
  | { kind: 'ir'; id: ReverbId };

const REVERB_IDS: readonly ReverbId[] = ['outdoor', 'room', 'alley', 'hall'];

/**
 * Six horizontal probes (plus one cast upward separately) for the enclosure
 * estimate. Evenly spaced at 60°: an asymmetric rosette biases the mean
 * distance toward whichever axes happen to be over-sampled, so a corridor
 * would read differently depending only on which way the player faced.
 */
const PROBE_DIRS: ReadonlyArray<readonly [number, number, number]> = [
  [1, 0, 0],
  [0.5, 0, 0.8660254],
  [-0.5, 0, 0.8660254],
  [-1, 0, 0],
  [-0.5, 0, -0.8660254],
  [0.5, 0, -0.8660254],
];

// ─────────────────────────────────────────────────────────────────────────────
// AudioEngine
// ─────────────────────────────────────────────────────────────────────────────

export class AudioEngine implements IAudio {
  readonly name = 'audio';

  private graph: AudioGraph | null = null;
  private ctxRef: EngineContext | null = null;
  private failed = false;
  private unlockBound = false;
  private resuming: Promise<void> | null = null;

  /** Rendered sound bank. `null` entries have not been synthesised yet. */
  private readonly bank = new Map<SoundId, (AudioBuffer | null)[]>();
  private readonly roundRobin = new Map<SoundId, number>();
  private readonly irs = new Map<ReverbId, AudioBuffer>();
  /** IRs whose turn came round before an AudioContext existed. */
  private readonly deferredIrs: ReverbId[] = [];
  private warmup: WarmupTask[] = [];
  private warmIndex = 0;
  /** Set while the game is paused, so we do not emit unprompted cues. */
  private paused = false;
  private suspendRetry = 0;

  private readonly voices: Voice[] = [];
  /** Stolen voices whose release fade is still running; torn down after `at`. */
  private readonly dying: Array<{ v: Voice; at: number }> = [];
  private readonly unsubscribes: Array<() => void> = [];

  // Listener state, mirrored so we can compute distance/doppler without
  // reading back from the (async, audio-thread) AudioListener.
  private readonly listenerPos = { x: 0, y: 1.7, z: 0 };
  private readonly listenerFwd = { x: 0, y: 0, z: -1 };
  private readonly listenerUp = { x: 0, y: 1, z: 0 };
  private readonly listenerRight = { x: 1, y: 0, z: 0 };
  private readonly listenerVel = { x: 0, y: 0, z: 0 };

  // Ducking envelope, evaluated on the main thread and pushed to two params.
  private duckLevel = 0;
  private duckTau = 0.1;
  private appliedDuck = -1;
  private appliedVolume = -1;

  // Environment
  private blend: ReverbBlend = { ...DEFAULT_BLEND };
  private targetBlend: ReverbBlend = { ...DEFAULT_BLEND };
  /** False until the return gains have been written at least once. */
  private blendApplied = false;
  private probeTimer = 0;
  private readonly probeOrigin = new Vector3();
  private readonly probeDir = new Vector3();

  // Occlusion
  private readonly occlusionCache = new Map<string, { t: number; v: number }>();
  private rayBudget = OCCLUSION_RAY_BUDGET;

  private time = 0;
  private lastWhiz = -1;
  private health = 100;
  private maxHealth = 100;
  private nextHeartbeat = 0;
  private lowHealthAnnounced = false;
  private wasGrounded = true;

  // ── lifecycle ──────────────────────────────────────────────────────────────

  init(ctx: EngineContext): void {
    this.ctxRef = ctx;

    for (const id of WARMUP_ORDER) {
      this.bank.set(id, new Array<AudioBuffer | null>(SOUND_SPECS[id].variants).fill(null));
    }
    this.buildWarmupQueue();
    this.subscribe(ctx);
    this.bindUnlockGestures();
  }

  /**
   * Browsers refuse to start an AudioContext without a user gesture, and
   * constructing one early logs a console warning on every load. So the context
   * is created lazily inside `resume()`; until then `play` is a silent no-op.
   * The sound bank does *not* need a context (see Synthesis.makeAudioBuffer),
   * so warm-up can run during the menu and be finished before the first shot.
   */
  private bindUnlockGestures(): void {
    if (this.unlockBound || typeof window === 'undefined') return;
    this.unlockBound = true;
    const unlock = (): void => {
      void this.resume();
    };
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
    window.addEventListener('touchstart', unlock);
    this.unsubscribes.push(() => {
      window.removeEventListener('pointerdown', unlock);
      window.removeEventListener('keydown', unlock);
      window.removeEventListener('touchstart', unlock);
    });
  }

  get ready(): boolean {
    return !this.failed && this.graph !== null && this.graph.ctx.state === 'running';
  }

  /**
   * Unlock (or re-unlock) audio. Safe to call at any time, from anywhere, as
   * often as you like.
   *
   * The in-flight guard is deliberately NOT a cached promise that clears
   * itself in its own `finally`: when the body completes without ever
   * awaiting — the normal case, because a context created inside a user
   * gesture is already `running` — that `finally` runs *before* the assignment
   * that stores the promise, so the guard would stay set forever and every
   * later call would short-circuit. The engine could then never recover from a
   * browser-initiated suspend (tab hidden, phone call, audio-focus loss) and
   * would be permanently silent. Clearing it in the awaiting caller's `finally`
   * is ordered correctly by construction.
   */
  async resume(): Promise<void> {
    if (this.failed) return;
    if (this.graph && this.graph.ctx.state === 'running') return;
    if (this.resuming) return this.resuming;

    const attempt = this.doResume();
    this.resuming = attempt;
    try {
      await attempt;
    } finally {
      if (this.resuming === attempt) this.resuming = null;
    }
  }

  private async doResume(): Promise<void> {
    try {
      if (!this.graph) this.graph = this.buildGraph();
      const g = this.graph;
      if (!g) return;
      if (g.ctx.state !== 'running') await g.ctx.resume();
      this.attachImpulseResponses();
    } catch (err) {
      // A blocked or unavailable context must never take the game down.
      warnOnce('resume', '[audio] resume failed; running silent', err);
    }
  }

  /**
   * Attach any impulse responses that have been rendered.
   *
   * Each assignment is guarded individually: `ConvolverNode.buffer` throws
   * NotSupportedError if the buffer's sample rate differs from the context's,
   * and one bad preset must not prevent the other three from loading.
   */
  private attachImpulseResponses(): void {
    const g = this.graph;
    if (!g) return;
    for (const id of REVERB_IDS) {
      const buf = this.irs.get(id);
      if (!buf || g.convolvers[id].buffer === buf) continue;
      try {
        g.convolvers[id].buffer = buf;
      } catch (err) {
        warnOnce(`ir:${id}`, `[audio] could not attach the "${id}" impulse response`, err);
      }
    }
  }

  private buildGraph(): AudioGraph | null {
    let ctx: AudioContext | null = null;
    try {
      ctx = new AudioContext({ latencyHint: 'interactive' });

      // Brickwall-ish limiter. Hard knee + 20:1 + a 2ms attack behaves as a
      // safety catch rather than a compressor: it should be transparent until
      // an explosion and three guns land in the same 10ms.
      const limiter = ctx.createDynamicsCompressor();
      limiter.threshold.value = -3;
      limiter.knee.value = 0;
      limiter.ratio.value = 20;
      limiter.attack.value = 0.002;
      limiter.release.value = 0.12;
      limiter.connect(ctx.destination);

      const master = ctx.createGain();
      master.gain.value = this.ctxRef?.settings.masterVolume ?? 0.8;
      master.connect(limiter);

      const fatigue = ctx.createBiquadFilter();
      fatigue.type = 'lowpass';
      fatigue.frequency.value = 20000;
      fatigue.Q.value = 0.4;
      fatigue.connect(master);

      const buses = {
        weapons: ctx.createGain(),
        world: ctx.createGain(),
        ui: ctx.createGain(),
        music: ctx.createGain(),
      } satisfies Record<BusName, GainNode>;

      buses.weapons.gain.value = BUS_LEVELS.weapons;
      buses.world.gain.value = BUS_LEVELS.world;
      buses.ui.gain.value = BUS_LEVELS.ui;
      buses.music.gain.value = BUS_LEVELS.music;

      buses.weapons.connect(fatigue);
      buses.world.connect(fatigue);
      buses.music.connect(fatigue);
      buses.ui.connect(master); // deliberately bypasses ear fatigue

      const reverbSend = ctx.createGain();
      reverbSend.gain.value = 1;

      const convolvers = {} as Record<ReverbId, ConvolverNode>;
      const returns = {} as Record<ReverbId, GainNode>;
      for (const id of REVERB_IDS) {
        const conv = ctx.createConvolver();
        // Our IRs are RMS-normalised and level-trimmed per preset, so the
        // browser's own equal-power normalisation would fight the design.
        conv.normalize = false;
        const ret = ctx.createGain();
        // Seeded from the current blend, NOT from zero: the per-frame update
        // only writes these when the blend *changes*, so starting at zero
        // while the blend is already settled (the usual case — the default
        // blend is fully outdoor, and it is also what we keep when no physics
        // system exists) would leave every convolver muted for the entire
        // session.
        ret.gain.value = this.blend[id];
        reverbSend.connect(conv);
        conv.connect(ret);
        ret.connect(fatigue);
        convolvers[id] = conv;
        returns[id] = ret;
      }

      return { ctx, master, limiter, fatigue, buses, reverbSend, convolvers, returns };
    } catch (err) {
      warnOnce('ctx', '[audio] could not create AudioContext; running silent', err);
      this.failed = true;
      // Do not leak a live context if construction succeeded but wiring threw.
      if (ctx) void ctx.close().catch(() => undefined);
      return null;
    }
  }

  dispose(): void {
    for (const u of this.unsubscribes) u();
    this.unsubscribes.length = 0;
    for (const v of this.voices.slice()) this.retire(v);
    this.voices.length = 0;
    for (const d of this.dying) this.disconnectVoice(d.v);
    this.dying.length = 0;
    if (this.graph) {
      void this.graph.ctx.close().catch(() => undefined);
      this.graph = null;
    }
    this.bank.clear();
    this.irs.clear();
    this.deferredIrs.length = 0;
    this.occlusionCache.clear();
    this.roundRobin.clear();
    // Reset enough state that a fresh init() on this instance would work.
    this.ctxRef = null;
    this.unlockBound = false;
    this.resuming = null;
    this.blendApplied = false;
    this.warmup = [];
    this.warmIndex = 0;
    this.appliedDuck = -1;
    this.appliedVolume = -1;
    this.duckLevel = 0;
  }

  // ── bank warm-up ───────────────────────────────────────────────────────────

  private buildWarmupQueue(): void {
    const q: WarmupTask[] = [];
    for (const id of WARMUP_ORDER) {
      for (let v = 0; v < SOUND_SPECS[id].variants; v++) q.push({ kind: 'sound', id, variant: v });
    }
    // Slot the impulse responses in early — the outdoor one first, since most
    // engagements start in the open and a dry first shot is very noticeable.
    q.splice(Math.min(6, q.length), 0, { kind: 'ir', id: 'outdoor' });
    q.splice(Math.min(20, q.length), 0, { kind: 'ir', id: 'room' });
    q.splice(Math.min(34, q.length), 0, { kind: 'ir', id: 'alley' });
    q.splice(Math.min(48, q.length), 0, { kind: 'ir', id: 'hall' });
    this.warmup = q;
    this.warmIndex = 0;
  }

  /**
   * Render as much of the bank as fits in the frame budget. Each item is a few
   * hundred microseconds to a few milliseconds; the queue always completes the
   * item it started so the worst case is one slightly long frame during load,
   * never a torn buffer.
   */
  private pumpWarmup(): void {
    if (this.warmIndex >= this.warmup.length && this.deferredIrs.length === 0) return;
    const deadline = performance.now() + WARMUP_BUDGET_MS;

    // Impulse responses that could not be rendered before the context existed.
    while (this.deferredIrs.length > 0 && this.graph && performance.now() < deadline) {
      const id = this.deferredIrs.shift();
      if (id) this.renderIr(id);
    }

    while (this.warmIndex < this.warmup.length && performance.now() < deadline) {
      const task = this.warmup[this.warmIndex++];
      try {
        if (task.kind === 'sound') {
          const arr = this.bank.get(task.id);
          if (arr && !arr[task.variant]) arr[task.variant] = renderSound(task.id, task.variant);
        } else if (this.graph) {
          this.renderIr(task.id);
        } else {
          // Cannot render an IR yet — see `renderIr` for why it needs the
          // context. Re-queue it rather than dropping it.
          this.deferredIrs.push(task.id);
        }
      } catch (err) {
        warnOnce(`warm:${task.kind}`, '[audio] synthesis failed for', task, err);
      }
    }
  }

  /**
   * Render one impulse response AT THE CONTEXT'S OWN SAMPLE RATE.
   *
   * This is not an optimisation, it is a correctness requirement: the Web
   * Audio spec mandates that `ConvolverNode.buffer` throw NotSupportedError if
   * the buffer's sample rate differs from the context's. Ordinary sounds are
   * exempt because AudioBufferSourceNode resamples transparently, but a
   * convolver does not — so on the very common 44.1kHz device a bank rendered
   * at a hard-coded 48kHz would lose reverb entirely.
   */
  private renderIr(id: ReverbId): void {
    const g = this.graph;
    if (!g) {
      this.deferredIrs.push(id);
      return;
    }
    try {
      const buf = renderImpulseResponse(REVERB_SPECS[id], g.ctx.sampleRate);
      this.irs.set(id, buf);
      g.convolvers[id].buffer = buf;
    } catch (err) {
      warnOnce(`ir:${id}`, `[audio] could not build the "${id}" impulse response`, err);
    }
  }

  private getBuffer(id: SoundId, variant?: number, pick?: number): AudioBuffer | null {
    const arr = this.bank.get(id);
    if (!arr || arr.length === 0) return null;
    const v = variant === undefined ? this.pickVariant(id, arr.length, pick) : variant % arr.length;
    const existing = arr[v];
    if (existing) return existing;
    try {
      // On-demand fallback for a sound the warm-up has not reached yet. Most
      // entries are 1-3ms; the longest (the outdoor tail, the explosion) are
      // ~15ms and would drop a frame, which is still preferable to silence and
      // happens at most once per variant.
      const buf = renderSound(id, v);
      arr[v] = buf;
      return buf;
    } catch (err) {
      warnOnce(`synth:${id}`, '[audio] on-demand synthesis failed for', id, err);
      return null;
    }
  }

  /**
   * Choose a waveform variant, never repeating the previous one.
   *
   * Selecting uniformly from the variants *excluding* the last one played is
   * strictly better than round-robin (which at a fixed fire rate becomes its
   * own audible N-shot pattern) and than uniform random (which repeats 1/N of
   * the time — and at N=2, a naive "advance by 1 or 2" scheme repeats 35% of
   * the time, which is worse than random).
   *
   * `pick` lets a caller drive the choice from an event seed so a given shot
   * always sounds identical; without it we fall back to Math.random().
   */
  private pickVariant(id: SoundId, count: number, pick?: number): number {
    if (count <= 1) return 0;
    const r = clamp(pick ?? Math.random(), 0, 0.999999);
    const prev = this.roundRobin.get(id);
    let v: number;
    if (prev === undefined) {
      v = Math.min(count - 1, Math.floor(r * count));
    } else {
      const k = Math.min(count - 2, Math.floor(r * (count - 1)));
      v = k >= prev ? k + 1 : k;
    }
    this.roundRobin.set(id, v);
    return v;
  }

  // ── per-frame ──────────────────────────────────────────────────────────────

  update(ctx: EngineContext, frame: FrameContext): void {
    this.time += frame.rawDt;
    this.rayBudget = OCCLUSION_RAY_BUDGET;

    this.pumpWarmup();

    const g = this.graph;
    if (!g) return;

    // A context can be suspended by the browser at any time (tab hidden, an
    // incoming call, audio-focus loss). Retry periodically so the game does
    // not go permanently silent after an interruption.
    if (g.ctx.state === 'suspended') {
      this.suspendRetry -= frame.rawDt;
      if (this.suspendRetry <= 0) {
        this.suspendRetry = 1;
        void this.resume();
      }
    }

    const now = g.ctx.currentTime;
    const dt = Math.max(1e-4, frame.rawDt);

    // Duck envelope. Evaluated here rather than with scheduled ramps so that
    // overlapping ducks compose (max, not last-wins) and so a paused game does
    // not leave a half-finished automation hanging on the param.
    if (this.duckLevel > 0.0005) {
      this.duckLevel *= Math.exp(-dt / this.duckTau);
      if (this.duckLevel < 0.0005) this.duckLevel = 0;
    }
    if (Math.abs(this.duckLevel - this.appliedDuck) > 0.002) {
      this.appliedDuck = this.duckLevel;
      const d = this.duckLevel;
      g.buses.world.gain.setTargetAtTime(BUS_LEVELS.world * (1 - 0.75 * d), now, 0.02);
      g.buses.weapons.gain.setTargetAtTime(BUS_LEVELS.weapons * (1 - 0.30 * d), now, 0.02);
      g.fatigue.frequency.setTargetAtTime(lerpHz(20000, 850, d), now, 0.025);
    }

    const vol = ctx.settings.masterVolume;
    if (vol !== this.appliedVolume) {
      this.appliedVolume = vol;
      g.master.gain.setTargetAtTime(clamp(vol, 0, 1), now, 0.03);
    }

    // Environment estimate at 4Hz, crossfaded continuously.
    this.probeTimer -= dt;
    if (this.probeTimer <= 0) {
      this.probeTimer = PROBE_INTERVAL;
      this.probeEnvironment(ctx);
    }
    const k = 1 - Math.exp(-dt / 0.35);
    let changed = !this.blendApplied;
    for (const id of REVERB_IDS) {
      const cur = this.blend[id];
      const next = cur + (this.targetBlend[id] - cur) * k;
      if (Math.abs(next - cur) > 1e-4) changed = true;
      this.blend[id] = Number.isFinite(next) ? next : 0;
    }
    if (changed) {
      this.blendApplied = true;
      for (const id of REVERB_IDS) {
        g.returns[id].gain.setTargetAtTime(this.blend[id], now, 0.08);
      }
    }

    if (!this.paused) this.updateHealthCues();
    this.pruneVoices(now);
  }

  private pruneVoices(now: number): void {
    for (let i = this.voices.length - 1; i >= 0; i--) {
      const v = this.voices[i];
      if (v.dead || now > v.endsAt + 0.25) this.retire(v);
    }
    for (let i = this.dying.length - 1; i >= 0; i--) {
      if (now >= this.dying[i].at) {
        this.disconnectVoice(this.dying[i].v);
        this.dying.splice(i, 1);
      }
    }
  }

  // ── IAudio surface ─────────────────────────────────────────────────────────

  play(id: SoundId, gain = 1, rate = 1): void {
    this.spawn(id, { gain, rate, priority: busFor(id) === 'ui' ? PRIORITY.ui : PRIORITY.player });
  }

  play3D(id: SoundId, position: Vector3, gain = 1, rate = 1): void {
    this.spawn(id, { gain, rate, position, priority: PRIORITY.impact });
  }

  setListener(position: Vector3, forward: Vector3, up: Vector3, velocity: Vector3): void {
    this.listenerPos.x = position.x;
    this.listenerPos.y = position.y;
    this.listenerPos.z = position.z;
    this.listenerFwd.x = forward.x;
    this.listenerFwd.y = forward.y;
    this.listenerFwd.z = forward.z;
    this.listenerUp.x = up.x;
    this.listenerUp.y = up.y;
    this.listenerUp.z = up.z;
    this.listenerVel.x = velocity.x;
    this.listenerVel.y = velocity.y;
    this.listenerVel.z = velocity.z;

    // right = forward × up, used for the stereo-pan fallback path.
    this.listenerRight.x = forward.y * up.z - forward.z * up.y;
    this.listenerRight.y = forward.z * up.x - forward.x * up.z;
    this.listenerRight.z = forward.x * up.y - forward.y * up.x;

    const g = this.graph;
    if (!g) return;
    const l = g.ctx.listener;
    const now = g.ctx.currentTime;
    // Smoothed rather than stepped: at 60fps a hard set on positionX produces
    // audible zipper noise on fast turns, especially with HRTF.
    const tau = 0.02;
    const t = l as Partial<AudioListener>;
    if (t.positionX && t.forwardX && t.upX && t.positionY && t.positionZ && t.forwardY && t.forwardZ && t.upY && t.upZ) {
      t.positionX.setTargetAtTime(position.x, now, tau);
      t.positionY.setTargetAtTime(position.y, now, tau);
      t.positionZ.setTargetAtTime(position.z, now, tau);
      t.forwardX.setTargetAtTime(forward.x, now, tau);
      t.forwardY.setTargetAtTime(forward.y, now, tau);
      t.forwardZ.setTargetAtTime(forward.z, now, tau);
      t.upX.setTargetAtTime(up.x, now, tau);
      t.upY.setTargetAtTime(up.y, now, tau);
      t.upZ.setTargetAtTime(up.z, now, tau);
    } else if (typeof t.setPosition === 'function' && typeof t.setOrientation === 'function') {
      t.setPosition(position.x, position.y, position.z);
      t.setOrientation(forward.x, forward.y, forward.z, up.x, up.y, up.z);
    }
  }

  /**
   * Duck the mix and muffle the world for `duration` seconds.
   * Composition is by maximum, not replacement: a sniper shot followed 50ms
   * later by a pistol must not shorten the sniper's deafening.
   */
  duck(amount: number, duration: number): void {
    const a = clamp(amount, 0, 1);
    if (!(a > 0)) return;
    // tau = duration/3 puts the envelope ~95% recovered at `duration`.
    const tau = Math.max(0.02, duration / 3);

    if (a >= this.duckLevel || this.duckLevel < 0.02) {
      // A deeper event takes charge of the envelope, and so does any event
      // arriving once the previous duck has essentially finished. Assigning
      // (not maxing) the time constant here is what stops it ratcheting: if
      // tau could only ever grow, one grenade early in a match would leave
      // every subsequent rifle round muffling the world for two seconds.
      this.duckLevel = a;
      this.duckTau = tau;
    } else if (tau > this.duckTau) {
      // Shallower but longer-lasting: keep the deeper level already in flight
      // and just extend the recovery.
      this.duckTau = tau;
    }
  }

  // ── voice spawning ─────────────────────────────────────────────────────────

  private spawn(id: SoundId, o: PlayOptions): void {
    const g = this.graph;
    if (!g || this.failed || g.ctx.state !== 'running') return;

    const spec = SOUND_SPECS[id];
    if (!spec) return;

    const ctx = g.ctx;
    const now = ctx.currentTime;

    let gain = o.gain ?? 1;
    let rate = clamp(o.rate ?? 1, 0.25, 4);
    let delay = Math.max(0, o.delay ?? 0);
    let cutoff = 20000;
    let dist = 0;

    const pos = o.position;
    if (pos) {
      dist = distance(pos, this.listenerPos);

      // Propagation delay. Sample-accurate and free — no DelayNode needed.
      if (!o.instant && dist > MIN_DELAY_DISTANCE) {
        delay += Math.min(dist / SPEED_OF_SOUND, MAX_TRAVEL_DELAY);
      }

      // Air absorption.
      cutoff = clamp(20000 * Math.exp(-dist / AIR_ABSORPTION_SCALE), MIN_CUTOFF, 20000);

      if (!o.noOcclude) {
        const occ = this.occlusionAt(pos, dist);
        if (occ > 0.01) {
          cutoff = Math.min(cutoff, lerpHz(20000, 620, occ));
          gain *= 1 - 0.6 * occ;
        }
      }

      rate *= this.dopplerFor(pos, dist);
    }

    const audible = gain * (pos ? inverseGain(dist) : 1);
    // Below this the voice would be inaudible under any plausible mix; do not
    // burn a slot on it. This is what keeps a 200-round firefight affordable.
    if (audible < 0.0009) return;

    // Resolve the buffer BEFORE stealing a voice: if this fails we would
    // otherwise have silenced an existing sound to make room for nothing.
    const buf = this.getBuffer(id, o.variant, o.pick);
    if (!buf) return;

    const priority = o.priority ?? PRIORITY.impact;
    if (!this.ensureVoiceRoom(priority, audible)) return;

    try {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate;

      const head = ctx.createGain();
      const chain: AudioNode[] = [head];
      src.connect(head);
      let tail: AudioNode = head;

      if (cutoff < 18000) {
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        lp.frequency.value = cutoff;
        lp.Q.value = 0.7;
        tail.connect(lp);
        tail = lp;
        chain.push(lp);
      }

      // The reverb send is taken PRE-panner and, critically, PRE-distance-gain:
      // a room is excited by energy, not by direction, so feeding it the mono,
      // distance-filtered signal is both more correct and cheaper than
      // convolving an already-spatialised pair.
      //
      // Tapping here also keeps the two spatial paths consistent. The HRTF
      // path applies distance attenuation inside the PannerNode (i.e. after
      // this tap), so the stereo path must likewise apply its distance gain in
      // its own branch below rather than on `head` — otherwise the send would
      // be attenuated by the distance law twice on one path and once on the
      // other, and the documented wet/dry relationship would run backwards.
      head.gain.value = gain;
      if ((o.send ?? 0) > 0) {
        const send = ctx.createGain();
        // sqrt() of the distance law: the direct sound falls off faster than
        // the reverberant field, so the wet/dry ratio rises with distance.
        // That ratio is the strongest distance cue the ear has indoors.
        send.gain.value = (o.send ?? 0) * Math.sqrt(pos ? inverseGain(dist) : 1);
        tail.connect(send);
        send.connect(g.reverbSend);
        chain.push(send);
      }

      let spatial: AudioNode;
      if (pos && spec.channels === 1 && o.pan === undefined) {
        const p = ctx.createPanner();
        p.panningModel = 'HRTF';
        p.distanceModel = 'inverse';
        p.refDistance = REF_DISTANCE;
        p.rolloffFactor = ROLLOFF;
        p.maxDistance = MAX_DISTANCE;
        setPannerPosition(p, pos.x, pos.y, pos.z, now);
        if (o.orientation) {
          // Muzzle blasts are strongly directional: standing behind a rifle is
          // a very different sound from standing in front of it.
          p.coneInnerAngle = 80;
          p.coneOuterAngle = 260;
          p.coneOuterGain = 0.45;
          setPannerOrientation(p, o.orientation.x, o.orientation.y, o.orientation.z, now);
        }
        tail.connect(p);
        spatial = p;
        chain.push(p);
      } else {
        if (pos) {
          // Mirror of the PannerNode's inverse distance model for the stereo
          // path, applied after the send tap so both paths match.
          const dg = ctx.createGain();
          dg.gain.value = inverseGain(dist);
          tail.connect(dg);
          tail = dg;
          chain.push(dg);
        }
        const sp = ctx.createStereoPanner();
        sp.pan.value = clamp(o.pan ?? (pos ? this.lateralPan(pos, dist) : 0), -1, 1);
        tail.connect(sp);
        spatial = sp;
        chain.push(sp);
      }

      spatial.connect(g.buses[o.bus ?? busFor(id)]);

      const when = now + delay;
      src.start(when);

      const voice: Voice = {
        src,
        head,
        chain,
        startedAt: when,
        endsAt: when + buf.duration / rate,
        priority,
        audible,
        tag: o.tag ?? '',
        dead: false,
      };
      src.onended = () => {
        voice.dead = true;
      };
      this.voices.push(voice);
    } catch (err) {
      // Throttled: this path can be reached at gunfire rates.
      warnOnce('spawn', '[audio] failed to spawn', id, err);
    }
  }

  /**
   * Make room for one more voice.
   * Steal the lowest-priority voice, tie-broken by how quiet it is and then by
   * age. If the weakest existing voice still outranks the newcomer, the
   * newcomer is dropped instead — an explosion must never be evicted by the
   * twelfth footstep.
   */
  private ensureVoiceRoom(priority: number, audible: number): boolean {
    if (this.voices.length < MAX_VOICES) return true;

    // Ages must be measured on the SAME clock the voices were stamped with.
    // `startedAt` comes from AudioContext.currentTime, whose epoch is the
    // context's creation (i.e. the first user gesture), while `this.time` runs
    // from engine start. Mixing them makes the age term arbitrarily large and
    // silently erodes the priority ordering it is only meant to break ties in.
    const now = this.graph ? this.graph.ctx.currentTime : 0;

    let worst: Voice | null = null;
    let worstScore = Infinity;
    for (const v of this.voices) {
      if (v.dead) {
        this.retire(v);
        return true;
      }
      // Age is a tie-breaker only, so it is bounded: it must never be able to
      // outweigh a whole priority level.
      const age = Math.min(4, Math.max(0, now - v.startedAt));
      const score = v.priority * 10 + v.audible * 3 - age * 0.5;
      if (score < worstScore) {
        worstScore = score;
        worst = v;
      }
    }
    if (!worst) return true;
    if (worstScore > priority * 10 + audible * 3) return false;

    this.stopVoice(worst);
    return true;
  }

  /**
   * Free a voice's slot immediately, but let its 4ms release fade actually
   * happen before the nodes are torn down. Disconnecting in the same tick (as
   * a synchronous retire() would) discards the scheduled ramp entirely and
   * reintroduces exactly the click the fade exists to prevent.
   */
  private stopVoice(v: Voice): void {
    const i = this.voices.indexOf(v);
    if (i >= 0) this.voices.splice(i, 1);
    v.dead = true;

    const g = this.graph;
    if (!g) {
      this.disconnectVoice(v);
      return;
    }
    try {
      const now = g.ctx.currentTime;
      v.head.gain.cancelScheduledValues(now);
      v.head.gain.setValueAtTime(v.head.gain.value, now);
      v.head.gain.linearRampToValueAtTime(0, now + 0.004);
      v.src.stop(now + 0.006);
      // Torn down by pruneVoices once the fade has run. Deliberately not
      // relying on `onended`, which is not guaranteed for a source stopped
      // before its scheduled start time.
      this.dying.push({ v, at: now + 0.05 });
    } catch {
      this.disconnectVoice(v);
    }
  }

  private retire(v: Voice): void {
    const i = this.voices.indexOf(v);
    if (i >= 0) this.voices.splice(i, 1);
    this.disconnectVoice(v);
  }

  private disconnectVoice(v: Voice): void {
    try {
      v.src.onended = null;
      v.src.disconnect();
      for (const n of v.chain) n.disconnect();
    } catch {
      // Disconnecting an already-torn-down node is harmless.
    }
  }

  private cancelTag(tag: string): void {
    for (const v of this.voices.slice()) {
      if (v.tag === tag) this.stopVoice(v);
    }
  }

  private countTag(tag: string): number {
    let n = 0;
    for (const v of this.voices) if (v.tag === tag && !v.dead) n++;
    return n;
  }

  // ── spatial helpers ────────────────────────────────────────────────────────

  /** Signed lateral position of a source, for the non-HRTF stereo path. */
  private lateralPan(pos: Vec3, dist: number): number {
    if (dist < 0.2) return 0;
    const inv = 1 / dist;
    const dx = (pos.x - this.listenerPos.x) * inv;
    const dy = (pos.y - this.listenerPos.y) * inv;
    const dz = (pos.z - this.listenerPos.z) * inv;
    const r = this.listenerRight;
    // Softened, because a hard-panned environmental layer feels detached.
    return clamp((dx * r.x + dy * r.y + dz * r.z) * 0.7, -1, 1);
  }

  /**
   * Doppler from listener motion only.
   *
   * The Web Audio doppler API was removed from the spec, and these are all
   * one-shots from (mostly) static emitters, so source velocity is unknown.
   * What remains — pitch shifting as *you* sprint past a sound — is subtle but
   * adds real motion. The genuinely fast case, a round passing your head, has
   * its doppler sweep baked into the whizby waveform instead, which is both
   * cheaper and more controllable.
   */
  private dopplerFor(pos: Vec3, dist: number): number {
    if (dist < 0.5) return 1;
    const inv = 1 / dist;
    const dx = (pos.x - this.listenerPos.x) * inv;
    const dy = (pos.y - this.listenerPos.y) * inv;
    const dz = (pos.z - this.listenerPos.z) * inv;
    const radial = this.listenerVel.x * dx + this.listenerVel.y * dy + this.listenerVel.z * dz;
    return clamp((SPEED_OF_SOUND + radial) / SPEED_OF_SOUND, 0.94, 1.06);
  }

  /**
   * 0 = clear line of sight, 1 = fully blocked.
   *
   * Results are cached per 1.5m world cell for 250ms and *slewed* toward the
   * new value rather than snapped. That slew is what stops full-auto fire
   * across a doorway edge from chattering between filtered and unfiltered,
   * which is the classic occlusion popping artefact.
   */
  private occlusionAt(pos: Vec3, dist: number): number {
    if (dist < 2.5) return 0;
    const physics = this.ctxRef?.get<IPhysics>('physics');
    if (!physics) return 0;

    const key = `${Math.round(pos.x / OCCLUSION_CELL)},${Math.round(pos.y / OCCLUSION_CELL)},${Math.round(pos.z / OCCLUSION_CELL)}`;
    const cached = this.occlusionCache.get(key);
    if (cached && this.time - cached.t < OCCLUSION_CACHE_TTL) return cached.v;
    if (this.rayBudget <= 0) return cached ? cached.v : 0;
    this.rayBudget--;

    let raw = 0;
    try {
      const inv = 1 / dist;
      const dx = (pos.x - this.listenerPos.x) * inv;
      const dy = (pos.y - this.listenerPos.y) * inv;
      const dz = (pos.z - this.listenerPos.z) * inv;
      // Start the ray clear of the listener's own collider. We have no handle
      // on the player's object to put in a PhysicsFilter, and a capsule the
      // ray starts inside would report every sound in the world as occluded.
      this.probeOrigin.set(
        this.listenerPos.x + dx * LISTENER_CLEARANCE,
        this.listenerPos.y + dy * LISTENER_CLEARANCE,
        this.listenerPos.z + dz * LISTENER_CLEARANCE,
      );
      this.probeDir.set(dx, dy, dz);
      const hit = physics.raycast(this.probeOrigin, this.probeDir, dist - LISTENER_CLEARANCE - 0.4);
      raw = hit ? 0.85 : 0;
    } catch {
      raw = 0;
    }

    const v = cached ? cached.v + (raw - cached.v) * 0.5 : raw;
    // Evict by age, never wholesale. A `clear()` would throw away precisely
    // the slew state that stops occlusion popping, periodically reintroducing
    // the artefact this cache exists to suppress.
    if (this.occlusionCache.size > 192) {
      const cutoff = this.time - OCCLUSION_CACHE_TTL * 4;
      for (const [k, entry] of this.occlusionCache) {
        if (entry.t < cutoff) this.occlusionCache.delete(k);
      }
      // Still oversized (a very fast-moving listener): drop the oldest half.
      if (this.occlusionCache.size > 192) {
        let drop = this.occlusionCache.size >> 1;
        for (const k of this.occlusionCache.keys()) {
          if (drop-- <= 0) break;
          this.occlusionCache.delete(k);
        }
      }
    }
    this.occlusionCache.set(key, { t: this.time, v });
    return v;
  }

  /** Estimate the listener's enclosure and pick a reverb blend. */
  private probeEnvironment(ctx: EngineContext): void {
    const physics = ctx.get<IPhysics>('physics');
    if (!physics) {
      this.targetBlend = { ...DEFAULT_BLEND };
      return;
    }
    try {
      let hits = 0;
      let sum = 0;
      this.probeOrigin.set(this.listenerPos.x, this.listenerPos.y, this.listenerPos.z);
      for (const d of PROBE_DIRS) {
        this.probeDir.set(d[0], d[1], d[2]);
        const hit = physics.raycast(this.probeOrigin, this.probeDir, PROBE_RANGE);
        // Anything inside the listener's own clearance is the player's collider,
        // not a wall; counting it would report a broom cupboard everywhere.
        if (hit && Number.isFinite(hit.distance) && hit.distance > LISTENER_CLEARANCE) {
          hits++;
          sum += hit.distance;
        } else {
          sum += PROBE_RANGE;
        }
      }
      this.probeDir.set(0, 1, 0);
      const up = physics.raycast(this.probeOrigin, this.probeDir, PROBE_RANGE);
      const roofed = up !== null && Number.isFinite(up.distance) && up.distance > LISTENER_CLEARANCE;

      const blend = classifyEnvironment({
        enclosure: hits / PROBE_DIRS.length,
        meanDistance: sum / PROBE_DIRS.length,
        overhead: roofed ? 0 : 1,
      });
      // These values are written straight into AudioParams; a single NaN would
      // throw out of the engine's unguarded per-frame update loop.
      this.targetBlend = REVERB_IDS.every((id) => Number.isFinite(blend[id]))
        ? blend
        : { ...DEFAULT_BLEND };
    } catch {
      this.targetBlend = { ...DEFAULT_BLEND };
    }
  }

  // ── event wiring ───────────────────────────────────────────────────────────

  private subscribe(ctx: EngineContext): void {
    const b = ctx.bus;
    const u = this.unsubscribes;

    u.push(b.on('weapon:fire', (e) => this.onFire(e)));
    u.push(b.on('weapon:dryfire', () => {
      this.spawn('dryfire', { gain: 0.8, rate: 0.97 + Math.random() * 0.06, priority: PRIORITY.mechanical });
    }));
    u.push(b.on('weapon:reload:start', (e) => this.onReloadStart(e)));
    u.push(b.on('weapon:reload:end', () => this.onReloadEnd()));
    u.push(b.on('weapon:switch', (e) => this.onSwitch(e)));
    u.push(b.on('weapon:ads', (e) => {
      this.spawn(e.active ? 'ads_in' : 'ads_out', {
        gain: 0.7, rate: 0.96 + Math.random() * 0.08, priority: PRIORITY.mechanical,
      });
    }));

    u.push(b.on('impact', (e) => this.onImpact(e)));
    u.push(b.on('shell:eject', (e) => this.onShell(e)));

    u.push(b.on('player:footstep', (e) => this.onFootstep(e)));
    u.push(b.on('player:land', (e) => this.onLand(e)));
    u.push(b.on('player:slide:start', () => {
      this.spawn('slide', { gain: 0.7, rate: 0.95 + Math.random() * 0.1, priority: PRIORITY.player, tag: 'slide' });
    }));
    u.push(b.on('player:slide:end', () => this.cancelTag('slide')));

    u.push(b.on('damage:taken', (e) => this.onDamageTaken(e)));
    u.push(b.on('damage:dealt', (e) => {
      this.spawn(e.headshot ? 'headshot' : 'hitmarker', {
        gain: e.headshot ? 0.85 : 0.7, priority: PRIORITY.ui,
      });
    }));
    u.push(b.on('entity:killed', (e) => {
      if (e.byPlayer) this.spawn('kill', { gain: 0.8, priority: PRIORITY.ui });
    }));

    u.push(b.on('explosion', (e) => this.onExplosion(e)));

    // Additive, read-only subscriptions to two further typed events:
    //  • 'tracer' drives the whiz-by, which by definition can only be triggered
    //    by a round actually passing the camera;
    //  • 'player:state' keeps the health cues accurate through regeneration,
    //    which 'damage:taken' alone cannot observe.
    u.push(b.on('tracer', (e) => this.onTracer(e)));
    u.push(b.on('player:state', (e) => {
      this.health = e.health;
      this.maxHealth = Math.max(1, e.maxHealth);
      // There is no 'player:jump' event, so the takeoff is inferred from the
      // grounded flag. Without this the `jump` sound would be synthesised and
      // warmed but unreachable, and leaving the ground would be silent while
      // landing again is not.
      if (this.wasGrounded && !e.grounded && e.velocity.y > 1.5) {
        this.spawn('jump', { gain: 0.5, rate: 0.94 + Math.random() * 0.12, priority: PRIORITY.player, send: 0.2 });
      }
      this.wasGrounded = e.grounded;
    }));
    // Silence unprompted cues (the low-health heartbeat especially) while the
    // game is paused. The engine still calls update() when paused, and it does
    // so with a live rawDt, so nothing else would stop them.
    u.push(b.on('game:pause', (e) => {
      this.paused = e.paused;
      if (e.paused) this.cancelTag('slide');
    }));
  }

  private onFire(e: GameEvents['weapon:fire']): void {
    const id = WEAPON_SOUND[e.weapon];
    const dist = distance(e.origin, this.listenerPos);
    const firstPerson = dist < FIRST_PERSON_DISTANCE;

    // Per-shot variation, derived from the shot's own seed. Together with the
    // four rendered variants this is what stops full-auto from sounding like a
    // looped sample: pitch, level and waveform all move every trigger pull,
    // and `pick` guarantees the waveform is never the same two shots running.
    const [ja, jb, jc] = seedRandoms(e.seed);
    const rate = 1 + (ja * 2 - 1) * 0.04;
    const gvar = 1 + (jb * 2 - 1) * 0.10;

    const ind = indoorness(this.blend);
    const wet = 0.28 + 0.5 * ind;

    if (firstPerson) {
      this.spawn(id, {
        gain: 0.95 * gvar,
        rate,
        bus: 'weapons',
        // Slightly right of centre: the weapon is shouldered, not in your skull.
        pan: 0.12,
        priority: PRIORITY.localWeapon,
        send: wet,
        instant: true,
        pick: jc,
      });
    } else {
      const w = smoothstep(DISTANT_FADE_START, DISTANT_FADE_END, dist);
      if (w < 0.995) {
        this.spawn(id, {
          gain: gvar * (1 - w),
          rate,
          position: e.origin,
          orientation: e.direction,
          bus: 'weapons',
          priority: PRIORITY.remoteWeapon,
          send: wet,
          pick: jc,
        });
      }
      if (w > 0.005) {
        // The crossfade partner: a report with no shock front at all.
        this.spawn('fire_distant', {
          gain: gvar * w * 1.25,
          rate: WEAPON_DISTANT_RATE[e.weapon] * rate,
          position: e.origin,
          bus: 'weapons',
          priority: PRIORITY.remoteWeapon,
          send: wet * 0.6,
          pick: jc,
        });
      }
    }

    // Environmental tail. Indoors it slaps back almost immediately; outdoors
    // there is a real gap before the landscape answers.
    //
    // Two constraints here that are easy to get wrong:
    //  • the tail must NOT be `instant` when it carries a world position. It is
    //    the *echo* of the report, so it has to make the same trip through the
    //    air; scheduling it instantly puts the echo of a distant shot up to
    //    several hundred ms BEFORE the shot itself.
    //  • a 1.75s tail retriggered at 13 rounds/second would stack 20+ deep and
    //    pile up ~14dB over the shots it is supporting, as well as eating the
    //    whole voice pool. Real tails do not sum like that — the room is
    //    already excited — so gain is divided down by how many are in flight
    //    and hard-capped.
    const tails = this.countTag('tail');
    if (tails < MAX_TAIL_VOICES) {
      const crowd = 1 / (1 + 0.4 * tails);
      const tailBase = WEAPON_TAIL[e.weapon] * crowd * (firstPerson ? 1 : Math.max(0.25, 1 - dist / 160));
      if (ind > 0.08) {
        this.spawn('tail_indoor', {
          gain: tailBase * ind * 0.75,
          delay: 0.014,
          position: firstPerson ? undefined : e.origin,
          instant: firstPerson,
          priority: PRIORITY.tail,
          noOcclude: true,
          tag: 'tail',
        });
      }
      if (ind < 0.94) {
        this.spawn('tail_outdoor', {
          gain: tailBase * (1 - ind) * 0.62,
          delay: 0.045,
          position: firstPerson ? undefined : e.origin,
          instant: firstPerson,
          priority: PRIORITY.tail,
          noOcclude: true,
          tag: 'tail',
        });
      }
    }

    const [amount, dur] = WEAPON_DUCK[e.weapon];
    // Remote gunfire deafens you far less than your own muzzle 40cm away.
    const proximity = firstPerson ? 1 : clamp(1 - dist / 45, 0.08, 0.7);
    this.duck(amount * proximity, dur);
  }

  /**
   * Reloads are scheduled as three offsets inside the animation rather than
   * driven by further events, so the audio lines up with the viewmodel without
   * the weapon system having to emit sub-events. A tactical reload keeps a
   * round chambered, so the charging handle must NOT play — getting that wrong
   * is the kind of detail players notice without being able to name it.
   */
  private onReloadStart(e: GameEvents['weapon:reload:start']): void {
    const d = Math.max(0.4, e.duration);
    this.spawn('reload_magout', {
      gain: 0.75, rate: 0.97 + Math.random() * 0.06,
      delay: d * 0.12, priority: PRIORITY.mechanical, tag: 'reload',
    });
    this.spawn('reload_magin', {
      gain: 0.8, rate: 0.97 + Math.random() * 0.06,
      delay: d * 0.52, priority: PRIORITY.mechanical, tag: 'reload',
    });
    if (!e.tactical) {
      this.spawn('reload_charge', {
        gain: 0.8, rate: 0.97 + Math.random() * 0.06,
        delay: d * 0.80, priority: PRIORITY.mechanical, tag: 'reload',
      });
    }
  }

  private onReloadEnd(): void {
    // The scheduled voices are already in flight and correctly timed; the end
    // event only clears the cancellation tag so a later switch cannot kill a
    // sound belonging to a completed reload.
    for (const v of this.voices) {
      if (v.tag === 'reload') v.tag = '';
    }
  }

  private onSwitch(e: GameEvents['weapon:switch']): void {
    // Tagged 'reload' as well, so rapid switching cancels the previous
    // switch's own sounds rather than layering them.
    this.cancelTag('reload');
    if (e.from !== null) {
      this.spawn('reload_magout', { gain: 0.4, rate: 0.82, priority: PRIORITY.mechanical, tag: 'reload' });
    }
    this.spawn('reload_charge', {
      gain: 0.45, rate: 1.1, delay: 0.14, priority: PRIORITY.mechanical, tag: 'reload',
    });
  }

  private onImpact(e: GameEvents['impact']): void {
    const entry = IMPACT_MAP[e.surface] ?? IMPACT_MAP.concrete;
    const [id, baseRate, baseGain] = entry;
    // Energy drives level and, more subtly, pitch: a low-energy hit excites the
    // higher, shorter modes of a material, so a small rate lift reads as "weak".
    const energy = clamp(e.energy, 0, 1);
    this.spawn(id, {
      position: e.position,
      gain: baseGain * (0.32 + 0.68 * energy),
      rate: baseRate * (1.08 - 0.16 * energy) * (0.94 + Math.random() * 0.12),
      priority: PRIORITY.impact,
      send: 0.3,
    });
  }

  private onShell(e: GameEvents['shell:eject']): void {
    // Ballistic time-of-flight for the case, from a ~0.95m ejection height.
    const vy = e.velocity.y;
    const fall = clamp((vy + Math.sqrt(Math.max(0, vy * vy + 2 * 9.81 * 0.95))) / 9.81, 0.22, 0.95);
    this.spawn('shell_drop', {
      position: e.position,
      delay: fall,
      gain: 0.42,
      rate: (CALIBER_RATE[e.caliber] ?? 1) * (0.93 + Math.random() * 0.14),
      priority: PRIORITY.shell,
      noOcclude: true,
      send: 0.35,
    });
  }

  private onFootstep(e: GameEvents['player:footstep']): void {
    const [id, rate] = FOOTSTEP_MAP[e.surface] ?? FOOTSTEP_MAP.concrete;
    const own = distance(e.position, this.listenerPos) < FIRST_PERSON_DISTANCE;
    this.spawn(id, {
      position: own ? undefined : e.position,
      gain: (e.sprinting ? 0.62 : 0.42) * (own ? 0.75 : 1),
      rate: rate * (0.92 + Math.random() * 0.16) * (e.sprinting ? 1.04 : 1),
      priority: PRIORITY.footstep,
      send: 0.25,
      noOcclude: own,
    });
    // Stepping in water layers a splash over the step rather than replacing it.
    if (e.surface === 'water') {
      this.spawn('impact_water', {
        position: own ? undefined : e.position,
        gain: 0.3, rate: 0.8 + Math.random() * 0.2,
        priority: PRIORITY.footstep, send: 0.25, noOcclude: own,
      });
    }
  }

  private onLand(e: GameEvents['player:land']): void {
    const v = Math.abs(e.velocity);
    const hard = clamp((v - 2) / 9, 0, 1);
    if (v < 1.2) return;
    this.spawn('land', {
      gain: 0.35 + 0.55 * hard,
      rate: 1.08 - 0.18 * hard,
      priority: PRIORITY.player,
      send: 0.25,
    });
    if (hard > 0.6) this.duck(0.2 * hard, 0.18);
  }

  /**
   * CONVENTION ASSUMED: `damage:taken.direction` points FROM the player TOWARD
   * the attacker, matching what a HUD damage indicator needs to draw an arrow.
   * Types.ts does not document it, and the sibling `ImpactEvent.incoming` uses
   * the opposite sense ("direction of travel of the projectile"), so if the
   * emitter follows *that* convention instead, negate the three components
   * below — otherwise every damage cue is heard 180° wrong, which is worse
   * than no directional cue at all.
   */
  private onDamageTaken(e: GameEvents['damage:taken']): void {
    this.health = e.health;

    // Two metres out along the bearing to the attacker, so HRTF gives a real
    // directional cue — "where did that come from" is gameplay, not flavour.
    const p = {
      x: this.listenerPos.x + e.direction.x * 2,
      y: this.listenerPos.y + e.direction.y * 2,
      z: this.listenerPos.z + e.direction.z * 2,
    };
    this.spawn('damage', {
      position: p,
      gain: clamp(0.5 + e.amount / 60, 0.5, 1),
      rate: 0.94 + Math.random() * 0.12,
      // Routed to the UI bus, not `world`: this event also triggers a duck,
      // and the world bus is exactly what that duck attenuates. Leaving the
      // cue there would have it fight the ducking it caused, at the one moment
      // the player most needs to hear which direction they are being shot from.
      bus: 'ui',
      priority: PRIORITY.ui,
      instant: true,
      noOcclude: true,
      send: 0.15,
    });
    // A hit briefly muffles hearing, matching the "whoomp" baked into the cue.
    this.duck(clamp(0.25 + e.amount / 90, 0.25, 0.7), 0.42);
  }

  private onExplosion(e: GameEvents['explosion']): void {
    const dist = distance(e.position, this.listenerPos);
    const power = clamp(e.power, 0.2, 4);
    this.spawn('explosion', {
      position: e.position,
      gain: clamp(0.85 * power, 0.2, 1),
      rate: clamp(1.1 - 0.12 * power, 0.7, 1.2),
      priority: PRIORITY.explosion,
      send: 0.75,
    });
    // Deafening scales with proximity and falls off over the blast radius ×4.
    // Note there is no floor on the amount: a blast on the far side of the map
    // must not muffle your hearing at all, and a constant term would mean the
    // heaviest duck in the game fires no matter how distant the explosion.
    const prox = clamp(1 - dist / Math.max(8, e.radius * 4), 0, 1);
    const amount = clamp(0.95 * prox * power, 0, 1);
    if (amount > 0.02) this.duck(amount, 0.5 + 1.5 * prox);
  }

  /**
   * A round passing close by. Enormously effective for tension, and almost
   * free: find the closest approach of the tracer segment to the listener, and
   * if it is inside a few metres, schedule the whiz-by at the moment the round
   * actually reaches that point.
   */
  private onTracer(e: GameEvents['tracer']): void {
    if (this.time - this.lastWhiz < WHIZ_MIN_INTERVAL) return;

    const ax = e.to.x - e.from.x;
    const ay = e.to.y - e.from.y;
    const az = e.to.z - e.from.z;
    const len2 = ax * ax + ay * ay + az * az;
    if (len2 < 1) return;

    const lp = this.listenerPos;
    const t = ((lp.x - e.from.x) * ax + (lp.y - e.from.y) * ay + (lp.z - e.from.z) * az) / len2;
    // Only count rounds that genuinely pass us, not ones that start or end here.
    if (t <= 0.03 || t >= 0.99) return;

    const cx = e.from.x + ax * t;
    const cy = e.from.y + ay * t;
    const cz = e.from.z + az * t;
    const miss = distance({ x: cx, y: cy, z: cz }, lp);
    if (miss > WHIZ_RADIUS) return;
    // Our own outgoing rounds start at our muzzle; they must not whiz past us.
    if (distance(e.from, lp) < 4) return;

    // Set only once we are actually committing to a voice, so a whiz-by that
    // gets culled as inaudible does not consume the rate-limit window.
    this.lastWhiz = this.time;
    const flight = (Math.sqrt(len2) * t) / Math.max(80, e.speed);
    this.spawn('whizby', {
      position: { x: cx, y: cy, z: cz },
      gain: 0.75 * (1 - miss / WHIZ_RADIUS),
      rate: 0.9 + Math.random() * 0.2,
      delay: Math.min(flight, 0.6),
      priority: PRIORITY.whizby,
      instant: true,
      noOcclude: true,
      send: 0.2,
    });
  }

  /**
   * Low-health cues. The heartbeat accelerates as health falls, which is a
   * pure gameplay-readability device: the player learns the tempo and can
   * judge how close to death they are without reading the HUD.
   */
  private updateHealthCues(): void {
    const frac = this.health / this.maxHealth;
    if (frac > LOW_HEALTH_FRACTION || this.health <= 0) {
      this.lowHealthAnnounced = false;
      return;
    }
    if (!this.lowHealthAnnounced) {
      this.lowHealthAnnounced = true;
      this.spawn('lowhealth', { gain: 0.5, priority: PRIORITY.ui });
      this.nextHeartbeat = this.time + 0.35;
    }
    if (this.time >= this.nextHeartbeat) {
      const urgency = 1 - frac / LOW_HEALTH_FRACTION;
      this.spawn('heartbeat', { gain: 0.35 + 0.3 * urgency, rate: 0.95 + 0.15 * urgency, priority: PRIORITY.ui });
      this.nextHeartbeat = this.time + (0.95 - 0.35 * urgency);
    }
  }
}
