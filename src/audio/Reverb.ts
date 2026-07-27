/**
 * NOISEFIELD — procedural impulse responses.
 *
 * A ConvolverNode needs an impulse response, and an impulse response is just an
 * AudioBuffer. There is no reason to ship a .wav for one: an IR has a very
 * well-understood structure that is straightforward to synthesise.
 *
 * ── The model ────────────────────────────────────────────────────────────────
 * Every real room's response has three phases:
 *
 *   1. DIRECT + PRE-DELAY. Silence between the direct sound and the first
 *      reflection. Its length is the single strongest cue for room *size* —
 *      the brain converts it straight into "how far is the nearest wall".
 *   2. EARLY REFLECTIONS. A handful of discrete, still-recognisable echoes off
 *      the nearest surfaces. Their *pattern* is what distinguishes a corridor
 *      from a cathedral. Modelled here as individual bandpassed noise grains
 *      rather than naked impulses, because a real reflection is the source
 *      convolved with a surface, not a click.
 *   3. LATE FIELD. Once the echo density passes ~1000/s the ear stops resolving
 *      individual reflections and hears exponentially-decaying noise. Two
 *      details make it sound real rather than like a noise gate:
 *        • the field *builds* rather than starting at full density, and
 *        • high frequencies decay faster than low ones, because air and soft
 *          furnishings absorb them, so the tail progressively darkens.
 *
 * ── Choosing which reverb to use ─────────────────────────────────────────────
 * The engine estimates enclosure by casting a handful of probe rays from the
 * listener (see `classifyEnvironment`), and crossfades between the four
 * presets. It never hard-switches: walking through a doorway ramps the return
 * gains, which is inaudible, whereas swapping a ConvolverNode's buffer mid-tail
 * is a very audible click.
 */

import { hashString, makeAudioBuffer, Rng, SAMPLE_RATE } from './Synthesis';

export type ReverbId = 'room' | 'hall' | 'outdoor' | 'alley';

export interface ReverbSpec {
  readonly id: ReverbId;
  /** RT60 of the late field, seconds. */
  readonly decay: number;
  /** Silence before the first reflection, seconds. The primary size cue. */
  readonly preDelay: number;
  /** HF damping: the tail's lowpass glides from this cutoff... */
  readonly dampStart: number;
  /** ...to this one over the decay. Air and soft surfaces eat treble first. */
  readonly dampEnd: number;
  /** Fixed low cut, to keep the tail from muddying the sub range. */
  readonly highPass: number;
  /** Discrete early reflections as [timeSec, gain] pairs. */
  readonly earlyTaps: ReadonlyArray<readonly [number, number]>;
  /** Time constant of the late-field density build-up, seconds. */
  readonly buildTime: number;
  /** 0..1 stereo decorrelation of the early taps. */
  readonly width: number;
  /** Output trim, so switching presets does not jump in level. */
  readonly level: number;
}

/**
 * Four spaces, chosen to bracket everything a map can contain.
 *
 * `alley` deserves a note: its taps are deliberately *evenly spaced* (~11ms).
 * Parallel hard walls produce a periodic echo train, and that periodicity is
 * heard as the metallic "flutter" every narrow urban gap has. Randomising those
 * taps would technically be more diffuse and would sound completely wrong.
 */
export const REVERB_SPECS: Readonly<Record<ReverbId, ReverbSpec>> = {
  room: {
    id: 'room',
    decay: 0.42,
    preDelay: 0.004,
    dampStart: 5200,
    dampEnd: 1400,
    highPass: 180,
    earlyTaps: [[0.006, 0.70], [0.009, 0.58], [0.014, 0.46], [0.019, 0.37], [0.026, 0.29], [0.033, 0.22]],
    buildTime: 0.010,
    width: 0.45,
    level: 0.85,
  },
  hall: {
    id: 'hall',
    decay: 2.30,
    preDelay: 0.018,
    dampStart: 4200,
    dampEnd: 700,
    highPass: 90,
    earlyTaps: [[0.017, 0.62], [0.026, 0.52], [0.038, 0.44], [0.051, 0.36], [0.067, 0.28], [0.088, 0.21], [0.110, 0.16]],
    buildTime: 0.045,
    width: 0.80,
    level: 0.90,
  },
  // Not a room at all: a couple of distant hard returns off buildings and
  // terrain, with almost no diffuse field. Long pre-delay, dark, and quiet.
  outdoor: {
    id: 'outdoor',
    decay: 0.90,
    preDelay: 0.035,
    dampStart: 3000,
    dampEnd: 500,
    highPass: 130,
    earlyTaps: [[0.042, 0.55], [0.078, 0.34], [0.135, 0.20], [0.210, 0.11]],
    buildTime: 0.090,
    width: 1.00,
    level: 0.55,
  },
  alley: {
    id: 'alley',
    decay: 1.15,
    preDelay: 0.008,
    dampStart: 6000,
    dampEnd: 1100,
    highPass: 150,
    earlyTaps: [[0.008, 0.66], [0.019, 0.58], [0.030, 0.50], [0.041, 0.43], [0.052, 0.37], [0.063, 0.31], [0.074, 0.26], [0.085, 0.21]],
    buildTime: 0.020,
    width: 0.35,
    level: 0.80,
  },
};

/** −60dB expressed as a natural-log decay constant. */
const LN_1000 = 6.907755;

/**
 * Synthesise a stereo impulse response.
 *
 * Implementation notes:
 *  • One-pole filters (not biquads) are used for damping and the low cut. An IR
 *    is already noise; filter phase response is irrelevant, and one-poles are
 *    ~4× cheaper across the ~250k samples a hall costs.
 *  • The late field is gated by a *velvet-noise* style density ramp: early on
 *    only a fraction of grains pass, and that fraction rises to 1. This is what
 *    produces a natural build instead of a wall of noise starting instantly.
 *  • Channels use independent noise streams (full decorrelation → width) while
 *    the early taps are only jittered by `width`, keeping the image anchored.
 *  • Normalised by RMS rather than peak, then trimmed by `spec.level`, so the
 *    perceived wet level stays constant when the engine crossfades presets.
 *    The ConvolverNode must therefore run with `normalize = false`.
 */
export function renderImpulseResponse(spec: ReverbSpec, sr = SAMPLE_RATE): AudioBuffer {
  const dur = spec.preDelay + spec.decay * 1.12;
  const n = Math.ceil(dur * sr);
  const chans: Float32Array[] = [new Float32Array(n), new Float32Array(n)];

  const preN = Math.floor(spec.preDelay * sr);
  const decayK = LN_1000 / spec.decay;

  for (let c = 0; c < 2; c++) {
    const out = chans[c];
    // Hash the whole id, not just its first character: two presets sharing an
    // initial letter would otherwise render byte-identical impulse responses.
    const rng = new Rng(0x51ed270b ^ hashString(spec.id) ^ ((c + 1) * 0x9e3779b1));

    // ── late field ──────────────────────────────────────────────────────────
    // Time-varying one-pole damping. Recomputing the coefficient every sample
    // would be wasteful; every 32 samples is far below any audible resolution.
    let dampZ = 0;
    let dampA = 0;
    let hpZ = 0;
    const hpA = Math.exp((-2 * Math.PI * spec.highPass) / sr);

    const buildN = Math.max(1, spec.buildTime * sr);

    // Both envelopes are exponentials, so they advance by a constant multiply
    // instead of a `Math.exp` per sample. A hall IR is ~250k samples × 2
    // channels; this is the difference between a stutter and a warm-up slice.
    const decCoef = Math.exp(-decayK / sr);
    let dec = 1;
    const buildCoef = Math.exp(-1 / buildN);
    let buildState = 1; // tracks e^(-k/buildN); density = 1 - buildState

    for (let i = preN; i < n; i++) {
      const k = i - preN;

      if ((k & 31) === 0) {
        const u = Math.min(1, k / sr / spec.decay);
        // Exponential glide in log-frequency, which is how absorption behaves.
        const fc = spec.dampStart * Math.pow(spec.dampEnd / spec.dampStart, u);
        dampA = Math.exp((-2 * Math.PI * Math.min(fc, sr * 0.49)) / sr);
      }

      // Density build: probability a grain exists rises toward 1.
      let x = rng.bi();
      if (rng.next() > 1 - buildState) x *= 0.15;

      x *= dec;
      dec *= decCoef;
      buildState *= buildCoef;

      dampZ = x * (1 - dampA) + dampZ * dampA;
      hpZ = dampZ * (1 - hpA) + hpZ * hpA;
      out[i] += dampZ - hpZ;
    }

    // ── early reflections ───────────────────────────────────────────────────
    // Each tap is a ~2ms bandpassed noise grain: a real reflection carries the
    // surface's colouration, and a bare impulse sounds like a digital tick.
    for (const [tapT, tapG] of spec.earlyTaps) {
      const jitter = 1 + rng.bi() * spec.width * 0.06;
      const start = preN + Math.floor(tapT * jitter * sr);
      const grainN = Math.floor(0.0022 * sr);
      if (start >= n) continue;

      // Surfaces get progressively duller with each bounce (absorption is
      // frequency-dependent and cumulative), so later taps are darker.
      const bright = 1 - Math.min(0.75, tapT * 4);
      const fc = 700 + 5200 * bright * (1 + rng.bi() * 0.25);
      let z = 0;
      const a = Math.exp((-2 * Math.PI * Math.min(fc, sr * 0.49)) / sr);
      const g = tapG * (1 + rng.bi() * spec.width * 0.3);

      for (let k = 0; k < grainN && start + k < n; k++) {
        z = rng.bi() * (1 - a) + z * a;
        out[start + k] += z * Math.exp((-k / grainN) * 3) * g * 3.2;
      }
    }
  }

  // RMS normalisation across the pair keeps the stereo image and gives the
  // engine a predictable wet level to fade between.
  let energy = 0;
  for (const ch of chans) for (let i = 0; i < n; i++) energy += ch[i] * ch[i];
  const rms = Math.sqrt(energy / (n * 2)) || 1;
  const g = (0.055 / rms) * spec.level;
  for (const ch of chans) {
    for (let i = 0; i < n; i++) ch[i] *= g;
    // A short tail fade so a truncated IR never adds a click to every voice.
    const f = Math.min(Math.floor(0.02 * sr), n >> 1);
    for (let i = 0; i < f; i++) ch[n - 1 - i] *= i / f;
  }

  const buf = makeAudioBuffer(2, n, sr);
  buf.getChannelData(0).set(chans[0]);
  buf.getChannelData(1).set(chans[1]);
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Environment classification
// ─────────────────────────────────────────────────────────────────────────────

/** Normalised mix of the four presets. Components sum to 1. */
export interface ReverbBlend {
  room: number;
  hall: number;
  outdoor: number;
  alley: number;
}

/** Result of the listener's enclosure probe. */
export interface EnclosureProbe {
  /** Fraction of horizontal probe rays that hit something, 0..1. */
  enclosure: number;
  /** Mean distance to the nearest surface (probe range when nothing hit). */
  meanDistance: number;
  /** 0 = solid ceiling directly above, 1 = open sky. */
  overhead: number;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Turn a probe into a soft blend of the four presets.
 *
 * DESIGN CHOICE (documented as required): enclosure is estimated by raycasting
 * from the listener rather than by authoring volumes in the level. Rays are
 * free (the physics system already has a BVH), the estimate updates as the
 * player moves through doorways with no authoring burden, and — critically for
 * a procedurally generated map — it needs no hand-placed reverb zones at all.
 * The cost is that it cannot know about a room it cannot see, which for a
 * first-person listener is exactly the right approximation.
 *
 * The mapping is deliberately continuous, never branch-and-switch:
 *   openness      → outdoor
 *   enclosed + open sky + tight  → alley
 *   enclosed + small             → room
 *   enclosed + large             → hall
 */
export function classifyEnvironment(probe: EnclosureProbe): ReverbBlend {
  const open = 1 - Math.min(1, Math.max(0, probe.enclosure));
  const large = smoothstep(5, 15, probe.meanDistance);
  const tight = 1 - smoothstep(3, 9, probe.meanDistance);

  const enclosed = 1 - open;
  const alleyness = enclosed * probe.overhead * tight;
  const rest = Math.max(0, enclosed - alleyness);

  const blend: ReverbBlend = {
    outdoor: open,
    alley: alleyness,
    room: rest * (1 - large),
    hall: rest * large,
  };

  const sum = blend.outdoor + blend.alley + blend.room + blend.hall;
  if (sum > 1e-6) {
    blend.outdoor /= sum;
    blend.alley /= sum;
    blend.room /= sum;
    blend.hall /= sum;
  } else {
    blend.outdoor = 1;
  }
  return blend;
}

/** Default used before the first probe, or when no physics system exists. */
export const DEFAULT_BLEND: ReverbBlend = { room: 0, hall: 0, outdoor: 1, alley: 0 };

/**
 * How "inside" the listener is, 0..1. Drives the indoor/outdoor tail choice and
 * the overall wet send: a shot in the open sends far less to the convolvers
 * than the same shot in a stairwell.
 */
export function indoorness(b: ReverbBlend): number {
  return Math.min(1, b.room + b.hall + b.alley);
}
