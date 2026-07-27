/**
 * NOISEFIELD — procedural sound synthesis.
 *
 * ZERO external assets. Every sample the game will ever play is computed here
 * by DSP written from scratch: filtered noise, phase-accumulated sweeps, modal
 * (resonator bank) synthesis and grain clouds, glued together with percussive
 * envelopes and waveshaping.
 *
 * ── Why this file exists ─────────────────────────────────────────────────────
 * Recorded samples are a black box: you cannot ask a .wav to be 4% sharper, or
 * to have a slightly hotter transient on this particular shot. Synthesis gives
 * us a *parametric* sound bank, which is exactly what kills the single biggest
 * tell of cheap game audio — the "machine-gun clone" effect, where full-auto
 * fire is one identical sample retriggered 10× a second. Here every weapon
 * renders N genuinely different variants (each parameter jittered by its own
 * RNG stream), and the engine additionally randomises pitch and gain per shot.
 *
 * ── House style for a "sound" ────────────────────────────────────────────────
 * Real-world percussive sounds decompose into four recurring layers, and almost
 * every renderer below is some mix of them:
 *
 *   1. TRANSIENT — sub-5ms broadband spike. Carries "how hard" and "how close".
 *      Perceptually this is what makes something sound loud and *present*; it
 *      is also the layer that survives distance the worst (see air absorption).
 *   2. BODY      — filtered noise through a resonant band with an exponential
 *      decay. The band centre is the object's "voice" (a 5.56 muzzle blast
 *      resonates ~1.2kHz, a .338 nearer 500Hz).
 *   3. TONAL     — decaying sinusoids. Either a low sweep (the chest "thump" of
 *      a pressure wave) or a bank of *inharmonic* partials (modal synthesis),
 *      which is how struck metal, glass and wood actually ring.
 *   4. MECHANICAL/DETAIL — a small offset layer: a bolt click, a shell tink, a
 *      grit scatter. Almost inaudible alone; its absence reads as "synthetic".
 *
 * ── Cost ─────────────────────────────────────────────────────────────────────
 * Renders are plain Float32Array maths, no Web Audio graph, so they are
 * synchronous, deterministic and need no AudioContext (important: we must be
 * able to build the bank *before* the browser's autoplay gesture unlocks
 * audio).
 *
 * Measured on the full bank (95 buffers, 48kHz): ~305ms total, ~3.2ms mean,
 * ~18ms worst case (the 1.75s stereo outdoor tail). The engine amortises this
 * across frames with a time budget, so the bank is warm roughly 1.5s after
 * boot and no single frame does more than one heavy item. Two techniques keep
 * it there:
 *   • the hot loops contain no `Math.exp`/`Math.pow`/`Math.sin` — see the
 *     PERFORMANCE NOTE above the layer generators;
 *   • the two longest layers (explosion rumble, outdoor tail rumble) are
 *     generated at half rate and linearly upsampled. Both are lowpassed below
 *     1kHz, so the octave the half-rate buffer cannot represent is empty
 *     anyway: inaudible, and twice as cheap.
 */

import type { SoundId, WeaponId } from '../core/Types';

/**
 * Render rate for the whole bank. Fixed rather than taken from the AudioContext
 * because the bank is built before any context exists. If the device runs at
 * 44.1kHz the browser resamples on playback, which for short percussive content
 * is inaudible and costs nothing measurable.
 */
export const SAMPLE_RATE = 48000;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic RNG
// ─────────────────────────────────────────────────────────────────────────────

/**
 * mulberry32. Seeded so a given (sound, variant) pair always renders the same
 * buffer — variation must be *designed*, not accidental, or QA can never
 * reproduce "that shot sounded wrong".
 */
export class Rng {
  private s: number;

  constructor(seed: number) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }

  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [a, b). */
  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  /** Uniform in [-1, 1). */
  bi(): number {
    return this.next() * 2 - 1;
  }

  /** Multiplicative jitter: `jit(0.08)` returns a factor in [0.92, 1.08]. */
  jit(amount: number): number {
    return 1 + this.bi() * amount;
  }

  chance(p: number): boolean {
    return this.next() < p;
  }
}

/** FNV-1a over a string, used to derive a stable seed from a SoundId. */
export function hashString(s: string): number {
  return hash(s);
}

function hash(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioBuffer allocation without a context
// ─────────────────────────────────────────────────────────────────────────────

let bufferFactory: BaseAudioContext | null = null;

/**
 * Allocate an AudioBuffer with no AudioContext.
 *
 * The standalone `AudioBuffer` constructor (Chrome 55+, FF 53+, Safari 14.1+)
 * lets us build the entire bank during the loading screen, before the user has
 * gestured and before we are allowed to start an AudioContext. The fallback is
 * an OfflineAudioContext, which is also gesture-free but is a heavier object.
 */
export function makeAudioBuffer(channels: number, length: number, sampleRate = SAMPLE_RATE): AudioBuffer {
  const len = Math.max(1, Math.floor(length));
  try {
    return new AudioBuffer({ numberOfChannels: channels, length: len, sampleRate });
  } catch {
    if (!bufferFactory) bufferFactory = new OfflineAudioContext(1, 1, sampleRate);
    return bufferFactory.createBuffer(channels, len, sampleRate);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DSP primitives
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Topology-preserving-transform state variable filter (Zavalishin).
 *
 * Chosen over an RBJ biquad because almost every layer here *sweeps* its
 * cutoff (a muzzle blast's formant drops as gases vent; a ricochet whine
 * descends; air absorption darkens a tail over time). A TPT SVF stays stable
 * and keeps its tuning right up to Nyquist under fast modulation, where a
 * direct-form biquad with recomputed coefficients blows up or zipper-noises.
 */
class SVF {
  private g = 0;
  private k = 1;
  private a1 = 0;
  private a2 = 0;
  private a3 = 0;
  private ic1 = 0;
  private ic2 = 0;
  lp = 0;
  bp = 0;
  hp = 0;

  set(sr: number, fc: number, q: number): void {
    const f = clamp(fc, 10, sr * 0.49);
    this.g = Math.tan((Math.PI * f) / sr);
    this.k = 1 / Math.max(q, 0.05);
    this.a1 = 1 / (1 + this.g * (this.g + this.k));
    this.a2 = this.g * this.a1;
    this.a3 = this.g * this.a2;
  }

  /** Unity-peak bandpass scale factor (the raw band output is 1/Q down). */
  get bandScale(): number {
    return this.k;
  }

  process(x: number): void {
    const v3 = x - this.ic2;
    const v1 = this.a1 * this.ic1 + this.a2 * v3;
    const v2 = this.ic2 + this.a2 * this.ic1 + this.a3 * v3;
    this.ic1 = 2 * v1 - this.ic1;
    this.ic2 = 2 * v2 - this.ic2;
    this.lp = v2;
    this.bp = v1;
    this.hp = x - this.k * v1 - v2;
  }
}

/** One-pole lowpass. Used for cheap smoothing (gate slew, HF damping). */
class OnePole {
  private a = 0;
  private z = 0;

  set(sr: number, fc: number): void {
    this.a = Math.exp((-2 * Math.PI * clamp(fc, 0.1, sr * 0.49)) / sr);
  }

  lp(x: number): number {
    this.z = x * (1 - this.a) + this.z * this.a;
    return this.z;
  }
}

/**
 * Paul Kellet's economy pink-noise filter. Pink (1/f) noise is the right
 * spectrum for anything involving turbulent air or granular material — white
 * noise sounds thin and "hissy" as a gunshot body, pink sounds like pressure.
 */
class Pink {
  private b0 = 0;
  private b1 = 0;
  private b2 = 0;

  next(w: number): number {
    this.b0 = 0.99765 * this.b0 + w * 0.099046;
    this.b1 = 0.963 * this.b1 + w * 0.2965164;
    this.b2 = 0.57 * this.b2 + w * 0.1050186;
    return (this.b0 + this.b1 + this.b2 + w * 0.1848) * 0.36;
  }
}

/**
 * Asymmetric-free tanh soft clipper. Two jobs:
 *   • loudness — squashes the peak so normalisation can raise everything else,
 *     which is why a saturated shot sounds twice as big at the same peak level;
 *   • character — generates odd harmonics, the "grit" of a real transducer.
 * Normalised by tanh(drive) so drive does not also change the level.
 */
function softClip(x: number, drive: number): number {
  if (drive <= 0) return x; // tanh(0)/tanh(0) would be NaN
  return Math.tanh(x * drive) / Math.tanh(drive);
}

/** Gaussian bell, for fly-by style "approach and recede" amplitude curves. */
function bell(t: number, centre: number, width: number): number {
  if (width <= 0) return t === centre ? 1 : 0;
  const u = (t - centre) / width;
  return Math.exp(-u * u);
}

/** Remove any DC offset a sweep or asymmetric clip may have introduced. */
function dcBlock(d: Float32Array): void {
  let mean = 0;
  for (let i = 0; i < d.length; i++) mean += d[i];
  mean /= d.length || 1;
  if (Math.abs(mean) < 1e-6) return;
  for (let i = 0; i < d.length; i++) d[i] -= mean;
}

/** Scale so the absolute peak lands exactly on `peak`. */
function normalize(d: Float32Array, peak: number): void {
  let m = 0;
  for (let i = 0; i < d.length; i++) {
    const a = Math.abs(d[i]);
    if (a > m) m = a;
  }
  if (m < 1e-9) return;
  const g = peak / m;
  for (let i = 0; i < d.length; i++) d[i] *= g;
}

/**
 * Micro-fades at both ends. The head fade is deliberately tiny (0.1ms) so a
 * gunshot transient stays razor sharp; the tail fade is long enough that a
 * truncated decay never clicks.
 */
function fade(d: Float32Array, sr: number, inMs = 0.1, outMs = 3): void {
  const n = d.length;
  const a = Math.min(Math.floor((inMs * sr) / 1000), n >> 1);
  const b = Math.min(Math.floor((outMs * sr) / 1000), n >> 1);
  for (let i = 0; i < a; i++) d[i] *= i / a;
  for (let i = 0; i < b; i++) d[n - 1 - i] *= i / b;
}

/** Copy a mono working buffer into a fresh AudioBuffer. */
function toBuffer(data: Float32Array, sr: number): AudioBuffer {
  const buf = makeAudioBuffer(1, data.length, sr);
  buf.getChannelData(0).set(data);
  return buf;
}

/** Copy two working buffers into a fresh stereo AudioBuffer. */
function toStereoBuffer(l: Float32Array, r: Float32Array, sr: number): AudioBuffer {
  const buf = makeAudioBuffer(2, l.length, sr);
  buf.getChannelData(0).set(l);
  buf.getChannelData(1).set(r);
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer generators
// ─────────────────────────────────────────────────────────────────────────────

/** A band of noise with an envelope and (optionally) a swept filter. */
interface NoiseLayer {
  /** Start offset in seconds. */
  t0?: number;
  /** How long to run the generator. */
  dur: number;
  mode: 'lp' | 'bp' | 'hp';
  /** Filter cutoff/centre at t0. */
  f0: number;
  /** Filter cutoff/centre at t0 + sweepDur (defaults to f0 = no sweep). */
  f1?: number;
  sweepDur?: number;
  q?: number;
  atk?: number;
  tau: number;
  gain: number;
  /** Use 1/f noise instead of white — right for air, grit and rumble. */
  pink?: boolean;
  /**
   * Probability that a ~grainMs window is audible. Below 1 this turns a smooth
   * hiss into a *scatter* of discrete particles: concrete grit, sand crunch,
   * wood splinter. Gating per-sample would just be quieter noise, so the gate
   * is held for a grain and slewed to avoid clicks.
   */
  density?: number;
  grainMs?: number;
  /** Slow amplitude modulation, for friction "roll" and explosion rumble. */
  amDepth?: number;
  amRate?: number;
}

const FILTER_BLOCK = 16;

/**
 * PERFORMANCE NOTE for every generator below.
 *
 * These loops run tens of millions of iterations to build the bank, so the
 * transcendentals are hoisted out of them entirely using two identities:
 *
 *   • an exponential envelope is a constant multiply — e^(−(t+dt)/τ) =
 *     e^(−t/τ)·e^(−dt/τ) — so a decay costs one multiply per sample, not a
 *     `Math.exp`;
 *   • an exponential frequency sweep is likewise a constant multiply on the
 *     frequency, so no `Math.pow` per sample either.
 *
 * Sinusoids use a coupled-form ("magic circle") oscillator: two multiplies and
 * two adds per sample instead of a `Math.sin`. It is unconditionally stable and
 * its slight amplitude error is removed by the final normalise anyway.
 *
 * This is what keeps the heaviest single render (a 3.1s explosion) inside one
 * frame's warm-up budget instead of stalling the main thread.
 */

function addNoise(out: Float32Array, sr: number, rng: Rng, L: NoiseLayer): void {
  const start = Math.floor((L.t0 ?? 0) * sr);
  const n = Math.min(out.length, start + Math.ceil(L.dur * sr));
  if (n <= start) return;

  const svf = new SVF();
  const q = L.q ?? 0.7;
  const f0 = L.f0;
  const f1 = L.f1 ?? L.f0;
  const sweepDur = L.sweepDur ?? L.dur;
  const atk = L.atk ?? 0;
  const pink = L.pink ? new Pink() : null;

  const density = L.density ?? 1;
  const grainLen = Math.max(1, Math.floor(((L.grainMs ?? 1.6) * sr) / 1000));
  const gateSlew = new OnePole();
  gateSlew.set(sr, 900);
  let gate = 1;

  const amDepth = L.amDepth ?? 0;
  const amRate = L.amRate ?? 0;
  let am = 1;

  // Envelope recursions (see PERFORMANCE NOTE).
  const atkCoef = atk > 0 ? Math.exp(-1 / (atk * sr)) : 0;
  let atkState = atk > 0 ? 1 : 0; // tracks e^(-t/atk)
  const decCoef = Math.exp(-1 / (L.tau * sr));
  let dec = 1; // tracks e^(-t/tau)

  // Exponential (musical) filter sweep as a per-sample multiply. Guarded
  // against f0 === 0, where the ratio would be Infinity and 0·Infinity = NaN
  // would poison the whole layer.
  const sweeping = f1 !== f0 && f0 > 0 && f1 > 0;
  const sweepN = Math.max(1, sweepDur * sr);
  const fStep = sweeping ? Math.pow(f1 / f0, 1 / sweepN) : 1;
  let f = f0;

  svf.set(sr, f0, q);
  const bandScale = svf.bandScale;

  for (let i = start; i < n; i++) {
    const k = i - start;

    if (sweeping) {
      if (k < sweepN) f *= fStep;
      if ((k & (FILTER_BLOCK - 1)) === 0) svf.set(sr, f, q);
    }

    if (density < 1 && k % grainLen === 0) {
      gate = rng.chance(density) ? 1 : 0;
    }
    const g = density < 1 ? gateSlew.lp(gate) : 1;

    // The LFO is far slower than the block rate, so block-rate is transparent.
    if (amDepth > 0 && (k & 15) === 0) {
      am = 1 - amDepth + amDepth * (0.5 + 0.5 * Math.sin((2 * Math.PI * amRate * k) / sr));
    }

    const w = rng.bi();
    svf.process(pink ? pink.next(w) : w);

    let y: number;
    if (L.mode === 'lp') y = svf.lp;
    else if (L.mode === 'hp') y = svf.hp;
    else y = svf.bp * bandScale;

    out[i] += y * (1 - atkState) * dec * g * am * L.gain;

    atkState *= atkCoef;
    dec *= decCoef;
  }
}

/** One partial of a modal (resonator-bank) model. */
interface Mode {
  f: number;
  a: number;
  tau: number;
  /**
   * Fractional downward pitch drift over the decay. Struck metal and wood both
   * droop slightly as the amplitude falls (tension/nonlinearity), and adding a
   * couple of percent is the difference between "a bell" and "a real object".
   */
  droop?: number;
}

/**
 * Modal synthesis: a struck rigid body rings at a set of *inharmonic* partials
 * whose ratios come from its geometry (a free-free bar is 1 : 2.76 : 5.40 : 8.93,
 * nothing like a harmonic series). Summing decaying sines at those ratios is
 * the cheapest convincing model of metal, glass and wood there is.
 */
function addModal(out: Float32Array, sr: number, t0: number, modes: readonly Mode[], gain: number): void {
  const start = Math.floor(t0 * sr);
  for (const m of modes) {
    const n = Math.min(out.length, start + Math.ceil(m.tau * 6 * sr));
    const s0 = Math.max(0, start);
    if (n <= s0) continue;

    const droop = m.droop ?? 0;
    const decCoef = Math.exp(-1 / (m.tau * sr));
    let dec = 1;
    const amp = m.a * gain;

    // Coupled-form oscillator (see PERFORMANCE NOTE).
    let f = m.f;
    let eps = 2 * Math.sin((Math.PI * f) / sr);
    // The oscillator's own peak amplitude is sec(pi*f/sr), i.e. it runs hot by
    // a factor that grows with frequency (+76% at 15kHz, +176% at 18kHz).
    // Uncompensated, the top partial of a modal bank drowns the fundamental
    // and then steals all the headroom at the normalise step, which quietly
    // hollows out every metallic sound. One cosine per retune fixes it exactly.
    let scale = Math.cos((Math.PI * f) / sr);
    let sx = 1;
    let sy = 0;

    for (let i = s0; i < n; i++) {
      // Droop follows the amplitude envelope; retuning at block rate is well
      // below the ~1Hz at which a pitch glide becomes audible as steps.
      if (droop > 0 && ((i - s0) & 31) === 0) {
        f = m.f * (1 - droop * (1 - dec));
        eps = 2 * Math.sin((Math.PI * f) / sr);
        scale = Math.cos((Math.PI * f) / sr);
      }
      sx -= eps * sy;
      sy += eps * sx;
      out[i] += sy * dec * amp * scale;
      dec *= decCoef;
    }
  }
}

/**
 * Phase-accumulated sine with an exponential frequency sweep. This is the
 * "thump" generator: a pressure wave arriving at the ear is a rapid rise then
 * a fall in pressure, which reads as a pitch dropping fast into the sub range.
 */
function addSweep(
  out: Float32Array,
  sr: number,
  t0: number,
  f0: number,
  f1: number,
  sweepDur: number,
  atk: number,
  tau: number,
  gain: number,
  triangle = false,
): void {
  const start = Math.floor(t0 * sr);
  const n = Math.min(out.length, start + Math.ceil(tau * 7 * sr));
  const s0 = Math.max(0, start);
  if (n <= s0) return;

  const sweepN = Math.max(1, sweepDur * sr);
  // Guarded against a zero endpoint (see addNoise).
  const fStep = f1 === f0 || f0 <= 0 || f1 <= 0 ? 1 : Math.pow(f1 / f0, 1 / sweepN);
  let f = f0;

  const atkCoef = atk > 0 ? Math.exp(-1 / (atk * sr)) : 0;
  let atkState = atk > 0 ? 1 : 0;
  const decCoef = Math.exp(-1 / (tau * sr));
  let dec = 1;

  if (triangle) {
    // A triangle adds a little odd-harmonic edge so a sub thump still reads on
    // laptop speakers that cannot reproduce the fundamental at all. Generated
    // from a normalised phase directly — asin(sin(x)) is the same waveform for
    // two transcendentals a sample.
    let p = 0;
    for (let i = s0; i < n; i++) {
      if (i - s0 < sweepN) f *= fStep;
      p += f / sr;
      if (p >= 1) p -= 1;
      const tri = 1 - 4 * Math.abs(((p + 0.25) % 1) - 0.5);
      out[i] += tri * (1 - atkState) * dec * gain;
      atkState *= atkCoef;
      dec *= decCoef;
    }
  } else {
    let eps = 2 * Math.sin((Math.PI * f) / sr);
    let scale = Math.cos((Math.PI * f) / sr); // see addModal
    let sx = 1;
    let sy = 0;
    for (let i = s0; i < n; i++) {
      const k = i - s0;
      if (k < sweepN) {
        f *= fStep;
        if ((k & 7) === 0) {
          eps = 2 * Math.sin((Math.PI * f) / sr);
          scale = Math.cos((Math.PI * f) / sr);
        }
      }
      sx -= eps * sy;
      sy += eps * sx;
      out[i] += sy * (1 - atkState) * dec * gain * scale;
      atkState *= atkCoef;
      dec *= decCoef;
    }
  }
}

/** A cloud of short decaying grains — glass shards, debris, brass tinkle. */
interface GrainCloud {
  count: number;
  tStart: number;
  tEnd: number;
  fLo: number;
  fHi: number;
  tauLo: number;
  tauHi: number;
  gain: number;
  /** >1 clusters grains toward tStart (u = rand^bias); shatter is front-loaded. */
  bias?: number;
  /** Band-passed noise bursts instead of sines — debris rather than shards. */
  noise?: boolean;
  q?: number;
}

function addGrains(out: Float32Array, sr: number, rng: Rng, c: GrainCloud): void {
  const bias = c.bias ?? 1;
  for (let k = 0; k < c.count; k++) {
    const u = Math.pow(rng.next(), bias);
    const t0 = c.tStart + (c.tEnd - c.tStart) * u;
    const start = Math.floor(t0 * sr);
    if (start >= out.length) continue;

    const f = rng.range(c.fLo, c.fHi);
    const tau = rng.range(c.tauLo, c.tauHi);
    // Equal-loudness-ish tilt: high grains are perceptually louder, so scale
    // them down or a shatter turns into an ice-pick.
    const amp = c.gain * (1 / Math.sqrt(f / c.fLo)) * rng.range(0.5, 1);
    const n = Math.min(out.length, start + Math.ceil(tau * 6 * sr));

    const decCoef = Math.exp(-1 / (tau * sr));
    let dec = 1;

    if (c.noise) {
      const svf = new SVF();
      svf.set(sr, f, c.q ?? 4);
      const bs = svf.bandScale;
      for (let i = start; i < n; i++) {
        svf.process(rng.bi());
        out[i] += svf.bp * bs * dec * amp;
        dec *= decCoef;
      }
    } else {
      // Random start phase, so a cloud of grains does not sum into a click.
      const p0 = rng.range(0, 6.283);
      const eps = 2 * Math.sin((Math.PI * f) / sr);
      let sx = Math.cos(p0);
      let sy = Math.sin(p0);
      // Exact amplitude compensation. The coupled form conserves
      // x² + y² − eps·x·y, so its peak is sqrt(that) / cos(pi*f/sr); with a
      // random start phase the invariant is phase-dependent too, and glass
      // shards run up to 9.5kHz where the error is large.
      const inv = Math.sqrt(Math.max(1e-9, sx * sx + sy * sy - eps * sx * sy));
      const scale = (Math.cos((Math.PI * f) / sr) / inv) * amp;
      for (let i = start; i < n; i++) {
        sx -= eps * sy;
        sy += eps * sx;
        out[i] += sy * dec * scale;
        dec *= decCoef;
      }
    }
  }
}

/** Upsample a half-rate working buffer into `out` with linear interpolation. */
function addUpsampled(out: Float32Array, half: Float32Array, gain: number): void {
  const n = Math.min(out.length, half.length * 2 - 2);
  for (let i = 0; i < n; i++) {
    const h = i * 0.5;
    const i0 = h | 0;
    const fr = h - i0;
    out[i] += (half[i0] * (1 - fr) + half[i0 + 1] * fr) * gain;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GUNSHOTS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A muzzle report is four simultaneous physical events, and the profile below
 * is one knob set per event:
 *
 *   crack — the shock front. Sub-2ms, effectively broadband, band-limited only
 *           by the barrel geometry. This is the layer the ear uses to judge
 *           distance, because air absorption destroys it first.
 *   body  — expanding propellant gas: turbulent, so pink noise, resonating in
 *           the bore + muzzle device around a characteristic frequency that
 *           *falls* as the gas vents and the cavity effectively lengthens.
 *   thump — the low-frequency pressure pulse you feel in the chest, modelled as
 *           a fast downward sine sweep into the sub range.
 *   mech  — the action cycling a few tens of ms later: a small inharmonic
 *           metallic ring. Inaudible on its own, unmistakable when missing.
 */
interface GunProfile {
  duration: number;
  bodyTau: number;
  bodyFreq: number;
  bodyQ: number;
  lowFreq: number;
  thumpFrom: number;
  thumpTo: number;
  thumpTau: number;
  thumpGain: number;
  crackTau: number;
  crackFreq: number;
  crackGain: number;
  mechDelay: number;
  mechFreq: number;
  mechGain: number;
  drive: number;
  peak: number;
}

const GUN_PROFILES: Readonly<Record<WeaponId, GunProfile>> = {
  // 5.56 carbine: sharp, cracking, ~150ms of usable report.
  ar_vector: {
    duration: 0.19, bodyTau: 0.035, bodyFreq: 1250, bodyQ: 1.1, lowFreq: 380,
    thumpFrom: 190, thumpTo: 48, thumpTau: 0.045, thumpGain: 0.7,
    crackTau: 0.0011, crackFreq: 4200, crackGain: 0.95,
    mechDelay: 0.020, mechFreq: 3400, mechGain: 0.18, drive: 2.2, peak: 0.97,
  },
  // 9mm SMG: less powder, shorter bore — snappier and a good octave brighter.
  smg_wasp: {
    duration: 0.14, bodyTau: 0.024, bodyFreq: 1850, bodyQ: 1.3, lowFreq: 520,
    thumpFrom: 160, thumpTo: 60, thumpTau: 0.028, thumpGain: 0.5,
    crackTau: 0.0009, crackFreq: 5200, crackGain: 0.9,
    mechDelay: 0.014, mechFreq: 4200, mechGain: 0.22, drive: 2.0, peak: 0.95,
  },
  // .338 magnum: enormous charge, long barrel. Deep, and the tail runs ~900ms.
  sniper_longbow: {
    duration: 0.95, bodyTau: 0.16, bodyFreq: 520, bodyQ: 0.8, lowFreq: 150,
    thumpFrom: 95, thumpTo: 26, thumpTau: 0.30, thumpGain: 1.0,
    crackTau: 0.0016, crackFreq: 3200, crackGain: 1.0,
    mechDelay: 0.045, mechFreq: 2600, mechGain: 0.12, drive: 2.8, peak: 1.0,
  },
  // 12ga: low pressure, huge bore. Broad (low Q) and blunt rather than sharp.
  shotgun_breach: {
    duration: 0.46, bodyTau: 0.075, bodyFreq: 700, bodyQ: 0.55, lowFreq: 240,
    thumpFrom: 130, thumpTo: 34, thumpTau: 0.11, thumpGain: 0.95,
    crackTau: 0.0020, crackFreq: 2600, crackGain: 0.8,
    mechDelay: 0.055, mechFreq: 2200, mechGain: 0.16, drive: 2.6, peak: 0.99,
  },
  pistol_sidearm: {
    duration: 0.17, bodyTau: 0.030, bodyFreq: 1500, bodyQ: 1.2, lowFreq: 430,
    thumpFrom: 170, thumpTo: 52, thumpTau: 0.035, thumpGain: 0.6,
    crackTau: 0.0010, crackFreq: 4600, crackGain: 0.92,
    mechDelay: 0.026, mechFreq: 3800, mechGain: 0.20, drive: 2.1, peak: 0.96,
  },
};

/**
 * Rendered variants per weapon. Four is the point where, combined with the
 * per-shot pitch and level jitter applied at playback time, a sustained burst
 * stops being recognisable as a loop.
 */
export const GUN_VARIANTS = 4;

export function renderGunshot(weapon: WeaponId, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const base = GUN_PROFILES[weapon];
  const rng = new Rng(hash(weapon) ^ Math.imul(variant + 1, 0x9e3779b1));

  // Every variant is a genuinely different *instrument*, not the same render
  // with a different noise seed: the resonances, decays and layer balance all
  // move. Combined with per-shot pitch/gain jitter at playback time, a 10 r/s
  // full-auto burst never repeats an identical waveform.
  const p: GunProfile = {
    duration: base.duration * rng.jit(0.06),
    bodyTau: base.bodyTau * rng.jit(0.14),
    bodyFreq: base.bodyFreq * rng.jit(0.09),
    bodyQ: base.bodyQ * rng.jit(0.16),
    lowFreq: base.lowFreq * rng.jit(0.11),
    thumpFrom: base.thumpFrom * rng.jit(0.07),
    thumpTo: base.thumpTo * rng.jit(0.09),
    thumpTau: base.thumpTau * rng.jit(0.13),
    thumpGain: base.thumpGain * rng.jit(0.12),
    crackTau: base.crackTau * rng.jit(0.18),
    crackFreq: base.crackFreq * rng.jit(0.12),
    crackGain: base.crackGain * rng.jit(0.10),
    mechDelay: base.mechDelay * rng.jit(0.22),
    mechFreq: base.mechFreq * rng.jit(0.10),
    mechGain: base.mechGain * rng.jit(0.30),
    drive: base.drive * rng.jit(0.10),
    peak: base.peak,
  };

  const n = Math.ceil(p.duration * sr);
  const out = new Float32Array(n);

  // 1 — SHOCK FRONT. Two bands: a resonant crack and a raw broadband spike.
  addNoise(out, sr, rng, {
    dur: 0.02, mode: 'bp', f0: p.crackFreq, f1: p.crackFreq * 0.55, sweepDur: 0.006,
    q: 1.0, atk: 0.00004, tau: p.crackTau, gain: p.crackGain,
  });
  addNoise(out, sr, rng, {
    dur: 0.006, mode: 'hp', f0: 900, q: 0.7,
    atk: 0.00003, tau: p.crackTau * 0.55, gain: p.crackGain * 0.75,
  });

  // 2 — GAS BODY. Downward formant sweep: the resonant cavity effectively
  // lengthens as the gas column exits, dropping the centre by ~35%.
  addNoise(out, sr, rng, {
    dur: p.duration, mode: 'bp', f0: p.bodyFreq * 1.35, f1: p.bodyFreq * 0.82,
    sweepDur: p.bodyTau * 2.2, q: p.bodyQ, atk: 0.0004, tau: p.bodyTau, gain: 0.85, pink: true,
  });
  addNoise(out, sr, rng, {
    dur: p.duration, mode: 'bp', f0: p.lowFreq * 1.2, f1: p.lowFreq * 0.85,
    sweepDur: p.bodyTau * 3, q: 0.9, atk: 0.0008, tau: p.bodyTau * 1.9, gain: 0.7, pink: true,
  });
  // A wide-open hiss so the spectrum is not obviously two bumps.
  addNoise(out, sr, rng, {
    dur: p.duration, mode: 'hp', f0: 2600, q: 0.5,
    atk: 0.0002, tau: p.bodyTau * 0.7, gain: 0.28,
  });

  // 3 — PRESSURE PULSE.
  addSweep(out, sr, 0, p.thumpFrom, p.thumpTo, p.thumpTau * 1.4,
    0.0006, p.thumpTau, p.thumpGain, true);

  // 4 — ACTION. Inharmonic ratios: this is a small steel part, not a bell.
  addModal(out, sr, p.mechDelay, [
    { f: p.mechFreq, a: 1.0, tau: 0.012, droop: 0.02 },
    { f: p.mechFreq * 1.71, a: 0.55, tau: 0.008 },
    { f: p.mechFreq * 2.63, a: 0.3, tau: 0.005 },
  ], p.mechGain);
  addNoise(out, sr, rng, {
    t0: p.mechDelay, dur: 0.03, mode: 'bp', f0: p.mechFreq * 1.4, q: 1.2,
    atk: 0.00006, tau: 0.004, gain: p.mechGain * 0.9,
  });

  for (let i = 0; i < n; i++) out[i] = softClip(out[i], p.drive);
  dcBlock(out);
  // Fade before normalise — see `finish()`. A shock front lands within the
  // first handful of samples, exactly where the head fade is.
  fade(out, sr, 0.06, Math.min(6, p.duration * 1000 * 0.08));
  normalize(out, p.peak);
  return toBuffer(out, sr);
}

/**
 * The same shot heard from a long way off.
 *
 * Physics: air absorption is strongly frequency dependent (roughly ∝ f²), so
 * over 100m+ the shock front is simply gone and only the low-mid pressure and
 * the environmental reflections survive. The result is a *thud plus a rumble*
 * with no crack at all — which is exactly why crossfading to this variant with
 * distance reads as "far away" far more convincingly than turning the volume
 * down and adding a lowpass ever can.
 */
export function renderDistantReport(variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const rng = new Rng(hash('fire_distant') ^ Math.imul(variant + 1, 0x85ebca6b));
  const dur = 1.15 * rng.jit(0.08);
  const n = Math.ceil(dur * sr);
  const out = new Float32Array(n);

  // No crack layer at all — that is the whole point.
  addSweep(out, sr, 0, 88 * rng.jit(0.1), 30, 0.18, 0.004, 0.17, 1.0, true);
  addNoise(out, sr, rng, {
    dur: 0.5, mode: 'bp', f0: 380 * rng.jit(0.12), f1: 260, sweepDur: 0.12,
    q: 1.1, atk: 0.003, tau: 0.09, gain: 0.75, pink: true,
  });
  addNoise(out, sr, rng, {
    dur: 0.4, mode: 'lp', f0: 1500, f1: 700, sweepDur: 0.1, q: 0.6,
    atk: 0.0015, tau: 0.05, gain: 0.45, pink: true,
  });
  // Environmental return: the terrain slapping the report back at you.
  addNoise(out, sr, rng, {
    t0: 0.03, dur: dur - 0.03, mode: 'lp', f0: 1200, f1: 280, sweepDur: 0.5, q: 0.55,
    atk: 0.05, tau: 0.42, gain: 0.5, pink: true, amDepth: 0.25, amRate: 3.5,
  });

  for (let i = 0; i < n; i++) out[i] = softClip(out[i], 1.6);
  dcBlock(out);
  // Fade before normalise — see `finish()`. This one matters more than most:
  // the head fade is 1.5ms (72 samples), long enough to swallow a peak.
  fade(out, sr, 1.5, 40);
  normalize(out, 0.82);
  return toBuffer(out, sr);
}

// ─────────────────────────────────────────────────────────────────────────────
// TAILS — the environment answering the shot
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A tail is not reverb on the gunshot, it is a *separate, louder-than-you-think
 * layer* mixed under it. Ask anyone who has heard a rifle outdoors: the report
 * is 150ms, the countryside answering it runs for a second and a half. Games
 * that skip this layer sound like the weapon is firing in a vacuum.
 *
 * Indoor: short pre-delay, discrete bright slap-backs off nearby hard surfaces,
 *         fast dense decay.
 * Outdoor: long pre-delay before anything comes back at all, dark and diffuse,
 *          with a couple of distinct distant returns and a long low rumble.
 */
export function renderTail(kind: 'indoor' | 'outdoor', variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const indoor = kind === 'indoor';
  const dur = indoor ? 0.46 : 1.75;
  const n = Math.ceil(dur * sr);
  const chans: Float32Array[] = [new Float32Array(n), new Float32Array(n)];

  for (let c = 0; c < 2; c++) {
    // Independent noise streams per channel give a wide, decorrelated field;
    // the discrete taps below are only *slightly* offset so the image stays
    // anchored rather than turning into two unrelated sounds.
    const rng = new Rng(hash('tail_' + kind) ^ Math.imul(variant * 2 + c + 1, 0xc2b2ae35));
    const out = chans[c];
    const skew = c === 0 ? 1 : 1.0 + 0.06;

    if (indoor) {
      const taps: readonly number[] = [0.009, 0.016, 0.024, 0.034, 0.047, 0.062];
      for (let k = 0; k < taps.length; k++) {
        addNoise(out, sr, rng, {
          t0: taps[k] * skew * rng.jit(0.08), dur: 0.02, mode: 'bp',
          f0: 2600 * rng.jit(0.25), q: 0.8, atk: 0.0002,
          tau: 0.004, gain: 0.85 * Math.pow(0.72, k),
        });
      }
      addNoise(out, sr, rng, {
        dur: dur, mode: 'bp', f0: 2400, f1: 1100, sweepDur: 0.16, q: 0.5,
        atk: 0.008, tau: 0.075, gain: 0.55,
      });
      addNoise(out, sr, rng, {
        dur: dur, mode: 'lp', f0: 700, f1: 240, sweepDur: 0.2, q: 0.6,
        atk: 0.012, tau: 0.10, gain: 0.4, pink: true,
      });
    } else {
      // Distant hard returns — a treeline, a building 60m away.
      const taps: readonly [number, number][] = [[0.115, 0.42], [0.175, 0.26], [0.265, 0.16], [0.40, 0.09]];
      for (const [tt, gg] of taps) {
        addNoise(out, sr, rng, {
          t0: tt * skew * rng.jit(0.12), dur: 0.14, mode: 'lp',
          f0: 1500 * rng.jit(0.2), f1: 600, sweepDur: 0.06, q: 0.6,
          atk: 0.006, tau: 0.035, gain: gg, pink: true,
        });
      }
      // The long diffuse field. Generated at half rate: it is lowpassed under
      // 1kHz throughout, so the upper octave it cannot represent is empty
      // anyway, and we halve the cost of the single most expensive layer.
      const hn = Math.ceil(n / 2);
      const half = new Float32Array(hn);
      addNoise(half, sr / 2, rng, {
        dur: dur, mode: 'lp', f0: 900, f1: 170, sweepDur: 0.9, q: 0.55,
        atk: 0.09, tau: 0.52, gain: 1.0, pink: true, amDepth: 0.3, amRate: 2.2,
      });
      addUpsampled(out, half, 0.75);
      addNoise(out, sr, rng, {
        dur: 0.7, mode: 'bp', f0: 2200, f1: 900, sweepDur: 0.35, q: 0.5,
        atk: 0.03, tau: 0.16, gain: 0.22,
      });
    }

    dcBlock(out);
  }

  // De-click first, then normalise the pair together (see `finish`), so the
  // stereo image is preserved and the level lands where it was designed.
  for (const ch of chans) fade(ch, sr, 0.5, indoor ? 20 : 90);
  let peak = 0;
  for (const ch of chans) for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(ch[i]));
  if (peak > 1e-9) {
    const g = 0.6 / peak;
    for (const ch of chans) for (let i = 0; i < n; i++) ch[i] *= g;
  }
  return toStereoBuffer(chans[0], chans[1], sr);
}

// ─────────────────────────────────────────────────────────────────────────────
// IMPACTS
// ─────────────────────────────────────────────────────────────────────────────

export type ImpactKind = 'concrete' | 'metal' | 'wood' | 'sand' | 'glass' | 'flesh' | 'water';

export function renderImpact(kind: ImpactKind, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const rng = new Rng(hash('impact_' + kind) ^ Math.imul(variant + 1, 0x27d4eb2f));

  switch (kind) {
    // Dry, brittle, mineral. A hard crack, a scatter of pulverised grit, and a
    // barely-there dust puff. No ring: concrete is far too lossy to resonate.
    case 'concrete': {
      const n = Math.ceil(0.17 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, {
        dur: 0.02, mode: 'bp', f0: 1600 * rng.jit(0.2), q: 1.2,
        atk: 0.00005, tau: 0.0018, gain: 1.0,
      });
      addNoise(out, sr, rng, {
        dur: 0.09, mode: 'hp', f0: 2200 * rng.jit(0.15), q: 0.7,
        atk: 0.0002, tau: 0.030, gain: 0.45, density: 0.55, grainMs: 1.2,
      });
      addSweep(out, sr, 0, 150 * rng.jit(0.15), 88, 0.02, 0.0006, 0.020, 0.42, true);
      addNoise(out, sr, rng, { dur: 0.14, mode: 'lp', f0: 700, q: 0.6, atk: 0.003, tau: 0.07, gain: 0.16, pink: true });
      return finish(out, sr, 1.4, 0.8);
    }

    // Struck steel: a bright inharmonic ring on free-bar mode ratios, plus —
    // about half the time — the classic ricochet whine. That whine is a real
    // doppler artefact of a deformed round spinning away, and it is modelled
    // here as a narrow band descending exponentially with a little vibrato.
    case 'metal': {
      const n = Math.ceil(0.6 * sr);
      const out = new Float32Array(n);
      const f0 = rng.range(540, 780);
      addNoise(out, sr, rng, {
        dur: 0.02, mode: 'bp', f0: 5000 * rng.jit(0.2), q: 0.9,
        atk: 0.00004, tau: 0.0012, gain: 0.9,
      });
      addModal(out, sr, 0, [
        { f: f0 * rng.jit(0.01), a: 1.0, tau: 0.30 * rng.jit(0.2), droop: 0.015 },
        { f: f0 * 2.76 * rng.jit(0.012), a: 0.62, tau: 0.21 * rng.jit(0.2), droop: 0.012 },
        { f: f0 * 5.40 * rng.jit(0.014), a: 0.40, tau: 0.15 * rng.jit(0.2) },
        { f: f0 * 8.93 * rng.jit(0.016), a: 0.24, tau: 0.10 * rng.jit(0.2) },
        { f: f0 * 13.34 * rng.jit(0.02), a: 0.13, tau: 0.06 },
      ], 0.55);
      addNoise(out, sr, rng, {
        dur: 0.1, mode: 'bp', f0: 3000 * rng.jit(0.2), q: 1.4,
        atk: 0.0001, tau: 0.04, gain: 0.32,
      });
      if (rng.chance(0.55)) {
        const rico = new Float32Array(n);
        const from = rng.range(2500, 3300);
        const to = rng.range(700, 1000);
        addNoise(rico, sr, rng, {
          t0: 0.008, dur: 0.34, mode: 'bp', f0: from, f1: to, sweepDur: 0.28,
          q: 7, atk: 0.008, tau: 0.11, gain: 1.0,
        });
        // Vibrato: the round is tumbling, so the whine warbles.
        let phase = 0;
        for (let i = 0; i < n; i++) {
          phase += (2 * Math.PI * 6.5) / sr;
          out[i] += rico[i] * (0.75 + 0.25 * Math.sin(phase)) * 0.5;
        }
      }
      return finish(out, sr, 1.5, 0.85);
    }

    // Hollow and mid-low. Wood is stiff enough to ring but heavily damped, so
    // the modes die in ~100ms and the dominant impression is a "thock".
    case 'wood': {
      const n = Math.ceil(0.26 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 0.02, mode: 'bp', f0: 900 * rng.jit(0.2), q: 1.0, atk: 0.00008, tau: 0.0025, gain: 0.8 });
      addNoise(out, sr, rng, { dur: 0.14, mode: 'bp', f0: 420 * rng.jit(0.18), q: 2.2, atk: 0.0004, tau: 0.035, gain: 1.0, pink: true });
      addModal(out, sr, 0.0008, [
        { f: 220 * rng.jit(0.1), a: 1.0, tau: 0.12, droop: 0.01 },
        { f: 561 * rng.jit(0.1), a: 0.55, tau: 0.08 },
        { f: 1183 * rng.jit(0.12), a: 0.22, tau: 0.045 },
      ], 0.42);
      addNoise(out, sr, rng, { dur: 0.05, mode: 'hp', f0: 3500, q: 0.7, atk: 0.0002, tau: 0.012, gain: 0.26, density: 0.5, grainMs: 1.0 });
      return finish(out, sr, 1.3, 0.78);
    }

    // Loose granular material absorbs the shock entirely: there is no transient
    // and no resonance, only a broadband puff that darkens as it decays.
    case 'sand': {
      const n = Math.ceil(0.2 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, {
        dur: 0.16, mode: 'lp', f0: 1700 * rng.jit(0.15), f1: 480, sweepDur: 0.05,
        q: 0.6, atk: 0.006, tau: 0.045, gain: 1.0, pink: true,
      });
      addNoise(out, sr, rng, {
        dur: 0.12, mode: 'bp', f0: 3000 * rng.jit(0.2), q: 0.9,
        atk: 0.004, tau: 0.03, gain: 0.35, density: 0.35, grainMs: 0.9,
      });
      addNoise(out, sr, rng, { dur: 0.18, mode: 'lp', f0: 300, q: 0.6, atk: 0.008, tau: 0.05, gain: 0.22, pink: true });
      return finish(out, sr, 1.1, 0.55);
    }

    // Shatter = a large number of independent small resonators all excited at
    // once and dying at different rates. Front-loaded (bias 2.2) so the ear
    // hears one event with a tinkling aftermath, not a rattle.
    case 'glass': {
      const n = Math.ceil(0.6 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 0.02, mode: 'hp', f0: 3000, q: 0.7, atk: 0.00005, tau: 0.003, gain: 0.85 });
      addGrains(out, sr, rng, {
        count: 58, tStart: 0, tEnd: 0.32, fLo: 2800, fHi: 9500,
        tauLo: 0.006, tauHi: 0.035, gain: 0.5, bias: 2.2,
      });
      addGrains(out, sr, rng, {
        count: 22, tStart: 0.28, tEnd: 0.52, fLo: 3500, fHi: 11000,
        tauLo: 0.004, tauHi: 0.018, gain: 0.2, bias: 1.4,
      });
      addModal(out, sr, 0, [
        { f: 1450 * rng.jit(0.1), a: 1.0, tau: 0.09 },
        { f: 2310 * rng.jit(0.1), a: 0.6, tau: 0.07 },
      ], 0.2);
      return finish(out, sr, 1.2, 0.82);
    }

    // Soft tissue is almost all lowpass: a dull pressure thud plus a "squelch"
    // formant sliding down. Deliberately no HF transient — a bright flesh hit
    // is the classic uncanny mistake.
    case 'flesh': {
      const n = Math.ceil(0.3 * sr);
      const out = new Float32Array(n);
      addSweep(out, sr, 0, 120 * rng.jit(0.12), 55, 0.03, 0.0012, 0.045, 1.0, true);
      addNoise(out, sr, rng, {
        dur: 0.2, mode: 'lp', f0: 700 * rng.jit(0.15), f1: 250, sweepDur: 0.05,
        q: 0.7, atk: 0.0016, tau: 0.055, gain: 0.8, pink: true,
      });
      addNoise(out, sr, rng, {
        dur: 0.22, mode: 'bp', f0: 900 * rng.jit(0.2), f1: 300, sweepDur: 0.09,
        q: 3.0, atk: 0.004, tau: 0.09, gain: 0.32, amDepth: 0.35, amRate: 40,
      });
      return finish(out, sr, 1.2, 0.7);
    }

    // The pitch of a collapsing air cavity RISES as it shrinks (Minnaert
    // resonance, f ∝ 1/radius). Getting this direction right is the whole
    // difference between "plop" and "boop".
    case 'water': {
      const n = Math.ceil(0.38 * sr);
      const out = new Float32Array(n);
      addSweep(out, sr, 0.004, 280 * rng.jit(0.15), 940 * rng.jit(0.12), 0.05, 0.0015, 0.045, 1.0);
      addNoise(out, sr, rng, {
        dur: 0.16, mode: 'bp', f0: 1200, f1: 3200, sweepDur: 0.04, q: 0.8,
        atk: 0.0012, tau: 0.06, gain: 0.5,
      });
      addGrains(out, sr, rng, {
        count: 9, tStart: 0.09, tEnd: 0.30, fLo: 900, fHi: 2600,
        tauLo: 0.01, tauHi: 0.03, gain: 0.22, bias: 1.6,
      });
      addNoise(out, sr, rng, { dur: 0.25, mode: 'lp', f0: 250, q: 0.7, atk: 0.006, tau: 0.08, gain: 0.26, pink: true });
      return finish(out, sr, 1.1, 0.7);
    }

    default: {
      const out = new Float32Array(Math.ceil(0.1 * sr));
      return finish(out, sr, 1, 0.5);
    }
  }
}

/**
 * Shared post-chain: saturate, DC-block, de-click, normalise.
 *
 * Note the order: the fades are applied BEFORE normalisation. Several of these
 * sounds (short bright ticks especially) have their true peak inside the first
 * few samples, so normalising first and fading afterwards silently scales the
 * peak back down and the sound lands several dB under its designed level.
 */
function finish(out: Float32Array, sr: number, drive: number, peak: number): AudioBuffer {
  if (drive > 1.01) for (let i = 0; i < out.length; i++) out[i] = softClip(out[i], drive);
  dcBlock(out);
  fade(out, sr, 0.1, 5);
  normalize(out, peak);
  return toBuffer(out, sr);
}

// ─────────────────────────────────────────────────────────────────────────────
// FOOTSTEPS
// ─────────────────────────────────────────────────────────────────────────────

export type StepKind = 'concrete' | 'sand' | 'metal' | 'wood';

/**
 * A footstep is two impacts, not one: heel strike, then a quieter and slightly
 * brighter toe/roll 55-85ms later. Rendering the pair into a single buffer
 * keeps the voice count down and — because the gap is randomised per variant —
 * is a cheap extra axis of variation on top of the four variants themselves.
 */
export function renderFootstep(kind: StepKind, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const rng = new Rng(hash('step_' + kind) ^ Math.imul(variant + 1, 0x165667b1));
  const gap = rng.range(0.055, 0.088);
  const dur = kind === 'metal' ? 0.5 : 0.38;
  const n = Math.ceil(dur * sr);
  const out = new Float32Array(n);

  const part = (t0: number, g: number, rate: number): void => {
    switch (kind) {
      case 'concrete':
        addNoise(out, sr, rng, { t0, dur: 0.09, mode: 'lp', f0: 950 * rate, f1: 420, sweepDur: 0.02, q: 0.7, atk: 0.0004, tau: 0.020 * g, gain: 0.9 * g, pink: true });
        addNoise(out, sr, rng, { t0, dur: 0.06, mode: 'hp', f0: 2500 * rate, q: 0.7, atk: 0.0003, tau: 0.015, gain: 0.3 * g, density: 0.45, grainMs: 1.1 });
        addSweep(out, sr, t0, 125 * rate, 80, 0.02, 0.0008, 0.026, 0.5 * g, true);
        break;
      case 'sand':
        // Crunch: many tiny grains, a soft envelope, and no transient at all.
        addGrains(out, sr, rng, { count: 34, tStart: t0, tEnd: t0 + 0.075, fLo: 900 * rate, fHi: 4600 * rate, tauLo: 0.002, tauHi: 0.012, gain: 0.42 * g, bias: 1.5, noise: true, q: 3 });
        addNoise(out, sr, rng, { t0, dur: 0.12, mode: 'lp', f0: 700 * rate, f1: 320, sweepDur: 0.04, q: 0.6, atk: 0.004, tau: 0.05, gain: 0.5 * g, pink: true });
        break;
      case 'metal':
        addModal(out, sr, t0, [
          { f: 380 * rate * rng.jit(0.06), a: 1.0, tau: 0.18 * g, droop: 0.02 },
          { f: 912 * rate * rng.jit(0.06), a: 0.5, tau: 0.12 * g },
          { f: 1558 * rate * rng.jit(0.08), a: 0.3, tau: 0.08 },
          { f: 2546 * rate * rng.jit(0.08), a: 0.16, tau: 0.05 },
        ], 0.42 * g);
        addNoise(out, sr, rng, { t0, dur: 0.03, mode: 'bp', f0: 3600 * rate, q: 1.0, atk: 0.00006, tau: 0.004, gain: 0.5 * g });
        addSweep(out, sr, t0, 95, 62, 0.03, 0.0012, 0.05, 0.4 * g, true);
        break;
      case 'wood':
        addSweep(out, sr, t0, 165 * rate, 110, 0.025, 0.0008, 0.04, 0.8 * g, true);
        addModal(out, sr, t0, [
          { f: 240 * rate * rng.jit(0.08), a: 1.0, tau: 0.09 * g },
          { f: 470 * rate * rng.jit(0.08), a: 0.45, tau: 0.05 },
        ], 0.4 * g);
        addNoise(out, sr, rng, { t0, dur: 0.08, mode: 'bp', f0: 1100 * rate, q: 1.4, atk: 0.0006, tau: 0.025, gain: 0.45 * g, pink: true });
        break;
      default:
        break;
    }
  };

  part(0, 1, 1);
  part(gap, rng.range(0.45, 0.62), rng.range(1.03, 1.14));

  // A loose board creaking is the single most evocative wood-floor detail, but
  // it must be occasional or it becomes a comedy sound effect.
  if (kind === 'wood' && rng.chance(0.45)) {
    addNoise(out, sr, rng, {
      t0: gap * 0.4, dur: 0.22, mode: 'bp', f0: 700 * rng.jit(0.2), f1: 1150,
      sweepDur: 0.16, q: 9, atk: 0.02, tau: 0.075, gain: 0.3, amDepth: 0.5, amRate: 21,
    });
  }

  return finish(out, sr, 1.2, kind === 'sand' ? 0.42 : 0.58);
}

// ─────────────────────────────────────────────────────────────────────────────
// MECHANICAL — reloads, dry fire, ADS
// ─────────────────────────────────────────────────────────────────────────────

export type MechKind = 'magout' | 'magin' | 'charge' | 'dryfire' | 'ads_in' | 'ads_out';

export function renderMechanical(kind: MechKind, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const rng = new Rng(hash('mech_' + kind) ^ Math.imul(variant + 1, 0x2545f491));

  switch (kind) {
    // Polymer magazine leaving a steel well: a dull-bright clack (polymer modes
    // are heavily damped, so short taus and a lowpassed spectrum), a spring
    // rattle, and the mag body thumping the catch.
    case 'magout': {
      const n = Math.ceil(0.28 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 0.02, mode: 'bp', f0: 2200 * rng.jit(0.15), q: 1.1, atk: 0.00006, tau: 0.002, gain: 0.85 });
      addModal(out, sr, 0, [
        { f: 780 * rng.jit(0.08), a: 1.0, tau: 0.045 },
        { f: 1560 * rng.jit(0.08), a: 0.5, tau: 0.028 },
        { f: 2900 * rng.jit(0.1), a: 0.22, tau: 0.014 },
      ], 0.5);
      addSweep(out, sr, 0, 155, 105, 0.02, 0.001, 0.03, 0.4, true);
      for (const t of [0.042, 0.071, 0.108]) {
        addNoise(out, sr, rng, { t0: t * rng.jit(0.15), dur: 0.02, mode: 'bp', f0: rng.range(2600, 5200), q: 2.5, atk: 0.00008, tau: 0.0035, gain: 0.16 });
      }
      return finish(out, sr, 1.3, 0.62);
    }

    // Three beats: the mag sliding up the well, seating hard, then the catch
    // clicking over. The click *after* the thunk is what says "locked".
    case 'magin': {
      const n = Math.ceil(0.36 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 0.10, mode: 'bp', f0: 1700, f1: 3200, sweepDur: 0.09, q: 2.0, atk: 0.012, tau: 0.05, gain: 0.32 });
      const seat = 0.105 * rng.jit(0.12);
      addNoise(out, sr, rng, { t0: seat, dur: 0.12, mode: 'lp', f0: 620 * rng.jit(0.12), q: 0.7, atk: 0.0006, tau: 0.035, gain: 0.9, pink: true });
      addSweep(out, sr, seat, 132, 88, 0.03, 0.0009, 0.04, 0.85, true);
      const click = seat + 0.048 * rng.jit(0.2);
      addModal(out, sr, click, [
        { f: 3600 * rng.jit(0.1), a: 1.0, tau: 0.008 },
        { f: 5200 * rng.jit(0.1), a: 0.5, tau: 0.005 },
      ], 0.4);
      addNoise(out, sr, rng, { t0: click, dur: 0.01, mode: 'hp', f0: 3800, q: 0.8, atk: 0.00004, tau: 0.0015, gain: 0.35 });
      return finish(out, sr, 1.35, 0.72);
    }

    // Charging handle: a long metal-on-metal slide with a faint ratchet, then a
    // hard bolt-into-battery snap with a lot of HF.
    case 'charge': {
      const n = Math.ceil(0.42 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, {
        dur: 0.145, mode: 'bp', f0: 900, f1: 2600, sweepDur: 0.14, q: 1.6,
        atk: 0.010, tau: 0.28, gain: 0.4, amDepth: 0.45, amRate: 88,
      });
      const snap = 0.168 * rng.jit(0.08);
      addNoise(out, sr, rng, { t0: snap, dur: 0.015, mode: 'hp', f0: 4000, q: 0.7, atk: 0.00004, tau: 0.0016, gain: 0.95 });
      addModal(out, sr, snap, [
        { f: 2400 * rng.jit(0.08), a: 1.0, tau: 0.055, droop: 0.02 },
        { f: 4100 * rng.jit(0.08), a: 0.55, tau: 0.035 },
        { f: 6800 * rng.jit(0.1), a: 0.28, tau: 0.018 },
      ], 0.5);
      addSweep(out, sr, snap, 190, 120, 0.02, 0.0008, 0.026, 0.55, true);
      return finish(out, sr, 1.4, 0.8);
    }

    // Hammer on an empty chamber. Deliberately tiny: the *absence* of a shot is
    // the message, and a loud dry fire ruins the panic it is supposed to cause.
    case 'dryfire': {
      const n = Math.ceil(0.08 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 0.01, mode: 'bp', f0: 3000 * rng.jit(0.12), q: 1.2, atk: 0.00004, tau: 0.0012, gain: 1.0 });
      addModal(out, sr, 0, [
        { f: 3100 * rng.jit(0.08), a: 1.0, tau: 0.012 },
        { f: 5400 * rng.jit(0.1), a: 0.4, tau: 0.006 },
        { f: 220, a: 0.5, tau: 0.010 },
      ], 0.35);
      return finish(out, sr, 1.2, 0.42);
    }

    // Bringing the weapon up: mostly the sound of a sling and a sleeve moving
    // (a broadband swell, no transient), finished with one small metallic tick
    // as the optic meets the eye line.
    case 'ads_in':
    case 'ads_out': {
      const inward = kind === 'ads_in';
      const n = Math.ceil((inward ? 0.24 : 0.21) * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, {
        dur: 0.18, mode: 'bp', f0: inward ? 800 : 2200, f1: inward ? 2200 : 800,
        sweepDur: 0.12, q: 0.8, atk: inward ? 0.035 : 0.022, tau: 0.06, gain: 0.55, pink: true,
      });
      addNoise(out, sr, rng, { t0: 0.012, dur: 0.05, mode: 'lp', f0: 900, q: 0.6, atk: 0.004, tau: 0.02, gain: 0.3, density: 0.6, grainMs: 2.2 });
      const tick = inward ? 0.118 : 0.086;
      addModal(out, sr, tick * rng.jit(0.15), [
        { f: 4200 * rng.jit(0.1), a: 1.0, tau: 0.006 },
        { f: 6100 * rng.jit(0.1), a: 0.45, tau: 0.004 },
      ], 0.3);
      return finish(out, sr, 1.1, inward ? 0.36 : 0.31);
    }

    default: {
      return finish(new Float32Array(Math.ceil(0.05 * sr)), sr, 1, 0.3);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FEEDBACK — hitmarker, headshot, kill, damage, health
// ─────────────────────────────────────────────────────────────────────────────

export type FeedbackKind = 'hitmarker' | 'headshot' | 'kill' | 'damage' | 'lowhealth' | 'heartbeat';

export function renderFeedback(kind: FeedbackKind, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const rng = new Rng(hash('fb_' + kind) ^ Math.imul(variant + 1, 0x68e31da4));

  // Feedback cues are the only sounds allowed to be un-diegetic, so they live
  // in a spectral slot nothing else occupies (2-6kHz, short, stereo-wide). They
  // must stay legible when a machine gun and an explosion are both playing,
  // which is also why the UI bus bypasses ear-fatigue filtering downstream.
  const stereoOut = (dur: number): [Float32Array, Float32Array] => {
    const n = Math.ceil(dur * sr);
    return [new Float32Array(n), new Float32Array(n)];
  };

  switch (kind) {
    case 'hitmarker': {
      const [l, r] = stereoOut(0.1);
      for (let c = 0; c < 2; c++) {
        const o = c === 0 ? l : r;
        const d = c === 0 ? -4 : 4;
        addModal(o, sr, 0, [
          { f: 2350 + d, a: 1.0, tau: 0.022 },
          { f: 3520 + d * 1.5, a: 0.45, tau: 0.016 },
        ], 0.7);
        addNoise(o, sr, rng, { dur: 0.006, mode: 'hp', f0: 3000, q: 0.7, atk: 0.00003, tau: 0.0008, gain: 0.4 });
      }
      return finishStereo(l, r, sr, 0.55);
    }

    // Same family as the hitmarker but a fourth higher with a rising bend — the
    // upward pitch inflection is universally read as "better outcome".
    case 'headshot': {
      const [l, r] = stereoOut(0.15);
      for (let c = 0; c < 2; c++) {
        const o = c === 0 ? l : r;
        const d = c === 0 ? 0.997 : 1.003;
        addSweep(o, sr, 0, 3050 * d, 3450 * d, 0.03, 0.0004, 0.035, 0.8);
        addSweep(o, sr, 0, 4600 * d, 5150 * d, 0.03, 0.0004, 0.022, 0.4);
        addGrains(o, sr, rng, { count: 6, tStart: 0.004, tEnd: 0.05, fLo: 6000, fHi: 11000, tauLo: 0.004, tauHi: 0.012, gain: 0.25 });
        addNoise(o, sr, rng, { dur: 0.006, mode: 'hp', f0: 4000, q: 0.7, atk: 0.00003, tau: 0.0008, gain: 0.35 });
      }
      return finishStereo(l, r, sr, 0.62);
    }

    // Two beats a fifth apart plus a shimmer: long enough to register as an
    // event, short enough to survive back-to-back kills without smearing.
    case 'kill': {
      const [l, r] = stereoOut(0.34);
      for (let c = 0; c < 2; c++) {
        const o = c === 0 ? l : r;
        const d = c === 0 ? 0.996 : 1.004;
        addModal(o, sr, 0, [
          { f: 620 * d, a: 1.0, tau: 0.05 },
          { f: 1240 * d, a: 0.42, tau: 0.04 },
          { f: 1860 * d, a: 0.18, tau: 0.025 },
        ], 0.55);
        addModal(o, sr, 0.098, [
          { f: 930 * d, a: 1.0, tau: 0.075 },
          { f: 1860 * d, a: 0.45, tau: 0.05 },
          { f: 2790 * d, a: 0.2, tau: 0.03 },
        ], 0.6);
        addGrains(o, sr, rng, { count: 14, tStart: 0.1, tEnd: 0.3, fLo: 4000, fHi: 9000, tauLo: 0.008, tauHi: 0.03, gain: 0.16, bias: 1.3 });
        addSweep(o, sr, 0, 92, 62, 0.05, 0.001, 0.06, 0.32, true);
      }
      return finishStereo(l, r, sr, 0.6);
    }

    // Taking a hit: a body thud plus an audible "whoomp" of the ear reflex.
    // The whoomp is the sound-design twin of the duck() the engine applies at
    // the same moment — you hear the muffle arrive as well as feel it.
    case 'damage': {
      const n = Math.ceil(0.38 * sr);
      const out = new Float32Array(n);
      addSweep(out, sr, 0, 112 * rng.jit(0.1), 48, 0.05, 0.0012, 0.07, 1.0, true);
      addNoise(out, sr, rng, { dur: 0.2, mode: 'lp', f0: 430, q: 0.7, atk: 0.002, tau: 0.05, gain: 0.7, pink: true });
      addNoise(out, sr, rng, { dur: 0.34, mode: 'lp', f0: 320, f1: 180, sweepDur: 0.2, q: 0.6, atk: 0.05, tau: 0.12, gain: 0.34, pink: true });
      addNoise(out, sr, rng, { dur: 0.06, mode: 'bp', f0: 1400, q: 1.2, atk: 0.0004, tau: 0.014, gain: 0.22 });
      return finish(out, sr, 1.5, 0.8);
    }

    // Near death: a dark pulsing swell with a faint 2.8kHz tinnitus whine. Kept
    // low and slow so it raises tension without masking gameplay-critical cues.
    case 'lowhealth': {
      const [l, r] = stereoOut(0.95);
      for (let c = 0; c < 2; c++) {
        const o = c === 0 ? l : r;
        addNoise(o, sr, rng, { dur: 0.9, mode: 'lp', f0: 300, q: 0.7, atk: 0.18, tau: 0.45, gain: 0.9, pink: true, amDepth: 0.55, amRate: 1.6 });
        addSweep(o, sr, 0, 70, 62, 0.6, 0.15, 0.4, 0.5, true);
        addModal(o, sr, 0.05, [{ f: c === 0 ? 2790 : 2812, a: 1.0, tau: 0.5 }], 0.05);
      }
      return finishStereo(l, r, sr, 0.36);
    }

    // Lub-dub. The second beat is quieter, slightly higher and 195ms later,
    // which is the actual S1/S2 relationship — get the rhythm wrong and it
    // reads as a drum, not a heart.
    case 'heartbeat': {
      const [l, r] = stereoOut(0.66);
      for (let c = 0; c < 2; c++) {
        const o = c === 0 ? l : r;
        addSweep(o, sr, 0, 58, 33, 0.06, 0.006, 0.055, 1.0, true);
        addNoise(o, sr, rng, { dur: 0.1, mode: 'lp', f0: 120, q: 0.7, atk: 0.006, tau: 0.04, gain: 0.3, pink: true });
        addSweep(o, sr, 0.195, 65, 38, 0.05, 0.006, 0.045, 0.62, true);
        addNoise(o, sr, rng, { t0: 0.195, dur: 0.09, mode: 'lp', f0: 135, q: 0.7, atk: 0.006, tau: 0.032, gain: 0.2, pink: true });
      }
      return finishStereo(l, r, sr, 0.55);
    }

    default:
      return finish(new Float32Array(Math.ceil(0.05 * sr)), sr, 1, 0.3);
  }
}

/** As `finish`, but normalises the pair together so the image is not skewed. */
function finishStereo(l: Float32Array, r: Float32Array, sr: number, peak: number): AudioBuffer {
  dcBlock(l);
  dcBlock(r);
  fade(l, sr, 0.2, 8);
  fade(r, sr, 0.2, 8);
  let m = 0;
  for (let i = 0; i < l.length; i++) m = Math.max(m, Math.abs(l[i]), Math.abs(r[i]));
  if (m > 1e-9) {
    const g = peak / m;
    for (let i = 0; i < l.length; i++) {
      l[i] *= g;
      r[i] *= g;
    }
  }
  return toStereoBuffer(l, r, sr);
}

// ─────────────────────────────────────────────────────────────────────────────
// MISC WORLD SOUNDS
// ─────────────────────────────────────────────────────────────────────────────

export type MiscKind = 'shell' | 'jump' | 'land' | 'slide' | 'whizby' | 'explosion';

export function renderMisc(kind: MiscKind, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  const rng = new Rng(hash('misc_' + kind) ^ Math.imul(variant + 1, 0x7feb352d));

  switch (kind) {
    // Brass on concrete: three bounces, each shorter, quieter and *higher* (the
    // case loses energy to rotation, so later contacts excite the short modes).
    case 'shell': {
      const n = Math.ceil(0.34 * sr);
      const out = new Float32Array(n);
      const f0 = rng.range(2300, 2950);
      const bounces: readonly [number, number, number][] = [
        [0, 1.0, 1.0],
        [0.078 * rng.jit(0.25), 0.5, 1.06],
        [0.142 * rng.jit(0.25), 0.26, 1.11],
      ];
      for (const [t0, g, pitch] of bounces) {
        addModal(out, sr, t0, [
          { f: f0 * pitch, a: 1.0, tau: 0.045 * g, droop: 0.01 },
          { f: f0 * 2.31 * pitch, a: 0.55, tau: 0.03 * g },
          { f: f0 * 3.87 * pitch, a: 0.3, tau: 0.02 * g },
          { f: f0 * 5.6 * pitch, a: 0.15, tau: 0.012 * g },
        ], 0.6 * g);
        addNoise(out, sr, rng, { t0, dur: 0.008, mode: 'hp', f0: 4000, q: 0.7, atk: 0.00003, tau: 0.0015, gain: 0.35 * g });
      }
      return finish(out, sr, 1.1, 0.5);
    }

    // Gear and webbing shifting as the body leaves the ground, plus a short
    // exhale. Nothing percussive: a jump has no impact, only a release.
    case 'jump': {
      const n = Math.ceil(0.22 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 0.16, mode: 'bp', f0: 700, f1: 2600, sweepDur: 0.1, q: 0.7, atk: 0.008, tau: 0.05, gain: 0.6, pink: true });
      addGrains(out, sr, rng, { count: 5, tStart: 0.01, tEnd: 0.11, fLo: 2400, fHi: 5600, tauLo: 0.003, tauHi: 0.01, gain: 0.2, noise: true, q: 3 });
      addNoise(out, sr, rng, { dur: 0.2, mode: 'lp', f0: 500, q: 0.6, atk: 0.012, tau: 0.09, gain: 0.28, pink: true });
      return finish(out, sr, 1.1, 0.4);
    }

    // Boots + 30kg of kit arriving: a heavy sub thump, a broadband body, a
    // scuff, and the kit rattling ~40ms behind the impact.
    case 'land': {
      const n = Math.ceil(0.45 * sr);
      const out = new Float32Array(n);
      addSweep(out, sr, 0, 105 * rng.jit(0.1), 42, 0.06, 0.0012, 0.09, 1.0, true);
      addNoise(out, sr, rng, { dur: 0.2, mode: 'lp', f0: 520, q: 0.7, atk: 0.0012, tau: 0.06, gain: 0.7, pink: true });
      addNoise(out, sr, rng, { dur: 0.08, mode: 'hp', f0: 1800, q: 0.7, atk: 0.0005, tau: 0.03, gain: 0.26, density: 0.5, grainMs: 1.2 });
      addGrains(out, sr, rng, { count: 7, tStart: 0.03, tEnd: 0.2, fLo: 2500, fHi: 6000, tauLo: 0.003, tauHi: 0.014, gain: 0.28, noise: true, q: 3 });
      return finish(out, sr, 1.5, 0.85);
    }

    // Sustained friction. Two things sell it: a slow swell (contact area grows
    // as the body settles) and a noisy amplitude modulation from the surface
    // grain — constant-amplitude noise reads instantly as a synthesiser.
    case 'slide': {
      const n = Math.ceil(1.15 * sr);
      const out = new Float32Array(n);
      addNoise(out, sr, rng, { dur: 1.1, mode: 'bp', f0: 1200, f1: 750, sweepDur: 0.6, q: 1.1, atk: 0.11, tau: 0.45, gain: 0.9, pink: true, amDepth: 0.32, amRate: 27 });
      addNoise(out, sr, rng, { dur: 1.1, mode: 'lp', f0: 230, q: 0.6, atk: 0.08, tau: 0.5, gain: 0.5, pink: true });
      addNoise(out, sr, rng, { dur: 0.5, mode: 'bp', f0: 2600, q: 0.8, atk: 0.05, tau: 0.18, gain: 0.22, density: 0.7, grainMs: 3 });
      return finish(out, sr, 1.2, 0.55);
    }

    // Supersonic round passing within a couple of metres. The perceived pitch
    // sweeps *down* hard as the source flies past (classic doppler), the
    // amplitude is a tight gaussian because the near-field window is ~30ms, and
    // the little crack at the peak is the bow shock arriving.
    case 'whizby': {
      const n = Math.ceil(0.15 * sr);
      const out = new Float32Array(n);
      const band = new Float32Array(n);
      addNoise(band, sr, rng, {
        dur: 0.14, mode: 'bp', f0: 2900 * rng.jit(0.12), f1: 780 * rng.jit(0.12),
        sweepDur: 0.085, q: 6, atk: 0.0008, tau: 0.5, gain: 1.0,
      });
      for (let i = 0; i < n; i++) out[i] += band[i] * bell(i / sr, 0.045, 0.022);
      addNoise(out, sr, rng, { t0: 0.040, dur: 0.01, mode: 'hp', f0: 2200, q: 0.7, atk: 0.00004, tau: 0.0014, gain: 0.35 });
      addSweep(out, sr, 0.03, 180, 90, 0.04, 0.003, 0.025, 0.22, true);
      return finish(out, sr, 1.3, 0.7);
    }

    // Six layers, because a blast is genuinely six events:
    //   sub sweep   — the overpressure wave itself
    //   burst       — the fireball, broadband, razor attack, hard saturation
    //   crack       — the detonation front
    //   rumble      — the reflected/refracted energy rolling back for seconds
    //   debris      — fragments and material landing over the next second
    //   sub decay   — a residual 30Hz you feel more than hear
    case 'explosion': {
      const dur = 3.1;
      const n = Math.ceil(dur * sr);
      const out = new Float32Array(n);

      addSweep(out, sr, 0, 118 * rng.jit(0.08), 22, 0.55, 0.004, 0.26, 1.0, true);
      addNoise(out, sr, rng, { dur: 0.6, mode: 'hp', f0: 60, q: 0.6, atk: 0.0015, tau: 0.09, gain: 0.95, pink: true });
      addNoise(out, sr, rng, { dur: 0.03, mode: 'bp', f0: 1800, q: 0.8, atk: 0.00005, tau: 0.003, gain: 0.8 });

      // Rolling tail at half rate — it is lowpassed below 700Hz throughout.
      const hn = Math.ceil(n / 2);
      const half = new Float32Array(hn);
      addNoise(half, sr / 2, rng, {
        dur: dur, mode: 'lp', f0: 700, f1: 90, sweepDur: 1.6, q: 0.55,
        atk: 0.06, tau: 0.85, gain: 1.0, pink: true, amDepth: 0.35, amRate: 3.1,
      });
      addUpsampled(out, half, 0.8);

      addGrains(out, sr, rng, {
        count: 70, tStart: 0.12, tEnd: 1.45, fLo: 1200, fHi: 6500,
        tauLo: 0.004, tauHi: 0.02, gain: 0.3, bias: 1.8, noise: true, q: 4,
      });
      addSweep(out, sr, 0.02, 34, 26, 0.5, 0.02, 0.6, 0.35, true);

      return finish(out, sr, 3.2, 1.0);
    }

    default:
      return finish(new Float32Array(Math.ceil(0.05 * sr)), sr, 1, 0.3);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────────

export interface SoundSpec {
  /** How many distinct renders exist; the engine round-robins with jitter. */
  readonly variants: number;
  /** 1 = routed through an HRTF panner, 2 = stereo (tails, UI). */
  readonly channels: 1 | 2;
  /** Warm-up order: lower renders first. Gameplay-critical sounds go early. */
  readonly order: number;
}

/**
 * Warm-up ordering rationale: a player can shoot, hit something and take a
 * step within the first second of gameplay, so those families are rendered
 * before the bank finishes. Explosions, tails and the health cues can afford
 * to arrive a frame or two later, and are also the most expensive.
 */
export const SOUND_SPECS: Readonly<Record<SoundId, SoundSpec>> = {
  fire_ar: { variants: GUN_VARIANTS, channels: 1, order: 0 },
  fire_smg: { variants: GUN_VARIANTS, channels: 1, order: 1 },
  fire_pistol: { variants: GUN_VARIANTS, channels: 1, order: 2 },
  fire_shotgun: { variants: GUN_VARIANTS, channels: 1, order: 3 },
  fire_sniper: { variants: GUN_VARIANTS, channels: 1, order: 4 },

  impact_concrete: { variants: 3, channels: 1, order: 5 },
  impact_metal: { variants: 3, channels: 1, order: 6 },
  impact_flesh: { variants: 3, channels: 1, order: 7 },
  impact_wood: { variants: 3, channels: 1, order: 8 },
  impact_sand: { variants: 3, channels: 1, order: 9 },
  impact_glass: { variants: 3, channels: 1, order: 10 },
  impact_water: { variants: 3, channels: 1, order: 11 },

  hitmarker: { variants: 1, channels: 2, order: 12 },
  headshot: { variants: 1, channels: 2, order: 13 },

  footstep_concrete: { variants: 4, channels: 1, order: 14 },
  footstep_sand: { variants: 4, channels: 1, order: 15 },
  footstep_metal: { variants: 4, channels: 1, order: 16 },
  footstep_wood: { variants: 4, channels: 1, order: 17 },

  reload_magout: { variants: 2, channels: 1, order: 18 },
  reload_magin: { variants: 2, channels: 1, order: 19 },
  reload_charge: { variants: 2, channels: 1, order: 20 },
  dryfire: { variants: 2, channels: 1, order: 21 },
  ads_in: { variants: 2, channels: 1, order: 22 },
  ads_out: { variants: 2, channels: 1, order: 23 },

  shell_drop: { variants: 3, channels: 1, order: 24 },
  whizby: { variants: 3, channels: 1, order: 25 },

  tail_outdoor: { variants: 2, channels: 2, order: 26 },
  tail_indoor: { variants: 2, channels: 2, order: 27 },
  fire_distant: { variants: 2, channels: 1, order: 28 },

  kill: { variants: 1, channels: 2, order: 29 },
  damage: { variants: 2, channels: 1, order: 30 },
  land: { variants: 2, channels: 1, order: 31 },
  jump: { variants: 2, channels: 1, order: 32 },
  slide: { variants: 1, channels: 1, order: 33 },

  explosion: { variants: 2, channels: 1, order: 34 },
  heartbeat: { variants: 1, channels: 2, order: 35 },
  lowhealth: { variants: 1, channels: 2, order: 36 },
};

/** Every SoundId, ordered so the most gameplay-critical warms up first. */
export const WARMUP_ORDER: readonly SoundId[] = (Object.keys(SOUND_SPECS) as SoundId[]).sort(
  (a, b) => SOUND_SPECS[a].order - SOUND_SPECS[b].order,
);

/**
 * Render one variant of one sound. Pure, synchronous, and needs no
 * AudioContext — call it whenever there is a spare millisecond.
 */
export function renderSound(id: SoundId, variant: number, sr = SAMPLE_RATE): AudioBuffer {
  switch (id) {
    case 'fire_ar': return renderGunshot('ar_vector', variant, sr);
    case 'fire_smg': return renderGunshot('smg_wasp', variant, sr);
    case 'fire_sniper': return renderGunshot('sniper_longbow', variant, sr);
    case 'fire_shotgun': return renderGunshot('shotgun_breach', variant, sr);
    case 'fire_pistol': return renderGunshot('pistol_sidearm', variant, sr);
    case 'fire_distant': return renderDistantReport(variant, sr);

    case 'tail_indoor': return renderTail('indoor', variant, sr);
    case 'tail_outdoor': return renderTail('outdoor', variant, sr);

    case 'reload_magout': return renderMechanical('magout', variant, sr);
    case 'reload_magin': return renderMechanical('magin', variant, sr);
    case 'reload_charge': return renderMechanical('charge', variant, sr);
    case 'dryfire': return renderMechanical('dryfire', variant, sr);
    case 'ads_in': return renderMechanical('ads_in', variant, sr);
    case 'ads_out': return renderMechanical('ads_out', variant, sr);

    case 'impact_concrete': return renderImpact('concrete', variant, sr);
    case 'impact_metal': return renderImpact('metal', variant, sr);
    case 'impact_wood': return renderImpact('wood', variant, sr);
    case 'impact_sand': return renderImpact('sand', variant, sr);
    case 'impact_glass': return renderImpact('glass', variant, sr);
    case 'impact_flesh': return renderImpact('flesh', variant, sr);
    case 'impact_water': return renderImpact('water', variant, sr);

    case 'footstep_concrete': return renderFootstep('concrete', variant, sr);
    case 'footstep_sand': return renderFootstep('sand', variant, sr);
    case 'footstep_metal': return renderFootstep('metal', variant, sr);
    case 'footstep_wood': return renderFootstep('wood', variant, sr);

    case 'hitmarker': return renderFeedback('hitmarker', variant, sr);
    case 'headshot': return renderFeedback('headshot', variant, sr);
    case 'kill': return renderFeedback('kill', variant, sr);
    case 'damage': return renderFeedback('damage', variant, sr);
    case 'lowhealth': return renderFeedback('lowhealth', variant, sr);
    case 'heartbeat': return renderFeedback('heartbeat', variant, sr);

    case 'shell_drop': return renderMisc('shell', variant, sr);
    case 'jump': return renderMisc('jump', variant, sr);
    case 'land': return renderMisc('land', variant, sr);
    case 'slide': return renderMisc('slide', variant, sr);
    case 'whizby': return renderMisc('whizby', variant, sr);
    case 'explosion': return renderMisc('explosion', variant, sr);
  }
}
