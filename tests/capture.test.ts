import { describe, expect, it } from 'vitest';
import { PRESETS, WEAPON_IDS, parseVec } from '../src/core/Capture';
import { WEAPON_ORDER } from '../src/weapons/Specs';

/*
 * The capture harness parses untrusted query-string input, and the security
 * policy claims every parameter is either coerced to a number or matched
 * against a fixed allowlist. That claim was briefly untrue: `?weapon=` was an
 * unchecked `as WeaponId` cast, so any string reached the weapon system as a
 * key. These tests hold the claim to account.
 */

describe('parseVec', () => {
  it('parses three comma-separated numbers', () => {
    expect(parseVec('1,2,3')).toEqual([1, 2, 3]);
  });

  it('accepts negatives, decimals and surrounding whitespace', () => {
    expect(parseVec(' -14 , 1.7 ,-16 ')).toEqual([-14, 1.7, -16]);
  });

  it.each([
    ['null input', null],
    ['empty string', ''],
    ['too few components', '1,2'],
    ['too many components', '1,2,3,4'],
    ['non-numeric', 'a,b,c'],
    ['partially non-numeric', '1,two,3'],
    ['injection-ish', '1,2,3);drop'],
  ])('rejects %s', (_label, input) => {
    expect(parseVec(input)).toBeNull();
  });

  it('rejects non-finite values rather than propagating them', () => {
    // NaN or Infinity in a camera position produces an all-black frame and a
    // corrupted projection matrix, which is far harder to diagnose than a
    // rejected parameter.
    expect(parseVec('Infinity,0,0')).toBeNull();
    expect(parseVec('NaN,0,0')).toBeNull();
  });
});

describe('weapon parameter allowlist', () => {
  it('matches the real weapon roster exactly', () => {
    // Prevents drift: adding a weapon to Specs without adding it here would
    // silently make `?weapon=<new>` fall back to the preset default.
    expect([...WEAPON_IDS].sort()).toEqual([...WEAPON_ORDER].sort());
  });

  it.each(WEAPON_IDS)('accepts the real weapon id "%s"', (id) => {
    expect((WEAPON_IDS as readonly string[]).includes(id)).toBe(true);
  });

  it.each([
    'nonsense',
    '',
    'AR_VECTOR',
    'ar_vector ',
    '__proto__',
    'constructor',
    'toString',
  ])('rejects "%s"', (bad) => {
    expect((WEAPON_IDS as readonly string[]).includes(bad)).toBe(false);
  });

  it('is not fooled by inherited object properties', () => {
    // A naive `allowlist[value] !== undefined` check would accept
    // "constructor" and "__proto__". Array.includes does not, and this asserts
    // the property rather than trusting the current implementation shape.
    for (const bad of ['__proto__', 'constructor', 'hasOwnProperty']) {
      expect((WEAPON_IDS as readonly string[]).includes(bad)).toBe(false);
    }
  });
});

describe('shot presets', () => {
  const entries = Object.entries(PRESETS);

  it('exposes a reasonable number of presets', () => {
    expect(entries.length).toBeGreaterThanOrEqual(10);
  });

  it.each(entries)('preset "%s" is internally consistent', (key, preset) => {
    // The record key and the preset's own name must agree, or `?shot=x` logs
    // one name and renders another.
    expect(preset.name).toBe(key);
    expect(preset.description.length).toBeGreaterThan(10);
  });

  it.each(entries)('preset "%s" has a finite camera pose', (_key, preset) => {
    for (const v of [...preset.pos, ...preset.look]) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it.each(entries)('preset "%s" does not look at its own eye position', (_key, preset) => {
    // A zero-length view direction yields a NaN rotation and a black frame.
    const [px, py, pz] = preset.pos;
    const [lx, ly, lz] = preset.look;
    const d2 = (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2;
    expect(d2).toBeGreaterThan(1e-6);
  });

  it.each(entries)('preset "%s" names only a real weapon', (_key, preset) => {
    if (preset.weapon === undefined) return;
    expect((WEAPON_IDS as readonly string[]).includes(preset.weapon)).toBe(true);
  });

  it.each(entries)('preset "%s" requests a sane frame count', (_key, preset) => {
    if (preset.frames === undefined) return;
    expect(preset.frames).toBeGreaterThan(0);
    // Each frame is stepped synchronously; a huge count hangs the capture.
    expect(preset.frames).toBeLessThanOrEqual(600);
  });
});
