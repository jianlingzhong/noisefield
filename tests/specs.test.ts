import { describe, expect, it } from 'vitest';
import {
  PRIMARY_POOL,
  SIDEARM,
  WEAPON_ORDER,
  WEAPON_SPECS,
  falloffAt,
  shotInterval,
} from '../src/weapons/Specs';
import type { WeaponId } from '../src/core/Types';

/*
 * Weapon data invariants.
 *
 * Balance numbers are a judgement call and not something a test should pin
 * down. What a test CAN pin down is that the data is coherent: no division by
 * zero, no weapon missing from the selection order, no falloff curve that
 * increases damage with distance. These are the failures that produce
 * NaN-ammo, unselectable weapons and unkillable enemies, and they are all
 * invisible to the type system.
 */

const ALL: readonly WeaponId[] = WEAPON_ORDER;

describe('roster integrity', () => {
  it('defines a spec for every weapon in the order', () => {
    for (const id of ALL) expect(WEAPON_SPECS[id]).toBeTruthy();
  });

  it('has no weapon in WEAPON_SPECS that is missing from the order', () => {
    // Drift in this direction produces a weapon that exists but can never be
    // selected, which is easy to ship and hard to notice.
    expect([...Object.keys(WEAPON_SPECS)].sort()).toEqual([...ALL].sort());
  });

  it('lists each weapon exactly once', () => {
    expect(new Set(ALL).size).toBe(ALL.length);
  });

  it.each(ALL)('spec "%s" carries its own id', (id) => {
    expect(WEAPON_SPECS[id].id).toBe(id);
  });

  it.each(ALL)('spec "%s" has a display name', (id) => {
    expect(WEAPON_SPECS[id].displayName.trim().length).toBeGreaterThan(0);
  });
});

describe('selection pools', () => {
  it('keeps the sidearm out of the primary pool', () => {
    // `1` cycles primaries and `2` draws the sidearm. If the sidearm were in
    // both, cycling would land on it and the two keys would fight.
    expect(PRIMARY_POOL).not.toContain(SIDEARM);
  });

  it('references only real weapons', () => {
    for (const id of PRIMARY_POOL) expect(ALL).toContain(id);
    expect(ALL).toContain(SIDEARM);
  });

  it('covers the whole roster between primaries and the sidearm', () => {
    expect([...PRIMARY_POOL, SIDEARM].sort()).toEqual([...ALL].sort());
  });

  it('has no duplicates in the primary pool', () => {
    // A duplicate would make the `1` cycle visit one weapon twice.
    expect(new Set(PRIMARY_POOL).size).toBe(PRIMARY_POOL.length);
  });
});

describe('firing and ammunition', () => {
  it.each(ALL)('"%s" has a positive rate of fire', (id) => {
    expect(WEAPON_SPECS[id].rpm).toBeGreaterThan(0);
  });

  it.each(ALL)('"%s" has a finite, positive shot interval', (id) => {
    // shotInterval divides by rpm; a zero would yield Infinity and the weapon
    // would either never fire or fire every frame.
    const dt = shotInterval(WEAPON_SPECS[id]);
    expect(Number.isFinite(dt)).toBe(true);
    expect(dt).toBeGreaterThan(0);
  });

  it.each(ALL)('"%s" has a usable magazine and reserve', (id) => {
    const s = WEAPON_SPECS[id];
    expect(s.magSize).toBeGreaterThan(0);
    expect(Number.isInteger(s.magSize)).toBe(true);
    expect(s.reserveAmmo).toBeGreaterThanOrEqual(0);
  });

  it.each(ALL)('"%s" reloads in a positive, plausible time', (id) => {
    const s = WEAPON_SPECS[id];
    expect(s.reloadTime).toBeGreaterThan(0);
    expect(s.reloadTime).toBeLessThan(10);
    expect(s.reloadTacticalTime).toBeGreaterThan(0);
    // Keeping a round chambered is faster than reloading from empty.
    expect(s.reloadTacticalTime).toBeLessThanOrEqual(s.reloadTime);
  });

  it.each(ALL)('"%s" aims down sights in a positive, plausible time', (id) => {
    const s = WEAPON_SPECS[id];
    expect(s.adsTime).toBeGreaterThan(0);
    expect(s.adsTime).toBeLessThan(2);
  });
});

describe('damage and falloff', () => {
  it.each(ALL)('"%s" deals positive damage', (id) => {
    expect(WEAPON_SPECS[id].damage).toBeGreaterThan(0);
  });

  it.each(ALL)('"%s" rewards a headshot', (id) => {
    expect(WEAPON_SPECS[id].headshotMultiplier).toBeGreaterThan(1);
  });

  it.each(ALL)('"%s" has a well-ordered falloff window', (id) => {
    const s = WEAPON_SPECS[id];
    expect(s.falloffStart).toBeGreaterThanOrEqual(0);
    // An end at or before the start makes the interpolation divide by ~zero.
    expect(s.falloffEnd).toBeGreaterThan(s.falloffStart);
    // Distance may reduce damage but must never increase it.
    expect(s.falloffMin).toBeGreaterThan(0);
    expect(s.falloffMin).toBeLessThanOrEqual(1);
  });

  it.each(ALL)('"%s" deals full damage inside the falloff start', (id) => {
    const s = WEAPON_SPECS[id];
    expect(falloffAt(s, 0)).toBe(1);
    expect(falloffAt(s, s.falloffStart)).toBe(1);
  });

  it.each(ALL)('"%s" floors at falloffMin beyond the falloff end', (id) => {
    const s = WEAPON_SPECS[id];
    expect(falloffAt(s, s.falloffEnd)).toBeCloseTo(s.falloffMin, 6);
    expect(falloffAt(s, s.falloffEnd * 10)).toBeCloseTo(s.falloffMin, 6);
  });

  it.each(ALL)('"%s" falloff never increases with distance', (id) => {
    const s = WEAPON_SPECS[id];
    let prev = Infinity;
    for (let d = 0; d <= s.falloffEnd * 1.5; d += s.falloffEnd / 40) {
      const v = falloffAt(s, d);
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeLessThanOrEqual(prev + 1e-9);
      prev = v;
    }
  });

  it.each(ALL)('"%s" handles a negative distance without going out of range', (id) => {
    // Distances come from a raycast and should never be negative, but a
    // multiplier above 1 would silently buff damage if one ever were.
    const v = falloffAt(WEAPON_SPECS[id], -5);
    expect(v).toBeLessThanOrEqual(1);
    expect(v).toBeGreaterThan(0);
  });
});

describe('accuracy', () => {
  it.each(ALL)('"%s" is more accurate aimed than at the hip', (id) => {
    const s = WEAPON_SPECS[id];
    expect(s.spreadAds).toBeLessThanOrEqual(s.spreadHip);
    expect(s.spreadAds).toBeGreaterThanOrEqual(0);
  });

  it.each(ALL)('"%s" recovers the spread it accumulates', (id) => {
    const s = WEAPON_SPECS[id];
    expect(s.spreadPerShot).toBeGreaterThanOrEqual(0);
    // Without decay, spread accumulated while firing would never recover and
    // the weapon would become permanently inaccurate for the whole match.
    if (s.spreadPerShot > 0) expect(s.spreadDecay).toBeGreaterThan(0);
  });

  it.each(ALL)('"%s" keeps spread within a sane cone', (id) => {
    // Half-angles in radians; anything past ~0.5 rad is a 57-degree cone.
    expect(WEAPON_SPECS[id].spreadHip).toBeLessThan(0.5);
  });
});
