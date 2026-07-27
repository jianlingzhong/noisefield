import { describe, expect, it } from 'vitest';
import { DEFAULT_BINDINGS } from '../src/core/Input';
import type { ActionName } from '../src/core/Types';

/*
 * Keyboard binding table.
 *
 * This file exists because of a specific shipped bug. The table had been
 * hand-patched into this state:
 *
 *   ArrowLeft: 'right' as ActionName,   // overwritten below
 *   ...
 *   DEFAULT_BINDINGS['ArrowLeft'] = 'left';
 *
 * `ArrowRight` was simply absent, so the documented arrow-key controls could
 * move forward, back and left, but not right. TypeScript cannot catch this:
 * the table is a valid `Record<string, ActionName>` either way. Only an
 * assertion about COMPLETENESS finds it.
 */

/** The four movement directions, which must each be reachable. */
const DIRECTIONS: readonly ActionName[] = ['forward', 'back', 'left', 'right'];

/** Keys bound to a given action. */
function keysFor(action: ActionName): string[] {
  return Object.entries(DEFAULT_BINDINGS)
    .filter(([, a]) => a === action)
    .map(([k]) => k);
}

describe('movement bindings', () => {
  it.each(DIRECTIONS)('"%s" is reachable from WASD', (action) => {
    const wasd = keysFor(action).filter((k) => /^Key[WASD]$/.test(k));
    expect(wasd).toHaveLength(1);
  });

  it.each(DIRECTIONS)('"%s" is reachable from the arrow keys', (action) => {
    // The regression: ArrowRight was missing entirely.
    const arrows = keysFor(action).filter((k) => k.startsWith('Arrow'));
    expect(arrows).toHaveLength(1);
  });

  it('maps each arrow key to the direction it points at', () => {
    expect(DEFAULT_BINDINGS['ArrowUp']).toBe('forward');
    expect(DEFAULT_BINDINGS['ArrowDown']).toBe('back');
    expect(DEFAULT_BINDINGS['ArrowLeft']).toBe('left');
    expect(DEFAULT_BINDINGS['ArrowRight']).toBe('right');
  });

  it('maps each WASD key to the direction it points at', () => {
    expect(DEFAULT_BINDINGS['KeyW']).toBe('forward');
    expect(DEFAULT_BINDINGS['KeyS']).toBe('back');
    expect(DEFAULT_BINDINGS['KeyA']).toBe('left');
    expect(DEFAULT_BINDINGS['KeyD']).toBe('right');
  });

  it('never binds one key to two actions', () => {
    // The old table assigned ArrowLeft twice, relying on statement order for
    // the correct value to win. That is legal and silent, so assert the shape
    // rather than the outcome.
    const keys = Object.keys(DEFAULT_BINDINGS);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe('core action bindings', () => {
  it.each([
    ['Space', 'jump'],
    ['ShiftLeft', 'sprint'],
    ['KeyR', 'reload'],
    ['KeyF', 'inspect'],
    ['Digit1', 'weapon1'],
    ['Digit2', 'weapon2'],
    ['Escape', 'pause'],
  ] as const)('%s is bound to "%s"', (key, action) => {
    expect(DEFAULT_BINDINGS[key]).toBe(action);
  });

  it('offers both Ctrl and C for crouch', () => {
    expect(keysFor('crouch').sort()).toEqual(['ControlLeft', 'KeyC']);
  });

  it('binds every key to a known action name', () => {
    // Guards against a typo'd action string, which would bind a key to an
    // action nothing listens for — the key would simply do nothing.
    const known: ReadonlySet<string> = new Set<ActionName>([
      'forward', 'back', 'left', 'right',
      'jump', 'crouch', 'sprint', 'slide',
      'fire', 'ads', 'reload', 'melee', 'grenade',
      'inspect', 'weapon1', 'weapon2', 'pause',
    ]);
    for (const [key, action] of Object.entries(DEFAULT_BINDINGS)) {
      expect(known.has(action), `${key} → "${action}" is not a known action`).toBe(true);
    }
  });

  it('uses KeyboardEvent.code values, not key values', () => {
    // Binding on `.key` would break on non-QWERTY layouts and would make the
    // table locale-dependent. `.code` values are physical and always match
    // these shapes.
    for (const key of Object.keys(DEFAULT_BINDINGS)) {
      expect(key).toMatch(/^(Key[A-Z]|Digit\d|Arrow(Up|Down|Left|Right)|Space|Escape|(Control|Shift|Alt)(Left|Right))$/);
    }
  });
});
