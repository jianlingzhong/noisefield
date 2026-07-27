import type { ActionName, IInput, QualitySettings } from './Types';

/**
 * Physical key / button → logical action. Multiple bindings per action allowed.
 *
 * Exported so tests can assert the table is complete; a missing arrow key is
 * invisible at compile time and easy to miss by hand.
 */
export const DEFAULT_BINDINGS: Record<string, ActionName> = {
  KeyW: 'forward',
  ArrowUp: 'forward',
  KeyS: 'back',
  ArrowDown: 'back',
  KeyA: 'left',
  ArrowLeft: 'left',
  KeyD: 'right',
  ArrowRight: 'right',
  Space: 'jump',
  ControlLeft: 'crouch',
  KeyC: 'crouch',
  ShiftLeft: 'sprint',
  KeyR: 'reload',
  KeyV: 'melee',
  KeyG: 'grenade',
  KeyF: 'inspect',
  Digit1: 'weapon1',
  Digit2: 'weapon2',
  Escape: 'pause',
};

/**
 * Pointer-lock mouse + keyboard input with per-frame edge detection.
 *
 * Mouse deltas accumulate between frames and are converted to radians using the
 * sensitivity setting so consumers never deal with raw pixels. `endFrame` must
 * be called once per frame by the engine or `pressed`/`released` will latch.
 */
export class Input implements IInput {
  private readonly down = new Set<ActionName>();
  private readonly downPrev = new Set<ActionName>();
  private accumX = 0;
  private accumY = 0;
  private readonly delta = { x: 0, y: 0 };
  private locked = false;
  private readonly element: HTMLElement;
  private readonly settings: QualitySettings;
  private disposers: Array<() => void> = [];

  constructor(element: HTMLElement, settings: QualitySettings) {
    this.element = element;
    this.settings = settings;
    this.attach();
  }

  get lookDelta() {
    return this.delta;
  }

  get pointerLocked(): boolean {
    return this.locked;
  }

  isDown(action: ActionName): boolean {
    return this.down.has(action);
  }

  pressed(action: ActionName): boolean {
    return this.down.has(action) && !this.downPrev.has(action);
  }

  released(action: ActionName): boolean {
    return !this.down.has(action) && this.downPrev.has(action);
  }

  requestPointerLock(): void {
    // Chrome returns a promise here; swallow the rejection that occurs when the
    // user exits lock too quickly (the browser rate-limits re-entry).
    const r = this.element.requestPointerLock() as unknown as Promise<void> | undefined;
    if (r && typeof r.catch === 'function') r.catch(() => {});
  }

  /** Called by the engine at the very start of a frame to latch look input. */
  beginFrame(): void {
    this.delta.x = this.accumX * this.settings.sensitivity;
    this.delta.y = this.accumY * this.settings.sensitivity;
    this.accumX = 0;
    this.accumY = 0;
  }

  endFrame(): void {
    this.downPrev.clear();
    for (const a of this.down) this.downPrev.add(a);
  }

  private attach(): void {
    const onKey = (e: KeyboardEvent, isDown: boolean) => {
      const action = DEFAULT_BINDINGS[e.code];
      if (!action) return;
      // Never swallow F-keys or browser reload shortcuts.
      if (!e.metaKey && !e.ctrlKey) e.preventDefault();
      if (isDown) this.down.add(action);
      else this.down.delete(action);
    };

    const kd = (e: KeyboardEvent) => onKey(e, true);
    const ku = (e: KeyboardEvent) => onKey(e, false);

    const md = (e: MouseEvent) => {
      if (!this.locked) return;
      if (e.button === 0) this.down.add('fire');
      if (e.button === 2) this.down.add('ads');
      if (e.button === 1) this.down.add('melee');
    };
    const mu = (e: MouseEvent) => {
      if (e.button === 0) this.down.delete('fire');
      if (e.button === 2) this.down.delete('ads');
      if (e.button === 1) this.down.delete('melee');
    };
    const mm = (e: MouseEvent) => {
      if (!this.locked) return;
      // movementX/Y can spike enormously on some drivers when the pointer
      // re-enters; clamp to keep a bad frame from spinning the view.
      this.accumX += Math.max(-200, Math.min(200, e.movementX));
      this.accumY += Math.max(-200, Math.min(200, e.movementY));
    };
    const ctx = (e: Event) => e.preventDefault();
    const lockChange = () => {
      this.locked = document.pointerLockElement === this.element;
      if (!this.locked) {
        // Dropping lock must release held keys or the player keeps sprinting.
        this.down.clear();
        this.accumX = 0;
        this.accumY = 0;
      }
    };
    const blur = () => this.down.clear();

    window.addEventListener('keydown', kd);
    window.addEventListener('keyup', ku);
    window.addEventListener('mousedown', md);
    window.addEventListener('mouseup', mu);
    window.addEventListener('mousemove', mm);
    window.addEventListener('blur', blur);
    this.element.addEventListener('contextmenu', ctx);
    document.addEventListener('pointerlockchange', lockChange);

    this.disposers = [
      () => window.removeEventListener('keydown', kd),
      () => window.removeEventListener('keyup', ku),
      () => window.removeEventListener('mousedown', md),
      () => window.removeEventListener('mouseup', mu),
      () => window.removeEventListener('mousemove', mm),
      () => window.removeEventListener('blur', blur),
      () => this.element.removeEventListener('contextmenu', ctx),
      () => document.removeEventListener('pointerlockchange', lockChange),
    ];
  }

  /**
   * Test/automation hook: synthesise input without a real device. Used by the
   * headless capture harness to pose the game for screenshots.
   */
  __setAction(action: ActionName, isDown: boolean): void {
    if (isDown) this.down.add(action);
    else this.down.delete(action);
  }

  /** Test/automation hook: inject raw look delta in pixels. */
  __look(dx: number, dy: number): void {
    this.accumX += dx;
    this.accumY += dy;
  }

  dispose(): void {
    for (const d of this.disposers) d();
    this.disposers = [];
  }
}
