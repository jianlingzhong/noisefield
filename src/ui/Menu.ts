/**
 * NOISEFIELD — Pause menu and live settings.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW PAUSE IS TRIGGERED (three converging paths, one debounced entry point)
 * ═══════════════════════════════════════════════════════════════════════════
 * Relying on a single trigger here is fragile, for a reason specific to
 * pointer-locked browser games:
 *
 *   1. `input.pressed('pause')` — the clean path. But `Input` clears its
 *      entire held-action set on `pointerlockchange`, and pressing Escape
 *      *while pointer-locked* makes the browser drop the lock. The ordering of
 *      the Escape `keydown` against the `pointerlockchange` task is not
 *      specified, so this edge can legitimately be swallowed before the next
 *      frame reads it. It cannot be the only path.
 *
 *   2. A private `keydown` listener for Escape. Immediate, synchronous, and
 *      unaffected by the lock state. This is the reliable one.
 *
 *   3. Loss of pointer lock after having held it. Every browser FPS opens its
 *      menu here, because losing the lock means the cursor is loose on screen
 *      and the player is no longer really playing.
 *
 * All three funnel into one debounced toggle, so any two firing together still
 * results in exactly one state change.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW PAUSE IS COMMUNICATED
 * ═══════════════════════════════════════════════════════════════════════════
 * `Engine.setPaused` emits `game:pause`, and `EngineContext` deliberately does
 * not expose the engine. This module therefore treats `game:pause` as a
 * bidirectional channel: it announces its own state changes on it and accepts
 * external ones idempotently. Wiring `bus.on('game:pause', p =>
 * engine.setPaused(p.paused))` in the composition root makes the simulation
 * actually stop; without that line the menu still opens and settings still
 * apply, the world just keeps running behind the glass. Degrading rather than
 * breaking is the point — a UI module must never be able to wedge the game.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * SETTINGS
 * ═══════════════════════════════════════════════════════════════════════════
 * `ctx.settings` is the single source of truth and every control writes
 * straight into it. What makes a change *visible* differs per setting:
 *   • sensitivity  — `Input.beginFrame` reads it every frame. Instant.
 *   • fov          — the weapon system owns `camera.fov` and rebuilds it from
 *                    `settings.fov` every frame, so writing the setting is
 *                    both sufficient and the only correct move. Writing
 *                    `camera.fov` directly would be overwritten next frame.
 *   • masterVolume — the audio graph tracks it every frame. Instant.
 *   • renderScale  — `Engine.resize` reads it, so a synthetic `resize` event
 *                    applies it. This is the convention already used by the
 *                    adaptive scaler in the composition root.
 *   • shadows      — applied at the renderer, which is a renderer-level
 *                    concern and safe to drive from here.
 *   • ssao/bloom/motionBlur — baked into the post-processing chain when it is
 *                    constructed. An optional `applySettings()` hook is probed
 *                    on the relevant systems; where it is absent the value is
 *                    still stored and the row says so rather than lying.
 */

import type {
  EngineContext,
  FrameContext,
  QualitySettings,
  QualityTier,
  System,
} from '../core/Types';
import './hud.css';

const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls !== undefined) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Systems may optionally expose this to be told that quality settings moved.
 * Probed structurally so no other system has to implement it, and so adding it
 * later needs no change here.
 */
type SettingsAware = System & { applySettings?: (ctx: EngineContext) => void };

/** Systems worth notifying, in the order a rebuild would want them. */
const SETTINGS_CONSUMERS = ['pipeline', 'lighting', 'sky', 'audio', 'particles', 'decals'] as const;

/**
 * Tier presets. Mirrors the shape of the boot-time auto-detect in the
 * composition root so a manual choice and an automatic one produce comparable
 * configurations. renderScale and shadows are included specifically because
 * they are the two knobs that apply *without* a post-processing rebuild —
 * which is what makes a tier change visible immediately rather than on reload.
 */
const TIER_PRESETS: Readonly<Record<QualityTier, Partial<QualitySettings>>> = {
  low: {
    renderScale: 0.65, shadows: false, shadowMapSize: 1024, shadowCascades: 2,
    ssao: false, ssaoQuality: 'low', bloom: false, motionBlur: false, volumetrics: false,
    antialias: 'fxaa', anisotropy: 4, particleBudget: 4000, decalBudget: 64,
  },
  medium: {
    renderScale: 0.82, shadows: true, shadowMapSize: 1024, shadowCascades: 2,
    ssao: false, ssaoQuality: 'low', bloom: true, motionBlur: false, volumetrics: false,
    antialias: 'fxaa', anisotropy: 8, particleBudget: 8000, decalBudget: 128,
  },
  high: {
    renderScale: 0.9, shadows: true, shadowMapSize: 1536, shadowCascades: 3,
    ssao: true, ssaoQuality: 'medium', bloom: true, motionBlur: true, volumetrics: true,
    antialias: 'smaa', anisotropy: 8, particleBudget: 14000, decalBudget: 192,
  },
  ultra: {
    renderScale: 1.0, shadows: true, shadowMapSize: 2048, shadowCascades: 3,
    ssao: true, ssaoQuality: 'high', bloom: true, motionBlur: true, volumetrics: true,
    antialias: 'taa', anisotropy: 16, particleBudget: 20000, decalBudget: 256,
  },
};

const TIERS: readonly QualityTier[] = ['low', 'medium', 'high', 'ultra'];

export class Menu implements System {
  readonly name = 'menu';

  private ctx: EngineContext | null = null;
  private root: HTMLElement | null = null;
  private panel: HTMLElement | null = null;
  private settingsPane: HTMLElement | null = null;
  private restartBtn: HTMLButtonElement | null = null;

  private open = false;
  private pane: 'main' | 'settings' = 'main';

  /** Collapses simultaneous triggers into one toggle. */
  private toggleLock = 0;
  /** Two-step confirm on Restart; seconds remaining before it reverts. */
  private restartArmed = 0;

  private hadPointerLock = false;
  /** rAF handle coalescing repeated applyRender calls into one per frame. */
  private applyHandle = 0;
  private readonly disposers: Array<() => void> = [];
  /** Re-reads every control from `ctx.settings`; run whenever the pane opens. */
  private readonly syncers: Array<() => void> = [];

  // ═══════════════════════════════════════════════════════════════════════════
  // Lifecycle
  // ═══════════════════════════════════════════════════════════════════════════

  init(ctx: EngineContext): void {
    this.ctx = ctx;

    const mount = document.getElementById('ui-root');
    if (mount === null) {
      console.warn('[Menu] #ui-root not found — pause menu disabled.');
      return;
    }

    const root = h('div', 'nf-menu');
    root.setAttribute('data-pane', 'main');
    this.root = root;

    const scrim = h('div', 'nf-menu__scrim');
    root.appendChild(scrim);

    this.panel = this.buildPanel();
    root.appendChild(this.panel);

    this.settingsPane = this.buildSettings(ctx);
    root.appendChild(this.settingsPane);

    mount.appendChild(root);

    // ── Triggers (see the class comment for why there are three) ────────────
    const onKeyDown = (e: KeyboardEvent): void => {
      if (e.code === 'Escape') {
        e.preventDefault();
        // Escape inside the settings pane steps back rather than closing
        // everything: losing a whole menu because you wanted to leave one
        // sub-page is a small, constant, avoidable irritation.
        if (this.open && this.pane === 'settings') {
          this.showPane('main');
          this.toggleLock = 0.2;
          return;
        }
        this.requestToggle();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    this.disposers.push(() => window.removeEventListener('keydown', onKeyDown));

    const onLockChange = (): void => {
      const locked = document.pointerLockElement !== null;
      if (locked) this.hadPointerLock = true;
      else if (this.hadPointerLock && !this.open) this.requestToggle();
    };
    document.addEventListener('pointerlockchange', onLockChange);
    this.disposers.push(() => document.removeEventListener('pointerlockchange', onLockChange));

    this.disposers.push(
      ctx.bus.on('game:pause', (p) => {
        // External announcement — adopt it without re-announcing, which is
        // what keeps the engine↔menu round trip from looping.
        if (p.paused !== this.open) this.setOpen(p.paused, false);
      }),
    );

    // Swallow pointer events so a click on the menu can never reach the
    // canvas listener that re-acquires pointer lock.
    for (const type of ['pointerdown', 'pointerup', 'click', 'wheel', 'contextmenu'] as const) {
      const stop = (e: Event): void => {
        e.stopPropagation();
        if (type === 'contextmenu') e.preventDefault();
      };
      root.addEventListener(type, stop);
      this.disposers.push(() => root.removeEventListener(type, stop));
    }
  }

  // ── Panel ────────────────────────────────────────────────────────────────

  private buildPanel(): HTMLElement {
    const panel = h('div', 'nf-menu__panel');
    panel.append(
      h('div', 'nf-menu__brand', 'Paused'),
      h('div', 'nf-menu__kicker', 'Noisefield'),
    );

    const nav = h('div', 'nf-menu__nav');
    const mk = (label: string, cls: string, fn: () => void): HTMLButtonElement => {
      const b = h('button', `ap-menu__btn${cls}`, label);
      b.type = 'button';
      b.addEventListener('click', fn);
      nav.appendChild(b);
      return b;
    };

    mk('Resume', '', () => {
      this.setOpen(false, true);
      // Only re-acquire the lock from an explicit click. Requesting it right
      // after an Escape-driven close is rejected by Chrome's re-entry
      // cooldown and just logs an error the player cannot act on.
      this.ctx?.input.requestPointerLock();
    });
    mk('Settings', '', () => this.showPane(this.pane === 'settings' ? 'main' : 'settings'));
    this.restartBtn = mk('Restart', ' ap-menu__btn--danger', () => this.onRestart());

    panel.appendChild(nav);
    panel.appendChild(
      h(
        'div',
        'nf-menu__hint',
        'ESC resume · WASD move · SHIFT sprint · CTRL crouch · SPACE jump\nLMB fire · RMB aim · R reload · 1/2 weapons',
      ),
    );
    return panel;
  }

  private showPane(p: 'main' | 'settings'): void {
    this.pane = p;
    this.root?.setAttribute('data-pane', p);
    if (p === 'settings') this.syncAll();
  }

  private onRestart(): void {
    const btn = this.restartBtn;
    if (btn === null) return;
    if (this.restartArmed > 0) {
      // A full reload is the only honest "restart" available: there is no
      // reset event in the shared contract, and silently half-resetting a few
      // systems would leave the world in a state no one designed for.
      window.location.reload();
      return;
    }
    this.restartArmed = 4;
    btn.textContent = 'Confirm restart?';
  }

  // ── Settings ─────────────────────────────────────────────────────────────

  private buildSettings(ctx: EngineContext): HTMLElement {
    const pane = h('div', 'nf-set');
    const grid = h('div', 'nf-set__grid');
    pane.appendChild(grid);
    const s = ctx.settings;

    grid.appendChild(h('div', 'nf-set__head', 'Controls'));
    // Presented ×1000 so the number has a human-sized range; the underlying
    // unit is radians per pixel of mouse travel.
    this.slider(grid, 'Sensitivity', 0.0004, 0.008, 0.0001,
      () => s.sensitivity,
      (v) => { s.sensitivity = v; },
      (v) => (v * 1000).toFixed(2));

    grid.appendChild(h('div', 'nf-set__head', 'View'));
    this.slider(grid, 'Field of view', 60, 120, 1,
      () => s.fov,
      (v) => { s.fov = Math.round(v); },
      (v) => `${Math.round(v)}°`);

    grid.appendChild(h('div', 'nf-set__head', 'Audio'));
    this.slider(grid, 'Master volume', 0, 1, 0.01,
      () => s.masterVolume,
      (v) => { s.masterVolume = v; },
      (v) => `${Math.round(v * 100)}%`);

    grid.appendChild(h('div', 'nf-set__head', 'Graphics'));
    this.segmented(grid, 'Quality tier', TIERS,
      () => s.tier,
      (v) => this.applyTier(v),
      'Presets resolution, shadows and post-processing together.');

    this.slider(grid, 'Render scale', 0.5, 1, 0.05,
      () => s.renderScale,
      (v) => { s.renderScale = v; },
      (v) => `${Math.round(v * 100)}%`,
      'Adaptive scaling may move this again while the frame time is high.');

    this.toggle(grid, 'Shadows', () => s.shadows, (v) => { s.shadows = v; });
    this.toggle(grid, 'Ambient occlusion', () => s.ssao, (v) => { s.ssao = v; });
    this.toggle(grid, 'Bloom', () => s.bloom, (v) => { s.bloom = v; });
    this.toggle(grid, 'Motion blur', () => s.motionBlur, (v) => { s.motionBlur = v; });
    this.toggle(grid, 'Volumetric light', () => s.volumetrics, (v) => { s.volumetrics = v; });

    grid.appendChild(
      h(
        'div',
        'nf-set__note',
        'Resolution, shadows, field of view, sensitivity and volume apply instantly. ' +
          'Post-processing effects are compiled into the render chain and take hold ' +
          'when it is next rebuilt.',
      ),
    );
    return pane;
  }

  private row(grid: HTMLElement, label: string, control: HTMLElement, note?: string): HTMLElement {
    const row = h('div', 'nf-set__row');
    row.append(h('div', 'nf-set__label', label), control);
    const value = h('div', 'nf-set__value');
    row.appendChild(value);
    if (note !== undefined) row.appendChild(h('div', 'nf-set__note', note));
    grid.appendChild(row);
    return value;
  }

  private slider(
    grid: HTMLElement,
    label: string,
    min: number,
    max: number,
    step: number,
    get: () => number,
    set: (v: number) => void,
    fmt: (v: number) => string,
    note?: string,
  ): void {
    const input = h('input', 'nf-set__range');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    const value = this.row(grid, label, input, note);

    const paint = (v: number): void => {
      value.textContent = fmt(v);
      // The track gradient is driven by a custom property so the fill can be
      // pure paint; no layout is involved in dragging a slider.
      input.style.setProperty('--fill', `${((v - min) / (max - min)) * 100}%`);
    };

    input.addEventListener('input', () => {
      const v = clamp(Number(input.value), min, max);
      if (!Number.isFinite(v)) return;
      set(v);
      paint(v);
      this.requestApply();
    });

    const sync = (): void => {
      // Snap to the control's own grid before painting. A range input silently
      // quantises whatever it is given, so an off-grid value written by another
      // system (the adaptive render scaler lands on 0.82, which is not a
      // multiple of the 0.05 step) left the thumb and the numeric readout
      // disagreeing, with the number reporting a value the slider was not on.
      const raw = clamp(get(), min, max);
      const snapped = clamp(min + Math.round((raw - min) / step) * step, min, max);
      input.value = String(snapped);
      paint(Number(input.value));
    };
    sync();
    this.syncers.push(sync);
  }

  private toggle(
    grid: HTMLElement,
    label: string,
    get: () => boolean,
    set: (v: boolean) => void,
  ): void {
    const btn = h('button', 'nf-set__toggle');
    btn.type = 'button';
    const value = this.row(grid, label, btn);

    const paint = (v: boolean): void => {
      btn.setAttribute('aria-pressed', v ? 'true' : 'false');
      value.textContent = v ? 'ON' : 'OFF';
    };
    btn.addEventListener('click', () => {
      const v = !(btn.getAttribute('aria-pressed') === 'true');
      set(v);
      paint(v);
      this.requestApply();
    });

    const sync = (): void => paint(get());
    sync();
    this.syncers.push(sync);
  }

  private segmented(
    grid: HTMLElement,
    label: string,
    options: readonly QualityTier[],
    get: () => QualityTier,
    set: (v: QualityTier) => void,
    note?: string,
  ): void {
    const wrap = h('div', 'nf-set__seg');
    const buttons: HTMLButtonElement[] = [];
    const value = this.row(grid, label, wrap, note);

    const paint = (v: QualityTier): void => {
      for (let i = 0; i < buttons.length; i++) {
        buttons[i].setAttribute('aria-pressed', options[i] === v ? 'true' : 'false');
      }
      value.textContent = v.toUpperCase();
    };

    for (const opt of options) {
      const b = h('button', 'nf-set__segbtn', opt);
      b.type = 'button';
      b.addEventListener('click', () => {
        set(opt);
        paint(opt);
        // A tier writes half a dozen other rows; refresh them all so the pane
        // never shows a stale value next to the one just changed.
        this.syncAll();
      });
      buttons.push(b);
      wrap.appendChild(b);
    }

    paint(get());
    this.syncers.push(() => paint(get()));
  }

  private syncAll(): void {
    for (const fn of this.syncers) fn();
  }

  private applyTier(tier: QualityTier): void {
    const ctx = this.ctx;
    if (ctx === null) return;
    Object.assign(ctx.settings, TIER_PRESETS[tier], { tier });
    this.requestApply();
  }

  /**
   * Coalesce to at most one apply per frame.
   *
   * Dragging a slider fires `input` at pointer rate. The apply is not cheap —
   * it dispatches a synthetic global `resize`, which makes the engine resize
   * the drawing buffer and call `resize` on every registered system, and then
   * notifies every settings-aware system. Running that dozens of times per
   * second while a thumb is moving makes the drag itself stutter, which is a
   * uniquely bad experience for the control you are using to *fix* stutter.
   */
  private requestApply(): void {
    if (this.applyHandle !== 0) return;
    this.applyHandle = requestAnimationFrame(() => {
      this.applyHandle = 0;
      this.applyRender();
    });
  }

  /**
   * Push whatever can be pushed without rebuilding the render chain, then give
   * any system that wants one a chance to react.
   */
  private applyRender(): void {
    const ctx = this.ctx;
    if (ctx === null) return;
    const s = ctx.settings;

    try {
      const renderer = ctx.renderer;
      const wantShadows = s.shadows;
      if (renderer.shadowMap.enabled !== wantShadows) {
        renderer.shadowMap.enabled = wantShadows;
        // Toggling shadow support changes the shader permutation, so every
        // material has to be recompiled. It hitches — which is precisely why
        // doing it from a *paused* menu is the right place for it.
        ctx.scene.traverse((o) => {
          const mat = (o as { material?: unknown }).material;
          if (Array.isArray(mat)) {
            for (const m of mat) if (m && typeof m === 'object') (m as { needsUpdate: boolean }).needsUpdate = true;
          } else if (mat && typeof mat === 'object') {
            (mat as { needsUpdate: boolean }).needsUpdate = true;
          }
        });
        renderer.shadowMap.needsUpdate = true;
      }
    } catch {
      /* renderer state is best-effort; never let it break the menu */
    }

    // Applies renderScale (Engine.resize reads it) and re-lays out every
    // system's viewport-dependent state, including this HUD's canvases.
    try {
      window.dispatchEvent(new Event('resize'));
    } catch {
      /* ignore */
    }

    for (const nm of SETTINGS_CONSUMERS) {
      try {
        ctx.get<SettingsAware>(nm)?.applySettings?.(ctx);
      } catch {
        /* an optional hook that throws is the hook's problem, not the menu's */
      }
    }
  }

  // ── Open / close ─────────────────────────────────────────────────────────

  private requestToggle(): void {
    if (this.toggleLock > 0) return;
    this.toggleLock = 0.22;
    this.setOpen(!this.open, true);
  }

  private setOpen(v: boolean, announce: boolean): void {
    if (this.open === v || this.root === null) return;
    this.open = v;

    if (v) {
      this.root.setAttribute('data-open', '1');
      this.showPane('main');
      // A pause menu behind a hidden cursor is unusable.
      if (document.pointerLockElement !== null) document.exitPointerLock();
      // Focus the panel so the browser's own keyboard navigation lands
      // somewhere sensible instead of on the document body.
      this.panel?.querySelector<HTMLButtonElement>('.nf-menu__btn')?.focus();
    } else {
      this.root.removeAttribute('data-open');
      this.disarmRestart();
      // Blur so a stale focus ring cannot survive into gameplay and so a
      // later Space/Enter cannot re-activate a button that is not visible.
      const active = document.activeElement;
      if (active instanceof HTMLElement && this.root.contains(active)) active.blur();
    }

    if (announce) this.ctx?.bus.emit('game:pause', { paused: v });
  }

  private disarmRestart(): void {
    this.restartArmed = 0;
    if (this.restartBtn !== null) this.restartBtn.textContent = 'Restart';
  }

  get isOpen(): boolean {
    return this.open;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Frame
  // ═══════════════════════════════════════════════════════════════════════════

  update(ctx: EngineContext, frame: FrameContext): void {
    // `frame.dt` is zero while the engine is paused, which is exactly when the
    // menu is on screen. Menu timers therefore run off the unclamped wall
    // clock, bounded so an alt-tab cannot expire everything at once.
    const dt = Math.min(frame.rawDt, 0.1);

    if (this.toggleLock > 0) this.toggleLock -= dt;

    if (this.restartArmed > 0) {
      this.restartArmed -= dt;
      if (this.restartArmed <= 0) this.disarmRestart();
    }

    // Secondary trigger. Harmless when the private keydown listener already
    // handled this frame's Escape — `requestToggle` is debounced.
    if (ctx.input.pressed('pause')) this.requestToggle();
  }

  dispose(): void {
    for (const d of this.disposers) {
      try {
        d();
      } catch {
        /* ignore */
      }
    }
    if (this.applyHandle !== 0) {
      cancelAnimationFrame(this.applyHandle);
      this.applyHandle = 0;
    }
    this.disposers.length = 0;
    this.syncers.length = 0;
    this.root?.remove();
    this.root = null;
    this.panel = null;
    this.settingsPane = null;
    this.restartBtn = null;
    this.ctx = null;
  }
}
