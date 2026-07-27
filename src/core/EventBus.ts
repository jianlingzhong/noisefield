import type { EventHandler, EventName, GameEvents, IEventBus } from './Types';

/**
 * Synchronous typed pub/sub.
 *
 * Handlers are copied before dispatch so a handler may safely subscribe or
 * unsubscribe during its own emit. A throwing handler is isolated: it is logged
 * and the remaining handlers still run, because a VFX bug must never take down
 * the simulation.
 */
export class EventBus implements IEventBus {
  private readonly handlers = new Map<EventName, Set<(p: unknown) => void>>();
  private depth = 0;

  on<K extends EventName>(event: K, handler: EventHandler<K>): () => void {
    let set = this.handlers.get(event);
    if (!set) {
      set = new Set();
      this.handlers.set(event, set);
    }
    set.add(handler as (p: unknown) => void);
    return () => this.off(event, handler);
  }

  once<K extends EventName>(event: K, handler: EventHandler<K>): () => void {
    const wrapped = (payload: GameEvents[K]) => {
      this.off(event, wrapped);
      handler(payload);
    };
    return this.on(event, wrapped);
  }

  off<K extends EventName>(event: K, handler: EventHandler<K>): void {
    this.handlers.get(event)?.delete(handler as (p: unknown) => void);
  }

  emit<K extends EventName>(event: K, payload: GameEvents[K]): void {
    const set = this.handlers.get(event);
    if (!set || set.size === 0) return;

    // Guard against runaway recursion (a handler that re-emits its own event).
    if (this.depth > 16) {
      console.warn(`[EventBus] recursion depth exceeded on "${event}"`);
      return;
    }

    this.depth++;
    // Snapshot: handlers may mutate the set during dispatch.
    const snapshot = Array.from(set);
    for (const handler of snapshot) {
      try {
        handler(payload);
      } catch (err) {
        console.error(`[EventBus] handler for "${event}" threw:`, err);
      }
    }
    this.depth--;
  }

  clear(): void {
    this.handlers.clear();
  }
}
