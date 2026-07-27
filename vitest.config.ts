import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // happy-dom rather than node: several modules guard on `typeof document`
    // and take a different path in a browser-like environment, and it is that
    // path the game actually runs, so it is the one worth testing.
    environment: 'happy-dom',
    include: ['tests/**/*.test.ts'],
    // These are pure-logic and geometry tests; none should be slow. A test that
    // takes longer than this has almost certainly started doing real work by
    // accident (baking textures, for instance).
    testTimeout: 20_000,
  },
});
