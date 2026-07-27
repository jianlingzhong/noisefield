import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths.
  //
  // Vite defaults to `base: '/'`, which emits `<script src="/assets/…">`. That
  // is correct at a domain root but breaks on a GitHub Pages PROJECT site,
  // where the app is served from `https://<user>.github.io/<repo>/` — the
  // absolute path resolves to the user root and the bundle 404s, leaving a
  // black page with no error beyond a failed request.
  //
  // './' makes the build relocatable: it works at a domain root, under any
  // subpath, and from `vite preview` without further configuration.
  base: './',
  server: { host: '127.0.0.1', port: 5173, strictPort: false },
  preview: { host: '127.0.0.1', port: 4173, strictPort: false },
  build: {
    target: 'esnext',
    sourcemap: false,
    // The whole game is one bundle and there is nothing to code-split against:
    // every subsystem is needed before the first frame renders.
    chunkSizeWarningLimit: 4000,
  },
});
