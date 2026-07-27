/// <reference types="vite/client" />

// Vite resolves side-effect CSS imports at build time; TypeScript needs to be
// told they are valid modules or `import './hud.css'` is a TS2882 error.
declare module '*.css';
declare module '*.scss';
