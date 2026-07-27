/**
 * NOISEFIELD — material atlas harness (development only).
 *
 * Bakes the real `MaterialLibrary` and renders every surface twice: once as a
 * lit 3x3-tile quad under a single low sun, and once as its raw albedo /
 * normal / ORM maps.
 *
 * See `atlas.html` for why this exists. In short: a tiling artefact is obvious
 * on a flat 3x3 layout and nearly invisible in a game frame, and most of the
 * 17 surfaces are unreachable from the level at all, so this is the only place
 * they can be reviewed.
 *
 * The library is fed a hand-built `EngineContext`. Only `renderer` and
 * `settings` are ever read by `MaterialLibrary.init`, but the object has to
 * satisfy the whole interface, so the rest are minimal real instances rather
 * than casts — a cast here would silently rot the moment the library started
 * reading another field.
 */

import * as THREE from 'three';

import type {
  EngineContext,
  IEventBus,
  IInput,
  QualitySettings,
  QualityTier,
  System,
} from '../core/Types';
import { DEFAULT_SETTINGS } from '../core/Types';
import { MaterialLibrary } from './MaterialLibrary';

const PREVIEW = 256;
const MAP = 84;

function qs(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

function num(name: string, fallback: number): number {
  const raw = qs(name);
  const v = raw === null ? NaN : Number(raw);
  return Number.isFinite(v) ? v : fallback;
}

/** Minimal stand-ins so the context is structurally complete, not cast. */
const noopBus: IEventBus = {
  on: () => () => { /* no subscriptions in the harness */ },
  once: () => () => { /* no subscriptions in the harness */ },
  off: () => { /* no subscriptions in the harness */ },
  emit: () => { /* nothing listens */ },
};

const noopInput: IInput = {
  isDown: () => false,
  pressed: () => false,
  released: () => false,
  lookDelta: { x: 0, y: 0 },
  pointerLocked: false,
  requestPointerLock: () => { /* headless */ },
  endFrame: () => { /* headless */ },
};

async function main(): Promise<void> {
  const status = document.getElementById('status');
  const grid = document.getElementById('grid');
  if (!status || !grid) return;

  const tier = (qs('tier') ?? 'ultra') as QualityTier;
  const tiles = num('tiles', 3);
  const sunDeg = num('rot', 25);
  const only = qs('id');

  // One shared offscreen renderer for the bake AND for every preview: the
  // library's textures are GPU render targets, so they belong to exactly one
  // context and cannot be sampled by a second renderer.
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setSize(PREVIEW, PREVIEW, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const settings: QualitySettings = { ...DEFAULT_SETTINGS, tier, anisotropy: 16 };
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);

  const ctx: EngineContext = {
    renderer,
    scene,
    camera,
    viewScene: new THREE.Scene(),
    viewCamera: new THREE.PerspectiveCamera(45, 1, 0.1, 100),
    bus: noopBus,
    input: noopInput,
    settings,
    width: PREVIEW,
    height: PREVIEW,
    get: <T extends System>(): T | undefined => undefined,
  };

  const lib = new MaterialLibrary();
  const t0 = performance.now();
  await lib.init(ctx);
  await lib.whenReady();
  const bakeMs = (performance.now() - t0).toFixed(0);

  // ── preview scene ──────────────────────────────────────────────────────────
  // A single low sun plus a weak sky fill. Deliberately harsh and deliberately
  // low: relief and roughness variation only reveal themselves at a grazing
  // incidence, so a soft studio setup would flatter every surface equally and
  // tell us nothing.
  const sun = new THREE.DirectionalLight(0xfff0dd, 3.1);
  const el = (sunDeg * Math.PI) / 180;
  sun.position.set(Math.cos(el) * -6, Math.sin(el) * 6, 3);
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0x9dbbe0, 0x4a4034, 0.55));

  // A plain plane, UV-scaled so the texture repeats `tiles` times. Triplanar
  // materials ignore these UVs and project from world space instead, so the
  // plane is also sized in world units to match what they expect.
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1, 1, 1), lib.get('concrete'));
  plane.rotation.x = -Math.PI / 2;
  scene.add(plane);
  camera.position.set(0, 0.72, 0.78);
  camera.lookAt(0, 0, 0);

  const ids = only ? [only] : lib.ids();
  status.textContent =
    `${lib.ids().length} surfaces · ${(lib.vramBytes / 1048576).toFixed(1)} MB · baked in ${bakeMs} ms · `
    + `tier=${tier} · ${tiles} tiles across · sun ${sunDeg}deg`;

  for (const id of ids) {
    const material = lib.get(id);
    const tileM = lib.tileMeters(id);
    const desc = lib.descriptor(id);
    const set = lib.textures(id);

    // World size = tiles * tileMeters, and UVs run 0..tiles. Both projections
    // therefore show the same number of repeats.
    const world = tiles * tileM;
    plane.geometry.dispose();
    const geo = new THREE.PlaneGeometry(world, world, 1, 1);
    const uv = geo.getAttribute('uv') as THREE.BufferAttribute;
    for (let i = 0; i < uv.count; i++) {
      uv.setXY(i, uv.getX(i) * tiles, uv.getY(i) * tiles);
    }
    uv.needsUpdate = true;
    geo.computeTangents();
    plane.geometry = geo;
    plane.material = material;

    // Frame the quad from a low angle so relief throws shadow across it.
    camera.position.set(0, world * 0.58, world * 0.62);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    renderer.setSize(PREVIEW, PREVIEW, false);
    renderer.render(scene, camera);

    const card = document.createElement('div');
    card.className = 'card';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = id;
    card.appendChild(name);

    const row = document.createElement('div');
    row.className = 'row';
    row.appendChild(copyCanvas(renderer.domElement, PREVIEW));

    const maps = document.createElement('div');
    maps.className = 'maps';
    if (set) {
      maps.appendChild(labelled('albedo', renderTexture(renderer, set.map, MAP, false)));
      maps.appendChild(labelled('normal', renderTexture(renderer, set.normalMap, MAP, true)));
      maps.appendChild(labelled('ORM', renderTexture(renderer, set.roughnessMap, MAP, true)));
    }
    row.appendChild(maps);
    card.appendChild(row);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${desc?.surface ?? '?'} · tile ${tileM.toFixed(2)} m · `
      + `${material.userData['triplanar'] === true ? 'triplanar' : 'uv'}`;
    card.appendChild(meta);
    grid.appendChild(card);

    // Yield so the page paints progressively instead of hanging for a second.
    await new Promise((r) => { setTimeout(r, 0); });
  }
}

/** Snapshot the shared renderer's canvas; it is reused for the next surface. */
function copyCanvas(src: HTMLCanvasElement, size: number): HTMLCanvasElement {
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const g = out.getContext('2d');
  if (g) g.drawImage(src, 0, 0, size, size);
  return out;
}

/**
 * Blit a baked texture through a trivial unlit quad.
 *
 * These are render-target textures with no CPU-side image, so `drawImage` on
 * them is not an option — the only way to see them is to render them.
 */
function renderTexture(
  renderer: THREE.WebGLRenderer,
  tex: THREE.Texture | undefined,
  size: number,
  raw: boolean,
): HTMLCanvasElement {
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  if (!tex) return out;

  const scene = new THREE.Scene();
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const mat = new THREE.ShaderMaterial({
    vertexShader: 'varying vec2 v; void main(){ v = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }',
    // `raw` maps (normal, ORM) are data: show the stored bytes verbatim. Albedo
    // is sRGB-encoded in hardware, so it needs the inverse transfer to look
    // like what the artist authored.
    fragmentShader: raw
      ? 'varying vec2 v; uniform sampler2D t; void main(){ gl_FragColor = vec4( texture2D( t, v ).rgb, 1.0 ); }'
      : 'varying vec2 v; uniform sampler2D t; void main(){ vec3 c = texture2D( t, v ).rgb; '
        + 'gl_FragColor = vec4( pow( c, vec3( 1.0 / 2.2 ) ), 1.0 ); }',
    uniforms: { t: { value: tex } },
    depthTest: false,
    depthWrite: false,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
  quad.frustumCulled = false;
  scene.add(quad);

  renderer.setSize(size, size, false);
  renderer.render(scene, cam);
  const g = out.getContext('2d');
  if (g) g.drawImage(renderer.domElement, 0, 0, size, size);

  quad.geometry.dispose();
  mat.dispose();
  return out;
}

function labelled(text: string, canvas: HTMLCanvasElement): HTMLElement {
  const wrap = document.createElement('div');
  const lbl = document.createElement('div');
  lbl.className = 'lbl';
  lbl.textContent = text;
  wrap.appendChild(lbl);
  wrap.appendChild(canvas);
  return wrap;
}

void main();
