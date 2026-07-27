# Contributing

Thanks for taking an interest. This document covers how to get the project
running, the conventions the codebase follows, and what a good pull request
looks like here.

## Getting set up

Requires Node 20 or newer.

```bash
npm install
npm run dev          # http://127.0.0.1:5173
```

Before opening a pull request:

```bash
npm run check        # typecheck + tests + markdown lint
npm run build        # must pass
```

`npm run check` is what CI runs. If you are adding a guard against a bug you
just fixed, make sure the test actually FAILS with your fix reverted — a test
that cannot fail is worse than no test.

## The one rule that matters

**No external assets.** This project generates every texture, mesh and sound in
code at load time. A pull request that adds a `.png`, `.gltf`, `.fbx`, `.wav`,
`.mp3` or a font file will be declined regardless of how good it looks, because
it breaks the property that makes the project interesting: the entire game ships
as three files and roughly 594 KB gzipped.

If you want a new surface, write a new procedural recipe in
`src/materials/surfaces/`. If you want a new sound, synthesise it in
`src/audio/Synthesis.ts`.

The only images in the repository are the README screenshots in
`docs/screenshots/`, which are documentation and are not loaded by the game.

## Code conventions

- **TypeScript strict mode.** No `any` without a comment justifying it, and no
  `@ts-ignore`.
- **Comment the "why", not the "what".** The codebase leans heavily on this.
  Numeric constants especially: if a value is `0.014` rather than `0.05`,
  explain what went wrong at `0.05`. Several existing comments record exactly
  which visual artifact a given value was chosen to avoid, and that history is
  worth more than a restatement of the code.
- **Respect subsystem boundaries.** Systems talk through the interfaces in
  `src/core/Types.ts`. Cross-cutting wiring belongs in `src/core/Bridge.ts`, not
  scattered through the subsystems.
- **Registration order in `src/main.ts` is load-bearing.** Read the comment
  there before reordering anything.

## Test layout

Tests live in `tests/` and run under Vitest with no browser required:

| File | Covers |
| --- | --- |
| `tests/shaders.test.ts` | GLSL recipes: reserved words, balanced braces, entry point, unique ids |
| `tests/geometry.test.ts` | `beveledBox` and `sweptFinger` — smooth vs flat normals, finite vertices, dimensions |
| `tests/input.test.ts` | The key-binding table is complete and unambiguous |
| `tests/capture.test.ts` | Query-string parsing, the weapon allowlist, shot-preset integrity |
| `tests/specs.test.ts` | Weapon-data invariants and damage falloff |

Add tests next to the concern they cover rather than creating a new file per
change. If you are adding a rendering feature, prefer an assertion about a
measurable property of the geometry or the shader source over a screenshot
comparison — the existing suite runs in under a second precisely because it
never starts a GPU.

## Verifying visual changes

Rendering changes need visual evidence, not just a passing typecheck. The
project has a deterministic capture harness for this — the same URL produces the
same pixels every run:

```text
?shot=courtyard&hud=0          # a fixed camera preset
?shot=arms&isolate=1&hud=0     # the viewmodel on a neutral backdrop
```

The most useful technique when debugging a rendering issue is an A/B capture
with a single feature toggled, for example `&volumetrics=0` or `&ssao=0`. See
the URL parameters section of the README for the full list.

Two lessons from this codebase's own history, offered so they are not repeated:

- **Measure rather than eyeball.** Several bugs here looked like lighting
  problems and were not. Reading back pixel values or dumping live uniforms
  found causes that staring at screenshots did not.
- **Check that your change actually took effect.** One fix appeared to do
  nothing because the uniform set in a constructor was being overwritten every
  frame elsewhere. Verify the value at runtime before concluding a hypothesis is
  wrong.

## Pull requests

- Keep them focused; one concern per PR.
- Describe the user-visible effect, not just the mechanism.
- For visual changes, include before/after captures from the same preset.
- Note any change in draw calls or triangle count for renderer work.

## Reporting bugs

Use the issue templates. For rendering problems, please include your browser,
GPU, and the capture URL that reproduces it — the harness makes rendering bugs
far easier to reproduce than a prose description does.

Security vulnerabilities should **not** be filed as public issues. See
[SECURITY.md](SECURITY.md).
