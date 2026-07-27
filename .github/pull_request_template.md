## What this changes

<!-- Describe the user-visible effect, not only the mechanism. -->

## Why

<!-- What was wrong, or what this enables. Link an issue if there is one. -->

## Visual evidence

<!--
Required for any rendering change. Use the deterministic capture harness so
before/after are directly comparable, e.g.:

  ?shot=courtyard&hud=0
  ?shot=arms&isolate=1&hud=0

Paste before/after images from the SAME preset.
-->

| Before | After |
| ------ | ----- |
|        |       |

## Checklist

- [ ] `npm run typecheck` passes
- [ ] `npm run build` passes
- [ ] No external assets added (no `.png`, `.gltf`, `.wav`, fonts — everything
      stays procedural)
- [ ] Non-obvious constants are commented with *why* that value, not just what
      it does
- [ ] For renderer changes: draw call / triangle count impact noted below

<!-- Draw calls before → after, triangles before → after, if applicable. -->
