# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-07-27

Initial public release.

### Added

- Renderer: cascaded shadow maps, N8AO ambient occlusion, bloom, SMAA, motion
  blur and a filmic grade.
- Procedural material forge: 17 PBR surfaces baked to GPU render targets at load
  time, with no external texture files.
- "Compound Zulu": a 90 x 90 m three-lane map with props and decals.
- Five weapons with distinct recoil patterns, ballistics with wall penetration,
  and a fully procedural viewmodel.
- Player controller, enemy AI and ragdoll physics.
- BVH-accelerated raycasting and a custom rigid-body solver.
- VFX: analytic GPU particles, impact decals, tracers, ground fog and dust motes.
- Fully synthesised audio — 37 sound IDs across 95 buffers, no audio files.
- HUD, crosshair, compass, minimap and a pause menu with live quality settings.
- Deterministic capture harness (`?shot=`) for reproducible screenshots.
- Test suite: 271 tests covering GLSL reserved words, geometry normals, key
  bindings, capture-parameter validation and weapon-data invariants.

### Fixed

These are recorded because each was a genuine rendering defect whose cause was
non-obvious, and the diagnosis is worth preserving.

- Two materials baked to pure black. `patch` is a reserved word in GLSL ES 3.00,
  so the shader failed to compile and the bake silently returned the clear
  colour.
- Every chamfered edge on every weapon was flat-shaded. `ExtrudeGeometry`
  returns non-indexed geometry, so `computeVertexNormals()` assigns a face
  normal to all three vertices of each triangle. Resolved with creased normals
  at a 50 degree threshold, which keeps genuine box corners hard.
- A full-screen additive heat-shimmer overlay with depth testing disabled was
  washing out the entire frame, making solid geometry appear translucent.
  Reduced to a faint horizon warmth; see the known limitations in the README.
- Fingers were built as three overlapping boxes per digit and read as a chain of
  beads. Replaced with a single swept tube per finger with knuckle swells and a
  rounded tip.
- The glove and sleeve had no albedo map, only a flat colour, and read as
  moulded plastic rather than worn fabric.
- The `combat` capture preset, which is meant to catch a burst in flight,
  fired and then advanced far enough for every
  transient to expire, so it never actually captured a burst.

[Unreleased]: https://github.com/jianlingzhong/noisefield/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jianlingzhong/noisefield/releases/tag/v1.0.0
