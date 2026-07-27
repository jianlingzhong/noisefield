import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { beveledBox, sweptFinger } from '../src/weapons/WeaponModels';

/*
 * Geometry invariants for the two builders the whole weapon and hand set is
 * made of. Both encode a specific regression.
 *
 * `beveledBox` is built on `ExtrudeGeometry`, which returns NON-INDEXED
 * geometry and finishes by calling `computeVertexNormals()`. With no shared
 * vertices to average across, that writes the face normal to all three corners
 * of every triangle — flat shading, on every part, regardless of the
 * `flatShading` flag being left false. The chamfer that the design pays for on
 * all twelve edges then renders as more flat facets instead of the soft rolling
 * highlight that makes a shape read as machined metal. The fix re-derives
 * normals with a crease threshold; the test below asserts the RESULT of that
 * (smooth shading exists) rather than the mechanism, so it keeps working if the
 * implementation changes.
 *
 * `sweptFinger` replaced three overlapping boxes per finger, which read as a
 * chain of beads because three convex boxes on a curve have three silhouettes.
 */

/** Every finite position/normal, or a list of what is wrong. */
function findNonFinite(geo: THREE.BufferGeometry, attr: string): number {
  const a = geo.getAttribute(attr);
  if (!a) return -1;
  let bad = 0;
  for (let i = 0; i < a.count * a.itemSize; i++) {
    if (!Number.isFinite(a.array[i] as number)) bad++;
  }
  return bad;
}

/**
 * Fraction of triangles whose three vertex normals are identical.
 *
 * This is the direct signal for flat shading. Under flat shading it is 1.0 for
 * every triangle. A correctly creased bevelled box still has large genuinely
 * flat faces — those SHOULD report identical normals — but the fillet triangles
 * must not, so the fraction has to sit meaningfully below 1.
 */
function flatTriangleFraction(geo: THREE.BufferGeometry): number {
  const n = geo.getAttribute('normal');
  expect(n).toBeTruthy();
  const tris = Math.floor(n.count / 3);
  let flat = 0;
  for (let t = 0; t < tris; t++) {
    const i = t * 3;
    const same =
      Math.abs(n.getX(i) - n.getX(i + 1)) < 1e-6 &&
      Math.abs(n.getY(i) - n.getY(i + 1)) < 1e-6 &&
      Math.abs(n.getZ(i) - n.getZ(i + 1)) < 1e-6 &&
      Math.abs(n.getX(i) - n.getX(i + 2)) < 1e-6 &&
      Math.abs(n.getY(i) - n.getY(i + 2)) < 1e-6 &&
      Math.abs(n.getZ(i) - n.getZ(i + 2)) < 1e-6;
    if (same) flat++;
  }
  return tris === 0 ? 1 : flat / tris;
}

describe('beveledBox', () => {
  const box = beveledBox(0.04, 0.03, 0.12, 0.0022);

  it('produces geometry with positions and normals', () => {
    expect(box.getAttribute('position').count).toBeGreaterThan(0);
    expect(box.getAttribute('normal')).toBeTruthy();
    expect(box.getAttribute('uv')).toBeTruthy();
  });

  it('contains no NaN or Infinity', () => {
    // A single non-finite vertex silently destroys the bounding sphere, which
    // makes the whole mesh get frustum-culled at random angles.
    expect(findNonFinite(box, 'position')).toBe(0);
    expect(findNonFinite(box, 'normal')).toBe(0);
  });

  it('has unit-length normals', () => {
    const n = box.getAttribute('normal');
    for (let i = 0; i < n.count; i++) {
      const len = Math.hypot(n.getX(i), n.getY(i), n.getZ(i));
      expect(len).toBeGreaterThan(0.99);
      expect(len).toBeLessThan(1.01);
    }
  });

  it('is SMOOTH-shaded across its chamfers, not flat', () => {
    // The regression. Under the old behaviour this was exactly 1.0.
    expect(flatTriangleFraction(box)).toBeLessThan(0.9);
  });

  it('still keeps hard edges on its flat faces', () => {
    // The opposite failure: creasing at too high a threshold rounds a box into
    // a pillow. Some triangles must still be uniformly flat.
    expect(flatTriangleFraction(box)).toBeGreaterThan(0.15);
  });

  it('respects the requested outer dimensions', () => {
    box.computeBoundingBox();
    const bb = box.boundingBox;
    expect(bb).toBeTruthy();
    const size = new THREE.Vector3();
    bb!.getSize(size);
    // Dimensions are the FINAL outer size including the bevel, so they should
    // land on the requested values within a small tolerance.
    expect(size.x).toBeCloseTo(0.04, 3);
    expect(size.y).toBeCloseTo(0.03, 3);
    expect(size.z).toBeCloseTo(0.12, 3);
  });

  it('is centred on the origin', () => {
    box.computeBoundingBox();
    const c = new THREE.Vector3();
    box.boundingBox!.getCenter(c);
    expect(c.length()).toBeLessThan(1e-6);
  });

  it('survives degenerate requests without producing NaN', () => {
    // Call sites range from a 5.5 mm gloved palm to a 0.1 mm reticle bar, and
    // the bevel is clamped against the smallest dimension. A sub-bevel-sized
    // box must still yield usable geometry.
    const tiny = beveledBox(0.0004, 0.0004, 0.0004, 0.01);
    expect(findNonFinite(tiny, 'position')).toBe(0);
    expect(findNonFinite(tiny, 'normal')).toBe(0);
    expect(tiny.getAttribute('position').count).toBeGreaterThan(0);
  });
});

describe('sweptFinger', () => {
  // A representative index finger wrapping a pistol grip.
  const finger = sweptFinger(1, 0.019, 0.014, 0.0, 0.24, 1.6, 0.017, 1.0);

  it('produces a single continuous surface', () => {
    expect(finger.getAttribute('position').count).toBeGreaterThan(100);
    expect(finger.getAttribute('normal')).toBeTruthy();
    expect(finger.getAttribute('uv')).toBeTruthy();
  });

  it('contains no NaN or Infinity', () => {
    expect(findNonFinite(finger, 'position')).toBe(0);
    expect(findNonFinite(finger, 'normal')).toBe(0);
    expect(findNonFinite(finger, 'uv')).toBe(0);
  });

  it('is smooth-shaded along its length', () => {
    // The entire point of the rewrite: one surface with averaged normals
    // through the knuckles, rather than three boxes with hard waists.
    expect(flatTriangleFraction(finger)).toBeLessThan(0.5);
  });

  it('is anatomically plausible in size', () => {
    finger.computeBoundingBox();
    const size = new THREE.Vector3();
    finger.boundingBox!.getSize(size);
    // A curled finger occupies a few centimetres in every axis. This catches
    // unit errors (metres vs millimetres) which otherwise show up only as a
    // hand the size of a building or one invisible at the origin.
    const longest = Math.max(size.x, size.y, size.z);
    expect(longest).toBeGreaterThan(0.01);
    expect(longest).toBeLessThan(0.12);
  });

  it('is no wider than a finger across its row axis', () => {
    finger.computeBoundingBox();
    const size = new THREE.Vector3();
    finger.boundingBox!.getSize(size);
    // Y is the across-the-finger axis; 17 mm requested, so it must not exceed
    // that by much regardless of the knuckle swells.
    expect(size.y).toBeLessThan(0.025);
  });

  it('mirrors cleanly for the left hand', () => {
    const left = sweptFinger(-1, 0.019, 0.014, 0.0, 0.24, 1.6, 0.017, 1.0);
    expect(findNonFinite(left, 'position')).toBe(0);

    finger.computeBoundingBox();
    left.computeBoundingBox();
    const r = new THREE.Vector3();
    const l = new THREE.Vector3();
    finger.boundingBox!.getSize(r);
    left.boundingBox!.getSize(l);
    // Mirroring must not change the finger's dimensions.
    expect(l.x).toBeCloseTo(r.x, 6);
    expect(l.y).toBeCloseTo(r.y, 6);
    expect(l.z).toBeCloseTo(r.z, 6);
  });

  it('takes its physical extent from the arc, not from lengthScale', () => {
    /*
     * Worth stating explicitly, because the parameter name suggests otherwise
     * and a first version of this test asserted the opposite.
     *
     * The caller folds the finger's length into `sweep` before calling here
     * (`want = fingerSum * FINGER_SCALE * wrap / meanR`), so the arc from
     * theta0 to theta0+sweep already IS the finger's length. `lengthScale`
     * therefore only puts the UVs in metre space. Changing it must not move a
     * single vertex.
     */
    const other = sweptFinger(1, 0.019, 0.014, 0, 0.24, 1.6, 0.017, 0.75);
    const a = other.getAttribute('position');
    const b = finger.getAttribute('position');
    expect(a.count).toBe(b.count);
    for (let i = 0; i < a.count; i++) {
      expect(a.getX(i)).toBeCloseTo(b.getX(i), 9);
      expect(a.getY(i)).toBeCloseTo(b.getY(i), 9);
      expect(a.getZ(i)).toBeCloseTo(b.getZ(i), 9);
    }
  });

  it('scales the UV span with lengthScale', () => {
    // The flip side: lengthScale must still reach the UVs, or the glove weave
    // would run at a different gauge on a long finger than a short one.
    const vSpan = (g: THREE.BufferGeometry): number => {
      const uv = g.getAttribute('uv');
      let max = 0;
      for (let i = 0; i < uv.count; i++) max = Math.max(max, uv.getY(i));
      return max;
    };
    const short = sweptFinger(1, 0.019, 0.014, 0, 0.24, 1.6, 0.017, 0.5);
    expect(vSpan(short)).toBeCloseTo(vSpan(finger) * 0.5, 6);
  });

  it('changes shape when the sweep changes', () => {
    // Since sweep is what actually drives the geometry, it must demonstrably
    // do so — otherwise every finger on the hand would be identical.
    const tighter = sweptFinger(1, 0.019, 0.014, 0, 0.24, 2.2, 0.017, 1.0);
    const a = tighter.getAttribute('position');
    const b = finger.getAttribute('position');
    let moved = 0;
    for (let i = 0; i < Math.min(a.count, b.count); i++) {
      if (Math.abs(a.getZ(i) - b.getZ(i)) > 1e-4) moved++;
    }
    expect(moved).toBeGreaterThan(0);
  });

  it('produces a closed tip rather than an open tube', () => {
    // The tip is rounded by collapsing the last rings toward the centre line.
    // If that regressed, the finger would be a pipe you can see down.
    const pos = finger.getAttribute('position');
    const uv = finger.getAttribute('uv');
    let maxV = 0;
    for (let i = 0; i < uv.count; i++) maxV = Math.max(maxV, uv.getY(i));

    // Gather the ring of vertices at the very end of the sweep and check that
    // they have collapsed to nearly a single point.
    const tip: THREE.Vector3[] = [];
    for (let i = 0; i < uv.count; i++) {
      if (uv.getY(i) > maxV - 1e-6) tip.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)));
    }
    expect(tip.length).toBeGreaterThan(2);
    const centre = tip.reduce((acc, v) => acc.add(v), new THREE.Vector3()).divideScalar(tip.length);
    for (const v of tip) expect(v.distanceTo(centre)).toBeLessThan(1e-3);
  });
});
