/**
 * NOISEFIELD — Prop factory library.
 *
 * Every function here returns raw geometry (or a list of keyed geometry parts)
 * in a canonical local frame, ready to be either merged into a static batch or
 * used as the source geometry of an InstancedMesh. Nothing here touches the
 * scene graph, creates a material, or knows what a Level is — which is what
 * makes the batching strategy in Level.ts possible.
 *
 * CANONICAL FRAME: props sit with their base at y = 0, centred on X/Z, facing
 * +Z where "facing" means anything. Sticking to one convention removes a whole
 * class of "why is the crate buried in the floor" placement bugs.
 *
 * SCALE IS NOT NEGOTIABLE. Every dimension below is a real measurement. A
 * player is 1.80 m tall with eyes at 1.70 m, so a 0.81 m jersey barrier is
 * chest-high to a crouching player and thigh-high to a standing one, and a
 * 2.591 m shipping container is exactly tall enough to be unpeekable. Get these
 * numbers wrong by 20% and the map stops reading as a place, instantly and
 * unfixably.
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import {
  Rng,
  catenary,
  chamferedBox,
  chamferedCylinder,
  corrugatedPanel,
  normalizeAttributes,
  rockGeometry,
  saggingCloth,
  tubeThrough,
} from './Geometry';

/**
 * A chunk of geometry tagged with the material bucket it belongs in. Multi-
 * material props (a car is steel + rubber + glass) return a list of these so
 * the batcher can split them across buckets without the prop knowing how
 * batching works.
 */
export interface GeoPart {
  key: string;
  geo: THREE.BufferGeometry;
  /** Preserve authored UVs rather than world-projecting them. */
  keepUV?: boolean;
}

export type GeoParts = GeoPart[];

// ─────────────────────────────────────────────────────────────────────────────
// Local construction helpers
// ─────────────────────────────────────────────────────────────────────────────

function merge(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  if (parts.length === 0) return new THREE.BufferGeometry();
  if (parts.length === 1) return normalizeAttributes(parts[0]);
  return mergeGeometries(parts.map(normalizeAttributes));
}

/** Push a chamfered box positioned by its centre. */
function box(
  out: THREE.BufferGeometry[],
  w: number, h: number, d: number,
  x: number, y: number, z: number,
  bevel = 0.02, ry = 0, rx = 0, rz = 0,
): void {
  const g = chamferedBox(w, h, d, bevel);
  if (rx || ry || rz) g.applyQuaternion(new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'YXZ')));
  g.translate(x, y, z);
  out.push(g);
}

/** Push a chamfered box that spans from point A to point B (a strut/beam). */
function strut(out: THREE.BufferGeometry[], a: THREE.Vector3, b: THREE.Vector3, thickness: number): void {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-4) return;
  const g = chamferedBox(thickness, thickness, len, thickness * 0.26);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), dir.clone().normalize());
  g.applyMatrix4(new THREE.Matrix4().compose(
    new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5),
    q,
    new THREE.Vector3(1, 1, 1),
  ));
  out.push(g);
}

/** Push a cylinder aligned to Y, centred at (x,y,z). */
function cyl(
  out: THREE.BufferGeometry[],
  r: number, h: number,
  x: number, y: number, z: number,
  seg = 10, rx = 0, rz = 0,
): void {
  const g = new THREE.CylinderGeometry(r, r, h, seg, 1, false);
  if (rx || rz) g.applyQuaternion(new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, 0, rz, 'YXZ')));
  g.translate(x, y, z);
  out.push(g);
}

// ─────────────────────────────────────────────────────────────────────────────

export class Props {
  // ═══════════════════════════════════════════════════════════════════════════
  // COVER — the load-bearing gameplay geometry
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * A standard-profile concrete jersey barrier. 3.05 m long, 0.81 m tall,
   * 0.61 m at the foot narrowing to 0.15 m at the crown.
   *
   * GAMEPLAY: 0.81 m is the canonical "vault it or crouch behind it" height —
   * it breaks a standing player's legs from a sightline and fully conceals a
   * crouching one. Chained into a chicane they turn an open yard into a
   * readable series of cover beats without blocking movement.
   */
  static jerseyBarrier(length = 3.05): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    // Stacked slabs approximate the classic swept profile: wide splayed foot,
    // steep shoulder, narrow crown.
    box(parts, length, 0.10, 0.61, 0, 0.05, 0, 0.02);
    box(parts, length, 0.14, 0.52, 0, 0.17, 0, 0.02);
    box(parts, length, 0.16, 0.36, 0, 0.32, 0, 0.02);
    box(parts, length, 0.41, 0.20, 0, 0.605, 0, 0.025);
    // Lifting slot and the moulded end-connector loop.
    box(parts, 0.30, 0.05, 0.10, 0, 0.80, 0, 0.015);
    for (const sx of [-1, 1]) box(parts, 0.05, 0.30, 0.16, sx * (length * 0.5 - 0.04), 0.5, 0, 0.012);
    return merge(parts);
  }

  /**
   * A Gabion barrier: a wire cage of geotextile filled with spoil.
   * 1.06 m cube, the "Bastion 1" unit. Stack two for full-height hard cover.
   */
  static gabion(size = 1.06): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    // The fill bulges: a filled gabion is never a crisp cube.
    box(parts, size * 1.02, size * 0.96, size * 1.02, 0, size * 0.5, 0, 0.06, 0);
    const t = 0.035;
    // Wire cage: vertical corner posts plus two horizontal bands. The cage is
    // what makes it read as a filled cage rather than as a sandy box.
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        box(parts, t, size, t, sx * size * 0.5, size * 0.5, sz * size * 0.5, t * 0.3);
      }
    }
    // One band ring rather than two: 36 instances of this thing sit in the map.
    for (const y of [size * 0.66]) {
      for (const sz of [-1, 1]) box(parts, size + t, t, t, 0, y, sz * size * 0.5, t * 0.3);
      for (const sx of [-1, 1]) box(parts, t, t, size + t, sx * size * 0.5, y, 0, t * 0.3);
    }
    return merge(parts);
  }

  /** A single sandbag: a squashed, slumped pillow. 0.50 x 0.18 x 0.30 m. */
  static sandbag(rng: Rng): THREE.BufferGeometry {
    const w = 0.50 * rng.range(0.94, 1.06);
    const h = 0.17 * rng.range(0.9, 1.1);
    const d = 0.30 * rng.range(0.94, 1.06);
    // A heavy chamfer on a flat box is exactly the silhouette of a filled bag,
    // and 2 segments rounds it just enough to lose the facets in shadow.
    return chamferedBox(w, h, d, Math.min(w, h, d) * 0.44, 2);
  }

  /**
   * A sandbag emplacement: courses of bags laid stretcher-bond, each course
   * stepped back slightly and every bag jittered.
   *
   * STORYTELLING: emplacements always face *outward* from what they protect.
   * Placing them with a deliberate facing is a one-line change that turns
   * scenery into evidence that people fought here.
   */
  static sandbagWall(length: number, rows: number, rng: Rng): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const bagW = 0.50;
    const bagH = 0.17;
    for (let r = 0; r < rows; r++) {
      const y = bagH * (r + 0.5) + r * 0.005;
      const stagger = (r % 2) * bagW * 0.5;
      const n = Math.max(1, Math.floor((length - stagger) / bagW));
      const setback = r * 0.035;
      for (let i = 0; i < n; i++) {
        const g = Props.sandbag(rng);
        g.rotateY(rng.range(-0.09, 0.09));
        g.rotateZ(rng.range(-0.05, 0.05));
        g.translate(
          -length * 0.5 + stagger + bagW * (i + 0.5) + rng.range(-0.015, 0.015),
          y,
          setback + rng.range(-0.02, 0.02),
        );
        parts.push(g);
      }
    }
    return merge(parts);
  }

  /**
   * A 20 ft ISO shipping container: 6.058 x 2.438 x 2.591 m.
   *
   * GAMEPLAY: the most valuable single prop in an arena map. It is full cover
   * standing, it is climbable-onto for a 2.6 m firing platform, it blocks a
   * 6 m sightline, and rotating it 20 degrees off-grid instantly makes a yard
   * feel unplanned. Corrugation matters: at 40 m you read the *silhouette* of
   * the ribs against the sky, and a flat box with a normal map does not have one.
   */
  static shippingContainer(length = 6.058): GeoParts {
    const H = 2.591;
    const W = 2.438;
    const steel: THREE.BufferGeometry[] = [];

    // Side walls (corrugated, ribs vertical) and end walls.
    for (const sz of [-1, 1]) {
      const p = corrugatedPanel(length - 0.16, H - 0.24, Math.round(length * 3.4), 0.026, 0.05);
      p.rotateY(sz > 0 ? 0 : Math.PI);
      p.translate(0, H * 0.5, sz * (W * 0.5 - 0.025));
      steel.push(p);
    }
    // Front end: plain panel. Rear end: cargo doors with locking bars.
    const front = corrugatedPanel(W - 0.16, H - 0.24, 8, 0.026, 0.05);
    front.rotateY(-Math.PI / 2);
    front.translate(-(length * 0.5 - 0.025), H * 0.5, 0);
    steel.push(front);

    for (const sz of [-1, 1]) {
      const door = corrugatedPanel(W * 0.5 - 0.1, H - 0.24, 5, 0.02, 0.05);
      door.rotateY(Math.PI / 2);
      door.translate(length * 0.5 - 0.03, H * 0.5, sz * (W * 0.25 - 0.01));
      steel.push(door);
      // Vertical locking bars + cam handles: the detail that says "container".
      for (const off of [0.30, 0.72]) {
        cyl(steel, 0.032, H - 0.34, length * 0.5 + 0.03, H * 0.5, sz * (W * 0.5 - off), 8);
        box(steel, 0.14, 0.05, 0.05, length * 0.5 + 0.07, 1.15, sz * (W * 0.5 - off), 0.012);
      }
    }

    // Roof, floor rails and the eight corner castings.
    box(steel, length - 0.14, 0.06, W - 0.14, 0, H - 0.03, 0, 0.02);
    box(steel, length, 0.14, W, 0, 0.07, 0, 0.03);
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        box(steel, 0.17, H, 0.17, sx * (length * 0.5 - 0.085), H * 0.5, sz * (W * 0.5 - 0.085), 0.02);
        for (const y of [0.09, H - 0.09]) {
          box(steel, 0.25, 0.18, 0.20, sx * (length * 0.5 - 0.12), y, sz * (W * 0.5 - 0.1), 0.03);
        }
      }
    }
    // Top and bottom rails frame the corrugation.
    for (const sz of [-1, 1]) {
      box(steel, length - 0.2, 0.13, 0.10, 0, H - 0.07, sz * (W * 0.5 - 0.05), 0.02);
      box(steel, length - 0.2, 0.15, 0.12, 0, 0.16, sz * (W * 0.5 - 0.05), 0.02);
    }

    return [{ key: 'rust', geo: merge(steel) }];
  }

  /**
   * A 200-litre oil drum: 0.880 m tall, 0.572 m diameter, with the two rolling
   * hoops and the chime rims. Ubiquitous, cheap, and instantly legible as cover
   * — and as something that ought to explode.
   */
  static oilDrum(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const R = 0.286;
    const H = 0.88;
    // 14 radial segments: at 0.57 m diameter the facets are ~13 cm of arc, which
    // is below the point where the silhouette reads as polygonal at 2 m. This
    // prop runs at 50+ instances so the segment count is worth being mean about.
    const SEG = 14;
    const body = chamferedCylinder(R - 0.012, H, 0.02, SEG);
    body.translate(0, H * 0.5, 0);
    parts.push(body);
    // Rolling hoops at 1/3 and 2/3 height, plus the top chime.
    for (const y of [H * 0.32, H * 0.68, H - 0.028]) {
      const ring = new THREE.CylinderGeometry(R, R, 0.055, SEG, 1, true);
      ring.translate(0, y, 0);
      parts.push(ring);
    }
    // Bung plugs on the lid.
    cyl(parts, 0.035, 0.02, R * 0.5, H + 0.005, 0, 6);
    return merge(parts);
  }

  /**
   * A wooden crate with plank slats and corner battens.
   *
   * PERF: this is the single most-instanced prop in the map (70+ copies), so
   * its triangle count is multiplied harder than anything else here. Two planks
   * per face rather than three, and no separate lid frame — the corner battens
   * already close the top edge in silhouette. 572 triangles instead of 924 for
   * a read that is identical past about 3 m, which is where crates live.
   */
  static crate(size = 0.86, rng?: Rng): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const s = size;
    const t = 0.035;
    // Core body slightly inset so the battens stand proud and cast micro-shadow.
    box(parts, s - t * 2, s - t * 2, s - t * 2, 0, s * 0.5, 0, 0.012);
    // Plank slats on all four sides, with a visible gap between them.
    const planks = 2;
    for (let f = 0; f < 4; f++) {
      const ry = (f * Math.PI) / 2;
      for (let i = 0; i < planks; i++) {
        const ph = (s - t * 2.2) / planks - 0.03;
        const y = t + ((s - t * 2) / planks) * (i + 0.5);
        const jitter = rng ? rng.range(-0.004, 0.004) : 0;
        const g = chamferedBox(s - t * 1.6, ph, t, 0.008);
        g.translate(0, y + jitter, s * 0.5 - t * 0.5);
        g.applyQuaternion(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, ry, 0)));
        parts.push(g);
      }
    }
    // Corner battens: the silhouette, and the top edge.
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        box(parts, t * 1.4, s, t * 1.4, sx * (s * 0.5 - t * 0.6), s * 0.5, sz * (s * 0.5 - t * 0.6), 0.01);
      }
    }
    return merge(parts);
  }

  /** A EUR pallet: 1.20 x 0.80 x 0.144 m. Deck boards, bearers, blocks. */
  static pallet(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const L = 1.2;
    const W = 0.8;
    // Nine blocks in a 3x3 grid.
    for (const x of [-L * 0.5 + 0.05, 0, L * 0.5 - 0.05]) {
      for (const z of [-W * 0.5 + 0.05, 0, W * 0.5 - 0.05]) {
        box(parts, 0.1, 0.078, 0.1, x, 0.039 + 0.022, z, 0.008);
      }
    }
    // Bottom bearers.
    for (const z of [-W * 0.5 + 0.05, 0, W * 0.5 - 0.05]) box(parts, L, 0.022, 0.1, 0, 0.011, z, 0.006);
    // Top deck: five boards with gaps.
    for (let i = 0; i < 5; i++) {
      const z = -W * 0.5 + 0.05 + (i * (W - 0.1)) / 4;
      box(parts, L, 0.022, 0.1, 0, 0.139 + 0.005, z, 0.006);
    }
    return merge(parts);
  }

  /** A car tyre. 0.63 m outer diameter, 0.20 m section. Lies flat (axis = Y). */
  static tyre(): THREE.BufferGeometry {
    const g = new THREE.TorusGeometry(0.215, 0.1, 8, 14);
    g.rotateX(Math.PI / 2);
    g.translate(0, 0.1, 0);
    return g;
  }

  /** A concrete bollard: 0.24 m diameter, 0.95 m tall, domed cap. */
  static bollard(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const g = chamferedCylinder(0.12, 0.9, 0.02, 12);
    g.translate(0, 0.45, 0);
    parts.push(g);
    const cap = new THREE.SphereGeometry(0.12, 12, 6, 0, Math.PI * 2, 0, Math.PI * 0.5);
    cap.translate(0, 0.9, 0);
    parts.push(cap);
    box(parts, 0.34, 0.06, 0.34, 0, 0.03, 0, 0.015);
    return merge(parts);
  }

  /** A brick. 0.215 x 0.065 x 0.1025 m. For rubble and half-built walls. */
  static brick(): THREE.BufferGeometry {
    return chamferedBox(0.215, 0.065, 0.1025, 0.008);
  }

  /** A concrete block. 0.44 x 0.215 x 0.215 m. */
  static cinderBlock(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.44, 0.215, 0.215, 0, 0.1075, 0, 0.012);
    // Two recessed voids in the top face read as a real block, not a lump.
    for (const sx of [-1, 1]) box(parts, 0.13, 0.03, 0.11, sx * 0.1, 0.215, 0, 0.01);
    return merge(parts);
  }

  /** A large-bore concrete drainage pipe section. Doubles as crouch cover. */
  static concretePipe(length = 2.4, outerR = 0.62): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const inner = outerR - 0.09;
    const outer = new THREE.CylinderGeometry(outerR, outerR, length, 20, 1, true);
    const bore = new THREE.CylinderGeometry(inner, inner, length, 20, 1, true);
    // Annular end caps close the tube so it reads as thick-walled.
    const capA = new THREE.RingGeometry(inner, outerR, 20);
    const capB = new THREE.RingGeometry(inner, outerR, 20);
    capA.rotateX(Math.PI / 2);
    capA.translate(0, length * 0.5, 0);
    capB.rotateX(-Math.PI / 2);
    capB.translate(0, -length * 0.5, 0);
    parts.push(outer, bore, capA, capB);
    const g = merge(parts);
    g.rotateZ(Math.PI / 2); // lie it down along X
    g.translate(0, outerR, 0);
    return g;
  }

  /** A raised planter box — waist-high soft cover that dresses a plaza. */
  static planter(w = 1.8, d = 0.9, h = 0.95): GeoParts {
    const con: THREE.BufferGeometry[] = [];
    const t = 0.12;
    box(con, w, h, t, 0, h * 0.5, d * 0.5 - t * 0.5, 0.03);
    box(con, w, h, t, 0, h * 0.5, -d * 0.5 + t * 0.5, 0.03);
    box(con, t, h, d - t * 2, -w * 0.5 + t * 0.5, h * 0.5, 0, 0.03);
    box(con, t, h, d - t * 2, w * 0.5 - t * 0.5, h * 0.5, 0, 0.03);
    box(con, w + 0.06, 0.07, d + 0.06, 0, h + 0.02, 0, 0.02); // coping
    const soil: THREE.BufferGeometry[] = [];
    box(soil, w - t * 2, 0.2, d - t * 2, 0, h - 0.16, 0, 0.03);
    return [{ key: 'concrete', geo: merge(con) }, { key: 'dirt', geo: merge(soil) }];
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ARCHITECTURAL DETAIL — greeble and silhouette
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * A recessed window frame: an inner reveal box, a moulded outer surround and
   * a sloped sill that throws water (and shadow) clear of the wall.
   *
   * A window MUST be an inset. The reveal depth is what produces the hard
   * shadow line that reads as "hole in a thick wall"; painting a dark rectangle
   * on a flat facade never reads as anything but a texture, at any resolution.
   */
  static windowFrame(w = 1.1, h = 1.3, wallThickness = 0.35): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const t = 0.09;
    const d = wallThickness * 0.86;
    const face = wallThickness * 0.5; // outward face of the wall, local +Z
    // Reveal: four faces lining the hole, set back into the wall so the opening
    // reads as a hole in something *thick*.
    box(parts, w + t * 2, t, d, 0, h + t * 0.5, -0.02, 0.015);
    box(parts, w + t * 2, t, d, 0, -t * 0.5, -0.02, 0.015);
    box(parts, t, h, d, -(w + t) * 0.5, h * 0.5, -0.02, 0.015);
    box(parts, t, h, d, (w + t) * 0.5, h * 0.5, -0.02, 0.015);
    // Architrave. The original stood 2 cm proud, which at 15 m is nothing — the
    // window read as a decal. 7 cm is the point at which the low sun throws a
    // visible bar of shadow down one jamb and the opening starts to read as
    // built. This is the single highest-value centimetre in the whole map.
    const p = 0.075;
    const proud = 0.075;
    for (const sx of [-1, 1]) {
      box(parts, p, h + t * 2 + p * 2, proud, sx * (w + t * 2 + p) * 0.5, h * 0.5 + t * 0.5, face + proud * 0.5 - 0.012, 0.012);
    }
    box(parts, w + t * 2 + p * 2, p, proud, 0, h + t + p * 0.5, face + proud * 0.5 - 0.012, 0.012);
    // Head / lintel: wider and deeper again, so the top of the opening carries
    // a hard horizontal shadow across the wall either side of it.
    box(parts, w + 0.54, 0.125, wallThickness * 0.5 + 0.04, 0, h + t + p + 0.062, face * 0.5 + 0.10, 0.018);
    box(parts, w + 0.38, 0.055, wallThickness * 0.5 + 0.09, 0, h + t + p + 0.152, face * 0.5 + 0.115, 0.014);
    // Sill: projects 16 cm, slopes to throw water, and carries a drip nose on
    // its underside so the shadow separates cleanly from the wall below.
    const sill = chamferedBox(w + t * 2 + 0.24, 0.085, wallThickness * 0.5 + 0.22, 0.015);
    sill.rotateX(-0.10);
    sill.translate(0, -t - 0.03, face * 0.5 + 0.07);
    parts.push(sill);
    box(parts, w + t * 2 + 0.18, 0.05, 0.05, 0, -t - 0.10, face + 0.155, 0.01);
    // Central mullion + transom: cheap, and it stops a big opening looking bald.
    box(parts, 0.045, h, 0.05, 0, h * 0.5, -0.06, 0.01);
    box(parts, w, 0.045, 0.05, 0, h * 0.62, -0.06, 0.01);
    return merge(parts);
  }

  /** A glass pane sized to sit inside a windowFrame opening. */
  static windowGlass(w = 1.1, h = 1.3): THREE.BufferGeometry {
    const g = new THREE.PlaneGeometry(w, h);
    g.translate(0, h * 0.5, 0);
    return g;
  }

  /** A door surround with a lintel. Opening is `w` x `h`; 2.10 m is a door. */
  static doorFrame(w = 1.05, h = 2.1, wallThickness = 0.35): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const t = 0.1;
    const d = wallThickness * 0.9;
    const face = wallThickness * 0.5;
    box(parts, w + t * 2, t, d, 0, h + t * 0.5, 0, 0.018);
    for (const sx of [-1, 1]) box(parts, t, h + t, d, sx * (w + t) * 0.5, (h + t) * 0.5, 0, 0.018);
    // Architrave standing 8 cm proud, wrapping jambs and head. A doorway is
    // where the eye goes first; a flush opening reads as a cut-out.
    const p = 0.09;
    for (const sx of [-1, 1]) {
      box(parts, p, h + t * 2 + p, 0.08, sx * (w + t * 2 + p) * 0.5, (h + t) * 0.5 + t * 0.5, face + 0.028, 0.014);
    }
    box(parts, w + t * 2 + p * 2, p, 0.08, 0, h + t + p * 0.5, face + 0.028, 0.014);
    // Proud lintel above, throwing a hard bar of shadow across the doorway.
    box(parts, w + 0.5, 0.14, wallThickness + 0.22, 0, h + t + p + 0.07, 0.03, 0.02);
    box(parts, w + 0.34, 0.06, wallThickness + 0.3, 0, h + t + p + 0.17, 0.045, 0.015);
    // Worn stone threshold + a shallow step, so the floor level reads.
    box(parts, w + 0.24, 0.045, wallThickness + 0.1, 0, 0.022, 0, 0.015);
    box(parts, w + 0.44, 0.09, 0.34, 0, 0.045, face + 0.15, 0.018);
    return merge(parts);
  }

  /** A panelled timber door leaf, hung open by `openRad`. */
  static doorLeaf(w = 1.0, h = 2.05, openRad = 0): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, w, h, 0.05, 0, h * 0.5, 0, 0.012);
    // Two raised panels + rails; the classic four-plane door reads at distance.
    for (const [py, ph] of [[0.32, 0.62], [0.68, 0.5]] as const) {
      box(parts, w * 0.7, h * ph * 0.72, 0.022, 0, h * py, 0.032, 0.01);
    }
    box(parts, w, 0.11, 0.03, 0, h * 0.5, 0.03, 0.008);
    // Handle.
    cyl(parts, 0.022, 0.11, w * 0.36, h * 0.47, 0.075, 8, Math.PI / 2);
    const g = merge(parts);
    if (openRad) {
      // Hinge on the left stile, so rotate about the leaf's -X edge.
      g.translate(w * 0.5, 0, 0);
      g.rotateY(openRad);
      g.translate(-w * 0.5, 0, 0);
    }
    return g;
  }

  /** A corrugated roller shutter, partially raised. `open` in 0..1. */
  static rollShutter(w = 2.4, h = 2.6, open = 0.0): GeoParts {
    const parts: THREE.BufferGeometry[] = [];
    const visible = h * (1 - open);
    if (visible > 0.08) {
      // Ribs run horizontally on a roller shutter.
      const p = corrugatedPanel(visible - 0.02, w - 0.04, Math.max(3, Math.round(visible * 7)), 0.018, 0.03);
      p.rotateZ(Math.PI / 2);
      p.translate(0, h - visible * 0.5, 0);
      parts.push(p);
      box(parts, w, 0.07, 0.06, 0, h - visible + 0.03, 0, 0.014); // bottom bar
    }
    // Roller housing and guide channels: present whether open or shut.
    cyl(parts, 0.11, w + 0.06, 0, h + 0.13, 0, 12, 0, Math.PI / 2);
    box(parts, w + 0.3, 0.1, 0.3, 0, h + 0.26, 0, 0.02);
    for (const sx of [-1, 1]) box(parts, 0.08, h, 0.13, sx * (w * 0.5 + 0.04), h * 0.5, 0, 0.015);
    return [{ key: 'rust', geo: merge(parts) }];
  }

  /**
   * A window-unit air-conditioner bolted to a facade. 0.82 x 0.62 x 0.40 m.
   * Roof and wall clutter is what stops a building looking like a monolith,
   * and AC units are the single most recognisable urban greeble there is.
   */
  static acUnit(): GeoParts {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.82, 0.62, 0.40, 0, 0.31, 0, 0.02);
    // Fan cowl: a recessed circle with a cross-brace guard.
    const cowl = new THREE.CylinderGeometry(0.24, 0.26, 0.045, 16);
    cowl.rotateX(Math.PI / 2);
    cowl.translate(0, 0.31, 0.205);
    parts.push(cowl);
    // Two guard bars, not four: this prop lives on rooftops at 20 m+.
    for (let i = 0; i < 2; i++) {
      box(parts, 0.46, 0.018, 0.018, 0, 0.31, 0.23, 0.004, 0, 0, (i * Math.PI) / 2);
    }
    // Louvre fins down the side + a mounting bracket underneath.
    for (let i = 0; i < 3; i++) box(parts, 0.02, 0.05, 0.36, -0.41, 0.14 + i * 0.15, 0, 0.005, 0, 0.35);
    for (const sx of [-1, 1]) {
      strut(parts, new THREE.Vector3(sx * 0.34, 0.02, -0.18), new THREE.Vector3(sx * 0.34, 0.02, 0.2), 0.045);
      strut(parts, new THREE.Vector3(sx * 0.34, 0.02, 0.18), new THREE.Vector3(sx * 0.34, -0.42, -0.19), 0.035);
    }
    // Condensate drip stain is handled by the decal system, not here.
    return [{ key: 'metal', geo: merge(parts) }];
  }

  /** A wall ventilation grille with horizontal louvre blades. */
  static ventGrille(w = 0.7, h = 0.5): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, w, h, 0.06, 0, 0, 0, 0.012);
    const blades = Math.max(3, Math.round(h / 0.075));
    for (let i = 0; i < blades; i++) {
      const y = -h * 0.5 + (h / blades) * (i + 0.5);
      box(parts, w - 0.07, 0.05, 0.07, 0, y, 0.045, 0.008, 0, -0.5);
    }
    return merge(parts);
  }

  /** A parabolic satellite dish on a wall/pole bracket, facing +Z. */
  static satelliteDish(radius = 0.62): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const focal = radius * 0.42;
    const profile: THREE.Vector2[] = [];
    // 6 profile rings x 14 segments. The dish is a 60 cm object usually seen at
    // 15 m or more; at 20x10 it cost 650 triangles per copy and there are three
    // dozen of them, which is a quarter of a percent of the frame spent on
    // curvature nobody can resolve.
    const N = 6;
    for (let i = 0; i <= N; i++) {
      const r = (radius * i) / N;
      profile.push(new THREE.Vector2(r, (r * r) / (4 * focal)));
    }
    profile.push(new THREE.Vector2(radius, (radius * radius) / (4 * focal) + 0.05));
    const dish = new THREE.LatheGeometry(profile, 14);
    dish.rotateX(-Math.PI / 2); // open face toward +Z
    parts.push(dish);
    // Feed horn on a tripod at the focus.
    cyl(parts, 0.05, 0.14, 0, 0, focal, 8, Math.PI / 2);
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2;
      strut(
        parts,
        new THREE.Vector3(Math.cos(a) * radius * 0.82, Math.sin(a) * radius * 0.82, radius * 0.17),
        new THREE.Vector3(0, 0, focal),
        0.022,
      );
    }
    // Mount arm behind.
    strut(parts, new THREE.Vector3(0, -radius * 0.3, -0.02), new THREE.Vector3(0, -radius * 0.75, -0.34), 0.06);
    return merge(parts);
  }

  /** A whip antenna / mast with a couple of stand-off insulators. */
  static antenna(height = 2.6): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    cyl(parts, 0.03, height * 0.55, 0, height * 0.275, 0, 6);
    cyl(parts, 0.017, height * 0.5, 0, height * 0.75, 0, 6);
    for (const y of [height * 0.3, height * 0.62]) box(parts, 0.11, 0.03, 0.03, 0, y, 0, 0.008);
    box(parts, 0.16, 0.05, 0.16, 0, 0.025, 0, 0.012);
    return merge(parts);
  }

  /** A wall-mounted or free-standing ladder. Rungs face +Z. */
  static ladder(height: number, width = 0.44): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    for (const sx of [-1, 1]) box(parts, 0.05, height, 0.09, sx * width * 0.5, height * 0.5, 0, 0.012);
    const rungs = Math.max(2, Math.floor(height / 0.3));
    for (let i = 0; i < rungs; i++) {
      cyl(parts, 0.017, width, 0, 0.22 + (i * (height - 0.35)) / (rungs - 1 || 1), 0, 8, 0, Math.PI / 2);
    }
    // Stand-off brackets hold it clear of the wall; the gap is what makes it
    // read as a ladder rather than as a painted stripe.
    for (const y of [height * 0.15, height * 0.85]) {
      for (const sx of [-1, 1]) box(parts, 0.04, 0.05, 0.18, sx * width * 0.5, y, -0.13, 0.008);
    }
    return merge(parts);
  }

  /** A guard rail along +X. Posts, top rail, mid rail, toe board. */
  static railing(length: number, height = 1.05): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const posts = Math.max(2, Math.round(length / 1.4) + 1);
    for (let i = 0; i < posts; i++) {
      const x = -length * 0.5 + (length * i) / (posts - 1);
      box(parts, 0.05, height, 0.05, x, height * 0.5, 0, 0.01);
    }
    cyl(parts, 0.026, length, 0, height, 0, 8, 0, Math.PI / 2);
    cyl(parts, 0.02, length, 0, height * 0.52, 0, 6, 0, Math.PI / 2);
    box(parts, length, 0.1, 0.03, 0, 0.06, 0, 0.008);
    return merge(parts);
  }

  /** An open steel grating deck with edge angle and cross ribs. */
  static gratedDeck(w: number, d: number): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, w, 0.035, d, 0, 0, 0, 0.008);
    for (const sx of [-1, 1]) box(parts, 0.05, 0.09, d, sx * (w * 0.5 - 0.025), 0.02, 0, 0.01);
    for (const sz of [-1, 1]) box(parts, w, 0.09, 0.05, 0, 0.02, sz * (d * 0.5 - 0.025), 0.01);
    const ribs = Math.max(2, Math.round(d / 0.35));
    for (let i = 0; i < ribs; i++) {
      const z = -d * 0.5 + (d / ribs) * (i + 0.5);
      box(parts, w - 0.1, 0.045, 0.035, 0, -0.03, z, 0.006);
    }
    return merge(parts);
  }

  /** A half-round rain gutter running along X, with brackets. */
  static gutter(length: number): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const g = new THREE.CylinderGeometry(0.075, 0.075, length, 10, 1, true, 0, Math.PI);
    g.rotateZ(Math.PI / 2);
    g.rotateX(Math.PI);
    parts.push(g);
    const n = Math.max(2, Math.round(length / 1.6));
    for (let i = 0; i < n; i++) {
      const x = -length * 0.5 + (length * (i + 0.5)) / n;
      box(parts, 0.03, 0.03, 0.2, x, 0.05, -0.08, 0.006);
    }
    return merge(parts);
  }

  /** A vertical downpipe with wall clips. */
  static downpipe(height: number): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    cyl(parts, 0.045, height, 0, height * 0.5, 0, 8);
    const n = Math.max(2, Math.round(height / 1.8));
    for (let i = 0; i < n; i++) {
      box(parts, 0.13, 0.03, 0.09, 0, 0.4 + (i * (height - 0.7)) / (n - 1 || 1), -0.05, 0.006);
    }
    // Shoe at the bottom, kicking clear of the wall.
    cyl(parts, 0.045, 0.3, 0, 0.15, 0.09, 8, 0.5);
    return merge(parts);
  }

  /**
   * A run of service pipes clipped to a wall, following a polyline.
   * Nothing breaks up a blank stucco wall faster or cheaper than three pipes
   * and a bracket every two metres.
   */
  static pipeRun(points: THREE.Vector3[], radius = 0.055, count = 2, spacing = 0.14): GeoParts {
    const parts: THREE.BufferGeometry[] = [];
    for (let c = 0; c < count; c++) {
      const off = (c - (count - 1) * 0.5) * spacing;
      const shifted = points.map((p) => new THREE.Vector3(p.x, p.y + off, p.z));
      parts.push(tubeThrough(shifted, radius * (c === 0 ? 1 : 0.78), 6, Math.max(8, points.length * 5)));
    }
    // Support brackets at each control point.
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const g = chamferedBox(0.05, count * spacing + 0.12, 0.05, 0.01);
      g.translate(p.x, p.y, p.z);
      parts.push(g);
    }
    return [{ key: 'rust', geo: merge(parts) }];
  }

  /** A slack cable / guy-wire spanning two points. */
  static cable(a: THREE.Vector3, b: THREE.Vector3, sag: number, radius = 0.018): THREE.BufferGeometry {
    return tubeThrough(catenary(a, b, sag, 10), radius, 4, 18);
  }

  /**
   * A laundry line strung across an alley, with hanging sheets.
   * Cloth at head height over a lane does three jobs at once: it dresses the
   * space, it softly occludes a long sightline without blocking movement, and
   * it moves the eye upward so players notice the rooftop route.
   */
  static laundryLine(a: THREE.Vector3, b: THREE.Vector3, rng: Rng): GeoParts {
    const sag = a.distanceTo(b) * 0.05;
    const line = Props.cable(a, b, sag, 0.012);
    const cloths: THREE.BufferGeometry[] = [];
    const pts = catenary(a, b, sag, 10);
    const n = rng.int(3, 5);
    for (let i = 0; i < n; i++) {
      const t = (i + 0.7) / (n + 0.4);
      const idx = Math.min(pts.length - 1, Math.round(t * (pts.length - 1)));
      const p = pts[idx];
      const w = rng.range(0.5, 0.95);
      const h = rng.range(0.6, 1.15);
      const cloth = saggingCloth(w, h, 0.06, 5, 4, rng);
      // Rotate the sheet upright so it hangs from the line.
      cloth.rotateX(Math.PI / 2);
      cloth.rotateY(rng.range(-0.35, 0.35));
      cloth.translate(p.x, p.y - 0.02, p.z);
      cloths.push(cloth);
    }
    return [
      { key: 'metal', geo: line },
      { key: 'fabric', geo: merge(cloths) },
    ];
  }

  /**
   * A shop awning: sloped cloth on a steel frame, projecting from a facade.
   * Frame at +Z = out from the wall.
   */
  static awning(w = 2.6, depth = 1.5, height = 2.7, rng?: Rng): GeoParts {
    const frame: THREE.BufferGeometry[] = [];
    const drop = 0.42;
    for (const sx of [-1, 1]) {
      const x = sx * w * 0.5;
      strut(frame, new THREE.Vector3(x, height, 0), new THREE.Vector3(x, height - drop, depth), 0.045);
      strut(frame, new THREE.Vector3(x, height - 0.55, 0), new THREE.Vector3(x, height - drop, depth * 0.94), 0.035);
    }
    strut(frame, new THREE.Vector3(-w * 0.5, height - drop, depth), new THREE.Vector3(w * 0.5, height - drop, depth), 0.04);
    strut(frame, new THREE.Vector3(-w * 0.5, height, 0.03), new THREE.Vector3(w * 0.5, height, 0.03), 0.04);

    const cloth = saggingCloth(w, Math.hypot(depth, drop), 0.11, 10, 4, rng);
    cloth.rotateX(Math.atan2(drop, depth));
    cloth.translate(0, height - drop * 0.5 + 0.02, depth * 0.5);
    // Scalloped valance hanging off the front edge.
    const valance = saggingCloth(w, 0.28, 0.05, 10, 2, rng);
    valance.rotateX(Math.PI / 2);
    valance.translate(0, height - drop - 0.14, depth);

    return [
      { key: 'metal', geo: merge(frame) },
      { key: 'fabric', geo: merge([cloth, valance]) },
    ];
  }

  /**
   * A market stall: counter, posts, awning and produce.
   * Counter top lands at 1.05 m — deliberately in the crouch-cover band, so the
   * market street plays as a rhythm of short hops between hard points.
   */
  static marketStall(rng: Rng, w = 2.4, d = 1.15, toppled = false): GeoParts {
    const wood: THREE.BufferGeometry[] = [];
    const cloth: THREE.BufferGeometry[] = [];
    const goods: THREE.BufferGeometry[] = [];

    // Counter: top slab, apron, trestle legs.
    box(wood, w, 0.06, d, 0, 1.02, 0, 0.012);
    box(wood, w - 0.1, 0.16, 0.05, 0, 0.9, d * 0.5 - 0.04, 0.01);
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        box(wood, 0.07, 0.99, 0.07, sx * (w * 0.5 - 0.09), 0.495, sz * (d * 0.5 - 0.08), 0.012);
      }
      box(wood, 0.05, 0.05, d - 0.2, sx * (w * 0.5 - 0.09), 0.28, 0, 0.01);
    }
    // Canopy posts and the cloth roof.
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        box(wood, 0.06, 2.25, 0.06, sx * w * 0.5, 1.125, sz * d * 0.5, 0.012);
      }
    }
    box(wood, w + 0.12, 0.05, 0.05, 0, 2.25, -d * 0.5, 0.01);
    box(wood, w + 0.12, 0.05, 0.05, 0, 2.08, d * 0.5, 0.01);
    const roof = saggingCloth(w + 0.2, d + 0.35, 0.13, 8, 4, rng);
    roof.rotateX(-0.14);
    roof.translate(0, 2.2, 0.04);
    cloth.push(roof);
    const skirt = saggingCloth(w + 0.2, 0.4, 0.05, 8, 2, rng);
    skirt.rotateX(Math.PI / 2);
    skirt.translate(0, 2.02, d * 0.5 + 0.16);
    cloth.push(skirt);

    // Produce: crates and stacked spheres on the counter.
    const rows = rng.int(2, 4);
    for (let i = 0; i < rows; i++) {
      const x = -w * 0.5 + 0.35 + (i * (w - 0.7)) / Math.max(1, rows - 1);
      const c = chamferedBox(0.42, 0.2, 0.32, 0.03);
      c.rotateY(rng.range(-0.2, 0.2));
      c.translate(x, 1.15, rng.range(-0.1, 0.1));
      goods.push(c);
      for (let j = 0; j < rng.int(3, 6); j++) {
        const s = new THREE.SphereGeometry(rng.range(0.045, 0.075), 5, 4);
        s.translate(x + rng.range(-0.15, 0.15), 1.28, rng.range(-0.11, 0.11));
        goods.push(s);
      }
    }

    const out: GeoParts = [
      { key: 'wood', geo: merge(wood) },
      { key: 'fabric', geo: merge(cloth) },
      { key: 'accentOchre', geo: merge(goods) },
    ];

    if (toppled) {
      // Knock the whole stall over. A single toppled stall in an otherwise tidy
      // market is the cheapest possible "something happened here" beat.
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0.4, -1.28, 'YXZ'));
      const m = new THREE.Matrix4().compose(new THREE.Vector3(0, 0.55, 0), q, new THREE.Vector3(1, 1, 1));
      for (const p of out) p.geo.applyMatrix4(m);
    }
    return out;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // VEHICLES — silhouette-first, low poly, heavily chamfered
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * A derelict saloon car. 4.35 x 1.78 x 1.46 m, sitting low on flat tyres.
   *
   * Cars are the best mid-map cover in the genre: they are exactly tall enough
   * to break a standing sightline, low enough to shoot over from a crouch at
   * the bonnet, and they have a natural "peek left / peek right" pair of ends.
   */
  static derelictCar(rng: Rng): GeoParts {
    const body: THREE.BufferGeometry[] = [];
    const rubber: THREE.BufferGeometry[] = [];
    const dark: THREE.BufferGeometry[] = [];
    const L = 4.35;
    const W = 1.78;

    // Lower body: sill-to-waistline slab, generously chamfered so the shoulder
    // line catches the sun the way sheet metal does.
    box(body, L, 0.62, W, 0, 0.76, 0, 0.13, 0, 0, 0);
    // Bonnet and boot: shallow wedges either end.
    box(body, 1.35, 0.16, W - 0.1, -L * 0.5 + 0.72, 1.02, 0, 0.05, 0, -0.05);
    box(body, 1.05, 0.16, W - 0.1, L * 0.5 - 0.58, 1.03, 0, 0.05, 0, 0.05);
    // Cabin: narrower and shorter, tapered by a second smaller block on top.
    box(body, 2.05, 0.52, W - 0.16, 0.02, 1.32, 0, 0.12);
    box(body, 1.62, 0.1, W - 0.34, 0.06, 1.6, 0, 0.06);
    // A/B/C pillars.
    for (const sz of [-1, 1]) {
      for (const px of [-0.92, 0.1, 1.0]) {
        box(body, 0.09, 0.5, 0.09, px, 1.35, sz * (W * 0.5 - 0.12), 0.02, 0, 0, px * 0.06);
      }
    }
    // Bumpers, sills and a crumpled wing.
    box(body, 0.22, 0.18, W - 0.06, -L * 0.5 + 0.06, 0.55, 0, 0.05);
    box(body, 0.22, 0.18, W - 0.06, L * 0.5 - 0.06, 0.56, 0, 0.05);
    for (const sz of [-1, 1]) box(body, L - 1.5, 0.12, 0.1, 0, 0.44, sz * W * 0.5, 0.025);
    box(body, 0.8, 0.42, 0.3, -L * 0.5 + 0.45, 0.8, W * 0.5 - 0.1, 0.08, 0.2, 0.15);

    // Grille + light recesses read as the "face" and cost almost nothing.
    box(dark, 0.06, 0.2, 0.9, -L * 0.5 + 0.02, 0.8, 0, 0.02);
    for (const sz of [-1, 1]) {
      box(dark, 0.07, 0.16, 0.3, -L * 0.5 + 0.03, 0.86, sz * 0.62, 0.02);
      box(dark, 0.07, 0.13, 0.26, L * 0.5 - 0.03, 0.86, sz * 0.6, 0.02);
    }
    // Blown-out glazing: only the frame rubbers remain.
    for (const sz of [-1, 1]) box(dark, 1.9, 0.44, 0.03, 0.02, 1.33, sz * (W * 0.5 - 0.08), 0.01);

    // Wheels: three flat tyres, one missing entirely so it rests on the drum.
    const missing = rng.int(0, 3);
    let wi = 0;
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const x = sx * (L * 0.5 - 0.86);
        const z = sz * (W * 0.5 - 0.12);
        if (wi !== missing) {
          const t = new THREE.TorusGeometry(0.2, 0.105, 7, 12);
          t.scale(1, 0.82, 1); // deflated
          t.rotateY(Math.PI / 2);
          t.translate(x, 0.29, z);
          rubber.push(t);
        }
        cyl(dark, 0.13, 0.14, x, 0.29, z, 10, 0, Math.PI / 2);
        wi++;
      }
    }

    return [
      { key: 'paintFaded', geo: merge(body) },
      { key: 'rubber', geo: merge(rubber) },
      { key: 'metal', geo: merge(dark) },
    ];
  }

  /**
   * A burnt-out flatbed truck. Larger footprint, taller cover, and the
   * scorched hulk anchors the story of the compound.
   */
  static burntTruck(): GeoParts {
    const parts: THREE.BufferGeometry[] = [];
    const L = 6.2;
    const W = 2.2;
    // Chassis rails and axles.
    for (const sz of [-1, 1]) box(parts, L, 0.18, 0.16, 0, 0.62, sz * 0.42, 0.03);
    for (const x of [-L * 0.5 + 1.1, L * 0.5 - 1.3, L * 0.5 - 0.35]) {
      cyl(parts, 0.09, W - 0.35, x, 0.44, 0, 8, 0, Math.PI / 2);
    }
    // Cab: burnt to the shell, roof partly caved.
    box(parts, 1.9, 1.25, W - 0.15, -L * 0.5 + 1.15, 1.35, 0, 0.09);
    box(parts, 1.7, 0.08, W - 0.45, -L * 0.5 + 1.2, 1.99, 0, 0.04, 0, -0.09);
    box(parts, 1.0, 0.7, W - 0.2, -L * 0.5 + 0.25, 0.95, 0, 0.07);
    // Flatbed with drop-side stakes, one side blown outward.
    box(parts, 3.6, 0.12, W, L * 0.5 - 1.95, 1.02, 0, 0.03);
    for (const sz of [-1, 1]) {
      for (let i = 0; i < 4; i++) {
        const x = L * 0.5 - 3.5 + i * 1.0;
        box(parts, 0.09, 0.75, 0.09, x, 1.42, sz * (W * 0.5 - 0.06), 0.02, 0, 0, sz * i * 0.05);
      }
      box(parts, 3.4, 0.5, 0.06, L * 0.5 - 1.95, 1.32, sz * (W * 0.5 - 0.03), 0.015, 0, 0, sz * 0.12);
    }
    // Rear frame and a torn tailgate hanging open.
    box(parts, 0.1, 0.85, W, L * 0.5 - 0.2, 1.45, 0, 0.03, 0, 0.55);
    // Wheels: burnt down to the rims.
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        cyl(parts, 0.34, 0.2, sx * (L * 0.5 - 1.2), 0.36, sz * (W * 0.5 - 0.12), 12, 0, Math.PI / 2);
      }
    }
    return [{ key: 'rust', geo: merge(parts) }];
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // RUBBLE & DEBRIS
  // ═══════════════════════════════════════════════════════════════════════════

  /** A single angular chunk of broken masonry. */
  static rubbleChunk(size: number, rng: Rng): THREE.BufferGeometry {
    if (rng.chance(0.45)) {
      const g = chamferedBox(size * rng.range(0.7, 1.4), size * rng.range(0.35, 0.8), size * rng.range(0.7, 1.3), size * 0.12);
      g.rotateY(rng.range(0, Math.PI));
      g.rotateX(rng.range(-0.4, 0.4));
      return g;
    }
    return rockGeometry(size * 0.6, rng, 0);
  }

  /**
   * A pile of rubble: a low mound of chunks with a few reinforcing bars poking
   * out. Placed at the foot of every damaged wall so the damage has consequence.
   */
  static rubblePile(radius: number, count: number, rng: Rng): GeoParts {
    const stone: THREE.BufferGeometry[] = [];
    const rebar: THREE.BufferGeometry[] = [];
    for (let i = 0; i < count; i++) {
      const a = rng.range(0, Math.PI * 2);
      const r = radius * Math.sqrt(rng.next());
      const falloff = 1 - r / radius;
      const s = rng.range(0.1, 0.34) * (0.55 + falloff);
      const g = Props.rubbleChunk(s, rng);
      g.translate(Math.cos(a) * r, s * 0.35 + falloff * radius * 0.22 * rng.next(), Math.sin(a) * r);
      stone.push(g);
    }
    for (let i = 0; i < Math.max(1, Math.floor(count / 9)); i++) {
      const a = rng.range(0, Math.PI * 2);
      const r = radius * 0.55 * rng.next();
      const len = rng.range(0.5, 1.3);
      const bar = tubeThrough(
        [
          new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r),
          new THREE.Vector3(Math.cos(a) * r + rng.range(-0.1, 0.1), len * 0.6, Math.sin(a) * r + rng.range(-0.1, 0.1)),
          new THREE.Vector3(Math.cos(a) * r + rng.range(-0.35, 0.35), len, Math.sin(a) * r + rng.range(-0.3, 0.3)),
        ],
        0.011,
        4,
        6,
      );
      rebar.push(bar);
    }
    return [
      { key: 'concrete', geo: merge(stone) },
      { key: 'rust', geo: merge(rebar) },
    ];
  }

  /**
   * A wall with its top course blown away: a stepped, ragged crown.
   * `length` runs along X; the wall base sits at y = 0.
   */
  static collapsedWall(length: number, maxHeight: number, thickness: number, rng: Rng): GeoParts {
    const con: THREE.BufferGeometry[] = [];
    const segs = Math.max(4, Math.round(length / 0.75));
    for (let i = 0; i < segs; i++) {
      const w = length / segs;
      const x = -length * 0.5 + w * (i + 0.5);
      // A smooth-ish decay across the break plus per-segment noise: a truly
      // random crown looks like teeth, a decaying one looks like a collapse.
      const t = i / (segs - 1);
      const decay = Math.min(1, 0.25 + 1.35 * Math.pow(1 - t, 1.6));
      const h = Math.max(0.35, maxHeight * decay * rng.range(0.82, 1.06));
      const g = chamferedBox(w * 1.01, h, thickness * rng.range(0.9, 1.0), 0.035);
      g.rotateY(rng.range(-0.02, 0.02));
      g.translate(x, h * 0.5, rng.range(-0.02, 0.02));
      con.push(g);
      // Loose bricks perched on the broken crown.
      if (rng.chance(0.35)) {
        const b = Props.brick();
        b.rotateY(rng.range(0, 3.1));
        b.rotateZ(rng.range(-0.25, 0.25));
        b.translate(x + rng.range(-0.2, 0.2), h + 0.035, rng.range(-0.1, 0.1));
        con.push(b);
      }
    }
    return [{ key: 'concrete', geo: merge(con) }];
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // HERO LANDMARK — the comms tower
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * The compound's comms tower, standing in a dry fountain basin.
   *
   * WHY THIS, AND WHY HERE: an arena map needs one silhouette you can navigate
   * by from anywhere, and one central object that is contested rather than
   * merely decorative. A 15 m lattice mast does both. It is legible from every
   * lane, it throws a long articulated shadow across the courtyard at golden
   * hour (lattice shadows are far more interesting than block shadows), and its
   * 4.6 m maintenance platform is a genuine risk/reward perch — commanding, but
   * with no cover and one way down.
   *
   * The basin is the other half of the design: a 0.95 m rim is a perfect ring
   * of crouch cover around the base, so the centre of the map is contestable
   * from the ground instead of being pure open death-space. And the collapsed
   * upper section tells the story without a single line of dialogue.
   *
   * Returned in local space with the basin floor at y = 0.
   */
  static commsTower(rng: Rng): GeoParts {
    const steel: THREE.BufferGeometry[] = [];
    const concrete: THREE.BufferGeometry[] = [];
    const sand: THREE.BufferGeometry[] = [];
    const accent: THREE.BufferGeometry[] = [];

    // ── Dry fountain basin ───────────────────────────────────────────────────
    const OUT = 4.2;
    const RIM = 0.95;
    const basinProfile: THREE.Vector2[] = [
      new THREE.Vector2(OUT - 0.38, 0),
      new THREE.Vector2(OUT, 0),
      new THREE.Vector2(OUT, RIM - 0.07),
      new THREE.Vector2(OUT - 0.07, RIM),
      new THREE.Vector2(OUT - 0.31, RIM),
      new THREE.Vector2(OUT - 0.38, RIM - 0.07),
      new THREE.Vector2(OUT - 0.38, 0),
    ];
    const basin = new THREE.LatheGeometry(basinProfile, 32);
    concrete.push(basin);
    // Basin floor, drifted with sand — the fountain has not run in years.
    const floor = new THREE.CircleGeometry(OUT - 0.34, 32);
    floor.rotateX(-Math.PI / 2);
    floor.translate(0, 0.14, 0);
    sand.push(floor);
    // Broken central plinth where the fountain head used to be.
    const plinth = chamferedCylinder(0.62, 0.66, 0.05, 8);
    plinth.translate(0, 0.47, 0);
    concrete.push(plinth);
    box(concrete, 0.9, 0.16, 0.9, 0, 0.22, 0, 0.03);
    for (let i = 0; i < 5; i++) {
      const a = rng.range(0, Math.PI * 2);
      const c = Props.rubbleChunk(rng.range(0.12, 0.26), rng);
      c.translate(Math.cos(a) * rng.range(0.9, 3.4), 0.24, Math.sin(a) * rng.range(0.9, 3.4));
      concrete.push(c);
    }

    // ── Lattice mast ─────────────────────────────────────────────────────────
    const BASE = 1.75;   // half-spread of the legs at the footing
    const TOP = 0.5;     // half-spread at the point of the break
    const BREAK = 11.2;  // height at which the mast is bent
    const TIP = 15.4;    // notional height of the intact mast
    const legT = 0.13;

    const halfAt = (y: number): number => BASE + ((TOP - BASE) * Math.min(1, y / TIP));
    // Bend: everything above BREAK leans to the south-east.
    const bendDir = new THREE.Vector3(0.62, 0, 0.48).normalize();
    const bendAngle = 0.62;
    const nodeAt = (y: number, sx: number, sz: number): THREE.Vector3 => {
      const h = halfAt(y);
      const p = new THREE.Vector3(sx * h, y, sz * h);
      if (y > BREAK) {
        const dy = y - BREAK;
        const lean = Math.sin(bendAngle) * dy;
        p.y = BREAK + Math.cos(bendAngle) * dy;
        p.x += bendDir.x * lean;
        p.z += bendDir.z * lean;
      }
      return p;
    };

    const CORNERS: Array<[number, number]> = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    // Node heights: denser near the base where the loads (and the eye) are.
    const levels = [0.9, 2.4, 3.9, 4.6, 6.1, 7.6, 9.1, 10.6, BREAK, 12.4, 13.6, 14.6];

    // Legs, as a chain of struts through every node so the taper is faceted.
    for (const [sx, sz] of CORNERS) {
      let prev = nodeAt(levels[0] - 0.9, sx, sz);
      for (const y of levels) {
        const cur = nodeAt(y, sx, sz);
        strut(steel, prev, cur, legT * (1 - 0.4 * (y / TIP)));
        prev = cur;
      }
    }
    // Horizontal ring braces + face diagonals.
    for (let li = 0; li < levels.length; li++) {
      const y = levels[li];
      for (let c = 0; c < 4; c++) {
        const a = CORNERS[c];
        const b = CORNERS[(c + 1) % 4];
        strut(steel, nodeAt(y, a[0], a[1]), nodeAt(y, b[0], b[1]), 0.06);
        if (li + 1 < levels.length) {
          const y2 = levels[li + 1];
          // Alternate the diagonal direction per bay: real towers do, and it
          // makes the shadow pattern read as a zig-zag instead of a hatch.
          if ((li + c) % 2 === 0) strut(steel, nodeAt(y, a[0], a[1]), nodeAt(y2, b[0], b[1]), 0.045);
          else strut(steel, nodeAt(y, b[0], b[1]), nodeAt(y2, a[0], a[1]), 0.045);
        }
      }
    }
    // Torn steel at the break: three snapped members flapping loose.
    for (let i = 0; i < 3; i++) {
      const a = nodeAt(BREAK, CORNERS[i][0], CORNERS[i][1]);
      const b = a.clone().add(new THREE.Vector3(rng.range(-0.7, 0.7), rng.range(0.5, 1.5), rng.range(-0.7, 0.7)));
      strut(steel, a, b, 0.045);
    }

    // Concrete footings under each leg, sitting in the basin.
    for (const [sx, sz] of CORNERS) {
      box(concrete, 0.62, 0.5, 0.62, sx * BASE, 0.28, sz * BASE, 0.035);
    }

    // ── Maintenance platform at 4.6 m ────────────────────────────────────────
    const PY = 4.6;
    const ph = halfAt(PY) + 0.62;
    const deck = Props.gratedDeck(ph * 2, ph * 2);
    deck.translate(0, PY, 0);
    steel.push(deck);
    for (const [sxr, szr, len, ry] of [
      [0, -1, ph * 2, 0],
      [1, 0, ph * 2, Math.PI / 2],
      [0, 1, ph * 2, 0],
    ] as const) {
      const r = Props.railing(len, 1.05);
      r.rotateY(ry);
      r.translate(sxr * ph, PY + 0.03, szr * ph);
      steel.push(r);
    }
    // Cantilever brackets under the deck.
    for (const [sx, sz] of CORNERS) {
      strut(
        steel,
        new THREE.Vector3(sx * ph * 0.95, PY - 0.05, sz * ph * 0.95),
        new THREE.Vector3(sx * halfAt(3.5) * 0.9, 3.5, sz * halfAt(3.5) * 0.9),
        0.05,
      );
    }
    // Caged ladder from the basin rim up to the platform, on the -Z face.
    const lad = Props.ladder(PY - 0.9, 0.46);
    lad.translate(0, 0.9, -halfAt(2) - 0.28);
    steel.push(lad);
    for (let i = 0; i < 5; i++) {
      const hoop = new THREE.TorusGeometry(0.38, 0.016, 4, 12, Math.PI * 1.25);
      hoop.rotateY(Math.PI / 2);
      hoop.rotateZ(Math.PI * 0.375);
      hoop.translate(0, 1.6 + i * 0.62, -halfAt(2) - 0.5);
      steel.push(hoop);
    }

    // ── Payload: dishes, antennae, cable tray, warning light ─────────────────
    const dishSpecs: Array<[number, number, number]> = [
      [6.2, -2.3, 0.72],   // y, azimuth (rad), radius
      [8.8, 1.15, 0.55],
      [12.6, 2.7, 0.62],
    ];
    for (const [y, az, r] of dishSpecs) {
      const d = Props.satelliteDish(r);
      const node = nodeAt(y, 0, 0);
      d.rotateX(y > BREAK ? -0.5 : -0.06); // the high dish hangs askew after the collapse
      d.rotateY(az);
      d.translate(
        node.x + Math.sin(az) * (halfAt(y) + r * 0.75),
        y > BREAK ? node.y : y,
        node.z + Math.cos(az) * (halfAt(y) + r * 0.75),
      );
      steel.push(d);
    }
    for (const [sx, sz] of [[-1, 1], [1, -1]] as const) {
      const ant = Props.antenna(2.4);
      const n = nodeAt(levels[levels.length - 1], sx, sz);
      ant.translate(n.x, n.y, n.z);
      steel.push(ant);
    }
    // Cable tray running down the -X leg into a junction box at the base.
    const trayPts: THREE.Vector3[] = [];
    for (const y of [13.4, 10.6, 7.6, 4.6, 2.4, 1.0]) {
      const n = nodeAt(y, -1, -1);
      trayPts.push(new THREE.Vector3(n.x - 0.1, n.y, n.z - 0.1));
    }
    trayPts.push(new THREE.Vector3(-BASE - 0.4, 0.5, -BASE - 0.5));
    steel.push(tubeThrough(trayPts, 0.055, 6, 34));
    box(steel, 0.42, 0.55, 0.26, -BASE - 0.45, 0.62, -BASE - 0.62, 0.03);
    // Aircraft warning light at the tip.
    const tipNode = nodeAt(TIP, 0, 0);
    box(accent, 0.2, 0.24, 0.2, tipNode.x, tipNode.y - 0.5, tipNode.z, 0.05);

    // ── Guy wires, anchored well out into the courtyard ──────────────────────
    for (let i = 0; i < 3; i++) {
      const az = (i / 3) * Math.PI * 2 + 0.5;
      const top = nodeAt(9.1, Math.cos(az) > 0 ? 1 : -1, Math.sin(az) > 0 ? 1 : -1);
      const anchor = new THREE.Vector3(Math.cos(az) * 8.6, 0.35, Math.sin(az) * 8.6);
      steel.push(Props.cable(top, anchor, 0.35, 0.022));
      box(concrete, 0.75, 0.55, 0.75, anchor.x, 0.22, anchor.z, 0.04);
    }

    return [
      { key: 'rust', geo: merge(steel) },
      { key: 'concrete', geo: merge(concrete) },
      { key: 'sand', geo: merge(sand) },
      { key: 'accentOchre', geo: merge(accent) },
    ];
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SCAFFOLDING
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * A tube-and-board scaffold with `decks` walkable levels and a ladder run.
   * Stands along +X with the building face at -Z.
   *
   * Scaffolding is free verticality: it adds two firing positions and a
   * climbable route without carving anything out of the level's footprint,
   * and its open frame keeps the lane below readable from above.
   */
  static scaffold(length: number, decks: number, deckHeight = 2.6, depth = 1.35): GeoParts {
    const steel: THREE.BufferGeometry[] = [];
    const wood: THREE.BufferGeometry[] = [];
    const bays = Math.max(1, Math.round(length / 2.5));
    const total = decks * deckHeight;

    for (let i = 0; i <= bays; i++) {
      const x = -length * 0.5 + (length * i) / bays;
      for (const sz of [-1, 1]) {
        cyl(steel, 0.026, total + 0.5, x, (total + 0.5) * 0.5, sz * depth * 0.5, 6);
        box(steel, 0.16, 0.05, 0.16, x, 0.025, sz * depth * 0.5, 0.012); // base plate
      }
      // Transoms tying the two standards together at every deck.
      for (let d = 1; d <= decks; d++) {
        cyl(steel, 0.022, depth, x, d * deckHeight, 0, 6, Math.PI / 2);
      }
    }
    for (let d = 1; d <= decks; d++) {
      const y = d * deckHeight;
      for (const sz of [-1, 1]) {
        cyl(steel, 0.024, length, 0, y, sz * depth * 0.5, 6, 0, Math.PI / 2);
        cyl(steel, 0.024, length, 0, y + 1.0, sz * depth * 0.5, 6, 0, Math.PI / 2); // guard rail
      }
      // Boards.
      for (let b = 0; b < 3; b++) {
        box(wood, length, 0.038, depth / 3 - 0.02, 0, y + 0.045, -depth * 0.5 + (depth / 3) * (b + 0.5), 0.008);
      }
      // Facade braces, alternating direction bay to bay.
      for (let i = 0; i < bays; i++) {
        const x0 = -length * 0.5 + (length * i) / bays;
        const x1 = -length * 0.5 + (length * (i + 1)) / bays;
        const flip = (i + d) % 2 === 0;
        strut(
          steel,
          new THREE.Vector3(flip ? x0 : x1, y - deckHeight, depth * 0.5),
          new THREE.Vector3(flip ? x1 : x0, y, depth * 0.5),
          0.038,
        );
      }
    }
    // Access ladder at the far end, running the full height.
    const lad = Props.ladder(total + 0.4, 0.42);
    lad.rotateY(Math.PI / 2);
    lad.translate(length * 0.5 - 0.6, 0, depth * 0.5 + 0.16);
    steel.push(lad);

    return [
      { key: 'metal', geo: merge(steel) },
      { key: 'wood', geo: merge(wood) },
    ];
  }

  /**
   * A plank gangway spanning two rooftops. Boards, kerb rails and a pair of
   * carrying beams. Rooftop-to-rooftop links are what turn a set of roofs into
   * an actual third lane rather than four disconnected sniper nests.
   */
  static gangway(span: number, width = 1.2): GeoParts {
    const wood: THREE.BufferGeometry[] = [];
    const steel: THREE.BufferGeometry[] = [];
    for (const sx of [-1, 1]) box(steel, 0.09, 0.16, span, sx * (width * 0.5 - 0.1), -0.09, 0, 0.02);
    const boards = Math.max(4, Math.round(span / 0.42));
    for (let i = 0; i < boards; i++) {
      const z = -span * 0.5 + (span / boards) * (i + 0.5);
      box(wood, width, 0.04, span / boards - 0.014, 0, 0, z, 0.008);
    }
    for (const sx of [-1, 1]) {
      box(wood, 0.07, 0.14, span, sx * width * 0.5, 0.07, 0, 0.012);
      // Single-rail handhold: enough to feel safe, low enough to shoot over.
      for (let i = 0; i <= 3; i++) {
        box(steel, 0.04, 0.95, 0.04, sx * width * 0.5, 0.5, -span * 0.5 + (span * i) / 3, 0.008);
      }
      cyl(steel, 0.022, span, sx * width * 0.5, 0.95, 0, 6, Math.PI / 2);
    }
    return [
      { key: 'wood', geo: merge(wood) },
      { key: 'metal', geo: merge(steel) },
    ];
  }

  /** A barrel with a fire in it — burnt, blackened, ringed with soot. */
  static fireBarrel(): GeoParts {
    const parts: THREE.BufferGeometry[] = [];
    const drum = Props.oilDrum();
    parts.push(drum);
    // Vent holes punched round the base and a ragged open top.
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2;
      box(parts, 0.09, 0.11, 0.06, Math.cos(a) * 0.26, 0.18, Math.sin(a) * 0.26, 0.012, -a);
    }
    return [{ key: 'rust', geo: merge(parts) }];
  }

  /** A stack of tyres. */
  static tyrePile(count: number, rng: Rng): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    for (let i = 0; i < count; i++) {
      const t = Props.tyre();
      t.rotateY(rng.range(0, Math.PI));
      t.rotateX(rng.range(-0.05, 0.05));
      t.translate(rng.range(-0.05, 0.05), i * 0.19, rng.range(-0.05, 0.05));
      parts.push(t);
    }
    return merge(parts);
  }

  /** A roof-edge coping / parapet cap running along X. */
  static coping(length: number, width: number): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, length, 0.075, width + 0.12, 0, 0, 0, 0.02);
    // Drip nose either side so the cap throws a shadow line down the facade.
    for (const sz of [-1, 1]) box(parts, length, 0.045, 0.05, 0, -0.05, sz * (width * 0.5 + 0.05), 0.012);
    return merge(parts);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FACADE RELIEF
  //
  // Everything below exists for one reason: an extruded rectangle with a
  // texture on it is not architecture. What makes a building read is the
  // *silhouette break* and the *self-shadow* — a 12 cm projecting band throws a
  // hard horizontal shadow line across a wall at golden hour, and that single
  // line does more work than any amount of albedo detail. All of it merges into
  // the wall's existing material bucket, so the entire relief pass costs zero
  // additional draw calls; only triangles.
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * A single louvred shutter leaf, hinged on its local -X edge, face toward +Z.
   * Geometry is centred on the origin so the caller positions the leaf centre.
   *
   * Real shutters are almost never all at the same angle — half are latched
   * back flat, some hang at 30 degrees, one is closed because that room is not
   * in use. That variation is most of what stops a terrace looking stamped.
   */
  static shutterLeaf(w = 0.55, h = 1.3, openRad = 0, side: -1 | 1 = -1): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, w, h, 0.032, 0, 0, 0, 0.008);
    // Stiles and rails standing proud of the panel.
    for (const sx of [-1, 1]) box(parts, 0.055, h, 0.022, sx * (w * 0.5 - 0.028), 0, 0.026, 0.005);
    for (const sy of [-1, 1]) box(parts, w - 0.09, 0.065, 0.022, 0, sy * (h * 0.5 - 0.035), 0.026, 0.005);
    // Tilted louvre slats. The tilt is what catches a rim of light per slat.
    const n = Math.max(3, Math.round(h / 0.13));
    for (let i = 0; i < n; i++) {
      const y = -h * 0.5 + ((h - 0.1) / n) * (i + 0.5) + 0.05;
      box(parts, w - 0.11, 0.05, 0.028, 0, y, 0.02, 0.005, 0, 0, -0.5);
    }
    // Hinge pintles on the hung edge.
    for (const sy of [-0.36, 0.36]) cyl(parts, 0.018, 0.07, -side * (w * 0.5 - 0.01), sy * h, 0.012, 6);
    const g = merge(parts);
    // Re-origin onto the hinge edge so the caller positions the *hinge*, not
    // the leaf centre — which is the only way a closed leaf and an open one can
    // share a placement. `side` is which side of the opening the leaf hangs on:
    // -1 hinges at its own -X edge, +1 at its +X edge.
    g.translate(-side * w * 0.5, 0, 0);
    // Sign convention: `openRad` of the same sign as `side` swings the free
    // edge toward +Z, i.e. out of the wall, which is the only useful direction.
    if (openRad) g.rotateY(openRad);
    return g;
  }

  /**
   * Rough planks nailed across an opening. Centred on the origin, plane in XY.
   *
   * Boarded windows are the cheapest way to say "this place emptied out in a
   * hurry", and they break up a repeated window rhythm without costing a new
   * material.
   */
  static windowBoards(w: number, h: number, rng: Rng): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const n = rng.int(3, 5);
    for (let i = 0; i < n; i++) {
      const y = (((i + 0.5) / n) - 0.5) * h * 1.02;
      const len = (w + 0.22) * rng.range(1.0, 1.1);
      box(parts, len, rng.range(0.12, 0.19), 0.026, rng.range(-0.05, 0.05), y, 0, 0.006, 0, 0, rng.range(-0.09, 0.09));
    }
    // One diagonal brace: the give-away that a human did this in a hurry.
    box(
      parts,
      Math.hypot(w, h) * 1.02, rng.range(0.13, 0.17), 0.024,
      0, 0, 0.024, 0.006, 0, 0, Math.atan2(h, w) * rng.sign(),
    );
    return merge(parts);
  }

  /**
   * Shards of glass still clinging to a broken pane. Plane in XY, centred.
   *
   * Triangles anchored on the frame edges with the apex pulled toward the
   * centre: that is exactly the fracture pattern of a pane hit once and cleared
   * by hand, and it costs about 20 triangles.
   */
  static brokenGlass(w: number, h: number, rng: Rng): THREE.BufferGeometry {
    const pos: number[] = [];
    const hw = w * 0.5;
    const hh = h * 0.5;
    const edges: ReadonlyArray<readonly [number, number, number, number]> = [
      [-hw, hh, hw, hh],
      [hw, -hh, hw, hh],
      [-hw, -hh, hw, -hh],
      [-hw, -hh, -hw, hh],
    ];
    for (const [x0, y0, x1, y1] of edges) {
      const n = rng.int(2, 4);
      for (let i = 0; i < n; i++) {
        const t0 = i / n + rng.range(0, 0.14);
        const t1 = (i + 1) / n - rng.range(0, 0.14);
        if (t1 - t0 < 0.05) continue;
        const ax = x0 + (x1 - x0) * t0;
        const ay = y0 + (y1 - y0) * t0;
        const bx = x0 + (x1 - x0) * t1;
        const by = y0 + (y1 - y0) * t1;
        const k = rng.range(0.2, 0.6);
        pos.push(ax, ay, 0, bx, by, 0, (ax + bx) * 0.5 * (1 - k), (ay + by) * 0.5 * (1 - k), 0);
      }
    }
    const g = new THREE.BufferGeometry();
    if (pos.length === 0) return g;
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
  }

  /**
   * The dark box behind a window on a building you cannot enter.
   *
   * This is the single most valuable window detail there is. A glazed opening
   * with nothing behind it reads as a light rectangle painted on a wall no
   * matter how good the frame is; 60 cm of unlit recess reads as a room. Six
   * faces of a box, minus the one facing the player.
   */
  static windowRecess(w: number, h: number, depth = 0.6): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const t = 0.05;
    box(parts, w, h, t, 0, 0, -depth, 0.01);                                   // back
    for (const sx of [-1, 1]) box(parts, t, h, depth, sx * (w * 0.5), 0, -depth * 0.5, 0.01);
    for (const sy of [-1, 1]) box(parts, w, t, depth, 0, sy * (h * 0.5), -depth * 0.5, 0.01);
    return merge(parts);
  }

  /**
   * A drainage spout through a parapet, projecting toward +Z and tipped down.
   * Flat roofs in this region drain through the parapet, and the stain running
   * down the wall below each spout is half the character of an old building.
   */
  static wallScupper(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.15, 0.10, 0.44, 0, 0, 0.14, 0.012, 0, -0.16);
    box(parts, 0.26, 0.17, 0.07, 0, 0.02, -0.06, 0.012);
    return merge(parts);
  }

  /** An electrical junction / meter box bolted to a wall, lid facing +Z. */
  static conduitBox(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.26, 0.34, 0.12, 0, 0, 0.06, 0.014);
    box(parts, 0.20, 0.27, 0.03, 0, 0, 0.13, 0.008);
    cyl(parts, 0.02, 0.07, 0, -0.19, 0.06, 6);
    cyl(parts, 0.02, 0.07, 0, 0.19, 0.06, 6);
    return merge(parts);
  }

  /**
   * A gooseneck wall lamp with a conical shade. Mounts at the origin on a wall
   * whose outward normal is +Z; the shade hangs about 40 cm proud.
   */
  static wallLamp(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.11, 0.15, 0.05, 0, 0, 0.025, 0.01);
    parts.push(
      tubeThrough(
        [
          new THREE.Vector3(0, 0.02, 0.05),
          new THREE.Vector3(0, 0.16, 0.22),
          new THREE.Vector3(0, 0.13, 0.40),
        ],
        0.014, 5, 8,
      ),
    );
    const shade = new THREE.CylinderGeometry(0.145, 0.05, 0.11, 12, 1, true);
    shade.translate(0, -0.05, 0);
    shade.translate(0, 0.13, 0.40);
    parts.push(shade);
    return merge(parts);
  }

  /** A painted sign panel on stand-off brackets, face toward +Z. */
  static signBoard(w = 1.6, h = 0.5): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, w, h, 0.045, 0, 0, 0.055, 0.010);
    for (const sy of [-1, 1]) box(parts, w + 0.07, 0.055, 0.075, 0, sy * (h * 0.5 + 0.01), 0.055, 0.01);
    for (const sx of [-1, 1]) box(parts, 0.035, 0.05, 0.09, sx * w * 0.38, 0, 0.01, 0.008);
    return merge(parts);
  }

  /**
   * A short raised pier on a parapet run. Repeated at intervals it turns a dead
   * horizontal roof line into a punctuated one, which is most of what makes a
   * skyline read as built rather than extruded.
   */
  static parapetPier(w = 0.5, h = 0.42, t = 0.3): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, w, h, t + 0.05, 0, h * 0.5, 0, 0.02);
    box(parts, w + 0.11, 0.07, t + 0.16, 0, h + 0.035, 0, 0.015);
    return merge(parts);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // ROOFSCAPE AND SKYLINE
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * A roof-access stair bulkhead: a small hut with a mono-pitch roof and a
   * painted steel door. Every flat roof in the region has one, it is the
   * largest single object on a rooftop, and it doubles as hard cover on what is
   * otherwise an open deck.
   */
  static stairBulkhead(w = 2.2, d = 2.0, h = 2.35, accent = 'accentTeal'): GeoParts {
    const wall: THREE.BufferGeometry[] = [];
    const cap: THREE.BufferGeometry[] = [];
    const door: THREE.BufferGeometry[] = [];
    const t = 0.16;
    // Three solid walls; the +Z wall carries the door opening as two piers plus
    // a lintel, which is cheaper and cleaner than punching a hole.
    box(wall, w, h, t, 0, h * 0.5, -d * 0.5 + t * 0.5, 0.025);
    for (const sx of [-1, 1]) box(wall, t, h, d - t * 2, sx * (w * 0.5 - t * 0.5), h * 0.5, 0, 0.025);
    const openW = 0.95;
    for (const sx of [-1, 1]) {
      box(wall, (w - openW) * 0.5, h, t, sx * (w + openW) * 0.25, h * 0.5, d * 0.5 - t * 0.5, 0.025);
    }
    box(wall, openW, h - 2.1, t, 0, (h + 2.1) * 0.5, d * 0.5 - t * 0.5, 0.02);
    // Roof slab, tipped so water runs off the back, with a projecting drip edge.
    const slab = chamferedBox(w + 0.24, 0.13, d + 0.24, 0.025);
    slab.rotateX(0.07);
    slab.translate(0, h + 0.08, 0);
    cap.push(slab);
    box(door, openW - 0.05, 2.05, 0.045, 0.08, 1.03, d * 0.5 + 0.07, 0.012, 0.32);
    return [
      { key: 'plasterAlt', geo: merge(wall) },
      { key: 'concrete', geo: merge(cap) },
      { key: accent, geo: merge(door) },
    ];
  }

  /**
   * A rooftop water tank on a block stand. Extracted from the level so every
   * roof gets the same silhouette vocabulary, with the feed pipe and the
   * overflow dribble included.
   */
  static waterTank(radius = 0.55, height = 1.15): GeoParts {
    const shell: THREE.BufferGeometry[] = [];
    const steel: THREE.BufferGeometry[] = [];
    const stand: THREE.BufferGeometry[] = [];
    const body = chamferedCylinder(radius, height, 0.06, 14);
    body.translate(0, 0.62 + height * 0.5, 0);
    shell.push(body);
    // Ribbing hoops: two lathe bands read as a pressed-steel tank at 30 m.
    for (const f of [0.3, 0.72]) {
      const hoop = chamferedCylinder(radius * 1.03, 0.05, 0.012, 14, false);
      hoop.translate(0, 0.62 + height * f, 0);
      shell.push(hoop);
    }
    const lid = chamferedCylinder(radius * 0.34, 0.07, 0.02, 10);
    lid.translate(radius * 0.35, 0.62 + height + 0.035, 0);
    steel.push(lid);
    box(steel, radius * 1.8, 0.08, radius * 1.8, 0, 0.6, 0, 0.02);
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        box(stand, 0.22, 0.58, 0.22, sx * radius * 0.62, 0.29, sz * radius * 0.62, 0.02);
      }
    }
    cyl(steel, 0.032, 0.66, radius * 0.85, 0.33, 0, 8);
    return [
      { key: 'plasterPale', geo: merge(shell) },
      { key: 'metal', geo: merge(steel) },
      { key: 'concrete', geo: merge(stand) },
    ];
  }

  /**
   * A legged water tower — a tank carried 5-8 m up on a braced steel frame.
   * One of these on the skyline is worth a dozen rooftop props, because it is
   * the only thing in the compound taller than a building other than the mast.
   */
  static waterTower(height = 7.0, radius = 1.35): GeoParts {
    const steel: THREE.BufferGeometry[] = [];
    const tank: THREE.BufferGeometry[] = [];
    const spread = radius * 1.05;
    const legs: THREE.Vector3[] = [];
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + Math.PI * 0.25;
      legs.push(new THREE.Vector3(Math.cos(a) * spread, 0, Math.sin(a) * spread));
    }
    for (let i = 0; i < 4; i++) {
      const foot = legs[i];
      // Legs rake inward, which is both structurally honest and reads as depth.
      const top = new THREE.Vector3(foot.x * 0.62, height, foot.z * 0.62);
      strut(steel, foot, top, 0.11);
      const nFoot = legs[(i + 1) % 4];
      const nTop = new THREE.Vector3(nFoot.x * 0.62, height, nFoot.z * 0.62);
      // Two levels of X-bracing: the lattice shadow is the point.
      for (const [a, b] of [[0.08, 0.5], [0.5, 0.94]] as const) {
        strut(
          steel,
          new THREE.Vector3().lerpVectors(foot, top, a),
          new THREE.Vector3().lerpVectors(nFoot, nTop, b),
          0.045,
        );
        strut(
          steel,
          new THREE.Vector3().lerpVectors(nFoot, nTop, a),
          new THREE.Vector3().lerpVectors(foot, top, b),
          0.045,
        );
      }
      strut(
        steel,
        new THREE.Vector3().lerpVectors(foot, top, 0.5),
        new THREE.Vector3().lerpVectors(nFoot, nTop, 0.5),
        0.05,
      );
    }
    const body = chamferedCylinder(radius, 2.0, 0.14, 16);
    body.translate(0, height + 1.0, 0);
    tank.push(body);
    const cone = new THREE.ConeGeometry(radius * 1.02, 0.55, 16, 1, true);
    cone.translate(0, height + 2.24, 0);
    tank.push(cone);
    const skirt = new THREE.ConeGeometry(radius * 1.02, 0.5, 16, 1, true);
    skirt.rotateX(Math.PI);
    skirt.translate(0, height - 0.24, 0);
    tank.push(skirt);
    cyl(steel, 0.05, height, radius * 0.5, height * 0.5, radius * 0.5, 6);
    return [
      { key: 'rust', geo: merge(steel) },
      { key: 'metal', geo: merge(tank) },
    ];
  }

  /**
   * A rooftop TV aerial: a mast with a graduated ladder of crossbars.
   * Pure silhouette, ~120 triangles, and instantly reads as "inhabited".
   */
  static tvAerial(height = 2.4): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    cyl(parts, 0.022, height, 0, height * 0.5, 0, 6);
    const n = 6;
    for (let i = 0; i < n; i++) {
      const y = height * (0.42 + 0.55 * (i / (n - 1)));
      const len = 0.95 - 0.1 * i;
      cyl(parts, 0.011, len, 0, y, 0, 5, 0, Math.PI / 2);
    }
    // Reflector at the back and a folded dipole at the front.
    cyl(parts, 0.011, 1.02, 0, height * 0.38, -0.14, 5, 0, Math.PI / 2);
    box(parts, 0.16, 0.05, 0.16, 0, 0.03, 0, 0.012);
    for (const sx of [-1, 1]) strut(parts, new THREE.Vector3(0, height * 0.3, 0), new THREE.Vector3(sx * 0.3, 0.05, 0), 0.018);
    return merge(parts);
  }

  /** A rendered masonry flue with a projecting cap. Roof silhouette, 2 boxes. */
  static chimneyStack(height = 1.5, w = 0.5): GeoParts {
    const body: THREE.BufferGeometry[] = [];
    const cap: THREE.BufferGeometry[] = [];
    box(body, w, height, w, 0, height * 0.5, 0, 0.025);
    box(cap, w + 0.16, 0.1, w + 0.16, 0, height + 0.05, 0, 0.02);
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) box(cap, 0.09, 0.2, 0.09, sx * w * 0.3, height + 0.2, sz * w * 0.3, 0.015);
    }
    box(cap, w * 0.85, 0.06, w * 0.85, 0, height + 0.31, 0, 0.015);
    return [{ key: 'plasterAlt', geo: merge(body) }, { key: 'concrete', geo: merge(cap) }];
  }

  /**
   * A timber utility pole with a crossarm and insulators. Poles are the cheapest
   * possible vertical punctuation for a skyline, and the wires between them do
   * the rest of the work for a few hundred triangles.
   */
  static utilityPole(height = 8.0): GeoParts {
    const timber: THREE.BufferGeometry[] = [];
    const steel: THREE.BufferGeometry[] = [];
    const pole = chamferedCylinder(0.115, height, 0.03, 8);
    pole.translate(0, height * 0.5, 0);
    timber.push(pole);
    for (const [y, len] of [[height - 0.5, 1.85], [height - 1.25, 1.35]] as const) {
      box(timber, len, 0.11, 0.13, 0, y, 0, 0.02);
      const n = len > 1.6 ? 4 : 3;
      for (let i = 0; i < n; i++) {
        const x = -len * 0.5 + (len * (i + 0.5)) / n;
        cyl(steel, 0.045, 0.14, x, y + 0.13, 0, 7);
        cyl(steel, 0.016, 0.11, x, y + 0.06, 0, 5);
      }
      // Diagonal knee braces back to the pole.
      for (const sx of [-1, 1]) {
        strut(steel, new THREE.Vector3(sx * len * 0.38, y - 0.05, 0), new THREE.Vector3(sx * 0.1, y - 0.55, 0), 0.035);
      }
    }
    // Step bolts and a transformer can on the bigger poles.
    for (let i = 0; i < 5; i++) {
      cyl(steel, 0.014, 0.3, 0, 1.6 + i * 0.55, 0, 5, 0, Math.PI / 2);
    }
    const can = chamferedCylinder(0.22, 0.55, 0.05, 10);
    can.translate(0.28, height - 2.4, 0);
    steel.push(can);
    return [{ key: 'wood', geo: merge(timber) }, { key: 'metal', geo: merge(steel) }];
  }

  /**
   * A minaret standing beyond the compound wall: octagonal shaft, corbelled
   * gallery, conical cap, finial.
   *
   * WHY: the map's horizon is otherwise a flat band of two-storey parapets. One
   * tall, unmistakable, *unreachable* form outside the play space gives the
   * whole compound a sense of a town continuing past it, and gives players a
   * fixed compass bearing that is visible from every lane.
   */
  static minaret(height = 21): GeoParts {
    const stone: THREE.BufferGeometry[] = [];
    const accent: THREE.BufferGeometry[] = [];
    const baseH = height * 0.1;
    const shaftH = height * 0.6;
    const galleryY = baseH + shaftH;
    box(stone, 3.0, baseH, 3.0, 0, baseH * 0.5, 0, 0.06);
    box(stone, 2.7, 0.18, 2.7, 0, baseH + 0.09, 0, 0.04);
    const shaft = chamferedCylinder(1.05, shaftH, 0.08, 8);
    shaft.translate(0, baseH + shaftH * 0.5, 0);
    stone.push(shaft);
    // Two banding courses up the shaft, and slit openings punched as recesses.
    for (const f of [0.34, 0.68]) {
      const band = chamferedCylinder(1.14, 0.16, 0.03, 8, false);
      band.translate(0, baseH + shaftH * f, 0);
      stone.push(band);
    }
    for (let i = 0; i < 4; i++) {
      const a = (i / 4) * Math.PI * 2 + 0.4;
      box(accent, 0.32, 0.95, 0.3, Math.cos(a) * 1.0, baseH + shaftH * 0.52, Math.sin(a) * 1.0, 0.02, -a);
    }
    // Corbelled gallery.
    const corbel = chamferedCylinder(1.5, 0.22, 0.06, 12, false);
    corbel.translate(0, galleryY + 0.11, 0);
    stone.push(corbel);
    const rail = chamferedCylinder(1.55, 0.85, 0.05, 12, false);
    rail.translate(0, galleryY + 0.65, 0);
    stone.push(rail);
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      box(accent, 0.14, 0.55, 0.14, Math.cos(a) * 1.52, galleryY + 0.5, Math.sin(a) * 1.52, 0.02, -a);
    }
    // Upper stage and cap.
    const upperH = height - galleryY - 1.2 - height * 0.14;
    const upper = chamferedCylinder(0.78, Math.max(0.6, upperH), 0.06, 8);
    upper.translate(0, galleryY + 1.1 + Math.max(0.6, upperH) * 0.5, 0);
    stone.push(upper);
    const capY = galleryY + 1.1 + Math.max(0.6, upperH);
    const cone = new THREE.ConeGeometry(0.95, height * 0.14, 10);
    cone.translate(0, capY + height * 0.07, 0);
    accent.push(cone);
    cyl(accent, 0.05, 0.8, 0, capY + height * 0.14 + 0.4, 0, 6);
    const finial = new THREE.SphereGeometry(0.17, 8, 6);
    finial.translate(0, capY + height * 0.14 + 0.82, 0);
    accent.push(finial);
    return [{ key: 'plasterPale', geo: merge(stone) }, { key: 'accentTeal', geo: merge(accent) }];
  }

  /**
   * A simple background massing block with a parapet and recessed window
   * pockets, for the town beyond the compound wall.
   *
   * These are never entered and never shot at, so they are pure silhouette:
   * a box, a coping, and shallow recesses that catch enough shadow to stop the
   * face reading as a flat card.
   */
  static backdropBlock(w: number, h: number, d: number, rng: Rng): GeoParts {
    const wall: THREE.BufferGeometry[] = [];
    const cap: THREE.BufferGeometry[] = [];
    box(wall, w, h, d, 0, h * 0.5, 0, 0.06);
    // Parapet ring: a lip standing above the roof deck on all four sides.
    const pH = rng.range(0.55, 1.1);
    for (const sz of [-1, 1]) box(cap, w, pH, 0.26, 0, h + pH * 0.5, sz * (d * 0.5 - 0.13), 0.03);
    for (const sx of [-1, 1]) box(cap, 0.26, pH, d - 0.52, sx * (w * 0.5 - 0.13), h + pH * 0.5, 0, 0.03);
    for (const sz of [-1, 1]) box(cap, w + 0.16, 0.08, 0.4, 0, h + pH + 0.04, sz * (d * 0.5 - 0.13), 0.02);
    // Window pockets on the two faces that can be seen from inside the compound.
    const cols = Math.max(2, Math.floor(w / 3.2));
    const rows = Math.max(1, Math.floor(h / 3.3));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (rng.chance(0.22)) continue;
        const x = -w * 0.5 + (w * (c + 0.5)) / cols;
        const y = 1.2 + r * 3.3;
        if (y + 0.7 > h) continue;
        for (const sz of [-1, 1]) {
          box(wall, 1.05, 1.25, 0.5, x, y, sz * (d * 0.5 - 0.24), 0.02);
        }
      }
    }
    return [{ key: 'plasterAlt', geo: merge(wall) }, { key: 'plasterPale', geo: merge(cap) }];
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // GROUND CLUTTER
  //
  // Real debris is not evenly distributed. It piles against obstructions —
  // kerbs, wall feet, the lee side of anything solid — because that is where
  // wind and foot traffic leave it. Everything here is small, instanced, and
  // mostly shadow-free: at 25 cm the shadow is two pixels and costs a full
  // shadow-map draw for nothing.
  // ═══════════════════════════════════════════════════════════════════════════

  /** A builder's wheelbarrow, tipped or upright. Steel tray, timber handles. */
  static wheelbarrow(rng: Rng): GeoParts {
    const steel: THREE.BufferGeometry[] = [];
    const timber: THREE.BufferGeometry[] = [];
    const rubber: THREE.BufferGeometry[] = [];
    // Tray: a shallow tapered pan, faked as a floor plus four splayed sides.
    box(steel, 0.62, 0.045, 0.86, 0, 0.42, 0, 0.015);
    for (const sx of [-1, 1]) box(steel, 0.05, 0.30, 0.86, sx * 0.32, 0.55, 0, 0.015, 0, 0, sx * -0.22);
    box(steel, 0.62, 0.28, 0.05, 0, 0.55, -0.43, 0.015, 0, 0.24);
    box(steel, 0.72, 0.24, 0.05, 0, 0.53, 0.43, 0.015, 0, -0.3);
    for (const sx of [-1, 1]) {
      box(timber, 0.055, 0.055, 1.5, sx * 0.26, 0.34, 0.24, 0.012, 0, -0.13);
      box(steel, 0.05, 0.34, 0.05, sx * 0.26, 0.17, -0.2, 0.01);
    }
    const wheel = new THREE.TorusGeometry(0.14, 0.055, 6, 12);
    wheel.rotateY(Math.PI / 2);
    wheel.translate(0, 0.15, -0.62);
    rubber.push(wheel);
    cyl(steel, 0.035, 0.1, 0, 0.15, -0.62, 8, 0, Math.PI / 2);
    const g: GeoParts = [
      { key: 'metal', geo: merge(steel) },
      { key: 'wood', geo: merge(timber) },
      { key: 'rubber', geo: merge(rubber) },
    ];
    if (rng.chance(0.35)) for (const p of g) p.geo.rotateZ(1.35);
    return g;
  }

  /** A collapsed cardboard box or a stacked one. ~0.5 m. */
  static cardboardBox(size = 0.46): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, size, size * 0.72, size * 0.8, 0, size * 0.36, 0, 0.012);
    // Open flaps, splayed: the silhouette break that says "cardboard".
    for (const sz of [-1, 1]) {
      box(parts, size, 0.014, size * 0.36, 0, size * 0.76, sz * size * 0.52, 0.006, 0, sz * 0.6);
    }
    return merge(parts);
  }

  /** A slumped hessian sack of sand or cement. */
  static sack(rng: Rng): THREE.BufferGeometry {
    const g = new THREE.SphereGeometry(0.28, 8, 6);
    const pos = g.getAttribute('position');
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      // Squash vertically and pinch the ends: a sack, not a ball.
      const pinch = 1 - 0.42 * Math.abs(x) / 0.28;
      pos.setXYZ(i, x * 1.5, Math.max(-0.16, y * 0.6) * pinch + 0.16, z * 0.95 * pinch);
    }
    g.computeVertexNormals();
    g.rotateY(rng.range(0, Math.PI));
    return g;
  }

  /** A stamped-steel jerrycan. 0.47 m tall, the standard 20 litre pattern. */
  static jerrycan(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.34, 0.44, 0.17, 0, 0.24, 0, 0.025);
    // The two pressed X-channels on the face, and the triple handle.
    for (const sz of [-1, 1]) {
      box(parts, 0.24, 0.05, 0.02, 0, 0.26, sz * 0.088, 0.008, 0, 0, 0.62);
      box(parts, 0.24, 0.05, 0.02, 0, 0.26, sz * 0.088, 0.008, 0, 0, -0.62);
    }
    for (let i = -1; i <= 1; i++) box(parts, 0.05, 0.05, 0.12, i * 0.11, 0.48, 0, 0.012);
    box(parts, 0.34, 0.045, 0.13, 0, 0.47, 0, 0.014);
    cyl(parts, 0.035, 0.06, 0.11, 0.5, 0.055, 8);
    return merge(parts);
  }

  /** A timber cable drum lying on its side or standing. 1.1 m diameter. */
  static cableSpool(): GeoParts {
    const timber: THREE.BufferGeometry[] = [];
    const cable: THREE.BufferGeometry[] = [];
    for (const sx of [-1, 1]) {
      const cheek = chamferedCylinder(0.55, 0.06, 0.02, 16);
      cheek.rotateZ(Math.PI / 2);
      cheek.translate(sx * 0.28, 0.55, 0);
      timber.push(cheek);
    }
    const hub = chamferedCylinder(0.24, 0.5, 0.02, 12);
    hub.rotateZ(Math.PI / 2);
    hub.translate(0, 0.55, 0);
    timber.push(hub);
    const wound = chamferedCylinder(0.42, 0.46, 0.03, 16);
    wound.rotateZ(Math.PI / 2);
    wound.translate(0, 0.55, 0);
    cable.push(wound);
    return [{ key: 'wood', geo: merge(timber) }, { key: 'rubber', geo: merge(cable) }];
  }

  /**
   * A scrap of windblown litter: a crumpled sheet lying nearly flat.
   * Two triangles' worth of nothing that reads, in aggregate, as a lived-in
   * street. Never casts a shadow.
   */
  static litterScrap(rng: Rng): THREE.BufferGeometry {
    const g = new THREE.PlaneGeometry(rng.range(0.16, 0.32), rng.range(0.12, 0.26), 2, 2);
    g.rotateX(-Math.PI / 2);
    const pos = g.getAttribute('position');
    for (let i = 0; i < pos.count; i++) pos.setY(i, rng.range(0, 0.045));
    g.computeVertexNormals();
    g.rotateY(rng.range(0, Math.PI * 2));
    return g;
  }

  /**
   * A tarpaulin thrown over a stack and lashed down. Four draped faces plus a
   * sagging top; the only saturated blue in the compound.
   */
  static tarpSheet(w: number, d: number, h: number, rng: Rng): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    const top = saggingCloth(w, d, 0.09, 6, 5, rng);
    top.translate(0, h, 0);
    parts.push(top);
    // Skirts hanging down the four sides, each with its own droop and flare.
    for (const [ry, span] of [[0, w], [Math.PI / 2, d], [Math.PI, w], [-Math.PI / 2, d]] as const) {
      const drop = rng.range(0.28, 0.55);
      const skirt = saggingCloth(span * 1.02, drop, 0.05, 5, 3, rng);
      skirt.rotateX(Math.PI / 2);
      skirt.rotateY(ry);
      skirt.translate(
        Math.sin(ry) * (ry === 0 || Math.abs(ry) === Math.PI ? 0 : (span === d ? w : d) * 0.5),
        h - 0.01,
        Math.cos(ry) * (span === w ? d * 0.5 : 0),
      );
      parts.push(skirt);
    }
    return merge(parts);
  }

  /** A short length of loose cable snaking across the ground. */
  static groundCable(length: number, rng: Rng): THREE.BufferGeometry {
    const pts: THREE.Vector3[] = [];
    const n = 5;
    for (let i = 0; i <= n; i++) {
      pts.push(new THREE.Vector3((i / n - 0.5) * length, 0.022, rng.range(-0.45, 0.45)));
    }
    return tubeThrough(pts, 0.022, 5, 20);
  }

  /**
   * A shallow "Juliet" balcony under an upper window: a projecting slab on
   * corbels with a railing.
   *
   * Worth far more than its 400 triangles. It is the only thing on an upper
   * storey that projects far enough to cast a *cast* shadow onto the wall
   * beneath it rather than a contact one, and a hard diagonal shadow halfway up
   * an elevation is exactly what tells the eye the wall is a plane in space.
   */
  static julietBalcony(w = 1.7, depth = 0.55): GeoParts {
    const stone: THREE.BufferGeometry[] = [];
    const steel: THREE.BufferGeometry[] = [];
    box(stone, w, 0.1, depth, 0, 0.05, depth * 0.5 - 0.04, 0.02);
    box(stone, w + 0.12, 0.055, depth + 0.08, 0, 0.115, depth * 0.5 - 0.04, 0.016);
    // Corbels: three brackets, splayed, carrying the slab off the wall.
    for (const f of [-0.42, 0, 0.42]) {
      box(stone, 0.13, 0.24, depth * 0.9, f * w, -0.09, depth * 0.42, 0.02, 0, 0, 0);
      box(stone, 0.11, 0.3, 0.24, f * w, -0.22, 0.11, 0.02, 0, -0.7);
    }
    // Railing: uprights, a top rail and a mid rail.
    const posts = Math.max(4, Math.round(w / 0.22));
    for (let i = 0; i < posts; i++) {
      const x = -w * 0.5 + (w * (i + 0.5)) / posts;
      box(steel, 0.022, 0.86, 0.022, x, 0.57, depth - 0.07, 0.005);
    }
    for (const sz of [0.07, depth - 0.07]) {
      cyl(steel, 0.02, w, 0, 1.0, sz, 6, 0, Math.PI / 2);
    }
    cyl(steel, 0.016, w, 0, 0.6, depth - 0.07, 5, 0, Math.PI / 2);
    for (const sx of [-1, 1]) {
      box(steel, 0.022, 0.86, depth - 0.14, sx * w * 0.5, 0.57, depth * 0.5, 0.005);
    }
    return [{ key: 'concrete', geo: merge(stone) }, { key: 'rust', geo: merge(steel) }];
  }

  /**
   * A corner quoin pier: a vertical pilaster standing proud at a building
   * corner, running the full height.
   *
   * Corners are where two flat planes meet at a hard line, and a hard line with
   * nothing on it is the most box-like thing a building can do. A 55 cm pier
   * projecting 9 cm turns the corner into two shadowed reveals and gives the
   * elevation a vertical rhythm to bracket the horizontal bands against.
   */
  static cornerQuoin(height: number, size = 0.56, proud = 0.1): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, size + proud * 2, height, size + proud * 2, 0, height * 0.5, 0, 0.035);
    // Capital and base, so the pier terminates rather than just stopping.
    box(parts, size + proud * 2 + 0.14, 0.14, size + proud * 2 + 0.14, 0, height - 0.07, 0, 0.022);
    box(parts, size + proud * 2 + 0.18, 0.2, size + proud * 2 + 0.18, 0, 0.1, 0, 0.025);
    return merge(parts);
  }

  /** A cast-iron road gully grate, flush with the paving. */
  static drainGrate(): THREE.BufferGeometry {
    const parts: THREE.BufferGeometry[] = [];
    box(parts, 0.52, 0.05, 0.42, 0, 0.025, 0, 0.012);
    for (let i = 0; i < 5; i++) {
      box(parts, 0.4, 0.035, 0.028, 0, 0.055, -0.15 + i * 0.075, 0.006);
    }
    return merge(parts);
  }
}
