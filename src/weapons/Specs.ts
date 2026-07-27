/**
 * NOISEFIELD — Weapon data.
 *
 * Every number here is a design decision, so the rationale lives next to it.
 *
 * ── RECOIL PATTERNS ─────────────────────────────────────────────────────────
 * Patterns are HAND-AUTHORED SEQUENCES, not randomness. This is the single most
 * important balance decision in the file: a deterministic pattern is
 * *learnable*, so a player who puts hours into a weapon is rewarded with
 * measurably tighter groups at range. Randomised recoil caps skill at "hold
 * roughly down" and makes every gunfight feel like a coin flip.
 *
 * The shape of each pattern follows the modern-shooter convention:
 *   • a strong, almost purely VERTICAL climb for the first handful of rounds
 *     (this is the burst you must pull down through),
 *   • then a horizontal drift in one direction,
 *   • then a reversal, so holding a single counter-drift eventually fails.
 *
 * SIGN CONVENTION. Vertical is positive for muzzle rise. Horizontal is
 * positive for a drift to the LEFT: the kick is added to the camera's yaw, and
 * increasing yaw rotates a −Z forward axis toward −X, which is the shooter's
 * left. Getting this backwards is easy and silent — the pattern still feels
 * like a pattern, it is just mirrored from the one that was designed.
 *
 * Values are RADIANS OF VIEW KICK PER SHOT. 1° = 0.01745 rad. A first-shot
 * vertical of 0.0105 rad is 0.60°, which at 30 m is ~31 cm of muzzle rise — in
 * the right ballpark for an intermediate-cartridge rifle.
 *
 * `recoverFraction` is how much of the accumulated kick the view walks back on
 * its own. The remainder is PERMANENT climb the player must physically
 * compensate for. A high recoverFraction (pistol 0.94) means the sights settle
 * back on target for you; a low one (SMG 0.62) means sustained fire genuinely
 * walks off target.
 *
 * ── SPREAD ──────────────────────────────────────────────────────────────────
 * `spreadHip` / `spreadAds` are CONE HALF-ANGLES in radians.
 * `spreadPerShot` is bloom added per trigger pull, `spreadDecay` is bloom bled
 * off per second (linear, matching the mainstream model). The interesting quantity is
 * the ratio: at the weapon's own fire rate, does bloom win or does decay?
 *   AR : 11.67 sh/s × 0.0022 = 0.0257 rad/s in, 0.0130 out → net +0.0127
 *   SMG: 15.83 sh/s × 0.0034 = 0.0538 rad/s in, 0.0200 out → net +0.0338
 * So the SMG's cone opens ~2.7× faster than the rifle's under sustained fire,
 * which is what makes it a room-clearing weapon rather than a ranged one.
 *
 * ── VIEWMODEL POSITIONS ────────────────────────────────────────────────────────
 * `hipPosition` is the weapon's GRIP, in view-space metres, at the hip.
 *
 * HOW THESE WERE DERIVED. The viewmodel camera runs a 60° vertical FOV, so the
 * visible frame at depth |z| is 2·|z|·tan30° tall and 16/9 of that wide. At the
 * old |z| = 0.30 the frame was only 35 cm tall, which is why a 30 cm rifle ate
 * roughly 40% of it — the weapon was not too big, it was too CLOSE. Pushing the
 * grip out to 0.43 m grows the frame to 50 cm × 88 cm and drops the rifle to
 * about a quarter of it, which is where the genre sits an AR at the hip.
 *
 * Depth also controls how much perspective distortion the weapon suffers. At
 * 0.30 m a 0.30 m-long receiver spans a 45° arc and its near end is rendered
 * nearly 2× the size of its far end; at 0.43 m that falls to 33° and ~1.5×,
 * which is the difference between "fish-eyed" and "photographed on a long lens"
 * — the look real viewmodels are shot with.
 *
 * The X/Y offsets put the grip in the lower right without letting the receiver
 * touch the frame edge: at 0.43 m the half-frame is 0.44 m wide and 0.25 m
 * tall, so x = 0.122 is 28% of the way to the right edge and y = −0.150 is 60%
 * of the way to the bottom. The muzzle, 0.5 m further out where the frame is
 * more than twice as wide, therefore converges back toward the centre.
 *
 * Longer weapons sit further out (the sniper at 0.47 m) and shorter ones closer
 * (the pistol at 0.36 m), because what has to stay constant is the ANGULAR size
 * of the weapon, not its distance. `adsPosition` is only a COARSE SANITY
 * REFERENCE: the real aim pose is solved at runtime from the weapon model's
 * optic node, because that is the only way to guarantee the reticle sits on the
 * exact centre of the screen at any field of view. The viewmodel compares the
 * solved pose against this value and falls back to it, with a warning, only if
 * they differ by more than 25 cm — i.e. only if an optic node has been authored
 * somewhere clearly wrong. Ordinary disagreement of several centimetres is
 * expected and means nothing; do not hand-tune these to match the solve.
 */

import type { RecoilPattern, WeaponId, WeaponSpec } from '../core/Types';

/** Degrees → radians, so authored angles stay readable in source. */
const D = Math.PI / 180;

// ─────────────────────────────────────────────────────────────────────────────
// Recoil patterns
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AR — "climb then left then right".
 *
 * Rounds 1-6 go almost straight up hard (0.60° → 0.80°) so the opening burst
 * demands a real downward pull. From round 7 the vertical relaxes to a steady
 * ~0.46°/shot while the horizontal swings LEFT for six rounds, then RIGHT for
 * six, then a shallower left/right pair. The result is a shallow sideways "S"
 * laid over a vertical line — the classic learnable rifle pattern.
 */
const AR_RECOIL: RecoilPattern = {
  vertical: [
    0.0105, 0.0122, 0.0134, 0.0140, 0.0138, 0.0130, // opening climb
    0.0118, 0.0108, 0.0100, 0.0094, 0.0090, 0.0087, // settling
    0.0085, 0.0084, 0.0083, 0.0082, 0.0081, 0.0081,
    0.0080, 0.0080, 0.0079, 0.0079, 0.0080, 0.0081,
    0.0082, 0.0083, 0.0084, 0.0085, 0.0086, 0.0088, // slight late re-climb as the shooter fatigues
  ],
  horizontal: [
    0.0000, -0.0004, 0.0006, -0.0008, 0.0005, -0.0010, // essentially straight up
    0.0032, 0.0048, 0.0058, 0.0062, 0.0058, 0.0048, // hard left drift
    -0.0020, -0.0046, -0.0060, -0.0066, -0.0062, -0.0052, // swing right
    0.0018, 0.0040, 0.0050, 0.0046, 0.0030, 0.0010, // back left
    -0.0022, -0.0040, -0.0048, -0.0044, -0.0030, -0.0012, // final right drift
  ],
  recoverTime: 0.30,
  recoverFraction: 0.72,
};

/**
 * SMG — twitchy and wide. Lower per-shot vertical than the rifle (it is a
 * pistol cartridge) but about a third more horizontal on average (mean |h| of
 * 0.0046 vs the rifle's 0.0035), reversing every six rounds at 1.36× the
 * rifle's fire rate — a direction change every 0.38 s against the rifle's
 * 0.51 s. The pattern reads as a tight zig-zag rather than the rifle's lazy S.
 * Learnable, but only just, and only at short range.
 */
const SMG_RECOIL: RecoilPattern = {
  vertical: [
    0.0072, 0.0086, 0.0094, 0.0098, 0.0096, 0.0092,
    0.0086, 0.0080, 0.0076, 0.0073, 0.0071, 0.0070,
    0.0069, 0.0069, 0.0070, 0.0071, 0.0072, 0.0073,
    0.0074, 0.0075, 0.0076, 0.0077, 0.0078, 0.0079,
    0.0080, 0.0081, 0.0082, 0.0083, 0.0084, 0.0085,
    0.0086, 0.0087,
  ],
  horizontal: [
    -0.0006, 0.0014, 0.0034, 0.0052, // left
    0.0060, 0.0044, 0.0012, -0.0028, // unwind
    -0.0058, -0.0074, -0.0070, -0.0048, // right
    -0.0014, 0.0026, 0.0058, 0.0074, // left again, wider
    0.0070, 0.0046, 0.0008, -0.0034,
    -0.0068, -0.0082, -0.0078, -0.0054,
    -0.0016, 0.0030, 0.0064, 0.0080,
    0.0074, 0.0048, 0.0010, -0.0036,
  ],
  recoverTime: 0.26,
  recoverFraction: 0.62,
};

/**
 * Bolt-action — one enormous kick (4.7° up) and a long, slow, almost-complete
 * settle. `recoverFraction` 0.90 because a bipod-less but heavy rifle does
 * return close to point of aim, and because re-acquiring after every shot is
 * already the weapon's cost.
 */
const SNIPER_RECOIL: RecoilPattern = {
  vertical: [0.0820, 0.0790, 0.0805, 0.0775, 0.0810],
  horizontal: [-0.0100, 0.0084, -0.0092, 0.0078, -0.0088],
  recoverTime: 0.55,
  recoverFraction: 0.90,
};

/**
 * 12-gauge — brutal per-shot rise with a hard alternating shoulder-slap left
 * and right, so follow-up shots require a genuine re-plant rather than a
 * mag-dump.
 */
const SHOTGUN_RECOIL: RecoilPattern = {
  vertical: [0.0580, 0.0555, 0.0525, 0.0500, 0.0480, 0.0465],
  horizontal: [-0.0125, 0.0140, -0.0118, 0.0132, -0.0110, 0.0126],
  recoverTime: 0.42,
  recoverFraction: 0.82,
};

/**
 * Sidearm — snappy, small, and forgiving. High recoverFraction (0.94) and a
 * short recoverTime make it the most controllable weapon in the kit, which is
 * the whole point of a panic-swap sidearm.
 */
const PISTOL_RECOIL: RecoilPattern = {
  vertical: [
    0.0225, 0.0240, 0.0250, 0.0245, 0.0235, 0.0228,
    0.0222, 0.0218, 0.0216, 0.0215, 0.0215, 0.0216,
    0.0218, 0.0220, 0.0222,
  ],
  horizontal: [
    0.0018, -0.0026, 0.0030, -0.0034, 0.0028, -0.0022,
    0.0026, -0.0030, 0.0024, -0.0020, 0.0028, -0.0032,
    0.0022, -0.0026, 0.0020,
  ],
  recoverTime: 0.22,
  recoverFraction: 0.94,
};

// ─────────────────────────────────────────────────────────────────────────────
// Weapon table
// ─────────────────────────────────────────────────────────────────────────────

export const WEAPON_SPECS: Record<WeaponId, WeaponSpec> = {
  // ── VECTOR — 5.56 assault rifle. The reference weapon everything else is
  //    balanced against. 33 dmg = 4 shots to a 100 hp target inside 28 m, 5
  //    beyond 48 m; a headshot (×1.4 → 46.2) drops it to 3 shots close in.
  ar_vector: {
    id: 'ar_vector',
    displayName: 'VECTOR-5',
    caliber: '5.56',
    rpm: 700,
    damage: 33,
    headshotMultiplier: 1.4,
    falloffStart: 28,
    falloffEnd: 48,
    falloffMin: 0.7,
    magSize: 30,
    reserveAmmo: 210,
    reloadTime: 2.1,
    reloadTacticalTime: 1.75,
    adsTime: 0.24,
    spreadHip: 2.6 * D,
    spreadAds: 0.18 * D,
    spreadPerShot: 0.0022,
    spreadDecay: 0.0130,
    recoil: AR_RECOIL,
    muzzleVelocity: 920,
    pellets: 1,
    burst: 0,
    penetration: 0.55,
    hipPosition: [0.166, -0.168, -0.452],
    adsPosition: [0.0, -0.062, -0.215],
    fovScale: 0.80,
  },

  // ── WASP — 9mm SMG. Higher hipfire accuracy than the rifle (2.0° vs 2.6°)
  //    because it is meant to be fired from the hip, but far more bloom, so the
  //    advantage evaporates after ~8 rounds. 26 dmg = 4 shots inside 14 m,
  //    rising to 7 past 26 m: lethal in a room, useless across a courtyard.
  smg_wasp: {
    id: 'smg_wasp',
    displayName: 'WASP-9',
    caliber: '9mm',
    rpm: 950,
    damage: 26,
    headshotMultiplier: 1.3,
    falloffStart: 14,
    falloffEnd: 26,
    falloffMin: 0.55,
    magSize: 32,
    reserveAmmo: 256,
    reloadTime: 1.85,
    reloadTacticalTime: 1.5,
    adsTime: 0.18,
    spreadHip: 2.0 * D, // tighter hip cone than the AR — this is its identity
    spreadAds: 0.26 * D,
    // ...paid for with the worst bloom in the kit. Net per round at each
    // weapon's own cadence is +0.00214 rad for the SMG against +0.00109 for
    // the rifle, so the 0.6° hip advantage is gone by about round 10.
    spreadPerShot: 0.0034,
    spreadDecay: 0.0200,
    recoil: SMG_RECOIL,
    muzzleVelocity: 380,
    pellets: 1,
    burst: 0,
    penetration: 0.30,
    hipPosition: [0.156, -0.158, -0.416],
    adsPosition: [0.0, -0.058, -0.200],
    fovScale: 0.86,
  },

  // ── LONGBOW — .338 bolt action. 130 dmg one-shots an unarmoured 100 hp
  //    target anywhere in the upper body, and the ×1.5 head multiplier exists
  //    for armour rather than for lethality. Balanced by a 55 rpm cap
  //    (1.09 s between shots), a 420 ms ADS, and a hip cone so wide (9°) that
  //    unscoped fire is functionally a threat, not an attack.
  sniper_longbow: {
    id: 'sniper_longbow',
    displayName: 'LONGBOW .338',
    caliber: '.338',
    rpm: 55,
    damage: 130,
    headshotMultiplier: 1.5,
    falloffStart: 120,
    falloffEnd: 240,
    falloffMin: 0.92, // effectively no falloff — this is a rifle for distance
    magSize: 5,
    reserveAmmo: 30,
    reloadTime: 3.1,
    reloadTacticalTime: 2.65,
    adsTime: 0.42,
    spreadHip: 9.0 * D, // enormous; unscoped is a last resort
    spreadAds: 0.02 * D, // effectively pinpoint
    spreadPerShot: 0.0200,
    spreadDecay: 0.2500, // ...but the bloom is gone long before you cycle
    recoil: SNIPER_RECOIL,
    muzzleVelocity: 900,
    pellets: 1,
    burst: 1, // bolt action: one round per trigger pull
    penetration: 0.95, // punches concrete
    hipPosition: [0.176, -0.176, -0.492],
    adsPosition: [0.0, -0.048, -0.150],
    fovScale: 0.42,
  },

  // ── BREACH — 12 gauge. 8 pellets × 17 = 136 potential damage, so a fully
  //    connected pattern is a one-shot kill inside 8 m. At 16 m each pellet is
  //    worth 4.25, so even a perfect pattern only does 34 — the falloff, not
  //    the spread, is what enforces the range limit.
  shotgun_breach: {
    id: 'shotgun_breach',
    displayName: 'BREACH-12',
    caliber: '12ga',
    rpm: 90,
    damage: 17, // PER PELLET
    headshotMultiplier: 1.25,
    falloffStart: 8,
    falloffEnd: 16,
    falloffMin: 0.25,
    magSize: 6,
    reserveAmmo: 42,
    reloadTime: 3.4, // shell-by-shell, abstracted to a single timer
    reloadTacticalTime: 2.9,
    adsTime: 0.26,
    spreadHip: 3.0 * D, // aim cone; the pellet pattern is layered on top
    spreadAds: 1.7 * D,
    spreadPerShot: 0.0040,
    spreadDecay: 0.0500,
    recoil: SHOTGUN_RECOIL,
    muzzleVelocity: 400,
    pellets: 8,
    burst: 1,
    penetration: 0.12, // buckshot does not defeat cover
    hipPosition: [0.170, -0.170, -0.458],
    adsPosition: [0.0, -0.060, -0.205],
    fovScale: 0.85,
  },

  // ── SIDEARM — 9mm semi. 30 dmg is deliberately close to the SMG's 26 so the
  //    swap is a real option rather than a formality; the 420 rpm cap and 15
  //    round magazine are what keep it a sidearm.
  pistol_sidearm: {
    id: 'pistol_sidearm',
    displayName: 'SIDEARM-9',
    caliber: '9mm',
    rpm: 420, // fire-rate CAP for semi-auto; the trigger finger is the limit
    damage: 30,
    headshotMultiplier: 1.5,
    falloffStart: 16,
    falloffEnd: 30,
    falloffMin: 0.6,
    magSize: 15,
    reserveAmmo: 90,
    reloadTime: 1.55,
    reloadTacticalTime: 1.2,
    adsTime: 0.19,
    spreadHip: 2.2 * D,
    spreadAds: 0.30 * D,
    spreadPerShot: 0.0026,
    spreadDecay: 0.0320,
    recoil: PISTOL_RECOIL,
    muzzleVelocity: 360,
    pellets: 1,
    burst: 1,
    penetration: 0.28,
    hipPosition: [0.140, -0.150, -0.376],
    adsPosition: [0.0, -0.052, -0.190],
    fovScale: 0.88,
  },
};

/** Stable iteration order. Slot 0 is the primary pool, the sidearm is last. */
export const WEAPON_ORDER: readonly WeaponId[] = [
  'ar_vector',
  'smg_wasp',
  'sniper_longbow',
  'shotgun_breach',
  'pistol_sidearm',
];

/** The rotation of primaries reachable from the `weapon1` key. */
export const PRIMARY_POOL: readonly WeaponId[] = [
  'ar_vector',
  'smg_wasp',
  'sniper_longbow',
  'shotgun_breach',
];

/** The weapon bound to the `weapon2` key. */
export const SIDEARM: WeaponId = 'pistol_sidearm';

/** Seconds between rounds for a spec. Cached-free helper, safe in hot paths. */
export function shotInterval(spec: WeaponSpec): number {
  return 60 / Math.max(1, spec.rpm);
}

/**
 * Damage multiplier from range. Linear between falloffStart and falloffEnd,
 * flat outside. Linear (rather than a curve) is deliberate: players reason
 * about "my damage range" as a hard number, and a linear ramp keeps the
 * shots-to-kill table legible.
 */
export function falloffAt(spec: WeaponSpec, distance: number): number {
  if (distance <= spec.falloffStart) return 1;
  if (distance >= spec.falloffEnd) return spec.falloffMin;
  const span = Math.max(1e-4, spec.falloffEnd - spec.falloffStart);
  const t = (distance - spec.falloffStart) / span;
  return 1 + (spec.falloffMin - 1) * t;
}
