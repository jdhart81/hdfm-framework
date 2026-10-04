// Single-disturbance test for linked cores (DFM Build Spec, "Old-growth spine", SP6-SP7).
// Internal to dfm-core.
//
// Question: can one disturbance (a disk of a given width, centered anywhere) separate two cores
// that the connectivity check links?
//
// Method. The check links two cores when they touch the same connected part of E, the habitat
// eroded by r = minWidthM / 2 (I3). Removing a disk of radius rho from the habitat removes from E
// every point within rho + r of the disk's center, except points whose r-ball lies inside a core:
// a disturbance may overlap a core but does not remove core habitat (cores are the endpoints;
// their own loss is a different question). E is rasterized at spacing g into runs of cells, in a
// local frame (equirectangular around the data's center).
//
// Soundness of a "robust" claim:
//   * Raster connectivity never adds a connection (corridors only). A run comes from one interval
//     of a row line inside E, so its cells are joined along a segment inside E. Cells one row apart
//     are joined only in a column where no edge of E crosses the segment between their centers:
//     a pinch just under the minimum width leaves two pieces of E that can come within a cell of
//     each other, and this keeps them apart. A core's contact cells have centers inside the check's
//     reach of it. The frame is affine in longitude and latitude, so inside and outside are exact.
//   * A cut removes the cells whose centers lie within the tested radius, which exceeds the true
//     one by far more than a segment between neighboring cells can bow (g^2 / 8R), so the
//     segments between remaining cells stay outside the true disk.
//   * Tested radii are inflated for the frame's east-west stretch over the habitat's extent
//     (`scale`), so a true disk always fits inside the tested one.
//   * Disturbance centers are tested on a grid. A coarse grid with spacing S tests disks enlarged
//     by S * sqrt(2) / 2, so every possible center lies within the enlargement of a tested one
//     and the tested cut removes at least as much. Squares whose coarse cut separates a pair are
//     re-tested on a finer grid (spacing S / 4, enlarged accordingly).
//   * Stepping stones (gap > 0): the exact rule joins parts of E at most 2r + gap apart, unless the
//     join would cross a road. The raster grows each remaining cell by
//     growM = (r + gap / 2) * k - g (whole cells within it) and joins grown cells that touch.
//     k = kMin * (1 - 0.003): kMin is the frame's largest east-west shrinkage over the habitat's
//     extent, and 0.003 covers the exact rule's arcs drawn as chords. Touching grown cells come
//     from cells at most 2 growM + g apart in the frame, so at most 2r + gap apart on the ground:
//     a join the exact rule makes too. The exact rule keeps its reach r / 2 clear of roads; cells
//     closer to that than growM + 2g (on the ground, divided by k) keep their own cell but do not
//     grow, so every grown cell lies in the reach of its source and no join crosses a road. Joins
//     near the limit or near roads can be missed; such pairs are never reported robust.
//   * So a pair that no tested cut separates cannot be separated by any disturbance of the given
//     width. Reported failures can be conservative (up to the enlargement wider than nominal);
//     each is then re-tested exactly at nominal width and marked verified or not.

import * as turf from './turf.mjs';
import {buffer, intersect, difference, areaM2, parts, union} from './geo.mjs';
import {linkedPairs, pairKey} from './connectivity.mjs';

const SUBDIVISIONS = 4;
/** Share of a join distance the exact rule can lose to arcs drawn as chords (16 per quarter circle), with margin. */
export const CHORD_SHORTFALL = 0.003;

/** Rows of cell-index runs covering polygons (even-odd rule), on the grid {x0, y0, g, ny}. */
function scan(polysXY, grid) {
  const {x0, y0, g, ny} = grid;
  const rows = Array.from({length: ny}, () => []);
  for (const poly of polysXY) {
    const xs = Array.from({length: ny}, () => null);
    for (const ring of poly)
      for (let e = 1; e < ring.length; e++) {
        const [xa, ya] = ring[e - 1], [xb, yb] = ring[e];
        if (ya === yb) continue;
        const lo = Math.min(ya, yb), hi = Math.max(ya, yb);
        // Row centers y_j = y0 + (j + 0.5) g with lo <= y_j < hi.
        const j0 = Math.max(0, Math.ceil((lo - y0) / g - 0.5)), j1 = Math.min(ny - 1, Math.ceil((hi - y0) / g - 0.5) - 1);
        for (let j = j0; j <= j1; j++) {
          const y = y0 + (j + 0.5) * g;
          (xs[j] ??= []).push(xa + ((y - ya) * (xb - xa)) / (yb - ya));
        }
      }
    for (let j = 0; j < ny; j++) {
      const list = xs[j];
      if (!list) continue;
      list.sort((a, b) => a - b);
      for (let k = 0; k + 1 < list.length; k += 2) {
        const i0 = Math.ceil((list[k] - x0) / g - 0.5), i1 = Math.floor((list[k + 1] - x0) / g - 0.5);
        if (i0 <= i1) rows[j].push([i0, i1]);
      }
    }
  }
  for (const r of rows) r.sort((a, b) => a[0] - b[0]);
  return rows;
}

/** Merge overlapping or touching ranges in a sorted list. */
function merge(ranges) {
  const out = [];
  for (const [a, b] of ranges.slice().sort((p, q) => p[0] - q[0])) {
    const last = out[out.length - 1];
    if (last && a <= last[1] + 1) last[1] = Math.max(last[1], b); else out.push([a, b]);
  }
  return out;
}

/** Ranges of `runs` minus `cut` (both sorted, disjoint), keeping each run's label. */
function subtract(runs, cut) {
  if (!cut.length) return runs;
  const out = [];
  for (const [a, b, label] of runs) {
    let s = a;
    for (const [c, d] of cut) {
      if (d < s || c > b) continue;
      if (c > s) out.push([s, c - 1, label]);
      s = Math.max(s, d + 1);
      if (s > b) break;
    }
    if (s <= b) out.push([s, b, label]);
  }
  return out;
}

/**
 * Build the disturbance tester.
 * @param {object} o
 * @param {object} o.habitat - effective habitat (check rules applied)
 * @param {{id: string, feature: object}[]} o.cores
 * @param {number} o.minWidthM
 * @param {number} o.resolutionM - raster spacing g
 * @param {(c: number[]) => number[]} o.toXY - lon/lat to local meters
 * @param {number} [o.gapM] - stepping-stone gap (params.gapCrossingM); 0 for corridors only
 * @param {object} [o.barrier] - road surfaces after crossings: stepping stones never join across them
 * @param {number} [o.distanceFactor] - k in the header: kMin * (1 - CHORD_SHORTFALL), at most 1
 */
export function disturbanceModel({habitat, cores, minWidthM, resolutionM, toXY, gapM = 0, barrier = null, distanceFactor = 1 - CHORD_SHORTFALL}) {
  const r = minWidthM / 2, g = resolutionM;
  // Stepping stones: cells grow by growM (whole cells only); K rows reach that far (header, soundness).
  const gapMode = gapM > 0, growM = gapMode ? Math.max(0, (r + gapM / 2) * Math.min(1, distanceFactor) - g) : 0, K = gapMode ? Math.floor(growM / g) : 0;
  const halfWidth = Array.from({length: K + 1}, (_, dy) => Math.floor(Math.sqrt(Math.max(0, growM * growM - (dy * g) ** 2)) / g));
  const E = buffer(habitat, -r);
  const pieces = parts(E);
  const ringsXY = f => (f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates).map(p => p.map(ring => ring.map(toXY)));
  const allXY = pieces.flatMap(ringsXY);
  if (!allXY.length) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const poly of allXY) for (const [x, y] of poly[0]) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  // Rows extend K beyond E so grown cells near the edge still meet.
  const pad = K + 1;
  const grid = {x0: minX - g, y0: minY - pad * g, g, ny: Math.ceil((maxY - minY) / g) + 2 * pad + 1};
  const cellX = i => grid.x0 + (i + 0.5) * g, cellY = j => grid.y0 + (j + 0.5) * g;

  // E as labeled runs: one label per exact part, so cells never join across parts.
  const runs = Array.from({length: grid.ny}, () => []);
  pieces.forEach((p, k) => scan(ringsXY(p), grid).forEach((row, j) => { for (const [a, b] of row) runs[j].push([a, b, k]); }));
  for (const row of runs) row.sort((p, q) => p[0] - q[0]);

  // Cores: eligibility (the 50% rule), protected interiors, and contact cells strictly inside the reach.
  const eligible = [], deep = Array.from({length: grid.ny}, () => []), contact = [];
  for (const c of cores) {
    const remaining = intersect(c.feature, habitat);
    const ok = remaining && areaM2(remaining) >= 0.5 * areaM2(c.feature);
    eligible.push(Boolean(ok));
    if (!ok) { contact.push(null); continue; }
    const inner = buffer(remaining, -r);
    if (inner) scan(ringsXY(inner), grid).forEach((row, j) => deep[j].push(...row));
    // A cell whose center lies in E and within the check's reach (r + 0.5 m) is a true contact.
    contact.push(scan(ringsXY(buffer(remaining, r + 0.5)), grid));
  }
  for (let j = 0; j < grid.ny; j++) deep[j] = merge(deep[j]);

  // Corridors only: columns where an edge of E crosses the band between row centers k and k + 1.
  // Cells in rows k and k + 1 join only through a column outside these (header, soundness).
  const crossed = gapMode ? null : Array.from({length: grid.ny}, () => []);
  if (crossed) {
    for (const poly of allXY) for (const ring of poly) for (let e = 1; e < ring.length; e++) {
      const [xa, ya] = ring[e - 1], [xb, yb] = ring[e];
      const lo = Math.min(ya, yb), hi = Math.max(ya, yb);
      const k0 = Math.max(0, Math.ceil((lo - grid.y0) / g - 1.5)), k1 = Math.min(grid.ny - 2, Math.floor((hi - grid.y0) / g - 0.5));
      for (let k = k0; k <= k1; k++) {
        const ylo = Math.max(lo, cellY(k)), yhi = Math.min(hi, cellY(k + 1));
        if (ylo > yhi) continue;
        let xl = Math.min(xa, xb), xr = Math.max(xa, xb);
        if (ya !== yb) {
          const x1 = xa + ((ylo - ya) * (xb - xa)) / (yb - ya), x2 = xa + ((yhi - ya) * (xb - xa)) / (yb - ya);
          xl = Math.min(x1, x2); xr = Math.max(x1, x2);
        }
        const i0 = Math.ceil((xl - grid.x0) / g - 0.5), i1 = Math.floor((xr - grid.x0) / g - 0.5);
        if (i0 <= i1) crossed[k].push([i0, i1]);
      }
    }
    for (let k = 0; k < grid.ny; k++) if (crossed[k].length) crossed[k] = merge(crossed[k]);
  }
  /** True when some column lo..hi between rows k and k + 1 is not crossed by an edge of E. */
  const open = (k, lo, hi) => {
    const band = crossed[k];
    if (!band.length) return true;
    let L = 0, H = band.length - 1, at = -1; // last crossed range starting at or before lo
    while (L <= H) { const m = (L + H) >> 1; if (band[m][0] <= lo) { at = m; L = m + 1; } else H = m - 1; }
    return !(at >= 0 && band[at][1] >= hi);
  };

  /** Linked core pairs (keys) for the given rows of runs, by union-find over runs. */
  const link = rowsRuns => {
    const offset = new Int32Array(grid.ny + 1);
    for (let j = 0; j < grid.ny; j++) offset[j + 1] = offset[j] + rowsRuns[j].length;
    const parent = new Int32Array(offset[grid.ny]).map((_, i) => i);
    const find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    for (let j = 0; j + 1 < grid.ny; j++) {
      const A = rowsRuns[j], B = rowsRuns[j + 1];
      let a = 0, b = 0;
      while (a < A.length && b < B.length) {
        if (A[a][1] < B[b][0]) { a++; continue; }
        if (B[b][1] < A[a][0]) { b++; continue; }
        if (A[a][2] === B[b][2] && open(j, Math.max(A[a][0], B[b][0]), Math.min(A[a][1], B[b][1]))) {
          const ra = find(offset[j] + a), rb = find(offset[j + 1] + b);
          if (ra !== rb) parent[ra] = rb;
        }
        if (A[a][1] < B[b][1]) a++; else b++;
      }
    }
    const roots = cores.map((_, ci) => {
      const set = new Set();
      if (!eligible[ci]) return set;
      contact[ci].forEach((cr, j) => {
        const R = rowsRuns[j];
        let a = 0;
        for (const [c, d] of cr) {
          while (a < R.length && R[a][1] < c) a++;
          for (let t = a; t < R.length && R[t][0] <= d; t++) set.add(find(offset[j] + t));
        }
      });
      return set;
    });
    return pairsOf(roots);
  };
  /** Linked core pairs from the sets of roots each core touches. */
  const pairsOf = roots => {
    const out = new Set();
    for (let i = 0; i < cores.length; i++) for (let j = i + 1; j < cores.length; j++)
      if ([...roots[i]].some(x => roots[j].has(x))) out.add(pairKey(cores[i].id, cores[j].id));
    return out;
  };

  // Stepping stones near roads: cells within reach of the kept-clear road (the exact rule keeps
  // reach r / 2 clear of it) keep their own cell but do not grow.
  let near = null;
  if (gapMode && barrier) {
    const reach = buffer(barrier, r / 2 + (growM + 2 * g) / Math.min(1, distanceFactor) + 1);
    if (reach) near = scan(ringsXY(reach), grid).map(row => (row.length ? merge(row) : row));
  }
  /**
   * Every interval grown row j receives: the far cells of rows j - K .. j + K widened by the disk's
   * half-width at that offset, and the near cells of row j itself.
   */
  const contribute = (rowsRuns, j, add) => {
    for (let dy = -K; dy <= K; dy++) {
      const row = rowsRuns[j + dy];
      if (!row || !row.length) continue;
      const h = halfWidth[dy < 0 ? -dy : dy], nearRow = near?.[j + dy], self = dy === 0;
      for (let t = 0; t < row.length; t++) {
        const p = row[t][0], q = row[t][1];
        let s = p;
        if (nearRow && nearRow.length) for (const [c, d] of nearRow) {
          if (d < s) continue;
          if (c > q) break;
          if (c > s) add(s - h, c - 1 + h);
          if (self) add(Math.max(s, c), Math.min(q, d));
          s = d + 1;
          if (s > q) break;
        }
        if (s <= q) add(s - h, q + h);
      }
    }
  };
  /** Row j of the remaining cells, grown: union of its intervals by a sweep over sorted starts and ends. */
  const grow = (rowsRuns, j) => {
    const starts = [], ends = [];
    contribute(rowsRuns, j, (s, e) => { starts.push(s); ends.push(e); });
    const n = starts.length;
    if (!n) return [];
    const S = Float64Array.from(starts).sort(), E2 = Float64Array.from(ends).sort();
    const out = [];
    let i = 0, e = 0, depth = 0, from = 0;
    while (i < n || e < n) {
      if (i < n && S[i] <= E2[e] + 1) { if (depth++ === 0) from = S[i]; i++; }
      else { if (--depth === 0) out.push([from, E2[e]]); e++; }
    }
    return out;
  };
  /**
   * Row j grown after a cut, recomputed only in columns wlo..whi (the cut's reach plus the growth);
   * outside them nothing within growM of a removed cell exists, so the baseline row stands.
   * Coverage in the window by a difference array; touching pieces at the window edges merge.
   */
  let diff = new Int32Array(1024);
  const growWindow = (rowsRuns, j, wlo, whi) => {
    const width = whi - wlo + 1;
    if (width + 1 > diff.length) diff = new Int32Array(2 * (width + 1));
    diff.fill(0, 0, width + 1);
    contribute(rowsRuns, j, (s0, e0) => {
      const s = Math.max(s0, wlo), e = Math.min(e0, whi);
      if (s <= e) { diff[s - wlo]++; diff[e - wlo + 1]--; }
    });
    const pieces = [];
    for (const [s, e] of baseGrown[j]) if (s < wlo) pieces.push([s, Math.min(e, wlo - 1)]);
    let depth = 0, from = -1;
    for (let x = 0; x <= width; x++) {
      depth += diff[x];
      if (depth > 0 && from < 0) from = x;
      else if (depth === 0 && from >= 0) { pieces.push([wlo + from, wlo + x - 1]); from = -1; }
    }
    for (const [s, e] of baseGrown[j]) if (e > whi) pieces.push([Math.max(s, whi + 1), e]);
    const out = [];
    for (const piece of pieces) {
      const last = out[out.length - 1];
      if (last && piece[0] <= last[1] + 1) last[1] = Math.max(last[1], piece[1]); else out.push(piece);
    }
    return out;
  };
  /** Stepping stones: linked core pairs over the grown rows; contacts are remaining cells of E. */
  const linkGrown = (rowsRuns, grown) => {
    const offset = new Int32Array(grid.ny + 1);
    for (let j = 0; j < grid.ny; j++) offset[j + 1] = offset[j] + grown[j].length;
    const parent = new Int32Array(offset[grid.ny]).map((_, i) => i);
    const find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    for (let j = 0; j + 1 < grid.ny; j++) {
      const A = grown[j], B = grown[j + 1];
      let p = 0, q = 0;
      while (p < A.length && q < B.length) {
        if (A[p][1] < B[q][0]) { p++; continue; }
        if (B[q][1] < A[p][0]) { q++; continue; }
        const ra = find(offset[j] + p), rb = find(offset[j + 1] + q);
        if (ra !== rb) parent[ra] = rb;
        if (A[p][1] < B[q][1]) p++; else q++;
      }
    }
    const roots = cores.map((_, ci) => {
      const set = new Set();
      if (!eligible[ci]) return set;
      contact[ci].forEach((cr, j) => {
        const R = rowsRuns[j], G = grown[j];
        let t = 0, u = 0;
        for (const [c, d] of cr) {
          while (t < R.length && R[t][1] < c) t++;
          for (let s = t; s < R.length && R[s][0] <= d; s++) {
            const x = Math.max(c, R[s][0]); // a remaining cell of E inside the contact range
            while (u < G.length && G[u][1] < x) u++;
            if (u < G.length && G[u][0] <= x) set.add(find(offset[j] + u));
          }
        }
      });
      return set;
    });
    return pairsOf(roots);
  };
  const baseGrown = gapMode ? Array.from({length: grid.ny}, (_, j) => grow(runs, j)) : null;
  const baseline = gapMode ? linkGrown(runs, baseGrown) : link(runs);

  /** Linked pairs after removing a disk of radius R (in eroded space) centered at (cx, cy). */
  const cut = (cx, cy, R) => {
    const j0 = Math.max(0, Math.ceil((cy - R - grid.y0) / g - 0.5)), j1 = Math.min(grid.ny - 1, Math.floor((cy + R - grid.y0) / g - 0.5));
    if (j0 > j1) return baseline;
    let removed = false;
    const rowsRuns = runs.slice();
    for (let j = j0; j <= j1; j++) {
      const dy = cellY(j) - cy, h2 = R * R - dy * dy;
      if (h2 < 0 || !runs[j].length) continue;
      const h = Math.sqrt(h2);
      const a = Math.ceil((cx - h - grid.x0) / g - 0.5), b = Math.floor((cx + h - grid.x0) / g - 0.5);
      if (a > b) continue;
      const removal = subtract([[a, b, 0]], deep[j]).map(([p, q]) => [p, q]);
      const next = subtract(runs[j], removal);
      if (next.length !== runs[j].length || next.some((x, t) => x[0] !== runs[j][t][0] || x[1] !== runs[j][t][1])) { rowsRuns[j] = next; removed = true; }
    }
    if (!removed) return baseline;
    if (!gapMode) return link(rowsRuns);
    const wlo = Math.floor((cx - R - grid.x0) / g - 0.5) - K - 2, whi = Math.ceil((cx + R - grid.x0) / g - 0.5) + K + 2;
    const grown = baseGrown.slice();
    for (let j = Math.max(0, j0 - K); j <= Math.min(grid.ny - 1, j1 + K); j++) grown[j] = growWindow(rowsRuns, j, wlo, whi);
    return linkGrown(rowsRuns, grown);
  };

  // Occupancy of E on a coarse binning, to skip centers whose cut cannot touch E.
  const occupied = (cx, cy, R) => {
    const j0 = Math.max(0, Math.ceil((cy - R - grid.y0) / g - 0.5)), j1 = Math.min(grid.ny - 1, Math.floor((cy + R - grid.y0) / g - 0.5));
    for (let j = j0; j <= j1; j++) {
      const dy = cellY(j) - cy, h2 = R * R - dy * dy;
      if (h2 < 0) continue;
      const h = Math.sqrt(h2), a = Math.ceil((cx - h - grid.x0) / g - 0.5), b = Math.floor((cx + h - grid.x0) / g - 0.5);
      for (const [p, q] of runs[j]) if (p <= b && q >= a) return true;
    }
    return false;
  };

  return {grid, baseline, cut, occupied, bounds: {minX, minY, maxX, maxY}, cellX, cellY};
}

/**
 * Search disturbance centers (coarse grid, then refine where a cut separates a pair).
 * `scale` (>= 1) inflates every tested radius for the local frame's east-west distortion, and
 * `budget` caps the number of cuts: past it the search stops and returns {exhausted: true}, so a
 * caller never reports robustness from a partial search.
 * @returns {{tested, failures: {x, y, fi, fj, lost}[], fineSpacing, testedRadius, exhausted}}
 */
export function searchDisturbances(model, {rhoM, minWidthM, pairs, scale = 1, budget = Infinity}) {
  const r = minWidthM / 2, R = rhoM + r;
  const S = R / 2, s = S / SUBDIVISIONS, slack = 0.5; // 0.5 m for polygon approximation
  const Rc = (R + (S * Math.SQRT2) / 2) * scale + slack, Rf = (R + (s * Math.SQRT2) / 2) * scale + slack;
  const {minX, minY, maxX, maxY} = model.bounds;
  const ax0 = Math.floor((minX - Rc) / S), ax1 = Math.ceil((maxX + Rc) / S), by0 = Math.floor((minY - Rc) / S), by1 = Math.ceil((maxY + Rc) / S);
  const targets = [...pairs].filter(k => model.baseline.has(k));
  const failures = [];
  let tested = 0;
  const result = exhausted => ({tested, failures, fineSpacing: s, testedRadius: {coarse: Rc - r, fine: Rf - r}, exhausted});
  for (let b = by0; b <= by1; b++) for (let a = ax0; a <= ax1; a++) {
    const cx = (a + 0.5) * S, cy = (b + 0.5) * S;
    if (!model.occupied(cx, cy, Rc)) continue;
    if (tested >= budget) return result(true);
    tested++;
    const after = model.cut(cx, cy, Rc);
    if (targets.every(k => after.has(k))) continue;
    for (let j = 0; j < SUBDIVISIONS; j++) for (let i = 0; i < SUBDIVISIONS; i++) {
      const fx = a * S + (i + 0.5) * s, fy = b * S + (j + 0.5) * s;
      if (!model.occupied(fx, fy, Rf)) continue;
      if (tested >= budget) return result(true);
      tested++;
      const fineAfter = model.cut(fx, fy, Rf);
      const lost = targets.filter(k => !fineAfter.has(k));
      if (lost.length) failures.push({x: fx, y: fy, fi: a * SUBDIVISIONS + i, fj: b * SUBDIVISIONS + j, lost});
    }
  }
  return result(false);
}

/**
 * Group failing fine centers into zones: grid neighbors (8-connected) that separate exactly the
 * same pairs, so each zone names one weakness (a branch, a junction) rather than merging them.
 */
export function zones(failures) {
  const sig = f => f.lost.slice().sort().join(';');
  const index = new Map(failures.map((f, i) => [`${f.fi},${f.fj}`, i]));
  const seen = new Set(), out = [];
  for (let i = 0; i < failures.length; i++) {
    if (seen.has(i)) continue;
    const members = [], queue = [i]; seen.add(i);
    while (queue.length) {
      const k = queue.pop(); members.push(failures[k]);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const n = index.get(`${failures[k].fi + di},${failures[k].fj + dj}`);
        if (n != null && !seen.has(n) && sig(failures[n]) === sig(failures[k])) { seen.add(n); queue.push(n); }
      }
    }
    out.push(members);
  }
  return out;
}

/**
 * A disk of `rhoM` meters around a lon/lat center as a 64-sided polygon. Inscribed (vertices on
 * the circle, slightly smaller than the disk) by default; `circumscribed` makes it slightly larger.
 */
export function diskPolygon(center, rhoM, {circumscribed = false} = {}) {
  const n = 64, R = 6371008.8, lat = center[1] * Math.PI / 180, rad = circumscribed ? rhoM / Math.cos(Math.PI / n) : rhoM;
  const ring = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    ring.push([center[0] + ((rad * Math.cos(t)) / (R * Math.cos(lat))) * (180 / Math.PI), center[1] + ((rad * Math.sin(t)) / R) * (180 / Math.PI)]);
  }
  ring.push(ring[0]);
  return turf.polygon([ring]);
}

/**
 * Exact re-test of one disturbance at nominal width: the check's own link rule (linkedPairs, with
 * stepping stones when gapM > 0, never across `barrier`) on the habitat less the disk. Cores are not
 * damaged: the disk is applied outside them.
 */
export function exactCut({habitat, cores, minWidthM, center, rhoM, gapM = 0, barrier = null}) {
  const cutArea = difference(diskPolygon(center, rhoM), union(cores.map(c => c.feature)));
  const h = cutArea ? difference(habitat, cutArea) : habitat;
  return new Set(h ? linkedPairs(h, cores, minWidthM, {gapM, barrier}).pairs : []);
}
