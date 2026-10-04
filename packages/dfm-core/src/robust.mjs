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
// their own loss is a different question). E is rasterized at spacing g into runs of cells.
//
// Soundness of a "robust" claim:
//   * Raster connectivity never adds a connection: cells join only inside one exact part of E,
//     and a core's contact cells have centers inside the check's reach of it. Removal and
//     adjacency cannot jump a gap: gaps in E are at least 2r wide, the raster spacing is at most
//     r / 2 (so every gap holds a row and a column of cell centers), and removed disks are convex
//     with radius above r.
//   * Tested radii are inflated for the local frame's east-west scale error over the extent, so
//     a true disk always fits inside the tested one. The frame can also shrink east-west gaps,
//     by under 1% within the engine's 0.5° extent limit, far inside the raster's 4x margin.
//   * Disturbance centers are tested on a grid. A coarse grid with spacing S tests disks enlarged
//     by S * sqrt(2) / 2, so every possible center lies within the enlargement of a tested one
//     and the tested cut removes at least as much. Squares whose coarse cut separates a pair are
//     re-tested on a finer grid (spacing S / 4, enlarged accordingly).
//   * So a pair that no tested cut separates cannot be separated by any disturbance of the given
//     width. Reported failures can be conservative (up to the enlargement wider than nominal);
//     each is then re-tested exactly at nominal width and marked verified or not.

import * as turf from './turf.mjs';
import {buffer, intersect, difference, areaM2, parts, union} from './geo.mjs';
import {linkedPairs} from './connectivity.mjs';

const SUBDIVISIONS = 4;

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
 */
export function disturbanceModel({habitat, cores, minWidthM, resolutionM, toXY}) {
  const r = minWidthM / 2, g = resolutionM;
  const E = buffer(habitat, -r);
  const pieces = parts(E);
  const ringsXY = f => (f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates).map(p => p.map(ring => ring.map(toXY)));
  const allXY = pieces.flatMap(ringsXY);
  if (!allXY.length) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const poly of allXY) for (const [x, y] of poly[0]) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  const grid = {x0: minX - g, y0: minY - g, g, ny: Math.ceil((maxY - minY) / g) + 3};
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
        if (A[a][2] === B[b][2]) { const ra = find(offset[j] + a), rb = find(offset[j + 1] + b); if (ra !== rb) parent[ra] = rb; }
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
    const out = new Set();
    for (let a = 0; a < cores.length; a++) for (let b = a + 1; b < cores.length; b++)
      if ([...roots[a]].some(x => roots[b].has(x))) out.add(cores[a].id < cores[b].id ? `${cores[a].id}|${cores[b].id}` : `${cores[b].id}|${cores[a].id}`);
    return out;
  };
  const baseline = link(runs);

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
    return removed ? link(rowsRuns) : baseline;
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
 * Exact re-test of one disturbance at nominal width: the check's own link rule (linkedPairs) on
 * the habitat less the disk. Cores are not damaged: the disk is applied outside them.
 */
export function exactCut({habitat, cores, minWidthM, center, rhoM}) {
  const cutArea = difference(diskPolygon(center, rhoM), union(cores.map(c => c.feature)));
  const h = cutArea ? difference(habitat, cutArea) : habitat;
  return new Set(h ? linkedPairs(h, cores, minWidthM).pairs : []);
}
