// Geometry helpers shared by the connectivity check, the spine network and the
// projections. Internal to dfm-core; not part of the public API.
//
// Lengths are meters and areas square meters (I10). Coordinates are WGS84
// longitude/latitude; Turf measures on a sphere of radius 6,371,008.8 m.

import * as turf from './turf.mjs';

/** Seams narrower than 2 * CLOSE_M between adjacent habitat polygons are closed (digitizing gaps). */
export const CLOSE_M = 0.4;

export const fc = features => turf.featureCollection(features.filter(Boolean));
export const isPoly = f => f?.geometry && ['Polygon', 'MultiPolygon'].includes(f.geometry.type);
export const isLine = f => f?.geometry && ['LineString', 'MultiLineString'].includes(f.geometry.type);
export const id = f => String(f?.properties?.dfm_id ?? '');

// Polygon clipping can fail on near-coincident edges (e.g. habitat drawn exactly to a road
// edge). Each operation retries once on a 1e-8 degree grid (about 1 mm), which removes the
// near-coincidence without changing any width or area materially.
const snap = f => turf.truncate(f, {precision: 8, coordinates: 2});
export function robust(op, features) {
  try { return op(fc(features)); } catch (first) {
    try { return op(fc(features.map(snap))); } catch { throw first; }
  }
}
export function union(features) {
  const polys = features.filter(f => isPoly(f));
  if (!polys.length) return null;
  if (polys.length === 1) return turf.feature(structuredClone(polys[0].geometry));
  return robust(turf.union, polys.map(f => turf.feature(f.geometry)));
}
export const intersect = (a, b) => (a && b ? robust(turf.intersect, [a, b]) : null);
export const difference = (a, b) => (a && b ? robust(turf.difference, [a, b]) : a);
export const buffer = (f, m) => (f ? turf.buffer(f, m, {units: 'meters', steps: 16}) ?? null : null);
export const close = f => (f ? buffer(buffer(f, CLOSE_M), -CLOSE_M) ?? f : null);
export const areaM2 = f => (f ? turf.area(f) : 0);
/** Mean width of a polygon, 2 x area / perimeter (meters): small for slivers. */
export const meanWidthM = f => {
  let perimeter = 0;
  for (const ring of f.geometry.coordinates) for (let i = 1; i < ring.length; i++) perimeter += turf.rhumbDistance(ring[i - 1], ring[i], {units: 'meters'});
  return perimeter ? (2 * areaM2(f)) / perimeter : 0;
};
export const parts = f => (!f ? [] : f.geometry.type === 'Polygon' ? [turf.polygon(f.geometry.coordinates)] : f.geometry.coordinates.map(c => turf.polygon(c)));

/** Single LineString parts of a line feature. */
export const lineParts = f => (f.geometry.type === 'LineString' ? [f.geometry.coordinates] : f.geometry.coordinates);

/**
 * Area-weighted centroid of a polygon or multipolygon, [lon, lat]. Uses a local
 * equirectangular frame (exact enough for direction labels at woodlot scale).
 */
export function centroid(f) {
  let ax = 0, ay = 0, a = 0;
  const [w0, s0, e0, n0] = turf.bbox(f);
  const lon0 = (w0 + e0) / 2, lat0 = (s0 + n0) / 2; // local origin keeps the arithmetic well conditioned
  const k = Math.cos(lat0 * Math.PI / 180);
  const xy = c => [(c[0] - lon0) * k, c[1] - lat0];
  for (const poly of f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates)
    for (const [r, ring] of poly.entries()) {
      let cr = 0, cx = 0, cy = 0;
      for (let i = 1; i < ring.length; i++) {
        const [x0, y0] = xy(ring[i - 1]), [x1, y1] = xy(ring[i]);
        const cross = x0 * y1 - x1 * y0;
        cr += cross; cx += (x0 + x1) * cross; cy += (y0 + y1) * cross;
      }
      // Normalize each ring's winding so outer rings add and holes subtract, whatever their order.
      const w = (r === 0 ? 1 : -1) * (Math.sign(cr) || 1);
      a += w * cr; ax += w * cx; ay += w * cy;
    }
  if (Math.abs(a) < 1e-20) return [lon0, lat0];
  return [lon0 + ax / (3 * a) / k, lat0 + ay / (3 * a)];
}

const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
/** Eight-point compass label for a bearing in degrees (0 = north, clockwise). */
export const compass = bearing => COMPASS[Math.round((((bearing % 360) + 360) % 360) / 45) % 8];
