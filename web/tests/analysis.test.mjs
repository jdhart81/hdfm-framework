import test from 'node:test';
import assert from 'node:assert/strict';
import * as t from '@turf/turf';
import { analyze } from '../server/analysis.mjs';
import { empty } from '../public/map-core.mjs';

const params = { waterWidth: 50, roadWidth: 6 };

function square(west, south, east, north) {
  return t.polygon([[[west, south], [east, south], [east, north], [west, north], [west, south]]]);
}

function scenario({ retention = [], units = [] } = {}) {
  const state = empty();
  state.boundary = [square(0, 0, 0.01, 0.01)];
  state.forest = [square(0, 0, 0.01, 0.01)];
  state.waterways = [t.lineString([[-0.01, 0.005], [0.02, 0.005]])];
  state.roads = [
    t.lineString([[0.005, -0.01], [0.005, 0.02]]),
    t.lineString([[0.015, -0.01], [0.015, 0.02]]),
  ];
  state.retention = retention;
  state.units = units;
  return state;
}

function closeTo(actual, expected, tolerance = 0.01) {
  assert.ok(
    Math.abs(actual - expected) <= Math.abs(expected) * tolerance,
    `expected ${actual} to be within ${tolerance * 100}% of ${expected}`,
  );
}

function metrics(state) {
  return analyze(state, params, {}).metrics;
}

test('analysis metrics use the expected units and exclude outside crossings', () => {
  const result = metrics(scenario());
  closeTo(result.bufferHa, 11.12);
  closeTo(result.roadSurfaceHa, 0.667);
  assert.equal(result.crossingCandidates, 1);
});

test('retention area is counted inside the boundary and clipped at its edge', () => {
  const base = metrics(scenario()).retainedForestHa;
  const interior = square(0.001, 0.008, 0.003, 0.01);
  const interiorDelta = metrics(scenario({ retention: [interior] })).retainedForestHa - base;
  closeTo(interiorDelta, t.area(interior) / 10_000);

  const boundary = square(0, 0, 0.01, 0.01);
  const straddling = square(0.009, 0.008, 0.011, 0.01);
  const clipped = t.intersect(t.featureCollection([boundary, straddling]));
  assert.ok(clipped);
  const edgeDelta = metrics(scenario({ retention: [straddling] })).retainedForestHa - base;
  closeTo(edgeDelta, t.area(clipped) / 10_000);
});

test('management overlap equals the area of its part of a retention square', () => {
  const unit = square(0.001, 0.008, 0.002, 0.01);
  const retention = square(0.001, 0.008, 0.003, 0.01);
  const result = metrics(scenario({ retention: [retention], units: [unit] }));
  closeTo(result.managementOverlapHa, t.area(unit) / 10_000);
});
