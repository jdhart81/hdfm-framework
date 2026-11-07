// Runs one dfm-core analysis off the main thread, so the stdio server stays responsive
// and a call over the time limit can be stopped (invariant H6).
import {parentPort, workerData} from 'node:worker_threads';
import * as core from '@viridis/dfm-core';

const RUNNERS = {
  check: (input) => core.checkConnectivitySync(input),
  deriveSpine: (input) => core.deriveSpine(input),
  spineNetwork: (input, opts) => core.spineNetwork(input, opts),
  projectSpine: (input, opts) => core.projectSpine(input, opts),
  climateRoutes: (input) => core.climateRoutes(input),
  buildOutFrontier: (input) => core.buildOutFrontier(input),
};

const {fn, input, opts} = workerData;
try {
  parentPort.postMessage({ok: true, value: RUNNERS[fn](input, opts ?? {})});
} catch (e) {
  parentPort.postMessage({ok: false, error: `${e.name}: ${e.message}`});
}
