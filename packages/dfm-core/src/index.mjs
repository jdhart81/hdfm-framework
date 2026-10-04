// @viridis/dfm-core: pure DFM analysis functions (no storage, no network).
export {checkConnectivity, checkConnectivitySync, validateInput, ENGINE_VERSION, CORE_CLASSES, CROSSING_STATUS, LIGHT_INTENSITIES} from './connectivity.mjs';
export {canonicalJSON, canonicalHash, canonicalHashSync, sha256Hex} from './hash.mjs';
export {SCHEMA_VERSION, LAYERS, toLandscapePackage, fromLandscapePackage} from './package.mjs';
export {deriveSpine, spineNetwork, SPINE_VERSION, SPINE_LINK_KINDS} from './spine.mjs';
export {projectSpine, climateRoutes, buildOutFrontier, OUTLOOK_VERSION} from './outlook.mjs';
