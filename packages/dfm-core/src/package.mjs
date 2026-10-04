// DFM Landscape Package v1: the file both DFM and VergeCommon read and write.
// One JSON object, WGS84 GeoJSON features grouped by layer, plus parameters and
// a source record per layer. Schema: packages/dfm-schema/landscape-package.schema.json
export const SCHEMA_VERSION = '1.0';
export const LAYERS = ['boundary', 'parcels', 'coreAreas', 'retained', 'roads', 'water', 'crossings', 'treatments'];
/** Spine layers: written only when present, so a package without them is unchanged. */
export const SPINE_LAYERS = ['streams', 'connectors'];

/** Wrap connectivity inputs as a Landscape Package. */
export function toLandscapePackage(input, {name, generator = 'dfm-core', sources = {}, created = new Date().toISOString()} = {}) {
  const layers = Object.fromEntries(LAYERS.map(k => [k, structuredClone(input[k] ?? [])]));
  for (const k of SPINE_LAYERS) if (input[k]?.length) layers[k] = structuredClone(input[k]);
  return {dfm_package: SCHEMA_VERSION, name: name ?? 'Untitled landscape', created, generator, crs: 'EPSG:4326', units: {length: 'm', area: 'm2'}, params: structuredClone(input.params ?? {}), sources: structuredClone(sources), layers};
}

/** Read a Landscape Package into connectivity inputs. Throws on an unsupported version. */
export function fromLandscapePackage(pkg) {
  if (pkg?.dfm_package !== SCHEMA_VERSION) throw new Error(`Unsupported DFM package version ${pkg?.dfm_package}; expected ${SCHEMA_VERSION}.`);
  if (pkg.crs !== 'EPSG:4326') throw new Error('DFM packages use WGS84 longitude/latitude (EPSG:4326).');
  const spine = Object.fromEntries(SPINE_LAYERS.filter(k => pkg.layers?.[k]?.length).map(k => [k, pkg.layers[k]]));
  // Spine layers appear only when present, so a package without them reads exactly as before (same checksum).
  return {...Object.fromEntries(LAYERS.map(k => [k, pkg.layers?.[k] ?? []])), ...spine, params: pkg.params ?? {}};
}
