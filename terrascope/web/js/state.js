// Shared application state, a tiny event bus and the static catalogues (layers, presets).
import { store } from './util.js';

const bus = new EventTarget();
export const on = (ev, fn) => bus.addEventListener(ev, e => fn(e.detail));
export const emit = (ev, detail) => bus.dispatchEvent(new CustomEvent(ev, { detail }));

// Layer catalogue, listed top → bottom (first row draws on top). `kind` decides the renderer.
export const GROUPS = [['pred', 'Predictions'], ['fm', 'Foundation features'], ['in', 'Inputs']];
export const LAYERS = [
  { k: 'flood', g: 'pred', name: 'Flood extent', kind: 'flood', need: 'pix.flood', truth: 'flood_true', desc: 'Flood probability re-thresholded live on the GPU' },
  { k: 'lc', g: 'pred', name: 'Land cover', kind: 'lc', need: 'pix.lc', truth: 'lc_true', desc: 'Predicted class per pixel; hover a class to isolate it' },
  { k: 'veg', g: 'pred', name: 'Vegetation health', kind: 'png', truth: 'veg_true', desc: 'Predicted vegetation health index, 0 to 0.85' },
  { k: 'unc', g: 'pred', name: 'Uncertainty', kind: 'png', desc: 'Normalised entropy of the land-cover prediction' },
  { k: 'sim', g: 'fm', name: 'Similarity', kind: 'sim', desc: 'Click the map: cosine similarity of encoder features to that point' },
  { k: 'pca', g: 'fm', name: 'Embeddings', kind: 'png', px: true, desc: 'First three principal components of encoder tokens as RGB' },
  { k: 'recon', g: 'fm', name: 'Cloud-free', kind: 'recon', pair: 'rgb', desc: 'Clear pixels kept, clouded pixels rebuilt from SAR, terrain and context' },
  { k: 'rgb', g: 'in', name: 'True colour', kind: 'png', sensor: 0, desc: 'B4, B3, B2 as the sensor recorded them' },
  { k: 'fcc', g: 'in', name: 'False colour', kind: 'png', sensor: 0, desc: 'B8, B4, B3: vegetation appears red' },
  { k: 'ndvi', g: 'in', name: 'NDVI', kind: 'png', sensor: 0, desc: 'Normalised difference vegetation index from the optical input' },
  { k: 'sar', g: 'in', name: 'Radar', kind: 'png', sensor: 1, desc: 'R = VV, G = VH, B = VV − VH; open water is dark' },
  { k: 'dem', g: 'in', name: 'Terrain', kind: 'png', sensor: 2, desc: 'Elevation tint with hillshade' },
];
export const layerDef = k => LAYERS.find(l => l.k === k);

export const LEGENDS = {
  veg: { ramp: 'linear-gradient(90deg,#a50026,#f46d43,#fee08b,#a6d96a,#006837)', a: '0', b: '0.85', label: 'Vegetation health' },
  ndvi: { ramp: 'linear-gradient(90deg,#a50026,#f46d43,#fee08b,#a6d96a,#006837)', a: '−0.2', b: '0.9', label: 'NDVI' },
  unc: { ramp: 'linear-gradient(90deg,#000004,#51127c,#b73779,#fc8761,#fcfdbf)', a: '0', b: '0.8', label: 'Normalised entropy' },
  dem: { ramp: 'linear-gradient(90deg,#333399,#00b3b3,#7ed97e,#f0e68c,#a0764f,#ffffff)', a: 'low', b: 'high', label: 'Elevation' },
  sim: { ramp: 'linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)', a: 'dissimilar', b: 'similar', label: 'Cosine similarity' },
};

export const PRESETS = [
  { n: 'Clear day', seed: 12, flood: 0, cloud: 0, drought: 0 },
  { n: 'Flash flood', seed: 7, flood: 0.8, cloud: 0.05, drought: 0 },
  { n: 'Monsoon flood under cloud', seed: 7, flood: 0.8, cloud: 0.7, drought: 0.1 },
  { n: 'Drought in haze', seed: 31, flood: 0, cloud: 0.3, drought: 0.9 },
  { n: 'River delta', seed: 52, flood: 0.55, cloud: 0.4, drought: 0.3 },
];
export const SENSORS = [['optical', 'Optical', '6 bands, B2–B12'], ['sar', 'SAR', 'VV, VH'], ['dem', 'DEM', 'elevation, slope']];
export const STAGES = ['queued', 'preparing', 'encoding', 'decoding', 'rendering', 'done'];

const defaults = LAYERS.map(l => ({ k: l.k, on: ['recon', 'flood'].includes(l.k), op: 1 }));
const saved = store.get('layers', null);
export const S = {
  route: 'explorer', health: null, models: [], model: store.get('model', 'terrafm'),
  scen: { source: 'sim', seed: 7, size: 128, flood: 0.55, cloud: 0.4, drought: 0.3, sensors: [true, true, true], scene_id: null, name: null },
  res: null, pix: {}, job: null, busy: false,
  layers: Array.isArray(saved) && saved.length === defaults.length && saved.every(x => layerDef(x.k)) ? saved : defaults,
  active: store.get('active', 'recon'), thr: 0.5, hoverClass: -1,
  cursor: null, pin: null, sim: null,
  cmpTruth: false, cmpMode: 'swipe', cmpWith: null, blendT: 0.5,
  scenes: [], uploads: [],
};
export const saveLayers = () => { store.set('layers', S.layers); store.set('active', S.active); };

/** Is a layer present in the current result? (the UI hides absent layers) */
export function available(k) {
  const d = layerDef(k), r = S.res;
  if (!r) return false;
  if (d.kind === 'lc') return !!S.pix.lc;
  if (d.kind === 'flood') return !!S.pix.flood;
  if (d.kind === 'sim') return true;
  return r.layers.includes(k);
}
export const modelById = id => S.models.find(m => m.id === id);
