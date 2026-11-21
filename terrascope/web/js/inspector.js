// Inspector: pixel values under the cursor, scene summary (heuristic indices, composition,
// flood area at the live threshold, accuracy) and provenance of the current analysis.
import { el, fmt, tweenNum, $ } from './util.js';
import { S, modelById } from './state.js';

const RISK = [['composite', 'Composite'], ['flood', 'Flood exposure'], ['drought', 'Vegetation stress'], ['urban', 'Urban pressure']];
const prevRisk = {};
const BREACH = 50; // a heuristic index ≥ 50 is drawn in the warning colour

/** Accept the old numeric shape and the new {value, formula, inputs, kind} shape. */
export function riskItem(v) {
  if (v == null) return null;
  if (typeof v === 'number') return { value: v, formula: null, inputs: null, kind: 'heuristic' };
  return { value: v.value, formula: v.formula ?? null, inputs: v.inputs ?? null, kind: v.kind || 'heuristic', note: v.note ?? null };
}
export const gsdOf = r => r?.gsd_m ?? r?.stats?.gsd_m ?? 30;
export function floodArea(pix, thr, gsd) {
  const a = pix?.flood; if (!a) return null;
  const t = Math.round(thr * 255); let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] >= t) n++;
  return { km2: n * (gsd / 1000) ** 2, frac: n / a.length };
}
export const isSimulated = r => (r?.provenance?.data_source ?? (r?.meta?.source === 'sim' ? 'simulated' : r?.meta?.source)) === 'simulated';

const row = (k, v, cls = '') => el('div', { class: 'kv ' + cls }, el('span', { class: 'k' }, k), el('span', { class: 'v num' }, v));

export function buildInspector(host, { idp = 'ex' } = {}) {
  host.append(
    el('div', { class: 'phead' }, el('h2', {}, 'Inspector')),
    el('div', { class: 'pbody' },
      el('section', { class: 'sec', 'aria-labelledby': idp + '-px-h' },
        el('h3', { id: idp + '-px-h' }, 'Pixel', el('span', { class: 'meta num', id: idp + '-xy' }, '–')),
        el('p', { class: 'sr-only', 'aria-live': 'polite', id: idp + '-live' }),
        el('div', { id: idp + '-pix', class: 'rows' }, el('p', { class: 'muted' }, 'Hover the map, or tap it on touch screens'))),
      el('section', { class: 'sec', 'aria-labelledby': idp + '-sc-h' },
        el('h3', { id: idp + '-sc-h' }, 'Scene'), el('div', { id: idp + '-scene' })),
      el('section', { class: 'sec', 'aria-labelledby': idp + '-pv-h' },
        el('h3', { id: idp + '-pv-h' }, 'Provenance'), el('div', { id: idp + '-prov', class: 'rows' }))));
  return {
    cursor: px => renderPixel(idp, px),
    result: () => { renderScene(idp); renderProv(idp); },
    threshold: () => renderFloodArea(idp),
  };
}

export function pixelValues(px) {
  const r = S.res, p = S.pix; if (!r || !px) return null;
  const [H, W] = r.stats.size, i = px[1] * W + px[0];
  if (px[0] < 0 || px[1] < 0 || px[0] >= W || px[1] >= H) return null;
  const cls = r.classes, o = { x: px[0], y: px[1] };
  if (p.lc) o.lc = cls[p.lc[i]];
  if (p.lc_true) o.lcTrue = cls[p.lc_true[i]];
  if (p.flood) o.flood = p.flood[i] / 255;
  if (p.flood_true) o.floodTrue = p.flood_true[i] > 0;
  if (p.veg) o.veg = p.veg[i] / 255;
  if (p.unc) o.unc = p.unc[i] / 255;
  if (p.cloud_mask) o.rebuilt = p.cloud_mask[i] > 0;
  if (p.lc_prob) o.prob = p.lc_prob; // not sent by current servers
  return o;
}

function renderPixel(idp, px) {
  const box = $('#' + idp + '-pix'), v = pixelValues(px);
  $('#' + idp + '-xy').textContent = v ? `x ${String(v.x).padStart(3, '0')}  y ${String(v.y).padStart(3, '0')}` : '–';
  if (!v) return;
  const rows = [];
  if (v.lc) rows.push(el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Land cover'), el('span', { class: 'v' }, el('i', { class: 'sw', style: { background: v.lc.color } }), v.lc.name)));
  if (v.lcTrue) rows.push(el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Truth class'), el('span', { class: 'v' }, el('i', { class: 'sw', style: { background: v.lcTrue.color } }), v.lcTrue.name)));
  if (v.flood != null) rows.push(row('Flood probability', fmt.n(v.flood, 2) + (v.flood >= S.thr ? '  flooded' : '')));
  if (v.floodTrue != null) rows.push(row('Truth flood', v.floodTrue ? 'water' : 'dry'));
  if (v.veg != null) rows.push(row('Vegetation index', fmt.n(v.veg, 2)));
  if (v.unc != null) rows.push(row('Uncertainty', fmt.n(v.unc, 2)));
  if (v.rebuilt != null) rows.push(row('Optical pixel', v.rebuilt ? 'rebuilt' : 'observed'));
  rows.push(el('p', { class: 'muted small' }, 'Per-class probabilities are not sent to the browser'));
  box.replaceChildren(...rows);
  $('#' + idp + '-live').textContent = [`Pixel ${v.x}, ${v.y}`, v.lc?.name, v.flood != null && `flood ${Math.round(v.flood * 100)}%`, v.veg != null && `vegetation ${v.veg.toFixed(2)}`].filter(Boolean).join(', ');
}

function riskRow(idp, key, label, raw) {
  const r = riskItem(raw); if (!r) return null;
  const hot = r.value >= BREACH, id = `${idp}-tip-${key}`;
  const inputs = r.inputs ? Object.entries(r.inputs).map(([k, v]) => `${k.replace(/_/g, ' ')} = ${typeof v === 'number' ? +v.toFixed(4) : v}`).join('\n') : null;
  const val = el('span', { class: 'v num', 'data-v': prevRisk[id] ?? '' }, Math.round(prevRisk[id] ?? r.value) + '');
  prevRisk[id] = r.value; queueMicrotask(() => tweenNum(val, r.value, v => Math.round(v) + ''));
  return el('div', { class: 'risk' + (hot ? ' hot' : ''), tabindex: '0', 'aria-describedby': id },
    el('div', { class: 'kv' }, el('span', { class: 'k' }, label), val),
    el('div', { class: 'track', 'aria-hidden': 'true' }, el('i', { style: { transform: `scaleX(${Math.min(1, r.value / 100)})` } })),
    el('div', { class: 'tip', role: 'tooltip', id },
      el('b', {}, `Heuristic index, 0–100${hot ? ', above ' + BREACH : ''}`),
      el('code', {}, r.formula || 'Formula not reported by this server version'),
      inputs ? el('pre', {}, inputs) : null, r.note ? el('span', { class: 'muted' }, r.note) : null));
}

function renderScene(idp) {
  const r = S.res, st = r.stats, box = $('#' + idp + '-scene');
  const risk = st.risk || {};
  const comp = st.composition || [];
  const parts = [
    el('h4', {}, 'Heuristic indices', el('span', { class: 'meta' }, 'not calibrated risk')),
    el('div', { class: 'risks' }, RISK.map(([k, l]) => riskRow(idp, k, l, risk[k]))),
    el('h4', {}, 'Flood area', el('span', { class: 'meta num', id: idp + '-thr' }, `p ≥ ${S.thr.toFixed(2)}`)),
    el('div', { class: 'kv big' }, el('span', { class: 'v num', id: idp + '-farea', 'data-v': '' }, '–'), el('span', { class: 'k num', id: idp + '-ffrac' }, '')),
    el('h4', {}, 'Land cover'),
    el('div', { class: 'stack', 'aria-hidden': 'true' }, comp.map((c, i) => el('i', { style: { flexGrow: Math.max(c.frac, 0.002), background: r.classes[i].color }, 'data-cls': i }))),
    el('div', { class: 'rows comp' }, comp.map((c, i) => [c, i]).sort((a, b) => b[0].frac - a[0].frac).map(([c, i]) =>
      el('div', { class: 'kv cls', 'data-cls': i, tabindex: '0', 'aria-label': `${c.name} ${fmt.pct(c.frac)}, hover or focus to isolate on the map` },
        el('span', { class: 'k' }, el('i', { class: 'sw', style: { background: r.classes[i].color } }), c.name),
        el('span', { class: 'v num' }, fmt.pct(c.frac), el('span', { class: 'muted' }, ' ' + fmt.km2(c.km2)))))),
    el('h4', {}, 'Model'),
    row('Mean confidence', fmt.pct(st.confidence, 0)),
    row('Calibrated confidence', st.confidence_calibrated == null ? (st.calibration && (st.calibration.applied ?? st.calibration.apply) === false ? 'not applied' : 'no calibration file') : fmt.pct(st.confidence_calibrated, 0)),
    st.calibration ? row('ECE before → after', `${fmt.n(st.calibration.ece_before, 3)} → ${fmt.n(st.calibration.ece_after, 3)}`) : null,
    st.calibration && (st.calibration.applied ?? st.calibration.apply) === false ? el('p', { class: 'muted small' }, 'Temperature scaling not applied: it did not lower validation ECE, so the raw confidence is shown') : null,
    st.cloud_frac != null ? row('Optical pixels under cloud', fmt.pct(st.cloud_frac, 0)) : null,
  ];
  if (st.insights?.length) parts.push(el('h4', {}, 'Findings'), el('ul', { class: 'findings' }, st.insights.map(i => el('li', { 'data-kind': i.kind }, i.text))));
  const a = st.accuracy;
  if (a) parts.push(el('h4', {}, 'Against simulator truth'),
    row('Land-cover mIoU', fmt.n(a.miou, 3)), row('Pixel accuracy', fmt.pct(a.oa)),
    row('Flood IoU at p ≥ 0.50', a.flood_iou == null ? 'no flood in scene' : fmt.n(a.flood_iou, 3)), row('Vegetation MAE', fmt.n(a.veg_mae, 3)));
  box.replaceChildren(...parts.filter(Boolean));
  renderFloodArea(idp);
}

function renderFloodArea(idp) {
  const n = $('#' + idp + '-farea'); if (!n || !S.res) return;
  const a = floodArea(S.pix, S.thr, gsdOf(S.res));
  $('#' + idp + '-thr').textContent = `p ≥ ${S.thr.toFixed(2)}`;
  if (!a) { n.textContent = 'not predicted by this model'; return; }
  tweenNum(n, a.km2, v => v.toFixed(2) + ' km²');
  $('#' + idp + '-ffrac').textContent = fmt.pct(a.frac) + ' of scene';
}

function renderProv(idp) {
  const r = S.res, p = r.provenance || {}, m = modelById(p.model_id || S.model) || {};
  const src = p.data_source || (r.meta?.source === 'sim' ? 'simulated' : r.meta?.source || 'unknown');
  $('#' + idp + '-prov').replaceChildren(
    row('Model', p.model_name || m.name || 'TerraFM'),
    row('Parameters', fmt.params(p.params ?? m.params ?? S.health?.params)),
    row('Weights', p.weights_file || m.weights_file || 'not reported'),
    el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Data source'), el('span', { class: 'v' }, src === 'simulated' ? el('span', { class: 'tag' }, 'simulated') : src)),
    r.meta?.seed != null && src === 'simulated' ? row('Simulator', `seed ${r.meta.seed} · flood ${r.meta.flood} · cloud ${r.meta.cloud} · drought ${r.meta.drought}`) : null,
    row('Acquisition date', p.acquisition_date || (src === 'simulated' ? 'none, synthetic scene' : 'unknown')),
    row('Processed', p.processed_at ? fmt.date(p.processed_at) : 'not reported'),
    row('Device', p.device || S.health?.device || 'not reported'),
    row('Runtime', fmt.ms(p.runtime_ms ?? r.stats.timing_ms)),
    row('Sensors used', ['optical', 'SAR', 'DEM'].filter((_, i) => (r.sensors || [1, 1, 1])[i]).join(', ')));
}
