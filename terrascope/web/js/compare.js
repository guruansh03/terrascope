// Compare screen: analysis A (from the Explorer) against analysis B (another model, or the same
// place under different simulated conditions / another scene). Swipe or split with synced pan/zoom,
// plus change layers from /api/change when the server offers them.
import { el, icon, $, $$, fmt, toast } from './util.js';
import { S, on, LAYERS, layerDef, SENSORS } from './state.js';
import * as api from './api.js';
import { MapView } from './mapview.js';
import { LayerStack } from './render.js';
import { Progress, analyzeOnce } from './run.js';
import { gsdOf } from './inspector.js';

const C = { a: null, aPix: null, b: null, bPix: null, k: 'lc', mode: 'swipe', bKind: null, bModel: null, bScen: null, change: null, chOn: new Set() };
const CH_LABEL = { flood_onset: 'Flood onset', veg_loss: 'Vegetation loss', urban_growth: 'Urban growth', lc_change: 'Land-cover change' };

export function buildCompare(root) {
  const left = el('aside', { class: 'panel left', 'aria-label': 'Comparison setup' });
  const right = el('aside', { class: 'panel right', 'aria-label': 'Comparison readout' });
  const maps = el('div', { class: 'cmp-maps', 'data-mode': C.mode }, el('div', { class: 'cmp-a' }), el('div', { class: 'cmp-b' }));
  root.append(maps, left, right);
  const sync = (src, dst) => v => { if (C.mode === 'split') dst().setView(v); };
  const mA = new MapView($('.cmp-a', maps), { label: 'Comparison map A', onHover: px => readout(px), onLeave: () => readout(null), onPick: px => readout(px), onView: sync(0, () => mB) });
  const mB = new MapView($('.cmp-b', maps), { label: 'Comparison map B', onHover: px => readout(px), onLeave: () => readout(null), onPick: px => readout(px), onView: sync(1, () => mA) });
  const prog = new Progress(mA.world, mA.host);
  const sA = new LayerStack(), sB = new LayerStack();
  const tagA = el('div', { class: 'maptag l' }), tagB = el('div', { class: 'maptag r' });
  mA.host.append(tagA); mB.host.append(tagB.cloneNode()); mA.host.append(tagB);

  // ----- setup panel
  const modelSel = el('select', { id: 'cmp-model', 'aria-label': 'Model for B' });
  const sceneSel = el('select', { id: 'cmp-scene', 'aria-label': 'Scene for B' });
  const cond = ['flood', 'cloud', 'drought'].map(k => el('div', { class: 'field' },
    el('label', { for: 'cmpb-' + k }, k[0].toUpperCase() + k.slice(1), el('output', { class: 'num', id: 'cmpb-o-' + k }, '')),
    el('input', { type: 'range', id: 'cmpb-' + k, 'data-k': k, min: '0', max: '1', step: '0.05', oninput: e => { C.bScen[k] = +e.target.value; $('#cmpb-o-' + k).textContent = C.bScen[k].toFixed(2); } })));
  const kindSeg = el('div', { class: 'seg', role: 'group', 'aria-label': 'What differs in B' },
    [['model', 'Model'], ['conditions', 'Conditions'], ['scene', 'Scene']].map(([k, n]) => el('button', { 'data-kind': k, 'aria-pressed': String(C.bKind === k), onclick: () => { C.bKind = k; setup(); } }, n)));
  const layerSel = el('select', { id: 'cmp-layer', 'aria-label': 'Layer to compare', onchange: e => { C.k = e.target.value; draw(); } });
  const modeSeg = el('div', { class: 'seg', role: 'group', 'aria-label': 'View mode' },
    [['swipe', 'Swipe'], ['split', 'Split']].map(([k, n]) => el('button', { 'data-mode': k, 'aria-pressed': String(C.mode === k), onclick: () => { C.mode = k; maps.dataset.mode = k; $$('[data-mode]', modeSeg).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === k))); draw(); } }, n)));
  left.append(el('div', { class: 'phead' }, el('h2', {}, 'Compare')), el('div', { class: 'pbody pad' },
    el('section', { class: 'sec flat' }, el('h3', {}, 'A', el('span', { class: 'meta' }, 'Explorer analysis')), el('p', { id: 'cmp-a-desc', class: 'muted' }, 'Run an analysis in the Explorer first')),
    el('section', { class: 'sec flat' }, el('h3', {}, 'B'), el('div', { class: 'field' }, kindSeg),
      el('div', { class: 'field', 'data-for': 'model' }, el('label', { for: 'cmp-model' }, 'Model'), modelSel, el('p', { id: 'cmp-model-note', class: 'muted small' })),
      el('div', { 'data-for': 'conditions' }, el('p', { class: 'muted small' }, 'Same seed as A, different simulated conditions, as if acquired on another date'), cond),
      el('div', { class: 'field', 'data-for': 'scene' }, el('label', { for: 'cmp-scene' }, 'Scene'), sceneSel, el('p', { id: 'cmp-scene-note', class: 'muted small' })),
      el('button', { class: 'btn primary', id: 'cmp-run', onclick: runB }, icon('play'), 'Run B')),
    el('section', { class: 'sec flat' }, el('h3', {}, 'View'),
      el('div', { class: 'field' }, el('label', { for: 'cmp-layer' }, 'Layer'), layerSel), el('div', { class: 'field' }, el('label', {}, 'Mode'), modeSeg)),
    el('section', { class: 'sec flat' }, el('h3', {}, 'Change layers', el('span', { class: 'meta' }, 'A → B')), el('div', { id: 'cmp-change' }, el('p', { class: 'muted' }, 'Run B to compute change')))));
  right.append(el('div', { class: 'phead' }, el('h2', {}, 'Readout')), el('div', { class: 'pbody' },
    el('section', { class: 'sec' }, el('h3', {}, 'Pixel', el('span', { class: 'meta num', id: 'cmp-xy' }, '–')), el('div', { id: 'cmp-px', class: 'rows' }, el('p', { class: 'muted' }, 'Hover either map'))),
    el('section', { class: 'sec' }, el('h3', {}, 'Scene'), el('div', { id: 'cmp-scene-stats', class: 'rows' }))));

  function setup() {
    $$('[data-kind]', kindSeg).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.kind === C.bKind)));
    $$('[data-for]', left).forEach(n => { n.hidden = n.dataset.for !== C.bKind; });
    modelSel.replaceChildren(...S.models.map(m => el('option', { value: m.id, disabled: m.status !== 'ready' ? '' : null }, m.name + (m.status !== 'ready' ? ` (${m.status.replace('_', ' ')})` : ''))));
    const ready = S.models.filter(m => m.status === 'ready');
    const other = ready.filter(m => m.id !== S.model);
    if (!C.bKind) { C.bKind = other.length ? 'model' : 'conditions'; $$('[data-kind]', kindSeg).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.kind === C.bKind))); $$('[data-for]', left).forEach(n => { n.hidden = n.dataset.for !== C.bKind; }); }
    C.bModel = other.find(m => m.id === C.bModel)?.id || other[0]?.id || null; modelSel.value = C.bModel || '';
    $('#cmp-model-note').textContent = other.length ? '' : 'Only one model is ready, so there is nothing to compare across models. Compare other conditions or another scene instead';
    modelSel.disabled = !other.length;
    modelSel.onchange = e => { C.bModel = e.target.value; };
    C.bScen ??= { ...S.scen, flood: Math.min(1, S.scen.flood + 0.3), drought: S.scen.drought };
    ['flood', 'cloud', 'drought'].forEach(k => { $('#cmpb-' + k).value = C.bScen[k]; $('#cmpb-o-' + k).textContent = (+C.bScen[k]).toFixed(2); });
    sceneSel.replaceChildren(...S.scenes.map(s => el('option', { value: s.id }, `${s.name || s.id}${s.acquisition_date ? ' · ' + s.acquisition_date : ''}`)));
    $('#cmp-scene-note').textContent = api.caps.scenes === false ? 'Scene catalogue not available on this server' : S.scenes.length ? '' : 'No scenes yet; add one on the Data screen';
    $('#cmp-run').disabled = !C.a || (C.bKind === 'scene' && !S.scenes.length) || (C.bKind === 'model' && !C.bModel);
  }

  async function runB() {
    if (!C.a) return toast('Run an analysis in the Explorer first');
    let scen = { ...S.scen }, model = S.model;
    if (C.bKind === 'model') model = C.bModel;
    else if (C.bKind === 'conditions') scen = { ...S.scen, ...C.bScen, source: 'sim', scene_id: null, seed: S.scen.seed };
    else scen = { ...S.scen, source: 'scene', scene_id: sceneSel.value };
    $('#cmp-run').disabled = true;
    const r = await analyzeOnce(scen, model, prog);
    $('#cmp-run').disabled = false;
    if (!r.ok) return toast(r.error || 'Analysis B failed', 'bad');
    C.b = r.data; C.bPix = r.pix; C.bLabel = C.bKind === 'model' ? (S.models.find(m => m.id === model)?.name || model) : C.bKind === 'conditions' ? `flood ${scen.flood.toFixed(2)} · cloud ${scen.cloud.toFixed(2)} · drought ${scen.drought.toFixed(2)}` : sceneSel.selectedOptions[0]?.textContent;
    sB.bind(C.b, C.bPix, [C.b.stats.size[1] >> 1, C.b.stats.size[0] >> 1]);
    await loadChange(); fillLayers(); draw(); stats();
    if (matchMedia('(max-width: 820px)').matches) maps.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  async function loadChange() {
    const box = $('#cmp-change'); C.change = null; C.chOn.clear();
    const r = await api.change(C.a.id, C.b.id);
    if (r.unsupported) return box.replaceChildren(el('p', { class: 'muted' }, 'This server has no change detection endpoint (/api/change)'));
    if (!r.ok) return box.replaceChildren(el('p', { class: 'muted' }, r.status === 409 ? r.error || 'A and B have different sizes' : r.error));
    C.change = r.data; const st = r.data.stats || {};
    box.replaceChildren(
      el('ul', { class: 'chlist' }, (r.data.layers || []).map(n => el('li', {}, el('label', {}, el('input', { type: 'checkbox', 'data-ch': n, onchange: e => { e.target.checked ? C.chOn.add(n) : C.chOn.delete(n); draw(); } }), CH_LABEL[n] || n)))),
      el('div', { class: 'rows' },
        el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Flood onset'), el('span', { class: 'v num' }, fmt.km2(st.flood_onset_km2))),
        el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Vegetation loss'), el('span', { class: 'v num' }, fmt.km2(st.veg_loss_km2))),
        el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Urban growth'), el('span', { class: 'v num' }, fmt.km2(st.urban_growth_km2))),
        el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Land cover changed'), el('span', { class: 'v num' }, fmt.pct(st.lc_changed_frac)))));
  }

  function fillLayers() {
    const ks = LAYERS.filter(l => sA.available(l.k) && (!C.b || sB.available(l.k)) && l.k !== 'sim').map(l => l.k);
    if (!ks.includes(C.k)) C.k = ks.includes('lc') ? 'lc' : ks[0];
    layerSel.replaceChildren(...ks.map(k => el('option', { value: k, selected: k === C.k ? '' : null }, layerDef(k).name)));
  }
  function nodesFor(stack) {
    const list = [{ k: C.k, on: true, op: 1 }];
    if (layerDef(C.k).kind === 'flood') list.push({ k: 'recon', on: true, op: 1 });
    const out = stack.ordered(list);
    if (C.k === 'flood') stack.drawFlood(S.thr);
    // change layers describe A → B, so they are drawn once, on the B side only
    if (stack === sB) for (const n of C.chOn) out.push({ key: 'ch-' + n, node: chNode('b', n) });
    return out;
  }
  const chNodes = {};
  function chNode(side, n) { const key = side + n + C.change.id; return chNodes[key] ??= el('img', { alt: '', src: `/api/layer/${C.change.id}/${n}.png`, class: 'px' }); }

  function draw() {
    if (!C.a) return;
    mA.setScene(C.a.stats.size, gsdOf(C.a)); mB.setScene((C.b || C.a).stats.size, gsdOf(C.b || C.a));
    mA.setLayers(nodesFor(sA));
    tagA.textContent = 'A · ' + C.aLabel; tagA.hidden = false;
    if (!C.b) { mA.setCompare(false); mB.setLayers([]); $$('.maptag.r', maps).forEach(t => t.hidden = true); return; }
    const bNodes = nodesFor(sB);
    $$('.maptag.r', maps).forEach(t => { t.hidden = false; t.textContent = 'B · ' + C.bLabel; });
    tagB.hidden = C.mode !== 'swipe';
    if (C.mode === 'swipe') { mB.setLayers([]); mA.setLayers(bNodes, mA.world2); mA.setCompare(true); }
    else { mA.setCompare(false); mA.setLayers([], mA.world2); mB.setLayers(bNodes); mB.setView(mA.v); }
  }

  function readout(px) {
    $('#cmp-xy').textContent = px ? `x ${px[0]}  y ${px[1]}` : '–';
    if (!px || !C.a) return;
    const val = (r, p) => { if (!r) return {}; const W = r.stats.size[1], i = px[1] * W + px[0]; return { lc: p.lc && r.classes[p.lc[i]]?.name, fl: p.flood && p.flood[i] / 255, veg: p.veg && p.veg[i] / 255, unc: p.unc && p.unc[i] / 255 }; };
    const a = val(C.a, C.aPix), b = val(C.b, C.bPix || {});
    const row = (k, x, y) => el('div', { class: 'kv three' }, el('span', { class: 'k' }, k), el('span', { class: 'v num' }, x ?? '–'), el('span', { class: 'v num' }, y ?? '–'));
    const f = v => v == null ? null : v.toFixed(2);
    $('#cmp-px').replaceChildren(el('div', { class: 'kv three head' }, el('span'), el('span', { class: 'v' }, 'A'), el('span', { class: 'v' }, 'B')),
      row('Land cover', a.lc, b.lc), row('Flood p', f(a.fl), f(b.fl)), row('Vegetation', f(a.veg), f(b.veg)), row('Uncertainty', f(a.unc), f(b.unc)));
  }
  function stats() {
    const s = r => r ? r.stats : null, a = s(C.a), b = s(C.b);
    const row = (k, x, y) => el('div', { class: 'kv three' }, el('span', { class: 'k' }, k), el('span', { class: 'v num' }, x), el('span', { class: 'v num' }, y));
    $('#cmp-scene-stats').replaceChildren(el('div', { class: 'kv three head' }, el('span'), el('span', { class: 'v' }, 'A'), el('span', { class: 'v' }, 'B')),
      row('Flood area', fmt.km2(a?.flood?.km2), b ? fmt.km2(b.flood?.km2) : '–'), row('Veg. stress', fmt.pct(a?.veg?.stress_frac, 0), b ? fmt.pct(b.veg?.stress_frac, 0) : '–'),
      row('Confidence', fmt.pct(a?.confidence, 0), b ? fmt.pct(b.confidence, 0) : '–'), row('Runtime', fmt.ms(a?.timing_ms), b ? fmt.ms(b.timing_ms) : '–'));
  }

  function takeA() {
    if (!S.res) return;
    C.a = S.res; C.aPix = S.pix; C.b = null; C.bPix = null; C.change = null; C.chOn.clear();
    C.aLabel = (S.models.find(m => m.id === S.model)?.name || 'TerraFM') + (S.res.meta?.source === 'sim' ? ` · seed ${S.res.meta.seed}` : '');
    sA.bind(C.a, C.aPix, S.sim);
    $('#cmp-a-desc').textContent = `${C.aLabel} · ${S.res.stats.size[1]} × ${S.res.stats.size[0]} px` + (S.res.meta?.source === 'sim' ? ` · flood ${S.res.meta.flood} cloud ${S.res.meta.cloud} drought ${S.res.meta.drought}` : '');
    $('#cmp-change').replaceChildren(el('p', { class: 'muted' }, 'Run B to compute change'));
    C.bScen = null; setup(); fillLayers(); draw(); stats();
  }
  on('result', () => { if (S.route === 'compare' || C.a) takeA(); });
  on('route', r => { if (r === 'compare') { if (!C.a || C.a !== S.res) takeA(); else setup(); } });
  setup(); takeA();
}
export { SENSORS };
