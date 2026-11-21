// Simulator scenario builder (seed, flood/cloud/drought, presets) and sensor toggles.
// The same state drives the compact Explorer popover and the full builder on the Data screen.
import { el, $$, toast } from './util.js';
import { S, emit, on, PRESETS, SENSORS } from './state.js';

let uid = 0;
export function buildScenario(host, { compact = false } = {}) {
  const id = 'sc' + uid++;
  const slider = (k, label) => el('div', { class: 'field' },
    el('label', { for: `${id}-${k}` }, label, el('output', { class: 'num', id: `${id}-o-${k}`, for: `${id}-${k}` }, S.scen[k].toFixed(2))),
    el('input', { type: 'range', id: `${id}-${k}`, 'data-k': k, min: '0', max: '1', step: '0.05', value: S.scen[k],
      oninput: e => { S.scen[k] = +e.target.value; S.scen.source = 'sim'; changed(true); } }));
  const seed = el('input', { type: 'number', id: `${id}-seed`, class: 'num', min: '0', max: '999999', value: S.scen.seed,
    onchange: e => { S.scen.seed = Math.max(0, Math.min(999999, +e.target.value | 0)); S.scen.source = 'sim'; changed(); } });
  const presets = el('div', { class: 'presets', role: 'group', 'aria-label': 'Presets' }, PRESETS.map((p, i) =>
    el('button', { class: 'row-btn', 'data-i': i, 'aria-pressed': 'false', onclick: () => { Object.assign(S.scen, { seed: p.seed, flood: p.flood, cloud: p.cloud, drought: p.drought, source: 'sim', scene_id: null, name: null }); changed(); } },
      el('span', {}, p.n), el('span', { class: 'meta num' }, `${p.flood.toFixed(1)} · ${p.cloud.toFixed(1)} · ${p.drought.toFixed(1)}`))));
  const sensors = el('div', { class: 'seg', role: 'group', 'aria-label': 'Sensors fed to the model' }, SENSORS.map(([k, n, d], i) =>
    el('button', { 'data-sensor': k, 'aria-pressed': 'true', title: d, onclick: () => {
      if (S.scen.sensors[i] && S.scen.sensors.filter(Boolean).length === 1) return toast('Keep at least one sensor on');
      S.scen.sensors[i] = !S.scen.sensors[i]; changed(true);
    } }, n)));
  const modelSel = el('select', { id: `${id}-model`, onchange: async e => (await import('./main.js')).setModel(e.target.value) });
  const fillModels = () => { modelSel.replaceChildren(...S.models.map(m => el('option', { value: m.id, disabled: m.status !== 'ready' ? '' : null, selected: m.id === S.model ? '' : null }, m.name + (m.status !== 'ready' ? ` (${m.status.replace('_', ' ')})` : '')))); };
  on('models', fillModels); on('model', fillModels); fillModels();
  host.append(el('div', { class: 'scenario' + (compact ? ' compact' : '') },
    compact ? el('div', { class: 'field mobile-only' }, el('label', { for: `${id}-model` }, 'Model'), modelSel) : null,
    el('div', { class: 'field' }, el('label', {}, 'Sensors fed to the model'), sensors),
    el('div', { class: 'field seedrow' }, el('label', { for: `${id}-seed` }, 'Seed'), seed,
      el('button', { class: 'btn', onclick: () => { S.scen.seed = Math.floor(Math.random() * 99999); S.scen.source = 'sim'; changed(); } }, 'Shuffle')),
    slider('flood', 'Flood'), slider('cloud', 'Cloud cover'), slider('drought', 'Drought'),
    el('div', { class: 'field' }, el('label', {}, 'Presets', el('span', { class: 'meta' }, 'flood · cloud · drought')), presets)));
  sync();
}

function changed(debounced) { sync(); emit('scenario', { debounced }); }

export function sync() {
  for (const inp of $$('.scenario input[type=range]')) {
    const k = inp.dataset.k; inp.value = S.scen[k];
    const o = document.getElementById(inp.id.replace(`-${k}`, `-o-${k}`)); if (o) o.textContent = (+S.scen[k]).toFixed(2);
  }
  for (const inp of $$('.scenario input[type=number]')) inp.value = S.scen.seed;
  for (const b of $$('.scenario .presets button')) {
    const p = PRESETS[b.dataset.i];
    b.setAttribute('aria-pressed', S.scen.source === 'sim' && p.seed === S.scen.seed && p.flood === S.scen.flood && p.cloud === S.scen.cloud && p.drought === S.scen.drought);
  }
  for (const b of $$('.scenario [data-sensor]')) b.setAttribute('aria-pressed', S.scen.sensors[SENSORS.findIndex(s => s[0] === b.dataset.sensor)]);
}
on('scenario-sync', sync);
