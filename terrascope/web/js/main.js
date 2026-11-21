// Shell: toolbar, hash router with View Transitions, model switcher, theme, status bar, shortcuts, boot.
import { el, icon, $, $$, fmt, store, reduced, toast } from './util.js';
import * as api from './api.js';
import { S, on, emit } from './state.js';
import { buildExplorer, ex, nthLayer } from './explorer.js';
import { openPalette, openShortcuts, closeDialogs } from './palette.js';

export const SCREENS = [['explorer', 'Explorer'], ['compare', 'Compare'], ['benchmarks', 'Benchmarks'], ['data', 'Data'], ['model', 'Model']];
const SHORT = { explorer: 'Explore', benchmarks: 'Bench' };
const built = {};
const loaders = {
  explorer: async r => buildExplorer(r),
  compare: async r => (await import('./compare.js')).buildCompare(r),
  benchmarks: async r => (await import('./bench.js')).buildBench(r),
  data: async r => (await import('./data.js')).buildData(r),
  model: async r => (await import('./modelpage.js')).buildModelPage(r),
};

// ---------- theme
const darkMQ = matchMedia('(prefers-color-scheme: dark)');
function theme(t) {
  const saved = t ?? store.get('theme', null);
  const mode = saved || (darkMQ.matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = mode;
  if (t) store.set('theme', t);
  const b = $('#theme-btn'); if (b) { b.replaceChildren(icon(mode === 'dark' ? 'sun' : 'moon')); b.setAttribute('aria-label', `Switch to ${mode === 'dark' ? 'light' : 'dark'} theme`); }
}
export const toggleTheme = () => theme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
darkMQ.addEventListener('change', () => theme());

// ---------- router
export async function go(name, push = true) {
  if (!SCREENS.find(s => s[0] === name)) name = 'explorer';
  if (push && location.hash !== '#' + name) history.pushState(null, '', '#' + name);
  const swap = async () => {
    S.route = name;
    for (const [k] of SCREENS) { const sc = $('#scr-' + k); sc.hidden = k !== name; $('#tab-' + k).setAttribute('aria-selected', String(k === name)); $('#tab-' + k).tabIndex = k === name ? 0 : -1; }
    if (!built[name]) { built[name] = true; await loaders[name]($('#scr-' + name)); }
    emit('route', name);
  };
  if (document.startViewTransition && !reduced() && S.route !== name) await document.startViewTransition(swap).updateCallbackDone;
  else await swap();
}
addEventListener('popstate', () => go(location.hash.slice(1), false));

// ---------- model switcher (from /api/models; unready models are listed but disabled with the reason)
const REASON = { weights_missing: 'weights not downloaded', not_finetuned: 'no fine-tuned heads yet' };
function renderModels() {
  const list = $('#model-list'), cur = S.models.find(m => m.id === S.model) || S.models[0];
  $('#model-btn .nm').textContent = cur ? cur.name : 'No model'; $('#model-btn').setAttribute('aria-label', `Model ${cur ? cur.name : 'none'}`);
  list.replaceChildren(...S.models.map(m => {
    const ready = m.status === 'ready';
    return el('li', { role: 'option', id: 'opt-' + m.id, 'data-id': m.id, 'aria-selected': String(m.id === S.model), 'aria-disabled': String(!ready), class: ready ? '' : 'disabled' },
      el('span', { class: 'nm' }, m.name, m.family === 'unet' ? el('span', { class: 'meta' }, ' · baseline') : null), el('span', { class: 'meta num' }, fmt.params(m.params)),
      ready ? null : el('span', { class: 'reason' }, REASON[m.status] || m.status));
  }));
}
export function setModel(id, quiet) {
  const m = S.models.find(x => x.id === id);
  if (!m) return;
  if (m.status !== 'ready') { toast(`${m.name}: ${REASON[m.status] || m.status}`, 'bad'); return; }
  if (S.model === id) return;
  S.model = id; store.set('model', id); renderModels(); emit('model', id);
  if (!quiet) emit('scenario', {});
}
export function cycleModel(dir) {
  const ready = S.models.filter(m => m.status === 'ready'); if (ready.length < 2) return toast('Only one model is ready');
  const i = ready.findIndex(m => m.id === S.model); setModel(ready[(i + dir + ready.length) % ready.length].id);
}
function modelSwitcher() {
  const btn = el('button', { class: 'btn', id: 'model-btn', 'aria-haspopup': 'listbox', 'aria-expanded': 'false', 'aria-controls': 'model-list' }, el('span', { class: 'nm' }, '…'), icon('down'));
  const list = el('ul', { class: 'card menu', id: 'model-list', role: 'listbox', 'aria-label': 'Models', tabindex: '-1', hidden: true });
  const open = o => { list.hidden = !o; btn.setAttribute('aria-expanded', String(o)); if (o) { const cur = $(`[aria-selected=true]`, list) || list.firstChild; list.setAttribute('aria-activedescendant', cur?.id || ''); list.focus(); } };
  btn.onclick = () => open(list.hidden);
  list.addEventListener('click', e => { const li = e.target.closest('li'); if (li && li.getAttribute('aria-disabled') !== 'true') { setModel(li.dataset.id); open(false); btn.focus(); } });
  list.addEventListener('keydown', e => {
    const items = $$('li', list), i = items.findIndex(x => x.id === list.getAttribute('aria-activedescendant'));
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { const j = (i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length; list.setAttribute('aria-activedescendant', items[j].id); items.forEach(x => x.classList.toggle('kbd', x === items[j])); e.preventDefault(); }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); items[i]?.click(); }
    else if (e.key === 'Escape' || e.key === 'Tab') { open(false); if (e.key === 'Escape') btn.focus(); }
  });
  document.addEventListener('pointerdown', e => { if (!list.hidden && !list.contains(e.target) && !btn.contains(e.target)) open(false); });
  return el('div', { class: 'msw' }, btn, list);
}

// ---------- shell
function shell() {
  const tabs = el('nav', { class: 'tabs', role: 'tablist', 'aria-label': 'Screens' }, SCREENS.map(([k, n]) =>
    el('button', { role: 'tab', id: 'tab-' + k, 'aria-controls': 'scr-' + k, 'aria-selected': 'false', 'aria-label': n, tabindex: '-1', onclick: () => go(k) }, el('span', { class: 'full' }, n), el('span', { class: 'short' }, SHORT[k] || n))));
  tabs.addEventListener('keydown', e => {
    const t = $$('[role=tab]', tabs), i = t.indexOf(document.activeElement); if (i < 0) return;
    let j = null; if (e.key === 'ArrowRight') j = (i + 1) % t.length; else if (e.key === 'ArrowLeft') j = (i - 1 + t.length) % t.length; else if (e.key === 'Home') j = 0; else if (e.key === 'End') j = t.length - 1;
    if (j == null) return; e.preventDefault(); t[j].focus(); go(SCREENS[j][0]);
  });
  $('#toolbar').append(
    el('a', { class: 'brand', href: '#explorer', 'aria-label': 'TerraScope, Explorer' }, el('i', { class: 'mark', 'aria-hidden': 'true', html: '<svg viewBox="0 0 16 16" width="16" height="16"><rect x="1.5" y="1.5" width="13" height="13" rx="2" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".55"/><path d="M2 11.5c2.2-2.8 4-.4 6-2.2s3.4-1.6 6-4.3" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>' }), el('span', {}, 'TerraScope')),
    tabs,
    el('div', { class: 'tb-right' }, modelSwitcher(),
      el('button', { class: 'btn primary', id: 'run-btn', onclick: () => ex.run?.(), title: 'Run analysis (R)' }, icon('play'), 'Run'),
      el('button', { class: 'btn', id: 'palette-btn', onclick: openPalette, 'aria-label': `Commands ${navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl K'}`, title: 'Command palette' }, icon('search'), el('kbd', {}, navigator.platform.includes('Mac') ? '⌘K' : 'Ctrl K')),
      el('button', { class: 'btn icon', id: 'theme-btn', onclick: toggleTheme })));
  for (const [k] of SCREENS) $('#screens').append(el('section', { class: 'screen scr-' + k, id: 'scr-' + k, role: 'tabpanel', 'aria-labelledby': 'tab-' + k, tabindex: '0', hidden: true }));
}

// ---------- status bar
function statusBar() {
  const it = (id, label) => el('span', { class: 'sb-item', id: 'sb-' + id, title: label }, el('span', { class: 'sr-only' }, label + ': '), el('span', { class: 'v num' }, '–'));
  $('#statusbar').append(el('span', { class: 'sb-item', id: 'sb-server' }, el('i', { class: 'dot' }), el('span', { class: 'v' }, 'connecting')),
    it('scene', 'Scene'), it('size', 'Size'), it('gsd', 'Ground sampling distance'), it('model', 'Model'), it('runtime', 'Runtime'), it('stage', 'Job stage'), el('span', { class: 'grow' }), it('xy', 'Cursor'));
  const set = (id, v) => { const n = $(`#sb-${id} .v`); if (n) n.textContent = v; };
  on('result', r => {
    const m = r.meta || {}, p = r.provenance || {};
    set('scene', m.source === 'sim' ? `simulated · seed ${m.seed}` : S.scen.name || m.scene_id || p.data_source || 'upload');
    set('size', `${r.stats.size[1]} × ${r.stats.size[0]} px`); set('gsd', `${r.gsd_m ?? r.stats.gsd_m} m`);
    set('model', p.model_name || S.models.find(x => x.id === S.model)?.name || 'TerraFM'); set('runtime', fmt.ms(p.runtime_ms ?? r.stats.timing_ms));
  });
  on('stage', s => set('stage', s === 'running' ? 'running' : s));
  on('cursor', px => set('xy', px ? `x ${String(px[0]).padStart(3, '0')}  y ${String(px[1]).padStart(3, '0')}` : '–'));
  on('announce', m => { $('#announce').textContent = m; });
  return set;
}

// ---------- shortcuts
function shortcuts() {
  addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); return openPalette(); }
    if (e.key === 'Escape') closeDialogs();
    const t = e.target, typing = t.closest?.('input, textarea, select, [contenteditable], dialog, [role=listbox], .vp, [role=slider]');
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === '?') { e.preventDefault(); openShortcuts(); }
    else if (/^[1-9]$/.test(e.key) && S.route === 'explorer') nthLayer(+e.key);
    else if (e.key === 'c' || e.key === 'C') go('compare');
    else if (e.key === 'r' || e.key === 'R') ex.run?.();
    else if (e.key === '[') cycleModel(-1); else if (e.key === ']') cycleModel(1);
  });
}

// ---------- boot
(async () => {
  shell(); theme(); const set = statusBar(); shortcuts();
  const h = await api.health();
  S.health = h.ok ? h.data : null;
  const sv = $('#sb-server');
  sv.dataset.state = h.ok && h.data.ready ? 'ok' : 'bad';
  $('.v', sv).textContent = !h.ok ? 'server offline' : h.data.ready ? `ready · ${h.data.device || 'cpu'}` : 'model not trained';
  S.models = await api.models(S.health);
  if (!S.models.find(m => m.id === S.model && m.status === 'ready')) S.model = S.models.find(m => m.status === 'ready')?.id || 'terrafm';
  renderModels(); emit('models');
  await go(location.hash.slice(1) || 'explorer', false);
  if (!built.explorer) { built.explorer = true; buildExplorer($('#scr-explorer')); }
  if (h.ok && h.data.ready) ex.run();
  else { set('stage', h.ok ? 'train the model first' : 'server offline'); emit('offline', h.ok ? 'No trained model on the server. Run train.bat (or ./run_training.sh), then retry' : 'The TerraScope server is not reachable. Start it with start_local.bat, then retry'); }
})();
