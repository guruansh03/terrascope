// Explorer screen: full-height map, Layers panel, Inspector, floating map tools, mobile bottom sheet.
import { el, icon, $, $$, toast, reduced } from './util.js';
import { S, on, emit, layerDef, saveLayers } from './state.js';
import { MapView } from './mapview.js';
import { LayerStack } from './render.js';
import { Progress, runAnalysis } from './run.js';
import { buildLayerPanel } from './layerpanel.js';
import { buildInspector, gsdOf } from './inspector.js';
import { buildScenario } from './scenario.js';
import { thresholdCard, revealCard, truthCompare, valueCard } from './overlays.js';

export const ex = {};
const mobile = matchMedia('(max-width: 820px)');

export function buildExplorer(root) {
  const mapHost = el('div', { class: 'maphost' });
  const left = el('aside', { class: 'panel left', id: 'ex-left', 'aria-label': 'Layers' });
  const right = el('aside', { class: 'panel right', id: 'ex-right', 'aria-label': 'Inspector' });
  const scen = el('div', { class: 'panel popover scen', id: 'ex-scen', role: 'dialog', 'aria-label': 'Scenario', hidden: true },
    el('div', { class: 'phead' }, el('h2', {}, 'Scenario'), el('span', { class: 'meta' }, 'simulated scene'),
      el('button', { class: 'btn icon close', 'aria-label': 'Close scenario', onclick: () => openScen(false) }, icon('x'))),
    el('div', { class: 'pbody' }));
  root.append(mapHost, left, right, scen);

  const stack = new LayerStack();
  const map = new MapView(mapHost, {
    label: 'Explorer map',
    onHover: px => { S.cursor = px; ins.cursor(px); emit('cursor', px); hoverDim(px); },
    onLeave: () => { S.cursor = null; ins.cursor(S.pin); emit('cursor', null); if (S.active === 'lc') stack.hoverClass(-1); },
    onPick: (px, type) => pick(px, type),
  });
  const prog = new Progress(map.world, map.host);
  const panel = buildLayerPanel(left, stack);
  const ins = buildInspector(right, { idp: 'ex' });
  buildScenario($('.pbody', scen), { compact: true });
  Object.assign(ex, { map, stack, panel, ins, prog, openScen, run: () => runAnalysis(prog) });

  // ----- floating tools
  const thr = thresholdCard(() => { stack.drawFlood(S.thr); ins.threshold(); });
  const rev = revealCard(stack);
  const cmp = truthCompare(map, stack);
  const vcard = valueCard(map);
  const cmpBtn = el('button', { class: 'btn', id: 'cmp-truth', 'aria-pressed': String(S.cmpTruth), onclick: () => { S.cmpTruth = !S.cmpTruth; compose(); } }, 'Compare');
  const modeSeg = el('div', { class: 'seg', role: 'group', 'aria-label': 'Comparison mode' },
    ['swipe', 'blend'].map(m => el('button', { 'data-mode': m, 'aria-pressed': String(S.cmpMode === m), onclick: () => { S.cmpMode = m; compose(); } }, m === 'swipe' ? 'Swipe' : 'Blend')));
  const tools = el('div', { class: 'maptools tl' },
    el('button', { class: 'btn icon', id: 'toggle-left', 'aria-label': 'Toggle layers panel', 'aria-expanded': 'true', onclick: () => togglePanel('left') }, icon('layers')),
    el('button', { class: 'btn', id: 'scen-btn', 'aria-expanded': 'false', 'aria-controls': 'ex-scen', onclick: () => openScen(scen.hidden) }, icon('sliders'), 'Scenario'),
    cmpBtn, modeSeg);
  modeSeg.hidden = true; cmpBtn.disabled = true;
  const tr = el('div', { class: 'maptools tr' },
    el('span', { class: 'tag', id: 'src-badge', hidden: true, title: 'Data source of the current analysis' }, 'simulated'),
    el('a', { class: 'btn disabled', id: 'export', 'aria-disabled': 'true', download: '', 'aria-label': 'Export .npz' }, icon('download'), el('span', {}, 'Export .npz')),
    el('div', { class: 'seg', role: 'group', 'aria-label': 'Zoom' },
      el('button', { 'aria-label': 'Zoom in', onclick: () => map.zoomBy(1.5) }, icon('plus')),
      el('button', { 'aria-label': 'Zoom out', onclick: () => map.zoomBy(1 / 1.5) }, icon('minus')),
      el('button', { 'aria-label': 'Fit scene', onclick: () => map.fit() }, icon('fit'))),
    el('button', { class: 'btn icon', id: 'toggle-right', 'aria-label': 'Toggle inspector', 'aria-expanded': 'true', onclick: () => togglePanel('right') }, icon('panel')));
  // controls docked in a strip under the map so they never cover the scene
  const dock = el('div', { class: 'dock', id: 'ex-dock', hidden: true, role: 'group', 'aria-label': 'Layer controls' }, thr.card, rev.card, cmp.blendCard,
    el('div', { class: 'card hint', id: 'sim-hint', hidden: true }, 'Click the map to choose a reference point'));
  const msg = el('div', { class: 'card mapmsg', id: 'map-msg', hidden: true, role: 'alert' });
  map.ovl.append(tools, tr, msg); root.append(dock);
  function message(text, stale) {
    msg.hidden = !text; msg.classList.toggle('stale', !!stale); if (!text) return;
    msg.replaceChildren(el('p', {}, text), el('button', { class: 'btn', id: 'retry-btn', onclick: () => { if (!S.health) location.reload(); else runAnalysis(prog); } }, 'Retry'));
  }
  ex.message = message;

  // ----- mobile bottom sheet
  const sheetTabs = el('div', { class: 'sheet-tabs', role: 'tablist', 'aria-label': 'Panels' },
    [['ex-left', 'Layers'], ['ex-right', 'Inspect'], ['ex-scen', 'Scenario']].map(([id, n], i) =>
      el('button', { role: 'tab', 'aria-controls': id, 'aria-selected': 'false', tabindex: i ? '-1' : '0', onclick: e => sheet(id, e.currentTarget) }, n)));
  const sheetEl = el('div', { class: 'sheet', id: 'ex-sheet' }, el('button', { class: 'handle', 'aria-label': 'Collapse panel', onclick: () => sheet(null) }, el('i')), sheetTabs);
  root.append(sheetEl);
  function sheet(id, btn) {
    const open = id && root.dataset.sheet !== id;
    root.dataset.sheet = open ? id : '';
    $$('[role=tab]', sheetTabs).forEach(b => { b.setAttribute('aria-selected', String(b.getAttribute('aria-controls') === root.dataset.sheet)); b.tabIndex = b === btn || (!btn && b === sheetTabs.firstChild) ? 0 : -1; });
    if (open) scen.hidden = id !== 'ex-scen';
  }
  sheetTabs.addEventListener('keydown', e => {
    const tabs = $$('[role=tab]', sheetTabs), i = tabs.indexOf(document.activeElement);
    if (i < 0 || !['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
    const j = (i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length; tabs.forEach(t => t.tabIndex = -1); tabs[j].tabIndex = 0; tabs[j].focus(); e.preventDefault();
  });

  function openScen(open) {
    if (mobile.matches) return sheet(open ? 'ex-scen' : null);
    scen.hidden = !open; $('#scen-btn').setAttribute('aria-expanded', String(open));
    if (open) $('input', scen)?.focus();
  }
  document.addEventListener('pointerdown', e => { if (!mobile.matches && !scen.hidden && !scen.contains(e.target) && !e.target.closest('#scen-btn')) openScen(false); });
  scen.addEventListener('keydown', e => { if (e.key === 'Escape') { openScen(false); $('#scen-btn').focus(); } });

  function togglePanel(side) {
    const k = side === 'left' ? 'leftClosed' : 'rightClosed';
    root.dataset[k] = root.dataset[k] === '1' ? '' : '1';
    $(`#toggle-${side}`).setAttribute('aria-expanded', String(root.dataset[k] !== '1'));
    insets(true);
  }
  ex.togglePanel = togglePanel;
  function insets(anim) {
    if (mobile.matches) return map.setInsets({ l: 0, r: 0, t: 48, b: 112 }, anim);
    map.setInsets({ l: root.dataset.leftClosed === '1' ? 0 : 280, r: root.dataset.rightClosed === '1' ? 0 : 320, t: 48, b: 56 }, anim);
  }
  mobile.addEventListener('change', () => insets(false)); insets(false);

  // ----- composition
  function compose() {
    if (!S.res) return;
    const p = cmp.pairOf(S.active), actOn = S.layers.find(s => s.k === S.active)?.on;
    let list = stack.ordered(S.layers);
    if (S.cmpTruth && p && actOn) { // comparison shows the same layer type on both sides over the same backdrop
      list = list.filter(l => l.key === S.active); const bd = cmp.backdrop(p); if (bd) list.unshift(bd);
    }
    map.setLayers(list);
    stack.drawFlood(S.thr);
    dock.hidden = false;
    const fl = S.layers.find(s => s.k === 'flood');
    thr.card.hidden = !(fl?.on && stack.available('flood'));
    rev.card.hidden = !(S.active === 'recon' && stack.available('recon')); rev.sync();
    $('#sim-hint').hidden = S.active !== 'sim';
    cmpBtn.setAttribute('aria-pressed', String(S.cmpTruth));
    $$('[data-mode]', modeSeg).forEach(b => b.setAttribute('aria-pressed', String(S.cmpMode === b.dataset.mode)));
    cmpBtn.disabled = !p; cmpBtn.title = p ? `Compare with ${p.labels[1].toLowerCase()}` : 'No comparison for this layer';
    cmpBtn.textContent = p ? `Compare with ${p.labels[1] === 'Cloudy input' ? 'input' : 'truth'}` : 'Compare';
    modeSeg.hidden = !(p && S.cmpTruth);
    cmp.update();
    if (S.active === 'sim') map.setPin(S.sim);
  }
  ex.compose = compose;
  function hoverDim(px) {
    if (S.active !== 'lc' || !S.pix.lc || !S.layers.find(s => s.k === 'lc').on) return;
    stack.hoverClass(S.pix.lc[px[1] * S.res.stats.size[1] + px[0]]);
  }
  function pick(px, type) {
    if (!S.res) return;
    if (S.active === 'sim' && S.layers.find(s => s.k === 'sim').on) { S.sim = px; map.setPin(px); stack.pickSim(px); emit('announce', `Similarity reference set to ${px[0]}, ${px[1]}`); return; }
    S.pin = px; map.setPin(px); ins.cursor(px);
    if (type === 'touch' || mobile.matches) { if (root.dataset.sheet) sheet(null); vcard.show(px); } else vcard.hide();
  }
  // legend / composition hover → isolate class (h)
  const iso = e => { const c = e.target.closest?.('[data-cls]'); if (c && S.pix.lc) { if (S.active !== 'lc') return; stack.hoverClass(+c.dataset.cls); } };
  const unIso = e => { if (e.target.closest?.('[data-cls]')) stack.hoverClass(-1); };
  for (const p of [left, right]) { p.addEventListener('pointerover', iso); p.addEventListener('pointerout', unIso); p.addEventListener('focusin', iso); p.addEventListener('focusout', unIso); }

  on('layers', compose);
  on('active', k => { vcard.hide(); if (k === 'sim' && S.res) map.setPin(S.sim); else map.setPin(S.pin); });
  on('result', r => {
    stack.bind(r, S.pix, S.sim);
    map.setScene(r.stats.size, gsdOf(r));
    S.pin = null; map.setPin(null); vcard.hide();
    if (!stack.available(S.active)) { S.active = 'recon'; saveLayers(); }
    panel.render(); ins.result(); compose(); thr.updateArea();
    const ex1 = $('#export'); ex1.href = `/api/export/${r.id}.npz`; ex1.classList.remove('disabled'); ex1.removeAttribute('aria-disabled');
    const src = r.provenance?.data_source || (r.meta?.source === 'sim' ? 'simulated' : r.meta?.source);
    $('#src-badge').hidden = src !== 'simulated';
    message(null);
    const recon = S.layers.find(s => s.k === 'recon');
    if (recon?.on && S.pix.cloud_mask && !reduced()) stack.reveal();
  });
  on('scenario', ({ debounced } = {}) => { clearTimeout(ex._t); ex._t = setTimeout(() => runAnalysis(prog), debounced ? 250 : 0); });
  on('busy', b => mapHost.classList.toggle('busy', b));
  on('error', r => message(S.res ? `Last run failed: ${r.error || 'analysis failed'}. The map still shows the previous result` : r.error || 'Analysis failed', !!S.res));
  on('offline', t => message(t));
}

export function nthLayer(n) { const k = ex.panel.nth(n); if (!k) return; ex.panel.toggle(k); emit('announce', `${layerDef(k).name} ${S.layers.find(s => s.k === k).on ? 'shown' : 'hidden'}`); }
export { toast };
