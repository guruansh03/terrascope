// Layers panel: grouped rows with visibility, opacity, selection and reordering
// (pointer drag on the grip, or Alt+Arrow keys). Arrow keys move between rows (roving tabindex).
import { el, icon, $$, $ } from './util.js';
import { S, GROUPS, layerDef, saveLayers, emit, LEGENDS } from './state.js';
import { layerUrl } from './render.js';

export function buildLayerPanel(host, stack) {
  const list = el('div', { class: 'layers', id: 'layer-list' });
  host.append(el('div', { class: 'phead' }, el('h2', {}, 'Layers'), el('span', { class: 'meta' }, 'keys 1–9')), el('div', { class: 'pbody' }, list, el('div', { id: 'layer-detail' })));

  function visibleOrder() { return S.layers.filter(s => stack.available(s.k)); }

  function render() {
    const focusK = document.activeElement?.closest?.('.lrow')?.dataset.k, focusCol = document.activeElement?.dataset?.col;
    const avail = visibleOrder();
    list.replaceChildren(...GROUPS.map(([g, title]) => {
      const rows = avail.filter(s => layerDef(s.k).g === g);
      if (!rows.length) return null;
      return el('section', { class: 'lgroup', 'aria-label': title }, el('h3', {}, title),
        el('ul', { 'data-g': g }, rows.map(s => rowFor(s, avail.indexOf(s) + 1))));
    }).filter(Boolean));
    const first = $(`.lrow[data-k="${focusK || S.active}"] [data-col="${focusCol || 'name'}"]`, list) || $('.lrow [data-col=name]', list);
    $$('[data-col]', list).forEach(b => b.tabIndex = -1); if (first) first.tabIndex = 0;
    if (focusK && first && list.contains(document.activeElement) === false && document.activeElement === document.body) first.focus();
    detail();
  }

  function rowFor(s, n) {
    const d = layerDef(s.k), sel = s.k === S.active, off = d.sensor != null && !S.res?.sensors?.[d.sensor];
    const thumbKey = ['lc', 'flood', 'recon'].includes(s.k) || S.res?.layers.includes(s.k) ? s.k : 'pca';
    return el('li', { class: 'lrow' + (sel ? ' sel' : '') + (s.on ? '' : ' hidden-layer'), 'data-k': s.k },
      el('span', { class: 'grip', 'aria-hidden': 'true', title: 'Drag to reorder' }, icon('grip')),
      el('button', { class: 'eye', 'data-col': 'eye', 'aria-pressed': String(s.on), 'aria-label': `Show ${d.name}`, onclick: () => toggle(s.k) }, icon(s.on ? 'eye' : 'eyeoff')),
      el('button', { class: 'lname', 'data-col': 'name', 'aria-current': sel ? 'true' : null, 'aria-label': `${d.name}${s.on ? '' : ', hidden'}${sel ? ', selected' : ''}`, onclick: () => select(s.k) },
        S.res ? el('img', { class: 'thumb' + (d.px || s.k === 'lc' ? ' px' : ''), alt: '', src: layerUrl(S.res.id, thumbKey), loading: 'lazy' }) : null,
        el('span', { class: 'nm' }, d.name), off ? el('span', { class: 'meta' }, 'not fed') : null),
      n <= 9 ? el('kbd', { 'aria-hidden': 'true' }, n) : null);
  }

  function detail() {
    const box = $('#layer-detail'), s = S.layers.find(x => x.k === S.active), d = layerDef(S.active);
    if (!s || !stack.available(S.active)) { box.replaceChildren(); return; }
    const op = el('input', { type: 'range', min: '0', max: '1', step: '0.05', value: s.op, id: 'layer-opacity', 'aria-label': `${d.name} opacity`,
      oninput: e => { s.op = +e.target.value; out.textContent = Math.round(s.op * 100) + '%'; emit('layers'); saveLayers(); } });
    const out = el('output', { class: 'num', for: 'layer-opacity' }, Math.round(s.op * 100) + '%');
    box.replaceChildren(el('div', { class: 'ldetail' },
      el('h3', {}, d.name), el('p', { class: 'muted' }, d.desc),
      el('div', { class: 'field' }, el('label', { for: 'layer-opacity' }, 'Opacity', out), op),
      legend(S.active)));
  }

  function toggle(k, force) {
    const s = S.layers.find(x => x.k === k); s.on = force ?? !s.on;
    if (s.on) S.active = k;
    saveLayers(); render(); emit('layers');
  }
  function select(k) {
    S.active = k; const s = S.layers.find(x => x.k === k); if (!s.on) s.on = true;
    saveLayers(); render(); emit('layers'); emit('active', k);
  }
  function move(k, dir) {
    const i = S.layers.findIndex(s => s.k === k), g = layerDef(k).g;
    let j = i + dir;
    while (j >= 0 && j < S.layers.length && !stack.available(S.layers[j].k)) j += dir;
    if (j < 0 || j >= S.layers.length || layerDef(S.layers[j].k).g !== g) return false;
    [S.layers[i], S.layers[j]] = [S.layers[j], S.layers[i]]; saveLayers(); render(); emit('layers');
    return true;
  }

  // keyboard: roving tabindex over a 2-column grid of (eye, name); Alt+Up/Down reorders
  list.addEventListener('keydown', e => {
    const b = e.target.closest('[data-col]'); if (!b) return;
    const k = b.closest('.lrow').dataset.k, col = b.dataset.col;
    const rows = $$('.lrow', list), i = rows.findIndex(r => r.dataset.k === k);
    let target = null;
    if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();
      if (move(k, e.key === 'ArrowUp' ? -1 : 1)) { const nb = $(`.lrow[data-k="${k}"] [data-col="${col}"]`, list); $$('[data-col]', list).forEach(x => x.tabIndex = -1); nb.tabIndex = 0; nb.focus(); emit('announce', `${layerDef(k).name} moved`); }
      return;
    }
    if (e.key === 'ArrowDown') target = rows[Math.min(rows.length - 1, i + 1)];
    else if (e.key === 'ArrowUp') target = rows[Math.max(0, i - 1)];
    else if (e.key === 'Home') target = rows[0]; else if (e.key === 'End') target = rows[rows.length - 1];
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { const nb = $(`[data-col="${e.key === 'ArrowLeft' ? 'eye' : 'name'}"]`, rows[i]); $$('[data-col]', list).forEach(x => x.tabIndex = -1); nb.tabIndex = 0; nb.focus(); e.preventDefault(); return; }
    if (!target) return;
    e.preventDefault();
    const nb = $(`[data-col="${col}"]`, target); $$('[data-col]', list).forEach(x => x.tabIndex = -1); nb.tabIndex = 0; nb.focus();
  });

  // pointer drag on the grip, restricted to the row's group
  list.addEventListener('pointerdown', e => {
    const grip = e.target.closest('.grip'); if (!grip) return;
    e.preventDefault();
    const row = grip.closest('.lrow'), ul = row.parentNode, k = row.dataset.k;
    row.classList.add('dragging'); grip.setPointerCapture(e.pointerId);
    const y0 = e.clientY;
    const mv = ev => {
      row.style.transform = `translateY(${ev.clientY - y0}px)`;
      const sibs = [...ul.children].filter(r => r !== row);
      const after = sibs.find(r => ev.clientY < r.getBoundingClientRect().top + r.offsetHeight / 2);
      row.dataset.to = after ? after.dataset.k : '';
    };
    const upf = () => {
      grip.removeEventListener('pointermove', mv); grip.removeEventListener('pointerup', upf); row.classList.remove('dragging'); row.style.transform = '';
      if (row.dataset.to === undefined) return;
      const keys = [...ul.children].map(r => r.dataset.k).filter(x => x !== k), to = row.dataset.to;
      keys.splice(to ? keys.indexOf(to) : keys.length, 0, k);
      const slots = S.layers.map((s, i) => keys.includes(s.k) || s.k === k ? i : -1).filter(i => i >= 0);
      const byK = Object.fromEntries(S.layers.map(s => [s.k, s]));
      slots.forEach((slot, j) => { S.layers[slot] = byK[keys[j]]; });
      saveLayers(); render(); emit('layers');
    };
    grip.addEventListener('pointermove', mv); grip.addEventListener('pointerup', upf);
  });

  return { render, toggle, select, move, nth: n => visibleOrder()[n - 1]?.k };
}

export function legend(k) {
  const r = S.res; if (!r) return null;
  if (k === 'lc') return el('div', { class: 'legend' }, r.classes.map((c, i) => el('div', { class: 'cls', 'data-cls': i, tabindex: '0', 'aria-label': `${c.name}, hover or focus to isolate` }, el('i', { class: 'sw', style: { background: c.color } }), c.name)));
  if (k === 'flood') return el('div', { class: 'legend' }, el('div', {}, el('i', { class: 'sw', style: { background: '#19c3f0' } }), 'Flood probability at or above the threshold'));
  if (k === 'recon') return el('div', { class: 'legend' }, el('p', { class: 'muted' }, S.pix.cloud_mask ? 'Only pixels flagged in the cloud mask were rebuilt' : 'Cloud mask not returned by this server; reveal unavailable'));
  const L = LEGENDS[k]; if (!L) return null;
  return el('div', { class: 'legend' }, el('div', { class: 'ramp' }, el('span', { class: 'num' }, L.a), el('i', { style: { background: L.ramp } }), el('span', { class: 'num' }, L.b)), el('p', { class: 'muted' }, L.label));
}
