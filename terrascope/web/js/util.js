// Small DOM, formatting and motion helpers shared by every screen.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const SVGNS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

// Line icons drawn on a 16px grid, stroke 1.5, currentColor.
const ICONS = {
  eye: 'M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8Z M8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z',
  eyeoff: 'M2 2l12 12 M6.3 3.8A6.6 6.6 0 0 1 8 3.5c4 0 6.5 4.5 6.5 4.5a11 11 0 0 1-1.7 2.2 M4.3 4.9A11 11 0 0 0 1.5 8S4 12.5 8 12.5a6.4 6.4 0 0 0 3-.8',
  grip: 'M6 4h.01 M10 4h.01 M6 8h.01 M10 8h.01 M6 12h.01 M10 12h.01',
  left: 'M10 3 5 8l5 5', right: 'M6 3l5 5-5 5', down: 'M3 6l5 5 5-5',
  plus: 'M8 3v10 M3 8h10', minus: 'M3 8h10', fit: 'M2.5 6V2.5H6 M10 2.5h3.5V6 M13.5 10v3.5H10 M6 13.5H2.5V10',
  sun: 'M8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z M8 1.5v1.5 M8 13v1.5 M1.5 8H3 M13 8h1.5 M3.4 3.4l1 1 M11.6 11.6l1 1 M3.4 12.6l1-1 M11.6 4.4l1-1',
  moon: 'M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7Z',
  play: 'M5 3.5v9l7-4.5-7-4.5Z', search: 'M7 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z M14 14l-3.5-3.5',
  upload: 'M8 10.5V2.5 M4.5 6 8 2.5 11.5 6 M2.5 10.5v3h11v-3', download: 'M8 2.5v8 M4.5 7 8 10.5 11.5 7 M2.5 10.5v3h11v-3',
  layers: 'M8 2 1.5 5.5 8 9l6.5-3.5L8 2Z M1.5 8.5 8 12l6.5-3.5', panel: 'M2.5 2.5h11v11h-11z M6 2.5v11',
  x: 'M4 4l8 8 M12 4l-8 8', pin: 'M8 14.5s4.5-4 4.5-7.5a4.5 4.5 0 0 0-9 0c0 3.5 4.5 7.5 4.5 7.5Z M8 8.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z',
  sliders: 'M2.5 4.5h7 M12.5 4.5h1 M2.5 11.5h1 M6.5 11.5h7 M11 3v3 M5 10v3', trash: 'M3 4.5h10 M6 4.5V3h4v1.5 M4.5 4.5l.5 9h6l.5-9',
  edit: 'M10.5 2.5l3 3-8 8h-3v-3l8-8Z', info: 'M8 14.5a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13Z M8 7.5v4 M8 5h.01',
};
export function icon(name, size = 16) {
  const s = svg('svg', { viewBox: '0 0 16 16', width: size, height: size, 'aria-hidden': 'true', class: 'ic', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.5', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
  s.append(svg('path', { d: ICONS[name] || '' }));
  return s;
}

// ---------- formatting (tabular figures are set in CSS via .num)
export const fmt = {
  n: (v, d = 2) => (v == null || !isFinite(v) ? '–' : Number(v).toFixed(d)),
  pct: (v, d = 1) => (v == null || !isFinite(v) ? '–' : (v * 100).toFixed(d) + '%'),
  km2: v => (v == null || !isFinite(v) ? '–' : (v < 10 ? v.toFixed(2) : v.toFixed(1)) + ' km²'),
  params: p => (p == null ? 'unknown' : p >= 1e6 ? (p / 1e6).toFixed(p >= 1e8 ? 0 : 2) + 'M' : (p / 1e3).toFixed(0) + 'k'),
  ms: v => (v == null ? '–' : v >= 1000 ? (v / 1000).toFixed(2) + ' s' : Math.round(v) + ' ms'),
  date: s => { if (!s) return '–'; const d = new Date(s); return isNaN(d) ? String(s) : d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC'; },
  int: v => (v == null ? '–' : Math.round(v).toLocaleString('en-US')),
};

// ---------- motion
const mq = matchMedia('(prefers-reduced-motion: reduce)');
export const reduced = () => mq.matches || document.documentElement.dataset.motion === 'off';
export const ease = t => 1 - Math.pow(1 - t, 3);

/** Animate from 0..1 over ms, calling step(p). Instant when reduced motion. Returns cancel(). */
export function animate(ms, step, done) {
  if (reduced() || ms <= 0) { step(1); done?.(); return () => {}; }
  let raf, t0;
  const f = t => { t0 ??= t; const p = Math.min(1, (t - t0) / ms); step(ease(p)); if (p < 1) raf = requestAnimationFrame(f); else done?.(); };
  raf = requestAnimationFrame(f);
  return () => cancelAnimationFrame(raf);
}

/** 250 ms number tween; the element keeps its final value in dataset for the next tween. */
export function tweenNum(node, to, format = v => v.toFixed(2), ms = 250) {
  if (!node) return;
  const from = parseFloat(node.dataset.v);
  node._cancel?.();
  node.dataset.v = to;
  if (!isFinite(from) || !isFinite(to)) { node.textContent = isFinite(to) ? format(to) : '–'; return; }
  node._cancel = animate(ms, p => { node.textContent = format(from + (to - from) * p); });
}

// ---------- fetch
export async function getJSON(url, opt) {
  let r;
  try { r = await fetch(url, opt); } catch { return { ok: false, status: 0, error: 'Server unreachable' }; }
  let body = null;
  try { body = await r.json(); } catch { /* not json */ }
  if (!r.ok) {
    const d = body?.detail;
    return { ok: false, status: r.status, error: typeof d === 'string' ? d : Array.isArray(d) ? d.map(x => x.msg).join('; ') : `HTTP ${r.status}` };
  }
  return { ok: true, status: r.status, data: body };
}
export const postJSON = (url, body, method = 'POST') => getJSON(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.decoding = 'async'; i.onload = () => res(i); i.onerror = () => rej(new Error('image ' + src)); i.src = src; });
}
export const b64u8 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));

export const store = {
  get(k, d) { try { const v = localStorage.getItem('ts.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('ts.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

/** Roving tabindex over items matched by selector inside root (arrow keys, Home, End). */
export function roving(root, selector, { orientation = 'vertical', onMove } = {}) {
  root.addEventListener('keydown', e => {
    const items = $$(selector, root).filter(x => !x.disabled && x.offsetParent !== null);
    const i = items.indexOf(document.activeElement);
    if (i < 0) return;
    const prev = orientation === 'vertical' ? 'ArrowUp' : 'ArrowLeft', next = orientation === 'vertical' ? 'ArrowDown' : 'ArrowRight';
    let j = null;
    if (e.key === prev && !e.altKey) j = (i - 1 + items.length) % items.length;
    else if (e.key === next && !e.altKey) j = (i + 1) % items.length;
    else if (e.key === 'Home') j = 0; else if (e.key === 'End') j = items.length - 1;
    if (j == null) return;
    e.preventDefault();
    items.forEach((x, k) => x.tabIndex = k === j ? 0 : -1);
    items[j].focus(); onMove?.(items[j]);
  });
}
export function setRoving(items, active) { items.forEach(x => x.tabIndex = x === active ? 0 : -1); }

let toastT;
export function toast(msg, kind = 'info') {
  const t = $('#toast'); if (!t) return;
  t.textContent = msg; t.dataset.kind = kind; t.classList.add('on');
  clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('on'), 5000);
}
