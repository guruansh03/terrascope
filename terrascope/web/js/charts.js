// Hand-rolled SVG charts: grouped bars with optional error bars, lines, reliability diagram.
// Colours come from CSS (series 1 = ink, series 2 = grey); every chart has a text summary.
import { svg, el } from './util.js';

const box = (W, H, label) => { const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label, class: 'chart-svg', style: `max-width:${W}px` }); s.append(svg('title', {}, label)); return s; };
const tick = (s, x, y, t, anchor = 'end') => s.append(svg('text', { x, y, 'text-anchor': anchor, class: 'tk' }, t));

/** groups: string[]; series: [{name, vals:number[], err?:number[]}] */
export function bars(groups, series, { label, W = 640, H = 260, max = 1, fmt = v => v.toFixed(2), lowerBetter = false } = {}) {
  const m = { l: 40, r: 8, t: 16, b: 40 }, s = box(W, H, label), iw = W - m.l - m.r, ih = H - m.t - m.b;
  const Y = v => m.t + ih - Math.max(0, Math.min(1, v / max)) * ih, gw = iw / groups.length;
  for (let i = 0; i <= 4; i++) { const v = max * i / 4; s.append(svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: i ? 'grid' : 'axis' })); tick(s, m.l - 6, Y(v) + 4, fmt(v)); }
  const bw = Math.min(28, gw * 0.7 / series.length);
  groups.forEach((g, i) => {
    const x0 = m.l + i * gw + (gw - bw * series.length) / 2;
    series.forEach((se, j) => {
      const v = se.vals[i]; if (v == null) return;
      const x = x0 + j * bw;
      s.append(svg('rect', { x: x + 1, y: Y(v), width: bw - 2, height: Y(0) - Y(v), class: 's' + (j + 1) }, null));
      const e = se.err?.[i];
      if (e) s.append(svg('path', { d: `M${x + bw / 2} ${Y(v - e)}V${Y(v + e)}M${x + bw / 2 - 4} ${Y(v - e)}h8M${x + bw / 2 - 4} ${Y(v + e)}h8`, class: 'err' }));
    });
    const words = g.split(' ');
    tick(s, m.l + i * gw + gw / 2, H - m.b + 16, words.slice(0, 2).join(' '), 'middle');
    if (words.length > 2) tick(s, m.l + i * gw + gw / 2, H - m.b + 30, words.slice(2).join(' '), 'middle');
  });
  return wrap(s, series, lowerBetter);
}

/** series: [{name, pts:[[x,y]...]}] */
export function lines(series, { label, xl, W = 640, H = 240, ymin = 0, ymax, fmt = v => v.toFixed(2) } = {}) {
  const m = { l: 44, r: 12, t: 12, b: 36 }, s = box(W, H, label);
  const xs = series.flatMap(x => x.pts.map(p => p[0])), ys = series.flatMap(x => x.pts.map(p => p[1]));
  if (!xs.length) return el('p', { class: 'muted' }, 'No data');
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = ymin ?? Math.min(...ys), y1 = ymax ?? Math.max(...ys);
  const X = v => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0 || 1) * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) { const v = y0 + (y1 - y0) * i / 4; s.append(svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: i ? 'grid' : 'axis' })); tick(s, m.l - 6, Y(v) + 4, fmt(v)); }
  for (let i = 0; i <= 4; i++) { const v = x0 + (x1 - x0) * i / 4; tick(s, X(v), H - m.b + 16, Math.round(v), 'middle'); }
  tick(s, W - m.r, H - 4, xl, 'end');
  series.forEach((se, j) => {
    s.append(svg('path', { d: se.pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(''), class: 'ln s' + (j + 1) }));
    const l = se.pts[se.pts.length - 1];
    s.append(svg('circle', { cx: X(l[0]), cy: Y(l[1]), r: 3, class: 'pt s' + (j + 1) }));
  });
  return wrap(s, series.length > 1 ? series.map(x => ({ name: `${x.name}, final ${fmt(x.pts[x.pts.length - 1][1])}` })) : null);
}

/** Reliability diagram. bins: [{conf, acc, count?, acc_after?}] (field names tolerated). */
export function reliability(bins, { label, W = 320, H = 300 } = {}) {
  const m = { l: 40, r: 8, t: 8, b: 36 }, s = box(W, H, label), iw = W - m.l - m.r, ih = H - m.t - m.b;
  const X = v => m.l + v * iw, Y = v => m.t + ih - v * ih;
  for (let i = 0; i <= 4; i++) { const v = i / 4; s.append(svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: i ? 'grid' : 'axis' })); tick(s, m.l - 6, Y(v) + 4, v.toFixed(2)); tick(s, X(v), H - m.b + 16, v.toFixed(2), 'middle'); }
  tick(s, W - m.r, H - 4, 'confidence', 'end');
  s.append(svg('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), class: 'diag' }));
  const get = (b, ...ks) => { for (const k of ks) if (b[k] != null) return b[k]; return null; };
  const pts = bins.map(b => ({ c: get(b, 'conf_before', 'conf', 'confidence', 'mean_conf', 'avg_conf'), a: get(b, 'acc_before', 'acc', 'accuracy'), a2: get(b, 'acc_after', 'accuracy_after', 'acc_calibrated'), c2: get(b, 'conf_after', 'confidence_after') })).filter(p => p.c != null && p.a != null);
  const path = (k, ck, cls) => { const q = pts.filter(p => p[k] != null); if (!q.length) return; s.append(svg('path', { d: q.map((p, i) => (i ? 'L' : 'M') + X(p[ck] ?? p.c).toFixed(1) + ' ' + Y(p[k]).toFixed(1)).join(''), class: 'ln ' + cls })); q.forEach(p => s.append(svg('circle', { cx: X(p[ck] ?? p.c), cy: Y(p[k]), r: 3, class: 'pt ' + cls }))); };
  path('a', 'c', 's2'); path('a2', 'c2', 's1');
  const legend = [{ name: 'Before calibration' }]; if (pts.some(p => p.a2 != null)) legend.unshift({ name: 'After calibration' });
  return wrap(s, legend.length > 1 ? legend : null, false, pts.some(p => p.a2 != null) ? null : ['s2']);
}

function wrap(s, series, lowerBetter, classes) {
  const lg = series ? el('div', { class: 'chart-lg' }, series.map((x, j) => el('span', {}, el('i', { class: (classes?.[j]) || 's' + (j + 1) }), x.name))) : null;
  return el('figure', { class: 'chart' }, s, lg, lowerBetter ? el('figcaption', { class: 'muted small' }, 'Lower is better') : null);
}
