// LayerStack: builds and updates the DOM nodes for one analysis result (PNG images and GPU canvases)
// and runs the data-driven reveals (cloud removal, similarity ripple, class hover-dim).
import { el, animate, loadImage, reduced } from './util.js';
import { LAYERS, layerDef } from './state.js';
import { renderLC, renderFlood, renderReveal, renderRipple, simRank, renderBlend } from './effects.js';

export const layerUrl = (aid, k, sim) => k === 'sim' ? `/api/similarity/${aid}.png?x=${sim[0]}&y=${sim[1]}` : `/api/layer/${aid}/${k}.png`;

export class LayerStack {
  constructor() { this.nodes = new Map(); this.res = null; this.dim = 1; this.hover = -1; }

  bind(res, pix, sim) {
    this.res = res; this.pix = pix; this.sim = sim; this.nodes.clear(); this.imgs = {};
    const [H, W] = res.stats.size;
    this.W = W; this.H = H;
    this.palette = res.classes.map(c => c.color);
  }
  available(k) {
    const d = layerDef(k), r = this.res;
    if (!r) return false;
    if (d.kind === 'lc') return !!this.pix.lc;
    if (d.kind === 'flood') return !!this.pix.flood;
    if (d.kind === 'sim') return r.layers.includes('pca');
    return r.layers.includes(k);
  }
  node(k) {
    if (this.nodes.has(k)) return this.nodes.get(k);
    const d = layerDef(k), aid = this.res.id;
    let n;
    if (d.kind === 'png') n = el('img', { alt: '', src: layerUrl(aid, k), draggable: 'false', class: d.px ? 'px' : '' });
    else if (d.kind === 'lc' || d.kind === 'flood') n = el('canvas', { class: d.kind === 'lc' ? 'px' : '' });
    else if (d.kind === 'recon') n = el('div', {}, el('img', { alt: '', src: layerUrl(aid, 'recon'), draggable: 'false' }), el('canvas', { hidden: true }));
    else if (d.kind === 'sim') n = el('div', {}, el('img', { alt: '', src: layerUrl(aid, 'sim', this.sim), draggable: 'false' }), el('canvas', { hidden: true }));
    n.dataset.k = k;
    this.nodes.set(k, n);
    if (d.kind === 'lc') this.drawLC();
    return n;
  }
  /** Visible layers bottom → top for the given layer state list. */
  ordered(state) {
    return state.filter(s => s.on && this.available(s.k)).reverse().map(s => {
      const n = this.node(s.k); n.style.opacity = s.op; return { key: s.k, node: n };
    });
  }
  drawLC(hover = this.hover, dim = this.dim) {
    const n = this.nodes.get('lc'); if (!n) return;
    renderLC(n, { key: this.res.id + ':lc', data: this.pix.lc, w: this.W, h: this.H, palette: this.palette, hover, dim });
  }
  drawFlood(thr) {
    const n = this.nodes.get('flood'); if (!n) return;
    renderFlood(n, { key: this.res.id + ':fl', data: this.pix.flood, w: this.W, h: this.H, thr });
  }
  /** (h) hover-dim: other classes fade to 40 % in 120 ms. */
  hoverClass(c) {
    if (c === this.hover && (c < 0 ? this.dim === 1 : this.dim <= 0.4)) return;
    this._dimAnim?.();
    const from = this.dim, to = c < 0 ? 1 : 0.4;
    if (c >= 0) this.hover = c;
    this._dimAnim = animate(120, p => { this.dim = from + (to - from) * p; this.drawLC(this.hover, this.dim); }, () => { if (c < 0) this.hover = -1; });
  }

  /** (c) cloud-removal reveal driven by pix.cloud_mask. Returns false when the mask is absent. */
  async reveal() {
    const n = this.nodes.get('recon'), mask = this.pix.cloud_mask;
    if (!n || !mask || !this.res.layers.includes('rgb')) return false;
    const [img, cv] = n.children, key = this.res.id;
    const [input, recon] = await Promise.all([loadImage(layerUrl(key, 'rgb')), loadImage(layerUrl(key, 'recon'))]);
    this._rev?.();
    if (reduced()) return true;
    cv.hidden = false; img.style.opacity = 0; const lift = veilAbove(n);
    this._rev = animate(800, p => renderReveal(cv, { key: key + ':rv', input, recon, mask, w: this.W, h: this.H, prog: p }), () => { img.style.opacity = ''; cv.hidden = true; lift(); });
    return true;
  }

  /** (e) similarity ripple from the picked pixel; pixels light up in order of similarity rank. */
  async pickSim(px) {
    this.sim = px;
    const n = this.nodes.get('sim'); if (!n) return;
    const [img, cv] = n.children, aid = this.res.id, url = layerUrl(aid, 'sim', px);
    const baseKey = this.res.layers.includes('rgb_clean') ? 'rgb_clean' : 'rgb';
    let sim, base;
    try { [sim, base] = await Promise.all([loadImage(url), this.imgs.base ??= loadImage(layerUrl(aid, baseKey))]); } catch { return; }
    if (this.sim !== px) return;
    if (reduced()) { img.src = url; return; }
    const rank = simRank(sim, base, this.W, this.H), at = [(px[0] + 0.5) / this.W, (px[1] + 0.5) / this.H];
    this._rip?.(); cv.hidden = false; img.style.opacity = 0; img.src = url; const lift = veilAbove(n);
    this._rip = animate(800, p => renderRipple(cv, { key: aid + ':sim', sim, base, rank, w: this.W, h: this.H, prog: p, at }), () => { img.style.opacity = ''; cv.hidden = true; lift(); });
  }

  /** Image-like source for blending (PNG layers load from the server, GPU layers are canvases). */
  async source(k, truthPix) {
    const d = layerDef(k) || {};
    if (truthPix) {
      const c = el('canvas');
      if (k === 'lc_true') renderLC(c, { key: this.res.id + ':lct', data: this.pix.lc_true, w: this.W, h: this.H, palette: this.palette });
      else renderFlood(c, { key: this.res.id + ':flt', data: truthFlood(this.pix.flood_true), w: this.W, h: this.H, thr: 0.5 });
      return c;
    }
    if (d.kind === 'lc' || d.kind === 'flood') { this.node(k); return this.nodes.get(k); }
    return loadImage(layerUrl(this.res.id, k, this.sim));
  }
}

/** Fade layers drawn above `node` while a reveal plays on it, so the animated layer is never hidden. */
function veilAbove(node) {
  const above = []; for (let n = node.nextElementSibling; n; n = n.nextElementSibling) if (n.classList.contains('lyr')) { n.classList.add('veiled'); above.push(n); }
  return () => above.forEach(n => n.classList.remove('veiled'));
}

/** Truth flood arrives as 0/1; scale to 0/255 so it thresholds like a probability. */
export function truthFlood(a) { if (!a) return a; if (a._scaled) return a._scaled; const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] > 0 ? 255 : 0; a._scaled = o; return o; }

/** (b) blend two sources into a canvas at mix t. */
export function blendInto(canvas, a, b, t, key) {
  const w = Math.max(a.naturalWidth || a.width, b.naturalWidth || b.width), h = Math.max(a.naturalHeight || a.height, b.naturalHeight || b.height);
  renderBlend(canvas, { keyA: key + ':A', srcA: a, keyB: key + ':B', srcB: b, w, h, t });
}
export { LAYERS };
