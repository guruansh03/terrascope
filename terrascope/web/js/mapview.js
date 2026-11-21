// Map viewport: stacked layers in a transformed "world", pan/zoom by pointer, wheel, pinch and
// keyboard, an optional swipe-compare overlay with an accessible divider, a scale bar and a pin.
import { el, reduced } from './util.js';

export class MapView {
  constructor(host, { label = 'Map', onHover, onLeave, onPick, onView } = {}) {
    this.host = host; this.size = [128, 128]; this.gsd = 30; this.v = { z: 1, cx: 0.5, cy: 0.5 };
    this.ins = { l: 0, r: 0, t: 0, b: 0 }; this.cb = { onHover, onLeave, onPick, onView }; this.cmpPos = 0.5;
    host.classList.add('map');
    this.vp = el('div', { class: 'vp', tabindex: '0', role: 'application', 'aria-roledescription': 'map', 'aria-label': `${label}. Arrow keys pan, plus and minus zoom, 0 fits the scene` });
    this.world = el('div', { class: 'world' });
    this.cmp = el('div', { class: 'cmpclip', hidden: true }, this.world2 = el('div', { class: 'world' }));
    this.pin = el('div', { class: 'pin', hidden: true, 'aria-hidden': 'true' });
    this.world.append(this.pin);
    this.div = el('div', { class: 'divider', hidden: true, role: 'slider', tabindex: '0', 'aria-label': 'Compare divider position', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '50', 'aria-valuetext': '50% model, 50% comparison' }, el('span'));
    this.scale = el('div', { class: 'scalebar', 'aria-hidden': 'true' }, el('i'), el('span', { class: 'num' }));
    this.ovl = el('div', { class: 'ovl' });
    this.vp.append(this.world, this.cmp);
    host.append(this.vp, this.div, this.scale, this.ovl);
    this.slots = new Map();
    this.#events();
    new ResizeObserver(() => this.apply(false)).observe(host);
  }

  setScene(size, gsd) { const same = size[0] === this.size[0] && size[1] === this.size[1]; this.size = size; this.gsd = gsd || this.gsd; if (!same) this.fit(false); else this.apply(false); }
  setInsets(ins, anim = true) { Object.assign(this.ins, ins); this.apply(anim); }

  /** Ordered list of {key, node} drawn bottom → top. */
  setLayers(list, target = this.world) {
    const keep = new Set(list.map(l => l.node));
    [...target.children].forEach(c => { if (c.classList.contains('lyr') && !keep.has(c)) c.remove(); });
    const ref = target === this.world ? this.pin : null;
    list.forEach(l => { l.node.classList.add('lyr'); if (l.node.nextSibling !== ref || l.node.parentNode !== target) target.insertBefore(l.node, ref); });
  }

  fit(anim = true) { this.v = { z: 1, cx: 0.5, cy: 0.5 }; this.apply(anim); }
  setView(v, anim = false, silent = true) { this.v = { ...v }; this.apply(anim, silent); }
  zoomBy(f, ax, ay, anim = true) {
    const z = Math.min(16, Math.max(1, this.v.z * f));
    if (ax != null) { const [sx, sy] = this.toScene(ax, ay, true); const k = this.v.z / z; this.v.cx = sx + (this.v.cx - sx) * k; this.v.cy = sy + (this.v.cy - sy) * k; }
    this.v.z = z; this.apply(anim);
  }
  pan(dxPx, dyPx, anim = false) { const [W, H] = this.dims(); this.v.cx -= dxPx / (W * this.v.z); this.v.cy -= dyPx / (H * this.v.z); this.apply(anim); }

  dims() {
    const r = this.host.getBoundingClientRect(), w = Math.max(40, r.width - this.ins.l - this.ins.r), h = Math.max(40, r.height - this.ins.t - this.ins.b);
    const [SH, SW] = this.size, k = Math.min(w / SW, h / SH) * 0.96;
    return [SW * k, SH * k, r, w, h];
  }
  apply(anim = false, silent = false) {
    const [W, H, , w, h] = this.dims(), { z } = this.v;
    this.host.style.setProperty('--il', this.ins.l + 'px'); this.host.style.setProperty('--ir', this.ins.r + 'px');
    // keep at least a quarter of the scene on screen
    this.v.cx = Math.min(1.25, Math.max(-0.25, this.v.cx)); this.v.cy = Math.min(1.25, Math.max(-0.25, this.v.cy));
    const tx = this.ins.l + w / 2 - this.v.cx * W * z, ty = this.ins.t + h / 2 - this.v.cy * H * z;
    const t = `translate(${tx.toFixed(2)}px,${ty.toFixed(2)}px) scale(${z.toFixed(4)})`;
    for (const wd of [this.world, this.world2]) {
      wd.style.width = W + 'px'; wd.style.height = H + 'px';
      wd.classList.toggle('anim', anim && !reduced()); wd.style.transform = t; wd.style.setProperty('--iz', 1 / z);
      wd.classList.toggle('px', z * W / this.size[1] >= 6);
    }
    this.#scalebar(W * z / this.size[1]);
    if (!silent) this.cb.onView?.(this.v);
  }
  #scalebar(screenPxPerPx) {
    const mPerPx = this.gsd / screenPxPerPx, steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
    const m = steps.filter(s => s / mPerPx <= 96).pop() || steps[0];
    this.scale.firstChild.style.transform = `scaleX(${(m / mPerPx / 96).toFixed(4)})`;
    this.scale.lastChild.textContent = m >= 1000 ? m / 1000 + ' km' : m + ' m';
  }

  /** Client point → scene fractional coords (0..1), or integer pixel when !frac. */
  toScene(cx, cy, frac = false) {
    const r = this.world.getBoundingClientRect(), fx = (cx - r.left) / r.width, fy = (cy - r.top) / r.height;
    if (frac) return [fx, fy];
    const [H, W] = this.size;
    if (fx < 0 || fy < 0 || fx >= 1 || fy >= 1) return null;
    return [Math.floor(fx * W), Math.floor(fy * H)];
  }
  setPin(px) {
    this.pin.hidden = !px; if (!px) return;
    const [H, W] = this.size; this.pin.style.left = (px[0] + 0.5) / W * 100 + '%'; this.pin.style.top = (px[1] + 0.5) / H * 100 + '%';
  }
  pinScreen() { const r = this.pin.getBoundingClientRect(), h = this.host.getBoundingClientRect(); return [r.left + r.width / 2 - h.left, r.top + r.height / 2 - h.top]; }

  setCompare(on, pos = this.cmpPos) {
    this.cmp.hidden = this.div.hidden = !on; if (on) this.setDivider(pos);
  }
  setDivider(p) {
    this.cmpPos = Math.min(1, Math.max(0, p)); const pc = Math.round(this.cmpPos * 100);
    this.cmp.style.clipPath = `inset(0 0 0 ${this.cmpPos * 100}%)`;
    this.div.style.transform = `translateX(${this.cmpPos * this.host.clientWidth}px)`;
    this.div.setAttribute('aria-valuenow', pc); this.div.setAttribute('aria-valuetext', `${pc}% left side, ${100 - pc}% right side`);
  }

  #events() {
    const vp = this.vp, ptrs = new Map();
    let drag = null, pinch = null;
    vp.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      vp.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), z: this.v.z }; drag = null; }
      else drag = { x: e.clientX, y: e.clientY, moved: false, type: e.pointerType };
    });
    vp.addEventListener('pointermove', e => {
      if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, [e.clientX, e.clientY]);
      if (pinch && ptrs.size === 2) { const [a, b] = [...ptrs.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]); this.v.z = Math.min(16, Math.max(1, pinch.z * d / pinch.d)); this.apply(); return; }
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (!drag.moved && Math.hypot(dx, dy) > 4) drag.moved = true;
        if (drag.moved) { this.pan(dx, dy); drag.x = e.clientX; drag.y = e.clientY; vp.classList.add('grabbing'); }
      }
      if (e.pointerType === 'mouse' && !drag?.moved) { const p = this.toScene(e.clientX, e.clientY); p ? this.cb.onHover?.(p) : this.cb.onLeave?.(); }
    });
    const up = e => {
      ptrs.delete(e.pointerId); vp.classList.remove('grabbing');
      if (pinch) { if (ptrs.size < 2) pinch = null; drag = null; return; }
      if (drag && !drag.moved && e.type === 'pointerup') { const p = this.toScene(e.clientX, e.clientY); if (p) this.cb.onPick?.(p, drag.type); }
      drag = null;
    };
    vp.addEventListener('pointerup', up); vp.addEventListener('pointercancel', up);
    vp.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') this.cb.onLeave?.(); });
    vp.addEventListener('wheel', e => { e.preventDefault(); this.zoomBy(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY, false); }, { passive: false });
    vp.addEventListener('keydown', e => {
      const step = e.shiftKey ? 120 : 40, k = e.key;
      if (k === 'ArrowLeft') this.pan(step, 0, true); else if (k === 'ArrowRight') this.pan(-step, 0, true);
      else if (k === 'ArrowUp') this.pan(0, step, true); else if (k === 'ArrowDown') this.pan(0, -step, true);
      else if (k === '+' || k === '=') this.zoomBy(1.4); else if (k === '-' || k === '_') this.zoomBy(1 / 1.4);
      else if (k === '0') this.fit(); else if (k === 'Enter' || k === ' ') { const r = this.host.getBoundingClientRect(), p = this.toScene(r.left + this.ins.l + (r.width - this.ins.l - this.ins.r) / 2, r.top + this.ins.t + (r.height - this.ins.t - this.ins.b) / 2); if (p) this.cb.onPick?.(p, 'keyboard'); }
      else return;
      e.preventDefault(); e.stopPropagation();
    });
    // divider: pointer drag + arrows
    const d = this.div;
    d.addEventListener('pointerdown', e => { e.stopPropagation(); d.setPointerCapture(e.pointerId); d.dataset.drag = '1'; });
    d.addEventListener('pointermove', e => { if (d.dataset.drag) { const r = this.host.getBoundingClientRect(); this.setDivider((e.clientX - r.left) / r.width); } });
    d.addEventListener('pointerup', () => delete d.dataset.drag);
    d.addEventListener('keydown', e => {
      const s = e.shiftKey ? 0.1 : 0.02, k = e.key;
      if (k === 'ArrowLeft' || k === 'ArrowDown') this.setDivider(this.cmpPos - s);
      else if (k === 'ArrowRight' || k === 'ArrowUp') this.setDivider(this.cmpPos + s);
      else if (k === 'Home') this.setDivider(0); else if (k === 'End') this.setDivider(1); else return;
      e.preventDefault(); e.stopPropagation();
    });
    new ResizeObserver(() => this.setDivider(this.cmpPos)).observe(this.host);
  }
}
