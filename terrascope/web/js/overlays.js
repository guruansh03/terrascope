// Floating map cards for the Explorer: flood threshold scrub (d), cloud-removal reveal (c),
// truth comparison by swipe or GPU blend (b), similarity hint and the tap value card.
import { el, icon, fmt, tweenNum, $, reduced, animate } from './util.js';
import { S, layerDef, emit } from './state.js';
import { floodArea, gsdOf, pixelValues } from './inspector.js';
import { truthFlood, layerUrl } from './render.js';
import { renderLC, renderFlood, BlendGL } from './effects.js';

export function thresholdCard(onInput) {
  const out = el('output', { class: 'num', for: 'thr' }, S.thr.toFixed(2));
  const area = el('span', { class: 'num strong', id: 'thr-area', 'data-v': '' }, '–');
  const inp = el('input', { type: 'range', id: 'thr', min: '0.1', max: '0.9', step: '0.01', value: S.thr, 'aria-describedby': 'thr-area',
    oninput: e => { S.thr = +e.target.value; out.textContent = S.thr.toFixed(2); onInput(); updateArea(); } });
  const card = el('div', { class: 'card thr', id: 'thr-card' },
    el('div', { class: 'kv' }, el('label', { for: 'thr' }, 'Flood threshold  p ≥ ', out), area), inp);
  function updateArea() { const a = floodArea(S.pix, S.thr, gsdOf(S.res)); if (a) tweenNum(area, a.km2, v => v.toFixed(2) + ' km²'); emit('threshold'); }
  return { card, updateArea, sync: () => { inp.value = S.thr; out.textContent = S.thr.toFixed(2); } };
}

export function revealCard(stack) {
  const info = el('span', { class: 'meta num' });
  const btn = el('button', { class: 'btn', id: 'reveal-btn', onclick: () => stack.reveal() }, icon('play'), 'Replay cloud removal');
  const card = el('div', { class: 'card reveal', id: 'reveal-card' }, btn, info);
  return { card, sync() {
    const m = S.pix.cloud_mask;
    btn.disabled = !m || reduced();
    if (!m) info.textContent = 'server sent no cloud mask';
    else { let n = 0; for (const v of m) if (v) n++; info.textContent = `${fmt.pct(n / m.length)} of pixels rebuilt` + (reduced() ? ', motion reduced' : ''); }
  } };
}

/** Truth / input comparison for the active layer: swipe (clip) or blend (WebGL mix with scrub). */
export function truthCompare(map, stack) {
  const tagL = el('div', { class: 'maptag l' }), tagR = el('div', { class: 'maptag r' });
  map.host.append(tagL, tagR);
  const blendCv = el('canvas', { class: 'blendcv' }), blender = new BlendGL(blendCv);
  const tOut = el('output', { class: 'num', for: 'blend-t' }, '');
  const scrub = el('input', { type: 'range', id: 'blend-t', min: '0', max: '1', step: '0.01', value: S.blendT,
    oninput: e => { S.blendT = +e.target.value; drawBlend(); } });
  const playB = el('button', { class: 'btn icon', 'aria-label': 'Play blend from model to comparison', onclick: () => { animate(800, p => { S.blendT = p; scrub.value = p; drawBlend(); }); } }, icon('play'));
  const blendCard = el('div', { class: 'card blend', id: 'blend-card', hidden: true },
    el('div', { class: 'kv' }, el('label', { for: 'blend-t' }, 'Blend'), tOut), el('div', { class: 'row' }, scrub, playB));
  let srcA = null, srcB = null, labels = ['', ''];

  function pairOf(k) {
    const d = layerDef(k), r = S.res;
    if (!r) return null;
    if (d.truth && r.truth) {
      if (d.kind === 'lc' && S.pix.lc_true) return { kind: 'lc_true', labels: ['Model', 'Simulator truth'] };
      if (d.kind === 'flood' && S.pix.flood_true) return { kind: 'flood_true', labels: ['Model', 'Simulator truth'] };
      if (r.layers.includes(d.truth)) return { kind: 'png', key: d.truth, labels: ['Model', 'Simulator truth'] };
    }
    if (d.pair && r.layers.includes(d.pair)) return { kind: 'png', key: d.pair, labels: ['Cloud-free', 'Cloudy input'] };
    return null;
  }
  const bdCache = {};
  /** Shared backdrop under transparent layers so both sides of a comparison sit on the same image. */
  function backdrop(p, side = 'a') {
    if (p?.kind !== 'flood_true') return null;
    const r = S.res, k = r.layers.includes('rgb_clean') ? 'rgb_clean' : 'recon', id = `${r.id}:${k}:${side}`;
    return { key: 'cmpbase', node: bdCache[id] ??= el('img', { alt: '', src: layerUrl(r.id, k) }) };
  }
  function pairNode(p) {
    const r = S.res, [H, W] = r.stats.size, nodes = [];
    if (p.kind === 'lc_true') { const c = el('canvas', { class: 'px' }); renderLC(c, { key: r.id + ':lct', data: S.pix.lc_true, w: W, h: H, palette: r.classes.map(c => c.color) }); nodes.push(c); }
    else if (p.kind === 'flood_true') {
      nodes.push(backdrop(p, 'b').node);
      const c = el('canvas'); renderFlood(c, { key: r.id + ':flt', data: truthFlood(S.pix.flood_true), w: W, h: H, thr: 0.5 }); nodes.push(c);
    } else nodes.push(el('img', { alt: '', src: layerUrl(r.id, p.key) }));
    return nodes.map(n => ({ key: 'cmp', node: n }));
  }
  async function drawBlend() {
    tOut.textContent = `${Math.round((1 - S.blendT) * 100)} / ${Math.round(S.blendT * 100)}`;
    scrub.setAttribute('aria-valuetext', `${Math.round((1 - S.blendT) * 100)}% ${labels[0]}, ${Math.round(S.blendT * 100)}% ${labels[1]}`);
    if (srcA && srcB) blender.draw(S.blendT);
  }

  /** Called after every compose; returns the blend canvas when it replaces the active layer. */
  async function update() {
    const p = S.cmpTruth ? pairOf(S.active) : null;
    const on = !!p && S.layers.find(s => s.k === S.active)?.on;
    labels = p?.labels || ['', ''];
    tagL.hidden = tagR.hidden = !on; tagL.textContent = labels[0]; tagR.textContent = labels[1];
    blendCard.hidden = !(on && S.cmpMode === 'blend');
    map.setCompare(on && S.cmpMode === 'swipe');
    if (on && S.cmpMode === 'swipe') map.setLayers(pairNode(p), map.world2);
    else map.setLayers([], map.world2);
    blendCv.remove();
    if (on && S.cmpMode === 'blend') {
      const act = stack.nodes.get(S.active); if (!act) return;
      act.after(blendCv); blendCv.classList.add('lyr'); blendCv.style.opacity = act.style.opacity;
      srcA = await stack.source(S.active);
      srcB = p.kind === 'png' ? await stack.source(p.key) : await stack.source(p.kind, true);
      blender.set(srcA, srcB); // textures uploaded once per comparison, the scrub only changes a uniform
      act.style.visibility = 'hidden';
      drawBlend();
    } else stack.nodes.forEach(n => { n.style.visibility = ''; });
    emit('compare-state', { available: !!pairOf(S.active), on });
  }
  return { update, blendCard, pairOf, backdrop, redraw: drawBlend };
}

/** Tap-to-inspect value card shown next to the pin on touch screens. */
export function valueCard(map) {
  const card = el('div', { class: 'card valuecard', hidden: true, role: 'dialog', 'aria-label': 'Pixel values' });
  map.host.append(card);
  return {
    show(px) {
      const v = pixelValues(px); if (!v) { card.hidden = true; return; }
      card.replaceChildren(
        el('div', { class: 'vc-head' }, el('span', { class: 'num' }, `x ${v.x}  y ${v.y}`), el('button', { class: 'btn icon', 'aria-label': 'Close', onclick: () => { card.hidden = true; map.setPin(null); } }, icon('x'))),
        v.lc ? el('div', { class: 'kv' }, el('span', { class: 'k' }, el('i', { class: 'sw', style: { background: v.lc.color } }), v.lc.name), el('span', { class: 'v num' }, v.unc != null ? `unc ${v.unc.toFixed(2)}` : '')) : null,
        v.flood != null ? el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Flood p'), el('span', { class: 'v num' }, v.flood.toFixed(2))) : null,
        v.veg != null ? el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Vegetation'), el('span', { class: 'v num' }, v.veg.toFixed(2))) : null);
      card.hidden = false;
      const [x, y] = map.pinScreen(), w = map.host.clientWidth;
      card.style.transform = `translate(${Math.min(w - 196, Math.max(8, x - 90))}px, ${Math.max(8, y - card.offsetHeight - 20)}px)`;
    },
    hide() { card.hidden = true; },
  };
}
export { $ };
