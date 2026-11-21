// Runs analyses. (a) The stage rail advances only when the server reports a real
// stage over SSE; with an older server a plain indeterminate bar is shown instead.
import { el, b64u8, fmt, toast } from './util.js';
import * as api from './api.js';
import { S, emit, STAGES } from './state.js';

const LABEL = { queued: 'Queued', preparing: 'Preparing scene', encoding: 'Encoding', decoding: 'Decoding', rendering: 'Rendering layers', done: 'Done' };

export class Progress {
  // A stage rail (not a spatial scan): the bar and the stage list advance only on server stage events.
  constructor(world, host) {
    this.node = el('div', { class: 'rail', hidden: true, 'aria-hidden': 'true' }, el('i'));
    this.card = el('div', { class: 'card stages', hidden: true, role: 'status', 'aria-live': 'polite' });
    this.bar = el('div', { class: 'indet', hidden: true, role: 'progressbar', 'aria-label': 'Analysis running' }, el('i'));
    host.append(this.node, this.card, this.bar);
  }
  start() { this.seen = []; this.set(0); this.node.hidden = false; this.card.hidden = true; this.bar.hidden = true; this.node.classList.remove('out'); }
  set(f) { this.node.style.setProperty('--f', f); }
  stage(name, t) {
    if (name === 'sync') { this.node.hidden = true; this.bar.hidden = false; emit('stage', 'running'); return; }
    const i = STAGES.indexOf(name);
    if (i < 0) return;
    this.seen.push([name, t]);
    this.set(i / (STAGES.length - 1));
    this.card.hidden = false;
    this.card.replaceChildren(el('ol', {}, STAGES.slice(0, -1).map(s => {
      const hit = this.seen.find(x => x[0] === s);
      return el('li', { class: hit ? (s === name ? 'now' : 'past') : '' }, el('span', {}, LABEL[s]), el('span', { class: 'num' }, hit && hit[1] != null ? fmt.ms(hit[1]) : ''));
    })));
    emit('stage', name);
  }
  end(ok) {
    this.bar.hidden = true; this.card.hidden = true;
    if (!this.node.hidden) { this.set(1); this.node.classList.add('out'); setTimeout(() => { this.node.hidden = true; }, 260); }
    emit('stage', ok ? 'done' : 'error');
  }
}

let req = 0, ctl = null;
export function body(scen = S.scen, model = S.model) {
  const b = { source: scen.source, seed: scen.seed, size: scen.size, flood: scen.flood, cloud: scen.cloud, drought: scen.drought, sensors: scen.sensors, model };
  if (scen.scene_id) b.scene_id = scen.scene_id;
  return b;
}
export function decodePix(pix) { const o = {}; for (const [k, v] of Object.entries(pix || {})) if (typeof v === 'string' && v) o[k] = b64u8(v); return o; }

/** Analyse the current scenario with the selected model and publish the result. */
export async function runAnalysis(progress) {
  const id = ++req; ctl?.abort(); ctl = new AbortController();
  S.busy = true; emit('busy', true); progress?.start();
  const r = await api.analyze(body(), (st, t) => { if (id === req) progress?.stage(st, t); }, ctl.signal);
  if (id !== req || r.aborted) return;
  S.busy = false; emit('busy', false); progress?.end(r.ok);
  if (!r.ok) { emit('error', r); toast(r.error || 'Analysis failed', 'bad'); return; }
  S.res = r.data; S.pix = decodePix(r.data.pix);
  S.sim = [r.data.stats.size[1] >> 1, r.data.stats.size[0] >> 1];
  emit('result', r.data);
}

/** One-off analysis used by Compare (does not touch the Explorer result). */
export async function analyzeOnce(scen, model, progress) {
  progress?.start();
  const r = await api.analyze(body(scen, model), (st, t) => progress?.stage(st, t));
  progress?.end(r.ok);
  if (r.ok) r.pix = decodePix(r.data.pix);
  return r;
}
