// Benchmarks: simulated results (with seed spread when present), robustness, calibration,
// real-data metrics from the PC and the laptop CPU benchmark. Missing parts get honest empty states.
import { el, fmt, $ } from './util.js';
import * as api from './api.js';
import { on } from './state.js';
import { bars, lines, reliability } from './charts.js';

const f3 = v => (v == null ? '–' : (+v).toFixed(3));
const empty = (title, what) => el('div', { class: 'notrun' }, el('b', {}, title), el('p', { class: 'muted' }, what));
const tbl = (head, rows, cls = '') => el('div', { class: 'tw' }, el('table', { class: 't ' + cls }, el('thead', {}, el('tr', {}, head.map((h, i) => el('th', { scope: 'col', class: i ? 'n' : '' }, h)))),
  el('tbody', {}, rows.map(r => el('tr', {}, r.map((c, i) => i ? el('td', { class: 'n num' }, c) : el('th', { scope: 'row' }, c)))))));
const ci = c => (Array.isArray(c) && c.length === 2 ? `95% CI ${f3(c[0])}–${f3(c[1])}` : c && typeof c === 'object' && 'lo' in c ? `95% CI ${f3(c.lo)}–${f3(c.hi)}` : null);
/** mean ± std (n seeds) and 95% CI when present; plain numbers fall back to a single value. */
const ms = (v, extra = {}) => {
  if (v && typeof v === 'object' && 'mean' in v) {
    const n = v.n ?? extra.n, c = ci(v.ci95_seeds ?? v.ci95 ?? extra.ci95_seeds) || ci(v.bootstrap_ci95 ?? extra.bootstrap_ci95);
    const main = `${f3(v.mean)}${v.std != null ? ' ± ' + f3(v.std) : ''}${n ? ` (n = ${n})` : ''}`;
    return c ? el('span', {}, main, el('br'), el('span', { class: 'meta' }, c)) : main;
  }
  return typeof v === 'number' ? f3(v) : v ?? '–';
};
const SKIP = new Set(['params', 'seeds', 'per_seed', 'ci95_seeds', 'bootstrap_ci95', 'n']);
function calDecision(cal) {
  const d = cal.decision, applied = cal.apply ?? cal.applied;
  if (applied == null && !d) return null;
  const val = d && d.ece_val_before != null ? ` Validation ECE ${f3(d.ece_val_before)} → ${f3(d.ece_val_after)}.` : '';
  const rule = typeof d === 'string' ? d : d?.rule ? `Rule: ${d.rule}${d.split ? ` (decided on ${d.split})` : ''}.` : '';
  return el('p', { class: applied ? 'muted' : 'warn' }, `${applied ? 'Temperature scaling is applied.' : 'Temperature scaling is not applied.'} ${rule}${val} The ECE values below are held-out test outcomes`);
}

export function buildBench(root) {
  root.classList.add('doc');
  const body = el('div', { class: 'docbody', id: 'bench-body' }, el('p', { class: 'muted' }, 'Loading benchmarks'));
  root.append(body);
  load(body);
  on('route', r => { if (r === 'benchmarks' && !body.dataset.ok) load(body); });
}

async function load(body) {
  const r = await api.benchmarks();
  if (!r.ok) { body.replaceChildren(el('h1', {}, 'Benchmarks'), empty('Benchmarks not available', r.status === 404 ? 'artifacts/metrics.json is missing. Run the training pipeline, then terrascope.core.evaluate' : r.error)); return; }
  body.dataset.ok = '1';
  const B = r.data, sim = B.sim || B, cfg = sim.config || B.config || {};
  const abl = sim.ablation || B.ablation, scr = sim.scratch || B.scratch;
  const parts = [el('header', { class: 'dochead' }, el('h1', {}, 'Benchmarks'), el('p', { class: 'muted' }, 'Held-out simulator scenes unless stated otherwise. Simulated results show the method works end to end; they are not claims about real-world accuracy'))];

  // ----- simulated
  const sec = el('section', { class: 'docsec', 'aria-labelledby': 'b-sim' }, el('h2', { id: 'b-sim' }, 'Simulated data ', el('span', { class: 'tag' }, 'simulated')));
  if (abl && scr) {
    const a = abl['All sensors'], s = scr['All sensors'], seeds = sim.seeds || B.seeds;
    const sd = k => { const st = seeds?.std; if (!st) return null; const v = st['finetuned|All sensors']?.[k] ?? st['All sensors']?.[k] ?? st[k]; return typeof v === 'number' ? v : null; };
    const stat = (v, k, label, ref, lower) => el('div', { class: 'stat' }, el('span', { class: 'k' }, label),
      el('span', { class: 'v num' }, f3(v), sd(k) != null ? el('span', { class: 'pm' }, ` ± ${f3(sd(k))}`) : null),
      el('span', { class: 'ref num' }, ref == null ? '' : `${f3(ref)} from scratch${lower ? ', lower is better' : ''}`));
    sec.append(el('p', { class: 'muted' }, `TerraFM, ${fmt.params(cfg.params)} parameters, fine-tuned on ${cfg.labels ?? '?'} labelled scenes after self-supervised pretraining on ${cfg.train_scenes ?? '?'} unlabelled scenes. Compared with the same network trained from scratch on the same labels.` + (seeds?.n > 1 ? ` Spread over ${seeds.n} seeds.` : ' One seed, so no spread is available.')),
      el('div', { class: 'stats' },
        stat(a.miou, 'miou', 'Land-cover mIoU', s.miou), stat(a.flood_iou, 'flood_iou', 'Flood IoU', s.flood_iou),
        stat(a.veg_mae, 'veg_mae', 'Vegetation MAE', s.veg_mae, true),
        el('div', { class: 'stat' }, el('span', { class: 'k' }, 'Reflectance RMSE under cloud'), el('span', { class: 'v num' }, a.cloud_rmse != null ? f3(a.cloud_rmse) : '–'), el('span', { class: 'ref num' }, a.cloud_rmse_input != null ? `${f3(a.cloud_rmse_input)} for the cloudy input` : ''))));
    const sets = Object.keys(abl), errOf = (src, k) => sets.map(x => { const v = seeds?.std?.[`${src === 'ablation' ? 'finetuned' : src}|${x}`]?.[k] ?? seeds?.std?.[src]?.[x]?.[k] ?? seeds?.std?.[x]?.[k]; return typeof v === 'number' ? v : null; });
    sec.append(el('div', { class: 'grid2' },
      el('div', {}, el('h3', {}, 'Land-cover mIoU by sensors available at test time'),
        bars(sets, [{ name: 'TerraFM, pretrained', vals: sets.map(k => abl[k].miou), err: errOf('ablation', 'miou') }, { name: 'Same network from scratch', vals: sets.map(k => scr[k].miou), err: errOf('scratch', 'miou') }],
          { label: `Bar chart of land-cover mIoU for ${sets.length} sensor subsets, pretrained versus scratch. Values are in the table below.` })),
      el('div', {}, el('h3', {}, 'Validation mIoU during fine-tuning'), curve(sim.curves || B.curves))));
    const best = (x, y, lower) => (lower ? x <= y : x >= y) ? el('span', { class: 'best' }, f3(x)) : f3(x);
    sec.append(el('h3', {}, 'Every sensor combination'), el('p', { class: 'muted small' }, 'Each cell: TerraFM / scratch. The better value is set in medium weight'),
      tbl(['Sensors', 'mIoU', 'Pixel accuracy', 'Flood IoU', 'Vegetation MAE'], sets.map(k => {
        const x = abl[k], y = scr[k];
        return [k, el('span', {}, best(x.miou, y.miou), ' / ', f3(y.miou)), el('span', {}, best(x.oa, y.oa), ' / ', f3(y.oa)), el('span', {}, best(x.flood_iou, y.flood_iou), ' / ', f3(y.flood_iou)), el('span', {}, best(x.veg_mae, y.veg_mae, true), ' / ', f3(y.veg_mae))];
      })));
    const hc = sim.heavy_cloud || B.heavy_cloud;
    if (hc) sec.append(el('h3', {}, `Heavily clouded scenes, over 30% cover (n = ${hc.n})`),
      tbl(['Model · input', 'mIoU', 'Flood IoU', 'Vegetation MAE'], [['TerraFM · optical only', hc.finetuned_optical], ['TerraFM · all sensors', hc.finetuned_all], ['Scratch · optical only', hc.scratch_optical], ['Scratch · all sensors', hc.scratch_all]].filter(x => x[1]).map(([n, m]) => [n, f3(m.miou), f3(m.flood_iou), f3(m.veg_mae)])));
    const z = (sim.size128 || B.size128)?.finetuned;
    if (z) sec.append(el('p', { class: 'muted' }, `Trained on 64 × 64 crops and run unchanged on 128 × 128 scenes: mIoU ${f3(z.miou)}, flood IoU ${f3(z.flood_iou)}.`));
  } else sec.append(empty('No simulated results', 'metrics.json has no ablation table'));
  parts.push(sec);

  // ----- robustness
  const rob = B.robustness;
  parts.push(el('section', { class: 'docsec', 'aria-labelledby': 'b-rob' }, el('h2', { id: 'b-rob' }, 'Robustness ', el('span', { class: 'tag' }, 'simulated')),
    rob ? robustness(rob) : empty('Not run yet', 'Run tests/ml/robustness_probe.py to fill this table')));

  // ----- calibration
  const cal = B.calibration;
  parts.push(el('section', { class: 'docsec', 'aria-labelledby': 'b-cal' }, el('h2', { id: 'b-cal' }, 'Calibration'),
    cal ? calDecision(cal) : null,
    cal && cal.bins ? el('div', { class: 'grid2 narrow' }, reliability(cal.bins, { label: `Reliability diagram, ${cal.bins.length} bins. Expected calibration error ${f3(cal.ece_before)} before and ${f3(cal.ece_after)} after calibration.` }),
      el('div', { class: 'rows' }, el('div', { class: 'kv' }, el('span', { class: 'k' }, 'ECE before'), el('span', { class: 'v num' }, f3(cal.ece_before))),
        el('div', { class: 'kv' }, el('span', { class: 'k' }, 'ECE after'), el('span', { class: 'v num' }, f3(cal.ece_after))),
        cal.method ? el('div', { class: 'kv' }, el('span', { class: 'k' }, 'Method'), el('span', { class: 'v' }, cal.method)) : null,
        el('p', { class: 'muted small' }, 'Points on the diagonal are perfectly calibrated')))
      : empty('No calibration file', 'Confidence shown in the Explorer is the raw model confidence')));

  // ----- real data
  parts.push(el('section', { class: 'docsec', 'aria-labelledby': 'b-real' }, el('h2', { id: 'b-real' }, 'Real Sentinel data'),
    B.real ? realTables(B.real) : empty('Not run yet', 'Run train.bat then evaluate.bat on the PC. Results appear here from artifacts/metrics_real.json')));

  // ----- laptop
  parts.push(el('section', { class: 'docsec', 'aria-labelledby': 'b-lap' }, el('h2', { id: 'b-lap' }, 'Laptop CPU benchmark'),
    B.laptop_bench ? genericTable(B.laptop_bench) : empty('Not run yet', 'Run bench_laptop.bat on the demo laptop. Results appear here from artifacts/bench_laptop.json')));
  body.replaceChildren(...parts);
}

function curve(c) {
  if (!c?.finetuned) return el('p', { class: 'muted' }, 'No training curves');
  const va = n => (c[n] || []).filter(d => d.val_miou != null).map(d => [d.step, d.val_miou]);
  return lines([{ name: 'TerraFM, pretrained', pts: va('finetuned') }, { name: 'From scratch', pts: va('scratch') }], { label: 'Validation land-cover mIoU against fine-tuning step for pretrained and scratch models', xl: 'fine-tuning step', ymin: 0, ymax: 1 });
}

function realTables(real) {
  const ds = real.datasets || {};
  const out = [el('p', { class: 'muted small' }, `Generated ${real.generated_at ? fmt.date(real.generated_at) : 'at an unknown time'} on ${real.device || 'an unknown device'}. Mean ± standard deviation over seeds, with 95% confidence intervals when reported.`)];
  for (const [name, d] of Object.entries(ds)) {
    const models = Object.entries(d.models || {});
    const keys = [...new Set(models.flatMap(([, m]) => Object.keys(m).filter(k => !SKIP.has(k))))];
    const nSeeds = m => Array.isArray(m.seeds) ? m.seeds.length : typeof m.seeds === 'number' ? m.seeds : '–';
    out.push(el('h3', {}, pretty(name), el('span', { class: 'meta' }, ` ${d.task || ''}${d.split_sizes ? ' · ' + Object.entries(d.split_sizes).map(([k, v]) => `${k} ${v}`).join(', ') : ''}`)),
      models.length ? tbl(['Model', 'Parameters', 'Seeds', ...keys.map(k => pretty(k))], models.map(([id, m]) => [pretty(id), fmt.params(m.params), nSeeds(m),
        ...keys.map(k => ms(m[k], { n: Array.isArray(m.seeds) ? m.seeds.length : undefined, ci95_seeds: m.ci95_seeds?.[k], bootstrap_ci95: m.bootstrap_ci95?.[k] }))])) : el('p', { class: 'muted' }, 'No models evaluated'));
    if (d.paired) out.push(el('p', { class: 'muted small' }, 'Paired comparison: ', typeof d.paired === 'string' ? d.paired : Object.entries(d.paired).map(([k, v]) => `${pretty(k)}: ${typeof v === 'object' ? Object.entries(v).map(([a, b]) => `${pretty(a)} ${typeof b === 'number' ? f3(b) : Array.isArray(b) ? b.map(f3).join('–') : b}`).join(', ') : typeof v === 'number' ? f3(v) : v}`).join(' · ')));
  }
  if (real.paper || real.sim_paper) out.push(genericTable(real.paper || real.sim_paper));
  return el('div', {}, out);
}

/** Render an unknown-shaped artifact as a table: array of rows, or object of objects. */
function genericTable(obj) {
  let rows = Array.isArray(obj) ? obj : obj.rows || obj.results || obj.models || null;
  if (rows && !Array.isArray(rows)) rows = Object.entries(rows).map(([k, v]) => ({ name: k, ...(typeof v === 'object' ? v : { value: v }) }));
  if (!rows) rows = Object.entries(obj).map(([k, v]) => ({ name: k, ...(v && typeof v === 'object' && !Array.isArray(v) ? v : { value: v }) }));
  const cols = [...new Set(rows.flatMap(r => Object.keys(r)))].filter(c => typeof rows.find(r => r[c] != null)?.[c] !== 'object' || rows.some(r => r[c]?.mean != null));
  const first = cols.find(c => /name|model|id|condition|probe|case/i.test(c)) || cols[0];
  const rest = cols.filter(c => c !== first);
  const cell = v => v == null ? '–' : typeof v === 'number' ? (Number.isInteger(v) ? fmt.int(v) : f3(v)) : typeof v === 'object' ? ms(v) : String(v);
  return tbl([pretty(first), ...rest.map(pretty)], rows.map(r => [typeof r[first] === 'string' ? pretty(r[first]) : cell(r[first]), ...rest.map(c => cell(r[c]))]));
}
export { $ };

const NAMES = { unet_r18_scratch: 'U-Net ResNet-18, scratch (baseline)', unet_r18_imagenet: 'U-Net ResNet-18, ImageNet (baseline)', prithvi_tiny: 'Prithvi-EO-2.0 tiny', prithvi_100m: 'Prithvi-EO-2.0 100M', terramind_tiny: 'TerraMind 1.0 tiny', finetuned: 'TerraFM, fine-tuned', finetuned_v1: 'TerraFM, fine-tuned (v1 normalisation)', finetuned_v2: 'TerraFM, fine-tuned (v2 normalisation)', scratch: 'From scratch', terrafm: 'TerraFM' };
export const pretty = k => NAMES[k] || String(k).replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());

/** Robustness probe: per model, each shift with mIoU, drop against in-distribution and the gate verdict. */
function robustness(rob) {
  const models = Object.entries(rob).filter(([, v]) => v && Array.isArray(v.rows));
  if (!models.length) return genericTable(rob);
  return el('div', {}, models.map(([id, m]) => [
    el('h3', {}, pretty(id), el('span', { class: 'meta' }, ` ${m.split ? 'split ' + m.split + ' · ' : ''}gate: mIoU drop ≤ ${f3(m.gate_miou_drop)} · ${m.all_within_gate ? 'all probes pass' : 'some probes fail'}`)),
    tbl(['Probe', 'mIoU', 'Drop vs in-distribution', 'Flood IoU', 'Vegetation MAE', 'Gate'], m.rows.map(r => [
      r.name, f3(r.miou), r.miou_drop == null ? '–' : f3(r.miou_drop), f3(r.flood_iou), f3(r.veg_mae),
      r.within_gate == null ? '–' : el('span', { class: r.within_gate ? 'ok' : 'warn' }, r.within_gate ? 'pass' : 'fail')]))]));
}
