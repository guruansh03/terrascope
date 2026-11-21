// Model screen: TerraFM architecture, training recipe, limitations, and model cards from /api/models.
import { el, fmt } from './util.js';
import * as api from './api.js';
import { S, on } from './state.js';

const STATUS = { ready: 'ready', weights_missing: 'weights not downloaded', not_finetuned: 'no fine-tuned heads yet' };

function diagram() {
  const box = (x, y, w, h, title, lines, cls = '') => `<g class="${cls}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4"/><text x="${x + 12}" y="${y + 22}" class="t">${title}</text>${lines.map((l, i) => `<text x="${x + 12}" y="${y + 42 + i * 16}" class="s">${l}</text>`).join('')}</g>`;
  const arrow = (x1, y1, x2, y2) => `<path d="M${x1} ${y1}H${x2 - 6}" class="ar"/><path d="M${x2 - 6} ${y2 - 4}L${x2} ${y2}L${x2 - 6} ${y2 + 4}" class="ah"/>`;
  return `<svg class="flow" viewBox="0 0 980 300" role="img" aria-labelledby="arch-t arch-d"><title id="arch-t">TerraFM architecture</title>
  <desc id="arch-d">Optical, radar and terrain inputs go to per-sensor patch embedders, then a shared four-block transformer encoder, then a cross-modal decoder that rebuilds all channels and a task decoder for land cover, flood and vegetation.</desc>
  ${[['Optical', '6 bands', 24], ['SAR', 'VV, VH', 116], ['DEM', 'elevation, slope', 208]].map(([n, s, y]) => box(0, y, 140, 64, n, [s]) + arrow(140, y + 32, 180, y + 32)).join('')}
  ${box(180, 24, 170, 248, 'Patch embedders', ['one per sensor, 4 × 4', '', 'absent sensor is left', 'out of the mean', '', '+ sensor tags', '+ 2-D sin-cos position'])}
  ${arrow(350, 148, 390, 148)}
  ${box(390, 24, 190, 248, 'Transformer encoder', ['4 blocks, 96 dim, 4 heads', 'shared by all sensors'], 'ink')}
  ${[0, 1, 2, 3].map(i => `<rect x="406" y="${110 + i * 36}" width="158" height="26" rx="4" class="blk"/>`).join('')}
  ${arrow(580, 92, 620, 92)}${arrow(580, 208, 620, 208)}
  ${box(620, 24, 190, 120, 'Cross-modal decoder', ['rebuilds all 10 channels', 'incl. cloud-free optical', 'pretraining objective'])}
  ${box(620, 156, 190, 116, 'Task decoder', ['upsample × 4 + pixel skip', 'fine-tuned on few labels'])}
  ${arrow(810, 84, 840, 84)}${arrow(810, 214, 840, 214)}
  <text x="848" y="80" class="t">Cloud-free image</text><text x="848" y="98" class="s">embeddings</text>
  <text x="848" y="196" class="t">Land cover</text><text x="848" y="214" class="t">Flood extent</text><text x="848" y="232" class="t">Vegetation health</text></svg>`;
}

export function buildModelPage(root) {
  root.classList.add('doc');
  const cards = el('div', { id: 'model-cards' }, el('p', { class: 'muted' }, 'Loading'));
  root.append(el('div', { class: 'docbody' },
    el('header', { class: 'dochead' }, el('h1', {}, 'Model'), el('p', { class: 'muted' }, 'A small multimodal vision transformer that learns how optical, radar and terrain relate before it sees a label')),
    el('section', { class: 'docsec', 'aria-labelledby': 'm-arch' }, el('h2', { id: 'm-arch' }, 'TerraFM architecture'), el('div', { class: 'flowwrap', html: diagram() })),
    el('section', { class: 'docsec grid2', 'aria-label': 'Training' },
      el('div', {}, el('h2', {}, 'Pretraining without labels'), el('ul', { class: 'steps' },
        el('li', {}, el('b', {}, 'Mask. '), '40–80% of patches are hidden from the encoder'),
        el('li', {}, el('b', {}, 'Drop sensors. '), 'Whole sensors disappear at random, so each is predicted from the others'),
        el('li', {}, el('b', {}, 'Add clouds. '), 'Optical inputs carry clouds and shadows while the target is the clean image'))),
      el('div', {}, el('h2', {}, 'Fine-tuning with few labels'), el('ul', { class: 'steps' },
        el('li', {}, el('b', {}, 'Three heads, one encoder. '), 'Land cover, flood extent and vegetation health train together'),
        el('li', {}, el('b', {}, 'Sensor dropout continues, '), 'so every sensor subset works at inference'),
        el('li', {}, el('b', {}, 'Reconstruction stays on '), 'as an auxiliary loss, so the cloud-free output remains')))),
    el('section', { class: 'docsec', 'aria-labelledby': 'm-cards' }, el('h2', { id: 'm-cards' }, 'Models on this server'), cards),
    el('section', { class: 'docsec', 'aria-labelledby': 'm-lim' }, el('h2', { id: 'm-lim' }, 'Limitations'), el('ul', { class: 'steps' },
      el('li', {}, 'TerraFM is trained and scored on simulated scenes. Real Sentinel results exist only after train.bat and evaluate.bat have run on the PC; see Benchmarks'),
      el('li', {}, 'Flood, drought, urban and composite scores are heuristic indices with fixed weights, not calibrated probabilities of harm'),
      el('li', {}, 'Uncertainty is the normalised entropy of the land-cover softmax. Raw confidence is uncalibrated unless a calibration file is present'),
      el('li', {}, 'Patches are 4 × 4 pixels, so heavily clouded areas are rebuilt in visible 4-pixel blocks and similarity maps work at a quarter of the scene resolution'),
      el('li', {}, 'Foundation models other than TerraFM need downloaded weights and fine-tuned heads before they can run here')))));
  load(cards);
  on('model', () => load(cards));
}

async function load(box) {
  const ms = S.models.length ? S.models : await api.models(S.health);
  const rows = ms.map(m => el('article', { class: 'mcard', 'aria-labelledby': 'mc-' + m.id },
    el('div', { class: 'mc-head' }, el('h3', { id: 'mc-' + m.id }, m.name), el('span', { class: 'meta' }, m.family === 'unet' ? 'supervised baseline (U-Net)' : m.family || ''),
      el('span', { class: 'status ' + (m.status === 'ready' ? 'ok' : 'off') }, STATUS[m.status] || m.status), m.id === S.model ? el('span', { class: 'meta' }, 'selected') : null),
    el('dl', {},
      el('dt', {}, 'Parameters'), el('dd', { class: 'num' }, m.params == null ? 'unknown' : `${fmt.int(m.params)} (${fmt.params(m.params)})`),
      el('dt', {}, 'Sensors'), el('dd', {}, (m.sensors || []).join(', ') || '–'),
      el('dt', {}, 'Tasks'), el('dd', {}, (m.tasks || []).join(', ') || '–'),
      el('dt', {}, 'Weights'), el('dd', { class: 'num' }, m.weights_file || 'not reported'),
      el('dt', {}, 'License'), el('dd', {}, m.license || 'not reported'),
      el('dt', {}, 'Source'), el('dd', { class: 'num' }, m.source_url || 'not reported'))));
  box.replaceChildren(...(api.caps.models === false ? [el('p', { class: 'muted' }, 'This server does not list models (/api/models); only TerraFM is available')] : []), el('div', { class: 'mcards' }, rows));
}
