// Data screen: scene catalogue (rename, delete, open), upload with validation, sample file,
// and the simulator scenario builder.
import { el, icon, fmt, $, toast } from './util.js';
import * as api from './api.js';
import { S, on, emit } from './state.js';
import { buildScenario } from './scenario.js';

const MAX_MB = 20, EXT = ['.npz', '.tif', '.tiff'];

export function buildData(root) {
  root.classList.add('doc');
  const input = el('input', { type: 'file', id: 'file', accept: EXT.join(','), class: 'sr-only', multiple: '', onchange: e => { handle([...e.target.files]); e.target.value = ''; } });
  const drop = el('label', { class: 'drop', for: 'file', id: 'dropzone' }, icon('upload', 16),
    el('span', {}, el('b', {}, 'Choose files'), ' or drop them here'), el('span', { class: 'muted small' }, `.npz or GeoTIFF, up to ${MAX_MB} MB, 32–256 px per side`));
  ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => handle([...e.dataTransfer.files]));
  const scen = el('div', { class: 'scen-full' });
  root.append(el('div', { class: 'docbody' },
    el('header', { class: 'dochead' }, el('h1', {}, 'Data'), el('p', { class: 'muted' }, 'Scenes the server knows about, your uploads, and the simulator')),
    el('section', { class: 'docsec', 'aria-labelledby': 'd-sc' }, el('h2', { id: 'd-sc' }, 'Scenes'), el('div', { id: 'scene-table' }, el('p', { class: 'muted' }, 'Loading'))),
    el('section', { class: 'docsec grid2', 'aria-labelledby': 'd-up' },
      el('div', {}, el('h2', { id: 'd-up' }, 'Upload'), input, drop, el('ul', { class: 'upl', id: 'upload-log', 'aria-live': 'polite' })),
      el('div', {}, el('h2', {}, 'Format'),
        el('p', {}, el('code', {}, '.npz'), ' with ', el('code', {}, 'optical'), ' 6 × H × W reflectance 0–1 (B2, B3, B4, B8, B11, B12), ', el('code', {}, 'sar'), ' 2 × H × W in dB (VV, VH), ', el('code', {}, 'dem'), ' 2 × H × W (elevation m, slope °).'),
        el('p', {}, el('code', {}, '.tif'), ' with 10 bands B2, B3, B4, B8, B11, B12, VV, VH, elevation, slope, or 6 optical bands only. CRS and transform set the bounds and ground sampling distance. GeoTIFF needs a server with the RS upload route.'),
        el('a', { class: 'btn', href: '/api/sample.npz', id: 'sample-link', download: '' }, icon('download'), 'Download a sample .npz'))),
    el('section', { class: 'docsec', 'aria-labelledby': 'd-sim' }, el('h2', { id: 'd-sim' }, 'Simulator ', el('span', { class: 'tag' }, 'simulated')),
      el('p', { class: 'muted' }, 'Physically motivated Sentinel-2, Sentinel-1 and DEM scenes. Changes apply to the Explorer immediately'), scen)));
  buildScenario(scen);
  refresh();
  on('route', r => { if (r === 'data') refresh(); });
}

async function refresh() {
  const box = $('#scene-table');
  const r = await api.scenes();
  S.scenes = r.list || [];
  if (r.unsupported) {
    box.replaceChildren(el('p', { class: 'muted' }, 'This server has no scene catalogue (/api/scenes). Uploads from this session are listed below; they expire when the server restarts'),
      S.uploads.length ? table(S.uploads.map(u => ({ id: u.scene_id, name: u.name, source: 'upload', size: u.size, legacy: true }))) : el('p', { class: 'muted' }, 'No uploads in this session'));
    return;
  }
  if (r.error) return box.replaceChildren(el('p', { class: 'muted' }, r.error));
  box.replaceChildren(S.scenes.length ? table(S.scenes) : el('p', { class: 'muted' }, 'No scenes yet. Upload one below or add demo scenes to data/demo/'));
}

function table(list) {
  return el('div', { class: 'tw' }, el('table', { class: 't scenes' },
    el('thead', {}, el('tr', {}, ['Name', 'Source', 'Acquired', 'Added', 'Size', 'Bounds (W S E N)', 'Actions'].map((h, i) => el('th', { scope: 'col', class: (i >= 4 ? 'n' : '') + (i === 2 || i === 5 ? ' wide' : '') }, i === 6 ? el('span', { class: 'sr-only' }, h) : h)))),
    el('tbody', {}, list.map(sc => {
      const nameCell = el('th', { scope: 'row' }, sc.name || sc.id, el('span', { class: 'meta num sid' }, ' ' + String(sc.id).slice(-6)));
      const b = sc.bounds;
      return el('tr', { 'data-id': sc.id }, nameCell,
        el('td', {}, sc.source === 'simulated' || sc.source === 'sim' ? el('span', { class: 'tag' }, 'simulated') : sc.source || '–'),
        el('td', { class: 'num wide' }, sc.acquisition_date || '–'),
        el('td', { class: 'num' }, sc.created_at ? fmt.date(sc.created_at).replace(' UTC', '') : '–'),
        el('td', { class: 'n num' }, sc.size ? `${sc.size[1]} × ${sc.size[0]}` : '–'),
        el('td', { class: 'n num wide' }, b ? [b.west, b.south, b.east, b.north].map(v => (+v).toFixed(3)).join(' ') : '–'),
        el('td', { class: 'acts' },
          el('button', { class: 'btn', onclick: () => open(sc) }, 'Open'),
          sc.legacy || sc.read_only ? null : el('button', { class: 'btn icon', 'aria-label': `Rename ${sc.name || sc.id}`, onclick: () => rename(sc, nameCell) }, icon('edit')),
          sc.legacy || sc.read_only ? null : delBtn(sc)));
    }))));
}

function open(sc) {
  Object.assign(S.scen, { source: sc.legacy ? 'upload' : 'scene', scene_id: sc.id, name: sc.name || sc.id });
  location.hash = '#explorer'; emit('scenario', {});
}
function rename(sc, cell) {
  const inp = el('input', { type: 'text', value: sc.name || '', 'aria-label': 'Scene name', maxlength: '80' });
  const done = async save => {
    if (save && inp.value.trim() && inp.value !== sc.name) {
      const r = await api.renameScene(sc.id, inp.value.trim());
      if (!r.ok) toast(r.error, 'bad'); else sc.name = inp.value.trim();
    }
    cell.replaceChildren(sc.name || sc.id);
  };
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') done(true); if (e.key === 'Escape') done(false); });
  inp.addEventListener('blur', () => done(true), { once: true });
  cell.replaceChildren(inp); inp.focus(); inp.select();
}
function delBtn(sc) {
  const b = el('button', { class: 'btn icon', 'aria-label': `Delete ${sc.name || sc.id}` }, icon('trash'));
  b.onclick = async () => {
    if (!b.dataset.armed) { b.dataset.armed = '1'; b.classList.add('danger'); b.replaceChildren('Confirm'); b.classList.remove('icon'); setTimeout(() => { if (b.isConnected) { delete b.dataset.armed; b.classList.remove('danger'); b.classList.add('icon'); b.replaceChildren(icon('trash')); } }, 4000); return; }
    const r = await api.deleteScene(sc.id);
    if (!r.ok) return toast(r.error, 'bad');
    refresh();
  };
  return b;
}

async function handle(files) {
  const log = $('#upload-log');
  for (const f of files) {
    const li = el('li', {}, el('span', { class: 'fn' }, f.name), el('span', { class: 'st muted' }, 'checking'));
    log.prepend(li);
    const st = $('.st', li), ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    const fail = m => { st.textContent = m; st.className = 'st bad'; };
    if (!EXT.includes(ext)) { fail(`Rejected: ${ext || 'no extension'} is not .npz, .tif or .tiff`); continue; }
    if (f.size > MAX_MB * 1048576) { fail(`Rejected: ${(f.size / 1048576).toFixed(1)} MB is over ${MAX_MB} MB`); continue; }
    if (f.size < 64) { fail('Rejected: file is empty or truncated'); continue; }
    st.textContent = 'uploading';
    const r = await api.upload(f);
    if (!r.ok) { fail(r.error || 'Upload failed'); continue; }
    const size = r.data.size;
    st.className = 'st'; st.replaceChildren(`Uploaded, ${size ? size[1] + ' × ' + size[0] + ' px' : 'size unknown'}, analysing `, el('a', { href: '#explorer' }, 'View in Explorer'));
    S.uploads.unshift({ scene_id: r.data.scene_id, size, name: f.name });
    Object.assign(S.scen, { source: 'upload', scene_id: r.data.scene_id, name: f.name });
    emit('scenario', {});
    refresh();
  }
}
export { fmt };
