// Command palette (Ctrl/Cmd+K) and the keyboard shortcut sheet (?), both native <dialog>s.
import { el, $, $$ } from './util.js';
import { S, LAYERS, PRESETS, emit } from './state.js';

let pal, sheet;
const REASON = { weights_missing: 'weights not downloaded', not_finetuned: 'no fine-tuned heads yet' };

async function commands() {
  const m = await import('./main.js'), { ex } = await import('./explorer.js');
  const list = [];
  m.SCREENS.forEach(([k, n]) => list.push({ g: 'Screen', t: n, run: () => m.go(k) }));
  list.push({ g: 'Action', t: 'Run analysis', k: 'R', run: () => ex.run?.() });
  list.push({ g: 'Action', t: 'Switch theme', run: m.toggleTheme });
  list.push({ g: 'Action', t: 'Fit map to scene', run: () => ex.map?.fit() });
  list.push({ g: 'Action', t: 'Toggle layers panel', run: () => ex.togglePanel?.('left') });
  list.push({ g: 'Action', t: 'Toggle inspector', run: () => ex.togglePanel?.('right') });
  list.push({ g: 'Action', t: 'Keyboard shortcuts', k: '?', run: openShortcuts });
  if (S.res) LAYERS.filter(l => ex.stack?.available(l.k)).forEach(l => {
    const s = S.layers.find(x => x.k === l.k);
    list.push({ g: 'Layer', t: `${s.on ? 'Hide' : 'Show'} ${l.name}`, run: () => { m.go('explorer'); ex.panel.toggle(l.k); } });
  });
  S.models.forEach(md => list.push({ g: 'Model', t: md.name + (md.family === 'unet' ? ' (baseline)' : ''), sub: md.status === 'ready' ? (md.id === S.model ? 'current' : '') : REASON[md.status] || md.status, disabled: md.status !== 'ready', run: () => m.setModel(md.id) }));
  PRESETS.forEach(p => list.push({ g: 'Scenario', t: p.n, sub: 'simulated', run: () => { Object.assign(S.scen, { seed: p.seed, flood: p.flood, cloud: p.cloud, drought: p.drought, source: 'sim', scene_id: null, name: null }); emit('scenario-sync'); m.go('explorer'); emit('scenario', {}); } }));
  S.scenes.forEach(sc => list.push({ g: 'Scene', t: sc.name || sc.id, sub: sc.source, run: () => { Object.assign(S.scen, { source: 'scene', scene_id: sc.id, name: sc.name }); m.go('explorer'); emit('scenario', {}); } }));
  return list;
}

export async function openPalette() {
  if (!pal) {
    pal = el('dialog', { class: 'dialog palette', 'aria-label': 'Command palette' });
    const input = el('input', { type: 'text', id: 'pal-input', role: 'combobox', 'aria-expanded': 'true', 'aria-controls': 'pal-list', 'aria-autocomplete': 'list', 'aria-label': 'Command', placeholder: 'Type a command, layer, scene or model', autocomplete: 'off', spellcheck: 'false' });
    const ul = el('ul', { id: 'pal-list', role: 'listbox', 'aria-label': 'Commands' });
    pal.append(input, ul);
    document.body.append(pal);
    pal.addEventListener('click', e => { if (e.target === pal) pal.close(); });
    let items = [], idx = 0;
    const draw = () => {
      const q = input.value.trim().toLowerCase();
      const shown = items.filter(c => !q || (c.g + ' ' + c.t + ' ' + (c.sub || '')).toLowerCase().includes(q)).slice(0, 40);
      idx = Math.min(idx, Math.max(0, shown.length - 1));
      while (shown[idx]?.disabled && idx < shown.length - 1) idx++;
      ul.replaceChildren(...shown.map((c, i) => el('li', { role: 'option', id: 'pal-' + i, 'aria-selected': String(i === idx), 'aria-disabled': c.disabled ? 'true' : null, class: c.disabled ? 'disabled' : '',
        onpointerdown: e => { e.preventDefault(); if (!c.disabled) { pal.close(); c.run(); } } },
        el('span', { class: 'g' }, c.g), el('span', { class: 't' }, c.t), c.sub ? el('span', { class: 'meta' }, c.sub) : null, c.k ? el('kbd', {}, c.k) : null)));
      if (!shown.length) ul.append(el('li', { class: 'none', role: 'option', 'aria-disabled': 'true', 'aria-selected': 'false' }, 'No matching command'));
      input.setAttribute('aria-activedescendant', shown.length ? 'pal-' + idx : '');
      $('[aria-selected=true]', ul)?.scrollIntoView({ block: 'nearest' });
      pal._shown = shown;
    };
    input.addEventListener('input', () => { idx = 0; draw(); });
    input.addEventListener('keydown', e => {
      const sh = pal._shown || [];
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); const d = e.key === 'ArrowDown' ? 1 : -1; let j = idx; do { j = (j + d + sh.length) % sh.length; } while (sh[j]?.disabled && j !== idx); idx = j; draw(); }
      else if (e.key === 'Enter') { e.preventDefault(); const c = sh[idx]; if (c && !c.disabled) { pal.close(); c.run(); } }
    });
    pal.refresh = async () => { items = await commands(); idx = 0; draw(); };
  }
  // open and focus synchronously so early keystrokes go to the input, not to global shortcuts
  if (!pal.open) { $('#pal-input').value = ''; pal.showModal(); }
  $('#pal-input').focus();
  await pal.refresh();
}

const KEYS = [
  ['Ctrl K / ⌘K', 'Command palette'], ['?', 'This sheet'], ['1 – 9', 'Show or hide layer n (Explorer)'], ['R', 'Run analysis'],
  ['C', 'Compare screen'], ['[  ]', 'Previous / next ready model'], ['Arrows', 'Pan the focused map'], ['+  −  0', 'Zoom in, out, fit'],
  ['Enter', 'Inspect pixel at map centre'], ['Alt ↑ ↓', 'Reorder focused layer'], ['← →', 'Move the compare divider when focused'], ['Esc', 'Close dialogs and popovers'],
];
export function openShortcuts() {
  if (!sheet) {
    sheet = el('dialog', { class: 'dialog shortcuts', 'aria-labelledby': 'sk-h' },
      el('div', { class: 'phead' }, el('h2', { id: 'sk-h' }, 'Keyboard shortcuts'), el('button', { class: 'btn', onclick: () => sheet.close() }, 'Close')),
      el('dl', {}, KEYS.map(([k, d]) => [el('dt', {}, el('kbd', {}, k)), el('dd', {}, d)])));
    sheet.addEventListener('click', e => { if (e.target === sheet) sheet.close(); });
    document.body.append(sheet);
  }
  if (!sheet.open) sheet.showModal();
}
export function closeDialogs() { $$('dialog[open]').forEach(d => d.close()); }
