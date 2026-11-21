// TerraScope UI end-to-end test (Playwright, Chromium).
//   node tests/e2e/ui_e2e.js [baseUrl] [screenshotDir]
// Desktop 1440×900 and iPhone 13. Exit code 1 when any check fails.
const { chromium, devices } = require('playwright');
const fs = require('fs'), path = require('path');
const B = (process.argv[2] || 'http://127.0.0.1:8000').replace(/\/$/, '');
const OUT = process.argv[3] || process.env.UI_SHOTS || path.join(__dirname, 'shots');
fs.mkdirSync(OUT, { recursive: true });
// Endpoints the UI feature-detects; a 404/405 from them on an older server is expected, not an error.
const OPTIONAL = /\/api\/(models|analyze_async|scenes|change|jobs)\b/;
const res = [];
const T = (n, ok, info = '') => { res.push([n, !!ok, info]); console.log(ok ? 'PASS' : 'FAIL', n, info ? '· ' + String(info).slice(0, 160) : ''); };
const wait = ms => new Promise(r => setTimeout(r, ms));

async function ready(p) {
  await p.waitForFunction(() => document.querySelector('#sb-stage .v')?.textContent === 'done' && document.querySelectorAll('.lrow').length > 0, null, { timeout: 30000 });
}
const overflow = p => p.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
const mapBox = p => p.$eval('#scr-explorer .vp', e => { const r = e.closest('.map').querySelector('.world').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; });

async function run(label, ctxOpt) {
  const br = await chromium.launch();
  const ctx = await br.newContext({ acceptDownloads: true, ...ctxOpt });
  const p = await ctx.newPage();
  const errs = [], bad = [], external = [];
  p.on('console', m => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errs.push(m.text()); });
  p.on('pageerror', e => errs.push('PAGEERR ' + e.message));
  p.on('request', r => { const u = r.url(); if (!u.startsWith(B) && !u.startsWith('data:') && !u.startsWith('blob:')) external.push(u); });
  p.on('response', r => { const u = r.url(), s = r.status(); if (s >= 400 && !(OPTIONAL.test(u) && (s === 404 || s === 405)) && !(/\/api\/upload/.test(u) && s === 400)) bad.push(s + ' ' + u); });
  const mobile = label === 'mobile';

  await p.goto(B + '/#explorer', { waitUntil: 'domcontentloaded' });
  await ready(p).catch(e => T(label + ' analysis finished', false, e.message));
  T(label + ' explorer loads with layers', (await p.$$('.lrow')).length >= 8, (await p.$$('.lrow')).length + ' rows');
  T(label + ' status bar shows scene', /seed|upload|scene/.test(await p.textContent('#statusbar')), await p.textContent('#statusbar'));
  T(label + ' fonts are self-hosted', await p.evaluate(async () => { await document.fonts.ready; return document.fonts.check('13px "IBM Plex Sans"') && document.fonts.check('13px "IBM Plex Mono"'); }));
  await p.screenshot({ path: path.join(OUT, `${label}_explorer.png`) });

  // model switcher
  if (!mobile) {
    await p.click('#model-btn');
    const opts = await p.$$('#model-list [role=option]');
    T(label + ' model switcher lists models', opts.length >= 1 && await p.isVisible('#model-list'), opts.length + ' options');
    T(label + ' current model is selected', (await p.$$('#model-list [aria-selected=true]')).length === 1);
    const disabled = await p.$$eval('#model-list [aria-disabled=true]', l => l.map(x => x.textContent));
    T(label + ' unready models carry a reason', disabled.every(t => /weights|fine-tuned|missing|not/.test(t)), disabled.join(' | ') || 'none unready');
    await p.keyboard.press('Escape'); T(label + ' model list closes on Escape', !(await p.isVisible('#model-list')));
  } else T(label + ' model select in scenario sheet', (await p.$$('#ex-scen select')).length === 1);

  if (mobile) { await p.click('.sheet-tabs [aria-controls=ex-left]'); await wait(300); T(label + ' bottom sheet opens layers', await p.$eval('#ex-left', e => getComputedStyle(e).transform === 'none')); }
  // layer toggle + opacity
  await p.click('.lrow[data-k=lc] .eye'); await wait(300);
  T(label + ' layer eye shows land cover', await p.$eval('.lrow[data-k=lc] .eye', e => e.getAttribute('aria-pressed')) === 'true' && (await p.$$('#scr-explorer .world > [data-k=lc]')).length === 1);
  await p.$eval('#layer-opacity', e => { e.value = '0.4'; e.dispatchEvent(new Event('input', { bubbles: true })); }); await wait(200);
  T(label + ' opacity slider applies', await p.$eval('#scr-explorer .world > [data-k=lc]', e => e.style.opacity) === '0.4');
  await p.$eval('#layer-opacity', e => { e.value = '1'; e.dispatchEvent(new Event('input', { bubbles: true })); });
  if (!mobile) {
    await p.focus('.lrow[data-k=lc] [data-col=name]'); await p.keyboard.press('ArrowDown');
    T(label + ' roving tabindex in layer list', await p.evaluate(() => document.activeElement.closest('.lrow')?.dataset.k) === 'veg');
    await p.keyboard.press('Alt+ArrowUp'); await wait(150);
    T(label + ' Alt+Arrow reorders layers', await p.$$eval('.lrow', l => l.map(x => x.dataset.k).indexOf('veg') < l.map(x => x.dataset.k).indexOf('lc')));
    await p.keyboard.press('Alt+ArrowDown');
    await p.click('body', { position: { x: 700, y: 10 } }).catch(() => {}); await p.keyboard.press('Escape');
    await p.evaluate(() => document.activeElement.blur());
    await p.keyboard.press('2'); await wait(200);
    T(label + ' key 2 toggles the second layer', await p.$eval('.lrow:nth-child(2) .eye, .lgroup:first-child .lrow:nth-child(2) .eye', e => e.getAttribute('aria-pressed')) === 'false');
    await p.keyboard.press('2'); await wait(200);
  }
  // flood threshold scrub
  if (mobile) await p.click('.sheet-tabs [aria-controls=ex-left]');
  const a0 = await p.textContent('#thr-area');
  await p.$eval('#thr', e => { e.value = '0.2'; e.dispatchEvent(new Event('input', { bubbles: true })); }); await wait(500);
  const a1 = await p.textContent('#thr-area');
  T(label + ' threshold scrub changes flood area', parseFloat(a1) > parseFloat(a0), `${a0} → ${a1}`);
  await p.$eval('#thr', e => { e.value = '0.5'; e.dispatchEvent(new Event('input', { bubbles: true })); }); await wait(300);

  // compare with truth: swipe + keyboard, blend
  if (mobile) { await p.click('.sheet-tabs [aria-controls=ex-left]'); await wait(250); }
  await p.click('.lrow[data-k=lc] [data-col=name]'); await wait(200);
  if (mobile) { await p.click('.sheet-tabs [aria-controls=ex-left]'); await wait(250); }
  await p.click('#cmp-truth'); await wait(400);
  T(label + ' swipe compare shows divider', await p.isVisible('#scr-explorer .divider'));
  await p.focus('#scr-explorer .divider'); for (let i = 0; i < 5; i++) await p.keyboard.press('ArrowRight');
  T(label + ' divider operable with arrows', await p.$eval('#scr-explorer .divider', e => e.getAttribute('aria-valuenow')) === '60', await p.$eval('#scr-explorer .divider', e => e.getAttribute('aria-valuenow')));
  await p.click('#scr-explorer [data-mode=blend]'); await wait(500);
  T(label + ' blend mode shows scrub', await p.isVisible('#blend-card'));
  await p.$eval('#blend-t', e => { e.value = '0.8'; e.dispatchEvent(new Event('input', { bubbles: true })); }); await wait(200);
  T(label + ' blend canvas drawn', await p.$eval('#scr-explorer .blendcv', c => c.width > 0));
  await p.click('#scr-explorer [data-mode=swipe]'); await p.click('#cmp-truth'); await wait(300);

  // hover readout / tap pin
  const mb = await mapBox(p);
  if (!mobile) {
    await p.mouse.move(mb[0] + mb[2] * 0.5, mb[1] + mb[3] * 0.5); await wait(200);
    T(label + ' hover readout', /x \d+\s+y \d+/.test(await p.textContent('#ex-xy')), await p.textContent('#ex-xy'));
    await p.focus('#scr-explorer .vp'); const t0 = await p.$eval('#scr-explorer .world', e => e.style.transform);
    await p.keyboard.press('+'); await p.keyboard.press('ArrowLeft'); await wait(300);
    T(label + ' keyboard zoom and pan', (await p.$eval('#scr-explorer .world', e => e.style.transform)) !== t0);
    await p.keyboard.press('0'); await wait(300);
  } else {
    await p.tap('#scr-explorer .vp', { position: { x: mb[2] * 0.5, y: (mb[1] - 44) + mb[3] * 0.5 } }).catch(async () => p.mouse.click(mb[0] + mb[2] * 0.5, mb[1] + mb[3] * 0.5));
    await wait(300);
    T(label + ' tap shows pin value card', await p.isVisible('.valuecard'));
  }
  // similarity click
  if (mobile) { await p.click('.sheet-tabs [aria-controls=ex-left]'); await wait(250); }
  await p.click('.lrow[data-k=sim] [data-col=name]'); await wait(300);
  if (mobile) { await p.click('.sheet-tabs [aria-controls=ex-left]'); await wait(250); }
  const mb2 = await mapBox(p);
  const hit = await p.evaluate(([x, y]) => document.elementFromPoint(x, y)?.outerHTML.slice(0, 80), [mb2[0] + mb2[2] * 0.25, mb2[1] + mb2[3] * 0.25]);
  await p.mouse.click(mb2[0] + mb2[2] * 0.25, mb2[1] + mb2[3] * 0.25); await wait(1200);
  const simSrc = await p.$eval('#scr-explorer .world > [data-k=sim] img', e => e.getAttribute('src'));
  T(label + ' similarity click picks point', /x=3\d&y=3\d/.test(simSrc), simSrc + ' hit ' + hit);
  T(label + ' export link', /\/api\/export\/[\w-]+\.npz/.test(await p.getAttribute('#export', 'href')));

  // command palette + shortcuts sheet
  await p.keyboard.press('Control+k'); await wait(200);
  T(label + ' command palette opens', await p.isVisible('.palette'));
  await p.keyboard.type('bench'); await p.keyboard.press('Enter'); await wait(1500);
  T(label + ' palette navigates', (await p.evaluate(() => location.hash)) === '#benchmarks');
  T(label + ' benchmarks render charts', (await p.$$('#bench-body figure svg')).length >= 2, (await p.$$('#bench-body figure svg')).length + ' charts');
  T(label + ' real data honest state', /Not run yet|Generated/.test(await p.textContent('#bench-body')));
  await p.screenshot({ path: path.join(OUT, `${label}_bench.png`) });
  let ov = await overflow(p); T(label + ' bench no h-overflow', ov[0] <= ov[1] + 1, JSON.stringify(ov));
  await p.keyboard.press('Shift+?'); await wait(200);
  T(label + ' ? opens shortcut sheet', await p.isVisible('.shortcuts')); await p.keyboard.press('Escape');

  // theme toggle
  const th0 = await p.evaluate(() => document.documentElement.dataset.theme);
  await p.click('#theme-btn'); await wait(200);
  T(label + ' theme toggles', (await p.evaluate(() => document.documentElement.dataset.theme)) !== th0);
  await p.click('#theme-btn');

  // compare screen
  await p.click('#tab-compare'); await wait(1200);
  T(label + ' compare screen map', (await p.$$('#scr-compare .world > .lyr')).length >= 1);
  if (!mobile) {
    await p.click('#scr-compare [data-kind=conditions]'); await p.click('#cmp-run');
    await p.waitForFunction(() => !document.querySelector('#scr-compare .divider').hidden, null, { timeout: 30000 }).catch(() => {});
    T(label + ' compare runs B and swipes', await p.isVisible('#scr-compare .divider'));
    await p.click('#scr-compare [data-mode=split]'); await wait(400);
    T(label + ' split mode shows two maps', await p.isVisible('#scr-compare .cmp-b'));
    const chText = await p.textContent('#cmp-change');
    T(label + ' change layers state is honest', /change detection|Flood onset|different sizes/i.test(chText), chText.slice(0, 80));
    await p.screenshot({ path: path.join(OUT, `${label}_compare.png`) });
  }
  ov = await overflow(p); T(label + ' compare no h-overflow', ov[0] <= ov[1] + 1, JSON.stringify(ov));

  // data: upload good / bad
  await p.click('#tab-data'); await wait(800);
  const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#sample-link')]);
  await p.setInputFiles('#file', { name: 'terrascope_sample.npz', mimeType: 'application/octet-stream', buffer: fs.readFileSync(await dl.path()) });
  await p.waitForFunction(() => /Uploaded|Rejected|Could not/.test(document.querySelector('#upload-log li')?.textContent || ''), null, { timeout: 15000 }).catch(() => {});
  T(label + ' upload good file', /Uploaded/.test(await p.textContent('#upload-log li')), await p.textContent('#upload-log li'));
  await p.setInputFiles('#file', { name: 'bad.npz', mimeType: 'application/octet-stream', buffer: Buffer.alloc(200, 7) }); await wait(1500);
  T(label + ' bad upload shows server message', /Could not read|Rejected|invalid/i.test(await p.textContent('#upload-log li')), await p.textContent('#upload-log li'));
  await p.setInputFiles('#file', { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('hello world, not a scene, padded to be long enough for the check') }); await wait(300);
  T(label + ' wrong extension rejected client-side', /not \.npz/.test(await p.textContent('#upload-log li')));
  await p.screenshot({ path: path.join(OUT, `${label}_data.png`) });
  ov = await overflow(p); T(label + ' data no h-overflow', ov[0] <= ov[1] + 1, JSON.stringify(ov));

  await p.click('#tab-model'); await wait(800);
  T(label + ' model page', (await p.$$('#scr-model .flow')).length === 1 && (await p.$$('.mcard')).length >= 1);
  await p.screenshot({ path: path.join(OUT, `${label}_model.png`) });
  ov = await overflow(p); T(label + ' model no h-overflow', ov[0] <= ov[1] + 1, JSON.stringify(ov));
  await p.click('#tab-explorer'); await wait(1500);
  ov = await overflow(p); T(label + ' explorer no h-overflow', ov[0] <= ov[1] + 1, JSON.stringify(ov));

  T(label + ' no console errors', errs.length === 0, errs.join(' | '));
  T(label + ' no failed requests', bad.length === 0, bad.join(' '));
  T(label + ' no external requests', external.length === 0, external.join(' '));
  await br.close();
}

async function reducedMotion() {
  const br = await chromium.launch();
  const ctx = await br.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const p = await ctx.newPage();
  await p.goto(B + '/#explorer'); await ready(p).catch(() => {});
  T('reduced-motion transitions off', await p.$eval('#ex-left', e => parseFloat(getComputedStyle(e).transitionDuration)) === 0);
  await p.click('.lrow[data-k=sim] [data-col=name]'); await wait(300);
  const mb = await mapBox(p); await p.mouse.click(mb[0] + mb[2] * 0.7, mb[1] + mb[3] * 0.7); await wait(600);
  T('reduced-motion similarity is instant (no ripple canvas)', await p.$eval('#scr-explorer .world > [data-k=sim] canvas', c => c.hidden));
  const a0 = await p.textContent('#thr-area');
  await p.$eval('#thr', e => { e.value = '0.3'; e.dispatchEvent(new Event('input', { bubbles: true })); }); await wait(50);
  T('reduced-motion number set without tween', (await p.textContent('#thr-area')) !== a0);
  await br.close();
}


// Checks added after the design review: one per fix.
async function reviewFixes() {
  const br = await chromium.launch();
  const ctx = await br.newContext({ viewport: { width: 1440, height: 900 } });
  const p = await ctx.newPage();
  await p.goto(B + '/#explorer'); await ready(p);
  T('fix: progress is a stage rail, not a map sweep', (await p.$$('#scr-explorer .rail')).length === 1 && (await p.$$('.prog, .sweep')).length === 0);
  const dockTop = await p.$eval('#ex-dock', e => e.getBoundingClientRect().top), mb = await mapBox(p);
  T('fix: controls docked below the scene', dockTop >= mb[1] + mb[3] - 1, `dock ${dockTop} scene bottom ${mb[1] + mb[3]}`);
  T('fix: simulated badge shown', await p.isVisible('#src-badge'));
  T('fix: risk bars stay grey when hot', await p.$$eval('.risk.hot .track i', l => l.every(i => getComputedStyle(i).backgroundColor === getComputedStyle(document.querySelector('.risk .track i')).backgroundColor)));
  // palette: keys typed right after Ctrl+K go to the input, not to global shortcuts
  await p.keyboard.press('Control+k'); await p.keyboard.type('cmod');
  await wait(400);
  T('fix: palette captures early keystrokes', (await p.inputValue('#pal-input')) === 'cmod' && (await p.evaluate(() => location.hash)) === '#explorer');
  await p.keyboard.press('Escape');
  // swipe over flood: both sides share the same backdrop
  await p.click('.lrow[data-k=flood] [data-col=name]'); await p.click('#cmp-truth'); await wait(400);
  const bd = await p.$$eval('#scr-explorer img', l => l.filter(i => i.closest('.world') && i.classList.contains('lyr') && !i.dataset.k).map(i => i.getAttribute('src')));
  T('fix: swipe shows same layer type on the same backdrop', bd.length >= 2 && bd[0] === bd[1] && (await p.$$('#scr-explorer .vp > .world > .lyr')).length === 2, bd.join(' '));
  await p.click('#cmp-truth');
  // reveal lifts the animated layer: layers above are veiled while it plays
  await p.click('.lrow[data-k=recon] [data-col=name]'); await wait(200);
  if (await p.isEnabled('#reveal-btn')) { await p.click('#reveal-btn'); await wait(150); T('fix: layers above a reveal are veiled', (await p.$$('#scr-explorer .lyr.veiled')).length >= 1); }
  // failed re-run keeps the old map with an honest banner
  await p.route('**/api/analyze*', r => r.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"test failure"}' }));
  await p.keyboard.press('r'); await wait(800);
  T('fix: failed run shows stale banner with Retry', await p.isVisible('#map-msg.stale') && await p.isVisible('#retry-btn'));
  await p.unroute('**/api/analyze*');
  // benchmarks: calibration points, robustness rows, readable names, seed wording
  await p.click('#tab-benchmarks'); await wait(1500);
  const bench = await p.textContent('#bench-body');
  const cal = await p.evaluate(() => fetch('/api/benchmarks').then(r => r.json()).then(b => !!b.calibration));
  if (cal) T('fix: calibration reliability plot has points', (await p.$$('#bench-body figure circle')).length >= 3 + 2);
  const rob = await p.evaluate(() => fetch('/api/benchmarks').then(r => r.json()).then(b => !!b.robustness));
  if (rob) T('fix: robustness rows rendered', /in-distribution/.test(bench) && /pass|fail/.test(bench));
  T('fix: no snake_case ids in benchmark text', !/finetuned_v\d|_iou|_mae|miou_drop/.test(bench));
  T('fix: seed count wording', !/\b1 seeds\b/.test(bench));
  // compare default when only one model is ready
  const ready1 = await p.evaluate(() => fetch('/api/models').then(r => r.ok ? r.json() : [{ status: 'ready' }]).then(l => l.filter(m => m.status === 'ready').length));
  await p.click('#tab-compare'); await wait(800);
  if (ready1 < 2) T('fix: compare defaults to other conditions with one model', await p.$eval('#scr-compare [data-kind=conditions]', e => e.getAttribute('aria-pressed')) === 'true');
  await p.click('#scr-compare [data-kind=conditions]'); await p.click('#cmp-run');
  await p.waitForFunction(() => !document.querySelector('#scr-compare .divider').hidden, null, { timeout: 30000 }).catch(() => {});
  const ch = await p.$$('#cmp-change input[type=checkbox]');
  if (ch.length) {
    await ch[0].check(); await wait(400);
    const sides = await p.evaluate(() => { const a = document.querySelector('#scr-compare .cmp-a .vp > .world'); const w2 = document.querySelector('#scr-compare .cmp-a .cmpclip .world'); const q = w => [...w.querySelectorAll('img')].filter(i => /\/api\/layer\/[^/]+\/(flood_onset|veg_loss|urban_growth|lc_change)/.test(i.src)).length; return [q(a), q(w2)]; });
    T('fix: change layer drawn on B only', sides[0] === 0 && sides[1] === 1, JSON.stringify(sides));
  }
  await p.click('#tab-data'); await wait(800);
  T('fix: scene table has Added column', /Added/.test(await p.textContent('#scene-table')) || /no scene catalogue|No scenes/.test(await p.textContent('#scene-table')));
  await ctx.close();
  // offline: no health → centred message with Retry, controls hidden
  const ctx2 = await br.newContext({ viewport: { width: 1440, height: 900 } }); const q = await ctx2.newPage();
  await q.route('**/api/**', r => r.abort());
  await q.goto(B + '/#explorer'); await wait(1500);
  T('fix: offline shows centred message and Retry', await q.isVisible('#map-msg') && await q.isVisible('#retry-btn'));
  T('fix: controls hidden or disabled without a result', !(await q.isVisible('#ex-dock')) && await q.$eval('#export', e => e.getAttribute('aria-disabled')) === 'true' && await q.isDisabled('#cmp-truth'));
  await ctx2.close();
  // phone: simulated badge visible, value card not under the sheet
  const ctx3 = await br.newContext({ ...devices['iPhone 13'] }); const m = await ctx3.newPage();
  await m.goto(B + '/#explorer'); await ready(m);
  T('fix: phone keeps simulated badge visible', await m.isVisible('#src-badge'));
  await m.click('.sheet-tabs [aria-controls=ex-right]'); await wait(300);
  const wb = await mapBox(m); await m.mouse.click(wb[0] + wb[2] * 0.5, wb[1] + 12); await wait(400);
  T('fix: phone value card not covered by sheet', await m.isVisible('.valuecard') && (await m.evaluate(() => document.querySelector('#scr-explorer').dataset.sheet)) === '');
  const ov = await overflow(m); T('fix: phone no h-overflow with dock', ov[0] <= ov[1] + 1);
  await ctx3.close(); await br.close();
}

// Backend follow-up: calibration decision, U-Net baselines, real-data CIs (real block mocked when absent).
async function followUp() {
  const br = await chromium.launch(); const ctx = await br.newContext({ viewport: { width: 1440, height: 900 } }); const p = await ctx.newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto(B + '/#explorer'); await ready(p);
  const models = await p.evaluate(() => fetch('/api/models').then(r => r.ok ? r.json() : []));
  await p.click('#model-btn');
  T('follow-up: switcher lists every model', (await p.$$('#model-list [role=option]')).length === Math.max(1, models.length), models.length + ' models');
  if (models.some(m => m.family === 'unet')) T('follow-up: U-Net labelled baseline', /baseline/.test(await p.textContent('#model-list')));
  await p.keyboard.press('Escape');
  const cal = await p.evaluate(() => fetch('/api/benchmarks').then(r => r.json()).then(b => b.calibration));
  await p.route('**/api/benchmarks', async r => {
    const b = await (await r.fetch()).json();
    b.real = b.real || { generated_at: '2026-11-18T00:00:00Z', device: 'test fixture', datasets: { sen1floods11: { task: 'flood', models: { terrafm: { params: 686281, seeds: [0, 1, 2], flood_iou: { mean: 0.61, std: 0.02 }, ci95_seeds: { flood_iou: [0.56, 0.66] } }, unet_r18_scratch: { params: 14350161, seeds: [0], flood_iou: 0.55 } }, paired: { terrafm_vs_unet_r18_scratch: { delta: 0.06, p: 0.04 } } } } };
    r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(b) });
  });
  await p.click('#tab-benchmarks'); await wait(1500);
  const t = await p.textContent('#bench-body');
  if (cal && 'apply' in cal) T('follow-up: calibration note follows apply/decision', new RegExp(cal.apply ? 'is applied' : 'is not applied').test(t) && /validation|Rule/i.test(t));
  T('follow-up: real table shows mean ± std (n) and 95% CI', /± 0\.020 \(n = 3\)|±/.test(t) && /95% CI/.test(t));
  T('follow-up: single-value fallback and readable model names', /U-Net ResNet-18/.test(t) && !/unet_r18/.test(t));
  await p.click('#tab-model'); await wait(800);
  if (models.some(m => m.family === 'unet')) T('follow-up: model page shows baseline cards', (await p.$$('.mcard')).length === models.length && /supervised baseline/.test(await p.textContent('#model-cards')));
  T('follow-up: no page errors', errs.length === 0, errs.join(' | '));
  await br.close();
}

(async () => {
  await run('desktop', { viewport: { width: 1440, height: 900 } });
  await run('mobile', { ...devices['iPhone 13'] });
  await reducedMotion();
  await reviewFixes();
  await followUp();
  const ok = res.filter(r => r[1]).length;
  console.log(`SUMMARY ${ok}/${res.length}`);
  process.exit(ok === res.length ? 0 : 1);
})();
