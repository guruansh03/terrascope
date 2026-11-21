// Accessibility check with axe-core on every screen in light and dark themes.
//   node tests/e2e/a11y.js [baseUrl]
// axe-core is looked up via require.resolve, then $AXE_PATH (path to axe.min.js).
// Install once with:  npm install axe-core   (in this repo or any folder on NODE_PATH)
const { chromium } = require('playwright');
const fs = require('fs');
const B = (process.argv[2] || 'http://127.0.0.1:8000').replace(/\/$/, '');
let axePath = process.env.AXE_PATH;
if (!axePath) { try { axePath = require.resolve('axe-core/axe.min.js'); } catch { /* not installed */ } }
if (!axePath || !fs.existsSync(axePath)) { console.error('axe-core not found: npm install axe-core, or set AXE_PATH'); process.exit(2); }
const AXE = fs.readFileSync(axePath, 'utf8');
const SCREENS = ['explorer', 'compare', 'benchmarks', 'data', 'model'];

(async () => {
  const br = await chromium.launch();
  let serious = 0, total = 0;
  for (const theme of ['light', 'dark']) {
    const ctx = await br.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: theme });
    const p = await ctx.newPage();
    await p.goto(B + '/#explorer');
    await p.waitForFunction(() => document.querySelector('#sb-stage .v')?.textContent === 'done', null, { timeout: 30000 }).catch(() => {});
    for (const s of SCREENS) {
      await p.evaluate(s => { location.hash = s; }, s);
      await p.waitForTimeout(s === 'benchmarks' || s === 'compare' ? 1500 : 700);
      // open the interactive states so they are checked too
      if (s === 'explorer') await p.evaluate(() => { document.querySelector('#cmp-truth')?.click(); });
      await p.waitForTimeout(300);
      await p.addScriptTag({ content: AXE });
      const r = await p.evaluate(async () => {
        const res = await window.axe.run(document, { resultTypes: ['violations'] });
        return res.violations.map(v => ({ id: v.id, impact: v.impact, n: v.nodes.length, help: v.help, where: v.nodes.slice(0, 3).map(n => n.target.join(' ')).join(' | ') }));
      });
      if (s === 'explorer') await p.evaluate(() => { document.querySelector('#cmp-truth')?.click(); });
      const bad = r.filter(v => v.impact === 'serious' || v.impact === 'critical');
      serious += bad.length; total += r.length;
      console.log(`${bad.length ? 'FAIL' : 'PASS'} ${theme} ${s}: ${r.length} violations, ${bad.length} serious/critical`);
      for (const v of r) console.log(`   ${v.impact} ${v.id} ×${v.n} — ${v.help} — ${v.where}`);
    }
    // the command palette dialog
    await p.keyboard.press('Control+k'); await p.waitForTimeout(600);
    const pr = await p.evaluate(async () => (await window.axe.run(document.querySelector('.palette'))).violations.map(v => ({ id: v.id, impact: v.impact })));
    const pb = pr.filter(v => v.impact === 'serious' || v.impact === 'critical'); serious += pb.length; total += pr.length;
    console.log(`${pb.length ? 'FAIL' : 'PASS'} ${theme} command palette: ${pr.length} violations ${pr.map(v => v.impact + ' ' + v.id).join(', ')}`);
    await ctx.close();
  }
  await br.close();
  console.log(`SUMMARY a11y: ${total} violations, ${serious} serious/critical`);
  process.exit(serious ? 1 : 0);
})();
