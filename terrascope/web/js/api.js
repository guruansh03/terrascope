// Server API with feature detection. Endpoints that a server version lacks return
// {unsupported:true} so the UI can show an honest state instead of inventing data.
import { getJSON, postJSON } from './util.js';

export const caps = { async: null, models: null, scenes: null, change: null };

export async function health() { return getJSON('/api/health'); }

export async function models(health) {
  const r = await getJSON('/api/models');
  if (r.ok && Array.isArray(r.data)) { caps.models = true; return r.data; }
  caps.models = false;
  // Older server: only TerraFM exists. Parameter count comes from /api/health, nothing else is assumed.
  return [{ id: 'terrafm', name: 'TerraFM', family: 'terrafm', params: health?.params ?? null, sensors: ['optical', 'sar', 'dem'],
    tasks: ['landcover', 'flood', 'veg', 'cloud'], status: health?.ready ? 'ready' : 'weights_missing', weights_file: null, license: null, source_url: null, legacy: true }];
}

export async function scenes() {
  const r = await getJSON('/api/scenes');
  if (r.status === 404 || r.status === 405) { caps.scenes = false; return { unsupported: true, list: [] }; }
  if (!r.ok) return { error: r.error, list: [] };
  caps.scenes = true;
  return { list: r.data };
}
export const renameScene = (id, name) => postJSON(`/api/scenes/${encodeURIComponent(id)}`, { name }, 'PATCH');
export const deleteScene = id => getJSON(`/api/scenes/${encodeURIComponent(id)}`, { method: 'DELETE' });

export async function upload(file) {
  const fd = new FormData(); fd.append('file', file);
  return getJSON('/api/upload', { method: 'POST', body: fd });
}

export async function change(a, b) {
  const r = await getJSON(`/api/change?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  if (r.status === 404 && r.error === 'Not Found' && caps.change !== true) return { unsupported: true };
  if (r.ok) caps.change = true;
  return r;
}

export const benchmarks = () => getJSON('/api/benchmarks');

/**
 * Run an analysis. onStage(stage, t_ms, detail) is called only with real server SSE events.
 * When the async endpoint is missing, onStage('sync') is called once and the plain endpoint is used.
 * Returns {ok, data} | {ok:false, status, error}.
 */
export async function analyze(body, onStage = () => {}, signal) {
  if (caps.async !== false) {
    const r = await postJSON('/api/analyze_async', body);
    if (r.status === 404 || r.status === 405) caps.async = false;
    else if (!r.ok) return r;
    else { caps.async = true; return followJob(r.data.job_id, onStage, signal); }
  }
  onStage('sync', null);
  return postJSON('/api/analyze', body);
}

function followJob(id, onStage, signal) {
  return new Promise(resolve => {
    let finished = false;
    const finish = async () => {
      if (finished) return; finished = true; es?.close();
      for (let i = 0; i < 40; i++) {
        const j = await getJSON(`/api/jobs/${id}`);
        if (!j.ok) return resolve(j);
        if (j.data.status === 'done' && j.data.result) return resolve({ ok: true, data: j.data.result });
        if (j.data.status === 'error') return resolve({ ok: false, status: 500, error: j.data.error || j.data.detail || 'Analysis failed' });
        await new Promise(r => setTimeout(r, 150));
      }
      resolve({ ok: false, status: 504, error: 'Job did not finish' });
    };
    let es;
    try { es = new EventSource(`/api/jobs/${id}/events`); } catch { return finish(); }
    signal?.addEventListener('abort', () => { finished = true; es.close(); resolve({ ok: false, aborted: true }); });
    es.onmessage = ev => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      onStage(m.stage, m.t_ms, m.detail);
      if (m.stage === 'done' || m.stage === 'error') finish();
    };
    es.onerror = () => { if (!finished) finish(); };
  });
}
