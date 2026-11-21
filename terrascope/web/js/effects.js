// GPU map effects (WebGL via gl.js) with Canvas2D fallbacks. Every effect draws real data:
// class indices, flood probabilities, the cloud mask, similarity ranks or server-rendered layers.
import { gpu, pixels } from './gl.js';

const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const UP = 4; // GPU layers render at 4× the scene grid, matching the server PNGs

function ensure(key, src, opt) { if (!gpu.has(key)) gpu.texture(key, src, opt); }

// ---------- land cover with hover-dim
const FS_LC = `uniform sampler2D idx; uniform vec3 pal[8]; uniform float hover; uniform float dim;
void main(){ float i = floor(texture2D(idx, uv).r*255.+.5); vec3 c = pal[0];
  for (int k=0;k<8;k++) if (float(k)==i) c = pal[k];
  float a = (hover < 0. || abs(i-hover) < .5) ? 1. : dim; gl_FragColor = vec4(c*a, a); }`;
export function renderLC(target, { key, data, w, h, palette, hover = -1, dim = 1 }) {
  if (gpu.ok) {
    ensure(key, { data, w, h }, { linear: false });
    const pal = new Float32Array(24); palette.slice(0, 8).forEach((c, i) => pal.set(hex(c), i * 3));
    gpu.draw('lc', FS_LC, target, { w: w * UP, h: h * UP, uniforms: { pal, hover, dim }, textures: { idx: key } });
    return;
  }
  const P = palette.map(hex), ctx = sized(target, w, h), im = ctx.createImageData(w, h), d = im.data;
  for (let i = 0; i < w * h; i++) {
    const c = P[data[i]] || P[0], a = hover < 0 || data[i] === hover ? 1 : dim;
    d[i * 4] = c[0] * 255; d[i * 4 + 1] = c[1] * 255; d[i * 4 + 2] = c[2] * 255; d[i * 4 + 3] = a * 255;
  }
  ctx.putImageData(im, 0, 0);
}

// ---------- flood probability re-thresholded live
const FS_FLOOD = `uniform sampler2D prob; uniform float thr; uniform vec3 col;
void main(){ float p = texture2D(prob, uv).r; float m = smoothstep(thr-.004, thr+.004, p);
  float e = smoothstep(thr-.03, thr, p) * (1.-m); gl_FragColor = vec4(col*(.78*m) + vec3(1.)*e*.5, .78*m + e*.5); }`;
export function renderFlood(target, { key, data, w, h, thr, color = '#19c3f0' }) {
  if (gpu.ok) {
    ensure(key, { data, w, h }, { linear: true });
    gpu.draw('flood', FS_FLOOD, target, { w: w * UP, h: h * UP, uniforms: { thr, col: hex(color) }, textures: { prob: key } });
    return;
  }
  const c = hex(color), t = thr * 255, ctx = sized(target, w, h), im = ctx.createImageData(w, h), d = im.data;
  for (let i = 0; i < w * h; i++) if (data[i] >= t) { d[i * 4] = c[0] * 255; d[i * 4 + 1] = c[1] * 255; d[i * 4 + 2] = c[2] * 255; d[i * 4 + 3] = 199; }
  ctx.putImageData(im, 0, 0);
}

// ---------- cross-fade between two layers
const FS_BLEND = `uniform sampler2D a; uniform sampler2D b; uniform float t;
void main(){ gl_FragColor = mix(texture2D(a, uv), texture2D(b, uv), t); }`;
export function renderBlend(target, { keyA, srcA, keyB, srcB, w, h, t }) {
  if (gpu.ok) {
    gpu.texture(keyA, srcA); gpu.texture(keyB, srcB); // sources may be live canvases, re-upload
    gpu.draw('blend', FS_BLEND, target, { w, h, uniforms: { t }, textures: { a: keyA, b: keyB } });
    return;
  }
  const ctx = sized(target, w, h);
  ctx.globalAlpha = 1; ctx.drawImage(srcA, 0, 0, w, h);
  ctx.globalAlpha = t; ctx.drawImage(srcB, 0, 0, w, h); ctx.globalAlpha = 1;
}

// ---------- cloud-removal reveal: only pixels in the cloud mask ever change
const FS_REVEAL = `uniform sampler2D a; uniform sampler2D b; uniform sampler2D m; uniform float prog; uniform vec2 grid;
void main(){ vec2 cell = floor(uv*grid); float h = fract(sin(dot(cell, vec2(12.9898,78.233)))*43758.5453);
  float mk = texture2D(m, uv).r; float s = step(h, prog) * step(.5/255., mk);
  gl_FragColor = mix(texture2D(a, uv), texture2D(b, uv), s); }`;
export function renderReveal(target, { key, input, recon, mask, w, h, prog }) {
  const W = recon.naturalWidth || w * UP, H = recon.naturalHeight || h * UP;
  if (gpu.ok) {
    ensure(key + ':in', input); ensure(key + ':rc', recon); ensure(key + ':m', { data: mask, w, h }, { linear: false });
    gpu.draw('reveal', FS_REVEAL, target, { w: W, h: H, uniforms: { prog, grid: [w, h] }, textures: { a: key + ':in', b: key + ':rc', m: key + ':m' } });
    return;
  }
  const ctx = sized(target, w, h);
  target._a ??= pixels(input, w, h); target._b ??= pixels(recon, w, h);
  const im = ctx.createImageData(w, h), d = im.data, A = target._a, B = target._b;
  for (let i = 0; i < w * h; i++) {
    const x = i % w, y = (i / w) | 0, r = Math.abs(Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
    const S = mask[i] > 0 && r < prog ? B : A;
    d[i * 4] = S[i * 4]; d[i * 4 + 1] = S[i * 4 + 1]; d[i * 4 + 2] = S[i * 4 + 2]; d[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

// ---------- similarity ripple: pixels light up in order of similarity rank
const FS_RIPPLE = `uniform sampler2D sim; uniform sampler2D base; uniform sampler2D rank; uniform float prog; uniform vec2 c;
void main(){ float r = texture2D(rank, uv).r; vec4 dimmed = vec4(texture2D(base, uv).rgb*.45, 1.);
  vec4 col = mix(dimmed, texture2D(sim, uv), step(r, prog));
  float d = distance(uv, c); float k = 1.-prog;
  float ring = (1.-smoothstep(0., .004, abs(d - prog*1.2))) + (1.-smoothstep(0., .004, abs(d - prog*.7)))*.6;
  col.rgb = mix(col.rgb, vec3(1.), clamp(ring*k, 0., 1.)*.85); gl_FragColor = col; }`;
export function renderRipple(target, { key, sim, base, rank, w, h, prog, at }) {
  const W = sim.naturalWidth, H = sim.naturalHeight;
  if (gpu.ok) {
    gpu.texture(key + ':s', sim); ensure(key + ':b', base); gpu.texture(key + ':r', { data: rank, w, h }, { linear: false });
    gpu.draw('ripple', FS_RIPPLE, target, { w: W, h: H, uniforms: { prog, c: at }, textures: { sim: key + ':s', base: key + ':b', rank: key + ':r' } });
    return;
  }
  const ctx = sized(target, w, h);
  const S = pixels(sim, w, h), B = target._base ??= pixels(base, w, h), im = ctx.createImageData(w, h), d = im.data, lim = prog * 255;
  for (let i = 0; i < w * h; i++) {
    const on = rank[i] <= lim;
    for (let k = 0; k < 3; k++) d[i * 4 + k] = on ? S[i * 4 + k] : B[i * 4 + k] * 0.45;
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

/**
 * Similarity rank per pixel (0 = most similar, 255 = least) derived from the server's similarity PNG.
 * The server draws heat = inferno(sim) with alpha ∝ sim over the base image dimmed to 45 %, so
 * luminance above the dimmed base increases monotonically with similarity.
 */
export function simRank(simImg, baseImg, w, h) {
  const S = pixels(simImg, w, h), B = pixels(baseImg, w, h), n = w * h, est = new Float32Array(n);
  const L = (a, i) => 0.2126 * a[i] + 0.7152 * a[i + 1] + 0.0722 * a[i + 2];
  for (let i = 0; i < n; i++) est[i] = L(S, i * 4) - 0.45 * L(B, i * 4);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => est[b] - est[a]);
  const rank = new Uint8Array(n);
  order.forEach((p, r) => { rank[p] = Math.floor(r / n * 255); });
  return rank;
}

function sized(c, w, h) { if (c.width !== w || c.height !== h) { c.width = w; c.height = h; } const x = c.getContext('2d'); x.clearRect(0, 0, w, h); return x; }

/** (b) Blend on its own visible WebGL canvas: textures are uploaded once in set(), draw(t) only sets a uniform. */
const VS2 = `attribute vec2 p; varying vec2 uv; void main(){ uv = vec2(p.x*.5+.5, .5-p.y*.5); gl_Position = vec4(p,0.,1.); }`;
export class BlendGL {
  constructor(canvas) {
    this.c = canvas;
    let gl = null; try { if (gpu.ok) gl = canvas.getContext('webgl', { premultipliedAlpha: true, antialias: false }); } catch { gl = null; }
    this.gl = gl;
    if (!gl) return;
    const sh = (t, src) => { const x = gl.createShader(t); gl.shaderSource(x, src); gl.compileShader(x); return x; };
    const pr = this.p = gl.createProgram();
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS2)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, 'precision mediump float; varying vec2 uv;\n' + FS_BLEND)); gl.linkProgram(pr);
    gl.useProgram(pr);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const a = gl.getAttribLocation(pr, 'p'); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    this.tex = [gl.createTexture(), gl.createTexture()];
    this.uT = gl.getUniformLocation(pr, 't');
    gl.uniform1i(gl.getUniformLocation(pr, 'a'), 0); gl.uniform1i(gl.getUniformLocation(pr, 'b'), 1);
  }
  set(a, b) {
    this.a = a; this.b = b;
    const w = Math.max(a.naturalWidth || a.width, b.naturalWidth || b.width), h = Math.max(a.naturalHeight || a.height, b.naturalHeight || b.height);
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; }
    const gl = this.gl; if (!gl) return;
    gl.viewport(0, 0, w, h);
    [a, b].forEach((src, i) => {
      gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, this.tex[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    });
  }
  draw(t) {
    const gl = this.gl;
    if (!gl) { const x = this.c.getContext('2d'); x.clearRect(0, 0, this.c.width, this.c.height); x.globalAlpha = 1; x.drawImage(this.a, 0, 0, this.c.width, this.c.height); x.globalAlpha = t; x.drawImage(this.b, 0, 0, this.c.width, this.c.height); x.globalAlpha = 1; return; }
    gl.uniform1f(this.uT, t); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}
