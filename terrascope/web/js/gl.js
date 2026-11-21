// One shared, hand-written WebGL1 context. Effects render into it and the result is copied
// onto ordinary 2D canvases on the map, so any number of GPU layers costs one GL context.
// When WebGL is unavailable `gpu.ok` is false and effects use their Canvas2D paths.

const VS = `attribute vec2 p; varying vec2 uv;
void main(){ uv = vec2(p.x*.5+.5, .5-p.y*.5); gl_Position = vec4(p,0.,1.); }`;

class GPU {
  constructor() {
    this.canvas = document.createElement('canvas');
    let gl = null;
    try { gl = this.canvas.getContext('webgl', { premultipliedAlpha: true, alpha: true, antialias: false, preserveDrawingBuffer: false }); } catch { gl = null; }
    this.gl = gl; this.ok = !!gl && !new URLSearchParams(location.search).has('nogl');
    this.progs = new Map(); this.tex = new Map();
    if (!this.ok) return;
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    this.canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); this.ok = false; });
  }
  program(name, fs) {
    if (this.progs.has(name)) return this.progs.get(name);
    const gl = this.gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, 'precision mediump float; varying vec2 uv;\n' + fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const info = { p, loc: {} };
    this.progs.set(name, info);
    return info;
  }
  /** Upload (or reuse) a texture. src: image/canvas, or {data:Uint8Array, w, h} single channel. */
  texture(key, src, { linear = true } = {}) {
    const gl = this.gl;
    let t = this.tex.get(key);
    if (t && !src) { this.tex.delete(key); this.tex.set(key, t); return t.t; } // LRU touch
    if (!t) { t = { t: gl.createTexture() }; this.tex.set(key, t); }
    gl.bindTexture(gl.TEXTURE_2D, t.t);
    if (src.data) gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, src.w, src.h, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, src.data);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    const f = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    while (this.tex.size > 40) { const [k, old] = this.tex.entries().next().value; gl.deleteTexture(old.t); this.tex.delete(k); }
    return t.t;
  }
  has(key) { return this.tex.has(key); }
  /** Draw program `name` at w×h and copy into target 2D canvas. */
  draw(name, fs, target, { w, h, uniforms = {}, textures = {} }) {
    const gl = this.gl, pr = this.program(name, fs);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.useProgram(pr.p);
    const loc = n => (n in pr.loc ? pr.loc[n] : (pr.loc[n] = gl.getUniformLocation(pr.p, n)));
    const a = gl.getAttribLocation(pr.p, 'p');
    gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    let unit = 0;
    for (const [n, key] of Object.entries(textures)) {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, this.texture(key)); gl.uniform1i(loc(n), unit++);
    }
    for (const [n, v] of Object.entries(uniforms)) {
      const l = loc(n); if (l == null) continue;
      if (typeof v === 'number') gl.uniform1f(l, v);
      else if (v.length === 2) gl.uniform2fv(l, v);
      else if (v.length === 3) gl.uniform3fv(l, v);
      else gl.uniform3fv(l, v); // flat vec3 arrays
    }
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (target.width !== w || target.height !== h) { target.width = w; target.height = h; }
    const c = target.getContext('2d');
    c.clearRect(0, 0, w, h); c.drawImage(this.canvas, 0, 0);
  }
}

export const gpu = new GPU();

/** Read RGBA pixels of an image scaled to w×h (for CPU fallbacks and similarity ranks). */
export function pixels(img, w, h) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(img, 0, 0, w, h);
  return x.getImageData(0, 0, w, h).data;
}
