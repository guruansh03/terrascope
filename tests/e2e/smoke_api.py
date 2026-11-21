import httpx, json, io, numpy as np, time, sys
B=sys.argv[1] if len(sys.argv)>1 else "http://localhost:8000"
c=httpx.Client(base_url=B, timeout=60, headers={"Connection":"close"}, limits=httpx.Limits(max_keepalive_connections=0))
R=[]
def t(name, fn, expect):
    try: got=fn()
    except Exception as e: got=f"EXC {e!r}"
    ok = expect(got) if not isinstance(got,str) or not got.startswith("EXC") else False
    R.append((name, ok, str(got)[:160])); print(("PASS" if ok else "FAIL"), name, "|", str(got)[:160])
def A(**k):
    """POST /api/analyze; waits out the per-IP rate limit (429 + Retry-After) so the suite runs with defaults."""
    for _ in range(30):
        r = c.post("/api/analyze", json=k)
        if r.status_code != 429: return r
        time.sleep(float(r.headers.get("retry-after", 2)))
    return r
t("GET /", lambda: c.get("/").status_code, lambda s: s==200)
import re
t("GET index <script src> assets", lambda: [c.get(u).status_code for u in re.findall(r'<script[^>]+src="(/static/[^"]+)"', c.get("/").text)], lambda l: l and set(l)=={200})
t("health", lambda: c.get("/api/health").json(), lambda j: j["ready"] and {"params","models","device","version"} <= set(j))
r=A(seed=7); aid=r.json()["id"]
t("analyze default 200", lambda: r.status_code, lambda s: s==200)
for L in r.json()["layers"]:
    pass
t("all layers PNG", lambda: [c.get(f"/api/layer/{aid}/{L}.png").status_code for L in r.json()["layers"]], lambda l: set(l)=={200})
t("layer unknown 404", lambda: c.get(f"/api/layer/{aid}/nope.png").status_code, lambda s: s==404)
t("similarity", lambda: c.get(f"/api/similarity/{aid}.png?x=10&y=10").headers["content-type"], lambda s: s=="image/png")
t("similarity neg coords", lambda: c.get(f"/api/similarity/{aid}.png?x=-500&y=-500").status_code, lambda s: s in (200,400,422))
t("similarity huge coords", lambda: c.get(f"/api/similarity/{aid}.png?x=99999&y=99999").status_code, lambda s: s in (200,400,422))
t("export npz", lambda: sorted(np.load(io.BytesIO(c.get(f"/api/export/{aid}.npz").content)).files), lambda f: "pred_lc" in f)
t("export unknown 404", lambda: c.get("/api/export/zzz.npz").status_code, lambda s: s==404)
t("benchmarks", lambda: list(c.get("/api/benchmarks").json().keys()), lambda k: {"ablation","sim","robustness","calibration","real","laptop_bench"} <= set(k))
j=r.json()
t("risk shape heuristic+formula", lambda: j["stats"]["risk"]["flood"], lambda x: x["kind"]=="heuristic" and "formula" in x and "inputs" in x)
t("provenance", lambda: j["provenance"], lambda p: p["model_id"]=="terrafm" and p["data_source"]=="simulated")
t("bounds/gsd/cloud_mask", lambda: (j["bounds"], j["gsd_m"], j["pix"]["cloud_mask"] is not None), lambda x: x[0] is None and x[1]==30.0 and x[2])
t("model unknown 409", lambda: A(seed=1, model="nope").status_code, lambda s: s==409)
def _job():
    for _ in range(30):
        q=c.post("/api/analyze_async", json=dict(seed=7, size=64, cloud=0.4))
        if q.status_code!=429: break
        time.sleep(2)
    jid=q.json()["job_id"]
    ev=[json.loads(l[5:]) for l in c.get(f"/api/jobs/{jid}/events").text.splitlines() if l.startswith("data:")]
    return [e["stage"] for e in ev], c.get(f"/api/jobs/{jid}").json()["status"]
t("async job SSE stages", _job, lambda x: x[0] in (["queued","preparing","encoding","decoding","rendering","done"], ["queued","preparing","rendering","done"]) and x[1]=="done")  # 2nd form = cached result
ca=A(seed=7, size=64, flood=0.0).json()["id"]; cb=A(seed=7, size=64, flood=1.0).json()["id"]
t("change a->b", lambda: c.get(f"/api/change?a={ca}&b={cb}").json(), lambda d: d["stats"]["flood_onset_km2"]>0 and len(d["layers"])==4)
t("change size mismatch 409", lambda: c.get(f"/api/change?a={ca}&b={aid}").status_code, lambda s: s==409)
t("upload .tif (RS) or 415", lambda: c.post("/api/upload", files={"file":("x.tif",b"junk")}).status_code, lambda s: s in (400,415))
t("sample.npz", lambda: sorted(np.load(io.BytesIO(c.get("/api/sample.npz").content)).files), lambda f: f==["dem","optical","sar"])
t("sample.npz size=4096 (DoS?)", lambda: c.get("/api/sample.npz?size=4096", timeout=20).status_code, lambda s: s in (400,422))
t("sample.npz size=3", lambda: c.get("/api/sample.npz?size=3").status_code, lambda s: s in (400,422))
t("sample.npz flood=5", lambda: c.get("/api/sample.npz?flood=5").status_code, lambda s: s in (400,422))
# sensor ablation
for s in ([1,0,0],[0,1,0],[0,0,1],[0,1,1]):
    t(f"analyze sensors {s}", lambda s=s: A(seed=7,sensors=[bool(x) for x in s]).status_code, lambda x: x==200)
t("analyze no sensors 400", lambda: A(sensors=[False,False,False]).status_code, lambda s: s in (400,413))
t("analyze sensors len2 400", lambda: A(sensors=[True,True]).status_code, lambda s: s in (400,413))
t("analyze size=33 (not /4)", lambda: A(size=33).status_code, lambda s: s in (200,422))
t("analyze size=256", lambda: A(size=256).status_code, lambda s: s==200)
t("analyze size=300 422", lambda: A(size=300).status_code, lambda s: s==422)
t("analyze flood=2 422", lambda: A(flood=2).status_code, lambda s: s==422)
t("analyze seed=-1", lambda: A(seed=-1).status_code, lambda s: s in (400,422))
t("analyze source=bogus", lambda: A(source="bogus").status_code, lambda s: s in (400,422))
t("analyze upload missing id 404", lambda: A(source="upload", scene_id="x").status_code, lambda s: s==404)
# uploads
smp=c.get("/api/sample.npz").content
u=c.post("/api/upload", files={"file":("s.npz",smp)})
t("upload sample", lambda: u.json(), lambda j: "scene_id" in j)
t("analyze upload", lambda: A(source="upload", scene_id=u.json()["scene_id"]).json()["truth"], lambda x: x is False)
t("upload garbage 400", lambda: c.post("/api/upload", files={"file":("x.npz",b"hello")}).status_code, lambda s: s in (400,413))
b=io.BytesIO(); np.savez(b, optical=np.zeros((6,64,64)), sar=np.zeros((2,64,64))); 
t("upload missing dem 400", lambda: c.post("/api/upload", files={"file":("x.npz",b.getvalue())}).status_code, lambda s: s in (400,413))
b=io.BytesIO(); np.savez(b, optical=np.zeros((6,1000,1000),np.float32), sar=np.zeros((2,1000,1000)), dem=np.zeros((2,1000,1000)))
t("upload 1000px rejected", lambda: c.post("/api/upload", files={"file":("x.npz",b.getvalue())}).status_code, lambda s: s in (400,413))
b=io.BytesIO(); np.savez(b, optical=np.full((6,64,64),np.nan,np.float32), sar=np.zeros((2,64,64)), dem=np.zeros((2,64,64)))
up=c.post("/api/upload", files={"file":("x.npz",b.getvalue())})
t("upload NaN optical", lambda: (up.status_code, A(source="upload", scene_id=up.json().get("scene_id")).status_code if up.status_code==200 else None), lambda x: x[0]==400 or x[1]==200)
b=io.BytesIO(); np.savez(b, optical=np.array([object()],dtype=object))
t("upload pickled object npz (allow_pickle?)", lambda: c.post("/api/upload", files={"file":("x.npz",b.getvalue())}).status_code, lambda s: s in (400,413))
b=io.BytesIO(); np.savez(b, optical=np.zeros((6,64,64)), sar=np.zeros((2,64,64)), dem=np.zeros((2,64,64)), lc=np.full((64,64),99))
up=c.post("/api/upload", files={"file":("x.npz",b.getvalue())})
t("upload lc out of range", lambda: A(source="upload", scene_id=up.json()["scene_id"]).status_code, lambda s: s in (200,400))
# upload only lc without flood/veg
b=io.BytesIO(); np.savez(b, optical=np.zeros((6,64,64)), sar=np.zeros((2,64,64)), dem=np.zeros((2,64,64)), lc=np.zeros((64,64)))
up=c.post("/api/upload", files={"file":("x.npz",b.getvalue())})
t("upload lc only (no flood/veg)", lambda: A(source="upload", scene_id=up.json()["scene_id"]).status_code, lambda s: s in (200,400))
# results persist in the disk cache (LRU by bytes, 500 MB default) -> old ids still served after 16 more analyses
ids=[A(seed=i, size=64).json()["id"] for i in range(16)]
t("disk cache keeps older results", lambda: c.get(f"/api/layer/{aid}/rgb.png").status_code, lambda s: s==200)
# path traversal
t("static traversal", lambda: c.get("/static/../core/model.py").status_code, lambda s: s==404)
t("static traversal enc", lambda: c.get("/static/%2e%2e/core/model.py").status_code, lambda s: s==404)
t("docs exposed /docs", lambda: c.get("/docs").status_code, lambda s: True)
t("CORS header", lambda: c.get("/api/health", headers={"Origin":"http://evil"}).headers.get("access-control-allow-origin"), lambda h: True)
# timing
t0=time.time(); [A(seed=100+i) for i in range(5)]; 
t("latency 5x 128px", lambda: round((time.time()-t0)/5,3), lambda s: s<3)
t0=time.time(); A(seed=1,size=256)
t("latency 256px", lambda: round(time.time()-t0,3), lambda s: s<8)  # shared 2-core CI box; laptop target measured separately by bench_laptop
# rate limit: a burst well above 30/min from one IP must produce 429 (unless the server runs with TERRASCOPE_RATE_LIMIT=0)
t("rate limit burst -> 429", lambda: sorted({c.post("/api/analyze", json=dict(seed=1,size=32)).status_code for _ in range(40)}), lambda s: 429 in s)
print("\nSUMMARY", sum(o for _,o,_ in R), "/", len(R))
json.dump(R, open("api_results.json","w"), indent=1)
