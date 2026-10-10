import io, json, os, time, httpx, torch, torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms, models

app = FastAPI(title="Kisan Saathi")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
HERE = Path(__file__).parent
CONF_MIN = 0.70  # below this, don't pretend to know

# Edit with verified advice from ICAR/KVK/SKNAU docs (later: replace with RAG)
ADVICE = json.loads((HERE/"advice.json").read_text(encoding="utf-8")) if (HERE/"advice.json").exists() else {}

classes, model = [], None
if (HERE/"model.pt").exists() and (HERE/"classes.json").exists():
    classes = json.loads((HERE/"classes.json").read_text(encoding="utf-8"))
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(torch.load(HERE/"model.pt", map_location="cpu")); model.eval()

tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@app.get("/api/health")
def health(): return {"ok": True, "model_loaded": model is not None}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model abhi train nahi hua. ml/train.py chalayein.")
    try: img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception: raise HTTPException(400, "Photo samajh nahi aayi, dobara kheenchein.")
    with torch.no_grad():
        pr = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
    top = pr.topk(min(3, len(classes)))
    res = [{"disease": classes[i], "confidence": round(float(c),3)} for c,i in zip(top.values, top.indices)]
    best = res[0]
    return {"results": res, "confident": best["confidence"] >= CONF_MIN,
            "advice": ADVICE.get(best["disease"], "Najdiki KVK ya krishi adhikari se salah lein.")}

# ---- Mandi prices (data.gov.in), 30-min cache ----
_cache = {}
RES = "9ef84268-d588-465a-a308-a864a43d0070"
@app.get("/api/mandi")
async def mandi(commodity: str = "Wheat", state: str = "Rajasthan"):
    key = (commodity, state)
    if key in _cache and time.time() - _cache[key][0] < 1800: return _cache[key][1]
    k = os.getenv("DATA_GOV_API_KEY")
    if not k: raise HTTPException(503, "DATA_GOV_API_KEY set karein (data.gov.in se free).")
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"https://api.data.gov.in/resource/{RES}", params={
            "api-key": k, "format": "json", "limit": 100,
            "filters[state.keyword]": state, "filters[commodity]": commodity})
    rows = [{"market": x.get("market"), "district": x.get("district"), "date": x.get("arrival_date"),
             "min": x.get("min_price"), "max": x.get("max_price"), "modal": x.get("modal_price")}
            for x in r.json().get("records", [])]
    rows.sort(key=lambda x: float(x["modal"] or 0), reverse=True)
    out = {"commodity": commodity, "rows": rows}; _cache[key] = (time.time(), out); return out

app.mount("/", StaticFiles(directory=HERE.parent/"frontend", html=True), name="web")
