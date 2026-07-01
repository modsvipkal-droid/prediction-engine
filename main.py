import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from config import settings
from database import db_manager
from model import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

training_lock = asyncio.Lock()
startup_ok = False


async def train():
    async with training_lock:
        logger.info("Training...")
        try:
            docs = await db_manager.fetch_all()
            stats = await db_manager.get_stats()
            hourly = await db_manager.analyze_by_hour()
            streaks = await db_manager.analyze_streaks()
            logger.info("Data: %d docs, %d red, %d green", stats["total"], stats["red"], stats["green"])
            engine.train(docs, hourly, streaks)
            logger.info("Trained — acc=%.4f loss=%.4f samples=%d", engine.accuracy, engine.loss, engine.samples_trained)
        except Exception as e:
            logger.error("Train failed: %s", e)


async def background_loop():
    global startup_ok
    while True:
        try:
            if not db_manager.is_connected:
                logger.warning("DB not connected, reconnecting...")
                try:
                    await db_manager.connect()
                except Exception:
                    await asyncio.sleep(10)
                    continue
            current = await db_manager.get_count()
            if current != background_loop.last_count:
                logger.info("New data: %d → %d", background_loop.last_count, current)
                await train()
                background_loop.last_count = current
            if not startup_ok:
                startup_ok = True
        except Exception as e:
            logger.error("Bg error: %s", e)
        await asyncio.sleep(settings.TRAIN_INTERVAL_SECONDS)


background_loop.last_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db_manager.connect()
        background_loop.last_count = await db_manager.get_count()
        logger.info("Connected, records: %d", background_loop.last_count)
    except Exception as e:
        logger.warning("Initial DB connect failed: %s — will retry in background", e)
    await train()
    bg = asyncio.create_task(background_loop())
    yield
    bg.cancel()
    await db_manager.disconnect()


app = FastAPI(title="AI Prediction Engine", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok", "app": "AI Prediction Engine", "docs": "/dashboard"}


@app.get("/predict")
async def predict():
    return engine.latest_prediction


@app.get("/model-info")
async def model_info():
    return {
        "accuracy": engine.accuracy,
        "loss": engine.loss,
        "samples_trained": engine.samples_trained,
        "total_data": engine.total_data,
        "total_red": engine.total_red,
        "total_green": engine.total_green,
        "strategy": engine.strategy_notes,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if db_manager.is_connected else "degraded",
        "db_connected": db_manager.is_connected,
        "total_data": engine.total_data,
        "samples_trained": engine.samples_trained,
    }


@app.get("/auto-predict")
async def auto_predict(request: Request):
    async def event_stream():
        last = ""
        while True:
            try:
                pred = engine.latest_prediction
                cur = pred.get("current_period", "")
                if cur and cur != last:
                    last = cur
                    yield f"data: {json.dumps(pred)}\n\n"
            except Exception:
                pass
            await asyncio.sleep(5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/dashboard")
async def dashboard():
    html = """<!DOCTYPE html>
<html><head><title>AI Prediction Dashboard</title>
<meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:20px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:10px 0}
.green{color:#3fb950}.red{color:#f85149}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.badge{display:inline-block;padding:4px 12px;border-radius:12px;font-size:.9em;margin:2px}
</style></head><body>
<h1>AI Prediction Engine</h1>
<div id="content"><p>Loading...</p></div>
<script>
async function load(){try{
const r=await fetch('/predict'),d=await r.json();
document.getElementById('content').innerHTML=`
<div class="card">
<div class="grid">
<div><b>Current Period:</b> ${d.current_period||'-'}</div>
<div><b>Next Period:</b> ${d.next_period||'-'}</div>
</div>
<h2>Prediction: <span class="${d.prediction==='Green'?'green':'red'}">${d.prediction||'-'}</span></h2>
<div class="grid">
<div><b>Confidence:</b> ${(d.confidence*100).toFixed(1)}%</div>
<div><b>Accuracy:</b> ${(d.accuracy*100).toFixed(1)}%</div>
</div>
<div>
<span class="badge" style="background:#3fb95033;color:#3fb950">Green ${(d.probabilities?.Green*100||0).toFixed(0)}%</span>
<span class="badge" style="background:#f8514933;color:#f85149">Red ${(d.probabilities?.Red*100||0).toFixed(0)}%</span>
</div>
<p><b>Hour Pattern:</b> ${d.hourly_pattern||'-'}</p>
<p><b>Strategy:</b></p><ul>${(d.strategy||[]).map(s=>'<li>'+s+'</li>').join('')}</ul>
</div>
<div class="card">
<div class="grid">
<div><b>Total Records:</b> ${d.total_data||0}</div>
<div><b>Trained:</b> ${d.samples_trained||0}</div>
<div><span style="color:#f85149">Red: ${d.total_red||0}</span></div>
<div><span style="color:#3fb950">Green: ${d.total_green||0}</span></div>
</div></div>`}catch(e){document.getElementById('content').innerHTML='<p>Error loading</p>'}}
load();setInterval(load,5000);
</script></body></html>"""
    return HTMLResponse(content=html)
