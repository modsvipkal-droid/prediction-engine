import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from config import settings
from database import db_manager
from model import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

training_lock = asyncio.Lock()


async def train():
    async with training_lock:
        logger.info("Exploring database...")
        try:
            docs = await db_manager.fetch_all()
            stats = await db_manager.get_stats()
            hourly = await db_manager.analyze_by_hour()
            streaks = await db_manager.analyze_streaks()
            logger.info(
                "Total=%d Red=%d Green=%d | Training model...",
                stats["total"], stats["red"], stats["green"],
            )
            engine.train(docs, hourly, streaks)
            logger.info(
                "Training done — accuracy=%.4f loss=%.4f samples=%d",
                engine.accuracy, engine.loss, engine.samples_trained,
            )
        except Exception as e:
            logger.error("Training failed: %s", e)


async def background_loop():
    last_count = await db_manager.get_count()
    logger.info("Initial records: %d", last_count)
    while True:
        try:
            current = await db_manager.get_count()
            if current != last_count:
                logger.info("Change detected: %d → %d. Retraining...", last_count, current)
                await train()
                last_count = current
        except Exception as e:
            logger.error("Bg error: %s", e)
        await asyncio.sleep(settings.TRAIN_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to MongoDB...")
    await db_manager.connect()
    logger.info("Connected. Running initial training...")
    await train()
    bg = asyncio.create_task(background_loop())
    yield
    bg.cancel()
    await db_manager.disconnect()
    logger.info("Shutdown.")


app = FastAPI(title="AI Prediction Engine", lifespan=lifespan)


@app.get("/predict")
async def predict():
    return engine.latest_prediction


@app.get("/auto-predict")
async def auto_predict(request: Request):
    async def event_stream():
        last_period = ""
        while True:
            try:
                pred = engine.latest_prediction
                current = pred.get("current_period", "")
                if current and current != last_period:
                    last_period = current
                    yield f"data: {json.dumps(pred)}\n\n"
            except:
                pass
            await asyncio.sleep(5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@app.get("/dashboard")
async def dashboard():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Prediction Dashboard</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
            .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 10px 0; }
            .green { color: #3fb950; }
            .red { color: #f85149; }
            .big { font-size: 2em; font-weight: bold; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>AI Prediction Engine</h1>
        <div id="content">Loading...</div>
        <script>
        async function load() {
            const r = await fetch('/predict');
            const d = await r.json();
            const html = `
            <div class="card">
                <div class="grid">
                    <div><b>Current Period:</b> ${d.current_period || '-'}</div>
                    <div><b>Next Period:</b> ${d.next_period || '-'}</div>
                </div>
                <h2>Prediction: <span class="${d.prediction === 'Green' ? 'green' : 'red'}">${d.prediction || '-'}</span></h2>
                <div class="grid">
                    <div><b>Confidence:</b> ${(d.confidence * 100).toFixed(1)}%</div>
                    <div><b>Accuracy:</b> ${(d.accuracy * 100).toFixed(1)}%</div>
                </div>
                <div class="grid">
                    <div><span class="badge" style="background:#3fb95033;color:#3fb950">Green ${(d.probabilities?.Green * 100 || 0).toFixed(0)}%</span></div>
                    <div><span class="badge" style="background:#f8514933;color:#f85149">Red ${(d.probabilities?.Red * 100 || 0).toFixed(0)}%</span></div>
                </div>
                <p><b>Hour Pattern:</b> ${d.hourly_pattern || '-'}</p>
                <p><b>Strategy:</b></p>
                <ul>${(d.strategy || []).map(s => '<li>' + s + '</li>').join('')}</ul>
            </div>
            <div class="card">
                <div class="grid">
                    <div><b>Total Records:</b> ${d.total_data || 0}</div>
                    <div><b>Samples Trained:</b> ${d.samples_trained || 0}</div>
                    <div><span style="color:#f85149">Red: ${d.total_red || 0}</span></div>
                    <div><span style="color:#3fb950">Green: ${d.total_green || 0}</span></div>
                </div>
            </div>
            `;
            document.getElementById('content').innerHTML = html;
        }
        load();
        setInterval(load, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
