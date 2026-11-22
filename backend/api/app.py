import os
import sys
import io
import time
import logging
import traceback
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any

from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.integration.pipeline import Pipeline, MAX_SIDE

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Run command documentation:
# & "C:\Users\ektac\OneDrive\Documents\Projects\College\NaviSense\.venv\Scripts\python.exe" -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads AI models once at application startup into app.state.
    Single global tracker is maintained across requests.
    Architectural limitation note: Designed for single-user phone guidance demo;
    concurrent users sharing one server would share tracker history.
    """
    logger.info("Server starting: loading Pipeline models into memory...")
    app.state.pipeline = Pipeline()
    # Concurrency lock: non-blocking frame drop for real-time mobile streaming
    app.state.pipeline_lock = threading.Lock()
    logger.info("Server startup complete. Models loaded and ready.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="NaviSense Backend API",
    description="FastAPI service for real-time indoor obstacle detection, depth estimation, and spoken advice.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration: Allow all origins so a phone on the local Wi-Fi can communicate
# freely with the laptop backend during the live demo (Demo only; restrict in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    """
    Health check endpoint confirming server responsiveness and model readiness.
    """
    models_ready = hasattr(request.app.state, "pipeline") and request.app.state.pipeline is not None
    return {
        "status": "ok",
        "models_loaded": bool(models_ready),
    }


@app.post("/reset")
def reset_tracker(request: Request) -> Dict[str, Any]:
    """
    Resets persistent tracker history and restarts track IDs at 1.
    """
    pipeline: Pipeline = request.app.state.pipeline
    with request.app.state.pipeline_lock:
        pipeline.reset()
    logger.info("POST /reset - Tracker state successfully cleared.")
    return {"status": "reset"}


@app.post("/analyze-frame")
def analyze_frame(
    request: Request,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Primary inference endpoint: accepts a single image frame (JPEG/PNG) from phone browser.
    Defined as a standard sync `def` function so FastAPI runs it inside the background threadpool,
    preventing heavy CPU inference from blocking the async event loop.
    """
    req_arrival_time = time.time()
    req_start_perf = time.perf_counter()

    # 1. Non-blocking concurrency check:
    # If a frame is currently being processed by the laptop CPU, drop new incoming frames immediately.
    # The phone camera streams continuously; skipping busy frames maintains real-time responsiveness.
    acquired = request.app.state.pipeline_lock.acquire(blocking=False)
    if not acquired:
        logger.info("POST /analyze-frame: Server busy with previous frame. Dropping current frame.")
        return {"status": "busy"}

    try:
        # 2. Read and validate uploaded file
        try:
            file_bytes = file.file.read()
            if not file_bytes:
                raise ValueError("Uploaded file is empty.")
            raw_image = Image.open(io.BytesIO(file_bytes))
            # Correct phone orientation from EXIF metadata (essential for phone cameras)
            raw_image = ImageOps.exif_transpose(raw_image)
            image = raw_image.convert("RGB")
        except Exception as exc:
            logger.warning(f"Invalid image upload: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or unreadable image upload: {str(exc)}",
            )

        # 3. Resize image so longest side is at most MAX_SIDE (640 px)
        # Guarantees sub-second CPU latency; all output coordinates refer to this resized frame.
        orig_w, orig_h = image.size
        scale = min(1.0, float(MAX_SIDE) / max(orig_w, orig_h))
        target_w = int(round(orig_w * scale))
        target_h = int(round(orig_h * scale))
        if scale < 1.0:
            resized_image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        else:
            resized_image = image

        # 4. Execute AI pipeline
        pipeline: Pipeline = request.app.state.pipeline
        pipeline_output = pipeline.process(resized_image, timestamp=req_arrival_time)

        total_req_ms = (time.perf_counter() - req_start_perf) * 1000.0

        # One-line structured request log with stage timings
        t_ms = pipeline_output.get("timing_ms", {})
        logger.info(
            f"POST /analyze-frame | {target_w}x{target_h} | "
            f"Objects: {len(pipeline_output['objects'])} | "
            f"Detect: {t_ms.get('detect', 0)}ms | "
            f"Track: {t_ms.get('track', 0)}ms | "
            f"Depth: {t_ms.get('depth', 0)}ms | "
            f"Risk: {t_ms.get('risk', 0)}ms | "
            f"Total: {round(total_req_ms, 1)}ms | "
            f"Advice: \"{pipeline_output['advice']}\""
        )

        return {
            "status": "ok",
            "advice": pipeline_output["advice"],
            "top_threat": pipeline_output["top_threat"],
            "objects": pipeline_output["objects"],
            "path_clear": pipeline_output["path_clear"],
            "image": pipeline_output["image"],
            "timing_ms": pipeline_output["timing_ms"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        tb_str = traceback.format_exc()
        logger.error(f"Internal pipeline error during /analyze-frame:\n{tb_str}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline processing failure: {str(exc)}",
        )
    finally:
        request.app.state.pipeline_lock.release()


# Mount frontend static files read-only if frontend/index.html exists.
# This allows the phone browser to load the web interface and speak advice from the exact same origin.
frontend_index = os.path.join("frontend", "index.html")
if os.path.isfile(frontend_index):
    logger.info("Found 'frontend/index.html'. Mounting 'frontend/' at '/' as static web root.")
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    logger.info("No 'frontend/index.html' detected; skipping static frontend mount.")