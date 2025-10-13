import asyncio
from contextlib import asynccontextmanager
import os
import uuid
from datetime import datetime
from typing import Dict
import logging
import cv2

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription

from ultralytics import YOLO

# aiortc video frame handling relies on av
# pip install av to get .to_image()

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = []

    for _ in range(3):  # dummy loop to use 'yield' for lifespan
        worker_task = asyncio.create_task(inference_worker())
        tasks.append(worker_task)
    logger.info("Inference workers started")

    yield

    try:
        for worker_task in tasks:
            worker_task.cancel()
            await worker_task
    except asyncio.CancelledError:
        pass

    logger.info("Inference workers stopped")

app = FastAPI(lifespan=lifespan)

inference_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of peer connections so we can close them later if needed
pcs: Dict[str, RTCPeerConnection] = {}

FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

model = YOLO("yolov8n.pt") 

class Offer(BaseModel):
    sdp: str
    type: str

async def frame_producer(track):
    """Receive frames and put them into the inference queue (non-blocking)."""
    while True:
        frame = await track.recv()
        if inference_queue.full():
            # Drop frame if queue is full (avoid lag)
            continue
        await inference_queue.put(frame)


async def inference_worker():
    """Continuously read frames from queue and run ML inference."""
    while True:
        frame = await inference_queue.get()
        try:
            img = frame.to_ndarray(format="bgr24")

            # ↓ Resize/downsample if needed
            img_small = cv2.resize(img, (640, 360))

            # ↓ Run model inference
            results = model.predict(img_small, verbose=False)

            # ↓ Process results (e.g. print or store)
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist() # type: ignore
            labels = results[0].boxes.cls.cpu().numpy().tolist() # type: ignore
            confs = results[0].boxes.conf.cpu().numpy().tolist() # type: ignore

            label_names = [model.names[int(lbl)] for lbl in labels]

            print(f"Detected {len(boxes)} objects in frame, labels: {label_names}, confs: {confs}")

            # Example: you could push results to a WebSocket or DB here
        except Exception as e:
            print("Inference error:", e)
        finally:
            inference_queue.task_done()


async def save_video_frames(pc_id: str, track):
    """
    Consume frames from the incoming video track and save them as JPEGs.
    Replace the body of this function with any processing you need.
    """
    counter = 0
    try:
        while True:
            frame = await track.recv()  # av.VideoFrame
            # Convert to PIL Image (aiortc uses av VideoFrame)
            try:
                img = frame.to_image()
            except Exception:
                # fallback: convert to ndarray then to PIL
                arr = frame.to_ndarray(format="bgr24")
                # local import to avoid mandatory pillow if not used
                from PIL import Image
                img = Image.fromarray(arr[..., ::-1])  # bgr -> rgb

            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%f")[:-3]
            filename = os.path.join(FRAME_DIR, f"{pc_id}_{timestamp}_{counter:06d}.jpg")
            # save (this may be slow for high FPS, replace with in-memory queue or faster processing)

            logger.info(f"Got frame {filename}")

            img.save(filename, quality=85)
            counter += 1

    except asyncio.CancelledError:
        # task cancelled while shutting down
        return
    except Exception as e:
        print("Frame consumer error:", e)


@app.post("/offer")
async def offer(offer: Offer, request: Request):
    """
    Accept a browser offer (SDP), create an answer, and consume incoming video frames.
    Body JSON: { "sdp": "<offer_sdp>", "type": "offer" }
    Response: { "sdp": "<answer_sdp>", "type": "answer" }
    """
    pc = RTCPeerConnection()
    pc_id = str(uuid.uuid4())[:8]
    pcs[pc_id] = pc
    print("Created for", pc_id)

    frame_tasks = []

    @pc.on("track")
    def on_track(track):
        print(f"PC {pc_id} Track received: kind={track.kind}")
        if track.kind == "video":
            # create a task to consume all incoming frames
            # task = asyncio.create_task(save_video_frames(pc_id, track))
            # frame_tasks.append(task)

            asyncio.create_task(frame_producer(track))

        @track.on("ended")
        async def on_ended():
            print(f"PC {pc_id} Track ended ({track.kind})")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"PC {pc_id} connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            # cleanup
            for t in frame_tasks:
                t.cancel()
            await pc.close()
            pcs.pop(pc_id, None)
            print(f"PC {pc_id} closed and cleaned up")

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SDP: {e}")

    # We do not create any outgoing tracks in this example (server is receive-only),
    # but we still need to create an answer.
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print(f"PC {pc_id} answered")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/health")
async def health():
    return {"status": "ok", "pcs": len(pcs)}
