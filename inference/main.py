from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av.video.frame import VideoFrame
from ultralytics.models import YOLO
from typing import NoReturn

import asyncio
import uuid
import logging
import cv2

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task[NoReturn]] = []

    for _ in range(3):
        worker_task = asyncio.create_task(inference_worker())
        tasks.append(worker_task)

    logger.info("Inference workers started")

    yield

    try:
        for worker_task in tasks:
            _ = worker_task.cancel()
            await worker_task
    except asyncio.CancelledError:
        pass

    logger.info("Inference workers stopped")


app = FastAPI(lifespan=lifespan)

inference_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=10)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: dict[str, RTCPeerConnection] = {}

model = YOLO("yolov8n.pt")


async def frame_producer(track: VideoStreamTrack):
    """Receive frames and put them into the inference queue (non-blocking)."""
    while True:
        frame: VideoFrame = await track.recv()  # pyright: ignore[reportAssignmentType]
        if inference_queue.full():
            logger.warning("Inference queue is full, dropping frame")
            continue

        await inference_queue.put(frame)


async def inference_worker():
    """Continuously read frames from queue and run ML inference."""
    while True:
        frame = await inference_queue.get()
        try:
            img = frame.to_ndarray(format="bgr24")
            img_small = cv2.resize(img, (640, 360))

            results = model.predict(img_small, verbose=False)  # pyright: ignore[reportUnknownMemberType]

            boxes = results[0].boxes

            if not boxes:
                continue

            bxs: list[list[float]] = boxes.xyxy.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            labels: list[float] = boxes.cls.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            confs: list[float] = boxes.conf.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

            label_names = [model.names[int(lbl)] for lbl in labels]

            print(
                f"Detected {len(bxs)} objects in frame, labels: {label_names}, confs: {confs}"
            )

        except Exception as e:
            print("Inference error:", e)
        finally:
            inference_queue.task_done()


class Offer(BaseModel):
    sdp: str
    type: str


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

    @pc.on("track")
    def on_track(track: VideoStreamTrack):
        print(f"PC {pc_id} Track received: kind={track.kind}")
        if track.kind == "video":
            _ = asyncio.create_task(frame_producer(track))

        @track.on("ended")
        async def on_ended():
            print(f"PC {pc_id} Track ended ({track.kind})")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"PC {pc_id} connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            _ = pcs.pop(pc_id, None)
            print(f"PC {pc_id} closed and cleaned up")

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
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
