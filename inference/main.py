from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pydantic import BaseModel
from aiortc import (
    RTCDataChannel,
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
import time
import json
import cv2

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task[NoReturn]] = []

    channel_manager = DataChannelManager()
    app.state.channel_manager = channel_manager

    for _ in range(3):
        worker_task = asyncio.create_task(inference_worker(channel_manager))
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

inference_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=3)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: dict[str, RTCPeerConnection] = {}

model = YOLO("yolov8n.pt")

@dataclass
class Prediction:
    boxes: list[list[float]]
    labels: list[str]
    confs: list[float]

class DataChannelManager:
    def __init__(self):
        self.channels: dict[str, RTCDataChannel] = {}

    def send_prediction(self, prediction: Prediction):
        for pc_id, channel in self.channels.items():
            if channel.readyState == "open":
                channel.send(json.dumps(prediction.__dict__))
            else:
                logger.warning(f"Channel for PC {pc_id} is not open")

    def add_channel(self, pc_id: str, channel: RTCDataChannel):
        self.channels[pc_id] = channel
        logger.info(f"Data channel added for PC {pc_id}")

    def remove_channel(self, pc_id: str):
        channel = self.channels.pop(pc_id, None)
        if channel:
            logger.info(f"Data channel removed for PC {pc_id}")
        else:
            logger.warning(f"No data channel found for PC {pc_id} to remove")


async def frame_producer(track: VideoStreamTrack):
    """Receive frames and put them into the inference queue (non-blocking)."""
    while True:
        frame: VideoFrame = await track.recv()  # pyright: ignore[reportAssignmentType]
        if inference_queue.full():
            logger.warning("Inference queue is full, dropping frame")
            _ = inference_queue.get_nowait()

        await inference_queue.put(frame)


async def inference_worker(channel_manager: DataChannelManager):
    """Continuously read frames from queue and run ML inference."""
    while True:
        frame = await inference_queue.get()
        try:
            img = frame.to_ndarray(format="bgr24")
            img_small = cv2.resize(img, (640, 360))

            start = time.process_time()
            results = model.predict(img_small, verbose=False)  # pyright: ignore[reportUnknownMemberType]
            elapsed = time.process_time() - start
            logger.info(f"Elapsed time for inference: {elapsed * 1000:.2f} ms")

            boxes = results[0].boxes

            if not boxes:
                continue

            bxs: list[list[float]] = boxes.xyxy.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            labels: list[float] = boxes.cls.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            confs: list[float] = boxes.conf.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

            label_names = [model.names[int(lbl)] for lbl in labels]
            confs = [round(float(conf), 3) for conf in confs]

            logger.info(
                f"Detected {len(bxs)} objects in frame, labels: {label_names}, confs: {confs}"
            )

            prediction = Prediction(boxes=bxs, labels=label_names, confs=confs)

            channel_manager.send_prediction(prediction)

        except Exception as e:
            logger.error("Inference error:", e)
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

    pc_id = str(uuid.uuid4())
    pcs[pc_id] = pc

    logger.info(f"Peer connection created: {pc_id}")

    @pc.on("track")
    def on_track(track: VideoStreamTrack):
        logger.info(f"PC {pc_id} Track received: kind={track.kind}")

        if track.kind == "video":
            _ = asyncio.create_task(frame_producer(track))

        @track.on("ended")
        async def on_ended():
            logger.info(f"PC {pc_id} Track ended ({track.kind})")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"PC {pc_id} connection state: {pc.connectionState}")

        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            _ = pcs.pop(pc_id, None)

            channel_manager: DataChannelManager = request.app.state.channel_manager  # pyright: ignore[reportAny]
            channel_manager.remove_channel(pc_id)

            logger.info(f"PC {pc_id} closed and cleaned up")

    @pc.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info(f"PC {pc_id} Data channel established: {channel.label}")
        channel_manager: DataChannelManager = request.app.state.channel_manager  # pyright: ignore[reportAny]
        channel_manager.add_channel(pc_id, channel)

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SDP: {e}")


    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"PC {pc_id} answered")

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

