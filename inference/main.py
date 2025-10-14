from typing_extensions import override
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
import time

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task[NoReturn]] = []

    track_manager = TrackManager()
    app.state.track_manager = track_manager

    for _ in range(3):
        worker_task = asyncio.create_task(inference_worker(track_manager))
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

inference_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=30)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: dict[str, RTCPeerConnection] = {}

model = YOLO("yolov8n.pt")


class TrackManager:
    """Manages video tracks without bottlenecks."""

    def __init__(self):
        self.active_tracks: list["ProcessedVideoTrack"] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    async def add_track(self, track: "ProcessedVideoTrack"):
        """Add a track to receive frames."""
        async with self._lock:
            self.active_tracks.append(track)

    async def remove_track(self, track: "ProcessedVideoTrack"):
        """Remove a track."""
        async with self._lock:
            if track in self.active_tracks:
                self.active_tracks.remove(track)

    async def remove_tracks_by_pc_id(self, pc_id: str):
        """Remove all tracks associated with a peer connection."""
        async with self._lock:
            self.active_tracks = [
                track for track in self.active_tracks if track.pc_id != pc_id
            ]

    def broadcast_frame(self, frame: VideoFrame):
        """Directly feed frame to all active tracks (non-blocking)."""
        for track in self.active_tracks[
            :
        ]:  # Copy to avoid modification during iteration
            try:
                track.put_frame_nowait(frame)
            except Exception as e:
                logger.error(f"Error feeding frame to track {track.pc_id}: {e}")

    def get_stats(self):
        """Get track statistics."""
        return {
            "active_tracks": len(self.active_tracks),
            "track_details": [
                {
                    "pc_id": track.pc_id,
                    "kind": track.kind,
                    "queue_size": track.frame_queue.qsize(),
                }
                for track in self.active_tracks
            ],
        }


class ProcessedVideoTrack(VideoStreamTrack):
    """
    Custom video track with direct frame feeding.
    """

    def __init__(self, pc_id: str, track_manager: TrackManager):
        super().__init__()
        self.kind: str = "video"
        self.pc_id: str = pc_id
        self.track_manager: TrackManager = track_manager
        self.frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=10)

        _ = asyncio.create_task(self.track_manager.add_track(self))

    def put_frame_nowait(self, frame: VideoFrame):
        """Put a frame into this track's queue (non-blocking)."""
        try:
            self.frame_queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Frame queue is full")
            try:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass

    @override
    async def recv(self):
        """
        Called by WebRTC to get the next video frame.
        """
        try:
            # Get frame from this track's dedicated queue
            frame = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

            # Update frame timestamp and presentation time
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base

            return frame

        except asyncio.TimeoutError:
            # If no frame available, create a black frame to maintain stream
            frame = VideoFrame.from_ndarray(
                cv2.zeros((360, 640, 3), dtype="uint8"), format="bgr24"
            )
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            return frame

        except asyncio.CancelledError:
            # Remove this track from active tracks
            await self.track_manager.remove_track(self)
            raise

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            # Return a black frame as fallback
            frame = VideoFrame.from_ndarray(
                cv2.zeros((360, 640, 3), dtype="uint8"), format="bgr24"
            )
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
            return frame


async def frame_producer(track: VideoStreamTrack):
    """Receive frames and put them into the inference queue (non-blocking)."""
    while True:
        frame: VideoFrame = await track.recv()  # pyright: ignore[reportAssignmentType]
        if inference_queue.full():
            logger.warning("Inference queue is full, dropping frame")
            continue

        await inference_queue.put(frame)


async def inference_worker(track_manager: TrackManager):
    """Continuously read frames from queue and run ML inference."""
    while True:
        frame = await inference_queue.get()
        try:
            start = time.process_time()
            img = frame.to_ndarray(format="bgr24")
            img_small = cv2.resize(img, (640, 360))
            elapsed = time.process_time() - start
            logger.info(f"Elapsed time for preprocessing: {elapsed * 1000:.2f} ms")

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

            logger.info(
                f"Detected {len(bxs)} objects in frame, labels: {label_names}, confs: {confs}"
            )

            # Directly feed frame to all tracks (for now just one)
            track_manager.broadcast_frame(frame)

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
    pc_id = str(uuid.uuid4())[:8]
    pcs[pc_id] = pc

    logger.info("Created for", pc_id)

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

            # Clean up any tracks associated with this PC
            track_manager: TrackManager = request.app.state.track_manager  # pyright: ignore[reportAny]
            await track_manager.remove_tracks_by_pc_id(pc_id)

            logger.info(f"PC {pc_id} closed and cleaned up")

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SDP: {e}")

    track_manager: TrackManager = request.app.state.track_manager  # pyright: ignore[reportAny]
    processed_track = ProcessedVideoTrack(pc_id, track_manager)

    _ = pc.addTrack(processed_track)
    logger.info(f"PC {pc_id} Added processed video track")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"PC {pc_id} answered")

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/health")
async def health(request: Request):
    track_manager: TrackManager = request.app.state.track_manager  # pyright: ignore[reportAny]
    track_stats = track_manager.get_stats()

    return {
        "status": "ok",
        "pcs": len(pcs),
        "inference_queue_size": inference_queue.qsize(),
        **track_stats,
    }
