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

    # Start frame broadcaster
    broadcaster_task = asyncio.create_task(frame_broadcaster())
    tasks.append(broadcaster_task)

    logger.info("Inference workers and frame broadcaster started")

    yield

    try:
        for worker_task in tasks:
            _ = worker_task.cancel()
            await worker_task
    except asyncio.CancelledError:
        pass

    logger.info("Inference workers and frame broadcaster stopped")


app = FastAPI(lifespan=lifespan)

inference_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=10)
outgoing_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=10)

# Shared frame for broadcasting to all tracks
latest_frame: VideoFrame | None = None
frame_ready = asyncio.Event()
active_tracks: list["ProcessedVideoTrack"] = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: dict[str, RTCPeerConnection] = {}

model = YOLO("yolov8n.pt")


class ProcessedVideoTrack(VideoStreamTrack):
    """
    Custom video track that sends processed frames using broadcast mechanism.
    """

    def __init__(self, pc_id: str):
        super().__init__()
        self.kind = "video"
        self.last_frame_time = 0
        self.pc_id = pc_id
        active_tracks.append(self)

    async def recv(self):
        """
        Called by WebRTC to get the next video frame.
        Returns the latest processed frame.
        """
        global latest_frame, frame_ready

        try:
            # Wait for a new frame to be available
            await asyncio.wait_for(frame_ready.wait(), timeout=1.0)

            if latest_frame is not None:
                # Create a copy of the frame for this track
                frame = latest_frame

                # Update frame timestamp and presentation time
                pts, time_base = await self.next_timestamp()
                frame.pts = pts
                frame.time_base = time_base

                return frame
            else:
                raise Exception("No frame available")

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
            if self in active_tracks:
                active_tracks.remove(self)
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


async def frame_broadcaster():
    """Broadcast frames from outgoing_queue to all active tracks."""
    global latest_frame, frame_ready

    while True:
        try:
            # Get processed frame from outgoing queue
            frame = await outgoing_queue.get()

            # Update the latest frame for all tracks
            latest_frame = frame
            frame_ready.set()

            # Small delay to allow tracks to consume the frame
            await asyncio.sleep(0.01)
            frame_ready.clear()

            outgoing_queue.task_done()

        except Exception as e:
            logger.error(f"Error in frame broadcaster: {e}")
            await asyncio.sleep(0.1)


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

            outgoing_queue.put_nowait(frame)

            logger.info("Sent frame to outgoing queue")

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

            # Clean up any tracks associated with this PC
            global active_tracks
            active_tracks = [track for track in active_tracks if track.pc_id != pc_id]

            print(f"PC {pc_id} closed and cleaned up")

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid SDP: {e}")

    # Add processed video track to send frames back
    processed_track = ProcessedVideoTrack(pc_id)
    pc.addTrack(processed_track)
    print(f"PC {pc_id} Added processed video track")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print(f"PC {pc_id} answered")

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pcs": len(pcs),
        "inference_queue_size": inference_queue.qsize(),
        "outgoing_queue_size": outgoing_queue.qsize(),
        "active_tracks": len(active_tracks),
        "latest_frame_available": latest_frame is not None,
        "track_details": [
            {
                "pc_id": track.pc_id,
                "kind": track.kind,
            }
            for track in active_tracks
        ],
    }
