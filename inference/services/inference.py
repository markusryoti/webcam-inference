import asyncio
import logging
import time

import cv2
from aiortc import (
    VideoStreamTrack,
)
from av.video.frame import VideoFrame
from ultralytics.models import YOLO

from services.aggregator import Aggregator
from services.datachannel import DataChannelManager
from shared.prediction import Prediction

logger = logging.getLogger("uvicorn")

model = YOLO("yolov8n.pt")

inference_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=3)


async def frame_producer(track: VideoStreamTrack):
    """Receive frames and put them into the inference queue (non-blocking)."""
    while True:
        try:
            frame: VideoFrame = await track.recv()  # pyright: ignore[reportAssignmentType]
            if inference_queue.full():
                logger.warning("Inference queue is full, dropping frame")
                _ = inference_queue.get_nowait()

            await inference_queue.put(frame)
        except Exception as e:
            logger.warning(f"Exception in frame producer: {e}, track might be closed")
            return


async def inference_worker(
    channel_manager: DataChannelManager,
    aggregator: Aggregator,
):
    """Continuously read frames from queue and run ML inference."""
    while True:
        frame = await inference_queue.get()
        try:
            prediction = await asyncio.to_thread(process_frame, frame)
            if prediction:
                channel_manager.send_prediction(prediction)
                aggregator.add_prediction(prediction)
        except Exception as e:
            logger.error("Inference error:", e)
        finally:
            inference_queue.task_done()


def process_frame(frame: VideoFrame) -> Prediction | None:
    img = frame.to_ndarray(format="bgr24")
    img_small = cv2.resize(img, (640, 360))

    start = time.process_time()
    results = model.track(img_small, persist=True)
    elapsed = time.process_time() - start
    logger.info(f"Elapsed time for inference: {elapsed * 1000:.2f} ms")

    boxes = results[0].boxes

    if not boxes:
        return None

    tracking_ids = boxes.id  # pyright: ignore[reportUnknownMemberType]
    if tracking_ids is None:
        logger.warning("No tracking IDs found, skipping frame")
        return None

    tracking_ids = tracking_ids.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    bxs: list[list[float]] = boxes.xyxy.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    labels: list[float] = boxes.cls.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    confs: list[float] = boxes.conf.cpu().numpy().tolist()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]

    label_names = [model.names[int(lbl)] for lbl in labels]
    confs = [round(float(conf), 3) for conf in confs]

    logger.info(
        f"Detected {len(bxs)} objects in frame, labels: {label_names}, confs: {confs}"
    )

    prediction = Prediction(
        tracking_ids=tracking_ids, boxes=bxs, labels=label_names, confs=confs
    )

    return prediction
