import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import webrtc
from services.aggregator import Aggregator
from services.datachannel import DataChannelManager
from services.inference import inference_worker

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


if os.path.exists("events.log"):
    os.remove("events.log")


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task[None]] = []

    channel_manager = DataChannelManager()
    aggregator = Aggregator()

    app.state.channel_manager = channel_manager

    for _ in range(3):
        worker_task = asyncio.create_task(inference_worker(channel_manager, aggregator))
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webrtc.router, prefix="/webrtc", tags=["WebRTC"])


@app.get("/")
async def root():
    return {"message": "Hello World"}
