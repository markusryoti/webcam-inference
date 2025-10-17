import asyncio
import logging
import uuid

from aiortc import (
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services.datachannel import DataChannelManager
from services.inference import frame_producer

logger = logging.getLogger("uvicorn")


router = APIRouter()

pcs: dict[str, RTCPeerConnection] = {}


class Offer(BaseModel):
    sdp: str
    type: str


@router.post("/offer")
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
