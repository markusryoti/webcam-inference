import json
import logging
from aiortc import RTCDataChannel


from shared.prediction import Prediction


logger = logging.getLogger("uvicorn")


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
