import logging
import time
from dataclasses import dataclass

from shared.prediction import Prediction


logger = logging.getLogger("uvicorn")


@dataclass
class TrackingEvent:
    timestamp: float
    label: str
    confidence: float


@dataclass
class AggregateEvent:
    tracking_id: int
    timestamp: float
    time_span: float
    label: str
    confidence: float

    def __str__(self) -> str:
        return f"AggregateEvent(id={self.tracking_id}, ts={self.timestamp}, span={self.time_span:.2f}s, pred={self.label}, conf={self.confidence})"


class Aggregator:
    def __init__(self):
        self.expiry_time: float = 5.0
        self.confidence_threshold: float = 0.6
        self.predictions: dict[int, list[TrackingEvent]] = {}

    def add_prediction(self, prediction: Prediction):
        timestamp = time.time()

        for tid, label, conf in zip(
            prediction.tracking_ids, prediction.labels, prediction.confs
        ):
            event = TrackingEvent(timestamp=timestamp, label=label, confidence=conf)

            if tid not in self.predictions:
                self.predictions[tid] = []

            self.predictions[tid].append(event)

        self.forward_expired()

    def forward_expired(self):
        for id, evts in list(self.predictions.items()):
            now = time.time()
            last = evts[-1]
            span = now - last.timestamp

            if span > self.expiry_time:
                mean_confidence = sum(evt.confidence for evt in evts) / len(evts)
                first = evts[0]
                event = AggregateEvent(
                    tracking_id=id,
                    timestamp=now,
                    time_span=last.timestamp - first.timestamp,
                    label=first.label,
                    confidence=mean_confidence,
                )

                logger.info(f"Aggregating event: {event}")

                # TODO
                # Send forward
                if mean_confidence > self.confidence_threshold:
                    with open("events.log", "a") as f:
                        _ = f.write(f"{event}\n")

                del self.predictions[id]
