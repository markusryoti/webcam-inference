from dataclasses import dataclass


@dataclass
class Prediction:
    tracking_ids: list[int]
    boxes: list[list[float]]
    labels: list[str]
    confs: list[float]
