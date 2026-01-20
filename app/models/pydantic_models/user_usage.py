from pydantic import BaseModel
from typing import List


class ModelTypeDistributionResponse(BaseModel):
    model_type: str
    count: int


class TypeSplitResponse(BaseModel):
    problem_type: str
    count: int


class LabelCount(BaseModel):
    label: str
    count: int


class MetricBucket(BaseModel):
    bucket: float
    count: int


class GroupedLabelDistributionResponse(BaseModel):
    classification: list[LabelCount]
    regression: list[LabelCount]


class GroupedMetricDistributionResponse(BaseModel):
    classification: list[MetricBucket]
    regression: list[MetricBucket]