from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    model_id: int
    feature_values: Dict[str, Any]


class PredictionResponse(BaseModel):
    id: int
    user_id: int
    model_id: int
    model_type: str
    input_data: Dict[str, Any]
    prediction_result: str
    created_at: datetime
    status: str

    model_config = ConfigDict(from_attributes=True)


