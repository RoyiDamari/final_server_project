from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime


class TrainedModelResponse(BaseModel):
    id: int
    user_id: int
    model_type: str
    features: list[str]
    feature_schema: dict[str, Any]
    model_params: Optional[Dict[str, Any]]
    label: str
    metrics: Optional[Dict[str, Any]]
    created_at: datetime
    status: str

    model_config = ConfigDict(from_attributes=True)



