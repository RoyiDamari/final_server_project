from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class MetadataResponse(BaseModel, Generic[T]):
    data: list[T]
    charged: bool
    balance: int


class ActionResponse(BaseModel, Generic[T]):
    data: T
    charged: bool
    balance: int