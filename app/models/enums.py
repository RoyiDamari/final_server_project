from enum import Enum


class ActionType(str, Enum):
    TRAINING   = ("training", 10)
    METADATA   = ("metadata", 1)
    PREDICTION = ("prediction", 5)
    ASSIST     = ("assist", 2)

    def __new__(cls, value: str, cost: int):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._cost = cost
        return obj

    @property
    def cost(self) -> int:
        return self._cost


class RowStatus(str, Enum):
    pending = "pending"
    applied = "applied"
    failed  = "failed"