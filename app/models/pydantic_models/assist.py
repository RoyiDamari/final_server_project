from pydantic import BaseModel


class AssistExplainRequest(BaseModel):
    model_type: str | None = None
    param_key: str | None = None
    context: str | None = None


class AssistExplainResponse(BaseModel):
    data: str
    charged: bool
    balance: int
