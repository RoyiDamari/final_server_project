from pydantic import BaseModel


class LoginUserRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    message: str
    access_token: str
    refresh_token: str
    expires_at: int
    balance: int


class RefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int


class LogoutRequest(BaseModel):
    refresh_token: str


class LogoutResponse(BaseModel):
    message: str
