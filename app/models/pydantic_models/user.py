from pydantic import BaseModel, EmailStr, UUID4, Field, field_validator
from app.exceptions.user import  UsernameFormatException, PasswordFormatException
import re


USERNAME_REGEX = r"^[a-zA-Z0-9_-]{3,20}$"
PASSWORD_REGEX = r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@#$%^&+=!]{6,20}$"

class RegisterUserRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    username: str
    email: EmailStr = Field(max_length=120)
    password: str

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def _strip_names(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("username", mode="before")
    @classmethod
    def _normalize_and_validate_username(cls, value: str) -> str:
        value = (value or "").strip().lower()
        if not re.match(USERNAME_REGEX, value):
            raise UsernameFormatException()
        return value

    @field_validator("email", mode="before")
    @classmethod
    def _normalize_email(cls, value: EmailStr) -> str:
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def _validate_password(cls, value: str) -> str:
        if not re.match(PASSWORD_REGEX, value or ""):
            raise PasswordFormatException()
        return value


class RegisterUserResponse(BaseModel):
    message: str


class UserTokensResponse(BaseModel):
    username: str
    tokens: int


class DeleteUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    password: str = Field(min_length=1, max_length=20)
    confirm_delete_with_balance: bool = False

    @field_validator("username", mode="before")
    @classmethod
    def _normalize_username(cls, value: str) -> str:
        return (value or "").strip().lower()


class DeleteUserResponse(BaseModel):
    message: str