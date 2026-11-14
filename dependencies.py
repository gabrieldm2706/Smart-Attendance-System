from fastapi import Cookie, HTTPException, status
from jwt.exceptions import InvalidTokenError

from app.auth import decode_access_token

import os
import secrets

from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()

KIOSK_API_KEY = os.getenv("KIOSK_API_KEY")


def verify_kiosk(x_api_key: str | None = Header(default=None)):
    if (
        not KIOSK_API_KEY
        or not x_api_key
        or not secrets.compare_digest(x_api_key, KIOSK_API_KEY)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid kiosk key.",
        )

def get_current_admin(
    access_token: str | None = Cookie(default=None)
):
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please log in.",
        )

    try:
        payload = decode_access_token(access_token)
        username = payload.get("sub")

        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token.",
            )

        return {"username": username}

    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        )