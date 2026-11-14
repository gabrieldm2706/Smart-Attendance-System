import os
import secrets
from fastapi import Header, HTTPException, status

import os
from dotenv import load_dotenv
load_dotenv()

KIOSK_API_KEY = os.getenv("KIOSK_API_KEY", "")

def verify_kiosk(x_api_key: str | None = Header(default=None)):
    if not KIOSK_API_KEY or not x_api_key or not secrets.compare_digest(
        x_api_key, KIOSK_API_KEY
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid kiosk key.",
        )