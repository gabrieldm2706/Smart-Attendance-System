
from fastapi import HTTPException, Request

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def verify_same_origin(request: Request):
    if request.method in SAFE_METHODS:
        return

    origin = request.headers.get("origin")
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"

    if not origin or origin != expected_origin:
        raise HTTPException(
            status_code=403,
            detail="Request origin is missing or not allowed."
        )
