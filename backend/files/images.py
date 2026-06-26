"""Convert uploaded image bytes to base64 data URIs for multimodal messages."""
import base64
import imghdr
from typing import Optional

ALLOWED_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


def to_data_uri(filename: str, data: bytes) -> Optional[str]:
    """Return a base64 data URI for an image, or None if unsupported."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mime = ALLOWED_MIME.get(ext)
    if not mime:
        # Detect by magic bytes
        detected = imghdr.what(None, h=data)
        mime = ALLOWED_MIME.get(detected or "")
    if not mime:
        return None
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def is_image(filename: str) -> bool:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return ext in ALLOWED_MIME


def build_vision_content(text: str, image_uris: list[str]) -> list[dict]:
    """Build a LangChain-compatible multimodal content list."""
    content: list[dict] = [{"type": "text", "text": text}]
    for uri in image_uris:
        content.append({
            "type": "image_url",
            "image_url": {"url": uri},
        })
    return content
