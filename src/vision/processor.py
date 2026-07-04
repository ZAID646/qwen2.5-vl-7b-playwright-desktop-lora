from __future__ import annotations

from io import BytesIO

from PIL import Image

MAX_PIXELS = 1024 * 1024  # 1024x1024 ceiling


def process_screenshot(image_data: bytes, max_pixels: int = MAX_PIXELS) -> bytes:
    img = Image.open(BytesIO(image_data))

    if img.mode != "RGB":
        img = img.convert("RGB")

    if img.width * img.height > max_pixels:
        ratio = (max_pixels / (img.width * img.height)) ** 0.5
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
