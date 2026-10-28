"""Safe image loading shared by forensics and the ML pipeline."""
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
from config import MAX_IMAGE_PIXELS


def load_image(source: "Path | bytes") -> tuple[Image.Image, dict]:
    """Decode an image safely. Returns (RGB image with EXIF rotation applied, raw EXIF dict).

    * Rejects images whose pixel count exceeds MAX_IMAGE_PIXELS BEFORE decoding (decompression-bomb guard).
    * Applies the EXIF Orientation tag: phone photos are often stored sideways, which would
      otherwise defeat face detection.
    """
    handle = BytesIO(source) if isinstance(source, (bytes, bytearray)) else open(source, "rb")
    with handle:
        try:
            image = Image.open(handle)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image is too large ({image.width}x{image.height}). Maximum is {MAX_IMAGE_PIXELS // 1_000_000} megapixels.")
            image.load()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Image could not be decoded.") from exc
        fmt = image.format
        exif = {}
        try:
            from PIL import ExifTags
            for k, v in image.getexif().items():
                if isinstance(v, (str, int, float)):
                    exif[ExifTags.TAGS.get(k, str(k))] = v
        except Exception:
            pass
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.info["format"] = fmt
        return rgb, exif
