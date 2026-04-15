import exifread
from langchain.tools import tool

@tool
def fetch_exif(image_path: str) -> dict:
    """Extract EXIF metadata from an image file. Returns camera settings and technical info."""
    try:
        with open(image_path, "rb") as file_handle:
            tags = exifread.process_file(file_handle, builtin_types=True, details=False)

        if not tags:
            return {"error": "No EXIF data found. The image may be a screenshot or have stripped metadata."}

        def get(key):
            return tags.get(key)

        return {
            "Camera Make":        get("Image Make"),
            "Camera Model":       get("Image Model"),
            "Lens Model":         get("EXIF LensModel"),
            "Shutter Speed":      get("EXIF ExposureTime"),
            "Aperture":           get("EXIF FNumber"),
            "ISO":                get("EXIF ISOSpeedRatings"),
            "Focal Length":       get("EXIF FocalLength"),
            "Exposure Program":   get("EXIF ExposureProgram"),
            "Exposure Bias":      get("EXIF ExposureBiasValue"),
            "Metering Mode":      get("EXIF MeteringMode"),
            "White Balance":      get("EXIF WhiteBalance"),
            "Flash":              get("EXIF Flash"),
        }
    except FileNotFoundError:
        return {"error": f"Image file not found: {image_path}"}
    except Exception as e:
        return {"error": f"Failed to read EXIF data: {str(e)}"}
    