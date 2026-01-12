import exifread
from langchain.tools import tool

@tool
def fetch_exif(image_path: str) -> dict:
    """Extract EXIF metadata from an image file."""
    
    with open(image_path, "rb") as file_handle:
        # Return Exif tags
        tags = exifread.process_file(file_handle, builtin_types=True, details=False)
    
    return {
        "Shutterspeed": tags.get("EXIF ExposureTime"),
        "Aperature": tags.get("EXIF FNumber"),
        "ISO": tags.get("EXIF ISOSpeedRatings")
    }
    