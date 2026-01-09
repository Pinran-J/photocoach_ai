import exifread
from langchain.tools import tool

@tool
def fetch_exif(file_path):
    """Extract EXIF metadata from an image file."""
    
    with open(file_path, "rb") as file_handle:
        # Return Exif tags
        tags = exifread.process_file(file_handle, builtin_types=True, details=False)
    
    return {
        "Shutterspeed": tags.get("EXIF ExposureTime"),
        "Aperature": tags.get("EXIF FNumber"),
        "ISO": tags.get("EXIF ISOSpeedRatings")
    }
    

# exif_tool_getter = exif_tool
# print(exif_tool_getter.invoke({"file_path": "data\DSCF0677.JPG"}))