import exifread

def exif_tool(file_path):
    with open(file_path, "rb") as file_handle:

        # Return Exif tags
        tags = exifread.process_file(file_handle, builtin_types=True, details=False)
        return {
            "Shutterspeed": tags.get("EXIF ExposureTime"),
            "Aperature": tags.get("EXIF FNumber"),
            "ISO": tags.get("EXIF ISOSpeedRatings")
        }