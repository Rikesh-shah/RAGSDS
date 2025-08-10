import os
from pathlib import Path
import uuid

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

def save_upload_file_bytes(upload_file, dest: Path = None):
    """
    upload_file: starlette UploadFile
    returns path (str)
    """
    if dest is None:
        dest = UPLOAD_DIR / f"{uuid.uuid4()}_{upload_file.filename}"
    else:
        dest = Path(dest)
    with open(dest, "wb") as f:
        f.write(upload_file.file.read())
    return str(dest)
