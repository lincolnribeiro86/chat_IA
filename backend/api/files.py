"""File upload endpoint — returns extracted text or base64 image."""
from fastapi import APIRouter, Depends, UploadFile, File
from typing import List
from auth import require_auth
from files.readers import read_file
from files.images import to_data_uri, is_image

router = APIRouter()


@router.post("/files/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    _user=Depends(require_auth),
):
    results = []
    for f in files:
        data = await f.read()
        name = f.filename or "file"

        if is_image(name):
            uri = to_data_uri(name, data)
            results.append({
                "name": name,
                "type": "image",
                "data_uri": uri,
                "error": None if uri else "Unsupported image format",
            })
        else:
            text, error = read_file(name, data)
            results.append({
                "name": name,
                "type": "text",
                "content": text,
                "error": error,
            })
    return results
