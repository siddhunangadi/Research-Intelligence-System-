"""
FastAPI file ingestion router.
Accepts PDF uploads, caches raw files, and executes indexing integrations.
"""

import os
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from indexing.index_builder import IndexBuilder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["ingestion"])

# Base cache path for document uploads
UPLOAD_DIR = "./data/raw_papers"

@router.post("")
def upload_research_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None)
):
    """
    Saves scientific papers and triggers index construction background tasks.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF file format uploads are supported.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save uploaded file payload
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"File cached locally at: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cache uploaded file locally: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store file: {str(e)}")

    # Trigger indexing task asynchronously to release API response threads
    try:
        builder = IndexBuilder()
        # Ensure target schema directories are present
        background_tasks.add_task(
            builder.build_index_for_pdf,
            pdf_path=file_path,
            title=title
        )
        return {
            "message": "File upload completed. Indexing scheduled in background.",
            "filename": file.filename,
            "target_path": file_path
        }
    except Exception as e:
        logger.error(f"Failed to schedule background index building: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to launch index builder: {str(e)}")
