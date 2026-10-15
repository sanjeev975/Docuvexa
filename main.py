"""
Docuvexa Web Application - FastAPI Server
Integrates OpenCV document perspective transformation, adaptive scanning filters,
Tesseract OCR text/tabular extraction, and multi-format PDF/ZIP export.
"""

import os
import io
import uuid
import base64
from typing import List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from scanner import detect_document_corners, four_point_transform, apply_filter, order_points
from exporter import extract_ocr_text, text_to_dataframe, generate_pdf_from_images, create_export_zip

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

app = FastAPI(
    title="Docuvexa Web Application",
    description="High-performance browser-based document scanner with OpenCV & Tesseract OCR",
    version="1.0.0"
)

# Static and Templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory storage for active session artifacts
IMAGE_STORE = {}       # image_id -> {"image": np.ndarray, "width": int, "height": int, "corners": list}
PROCESSED_STORE = {}   # processed_id -> {"image": np.ndarray, "image_id": str, "filter": str, "ocr_data": dict}


# Pydantic Models
class TransformRequest(BaseModel):
    image_id: str
    corners: Optional[List[List[float]]] = None
    filter_mode: str = "magic_bw"


class OcrRequest(BaseModel):
    processed_id: str
    lang: str = "eng"


class ExportPdfRequest(BaseModel):
    processed_ids: List[str]
    page_mode: str = "fit_image"  # 'fit_image' or 'A4'


class ExportZipRequest(BaseModel):
    processed_ids: List[str]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main Docuvexa single-page interface."""
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except TypeError:
        return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_image(
    file: Optional[UploadFile] = File(None),
    camera_data: Optional[str] = Form(None)
):
    """
    Accepts image file upload or live WebRTC canvas snapshot (Base64 data URL).
    Returns image ID, dimensions, and automatically detected document corner points.
    """
    image_bytes = None

    if file and file.filename:
        image_bytes = await file.read()
    elif camera_data:
        # Parse data:image/jpeg;base64,...
        if "," in camera_data:
            _, encoded = camera_data.split(",", 1)
        else:
            encoded = camera_data
        try:
            image_bytes = base64.b64decode(encoded)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 camera image: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No file or camera data provided.")

    # Decode bytes with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image format.")

    h, w = image.shape[:2]
    image_id = str(uuid.uuid4())

    # Detect 4 corners
    corners, detected = detect_document_corners(image)

    # Cache image in memory
    IMAGE_STORE[image_id] = {
        "image": image,
        "width": w,
        "height": h,
        "corners": corners,
        "detected": detected
    }

    return {
        "status": "success",
        "image_id": image_id,
        "width": w,
        "height": h,
        "corners": corners,
        "auto_detected": detected,
        "preview_url": f"/api/raw/{image_id}"
    }


@app.get("/api/raw/{image_id}")
async def get_raw_image(image_id: str):
    """Serves the original uncropped image."""
    data = IMAGE_STORE.get(image_id)
    if not data:
        raise HTTPException(status_code=404, detail="Image ID not found.")

    success, encoded = cv2.imencode(".jpg", data["image"], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")

    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/jpeg")


@app.post("/api/transform")
async def transform_document(req: TransformRequest):
    """
    Performs 4-point perspective warp rectification and applies the chosen scan filter
    ('magic_bw', 'grayscale', 'enhanced_color', 'original').
    """
    data = IMAGE_STORE.get(req.image_id)
    if not data:
        raise HTTPException(status_code=404, detail="Image not found in session.")

    image = data["image"]
    pts = req.corners if req.corners else data["corners"]

    try:
        pts_arr = np.array(pts, dtype="float32")
        warped = four_point_transform(image, pts_arr)
        processed = apply_filter(warped, req.filter_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warp transformation failed: {str(e)}")

    processed_id = str(uuid.uuid4())
    PROCESSED_STORE[processed_id] = {
        "image": processed,
        "image_id": req.image_id,
        "filter": req.filter_mode,
        "ocr_data": None
    }

    h, w = processed.shape[:2]

    return {
        "status": "success",
        "processed_id": processed_id,
        "filter": req.filter_mode,
        "width": w,
        "height": h,
        "processed_url": f"/api/processed/{processed_id}"
    }


@app.get("/api/processed/{processed_id}")
async def get_processed_image(processed_id: str):
    """Serves the warped and filtered scan image."""
    data = PROCESSED_STORE.get(processed_id)
    if not data:
        raise HTTPException(status_code=404, detail="Processed scan not found.")

    success, encoded = cv2.imencode(".png", data["image"])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")


@app.post("/api/ocr")
async def run_ocr(req: OcrRequest):
    """
    Extracts text using Tesseract OCR and parses structured data into tabular format.
    """
    data = PROCESSED_STORE.get(req.processed_id)
    if not data:
        raise HTTPException(status_code=404, detail="Processed scan not found.")

    try:
        ocr_res = extract_ocr_text(data["image"], lang=req.lang)
        data["ocr_data"] = ocr_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tesseract OCR failed: {str(e)}")

    # Heuristic parsing to tabular dataframe
    df = text_to_dataframe(ocr_res["text"])
    table_preview = {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records")
    }

    return {
        "status": "success",
        "processed_id": req.processed_id,
        "text": ocr_res["text"],
        "confidence": ocr_res["confidence"],
        "word_count": ocr_res["word_count"],
        "table": table_preview,
        "csv_text": df.to_csv(index=False)
    }


@app.post("/api/export/pdf")
async def export_pdf(req: ExportPdfRequest):
    """
    Combines single or multiple processed scans into a high-resolution, compressed PDF.
    """
    if not req.processed_ids:
        raise HTTPException(status_code=400, detail="No processed page IDs provided.")

    images = []
    for pid in req.processed_ids:
        pdata = PROCESSED_STORE.get(pid)
        if not pdata:
            raise HTTPException(status_code=404, detail=f"Page ID {pid} not found.")
        images.append(pdata["image"])

    try:
        pdf_bytes = generate_pdf_from_images(images, page_mode=req.page_mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=document_scan.pdf"}
    )


@app.get("/api/export/csv/{processed_id}")
async def export_csv(processed_id: str):
    """Downloads the tabular OCR data for a specific page as a .csv file."""
    pdata = PROCESSED_STORE.get(processed_id)
    if not pdata:
        raise HTTPException(status_code=404, detail="Page not found.")

    ocr_data = pdata.get("ocr_data")
    if not ocr_data or not ocr_data.get("text"):
        # Run OCR on the fly
        ocr_data = extract_ocr_text(pdata["image"])
        pdata["ocr_data"] = ocr_data

    df = text_to_dataframe(ocr_data["text"])
    csv_str = df.to_csv(index=False)

    return StreamingResponse(
        io.BytesIO(csv_str.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=page_{processed_id[:8]}_table.csv"}
    )


@app.post("/api/export/zip")
async def export_zip(req: ExportZipRequest):
    """
    Exports a comprehensive ZIP bundle containing all scanned pages (PNG),
    OCR extracted text (.txt), tabular CSVs (.csv), and compiled PDF.
    """
    if not req.processed_ids:
        raise HTTPException(status_code=400, detail="No processed IDs provided.")

    pages = []
    images = []
    for idx, pid in enumerate(req.processed_ids, start=1):
        pdata = PROCESSED_STORE.get(pid)
        if not pdata:
            continue

        images.append(pdata["image"])
        ocr_text = pdata["ocr_data"]["text"] if pdata.get("ocr_data") else ""
        pages.append({
            "id": f"scan_page_{idx}",
            "image": pdata["image"],
            "ocr_text": ocr_text
        })

    pdf_bytes = generate_pdf_from_images(images, page_mode="fit_image")
    zip_bytes = create_export_zip(pages, pdf_bytes=pdf_bytes)

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=docuvexa_bundle.zip"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
