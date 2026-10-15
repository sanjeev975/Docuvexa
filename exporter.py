"""
Multi-Format Export Engine:
- Tesseract OCR extraction (plain text & structured word confidence)
- Heuristic line/delimiter parsing to Pandas DataFrame for CSV tabular export
- Multi-page high-resolution PDF generation via ReportLab / PIL
- ZIP archive bundling of processed images, text, and tables
"""

import io
import re
import os
import zipfile
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import pandas as pd
import pytesseract
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4


def extract_ocr_text(image: np.ndarray, lang: str = "eng") -> Dict[str, Any]:
    """
    Extracts text from processed image using Tesseract OCR.
    Returns plain text and word-level bounding/confidence data.
    """
    # Convert BGR to RGB for PIL / Tesseract
    if len(image.shape) == 3:
        rgb_image = image[:, :, ::-1]
    else:
        rgb_image = image

    pil_img = Image.fromarray(rgb_image)

    # PSM 3: Fully automatic page segmentation, but no OSD
    custom_config = r"--oem 3 --psm 3"
    text = pytesseract.image_to_string(pil_img, lang=lang, config=custom_config)

    # Detailed word-level data
    try:
        data_df = pytesseract.image_to_data(pil_img, lang=lang, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
        # Filter valid words with non-empty text and confidence > 0
        valid_words = data_df[(data_df.text.notnull()) & (data_df.text.str.strip() != "") & (data_df.conf > 0)]
        mean_conf = float(valid_words.conf.mean()) if not valid_words.empty else 0.0
        word_count = len(valid_words)
    except Exception:
        mean_conf = 0.0
        word_count = len(text.split())

    return {
        "text": text.strip(),
        "confidence": round(mean_conf, 1),
        "word_count": word_count
    }


def text_to_dataframe(text: str) -> pd.DataFrame:
    """
    Parses OCR text into a structured Pandas DataFrame using regex and heuristic delimiters.
    Handles invoices, tables, receipts, and columnar lists.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame(columns=["Content"])

    # Delimiter regex: Pipe (|), Tab (\t), Semicolon (;), or 2+ consecutive spaces
    delim_pattern = re.compile(r"\||\t|;|\s{2,}")

    rows = []
    max_cols = 1

    for line in lines:
        # Strip outer pipes if markdown-like table
        cleaned = line.strip("|").strip()
        parts = [p.strip() for p in delim_pattern.split(cleaned) if p.strip()]
        if len(parts) > 1:
            rows.append(parts)
            max_cols = max(max_cols, len(parts))
        else:
            # Check for comma-separated values if comma present
            comma_parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(comma_parts) > 1:
                rows.append(comma_parts)
                max_cols = max(max_cols, len(comma_parts))
            else:
                rows.append([line])

    if max_cols > 1:
        # Normalize all rows to max_cols length
        normalized_rows = [r + [""] * (max_cols - len(r)) for r in rows]
        
        # Heuristic: Check if first row looks like a header (alphabetic, no pure numbers)
        first_row = normalized_rows[0]
        is_header = all(not item.replace("$", "").replace(".", "").isdigit() for item in first_row)
        
        if is_header and len(normalized_rows) > 1:
            columns = [f"Col_{i+1}: {name}" if name else f"Column_{i+1}" for i, name in enumerate(first_row)]
            data = normalized_rows[1:]
        else:
            columns = [f"Column_{i+1}" for i in range(max_cols)]
            data = normalized_rows
            
        df = pd.DataFrame(data, columns=columns)
    else:
        # Fallback to line-item structure
        df = pd.DataFrame({
            "Line": list(range(1, len(lines) + 1)),
            "Extracted_Text": lines
        })

    return df


def generate_pdf_from_images(images: List[np.ndarray], page_mode: str = "fit_image") -> bytes:
    """
    Compiles single or multiple processed scanned images into a compressed,
    high-resolution PDF using ReportLab.
    
    Args:
        images: List of BGR or Grayscale numpy arrays.
        page_mode: 'fit_image' (canvas matches image aspect) or 'A4' (fits image inside A4 margins).
    """
    pdf_buffer = io.BytesIO()

    # ReportLab canvas
    canv = canvas.Canvas(pdf_buffer)

    for img in images:
        # Convert BGR to RGB
        if len(img.shape) == 3:
            rgb = img[:, :, ::-1]
        else:
            rgb = img

        pil_img = Image.fromarray(rgb)
        img_w, img_h = pil_img.size

        # Save image to in-memory JPEG for ReportLab
        img_temp = io.BytesIO()
        pil_img.save(img_temp, format="JPEG", quality=92, optimize=True)
        img_temp.seek(0)

        from reportlab.lib.utils import ImageReader
        img_reader = ImageReader(img_temp)

        if page_mode == "A4":
            page_w, page_h = A4
            canv.setPageSize((page_w, page_h))
            
            # Compute aspect fit with 20pt margin
            margin = 20.0
            avail_w = page_w - (2 * margin)
            avail_h = page_h - (2 * margin)

            scale = min(avail_w / img_w, avail_h / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale
            draw_x = margin + (avail_w - draw_w) / 2.0
            draw_y = margin + (avail_h - draw_h) / 2.0

            canv.drawImage(img_reader, draw_x, draw_y, width=draw_w, height=draw_h)
        else:
            # Exact 1:1 image dimension canvas
            canv.setPageSize((img_w, img_h))
            canv.drawImage(img_reader, 0, 0, width=img_w, height=img_h)

        canv.showPage()

    canv.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def create_export_zip(
    scanned_pages: List[Dict[str, Any]], 
    pdf_bytes: Optional[bytes] = None
) -> bytes:
    """
    Bundles processed scans (JPG/PNG), extracted text (.txt), tabular CSVs (.csv),
    and compiled PDF into a single downloadable ZIP archive.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        combined_text = []

        for idx, page in enumerate(scanned_pages, start=1):
            img_arr = page["image"]
            page_id = page.get("id", f"page_{idx}")
            ocr_text = page.get("ocr_text", "")

            # 1. Image file
            pil_img = Image.fromarray(img_arr[:, :, ::-1] if len(img_arr.shape) == 3 else img_arr)
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="PNG")
            zf.writestr(f"images/{page_id}.png", img_buf.getvalue())

            # 2. Text file per page
            if ocr_text:
                zf.writestr(f"text/{page_id}_ocr.txt", ocr_text)
                combined_text.append(f"--- PAGE {idx} ({page_id}) ---\n{ocr_text}\n")

                # 3. CSV per page if table data parsed
                df = text_to_dataframe(ocr_text)
                csv_str = df.to_csv(index=False)
                zf.writestr(f"tables/{page_id}_table.csv", csv_str)

        # 4. Combined document text
        if combined_text:
            zf.writestr("full_document_ocr.txt", "\n".join(combined_text))

        # 5. Compiled PDF
        if pdf_bytes:
            zf.writestr("document_scan.pdf", pdf_bytes)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
