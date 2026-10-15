/**
 * CamScanner Web Client - Vanilla JS Controller
 * Features:
 * - Live WebRTC camera stream capture
 * - Interactive 4-point polygon canvas with touch & mouse drag handles
 * - OpenCV backend pipeline integration (Warp, Thresholding, CLAHE)
 * - Tesseract OCR parsing with tabular dataframe preview
 * - Multi-page batch management & PDF / ZIP downloads
 */

// Application State
const state = {
    currentImageId: null,
    currentProcessedId: null,
    rawImage: null,
    rawWidth: 0,
    rawHeight: 0,
    corners: [], // [[x, y], [x, y], [x, y], [x, y]] in original image coords
    defaultCorners: [],
    activeHandleIndex: -1,
    scaleFactor: 1.0,
    offsetX: 0,
    offsetY: 0,
    cameraStream: null,
    batchPages: [], // [{ id, processedId, thumbUrl, ocrText }]
    activePageIndex: -1
};

// DOM Elements
const videoEl = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const btnStartCamera = document.getElementById('btnStartCamera');
const btnCapture = document.getElementById('btnCapture');

const tabCamera = document.getElementById('tabCamera');
const tabUpload = document.getElementById('tabUpload');
const cameraSection = document.getElementById('cameraSection');
const uploadSection = document.getElementById('uploadSection');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

const cropCanvas = document.getElementById('cropCanvas');
const ctx = cropCanvas.getContext('2d');
const emptyState = document.getElementById('emptyState');
const processedView = document.getElementById('processedView');
const processedImg = document.getElementById('processedImg');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');

const viewCornersBtn = document.getElementById('viewCornersBtn');
const viewProcessedBtn = document.getElementById('viewProcessedBtn');
const btnResetCorners = document.getElementById('btnResetCorners');
const btnApplyWarp = document.getElementById('btnApplyWarp');

const btnRunOcr = document.getElementById('btnRunOcr');
const ocrMeta = document.getElementById('ocrMeta');
const ocrConfidence = document.getElementById('ocrConfidence');
const ocrWords = document.getElementById('ocrWords');
const ocrTextArea = document.getElementById('ocrTextArea');
const tabPlainText = document.getElementById('tabPlainText');
const tabTableData = document.getElementById('tabTableData');
const plainTextContainer = document.getElementById('plainTextContainer');
const tableDataContainer = document.getElementById('tableDataContainer');
const ocrTable = document.getElementById('ocrTable');

const btnCopyText = document.getElementById('btnCopyText');
const btnDownloadTxt = document.getElementById('btnDownloadTxt');
const btnDownloadCsv = document.getElementById('btnDownloadCsv');
const btnDownloadPng = document.getElementById('btnDownloadPng');
const btnDownloadJpg = document.getElementById('btnDownloadJpg');

const btnExportPdf = document.getElementById('btnExportPdf');
const btnExportZip = document.getElementById('btnExportZip');
const pageCountBadge = document.getElementById('pageCountBadge');
const filmstripItems = document.getElementById('filmstripItems');
const batchCountText = document.getElementById('batchCountText');

// --- 1. INITIALIZATION & CAMERA WEBRTC ---

function init() {
    setupInputTabs();
    setupCameraControls();
    setupFileUpload();
    setupCanvasInteractions();
    setupViewToggles();
    setupFilterAndActions();
    setupOcrControls();
    setupExportHandlers();
}

function setupInputTabs() {
    tabCamera.addEventListener('click', () => {
        tabCamera.classList.add('active');
        tabUpload.classList.remove('active');
        cameraSection.style.display = 'block';
        uploadSection.style.display = 'none';
    });

    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabCamera.classList.remove('active');
        uploadSection.style.display = 'block';
        cameraSection.style.display = 'none';
        stopCamera();
    });
}

async function startCamera() {
    try {
        const constraints = {
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };
        state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoEl.srcObject = state.cameraStream;
        btnStartCamera.textContent = 'Stop Camera';
        btnStartCamera.classList.replace('btn-outline', 'btn-secondary');
        btnCapture.disabled = false;
    } catch (err) {
        console.error('Camera access error:', err);
        alert('Could not access camera. Please allow camera permissions or upload an image file.');
    }
}

function stopCamera() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
        state.cameraStream = null;
        videoEl.srcObject = null;
        btnStartCamera.textContent = 'Start Camera';
        btnStartCamera.classList.replace('btn-secondary', 'btn-outline');
        btnCapture.disabled = true;
    }
}

function setupCameraControls() {
    btnStartCamera.addEventListener('click', () => {
        if (state.cameraStream) {
            stopCamera();
        } else {
            startCamera();
        }
    });

    btnCapture.addEventListener('click', () => {
        if (!state.cameraStream) return;

        cameraCanvas.width = videoEl.videoWidth || 1280;
        cameraCanvas.height = videoEl.videoHeight || 720;
        const camCtx = cameraCanvas.getContext('2d');
        camCtx.drawImage(videoEl, 0, 0, cameraCanvas.width, cameraCanvas.height);
        const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);

        uploadBase64Image(dataUrl);
    });
}

// --- 2. FILE UPLOADS ---

function setupFileUpload() {
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadFileObject(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadFileObject(e.dataTransfer.files[0]);
        }
    });
}

async function uploadFileObject(file) {
    showLoading('Uploading and detecting document edges...');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) {
            handleUploadSuccess(data);
        } else {
            alert('Upload failed: ' + (data.detail || 'Server error'));
        }
    } catch (err) {
        console.error(err);
        alert('Error uploading document file.');
    } finally {
        hideLoading();
    }
}

async function uploadBase64Image(base64Data) {
    showLoading('Processing snapshot & detecting edges...');
    const formData = new FormData();
    formData.append('camera_data', base64Data);

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) {
            handleUploadSuccess(data);
        } else {
            alert('Capture failed: ' + (data.detail || 'Server error'));
        }
    } catch (err) {
        console.error(err);
        alert('Error sending camera frame to server.');
    } finally {
        hideLoading();
    }
}

function handleUploadSuccess(data) {
    state.currentImageId = data.image_id;
    state.rawWidth = data.width;
    state.rawHeight = data.height;
    state.corners = JSON.parse(JSON.stringify(data.corners));
    state.defaultCorners = JSON.parse(JSON.stringify(data.corners));

    // Load original image for canvas rendering
    const img = new Image();
    img.onload = () => {
        state.rawImage = img;
        emptyState.style.display = 'none';
        cropCanvas.style.display = 'block';
        processedView.style.display = 'none';
        viewCornersBtn.classList.add('active');
        viewProcessedBtn.classList.remove('active');
        btnApplyWarp.disabled = false;
        resizeAndDrawCanvas();
    };
    img.src = data.preview_url;
}

// --- 3. INTERACTIVE 4-POINT CANVAS ---

function resizeAndDrawCanvas() {
    if (!state.rawImage) return;

    const container = document.getElementById('canvasContainer');
    const availW = container.clientWidth - 20;
    const availH = container.clientHeight - 20;

    const imgAspect = state.rawWidth / state.rawHeight;
    const containerAspect = availW / availH;

    let drawW, drawH;
    if (imgAspect > containerAspect) {
        drawW = availW;
        drawH = availW / imgAspect;
    } else {
        drawH = availH;
        drawW = availH * imgAspect;
    }

    cropCanvas.width = drawW;
    cropCanvas.height = drawH;
    state.scaleFactor = drawW / state.rawWidth;

    drawCanvas();
}

window.addEventListener('resize', () => {
    if (state.rawImage && cropCanvas.style.display !== 'none') {
        resizeAndDrawCanvas();
    }
});

function drawCanvas() {
    if (!state.rawImage) return;

    ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);

    // 1. Draw raw document image
    ctx.drawImage(state.rawImage, 0, 0, cropCanvas.width, cropCanvas.height);

    // Convert corners to canvas display coordinates
    const scale = state.scaleFactor;
    const canvasPts = state.corners.map(([x, y]) => [x * scale, y * scale]);

    // 2. Draw semi-transparent dark mask outside document polygon
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);

    // Cut out document area
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.moveTo(canvasPts[0][0], canvasPts[0][1]);
    for (let i = 1; i < canvasPts.length; i++) {
        ctx.lineTo(canvasPts[i][0], canvasPts[i][1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // 3. Draw polygon edges
    ctx.save();
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(canvasPts[0][0], canvasPts[0][1]);
    for (let i = 1; i < canvasPts.length; i++) {
        ctx.lineTo(canvasPts[i][0], canvasPts[i][1]);
    }
    ctx.closePath();
    ctx.stroke();

    // Grid cross lines inside quad
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(canvasPts[0][0], canvasPts[0][1]);
    ctx.lineTo(canvasPts[2][0], canvasPts[2][1]);
    ctx.moveTo(canvasPts[1][0], canvasPts[1][1]);
    ctx.lineTo(canvasPts[3][0], canvasPts[3][1]);
    ctx.stroke();
    ctx.restore();

    // 4. Draw corner handles
    const labels = ['TL', 'TR', 'BR', 'BL'];
    canvasPts.forEach(([cx, cy], idx) => {
        ctx.save();
        // Outer halo
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
        ctx.fill();

        // Inner solid circle
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
        ctx.fillStyle = idx === state.activeHandleIndex ? '#3b82f6' : '#10b981';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.fill();
        ctx.stroke();

        // Label tag
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 4;
        ctx.fillText(labels[idx], cx + 12, cy - 8);
        ctx.restore();
    });
}

function setupCanvasInteractions() {
    function getCanvasCoords(e) {
        const rect = cropCanvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function onPointerDown(e) {
        if (!state.rawImage) return;
        const { x, y } = getCanvasCoords(e);
        const scale = state.scaleFactor;
        const hitRadius = 24; // Generous touch target

        state.activeHandleIndex = -1;
        state.corners.forEach(([origX, origY], idx) => {
            const canvasX = origX * scale;
            const canvasY = origY * scale;
            const dist = Math.hypot(x - canvasX, y - canvasY);
            if (dist <= hitRadius) {
                state.activeHandleIndex = idx;
            }
        });

        if (state.activeHandleIndex !== -1) {
            e.preventDefault();
            drawCanvas();
        }
    }

    function onPointerMove(e) {
        if (state.activeHandleIndex === -1) return;
        e.preventDefault();
        const { x, y } = getCanvasCoords(e);
        const scale = state.scaleFactor;

        // Clamp to image dimensions
        const clampedX = Math.max(0, Math.min(state.rawWidth, x / scale));
        const clampedY = Math.max(0, Math.min(state.rawHeight, y / scale));

        state.corners[state.activeHandleIndex] = [clampedX, clampedY];
        drawCanvas();
    }

    function onPointerUp() {
        if (state.activeHandleIndex !== -1) {
            state.activeHandleIndex = -1;
            drawCanvas();
        }
    }

    cropCanvas.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);

    cropCanvas.addEventListener('touchstart', onPointerDown, { passive: false });
    window.addEventListener('touchmove', onPointerMove, { passive: false });
    window.addEventListener('touchend', onPointerUp);

    btnResetCorners.addEventListener('click', () => {
    state.corners = [
        [0, 0],
        [state.rawWidth, 0],
        [state.rawWidth, state.rawHeight],
        [0, state.rawHeight]
    ];
    drawCanvas();
  });
}

// --- 4. VIEW TOGGLES & WARPING ---

function setupViewToggles() {
    viewCornersBtn.addEventListener('click', () => {
        if (!state.rawImage) return;
        viewCornersBtn.classList.add('active');
        viewProcessedBtn.classList.remove('active');
        cropCanvas.style.display = 'block';
        processedView.style.display = 'none';
        resizeAndDrawCanvas();
    });

    viewProcessedBtn.addEventListener('click', () => {
        if (!state.currentProcessedId) return;
        viewProcessedBtn.classList.add('active');
        viewCornersBtn.classList.remove('active');
        cropCanvas.style.display = 'none';
        processedView.style.display = 'flex';
    });
}

function setupFilterAndActions() {
    btnApplyWarp.addEventListener('click', async () => {
        if (!state.currentImageId) return;

        const selectedFilter = document.querySelector('input[name="filterMode"]:checked').value;
        showLoading('Applying 4-point perspective warp & scan filter...');

        try {
            const res = await fetch('/api/transform', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_id: state.currentImageId,
                    corners: state.corners,
                    filter_mode: selectedFilter
                })
            });

            const data = await res.json();
            if (res.ok) {
                state.currentProcessedId = data.processed_id;
                processedImg.src = data.processed_url + '?t=' + Date.now();

                // Switch to processed view
                viewProcessedBtn.classList.add('active');
                viewCornersBtn.classList.remove('active');
                cropCanvas.style.display = 'none';
                processedView.style.display = 'flex';

                // Enable action buttons
                btnRunOcr.disabled = false;
                btnDownloadPng.disabled = false;
                btnDownloadJpg.disabled = false;

                // Add or update page in batch filmstrip
                addBatchPage(data.processed_id, data.processed_url);
            } else {
                alert('Transformation failed: ' + (data.detail || 'Server error'));
            }
        } catch (err) {
            console.error(err);
            alert('Error running perspective transform.');
        } finally {
            hideLoading();
        }
    });
}

// --- 5. BATCH FILMSTRIP MANAGEMENT ---

function addBatchPage(processedId, url) {
    const existingIndex = state.batchPages.findIndex(p => p.processedId === processedId);
    if (existingIndex !== -1) {
        state.batchPages[existingIndex].url = url;
    } else {
        state.batchPages.push({
            id: 'page_' + (state.batchPages.length + 1),
            processedId: processedId,
            url: url,
            ocrText: ''
        });
    }

    state.activePageIndex = state.batchPages.length - 1;
    renderFilmstrip();
    updateBatchCounters();
}

function renderFilmstrip() {
    if (state.batchPages.length === 0) {
        filmstripItems.innerHTML = '<div class="filmstrip-placeholder">Processed pages will appear here. Click to view or remove.</div>';
        return;
    }

    filmstripItems.innerHTML = '';
    state.batchPages.forEach((page, idx) => {
        const item = document.createElement('div');
        item.className = 'filmstrip-item' + (idx === state.activePageIndex ? ' active' : '');
        item.innerHTML = `
            <img src="${page.url}" alt="Page ${idx + 1}">
            <button class="remove-btn" title="Remove page">&times;</button>
        `;

        item.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-btn')) {
                e.stopPropagation();
                removeBatchPage(idx);
            } else {
                selectBatchPage(idx);
            }
        });

        filmstripItems.appendChild(item);
    });
}

function selectBatchPage(index) {
    state.activePageIndex = index;
    const page = state.batchPages[index];
    state.currentProcessedId = page.processedId;
    processedImg.src = page.url;

    viewProcessedBtn.classList.add('active');
    viewCornersBtn.classList.remove('active');
    cropCanvas.style.display = 'none';
    processedView.style.display = 'flex';

    if (page.ocrText) {
        ocrTextArea.value = page.ocrText;
    }

    renderFilmstrip();
}

function removeBatchPage(index) {
    state.batchPages.splice(index, 1);
    if (state.batchPages.length > 0) {
        state.activePageIndex = Math.max(0, index - 1);
        selectBatchPage(state.activePageIndex);
    } else {
        state.activePageIndex = -1;
        state.currentProcessedId = null;
        processedView.style.display = 'none';
        emptyState.style.display = 'block';
    }
    renderFilmstrip();
    updateBatchCounters();
}

function updateBatchCounters() {
    const count = state.batchPages.length;
    pageCountBadge.textContent = count;
    batchCountText.textContent = `${count} page${count === 1 ? '' : 's'}`;
    btnExportPdf.disabled = count === 0;
    btnExportZip.disabled = count === 0;
}

// --- 6. OCR & TABULAR PARSING ---

function setupOcrControls() {
    tabPlainText.addEventListener('click', () => {
        tabPlainText.classList.add('active');
        tabTableData.classList.remove('active');
        plainTextContainer.style.display = 'flex';
        tableDataContainer.style.display = 'none';
    });

    tabTableData.addEventListener('click', () => {
        tabTableData.classList.add('active');
        tabPlainText.classList.remove('active');
        tableDataContainer.style.display = 'flex';
        plainTextContainer.style.display = 'none';
    });

    btnRunOcr.addEventListener('click', async () => {
        if (!state.currentProcessedId) return;

        showLoading('Running Tesseract OCR & parsing structured tables...');

        try {
            const res = await fetch('/api/ocr', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    processed_id: state.currentProcessedId,
                    lang: 'eng'
                })
            });

            const data = await res.json();
            if (res.ok) {
                ocrTextArea.value = data.text;
                ocrConfidence.textContent = data.confidence + '%';
                ocrWords.textContent = data.word_count;
                ocrMeta.style.display = 'flex';
                btnDownloadCsv.disabled = false;

                // Save into batch page
                if (state.activePageIndex !== -1 && state.batchPages[state.activePageIndex]) {
                    state.batchPages[state.activePageIndex].ocrText = data.text;
                }

                // Render table preview
                renderTablePreview(data.table);
            } else {
                alert('OCR failed: ' + (data.detail || 'Server error'));
            }
        } catch (err) {
            console.error(err);
            alert('Error running OCR engine.');
        } finally {
            hideLoading();
        }
    });

    btnCopyText.addEventListener('click', () => {
        if (!ocrTextArea.value) return;
        navigator.clipboard.writeText(ocrTextArea.value);
        btnCopyText.textContent = 'Copied!';
        setTimeout(() => btnCopyText.textContent = 'Copy Text', 2000);
    });

    btnDownloadTxt.addEventListener('click', () => {
        if (!ocrTextArea.value) return;
        downloadBlob(ocrTextArea.value, 'text/plain', 'scanned_ocr.txt');
    });

    btnDownloadCsv.addEventListener('click', () => {
        if (!state.currentProcessedId) return;
        window.location.href = `/api/export/csv/${state.currentProcessedId}`;
    });
}

function renderTablePreview(tableData) {
    if (!tableData || !tableData.columns || tableData.columns.length === 0) {
        ocrTable.innerHTML = '<thead><tr><th>Extracted Data</th></tr></thead><tbody><tr><td>No tabular structure parsed.</td></tr></tbody>';
        return;
    }

    const thead = '<thead><tr>' + tableData.columns.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead>';
    const tbody = '<tbody>' + tableData.rows.map(row => {
        return '<tr>' + tableData.columns.map(c => `<td>${escapeHtml(String(row[c] || ''))}</td>`).join('') + '</tr>';
    }).join('') + '</tbody>';

    ocrTable.innerHTML = thead + tbody;
}

function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// --- 7. EXPORT HANDLERS ---

function setupExportHandlers() {
    btnDownloadPng.addEventListener('click', () => {
        if (!state.currentProcessedId) return;
        const link = document.createElement('a');
        link.href = `/api/processed/${state.currentProcessedId}`;
        link.download = `scan_${state.currentProcessedId.slice(0, 8)}.png`;
        link.click();
    });

    btnDownloadJpg.addEventListener('click', () => {
        if (!state.currentProcessedId) return;
        const link = document.createElement('a');
        link.href = `/api/processed/${state.currentProcessedId}`;
        link.download = `scan_${state.currentProcessedId.slice(0, 8)}.jpg`;
        link.click();
    });

    btnExportPdf.addEventListener('click', async () => {
        const ids = state.batchPages.map(p => p.processedId);
        if (ids.length === 0) return;

        showLoading('Compiling multi-page PDF document...');
        try {
            const res = await fetch('/api/export/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ processed_ids: ids, page_mode: 'fit_image' })
            });

            if (res.ok) {
                const blob = await res.blob();
                downloadBlob(blob, 'application/pdf', 'document_scan.pdf');
            } else {
                alert('PDF generation failed.');
            }
        } catch (err) {
            console.error(err);
            alert('Error generating PDF.');
        } finally {
            hideLoading();
        }
    });

    btnExportZip.addEventListener('click', async () => {
        const ids = state.batchPages.map(p => p.processedId);
        if (ids.length === 0) return;

        showLoading('Bundling scans, OCR text, and CSV tables into ZIP...');
        try {
            const res = await fetch('/api/export/zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ processed_ids: ids })
            });

            if (res.ok) {
                const blob = await res.blob();
                downloadBlob(blob, 'application/zip', 'camscanner_bundle.zip');
            } else {
                alert('ZIP bundling failed.');
            }
        } catch (err) {
            console.error(err);
            alert('Error bundling ZIP export.');
        } finally {
            hideLoading();
        }
    });
}

function downloadBlob(content, mimeType, fileName) {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function showLoading(msg) {
    loadingText.textContent = msg || 'Processing...';
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Start
document.addEventListener('DOMContentLoaded', init);
