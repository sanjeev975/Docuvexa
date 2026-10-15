"""
Computer Vision Document Scanner Pipeline using OpenCV and NumPy.
Provides automated document corner detection, 4-point perspective warping,
and adaptive thresholding / color enhancement filters.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinates in consistent order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates: top-right has smallest diff, bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies perspective transformation to rectify document angles and keystoning.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image: maximum distance between bottom right & bottom left or top right & top left
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Compute height of new image: maximum distance between top right & bottom right or top left & bottom left
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Guard against zero or degenerate dimensions
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)

    # Destination points for top-down perspective view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Calculate perspective transform matrix & warp
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height), flags=cv2.INTER_CUBIC)

    return warped


def detect_document_corners(image: np.ndarray, max_dim: int = 1200) -> Tuple[List[List[float]], bool]:
    """
    Automatically detects document boundaries using Gaussian blur, Canny edge detection,
    and contour polygon approximation.
    
    Returns:
        corners: List of 4 [x, y] coordinates in [TL, TR, BR, BL] order.
        detected: Boolean indicating if a high-confidence 4-point contour was found.
    """
    orig_h, orig_w = image.shape[:2]

    # Resize image for fast and robust edge detection while keeping aspect ratio
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()

    h, w = resized.shape[:2]

    # Preprocessing: Grayscale and Gaussian Blur
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Multi-pass edge detection: adaptive Canny using Otsu's thresholding
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edged = cv2.Canny(blurred, low_thresh, high_thresh)

    # Morphological dilation to close gaps in contour lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    document_contour = None
    min_area = (w * h) * 0.10  # Document must occupy at least 10% of frame

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Look for convex 4-corner polygon
        if len(approx) == 4 and cv2.isContourConvex(approx):
            document_contour = approx
            break

  # Return 100% full uncropped image bounds by default:
    full_image_corners = [
        [0.0, 0.0],                       # Top-Left
        [float(orig_w), 0.0],              # Top-Right
        [float(orig_w), float(orig_h)],    # Bottom-Right
        [0.0, float(orig_h)]               # Bottom-Left
    ]
    return full_image_corners, False


def apply_filter(image: np.ndarray, filter_mode: str = "magic_bw") -> np.ndarray:
    """
    Applies scan enhancement filters:
    - 'magic_bw': Adaptive thresholding + bilateral smoothing for crisp, shadow-free pages
    - 'grayscale': Standard high-clarity grayscale
    - 'enhanced_color': CLAHE on L-channel in LAB space + unsharp masking
    - 'original': Unmodified warped image
    """
    mode = filter_mode.lower()

    if mode in ("magic_bw", "magic", "adaptive_bw"):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Bilateral filter removes texture noise while preserving crisp text edges
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive Gaussian thresholding for clean document background removal
        thresh = cv2.adaptiveThreshold(
            smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 11
        )
        
        # Subtle median blur to clean stray speckles
        cleaned = cv2.medianBlur(thresh, 3)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    elif mode in ("grayscale", "gray"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Contrast stretch
        norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    elif mode in ("enhanced_color", "color", "vibrant"):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Contrast Limited Adaptive Histogram Equalization (CLAHE) on Luminance channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_chan)
        limg = cv2.merge((cl, a_chan, b_chan))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Unsharp masking for crisp text edges
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    else:
        # 'original' or unrecognized mode
        return image.copy()
