import numpy as np
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import exposure, filters, restoration, util
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter

def _to_uint8(img: np.ndarray) -> np.uint8:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img -= img.min()
    m = img.max()
    if m > 0:
        img /= m
    return (img * 255.0).clip(0, 255).astype(np.uint8)

def fft_gaussian_filter(image: np.ndarray, std: float, truncate: float = 4.0) -> np.ndarray:
    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    y, x = np.ogrid[-cy:rows - cy, -cx:cols - cx]
    g = np.exp(-(x**2 + y**2) / (2 * std**2))
    g /= g.sum()
    image_fft = fft2(image)
    kernel_fft = fft2(ifftshift(g), s=image.shape)
    filtered_fft = image_fft * kernel_fft
    return np.real(ifft2(filtered_fft))

def remove_dynamic_background_full(pattern, operation="subtract", filter_domain="frequency",
                                   std=None, truncate=4.0, dtype_out="uint8"):
    if std is None:
        std = pattern.shape[1] / 8.0
    f = pattern.astype(np.float32)
    if filter_domain == "frequency":
        bg = fft_gaussian_filter(f, std, truncate)
    elif filter_domain == "spatial":
        bg = cv2.GaussianBlur(f, (0, 0), std)
    else:
        raise ValueError("filter_domain must be 'frequency' or 'spatial'")
    if operation == "subtract":
        corr = f - bg
    elif operation == "divide":
        corr = f / (bg + 1e-8)
    else:
        raise ValueError("operation must be 'subtract' or 'divide'")
    corr -= corr.min()
    m = corr.max()
    if m > 0:
        corr /= m
    out = (corr * 255).clip(0, 255).astype(np.uint8)
    return out if dtype_out == "uint8" else corr.astype(np.float32)

def ski_remove_dynamic_background_full(pattern, operation="subtract", filter_domain="frequency",
                                       std=None, truncate=4.0, dtype_out="uint8"):
    if std is None:
        std = pattern.shape[1] / 8.0
    f = util.img_as_float(pattern)
    if filter_domain == "frequency":
        bg = fft_gaussian_filter(f, std, truncate)
    elif filter_domain == "spatial":
        bg = gaussian_filter(f, sigma=std, truncate=truncate)
    else:
        raise ValueError("filter_domain must be 'frequency' or 'spatial'")
    corr = f - bg if operation == "subtract" else f / (bg + 1e-8)
    corr -= corr.min()
    m = corr.max()
    if m > 0:
        corr /= m
    out = (corr * 255).clip(0, 255).astype(np.uint8)
    return out if dtype_out == "uint8" else corr.astype(np.float32)

def enhanced_kikuchi_contrast(image, std=10, clip_limit=0.01, truncate=4.0, kernel_size=(32, 32)):
    img = util.img_as_float(image)
    bg = fft_gaussian_filter(img, std, truncate)
    img = img - bg
    img = exposure.rescale_intensity(img, out_range=(0, 1))
    sharp = filters.unsharp_mask(img, radius=3, amount=1.5)
    eq = exposure.equalize_adapthist(sharp, kernel_size=kernel_size, clip_limit=clip_limit)
    eq_u8 = util.img_as_ubyte(eq)
    den = cv2.bilateralFilter(eq_u8, d=9, sigmaColor=75, sigmaSpace=75)
    return den

def process_diffraction_pipeline(image):
    img = image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.morphologyEx(_to_uint8(img), cv2.MORPH_TOPHAT,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
    acc = np.zeros_like(thr, dtype=np.float32)
    for theta in range(0, 180, 45):
        k = cv2.getGaborKernel((21, 21), 4.0, np.deg2rad(theta), 10.0, 0.5, 0, ktype=cv2.CV_32F)
        acc = np.maximum(acc, cv2.filter2D(thr, cv2.CV_32F, k))
    gab = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gab)
    cl_f = cl.astype(np.float32) / 255.0
    out = exposure.adjust_gamma(cl_f, 0.5)
    return (out * 255).clip(0, 255).astype(np.uint8)

def butterworth_lowpass_filter(image, cutoff=30.0, order=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    F = fft2(image)
    Fshift = fftshift(F)
    Ff = Fshift * H
    img_back = np.abs(ifft2(ifftshift(Ff)))
    return _to_uint8(img_back)

def sobel_edge_detection(image):
    im = image / 255.0 if image.max() > 1.0 else image
    edges = filters.sobel(im)
    return (edges * 255).astype(np.uint8)

def ready_hough(image_data, params, detect_indices=False):
    grey = image_data if image_data.ndim == 2 else cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    if grey.dtype != np.uint8:
        grey = cv2.normalize(grey, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grey_blurred = cv2.GaussianBlur(grey, (5, 5), 1)
    edge_image = cv2.Canny(grey_blurred, params['thresh1'], params['thresh2'], apertureSize=3)
    final_image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    if params['prob_lines_toggle']:
        linesP = cv2.HoughLinesP(edge_image, 1, np.pi/180, params['thresh3'],
                                 minLineLength=params['min_line_length'],
                                 maxLineGap=params['max_line_gap'])
        if linesP is not None:
            for l in linesP[:params['min_limit']]:
                x1, y1, x2, y2 = l[0]
                cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), params['line_thickness'])
        return final_image

    lines = cv2.HoughLines(edge_image, 1, np.pi/180, params['thresh3'],
                           min_theta=np.pi/180*params.get('min_theta', 0),
                           max_theta=np.pi/180*params.get('max_theta', 180))

    if not detect_indices:
        if lines is not None:
            for i in range(min(len(lines), params['min_limit'])):
                rho, theta = lines[i][0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a*rho, b*rho
                pt1 = (int(round(x0 + params['round_number'] * (-b))),
                       int(round(y0 + params['round_number'] * (a))))
                pt2 = (int(round(x0 - params['round_number'] * (-b))),
                       int(round(y0 - params['round_number'] * (a))))
                cv2.line(final_image, pt1, pt2, (0, 0, 255), params['line_thickness'], cv2.LINE_AA)
        return final_image

    expected = {"(111)": 0.0, "(001)": 54.7, "(110)": 45.0}
    colors = {"(001)": (255, 0, 0), "(110)": (0, 0, 255), "(111)": (0, 255, 255)}
    tol = 10.0
    shown = set()
    if lines is not None:
        limit = min(len(lines), params['min_limit'])
        for i in range(limit):
            rho, theta = lines[i][0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            pt1 = (int(round(x0 + params['round_number'] * (-b))),
                   int(round(y0 + params['round_number'] * (a))))
            pt2 = (int(round(x0 - params['round_number'] * (-b))),
                   int(round(y0 - params['round_number'] * (a))))
            deg = np.degrees(theta) % 180
            best, mind = None, 999
            for label, ang in expected.items():
                d = min(abs(deg-ang), 180-abs(deg-ang))
                if d < mind and d < tol:
                    best, mind = label, d
            if best:
                cv2.line(final_image, pt1, pt2, colors[best], params['line_thickness'], cv2.LINE_AA)
                if best not in shown:
                    off = 15
                    tx = (int(round(x0 - off*b)), int(round(y0 + off*a)))
                    tx = (max(0, min(tx[0], final_image.shape[1]-50)),
                          max(15, min(tx[1], final_image.shape[0]-10)))
                    (tw, th), _ = cv2.getTextSize(best, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(final_image, (tx[0], tx[1]-th-2), (tx[0]+tw, tx[1]+2), (255,255,255), -1)
                    cv2.putText(final_image, best, tx, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                    shown.add(best)
    # intersections clustering
    if lines is not None:
        intersections = []
        hough_lines = [(np.cos(l[0][1]), np.sin(l[0][1]), -l[0][0]) for l in lines[:params['min_limit']]]
        for i in range(len(hough_lines)):
            for j in range(i+1, len(hough_lines)):
                a1,b1,c1 = hough_lines[i]
                a2,b2,c2 = hough_lines[j]
                det = a1*b2 - a2*b1
                if abs(det) < 1e-10:
                    continue
                x = (b1*c2 - b2*c1)/det
                y = (a2*c1 - a1*c2)/det
                if 0 <= x < final_image.shape[1] and 0 <= y < final_image.shape[0]:
                    intersections.append((int(round(x)), int(round(y))))
        if intersections:
            pts = np.array(intersections)
            db = DBSCAN(eps=10, min_samples=2).fit(pts)
            for lab in set(db.labels_):
                if lab == -1:
                    continue
                center = tuple(np.mean(pts[db.labels_==lab], axis=0).astype(int))
                cv2.circle(final_image, center, 4, (0,255,0), -1)
    return final_image

def apply_gabor_filter(image, angles=None):
    if angles is None:
        angles = list(range(0, 180, 15))
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(_to_uint8(gray))
    acc = np.zeros_like(gray, dtype=np.float32)
    for th in angles:
        k = cv2.getGaborKernel((21, 21), 4.0, np.deg2rad(th), 10.0, 0.5, 0, ktype=cv2.CV_32F)
        acc += np.abs(cv2.filter2D(gray, cv2.CV_32F, k))
    acc = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX)
    return acc.astype(np.uint8)

def apply_tophat_clahe(image, kernel_size=15, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    th = cv2.morphologyEx(_to_uint8(gray), cv2.MORPH_TOPHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(th)

def gamma_correction(image, gamma=0.5):
    f = _to_uint8(image).astype(np.float32) / 255.0
    corrected = exposure.adjust_gamma(f, gamma)
    return (corrected * 255).clip(0, 255).astype(np.uint8)

def enhance_diffraction(image):
    img = util.img_as_float(image)
    sigma_bg = max(2.0, img.shape[1] / 5.0)
    bg = gaussian_filter(img, sigma=sigma_bg)
    img = np.clip(img - bg, 0, None)
    if img.max() > 0:
        img /= img.max()
    sharp = filters.unsharp_mask(img, radius=4, amount=1.2)
    clahe = exposure.equalize_adapthist(sharp, kernel_size=(64, 64), clip_limit=0.003, nbins=512)
    den = restoration.denoise_nl_means(clahe, h=0.08, patch_size=5, patch_distance=6,
                                       fast_mode=True, preserve_range=True)
    return (den * 255).clip(0, 255).astype(np.uint8)
