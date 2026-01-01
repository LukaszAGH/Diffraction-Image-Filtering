import os
import cv2
import numpy as np
from skimage import io as skio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def parse_param(kind: str, val):
    if kind == "int":
        return int(val)
    if kind == "float":
        return float(val)
    if kind == "str":
        return str(val)
    if kind == "bool":
        import tkinter as tk
        return bool(val.get() if isinstance(val, tk.BooleanVar) else val)
    if kind == "tuple2int":
        a, b = str(val).strip().split(",")
        return (int(a), int(b))
    if kind == "list_of_ints":
        return [int(x) for x in str(val).strip().split(",") if x.strip()]
    if kind == "float_or_none":
        s = str(val).strip().lower()
        return None if s in ("", "none", "null") else float(s)
    return val

def save_image_robust(path: str, arr: np.ndarray, log_fn=None) -> None:
    def log(*a):
        try:
            (log_fn or (lambda *x: None))(*a)
        except Exception:
            pass
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    out = arr
    if out.dtype != np.uint8:
        out = out.astype(np.float32)
        out = out - out.min()
        m = out.max()
        if m > 0:
            out = out / m
        out = (out * 255).clip(0, 255).astype(np.uint8)

    ok = cv2.imwrite(path, out)
    log("SAVE cv2.imwrite:", ok, "shape=", getattr(out, "shape", None), "dtype=", getattr(out, "dtype", None))
    if ok:
        return

    try:
        if out.ndim == 3 and out.shape[-1] == 3:
            out2 = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        else:
            out2 = out
        skio.imsave(path, out2)
        log("SAVE skimage.imsave OK")
    except Exception as e:
        log("SAVE fallback error:", e)
        raise

def _prepare_for_plot(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[-1] not in (3, 4):
        img = np.mean(img, axis=-1)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def save_comparison_plot(save_path: str, original: np.ndarray, processed: np.ndarray):
    ori = _prepare_for_plot(original)
    new = _prepare_for_plot(processed)

    vminNew, vmaxNew = 0, 255

    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(2, 1, 1)
    plt.axis('off')
    rect = Rectangle((0.88, 0.01), 0.1, 0.1, color='white', transform=ax1.transAxes)
    ax1.add_patch(rect)
    plt.text(0.93, 0.05, '(a)', color='black', weight='bold', fontsize=14,
             ha='center', va='center', transform=ax1.transAxes)
    plt.imshow(ori, cmap='gray')

    ax2 = plt.subplot(2, 1, 2)
    plt.axis('off')
    rect2 = Rectangle((0.88, 0.01), 0.1, 0.1, color='white', transform=ax2.transAxes)
    ax2.add_patch(rect2)
    plt.text(0.93, 0.05, '(b)', color='black', weight='bold', fontsize=14,
             ha='center', va='center', transform=ax2.transAxes)
    plt.imshow(new, cmap='gray', vmin=vminNew, vmax=vmaxNew)

    plt.subplots_adjust(hspace=0.001)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_histogram_plot(save_path: str, image: np.ndarray, title_suffix: str = ""):
    img = _prepare_for_plot(image)

    image_min = img.min()
    image_max = img.max()

    bins = 256
    range_values = (0, 256)

    plt.figure(figsize=(25, 10))

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')

    rect = Rectangle((0.91, 0.017), 0.07, 0.07, color='white', transform=ax1.transAxes)
    ax1.add_patch(rect)
    plt.text(0.945, 0.05, '(a)', color='black', fontsize=22, weight='bold',
             ha='center', va='center', transform=ax1.transAxes)

    ax2 = plt.subplot(1, 2, 2)
    plt.hist(img.ravel(), bins=bins, range=range_values, fc='k', ec='k')
    plt.xlabel('Natężenie sygnału [j. a.]', fontsize=22, weight='bold')
    plt.ylabel('Liczebność [zliczenia]', fontsize=22, weight='bold')
    plt.tick_params(axis='both', which='major', labelsize=22)

    rect2 = Rectangle((0.91, 0.017), 0.07, 0.07, color='white', transform=ax2.transAxes)
    ax2.add_patch(rect2)
    plt.text(0.945, 0.05, '(b)', color='black', fontsize=22, weight='bold',
             ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()