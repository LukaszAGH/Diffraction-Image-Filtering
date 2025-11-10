import os
import cv2
import numpy as np
from skimage import io as skio

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
