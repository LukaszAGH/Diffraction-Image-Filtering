import numpy as np
import h5py
import cv2
from skimage import io as skio

def convert_image_to_h5(input_path: str, h5_path: str, signal_type: str = "EBSD") -> None:
    img = skio.imread(input_path)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
        else:
            img = cv2.cvtColor(img8, cv2.COLOR_RGBA2GRAY)
    while img.ndim > 2:
        img = img[0]
    img = img.astype(np.float32)

    with h5py.File(h5_path, "w") as f:
        grp_meta = f.create_group("meta")
        grp_meta.create_dataset("signal_type", data=np.string_(signal_type))
        f.create_dataset("data", data=img, compression="gzip", compression_opts=4)

def _extract_first_2d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    x = np.squeeze(x)
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        x8 = x.astype(np.uint8) if x.dtype != np.uint8 else x
        gray = cv2.cvtColor(x8, cv2.COLOR_RGB2GRAY) if x.shape[-1] == 3 else cv2.cvtColor(x8, cv2.COLOR_RGBA2GRAY)
        return gray.astype(np.float32)
    if x.ndim > 2:
        sl = (0,) * (x.ndim - 2) + (slice(None), slice(None))
        x = np.squeeze(x[sl])
        if x.ndim == 3 and x.shape[-1] in (3, 4):
            x8 = x.astype(np.uint8) if x.dtype != np.uint8 else x
            x = cv2.cvtColor(x8, cv2.COLOR_RGB2GRAY) if x.shape[-1] == 3 else cv2.cvtColor(x8, cv2.COLOR_RGBA2GRAY)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)

def load_h5_first_2d(h5_file: str) -> np.ndarray:
    with h5py.File(h5_file, 'r') as f:
        if "data" in f and isinstance(f["data"], h5py.Dataset):
            return _extract_first_2d(np.array(f["data"]))
        for path in ['Experiments/__unnamed__/data', 'signals/EBSD/data', 'Dataset/data']:
            if path in f and isinstance(f[path], h5py.Dataset):
                return _extract_first_2d(np.array(f[path]))
        cand = []
        def finder(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape not in [(), (1,)]:
                cand.append(name)
        f.visititems(lambda n, o: finder(n, o))
        if not cand:
            raise ValueError("No suitable dataset in HDF5.")
        return _extract_first_2d(np.array(f[cand[0]]))
