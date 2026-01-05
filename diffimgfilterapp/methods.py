from . import img_ops as op

METHODS = {
    "OpenCV RDB": ("opencv_rdb", {
        "operation": ("str", "subtract"),
        "filter_domain": ("str", "frequency"),
        "std": ("float_or_none", None),
        "truncate": ("float", 4.0),
        "dtype_out": ("str", "uint8"),
    }),
    "skimage RDB": ("skimage_rdb", {
        "operation": ("str", "subtract"),
        "filter_domain": ("str", "frequency"),
        "std": ("float_or_none", None),
        "truncate": ("float", 4.0),
        "dtype_out": ("str", "uint8"),
    }),
    "SKI RDB + ADAPT_HIST_EQUAL": ("ski_rdb_adapthist", {
        "std": ("float_or_none", None),
        "truncate": ("float", 4.0),
        "kernel_size": ("tuple2int", "15,15"),
        "clip_limit": ("float", 0.01),
    }),
    "OPENCV RDB + ADAPT_HIST_EQUAL(CLAHE)": ("opencv_rdb_clahe", {
        "std": ("float_or_none", None),
        "truncate": ("float", 4.0),
        "clip_limit": ("float", 0.0),
        "tile_grid_size": ("tuple2int", "30,30"),
    }),
    "OPENCV RDB + ADAPT_HIST_EQUAL(TRADITIONAL)": ("opencv_rdb_trad", {
        "std": ("float_or_none", None),
        "truncate": ("float", 4.0),
    }),
    "Butterworth LPF": ("butterworth", {"cutoff": ("float", 30.0), "order": ("int", 2)}),
    "Hough Lines": ("hough", {
        "prob_lines_toggle": ("bool", False),
        "detect_indices": ("bool", False),
        "thresh1": ("int", 80),
        "thresh2": ("int", 180),
        "thresh3": ("int", 100),
        "min_line_length": ("int", 50),
        "max_line_gap": ("int", 10),
        "min_limit": ("int", 300),
        "round_number": ("int", 1500),
        "line_thickness": ("int", 1),
        "min_theta": ("int", 0),
        "max_theta": ("int", 180),
    }),
    "Gabor Sum": ("gabor", {"angles": ("list_of_ints", "0,15,30,45,60,75,90,105,120,135,150,165")}),
    "Top-hat + CLAHE": ("tophat_clahe", {
        "kernel_size": ("int", 15),
        "clip_limit": ("float", 2.0),
        "tile_grid_size": ("tuple2int", "8,8"),
    }),
    "Gamma": ("gamma", {"gamma": ("float", 0.5)}),
}

def run_method(method_key: str, img, kwargs: dict):
    if method_key == "opencv_rdb":
        return op.remove_dynamic_background_full(
            img,
            operation=kwargs["operation"],
            filter_domain=kwargs["filter_domain"],
            std=kwargs["std"],
            truncate=kwargs["truncate"],
            dtype_out=kwargs["dtype_out"]
        )
    if method_key == "skimage_rdb":
        return op.ski_remove_dynamic_background_full(
            img,
            operation=kwargs["operation"],
            filter_domain=kwargs["filter_domain"],
            std=kwargs["std"],
            truncate=kwargs["truncate"],
            dtype_out=kwargs["dtype_out"]
        )

    if method_key == "ski_rdb_adapthist":
        return op.pipeline_ski_rdb_adapthist(
            img,
            std=kwargs["std"],
            truncate=kwargs["truncate"],
            kernel_size=kwargs["kernel_size"],
            clip_limit=kwargs["clip_limit"]
        )
    if method_key == "opencv_rdb_clahe":
        return op.pipeline_opencv_rdb_clahe(
            img,
            std=kwargs["std"],
            truncate=kwargs["truncate"],
            clip_limit=kwargs["clip_limit"],
            tile_grid_size=kwargs["tile_grid_size"]
        )
    if method_key == "opencv_rdb_trad":
        return op.pipeline_opencv_rdb_trad(
            img,
            std=kwargs["std"],
            truncate=kwargs["truncate"]
        )
    if method_key == "butterworth":
        return op.butterworth_lowpass_filter(img, kwargs["cutoff"], kwargs["order"])
    if method_key == "hough":
        params = {
            'prob_lines_toggle': kwargs["prob_lines_toggle"],
            'thresh1': kwargs["thresh1"], 'thresh2': kwargs["thresh2"], 'thresh3': kwargs["thresh3"],
            'min_line_length': kwargs["min_line_length"], 'max_line_gap': kwargs["max_line_gap"],
            'min_limit': kwargs["min_limit"], 'round_number': kwargs["round_number"],
            'line_thickness': kwargs["line_thickness"], 'min_theta': kwargs["min_theta"], 'max_theta': kwargs["max_theta"],
        }
        return op.ready_hough(img, params, detect_indices=kwargs["detect_indices"])
    if method_key == "gabor":
        return op.apply_gabor_filter(img, kwargs["angles"])
    if method_key == "tophat_clahe":
        return op.apply_tophat_clahe(img, kwargs["kernel_size"], kwargs["clip_limit"], kwargs["tile_grid_size"])
    if method_key == "gamma":
        return op.gamma_correction(img, kwargs["gamma"])
    raise ValueError("Unknown method")