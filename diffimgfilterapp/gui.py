import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import logging, datetime, traceback
import numpy as np

from .io_h5 import convert_image_to_h5, load_h5_first_2d
from .methods import METHODS, run_method
from .utils import parse_param, save_image_robust
from .img_ops import _to_uint8

class DiffractionFilteringGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Diffraction Image Filtering App")
        self.geometry("1200x780")
        self.tempdir = tempfile.mkdtemp(prefix="diff_gui_")
        self.h5_path = None
        self.original = None
        self.processed = None
        self.tk_img_original = None
        self.tk_img_processed = None

        self._build_left_panel()
        self._build_right_panel()
        self._init_logger()

    def _build_left_panel(self):
        left = tk.Frame(self, padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(left, text="Load", width=16, command=self.on_load).grid(row=0, column=0, pady=4, sticky="ew")
        tk.Button(left, text="Run",  width=16, command=self.on_run ).grid(row=1, column=0, pady=4, sticky="ew")
        tk.Button(left, text="Clear",width=16, command=self.on_clear).grid(row=2, column=0, pady=4, sticky="ew")
        tk.Button(left, text="Save", width=16, command=self.on_save).grid(row=3, column=0, pady=4, sticky="ew")
        tk.Button(left, text="Quit", width=16, command=self.destroy).grid(row=4, column=0, pady=4, sticky="ew")

        tk.Label(left, text="Method:").grid(row=5, column=0, pady=(16, 4), sticky="w")
        self.method_var = tk.StringVar(value="OpenCV RDB")
        menu = tk.OptionMenu(left, self.method_var, *METHODS.keys(), command=lambda _: self._rebuild_params())
        menu.config(width=20)
        menu.grid(row=6, column=0, sticky="ew")

        tk.Label(left, text="Parameters:").grid(row=7, column=0, pady=(16, 4), sticky="w")
        self.params_frame = tk.LabelFrame(left, text="Method Parameters", padx=6, pady=6)
        self.params_frame.grid(row=8, column=0, sticky="nsew")
        left.grid_rowconfigure(8, weight=1)
        self.param_vars = {}
        self._rebuild_params()

    def _build_right_panel(self):
        self.right = tk.Frame(self, padx=8, pady=8)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        top = tk.Frame(self.right)
        top.pack(fill=tk.BOTH, expand=True)

        self.canvas_orig = tk.Label(top, text="Original", bd=1, relief=tk.SUNKEN)
        self.canvas_proc = tk.Label(top, text="Processed", bd=1, relief=tk.SUNKEN)
        self.canvas_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.canvas_proc.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._build_debug_panel(self.right)

    def _build_debug_panel(self, parent):
        bar = tk.Frame(parent)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 2))
        self.debug_shown = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Debug", variable=self.debug_shown,
                       command=self._toggle_debug).pack(anchor="w")

        self.debug_frame = tk.Frame(parent, height=160, bd=1, relief=tk.SUNKEN)
        self.debug_frame.pack(side=tk.BOTTOM, fill=tk.X)

        wrap = tk.Frame(self.debug_frame)
        wrap.pack(fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(wrap, orient=tk.VERTICAL)
        self.debug_text = tk.Text(wrap, height=8, yscrollcommand=sb.set)
        sb.config(command=self.debug_text.yview)
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _toggle_debug(self):
        if self.debug_shown.get():
            self.debug_frame.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.debug_frame.forget()

    def _init_logger(self):
        self.logfile = os.path.join(self.tempdir, "diff_gui.log")
        self.logger = logging.getLogger("diff_gui")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.logfile, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

    def log(self, *args):
        msg = " ".join(str(a) for a in args)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            self.debug_text.insert(tk.END, line)
            self.debug_text.see(tk.END)
        except Exception:
            pass
        try:
            self.logger.debug(msg)
        except Exception:
            pass

    def _rebuild_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_vars.clear()
        _, spec = METHODS[self.method_var.get()]
        row = 0
        for pname, (kind, default) in spec.items():
            if kind == "bool":
                var = tk.BooleanVar(value=bool(default))
                tk.Checkbutton(self.params_frame, text=pname, variable=var).grid(row=row, column=0, sticky="w", pady=2)
                self.param_vars[pname] = var
            else:
                tk.Label(self.params_frame, text=pname).grid(row=row, column=0, sticky="w")
                ent = tk.Entry(self.params_frame, width=22)
                ent.insert(0, "" if default is None else str(default))
                ent.grid(row=row, column=1, sticky="ew", pady=2)
                self.param_vars[pname] = ent
            row += 1

    # Actions
    def on_load(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.h5"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            self.h5_path = os.path.join(self.tempdir, f"{base}.h5")
            self.log("LOAD start:", path)
            convert_image_to_h5(path, self.h5_path, signal_type="EBSD")
            self.log("H5 written:", self.h5_path)
            with h5py.File(self.h5_path, "r") as f:
                pass
            img = load_h5_first_2d(self.h5_path)
            self.log("Loaded 2D image shape:", img.shape, "min/max:", float(np.min(img)), float(np.max(img)))
            self.original = img.astype(np.float32)
            self._show_image(self.canvas_orig, self.original, is_color=False)
            self.processed = None
            self._clear_canvas(self.canvas_proc, text="Processed")
        except Exception as e:
            tb = traceback.format_exc()
            self.log("LOAD error:", e, "\n", tb)
            messagebox.showerror("Load error", str(e))

    def on_run(self):
        if self.original is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        name, _ = METHODS[self.method_var.get()]
        try:
            _, spec = METHODS[self.method_var.get()]
            kwargs = {}
            for pname, (kind, _) in spec.items():
                w = self.param_vars[pname]
                if isinstance(w, tk.Entry):
                    kwargs[pname] = parse_param(kind, w.get())
                elif isinstance(w, tk.BooleanVar):
                    kwargs[pname] = parse_param(kind, w)
                else:
                    kwargs[pname] = w
            out = run_method(name, self.original.copy(), kwargs)
            self.processed = out
            self.log("RUN result shape=", out.shape, "dtype=", out.dtype,
                     "min/max=", float(np.min(out)), float(np.max(out)))
            self._show_image(self.canvas_proc, out, is_color=(out.ndim == 3 and out.shape[-1] == 3))
        except Exception as e:
            tb = traceback.format_exc()
            self.log("RUN error:", e, "\n", tb)
            messagebox.showerror("Run error", str(e))

    def on_clear(self):
        self.h5_path = None
        self.original = None
        self.processed = None
        self._clear_canvas(self.canvas_orig, text="Original")
        self._clear_canvas(self.canvas_proc, text="Processed")

    def on_save(self):
        if self.processed is None:
            messagebox.showwarning("Nothing to save", "Run a method first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif;*.tiff"), ("JPEG", "*.jpg;*.jpeg")]
        )
        if not out_path:
            return
        try:
            self.log("SAVE start ->", out_path)
            save_image_robust(out_path, self.processed, log_fn=self.log)
            self.log("Saved to", out_path)
            messagebox.showinfo("Saved", f"Saved to:\n{out_path}")
        except Exception as e:
            tb = traceback.format_exc()
            self.log("SAVE error:", e, "\n", tb)
            messagebox.showerror("Save error", f"{e}\nPath: {out_path}")

    # Rendering helpers
    def _show_image(self, widget: tk.Label, img: np.ndarray, is_color: bool):
        self.log("SHOW shape=", getattr(img, "shape", None), "dtype=", getattr(img, "dtype", None))
        widget.update_idletasks()
        w = max(300, widget.winfo_width() or 600)
        h = max(300, widget.winfo_height() or 600)
        if img.ndim == 2:
            pil = Image.fromarray(_to_uint8(img), mode="L")
        else:
            arr = _to_uint8(img)
            if arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            from PIL import Image as PILImage
            pil = PILImage.fromarray(arr)
        rw, rh = w / pil.width, h / pil.height
        r = min(rw, rh)
        new_size = (max(1, int(pil.width * r)), max(1, int(pil.height * r)))
        pil = pil.resize(new_size, Image.Resampling.BICUBIC)
        tkimg = ImageTk.PhotoImage(pil)
        widget.configure(image=tkimg, text="")
        if widget is self.canvas_orig:
            self.tk_img_original = tkimg
        else:
            self.tk_img_processed = tkimg

    def _clear_canvas(self, widget: tk.Label, text=""):
        widget.configure(image="", text=text)
        if widget is self.canvas_orig:
            self.tk_img_original = None
        else:
            self.tk_img_processed = None

import h5py
import cv2
