"""
========================================================
GUI KLASIFIKASI & DETEKSI USG — DenseNet121 + YOLO
========================================================
Alur:
1. Gambar diklasifikasi menggunakan model DenseNet121.
2. Jika hasil klasifikasi adalah "Normal", proses selesai.
3. Jika hasil klasifikasi adalah "Penyakit", dilakukan object detection
   menggunakan model YOLO untuk mendeteksi letak penyakit.
========================================================
"""

import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np

# Fix conflict C++ library di Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================
# KONFIGURASI
# =====================================================

DEFAULT_CLS_MODEL_PATH = "model_densenet121_klasifikasi.h5"
DEFAULT_OBJ_MODEL_PATH = "best (5).pt"

CLS_NAMES = ["Normal", "Penyakit"]

CLASS_NAMES = [
    "Kavitas",
    "PolaB",
    "Konsolidasi",
    "PenebalanPleura",
    "Bullae",
    "InfiltratTerkonsolidasi",
]

CLASS_COLORS_HEX = [
    "#00D4FF", "#FF4B6E", "#00FF99",
    "#FFB800", "#CC44FF", "#FF8C00",
]

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

CLASS_COLORS_BGR = [hex_to_bgr(c) for c in CLASS_COLORS_HEX]


# =====================================================
# APLIKASI UTAMA
# =====================================================

class CombinedDetectionApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("USG Analyzer — DenseNet121 + YOLO")
        self.geometry("1280x820")
        self.minsize(1000, 700)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.cls_model      = None
        self.obj_model      = None
        self.cls_model_path = tk.StringVar(value=DEFAULT_CLS_MODEL_PATH)
        self.obj_model_path = tk.StringVar(value=DEFAULT_OBJ_MODEL_PATH)
        
        self.conf_thresh    = tk.DoubleVar(value=0.25)
        self.iou_thresh     = tk.DoubleVar(value=0.45)
        
        self.current_image  = None
        self.result_image   = None
        self.is_predicting  = False
        self.image_path     = None
        self.model_type     = None

        self._build_ui()

        # Load models on startup if exist
        if os.path.exists(DEFAULT_CLS_MODEL_PATH):
            self._load_cls_model_thread(DEFAULT_CLS_MODEL_PATH)
        if os.path.exists(DEFAULT_OBJ_MODEL_PATH):
            self._load_obj_model_thread(DEFAULT_OBJ_MODEL_PATH)

    # ─────────────────────────────────────────────
    # BUILD UI
    # ─────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_left_panel()
        self._build_right_panel()
        self._build_status_bar()

    def _build_left_panel(self):
        outer = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#0F1117")
        outer.grid(row=0, column=0, sticky="nsew")
        outer.grid_propagate(False)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        panel = ctk.CTkScrollableFrame(
            outer, corner_radius=0, fg_color="#0F1117",
            scrollbar_button_color="#1E2D4A",
            scrollbar_button_hover_color="#2A3F6A")
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        # Judul
        title_frame = ctk.CTkFrame(panel, fg_color="#161B27", corner_radius=12)
        title_frame.grid(row=0, column=0, padx=16, pady=(20, 8), sticky="ew")
        ctk.CTkLabel(title_frame, text="🫁  USG ANALYZER",
                 font=ctk.CTkFont(family="Courier New", size=16, weight="bold"),
                 text_color="#00D4FF").pack(pady=(14, 2))
        ctk.CTkLabel(title_frame, text="DenseNet121 + YOLO",
                 font=ctk.CTkFont(size=11), text_color="#6B7A99").pack(pady=(0, 12))

        # Model Klasifikasi
        self._section_label(panel, row=1, text="MODEL KLASIFIKASI (DenseNet121)")
        cls_model_frame = ctk.CTkFrame(panel, fg_color="#161B27", corner_radius=10)
        cls_model_frame.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="ew")

        path_frame1 = ctk.CTkFrame(cls_model_frame, fg_color="transparent")
        path_frame1.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkEntry(
            path_frame1, textvariable=self.cls_model_path,
            font=ctk.CTkFont(size=10), height=32,
            fg_color="#0F1117", border_color="#2A3550"
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(path_frame1, text="📂", width=36, height=32,
                  fg_color="#1E2D4A", hover_color="#2A3F6A",
                  command=lambda: self._browse_model(self.cls_model_path, "h5")).pack(side="right")

        self.btn_load_cls = ctk.CTkButton(
            cls_model_frame, text="⚡ Load Klasifikasi", height=32, corner_radius=8,
            fg_color="#1A3A6B", hover_color="#1E4A8A",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_load_cls_model)
        self.btn_load_cls.pack(fill="x", padx=12, pady=(0, 8))

        self.lbl_cls_model_status = ctk.CTkLabel(
            cls_model_frame, text="● Model belum diload",
            font=ctk.CTkFont(size=11), text_color="#FF4B6E")
        self.lbl_cls_model_status.pack(pady=(0, 10))

        # Model Object Detection
        self._section_label(panel, row=3, text="MODEL DETEKSI (YOLO)")
        obj_model_frame = ctk.CTkFrame(panel, fg_color="#161B27", corner_radius=10)
        obj_model_frame.grid(row=4, column=0, padx=16, pady=(0, 8), sticky="ew")

        path_frame2 = ctk.CTkFrame(obj_model_frame, fg_color="transparent")
        path_frame2.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkEntry(
            path_frame2, textvariable=self.obj_model_path,
            font=ctk.CTkFont(size=10), height=32,
            fg_color="#0F1117", border_color="#2A3550"
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(path_frame2, text="📂", width=36, height=32,
                  fg_color="#1E2D4A", hover_color="#2A3F6A",
                  command=lambda: self._browse_model(self.obj_model_path, "pt")).pack(side="right")

        self.btn_load_obj = ctk.CTkButton(
            obj_model_frame, text="⚡ Load Deteksi", height=32, corner_radius=8,
            fg_color="#1A3A6B", hover_color="#1E4A8A",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_load_obj_model)
        self.btn_load_obj.pack(fill="x", padx=12, pady=(0, 8))

        self.lbl_obj_model_status = ctk.CTkLabel(
            obj_model_frame, text="● Model belum diload",
            font=ctk.CTkFont(size=11), text_color="#FF4B6E")
        self.lbl_obj_model_status.pack(pady=(0, 10))

        # Threshold
        self._section_label(panel, row=5, text="YOLO THRESHOLD")
        thresh_frame = ctk.CTkFrame(panel, fg_color="#161B27", corner_radius=10)
        thresh_frame.grid(row=6, column=0, padx=16, pady=(0, 8), sticky="ew")
        self._slider_row(thresh_frame, "Confidence", self.conf_thresh, 0.05, 0.95, "{:.0%}")
        self._slider_row(thresh_frame, "IoU (NMS)",  self.iou_thresh,  0.10, 0.90, "{:.0%}")

        # Gambar
        self._section_label(panel, row=7, text="GAMBAR")
        img_frame = ctk.CTkFrame(panel, fg_color="#161B27", corner_radius=10)
        img_frame.grid(row=8, column=0, padx=16, pady=(0, 8), sticky="ew")

        ctk.CTkButton(img_frame, text="🖼️  Buka Gambar USG", height=40,
                  fg_color="#1A3A2A", hover_color="#1E4A38",
                  font=ctk.CTkFont(size=12, weight="bold"),
                  command=self._open_image).pack(fill="x", padx=12, pady=(12, 6))

        self.btn_detect = ctk.CTkButton(
            img_frame, text="🔍  Analisis Sekarang", height=44,
            fg_color="#00497A", hover_color="#006AAD",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._run_analysis, state="disabled")
        self.btn_detect.pack(fill="x", padx=12, pady=(0, 6))

        self.btn_save = ctk.CTkButton(
            img_frame, text="💾  Simpan Hasil", height=36,
            fg_color="#2A1A4A", hover_color="#3A2060",
            font=ctk.CTkFont(size=12),
            command=self._save_result, state="disabled")
        self.btn_save.pack(fill="x", padx=12, pady=(0, 12))

        # Hasil deteksi
        self._section_label(panel, row=9, text="HASIL ANALISIS")
        self.result_box = ctk.CTkTextbox(
            panel, height=220, corner_radius=10,
            fg_color="#0A0E18", border_color="#1E2D4A",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color="#B0C0E0")
        self.result_box.grid(row=10, column=0, padx=16, pady=(0, 16), sticky="ew")
        self.result_box.insert("end", "Belum ada analisis.\n")
        self.result_box.configure(state="disabled")

    def _build_right_panel(self):
        panel = ctk.CTkFrame(self, fg_color="#080B12", corner_radius=0)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        tab_bar = ctk.CTkFrame(panel, fg_color="#0F1117", height=48, corner_radius=0)
        tab_bar.grid(row=0, column=0, sticky="ew")
        tab_bar.grid_columnconfigure((0, 1, 2), weight=1)

        self.tab_var = tk.StringVar(value="result")
        for col, (text, val) in enumerate([("Gambar Asli", "original"),
                                           ("Hasil Analisis", "result")]):
            ctk.CTkRadioButton(tab_bar, text=text, variable=self.tab_var, value=val,
                               font=ctk.CTkFont(size=12),
                               fg_color="#00D4FF", hover_color="#0099CC",
                               command=self._switch_tab
                               ).grid(row=0, column=col, padx=20, pady=12, sticky="w")

        self.lbl_filename = ctk.CTkLabel(tab_bar, text="—",
                                         font=ctk.CTkFont(size=11),
                                         text_color="#4A5568")
        self.lbl_filename.grid(row=0, column=2, padx=20, sticky="e")

        self.canvas_frame = ctk.CTkFrame(panel, fg_color="#080B12", corner_radius=0)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#080B12",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._show_canvas_placeholder()
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, height=32, fg_color="#0A0E18", corner_radius=0)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        self.lbl_status = ctk.CTkLabel(bar, text="Siap.",
                                       font=ctk.CTkFont(size=11),
                                       text_color="#4A6080")
        self.lbl_status.grid(row=0, column=0, padx=16, pady=6, sticky="w")

        self.lbl_time = ctk.CTkLabel(bar, text="",
                                     font=ctk.CTkFont(family="Courier New", size=11),
                                     text_color="#2A5070")
        self.lbl_time.grid(row=0, column=2, padx=16, sticky="e")

        self.progress = ctk.CTkProgressBar(bar, width=120, height=8,
                                           fg_color="#1A2030",
                                           progress_color="#00D4FF")
        self.progress.grid(row=0, column=1, padx=16, sticky="e")
        self.progress.set(0)

    # ─────────────────────────────────────────────
    # HELPER UI
    # ─────────────────────────────────────────────

    def _section_label(self, parent, row, text):
        ctk.CTkLabel(parent, text=f"  {text}",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#3A5080", anchor="w"
                     ).grid(row=row, column=0, padx=16, pady=(12, 2), sticky="ew")

    def _slider_row(self, parent, label, variable, from_, to, fmt):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=12, pady=6)

        val_label = ctk.CTkLabel(frame, text=fmt.format(variable.get()),
                                 font=ctk.CTkFont(family="Courier New",
                                                  size=12, weight="bold"),
                                 text_color="#00D4FF", width=42)

        def on_change(val):
            val_label.configure(text=fmt.format(float(val)))

        ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11),
                     text_color="#8892A4", width=80, anchor="w").pack(side="left")
        ctk.CTkSlider(frame, from_=from_, to=to, variable=variable,
                      width=110, height=16,
                      fg_color="#1A2535", progress_color="#005A8A",
                      button_color="#00D4FF", button_hover_color="#00AACC",
                      command=on_change).pack(side="left", padx=8)
        val_label.pack(side="right")

    def _show_canvas_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()  or 800
        h = self.canvas.winfo_height() or 600
        self.canvas.create_text(w // 2, h // 2,
                                text="Buka gambar USG untuk memulai analisis",
                                fill="#1E2D4A",
                                font=("Courier New", 14))

    # ─────────────────────────────────────────────
    # MODEL LOADING (Lazy Import)
    # ─────────────────────────────────────────────

    def _browse_model(self, var_path, ext):
        path = filedialog.askopenfilename(
            title=f"Pilih model (*.{ext})",
            filetypes=[(f"Model File", f"*.{ext}"), ("Semua file", "*.*")])
        if path:
            var_path.set(path)

    def _on_load_cls_model(self):
        path = self.cls_model_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "File model klasifikasi tidak ditemukan!")
            return
        self._load_cls_model_thread(path)

    def _on_load_obj_model(self):
        path = self.obj_model_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "File model deteksi tidak ditemukan!")
            return
        self._load_obj_model_thread(path)

    def _load_cls_model_thread(self, path):
        self.lbl_cls_model_status.configure(text="⏳ Loading...", text_color="#FFB800")
        self.btn_load_cls.configure(state="disabled")
        threading.Thread(target=self._load_cls_model_worker, args=(path,), daemon=True).start()

    def _load_cls_model_worker(self, path):
        try:
            # Import TF di dalam worker untuk menghindari hang
            import tensorflow as tf
            # Batasi GPU VRAM agar tidak habis
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
            from tensorflow.keras.models import load_model
            
            model = load_model(path)
            self.cls_model = model
            name = Path(path).name
            self.after(0, lambda: self._on_cls_model_loaded_ok(name))
        except Exception as e:
            self.after(0, lambda: self._on_cls_model_loaded_fail(str(e)))

    def _on_cls_model_loaded_ok(self, name):
        self.lbl_cls_model_status.configure(text=f"✅ {name}", text_color="#00FF99")
        self.btn_load_cls.configure(state="normal")
        self._set_status("Model klasifikasi berhasil diload.")
        self._update_detect_btn()

    def _on_cls_model_loaded_fail(self, err):
        self.lbl_cls_model_status.configure(text="❌ Gagal", text_color="#FF4B6E")
        self.btn_load_cls.configure(state="normal")
        messagebox.showerror("Error Load Klasifikasi", err)

    def _load_obj_model_thread(self, path):
        self.lbl_obj_model_status.configure(text="⏳ Loading...", text_color="#FFB800")
        self.btn_load_obj.configure(state="disabled")
        threading.Thread(target=self._load_obj_model_worker, args=(path,), daemon=True).start()

    def _load_obj_model_worker(self, path):
        try:
            # Import YOLO di dalam worker
            from ultralytics import YOLO
            
            model = YOLO(path)
            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False)

            task = getattr(model, "task", None) or ""
            if task == "obb" or "obb" in str(path).lower():
                self.model_type = "obb"
            else:
                self.model_type = "detect"

            self.obj_model = model
            name = Path(path).name
            self.after(0, lambda: self._on_obj_model_loaded_ok(name))
        except Exception as e:
            self.after(0, lambda: self._on_obj_model_loaded_fail(str(e)))

    def _on_obj_model_loaded_ok(self, name):
        self.lbl_obj_model_status.configure(text=f"✅ {name}", text_color="#00FF99")
        self.btn_load_obj.configure(state="normal")
        self._set_status("Model deteksi (YOLO) berhasil diload.")
        self._update_detect_btn()

    def _on_obj_model_loaded_fail(self, err):
        self.lbl_obj_model_status.configure(text="❌ Gagal", text_color="#FF4B6E")
        self.btn_load_obj.configure(state="normal")
        messagebox.showerror("Error Load YOLO", err)

    # ─────────────────────────────────────────────
    # IMAGE & ANALYSIS
    # ─────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Pilih gambar USG",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Semua file", "*.*")])
        if not path:
            return
        try:
            self.image_path    = path
            self.current_image = Image.open(path).convert("RGB")
            self.result_image  = None
            self.lbl_filename.configure(text=Path(path).name)
            self.tab_var.set("original")
            self._display_image(self.current_image)
            self._set_status(f"Gambar dibuka: {Path(path).name}")
            self._update_detect_btn()
            self.btn_save.configure(state="disabled")
            self._write_result("Gambar dimuat. Tekan 'Analisis Sekarang'.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal buka gambar:\n{e}")

    def _run_analysis(self):
        if self.is_predicting:
            return
        if self.cls_model is None or self.obj_model is None:
            messagebox.showwarning("Peringatan", "Load kedua model (Klasifikasi & Deteksi) terlebih dahulu!")
            return
        if self.current_image is None:
            messagebox.showwarning("Peringatan", "Buka gambar terlebih dahulu!")
            return
            
        self.is_predicting = True
        self.btn_detect.configure(state="disabled", text="⏳  Menganalisis...")
        self.progress.set(0)
        self._set_status("Menjalankan analisis...")
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        try:
            # Import dependencies untuk analisis di dalam thread
            from tensorflow.keras.preprocessing.image import img_to_array
            
            # 1. KLASIFIKASI DENSENET121
            t0 = time.time()
            self.after(0, lambda: self._set_status("Tahap 1: Klasifikasi (Normal/Penyakit)..."))
            
            img_cls = self.current_image.resize((224, 224))
            img_array = img_to_array(img_cls) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = self.cls_model.predict(img_array, verbose=0)[0]
            cls_idx = np.argmax(preds)
            cls_label = CLS_NAMES[cls_idx]
            cls_conf = preds[cls_idx]
            
            elapsed_cls = time.time() - t0
            self.after(0, lambda: self.progress.set(0.5))

            # 2. DECISION
            if cls_label == "Normal":
                elapsed = time.time() - t0
                summary = self._build_cls_summary(cls_label, cls_conf, preds, elapsed)
                self.after(0, lambda: self._on_analysis_done(self.current_image.copy(), summary, elapsed))
                return
                
            # 3. JIKA PENYAKIT -> DETEKSI YOLO
            self.after(0, lambda: self._set_status("Tahap 2: Deteksi Penyakit (YOLO)..."))
            
            img_cv = cv2.cvtColor(np.array(self.current_image), cv2.COLOR_RGB2BGR)
            conf   = self.conf_thresh.get()
            iou    = self.iou_thresh.get()

            results = self.obj_model.predict(img_cv, conf=conf, iou=iou, verbose=False)
            elapsed = time.time() - t0
            
            self.after(0, lambda: self.progress.set(0.9))

            result     = results[0]
            result_img = self._draw_boxes(self.current_image.copy(), result)
            summary    = self._build_combined_summary(cls_label, cls_conf, preds, result, elapsed)

            self.after(0, lambda: self._on_analysis_done(result_img, summary, elapsed))

        except Exception as e:
            self.after(0, lambda: self._on_analysis_error(str(e)))

    def _on_analysis_done(self, result_img, summary, elapsed):
        self.result_image  = result_img
        self.is_predicting = False
        self.progress.set(1.0)
        self.tab_var.set("result")
        self._display_image(result_img)
        self._write_result(summary)
        self.btn_detect.configure(state="normal", text="🔍  Analisis Sekarang")
        self.btn_save.configure(state="normal")
        self.lbl_time.configure(text=f"⏱ {elapsed*1000:.1f} ms")
        self._set_status(f"Analisis selesai dalam {elapsed*1000:.1f} ms")

    def _on_analysis_error(self, err):
        self.is_predicting = False
        self.progress.set(0)
        self.btn_detect.configure(state="normal", text="🔍  Analisis Sekarang")
        self._set_status("Analisis gagal.")
        messagebox.showerror("Error Analisis", err)

    def _save_result(self):
        if self.result_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile=f"hasil_analisis_{Path(self.image_path).stem}.png")
        if path:
            self.result_image.save(path)
            self._set_status(f"Hasil disimpan: {Path(path).name}")

    def _switch_tab(self):
        tab = self.tab_var.get()
        if tab == "original" and self.current_image:
            self._display_image(self.current_image)
        elif tab == "result" and self.result_image:
            self._display_image(self.result_image)

    def _on_canvas_resize(self, event):
        if self.tab_var.get() == "result" and self.result_image:
            self._display_image(self.result_image)
        elif self.current_image:
            self._display_image(self.current_image)
        else:
            self._show_canvas_placeholder()

    # ─────────────────────────────────────────────
    # ★ DRAW — OBB & NON-OBB
    # ─────────────────────────────────────────────

    def _draw_boxes(self, img: Image.Image, result) -> Image.Image:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        names  = self.obj_model.names

        if result.obb is not None and len(result.obb) > 0:
            for i in range(len(result.obb)):
                cls_id = int(result.obb.cls[i].item())
                conf   = float(result.obb.conf[i].item())
                color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
                name   = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))

                pts = result.obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                cv2.polylines(img_cv, [pts], isClosed=True, color=color, thickness=2)
                self._put_label(img_cv, name, conf, color, int(pts[:, 0].min()), int(pts[:, 1].min()))

        elif result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf   = float(result.boxes.conf[i].item())
                color  = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
                name   = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))

                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color=color, thickness=2)
                self._put_label(img_cv, name, conf, color, x1, y1)

        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def _put_label(self, img_cv, name, conf, color, x, y):
        label = f"{name} {conf:.0%}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.55, 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        y0 = max(y - 10, th + 6)
        cv2.rectangle(img_cv, (x, y0 - th - 4), (x + tw + 6, y0 + 2), color, -1)
        cv2.putText(img_cv, label, (x + 3, y0 - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    # ─────────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────────

    def _display_image(self, img: Image.Image):
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 600
        iw, ih  = img.size
        scale   = min(cw / iw, ch / ih, 1.0)
        nw, nh  = int(iw * scale), int(ih * scale)
        resized = img.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        cx = (cw - nw) // 2
        cy = (ch - nh) // 2
        self.canvas.create_image(cx, cy, anchor="nw", image=self._tk_img)

    # ─────────────────────────────────────────────
    # RESULT SUMMARY
    # ─────────────────────────────────────────────

    def _build_cls_summary(self, cls_label, cls_conf, preds, elapsed) -> str:
        lines = [
            "━━━ HASIL ANALISIS USG ━━━",
            f"Total Waktu : {elapsed*1000:.1f} ms",
            "",
            "1. KLASIFIKASI (DenseNet121)",
            f"   Hasil  : {cls_label} ({cls_conf:.2%})",
            f"   Prob   : Normal ({preds[0]:.2%}) | Penyakit ({preds[1]:.2%})",
            "",
            "➔ Karena terdeteksi Normal, object detection YOLO di-skip."
        ]
        return "\n".join(lines)

    def _build_combined_summary(self, cls_label, cls_conf, preds, result, elapsed) -> str:
        names      = self.obj_model.names
        mode_label = "OBB" if self.model_type == "obb" else "Regular"
        
        lines = [
            "━━━ HASIL ANALISIS USG ━━━",
            f"Total Waktu : {elapsed*1000:.1f} ms",
            "",
            "1. KLASIFIKASI (DenseNet121)",
            f"   Hasil  : {cls_label} ({cls_conf:.2%})",
            f"   Prob   : Normal ({preds[0]:.2%}) | Penyakit ({preds[1]:.2%})",
            "",
            f"2. DETEKSI OBJEK (YOLO - {mode_label})",
        ]

        detections = result.obb if (self.model_type == "obb" and result.obb is not None) else result.boxes

        if detections is None or len(detections) == 0:
            lines += ["   Total objek : 0", "",
                      "   ⚠️  Tidak ada temuan terdeteksi oleh YOLO.",
                      "   → Coba turunkan Confidence Threshold"]
            return "\n".join(lines)

        n = len(detections)
        lines.append(f"   Total objek : {n}\n")

        counts = {}
        for i in range(n):
            cls_id = int(detections.cls[i].item())
            nm = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))
            counts[nm] = counts.get(nm, 0) + 1

        lines.append("   Per Kelas:")
        for nm, cnt in sorted(counts.items()):
            lines.append(f"     {nm:<22}: {cnt} objek")

        lines.append("\n   Detail:")
        for i in range(n):
            cls_id = int(detections.cls[i].item())
            conf   = float(detections.conf[i].item())
            nm     = names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))

            if self.model_type == "obb" and result.obb is not None:
                pts = result.obb.xyxyxyxy[i].cpu().numpy()
                cx_ = int(pts[:, 0].mean())
                cy_ = int(pts[:, 1].mean())
                lines.append(f"     [{i+1}] {nm} {conf:.0%}  center=({cx_},{cy_}) [OBB]")
            else:
                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                lines.append(f"     [{i+1}] {nm} {conf:.0%}  box=({x1},{y1})-({x2},{y2})")

        return "\n".join(lines)

    def _write_result(self, text: str):
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("end", text)
        self.result_box.configure(state="disabled")

    def _set_status(self, msg: str):
        self.lbl_status.configure(text=msg)

    def _update_detect_btn(self):
        if self.cls_model and self.obj_model and self.current_image:
            self.btn_detect.configure(state="normal")
        else:
            self.btn_detect.configure(state="disabled")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    app = CombinedDetectionApp()
    app.mainloop()
