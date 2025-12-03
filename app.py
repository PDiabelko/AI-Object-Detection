"""
yolo_tkinter_app.py

Features included:
1) Confidence slider (adjust min detection confidence)
5) Side panel showing detected classes + confidence scores
6) Table of bounding boxes (x1, y1, x2, y2, class, confidence)
7) Save output to chosen folder
8) Clear image / reset button
9) Class colors legend
10) Thumbnail history of processed images

Requirements:
    pip install ultralytics pillow numpy opencv-python
Run:
    python yolo_tkinter_app.py
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
import numpy as np
import os
import time
import cv2
import threading
import shutil

# ---------- Configuration ----------
MODEL_PATH = "./best.pt"
CANVAS_MAX_W, CANVAS_MAX_H = 900, 700
THUMB_SIZE = (100, 70)
HISTORY_LIMIT = 8
# -----------------------------------

def ensure_list(x):
    try:
        return list(x)
    except Exception:
        return x

class YOLOTkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Tkinter Detection App")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Load model
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model load error", f"Could not load model at {MODEL_PATH}.\nError: {e}")
            raise

        # State
        self.image_path = None
        self.display_image_pil = None
        self.annotated_pil = None
        self.history = []  # list of dicts {path, thumb_pil}
        self.save_folder = os.getcwd()
        
        # Camera state
        self.camera_active = False
        self.camera_thread = None
        self.camera_capture = None
        self.captured_images = []  # Track all captured image paths
        
        # Create images folder
        self.images_folder = os.path.join(os.getcwd(), "images")
        os.makedirs(self.images_folder, exist_ok=True)

        # Colors for classes
        self.class_names = self.model.names if hasattr(self.model, "names") else {}
        self.class_colors = self._generate_colors_for_classes(self.class_names)

        # UI layout (left main, right side panel)
        self._build_ui()

    def _generate_colors_for_classes(self, names):
        # deterministic random-ish colors
        colors = {}
        n = max(1, len(names))
        rng = np.random.RandomState(0)
        for idx, name in names.items():
            c = tuple(int(x) for x in (rng.randint(30,230), rng.randint(30,230), rng.randint(30,230)))
            colors[name] = c
        return colors

    def _build_ui(self):
        # Top controls frame
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.upload_btn = tk.Button(ctrl_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=4)
        
        self.camera_btn = tk.Button(ctrl_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=4)

        self.run_btn = tk.Button(ctrl_frame, text="Run Detection", command=self.run_detection, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=4)

        self.clear_btn = tk.Button(ctrl_frame, text="Clear / Reset", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=4)
        
        self.stop_btn = tk.Button(ctrl_frame, text="Stop & Exit", command=self.stop_and_exit, bg="#ff4444", fg="white")
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl_frame, text="Min Confidence:").pack(side=tk.LEFT, padx=(12,4))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_scale = tk.Scale(ctrl_frame, variable=self.conf_var, from_=0.01, to=0.99, resolution=0.01,
                                   orient=tk.HORIZONTAL, length=200)
        self.conf_scale.pack(side=tk.LEFT)

        tk.Label(ctrl_frame, text="Save folder:").pack(side=tk.LEFT, padx=(12,4))
        self.save_label = tk.Label(ctrl_frame, text=self.save_folder, anchor="w")
        self.save_label.pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl_frame, text="Choose...", command=self.choose_save_folder).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl_frame, text="Save Output Now", command=self.save_output_to_folder, state=tk.DISABLED).pack(side=tk.LEFT, padx=8)

        # Main content frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: canvas + thumbnail history
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Canvas for image
        self.canvas = tk.Canvas(left_frame, width=CANVAS_MAX_W, height=CANVAS_MAX_H, bg="#222222")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Thumbnail history bar
        thumb_frame = tk.Frame(left_frame)
        thumb_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=6)
        self.thumb_container = tk.Frame(thumb_frame)
        self.thumb_container.pack(side=tk.LEFT, fill=tk.X)

        # Right side panel: detection summary and table
        right_panel = tk.Frame(main_frame, width=360)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # Detected classes + confidences (list)
        det_label = tk.Label(right_panel, text="Detected Objects", font=("Segoe UI", 10, "bold"))
        det_label.pack(anchor="nw")
        self.detect_listbox = tk.Listbox(right_panel, height=8)
        self.detect_listbox.pack(fill=tk.X, pady=(4,12))

        # Bounding boxes table using Treeview
        table_label = tk.Label(right_panel, text="Bounding Boxes (x1,y1,x2,y2,class,conf)", font=("Segoe UI", 10, "bold"))
        table_label.pack(anchor="nw")
        columns = ("x1","y1","x2","y2","class","conf")
        self.tree = ttk.Treeview(right_panel, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(4,12))

        # Make scrollable
        vsb = ttk.Scrollbar(right_panel, orient="vertical", command=self.tree.yview)
        vsb.place(in_=self.tree, relx=1.0, rely=0, relheight=1.0, bordermode="outside")
        self.tree.configure(yscrollcommand=vsb.set)

        # Status bar
        self.status = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _populate_legend(self):
        # Clear legend box
        for child in self.legend_box.winfo_children():
            child.destroy()
        # Populate with small swatches
        for name, color in self.class_colors.items():
            frm = tk.Frame(self.legend_box)
            frm.pack(anchor="w", pady=2)
            sw = tk.Canvas(frm, width=18, height=12)
            sw.create_rectangle(0, 0, 18, 12, fill=self._rgb_to_hex(color), outline="")
            sw.pack(side=tk.LEFT, padx=(0,6))
            tk.Label(frm, text=f"{name}").pack(side=tk.LEFT)

    def _rgb_to_hex(self, rgb):
        return "#%02x%02x%02x" % rgb

    def choose_save_folder(self):
        d = filedialog.askdirectory(initialdir=self.save_folder)
        if d:
            self.save_folder = d
            self.save_label.config(text=self.save_folder)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not file_path:
            return
        self.image_path = file_path
        self._load_and_show_image(file_path)
        self.run_btn.config(state=tk.NORMAL)
        self._update_status(f"Loaded image: {os.path.basename(file_path)}")

    def _load_and_show_image(self, path):
        pil = Image.open(path).convert("RGB")
        self.display_image_pil = pil
        self.annotated_pil = None
        self._draw_on_canvas(pil)

    def _draw_on_canvas(self, pil_img):
        # Fit to canvas while keeping aspect ratio
        w, h = pil_img.size
        maxw, maxh = CANVAS_MAX_W, CANVAS_MAX_H
        scale = min(maxw / w, maxh / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        display = pil_img.resize((new_w, new_h), Image.LANCZOS)
        # center on canvas
        self.canvas.delete("all")
        self.tkimg = ImageTk.PhotoImage(display)
        canvas_w = max(new_w, CANVAS_MAX_W)
        canvas_h = max(new_h, CANVAS_MAX_H)
        self.canvas.config(scrollregion=(0,0,canvas_w,canvas_h))
        self.canvas.create_image(CANVAS_MAX_W//2, CANVAS_MAX_H//2, image=self.tkimg)
        self.root.update_idletasks()

    def run_detection(self):
        if not self.image_path:
            return
        self._update_status("Running detection...")
        self.run_btn.config(state=tk.DISABLED)
        try:
            conf = float(self.conf_var.get())
            # Run model inference with confidence threshold
            results = self.model(self.image_path, conf=conf, imgsz=1280)  # imgsz can be adjusted
            res0 = results[0]

            # Get boxes, classes, confidences
            boxes_xyxy = ensure_list(res0.boxes.xyxy.tolist()) if hasattr(res0.boxes, "xyxy") else []
            confs = ensure_list(res0.boxes.conf.tolist()) if hasattr(res0.boxes, "conf") else []
            cls_idxs = ensure_list(res0.boxes.cls.tolist()) if hasattr(res0.boxes, "cls") else []

            # Build detected info
            detections = []
            for i in range(len(boxes_xyxy)):
                x1,y1,x2,y2 = [int(round(v)) for v in boxes_xyxy[i]]
                cls_idx = int(cls_idxs[i])
                cls_name = self.class_names.get(cls_idx, str(cls_idx))
                conf_score = float(confs[i])
                detections.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"class":cls_name,"conf":conf_score})

            # Create annotated image: prefer res0.plot() if available, else use save
            try:
                plotted = res0.plot()  # numpy array (BGR or RGB depending on version)
                if isinstance(plotted, np.ndarray):
                    # Convert to PIL
                    if plotted.ndim == 3 and plotted.shape[2] == 3:
                        annotated = Image.fromarray(plotted)
                    else:
                        annotated = Image.fromarray(plotted.astype(np.uint8))
                else:
                    annotated = None
            except Exception:
                annotated = None

            if annotated is None:
                # fallback: ask YOLO to save then open file (temporary)
                out_file = os.path.join(self.save_folder, f"__tmp_annot_{int(time.time())}.jpg")
                results[0].save(out_file)
                annotated = Image.open(out_file).convert("RGB")
                try:
                    os.remove(out_file)
                except Exception:
                    pass

            # store annotated PIL
            self.annotated_pil = annotated
            self._draw_on_canvas(self.annotated_pil)

            # Update side panels (listbox, table)
            self._update_detect_list_and_table(detections)

            # Add to history (thumbnail)
            self._add_to_history(self.annotated_pil, self.image_path)

            # Enable save button
            for w in self.root.winfo_children():
                pass
            # Directly enable the "Save Output Now" button by searching (quick hack)
            # A more structured approach would keep a reference; do simple search:
            for child in self.root.winfo_children():
                pass
            # instead, re-enable Run button and find Save button by text
            self.run_btn.config(state=tk.NORMAL)
            self._enable_save_button()

            self._update_status(f"Detection done — {len(detections)} object(s) found")
        except Exception as e:
            messagebox.showerror("Detection error", f"An error occurred during detection:\n{e}")
            self._update_status("Error during detection")
            self.run_btn.config(state=tk.NORMAL)

    def _update_detect_list_and_table(self, detections):
        # Listbox: class names with highest confidence for each detection
        self.detect_listbox.delete(0, tk.END)
        class_count = {}
        for d in detections:
            text = f"{d['class']}  —  {d['conf']:.2f}"
            self.detect_listbox.insert(tk.END, text)
            class_count.setdefault(d['class'], []).append(d['conf'])
        # Also show counts per class at top
        if class_count:
            self.detect_listbox.insert(0, "----- Summary -----")
            for cls, confs in class_count.items():
                self.detect_listbox.insert(0, f"{cls}: {len(confs)}")
        # Table: clear then insert rows
        for r in self.tree.get_children():
            self.tree.delete(r)
        for d in detections:
            self.tree.insert("", tk.END, values=(d['x1'], d['y1'], d['x2'], d['y2'], d['class'], f"{d['conf']:.3f}"))

    def _add_to_history(self, pil_img, original_path):
        # create thumbnail
        thumb = pil_img.copy()
        thumb.thumbnail(THUMB_SIZE)
        # keep history limited
        entry = {"path": original_path, "image": pil_img.copy(), "thumb": thumb}
        self.history.insert(0, entry)
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(-1)
        self._refresh_thumbs()

    def _refresh_thumbs(self):
        for child in self.thumb_container.winfo_children():
            child.destroy()
        for idx, entry in enumerate(self.history):
            img = ImageTk.PhotoImage(entry["thumb"])
            btn = tk.Button(self.thumb_container, image=img, command=lambda i=idx: self._load_history(i))
            btn.image = img  # keep ref
            btn.pack(side=tk.LEFT, padx=3)

    def _load_history(self, idx):
        if idx < 0 or idx >= len(self.history):
            return
        entry = self.history[idx]
        self.annotated_pil = entry["image"]
        self._draw_on_canvas(self.annotated_pil)
        self._update_status(f"Loaded history item: {os.path.basename(entry['path'])}")

    def _enable_save_button(self):
        # Find the Save button by iterating (not ideal but works)
        for child in self.root.winfo_children():
            for sub in child.winfo_children():
                for widget in sub.winfo_children():
                    if isinstance(widget, tk.Button) and widget.cget("text") == "Save Output Now":
                        widget.config(state=tk.NORMAL)

    def save_output_to_folder(self):
        if self.annotated_pil is None:
            messagebox.showinfo("No output", "There is no annotated image to save. Run detection first.")
            return
        # choose filename
        base = os.path.splitext(os.path.basename(self.image_path or "image"))[0]
        fname = f"{base}_annot_{int(time.time())}.jpg"
        path = os.path.join(self.save_folder, fname)
        try:
            self.annotated_pil.save(path, quality=90)
            messagebox.showinfo("Saved", f"Annotated image saved to:\n{path}")
            self._update_status(f"Saved annotated image to {path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save file:\n{e}")

    def clear_all(self):
        self.image_path = None
        self.display_image_pil = None
        self.annotated_pil = None
        self.canvas.delete("all")
        self.detect_listbox.delete(0, tk.END)
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.history = []
        self._refresh_thumbs()
        self.run_btn.config(state=tk.DISABLED)
        self._update_status("Cleared")

    def _update_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()
    
    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera capture with automatic detection"""
        try:
            self.camera_capture = cv2.VideoCapture(0)
            if not self.camera_capture.isOpened():
                messagebox.showerror("Camera Error", "Could not access camera. Please check if it's connected.")
                return
            
            self.camera_active = True
            self.camera_btn.config(text="Stop Camera")
            self.upload_btn.config(state=tk.DISABLED)
            self._update_status("Camera active - capturing every 2 seconds")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
        except Exception as e: 
            messagebox.showerror("Camera Error", f"Failed to start camera:\n{e}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_active = False
        if self.camera_capture:
            self.camera_capture.release()
            self.camera_capture = None
        self.camera_btn.config(text="Start Camera")
        self.upload_btn.config(state=tk.NORMAL)
        self._update_status("Camera stopped")
    
    def _camera_loop(self):
        """Camera capture loop - captures every 2 seconds, runs detection after 1 second"""
        while self.camera_active:
            try:
                # Capture frame
                ret, frame = self.camera_capture.read()
                if not ret:
                    self.root.after(0, lambda: messagebox.showerror("Camera Error", "Failed to capture frame"))
                    self.root.after(0, self.stop_camera)
                    break
                
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save to images folder
                timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
                image_path = os.path.join(self.images_folder, f"capture_{timestamp}.jpg")
                pil_image.save(image_path)
                self.image_path = image_path
                self.captured_images.append(image_path)
                
                # Update display in main thread
                self.root.after(0, lambda img=pil_image: self._load_camera_image(img))
                
                # Wait 1 second, then run detection
                time.sleep(1)
                if self.camera_active:
                    self.root.after(0, self._run_camera_detection)
                
                # Wait additional 1 second to complete 2 second cycle
                time.sleep(1)
                
            except Exception as e:
                self.root.after(0, lambda err=str(e): messagebox.showerror("Camera Error", f"Error in camera loop:\n{err}"))
                self.root.after(0, self.stop_camera)
                break
    
    def _load_camera_image(self, pil_img):
        """Load camera image into display (called from main thread)"""
        self.display_image_pil = pil_img
        self.annotated_pil = None
        self._draw_on_canvas(pil_img)
        self._update_status(f"Camera captured - running detection...")
    
    def _run_camera_detection(self):
        """Run detection on camera image (called from main thread)"""
        if not self.camera_active:
            return
        try:
            self.run_detection()
        except Exception as e:
            print(f"Detection error: {e}")
    
    def stop_and_exit(self):
        """Stop camera, delete all captured images, and close application"""
        # Stop camera if active
        if self.camera_active:
            self.stop_camera()
        
        # Delete images folder and all contents
        try:
            if os.path.exists(self.images_folder):
                shutil.rmtree(self.images_folder)
                print(f"Deleted images folder: {self.images_folder}")
        except Exception as e:
            print(f"Error deleting images folder: {e}")
        
        # Close application
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTkApp(root)
    
    # Clean up camera on close
    def on_closing():
        if app.camera_active:
            app.stop_camera()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
