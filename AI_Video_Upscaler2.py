import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import torch
import numpy as np
import os
import threading
from PIL import Image, ImageTk
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from moviepy.editor import VideoFileClip

class VideoUpscalerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Upscaler (Python 3.8 Fix)")
        self.root.geometry("900x750")
        
        # UI Setup
        tk.Label(root, text="AI Video Upscaler (Real-ESRGAN)", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.file_var = tk.StringVar(value="No file selected")
        tk.Entry(root, textvariable=self.file_var, width=80, state="readonly").pack(padx=20, pady=5)
        tk.Button(root, text="1. Select Video", command=self.select_video, bg="#FF9800", fg="white").pack(pady=5)
        
        # Upscale Settings
        settings_frame = tk.Frame(root)
        settings_frame.pack(pady=10)
        tk.Label(settings_frame, text="Scale Factor:").grid(row=0, column=0, padx=5)
        self.scale_var = tk.StringVar(value="4")
        ttk.Combobox(settings_frame, textvariable=self.scale_var, values=["2", "4"], width=5).grid(row=0, column=1)
        
        self.preview_btn = tk.Button(root, text="2. Generate Preview", command=self.generate_preview, state="disabled")
        self.preview_btn.pack(pady=10)
        
        self.preview_canvas = tk.Canvas(root, width=800, height=300, bg="black")
        self.preview_canvas.pack(pady=10)
        
        self.start_btn = tk.Button(root, text="3. START UPSCALING", command=self.start_upscale, bg="#4CAF50", fg="white", state="disabled", font=("Arial", 12, "bold"))
        self.start_btn.pack(pady=10)
        
        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=10)
        
        self.status_var = tk.StringVar(value="Initializing AI...")
        tk.Label(root, textvariable=self.status_var, wraplength=800).pack(pady=10)
        
        # Internal State
        self.video_path = None
        self.upsampler = None
        self.stop_flag = False
        
        # Load Model
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            # Fix for "Is a directory" error: explicitly point to the .pth file
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=model_url,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available()
            )
            self.status_var.set(f"Ready on {device.type.upper()}")
        except Exception as e:
            self.status_var.set(f"Model Error: {str(e)}")

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_path = path
            self.file_var.set(path)
            self.preview_btn.config(state="normal")
            self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def generate_preview(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = int(self.scale_var.get())
        
        self.status_var.set("Generating AI Preview...")
        self.root.update()
        
        output, _ = self.upsampler.enhance(frame_rgb, outscale=scale)
        
        before = Image.fromarray(frame_rgb).resize((400, 300))
        after = Image.fromarray(output).resize((400, 300))
        
        self.before_tk = ImageTk.PhotoImage(before)
        self.after_tk = ImageTk.PhotoImage(after)
        self.preview_canvas.create_image(200, 150, image=self.before_tk)
        self.preview_canvas.create_image(600, 150, image=self.after_tk)
        
        self.start_btn.config(state="normal")
        self.status_var.set("Preview ready.")

    def start_upscale(self):
        self.stop_flag = False
        self.start_btn.config(state="disabled")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        scale = int(self.scale_var.get())
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
        
        temp_out = "temp_upscaled_no_audio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
        
        self.progress["maximum"] = total_frames
        
        count = 0
        while cap.isOpened() and not self.stop_flag:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                output, _ = self.upsampler.enhance(frame_rgb, outscale=scale)
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                out.write(output_bgr)
            except Exception as e:
                out.write(cv2.resize(frame, (width, height)))
            
            count += 1
            if count % 2 == 0:
                self.progress["value"] = count
                self.status_var.set(f"Upscaling: {count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        # Python 3.8 Fix: Manual path split instead of .with_stem()
        file_dir = os.path.dirname(self.video_path)
        file_name = os.path.basename(self.video_path)
        name_part, ext_part = os.path.splitext(file_name)
        final_path = os.path.join(file_dir, f"{name_part}_upscaled{ext_part}")
        
        # Audio Merge
        self.status_var.set("Merging original audio... (Please wait)")
        try:
            with VideoFileClip(self.video_path) as original:
                with VideoFileClip(temp_out) as upscaled:
                    final_video = upscaled.set_audio(original.audio)
                    # logger=None prevents terminal hangs during long renders
                    final_video.write_videofile(final_path, codec="libx264", audio_codec="aac", logger=None)
            
            if os.path.exists(temp_out):
                os.remove(temp_out)
            
            self.status_var.set(f"Success!")
            messagebox.showinfo("Done", f"Video saved to:\n{final_path}")
            
        except Exception as e:
            self.status_var.set("Merge failed, temp file kept.")
            messagebox.showerror("Audio Error", f"Upscaling finished but audio merge failed: {e}")
            
        self.start_btn.config(state="normal")
        self.progress["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoUpscalerGUI(root)
    root.mainloop()

