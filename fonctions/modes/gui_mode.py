"""GUI processing mode with separate video and control windows."""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image
import cv2
from ultralytics import YOLO
from fonctions.utils import logger, DEFAULT_CAMERA_INDEX, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
from fonctions.core import process_img

try:
    from PIL import ImageTk
except ImportError:
    ImageTk = None


class FaceBlurrerGUI:
    """GUI interface for face blurrer with separate video and control windows."""

    def __init__(self, model: YOLO):
        """Initialize GUI.

        Args:
            model: YOLOv8 model instance
        """
        self.model = model
        self.blur_enabled = True
        self.custom_image_enabled = False
        self.custom_image = None
        self.running = True
        self.cap = None
        self.frame_count = 0

        self.root = tk.Tk()
        self.root.withdraw()

        self._create_video_window()
        self._create_control_window()

        self._open_camera()

        self.update_frame()

    def _create_video_window(self):
        """Create video display window (for OBS capture)."""
        self.video_window = tk.Toplevel(self.root)
        self.video_window.title("Face Blurrer - Video")
        self.video_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.video_label = ttk.Label(self.video_window)
        self.video_label.pack(padx=0, pady=0)

        self.video_window.geometry("+100+100")

    def _create_control_window(self):
        """Create control panel window."""
        self.control_window = tk.Toplevel(self.root)
        self.control_window.title("Face Blurrer - Contrôles")
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        main_frame = ttk.Frame(self.control_window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(
            main_frame,
            text="Contrôles Face Blurrer",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        self.toggle_btn = ttk.Button(
            main_frame,
            text="Désactiver le Flou",
            command=self.toggle_blur,
            width=25
        )
        self.toggle_btn.grid(row=1, column=0, columnspan=2, pady=10)

        self.status_label = ttk.Label(
            main_frame,
            text="Statut: Flou ACTIVÉ",
            font=("Arial", 12, "bold"),
            foreground="green"
        )
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Separator(main_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20
        )

        custom_label = ttk.Label(
            main_frame,
            text="Image Personnalisée",
            font=("Arial", 11, "bold")
        )
        custom_label.grid(row=4, column=0, columnspan=2, pady=(0, 10))

        self.custom_toggle_btn = ttk.Button(
            main_frame,
            text="Activer Image Custom",
            command=self.toggle_custom_image,
            width=25
        )
        self.custom_toggle_btn.grid(row=5, column=0, columnspan=2, pady=5)

        self.select_btn = ttk.Button(
            main_frame,
            text="Sélectionner Image",
            command=self.select_custom_image,
            width=25
        )
        self.select_btn.grid(row=6, column=0, columnspan=2, pady=5)

        self.custom_status_label = ttk.Label(
            main_frame,
            text="Statut: Aucune image",
            font=("Arial", 10),
            foreground="gray"
        )
        self.custom_status_label.grid(row=7, column=0, columnspan=2, pady=10)

        ttk.Separator(main_frame, orient='horizontal').grid(
            row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20
        )

        info_label = ttk.Label(
            main_frame,
            text="💡 Capturez la fenêtre 'Video'\n   dans OBS sans les contrôles",
            font=("Arial", 10),
            foreground="gray",
            justify=tk.CENTER
        )
        info_label.grid(row=9, column=0, columnspan=2, pady=(0, 15))

        quit_btn = ttk.Button(
            main_frame,
            text="Quitter",
            command=self.on_closing,
            width=25
        )
        quit_btn.grid(row=10, column=0, columnspan=2, pady=10)

        self.control_window.geometry("+1100+100")

    def _open_camera(self):
        """Open camera and configure settings."""
        try:
            self.cap = cv2.VideoCapture(DEFAULT_CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera opened - Resolution: {actual_width}x{actual_height}")

        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            self.on_closing()
            raise

    def toggle_blur(self):
        """Toggle blur on/off."""
        self.blur_enabled = not self.blur_enabled

        if self.blur_enabled:
            self.toggle_btn.config(text="Désactiver le Flou")
            self.status_label.config(text="Statut: Flou ACTIVÉ", foreground="green")
            logger.info("Blur ENABLED")
        else:
            self.toggle_btn.config(text="Activer le Flou")
            self.status_label.config(text="Statut: Flou DÉSACTIVÉ", foreground="red")
            logger.info("Blur DISABLED")

    def toggle_custom_image(self):
        """Toggle custom image replacement on/off."""
        if self.custom_image is None:
            logger.warning("No custom image loaded yet")
            return

        self.custom_image_enabled = not self.custom_image_enabled

        if self.custom_image_enabled:
            self.custom_toggle_btn.config(text="Désactiver Image Custom")
            self.custom_status_label.config(text="Statut: Image ACTIVÉE", foreground="green")
            logger.info("Custom image ENABLED")
        else:
            self.custom_toggle_btn.config(text="Activer Image Custom")
            self.custom_status_label.config(text="Statut: Image DÉSACTIVÉE", foreground="orange")
            logger.info("Custom image DISABLED")

    def select_custom_image(self):
        """Open file dialog to select custom image."""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )

        if file_path:
            try:
                custom_img = cv2.imread(file_path)
                if custom_img is None:
                    logger.error(f"Failed to load image: {file_path}")
                    self.custom_status_label.config(
                        text="Erreur: Image introuvable",
                        foreground="red"
                    )
                    return

                self.custom_image = custom_img
                img_name = file_path.split('/')[-1]
                logger.info(f"Custom image loaded: {file_path}")
                self.custom_status_label.config(
                    text=f"Image: {img_name}",
                    foreground="green"
                )

                if not self.custom_image_enabled:
                    self.custom_toggle_btn.config(state='normal')

            except Exception as e:
                logger.error(f"Error loading image: {e}")
                self.custom_status_label.config(
                    text="Erreur: Impossible de charger",
                    foreground="red"
                )

    def update_frame(self):
        """Update video frame in GUI."""
        if not self.running:
            return

        ret, frame = self.cap.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            self.on_closing()
            return

        try:
            processed_frame = process_img(
                frame,
                self.model,
                blur_enabled=self.blur_enabled,
                custom_image_enabled=self.custom_image_enabled,
                custom_image=self.custom_image
            )

            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(rgb_frame)

            display_width = 960
            display_height = int(display_width * pil_image.height / pil_image.width)
            pil_image = pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)

            if ImageTk:
                photo = ImageTk.PhotoImage(image=pil_image)
            else:
                logger.warning("ImageTk not available - using numpy conversion")
                import numpy as np
                rgb_array = np.array(pil_image)
                height, width = rgb_array.shape[:2]
                photo = tk.PhotoImage(width=width, height=height)
                ppm_data = "P6 {} {} 255 ".format(width, height).encode() + rgb_array.tobytes()
                photo.put(ppm_data)

            self.video_label.config(image=photo)
            self.video_label.image = photo

            self.frame_count += 1

            if self.frame_count % 30 == 0:
                logger.debug(f"Processed {self.frame_count} frames")

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

        self.root.after(33, self.update_frame)

    def on_closing(self):
        """Clean up resources and close windows."""
        logger.info("Closing GUI...")
        self.running = False

        if self.cap is not None:
            self.cap.release()
            logger.debug("Camera released")

        try:
            self.video_window.destroy()
        except:
            pass
        try:
            self.control_window.destroy()
        except:
            pass

        self.root.quit()
        self.root.destroy()
        logger.info(f"GUI closed - {self.frame_count} frames processed")

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def process_gui_mode(model: YOLO) -> None:
    """Process webcam stream with GUI controls.

    Args:
        model: YOLOv8 model instance

    Raises:
        RuntimeError: If camera fails or GUI fails
    """
    logger.info("Starting GUI mode")

    try:
        gui = FaceBlurrerGUI(model)
        gui.run()
    except Exception as e:
        logger.error(f"GUI mode failed: {e}")
        raise RuntimeError(f"GUI mode failed: {e}")
