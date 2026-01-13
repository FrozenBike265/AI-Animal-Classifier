#!/usr/bin/env python3
"""
Animal Classifier GUI Application
Using full TensorFlow module paths for compatibility
A user-friendly interface for classifying animal images using a trained deep learning model.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
import threading
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class AnimalClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Classifier - AI Image Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.class_names = None
        self.current_image = None
        self.processed_image = None
        
        # Model configuration
        self.IMG_SIZE = 224
        self.MODEL_PATH = "animal_classifier_model.keras"
        self.CLASS_INDICES_PATH = "class_indices.npy"
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.load_model()
        
        # Center window
        self.center_window()
        
    def setup_styles(self):
        """Configure modern styling for the GUI"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', 
                           font=('Arial', 20, 'bold'),
                           background='#f0f0f0',
                           foreground='#2c3e50')
        
        self.style.configure('Subtitle.TLabel',
                           font=('Arial', 12),
                           background='#f0f0f0',
                           foreground='#7f8c8d')
        
        self.style.configure('Result.TLabel',
                           font=('Arial', 16, 'bold'),
                           background='#f0f0f0',
                           foreground='#27ae60')
        
        self.style.configure('Confidence.TLabel',
                           font=('Arial', 14),
                           background='#f0f0f0',
                           foreground='#3498db')
        
        self.style.configure('Custom.TButton',
                           font=('Arial', 11, 'bold'),
                           padding=(15, 8))
        
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="25")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 25))
        
        title_label = ttk.Label(title_frame, 
                               text="Animal Classifier", 
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame,
                                  text="Upload an animal image to identify the species with AI",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(pady=(8, 0))
        
        # Content area with two columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Image display
        left_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="20")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Image canvas with border
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(canvas_frame, 
                                    width=400, height=400,
                                    bg='white',
                                    relief=tk.SUNKEN,
                                    borderwidth=2)
        self.image_canvas.pack(pady=15)
        
        # Placeholder text
        self.placeholder_text = self.image_canvas.create_text(
            200, 200,
            text="No image selected\nClick 'Upload Image' to start",
            font=('Arial', 14),
            fill='#bdc3c7',
            justify=tk.CENTER
        )
        
        # Upload button
        self.upload_btn = ttk.Button(left_frame,
                                   text="Upload Image",
                                   command=self.upload_image,
                                   style='Custom.TButton')
        self.upload_btn.pack(pady=15)
        
        # Right column - Results
        right_frame = ttk.LabelFrame(content_frame, text="Classification Results", padding="20")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Model status
        self.status_label = ttk.Label(right_frame,
                                    text="Loading model...",
                                    style='Subtitle.TLabel')
        self.status_label.pack(pady=(0, 25))
        
        # Results area
        results_frame = ttk.Frame(right_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Primary prediction
        ttk.Label(results_frame, text="Predicted Animal:", 
                 font=('Arial', 14, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.result_label = ttk.Label(results_frame,
                                    text="No prediction yet",
                                    style='Result.TLabel')
        self.result_label.pack(anchor=tk.W, pady=(5, 20))
        
        # Confidence
        ttk.Label(results_frame, text="Confidence:", 
                 font=('Arial', 14, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.confidence_label = ttk.Label(results_frame,
                                        text="---%",
                                        style='Confidence.TLabel')
        self.confidence_label.pack(anchor=tk.W, pady=(5, 15))
        
        # Progress bar
        ttk.Label(results_frame, text="Confidence Level:", 
                 font=('Arial', 11)).pack(anchor=tk.W, pady=(5, 5))
        
        self.confidence_bar = ttk.Progressbar(results_frame,
                                            length=250,
                                            mode='determinate')
        self.confidence_bar.pack(anchor=tk.W, pady=(5, 20))
        
        # Top 3 predictions
        ttk.Label(results_frame, text="Top 3 Predictions:", 
                 font=('Arial', 14, 'bold')).pack(anchor=tk.W, pady=(15, 8))
        
        self.top3_frame = ttk.Frame(results_frame)
        self.top3_frame.pack(fill=tk.X)
        
        self.top3_labels = []
        for i in range(3):
            label = ttk.Label(self.top3_frame, 
                            text=f"{i+1}. ---", 
                            font=('Arial', 11))
            label.pack(anchor=tk.W, pady=3)
            self.top3_labels.append(label)
        
        # Classify button
        self.classify_btn = ttk.Button(right_frame,
                                     text="Classify Image",
                                     command=self.classify_image,
                                     style='Custom.TButton',
                                     state='disabled')
        self.classify_btn.pack(pady=(25, 0))
        
        # Processing indicator
        self.processing_label = ttk.Label(right_frame,
                                        text="",
                                        font=('Arial', 11, 'italic'),
                                        foreground='#e67e22')
        self.processing_label.pack(pady=(15, 0))
        
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def load_model(self):
        """Load the trained model and class names using full TensorFlow paths"""
        def load_in_background():
            try:
                # Check if model file exists
                if not Path(self.MODEL_PATH).exists():
                    raise FileNotFoundError(f"Model file '{self.MODEL_PATH}' not found!")
                
                # Load model using full TensorFlow path
                self.model = tf.keras.models.load_model(self.MODEL_PATH)
                
                # Load class names
                if Path(self.CLASS_INDICES_PATH).exists():
                    class_indices = np.load(self.CLASS_INDICES_PATH, allow_pickle=True).item()
                    self.class_names = list(class_indices.keys())
                else:
                    # Fallback to hardcoded class names
                    self.class_names = [
                        'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
                        'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin',
                        'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish',
                        'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird',
                        'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito',
                        'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda',
                        'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer',
                        'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel',
                        'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
                    ]
                
                # Update UI in main thread
                self.root.after(0, self.model_loaded_successfully)
                
            except Exception as e:
                self.root.after(0, lambda: self.model_load_failed(str(e)))
        
        # Start loading in background thread
        threading.Thread(target=load_in_background, daemon=True).start()
        
    def model_loaded_successfully(self):
        """Called when model is successfully loaded"""
        self.status_label.config(text="Model loaded successfully! Ready to classify.")
        
    def model_load_failed(self, error_msg):
        """Called when model loading fails"""
        self.status_label.config(text="Failed to load model")
        messagebox.showerror("Model Error", 
                           f"Failed to load the trained model:\n{error_msg}\n\n"
                           "Please ensure 'animal_classifier_model.h5' is in the same directory.")
        
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an animal image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                self.load_and_display_image(file_path)
                self.classify_btn.config(state='normal')
                self.reset_results()
                
            except Exception as e:
                messagebox.showerror("Image Error", 
                                   f"Failed to load image:\n{str(e)}")
    
    def load_and_display_image(self, file_path):
        """Load and display the selected image"""
        # Load original image
        self.current_image = Image.open(file_path)
        
        # Create display version (square, maintaining aspect ratio)
        display_image = self.current_image.copy()
        display_image = ImageOps.fit(display_image, (380, 380), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage for tkinter
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Clear canvas and display image
        self.image_canvas.delete("all")
        self.image_canvas.create_image(200, 200, image=self.photo)
        
        # Prepare image for classification
        self.processed_image = self.preprocess_image(self.current_image)
        
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if necessary
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
        
    def classify_image(self):
        """Classify the uploaded image"""
        if self.model is None:
            messagebox.showerror("Model Error", "Model not loaded yet!")
            return
            
        if self.processed_image is None:
            messagebox.showerror("Image Error", "No image to classify!")
            return
        
        def classify_in_background():
            try:
                # Update UI to show processing
                self.root.after(0, self.start_processing)
                
                # Make prediction
                predictions = self.model.predict(self.processed_image, verbose=0)
                probabilities = predictions[0]
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_probs = probabilities[top_indices]
                top_classes = [self.class_names[i] for i in top_indices]
                
                # Update UI with results
                self.root.after(0, lambda: self.display_results(top_classes, top_probs))
                
            except Exception as e:
                self.root.after(0, lambda: self.classification_failed(str(e)))
        
        # Start classification in background thread
        threading.Thread(target=classify_in_background, daemon=True).start()
        
    def start_processing(self):
        """Show processing indicator"""
        self.processing_label.config(text="Analyzing image...")
        self.classify_btn.config(state='disabled')
        
    def display_results(self, top_classes, top_probs):
        """Display classification results"""
        # Primary result
        best_class = top_classes[0].title().replace('_', ' ')
        best_confidence = top_probs[0] * 100
        
        self.result_label.config(text=f"{best_class}")
        self.confidence_label.config(text=f"{best_confidence:.1f}%")
        
        # Update progress bar
        self.confidence_bar['value'] = best_confidence
        
        # Top 3 predictions
        for i, (class_name, prob) in enumerate(zip(top_classes, top_probs)):
            confidence_pct = prob * 100
            formatted_name = class_name.title().replace('_', ' ')
            self.top3_labels[i].config(
                text=f"{i+1}. {formatted_name}: {confidence_pct:.1f}%"
            )
        
        # Reset UI state
        self.processing_label.config(text="")
        self.classify_btn.config(state='normal')
        
        # Color code confidence
        if best_confidence >= 80:
            self.confidence_label.config(foreground='#27ae60')  # Green for high confidence
        elif best_confidence >= 60:
            self.confidence_label.config(foreground='#f39c12')  # Orange for medium confidence
        else:
            self.confidence_label.config(foreground='#e74c3c')  # Red for low confidence
        
    def classification_failed(self, error_msg):
        """Handle classification errors"""
        self.processing_label.config(text="")
        self.classify_btn.config(state='normal')
        messagebox.showerror("Classification Error", 
                           f"Failed to classify image:\n{error_msg}")
        
    def reset_results(self):
        """Reset all result displays"""
        self.result_label.config(text="No prediction yet")
        self.confidence_label.config(text="---%", foreground='#3498db')
        self.confidence_bar['value'] = 0
        
        for label in self.top3_labels:
            label.config(text="---")

def main():
    """Main application entry point"""
    try:
        # Create and run the application
        root = tk.Tk()
        app = AnimalClassifierApp(root)
        
        # Handle window closing
        def on_closing():
            root.quit()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Application Error", 
                           f"Failed to start application:\n{str(e)}")

if __name__ == "__main__":
    main()