import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import json
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont # Import ImageFont for better text rendering
import xml.etree.ElementTree as ET
from pathlib import Path
import threading
import time

class AutoLabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Labeling Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0') # Light grey background

        # Variables to store paths and settings
        self.reference_image_path = tk.StringVar()
        self.reference_label_path = tk.StringVar()
        self.target_folder_path = tk.StringVar()
        self.output_image_folder = tk.StringVar()
        self.output_label_folder = tk.StringVar()
        self.score_threshold = tk.DoubleVar(value=0.8) # Default confidence threshold
        self.output_format = tk.StringVar(value="YOLO") # Default output format
        self.force_jpg_txt = tk.BooleanVar(value=False) # Option to force JPG and TXT output

        # Path to classes.txt for YOLO output (newly added for better class mapping)
        self.classes_file_path = tk.StringVar() 
        self.class_name_to_id = {} # Stores {'class_name': id} mapping for YOLO

        # Initialize GUI components
        self.setup_gui()
        
    def setup_gui(self):
        """Sets up the graphical user interface elements."""
        # Main title label
        title_label = tk.Label(self.root, text="Auto Labeling Tool", 
                               font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Main frame to hold all sections
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # --- Reference Image Section ---
        ref_frame = ttk.LabelFrame(main_frame, text="Reference Image & Label", padding=10)
        ref_frame.pack(fill=tk.X, pady=5)
        
        # Reference Image selection
        ttk.Label(ref_frame, text="Reference Image:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(ref_frame, textvariable=self.reference_image_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(ref_frame, text="Browse", command=self.browse_reference_image).grid(row=0, column=2)
        
        # Reference Label selection
        ttk.Label(ref_frame, text="Reference Label:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(ref_frame, textvariable=self.reference_label_path, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(ref_frame, text="Browse", command=self.browse_reference_label).grid(row=1, column=2)
        
        # Classes.txt path for YOLO output (important for class ID mapping)
        ttk.Label(ref_frame, text="Classes File (for YOLO output):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(ref_frame, textvariable=self.classes_file_path, width=60).grid(row=2, column=1, padx=5)
        ttk.Button(ref_frame, text="Browse", command=self.browse_classes_file).grid(row=2, column=2)

        # --- Target Folder Section ---
        target_frame = ttk.LabelFrame(main_frame, text="Target Images Folder", padding=10)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="Images Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(target_frame, textvariable=self.target_folder_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(target_frame, text="Browse", command=self.browse_target_folder).grid(row=0, column=2)
        
        # --- Output Section ---
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, pady=5)
        
        # Output Image Folder selection
        ttk.Label(output_frame, text="Output Image Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_image_folder, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_image_folder).grid(row=0, column=2)
        
        # Output Label Folder selection
        ttk.Label(output_frame, text="Output Label Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_label_folder, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_label_folder).grid(row=1, column=2)
        
        # Output Format Selection and Force JPG/TXT option
        format_frame = ttk.Frame(output_frame)
        format_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=10)
        
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT)
        self.format_combo = ttk.Combobox(format_frame, textvariable=self.output_format, # Store reference to combobox
                                     values=["YOLO", "Pascal VOC", "JSON"], state="readonly")
        self.format_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(format_frame, text="Force JPG + TXT format", 
                        variable=self.force_jpg_txt, 
                        command=self.update_output_format_state).pack(side=tk.LEFT, padx=20)
        
        # --- Processing Settings ---
        settings_frame = ttk.LabelFrame(main_frame, text="Processing Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Score Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                    variable=self.score_threshold, orient=tk.HORIZONTAL, length=200)
        threshold_scale.grid(row=0, column=1, padx=5)
        self.threshold_label = ttk.Label(settings_frame, text=f"{self.score_threshold.get():.2f}")
        self.threshold_label.grid(row=0, column=2, padx=5)
        
        # Update label text when slider moves
        def update_threshold_label(val):
            self.threshold_label.config(text=f"{float(val):.2f}")
        threshold_scale.config(command=update_threshold_label)
        
        # --- Control Buttons ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Start Auto Labeling", 
                   command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Preview Sample", 
                   command=self.preview_sample).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear All", 
                   command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # --- Progress Section ---
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # --- Log Section ---
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, state=tk.DISABLED) # Make text read-only
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.update_output_format_state() # Set initial state of combobox
        
    def log(self, message):
        """Adds a timestamped message to the log text area and ensures GUI updates."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL) # Enable writing
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END) # Scroll to the end
        self.log_text.config(state=tk.DISABLED) # Disable writing
        self.root.update_idletasks() # Force GUI update for responsiveness
            
    def update_output_format_state(self):
        """Disables/enables the output format combobox based on 'Force JPG + TXT' checkbox."""
        if self.force_jpg_txt.get():
            self.output_format.set("YOLO") # Automatically set to YOLO if forced TXT
            self.format_combo.config(state="disabled") # Use direct reference
        else:
            self.format_combo.config(state="readonly") # Use direct reference

    def browse_reference_image(self):
        """Opens a file dialog to select the reference image."""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if file_path:
            self.reference_image_path.set(file_path)
            self.log(f"Reference Image selected: {file_path}")
            
    def browse_reference_label(self):
        """Opens a file dialog to select the reference label file."""
        file_path = filedialog.askopenfilename(
            title="Select Reference Label",
            filetypes=[("Label files", "*.json *.xml *.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.reference_label_path.set(file_path)
            self.log(f"Reference Label selected: {file_path}")

    def browse_classes_file(self):
        """Opens a file dialog to select the classes.txt file for YOLO output."""
        file_path = filedialog.askopenfilename(
            title="Select Classes.txt File (for YOLO output)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            self.classes_file_path.set(file_path)
            self.log(f"Classes File selected: {file_path}")
            
    def browse_target_folder(self):
        """Opens a directory dialog to select the folder containing target images."""
        folder_path = filedialog.askdirectory(title="Select Target Images Folder")
        if folder_path:
            self.target_folder_path.set(folder_path)
            self.log(f"Target Folder selected: {folder_path}")
            
    def browse_output_image_folder(self):
        """Opens a directory dialog to select the output folder for images."""
        folder_path = filedialog.askdirectory(title="Select Output Images Folder")
        if folder_path:
            self.output_image_folder.set(folder_path)
            self.log(f"Output Image Folder selected: {folder_path}")
            
    def browse_output_label_folder(self):
        """Opens a directory dialog to select the output folder for label files."""
        folder_path = filedialog.askdirectory(title="Select Output Labels Folder")
        if folder_path:
            self.output_label_folder.set(folder_path)
            self.log(f"Output Label Folder selected: {folder_path}")
            
    def clear_all(self):
        """Clears all input fields and resets progress/log."""
        self.reference_image_path.set("")
        self.reference_label_path.set("")
        self.classes_file_path.set("") 
        self.target_folder_path.set("")
        self.output_image_folder.set("")
        self.output_label_folder.set("")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.progress_var.set("Ready")
        self.progress_bar['value'] = 0
        self.class_name_to_id = {} # Clear class mapping
        self.log("All fields cleared.")
            
    def load_reference_annotation(self, label_path):
        """
        Loads annotations from various formats (JSON, XML, YOLO TXT) into a standardized list of dictionaries.
        Each dictionary contains 'label', 'x', 'y', 'width', 'height'.
        """
        annotations = []
        try:
            ext = Path(label_path).suffix.lower()
            
            if ext == '.json':
                with open(label_path, 'r') as f:
                    data = json.load(f)
                
                # Handle common JSON formats (e.g., custom, LabelMe)
                if isinstance(data, list) and data and 'annotations' in data[0]: # Custom format: [{ "image": ..., "annotations": [...] }]
                    for annotation_entry in data[0]['annotations']:
                        coords = annotation_entry.get('coordinates', {})
                        # Validate coordinates exist and are convertible to int
                        if all(k in coords and isinstance(coords[k], (int, float)) for k in ['x', 'y', 'width', 'height']):
                            x, y, width, height = int(coords['x']), int(coords['y']), int(coords['width']), int(coords['height'])
                            if width > 0 and height > 0: # Ensure valid dimensions
                                annotations.append({
                                    'label': annotation_entry.get('label', 'object'),
                                    'x': x, 'y': y, 'width': width, 'height': height
                                })
                            else:
                                self.log(f"Warning: JSON annotation with zero or negative width/height skipped: {annotation_entry}")
                        else:
                            self.log(f"Warning: JSON annotation missing complete or valid coordinates: {annotation_entry}")
                elif isinstance(data, dict) and 'shapes' in data: # Common LabelMe format check
                    for shape in data['shapes']:
                        if shape.get('shape_type') == 'rectangle' and len(shape.get('points', [])) == 2:
                            p1 = shape['points'][0]
                            p2 = shape['points'][1]
                            xmin = int(min(p1[0], p2[0]))
                            ymin = int(min(p1[1], p2[1]))
                            xmax = int(max(p1[0], p2[0]))
                            ymax = int(max(p1[1], p2[1]))
                            width = xmax - xmin
                            height = ymax - ymin
                            if width > 0 and height > 0:
                                annotations.append({
                                    'label': shape.get('label', 'object'),
                                    'x': xmin, 'y': ymin, 'width': width, 'height': height
                                })
                            else:
                                self.log(f"Warning: LabelMe annotation with zero or negative width/height skipped: {shape}")
                        else:
                            self.log(f"Warning: LabelMe shape is not a valid rectangle or missing points: {shape}")
                else:
                    self.log(f"Warning: Unsupported JSON format in {label_path}")
                        
            elif ext == '.xml': # Pascal VOC format
                tree = ET.parse(label_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    if bbox is not None:
                        try:
                            # Convert to float first, then int, to handle potential float coordinates
                            xmin = int(float(bbox.find('xmin').text))
                            ymin = int(float(bbox.find('ymin').text))
                            xmax = int(float(bbox.find('xmax').text))
                            ymax = int(float(bbox.find('ymax').text))
                            width = xmax - xmin
                            height = ymax - ymin
                            if width > 0 and height > 0: # Ensure valid dimensions
                                annotations.append({
                                    'label': obj.find('name').text if obj.find('name') is not None else 'object',
                                    'x': xmin, 'y': ymin, 'width': width, 'height': height
                                })
                            else:
                                self.log(f"Warning: Pascal VOC annotation with zero or negative width/height skipped for object: {obj.find('name').text}")
                        except (ValueError, AttributeError) as ve:
                            self.log(f"Warning: Invalid bounding box data in XML for object: {obj.find('name').text if obj.find('name') is not None else 'N/A'}, error: {ve}")
            elif ext == '.txt': # YOLO format
                ref_img_path = self.reference_image_path.get()
                if not ref_img_path or not Path(ref_img_path).exists():
                    self.log(f"Error: Reference image path is invalid or not set: {ref_img_path}. Cannot convert YOLO annotations without image dimensions.")
                    return []
                
                ref_img = cv2.imread(ref_img_path)
                if ref_img is None:
                    self.log(f"Error: Could not load reference image at {ref_img_path} for YOLO annotation conversion. Check if the image exists and is valid.")
                    return []
                    
                h, w = ref_img.shape[:2]
                if w == 0 or h == 0:
                    self.log(f"Error: Reference image has zero dimension ({w}x{h}). Cannot convert YOLO annotations.")
                    return []

                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # class_id (ignored for label, using 'class_X'), cx, cy, nw, nh (normalized)
                                class_id = parts[0] 
                                cx, cy, nw, nh = map(float, parts[1:5])
                                
                                # Convert normalized coordinates to pixel coordinates
                                x_center = cx * w
                                y_center = cy * h
                                box_width = nw * w
                                box_height = nh * h

                                x = int(x_center - box_width / 2)
                                y = int(y_center - box_height / 2)
                                width = int(box_width)
                                height = int(box_height)
                                
                                # Basic validation for pixel coordinates
                                if width > 0 and height > 0:
                                    # Try to map class_id back to actual label if classes file is available
                                    label_from_id = f'class_{class_id}' 
                                    if self.classes_file_path.get() and Path(self.classes_file_path.get()).exists():
                                        try:
                                            with open(self.classes_file_path.get(), 'r') as cf:
                                                class_names_list = [c.strip() for c in cf if c.strip()]
                                            if int(class_id) < len(class_names_list):
                                                label_from_id = class_names_list[int(class_id)]
                                            else:
                                                self.log(f"Warning: YOLO class_id {class_id} out of bounds for classes.txt. Using default label.")
                                        except Exception as e:
                                            self.log(f"Warning: Could not map YOLO class_id {class_id} to name from {self.classes_file_path.get()}: {e}. Using default label.")

                                    annotations.append({
                                        'label': label_from_id, 
                                        'x': x, 'y': y, 'width': width, 'height': height
                                    })
                                else:
                                    self.log(f"Warning: YOLO annotation with zero or negative width/height skipped: {line.strip()}")
                            except ValueError as ve:
                                self.log(f"Warning: Could not parse YOLO line: '{line.strip()}', error: {ve}")
                        else:
                            self.log(f"Warning: YOLO line has insufficient parts (expected >=5): '{line.strip()}'")
            else:
                self.log(f"Error: Unsupported reference label format: {ext}. Supported formats are .json, .xml, .txt.")
                
        except FileNotFoundError:
            self.log(f"Error: Reference label file not found at {label_path}")
        except json.JSONDecodeError as e:
            self.log(f"Error: Invalid JSON format in {label_path}: {e}")
        except ET.ParseError as e:
            self.log(f"Error: Invalid XML format in {label_path}: {e}")
        except Exception as e:
            self.log(f"An unexpected error occurred while loading annotation from {label_path}: {e}")
            
        return annotations
            
    def save_annotation(self, image_path, annotations_list, output_folder, format_type):
        """
        Saves a list of annotations for a single image in the specified format.
        Creates an empty label file if no annotations are provided.
        """
        try:
            image_name = Path(image_path).stem
            
            # Ensure output folder exists
            os.makedirs(output_folder, exist_ok=True)

            img = cv2.imread(image_path)
            if img is None:
                self.log(f"Error: Could not load image {image_path} for saving annotation. Skipping label save.")
                return False
            h, w = img.shape[:2]

            # Determine the actual format to save (considering force_jpg_txt)
            actual_format_type = "YOLO" if self.force_jpg_txt.get() else format_type

            if actual_format_type == "YOLO":
                output_path = os.path.join(output_folder, f"{image_name}.txt")
                with open(output_path, 'w') as f:
                    for ann in annotations_list:
                        if ann['width'] > 0 and ann['height'] > 0: # Ensure valid dimensions
                            cx = (ann['x'] + ann['width'] / 2) / w
                            cy = (ann['y'] + ann['height'] / 2) / h
                            nw = ann['width'] / w
                            nh = ann['height'] / h
                            
                            # Get class_id from class_name_to_id map
                            class_id = self.class_name_to_id.get(ann['label'])
                            if class_id is None: # If label not found in mapping
                                self.log(f"Warning: Label '{ann['label']}' not found in classes.txt mapping. Using class_id 0 for YOLO output for {image_name}.")
                                class_id = 0 

                            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                        else:
                            self.log(f"Warning: Skipping annotation with zero/negative dimensions for {image_name} (YOLO): {ann}")
                
            elif actual_format_type == "Pascal VOC":
                output_path = os.path.join(output_folder, f"{image_name}.xml")
                
                c = img.shape[2] if len(img.shape) == 3 else 1 # Depth for grayscale or color

                annotation_xml = ET.Element("annotation")
                ET.SubElement(annotation_xml, "folder").text = Path(image_path).parent.name
                ET.SubElement(annotation_xml, "filename").text = Path(image_path).name
                ET.SubElement(annotation_xml, "path").text = image_path
                
                source = ET.SubElement(annotation_xml, "source")
                ET.SubElement(source, "database").text = "Unknown"
                
                size = ET.SubElement(annotation_xml, "size")
                ET.SubElement(size, "width").text = str(w)
                ET.SubElement(size, "height").text = str(h)
                ET.SubElement(size, "depth").text = str(c)
                
                ET.SubElement(annotation_xml, "segmented").text = "0" # Standard Pascal VOC field

                for ann in annotations_list:
                    if ann['width'] > 0 and ann['height'] > 0: # Ensure valid dimensions
                        obj = ET.SubElement(annotation_xml, "object")
                        ET.SubElement(obj, "name").text = ann['label']
                        ET.SubElement(obj, "pose").text = "Unspecified"
                        ET.SubElement(obj, "truncated").text = "0"
                        ET.SubElement(obj, "difficult").text = "0"
                        
                        bndbox = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bndbox, "xmin").text = str(ann['x'])
                        ET.SubElement(bndbox, "ymin").text = str(ann['y'])
                        ET.SubElement(bndbox, "xmax").text = str(ann['x'] + ann['width'])
                        ET.SubElement(bndbox, "ymax").text = str(ann['y'] + ann['height'])
                    else:
                        self.log(f"Warning: Skipping annotation with zero/negative dimensions for {image_name} (Pascal VOC): {ann}")

                tree = ET.ElementTree(annotation_xml)
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
                
            elif actual_format_type == "JSON":
                output_path = os.path.join(output_folder, f"{image_name}.json")
                
                json_annotations = []
                for ann in annotations_list:
                    if ann['width'] > 0 and ann['height'] > 0: # Ensure valid dimensions
                        json_annotations.append({
                            "label": ann['label'],
                            "coordinates": {
                                "x": ann['x'],
                                "y": ann['y'],
                                "width": ann['width'],
                                "height": ann['height']
                            }
                        })
                    else:
                        self.log(f"Warning: Skipping annotation with zero/negative dimensions for {image_name} (JSON): {ann}")

                json_data = [{
                    "image": Path(image_path).name, # Use just filename for 'image' field
                    "annotations": json_annotations
                }]
                
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
            return True
            
        except Exception as e:
            self.log(f"Error saving annotation for {image_path} in {format_type} format: {e}")
            return False
            
    def find_and_match_template(self, reference_img_path, single_reference_annotation, target_img_path):
        """
        Finds a single template (defined by a reference annotation) in a target image
        using OpenCV's template matching.
        Returns the new annotation (x, y, width, height, label) and the confidence score,
        or None and 0 if no match above threshold is found or an error occurs.
        """
        try:
            # Load reference image
            ref_img = cv2.imread(reference_img_path)
            if ref_img is None:
                self.log(f"Error: Could not load reference image for template matching: {reference_img_path}")
                return None, 0
                
            # Extract template from the reference image based on the single annotation
            x, y, w, h = single_reference_annotation['x'], single_reference_annotation['y'], single_reference_annotation['width'], single_reference_annotation['height']
            
            # Ensure template coordinates are within bounds and positive
            if w <= 0 or h <= 0 or x < 0 or y < 0 or (x + w) > ref_img.shape[1] or (y + h) > ref_img.shape[0]:
                self.log(f"Warning: Invalid template coordinates ({x},{y},{w},{h}) for label '{single_reference_annotation['label']}' in {Path(reference_img_path).name}. Skipping template matching.")
                return None, 0

            template = ref_img[y : y + h, x : x + w]
            
            if template.shape[0] == 0 or template.shape[1] == 0:
                self.log(f"Warning: Cropped template for '{single_reference_annotation['label']}' is empty (dimensions {template.shape[1]}x{template.shape[0]}). Check reference bounding box. Skipping template matching.")
                return None, 0
                
            # Load target image
            target_img = cv2.imread(target_img_path)
            if target_img is None:
                self.log(f"Error: Could not load target image for matching: {target_img_path}. Skipping template matching.")
                return None, 0
                
            # Convert to grayscale for template matching
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            
            # Ensure template is not larger than target image dimensions
            if template_gray.shape[0] > target_gray.shape[0] or template_gray.shape[1] > target_gray.shape[1]:
                self.log(f"Warning: Template '{single_reference_annotation['label']}' ({template_gray.shape[1]}x{template_gray.shape[0]}) is larger than target image ({target_gray.shape[1]}x{target_gray.shape[0]}). Skipping matching.")
                return None, 0

            # Perform template matching
            res = cv2.matchTemplate(target_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= self.score_threshold.get():
                # Match found above threshold
                new_annotation = {
                    'label': single_reference_annotation['label'],
                    'x': max_loc[0],
                    'y': max_loc[1],
                    'width': w,
                    'height': h
                }
                return new_annotation, max_val
                
            return None, max_val # No match found above threshold
            
        except Exception as e:
            self.log(f"Error in template matching for {Path(target_img_path).name} with label '{single_reference_annotation.get('label', 'N/A')}': {e}")
            return None, 0
            
    def copy_and_convert_image(self, source_path, target_folder, force_jpg=False):
        """
        Copies an image to the target folder, optionally converting it to JPG.
        Returns the path to the copied image, or None on failure.
        """
        try:
            # Ensure target folder exists
            os.makedirs(target_folder, exist_ok=True)

            image_name = Path(source_path).stem
            
            img = cv2.imread(source_path)
            if img is None:
                self.log(f"Error: Could not load source image for copying: {source_path}. Skipping copy.")
                return None

            if force_jpg:
                target_path = os.path.join(target_folder, f"{image_name}.jpg")
                # Save with high quality JPEG compression
                cv2.imwrite(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return target_path
            else:
                # Keep original format
                source_ext = Path(source_path).suffix
                target_path = os.path.join(target_folder, f"{image_name}{source_ext}")
                cv2.imwrite(target_path, img)
                return target_path
                        
        except Exception as e:
            self.log(f"Error copying image {source_path} to {target_folder}: {e}")
            return None
            
    def preview_sample(self):
        """
        Previews template matching on the first image found in the target folder.
        Draws all found bounding boxes with labels and scores on the image and displays it.
        """
        # Validate essential inputs before preview
        if not all([self.reference_image_path.get(), self.reference_label_path.get(), self.target_folder_path.get()]):
            messagebox.showerror("Error", "Please select a reference image, a reference label file, and a target folder before previewing.")
            return

        # Check for classes.txt if YOLO output is selected or forced
        if (self.output_format.get() == "YOLO" or self.force_jpg_txt.get()):
            if not self.classes_file_path.get():
                messagebox.showerror("Error", "Please select a 'classes.txt' file for YOLO output format, or uncheck 'Force JPG + TXT format'.")
                return
            else:
                self.load_classes_map(self.classes_file_path.get())
                if not self.class_name_to_id:
                    messagebox.showerror("Error", "Could not load class names from classes.txt. Check file content and try again.")
                    return
            
        try:
            # Load all reference annotations
            ref_annotations = self.load_reference_annotation(self.reference_label_path.get())
            if not ref_annotations:
                messagebox.showerror("Error", "Could not load any reference annotations from the provided label file. Please check its content and format.")
                return
                
            self.log(f"Loaded {len(ref_annotations)} reference annotations for preview.")

            # Find the first image in the target folder
            target_folder = self.target_folder_path.get()
            image_extensions = {'.jpg', '.jpeg', '.png'}
            
            target_image_path = None
            for file_name in os.listdir(target_folder):
                full_path = os.path.join(target_folder, file_name)
                if Path(full_path).is_file() and Path(file_name).suffix.lower() in image_extensions:
                    target_image_path = full_path
                    break
            
            if target_image_path is None:
                messagebox.showerror("Error", "No images found in the target folder for preview. Please check the folder path and ensure it contains .jpg, .jpeg, or .png files.")
                return
                
            # Load the target image for drawing
            img = cv2.imread(target_image_path)
            if img is None:
                messagebox.showerror("Error", f"Could not load target image for preview: {target_image_path}. It might be corrupted or an unsupported format.")
                return

            found_annotations_count = 0
            # Iterate through each reference annotation and try to find it in the target image
            for i, ref_ann in enumerate(ref_annotations):
                annotation, score = self.find_and_match_template(
                    self.reference_image_path.get(), ref_ann, target_image_path
                )
                
                if annotation:
                    found_annotations_count += 1
                    # Draw the bounding box on the image (using OpenCV)
                    cv2.rectangle(img, (annotation['x'], annotation['y']), 
                                  (annotation['x'] + annotation['width'], annotation['y'] + annotation['height']), 
                                  (0, 255, 0), 2) # Green color, 2px thickness

                    # Convert to PIL Image for drawing text (better text rendering)
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    
                    # Try to load a default font
                    try:
                        font = ImageFont.truetype("arial.ttf", 15)
                    except IOError:
                        font = ImageFont.load_default() # Fallback to default PIL font

                    text_position = (annotation['x'], annotation['y'] - 18 if annotation['y'] - 18 > 0 else annotation['y'] + 5)
                    draw.text(text_position, f"{annotation['label']} ({score:.2f})", fill=(0, 255, 0), font=font) # Green text
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # Convert back to OpenCV format

            if found_annotations_count > 0:
                # Resize for display if too large to fit screen
                height, width = img.shape[:2]
                max_display_width = 1000
                max_display_height = 800 

                if width > max_display_width or height > max_display_height:
                    scale_w = max_display_width / width
                    scale_h = max_display_height / height
                    scale = min(scale_w, scale_h)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                cv2.imshow(f"Preview - Found {found_annotations_count} object(s) in {Path(target_image_path).name}", img)
                cv2.waitKey(0) # Wait indefinitely until a key is pressed
                cv2.destroyAllWindows() # Close all OpenCV windows
                
                self.log(f"Preview: Found {found_annotations_count} match(es) in {Path(target_image_path).name}.")
            else:
                messagebox.showinfo("Preview", f"No matches found for any reference object in {Path(target_image_path).name} above score threshold {self.score_threshold.get():.2f}.")
                self.log(f"Preview: No matches found in {Path(target_image_path).name}.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during preview: {e}")
            self.log(f"Preview error: {e}")

    def load_classes_map(self, classes_file_path):
        """
        Loads class names from classes.txt and creates a name-to-id mapping.
        This is crucial for generating YOLO format labels.
        """
        self.class_name_to_id = {}
        try:
            if not Path(classes_file_path).exists():
                self.log(f"Error: classes.txt file not found at {classes_file_path}. Cannot create class ID mapping.")
                return
            
            with open(classes_file_path, 'r') as f:
                for i, line in enumerate(f):
                    class_name = line.strip()
                    if class_name:
                        self.class_name_to_id[class_name] = i
            self.log(f"Loaded {len(self.class_name_to_id)} classes from {classes_file_path}.")
        except Exception as e:
            self.log(f"Error loading classes.txt from {classes_file_path}: {e}")
            self.class_name_to_id = {} # Clear in case of error
                
    def start_processing(self):
        """
        Initiates the auto-labeling process in a separate thread to keep the GUI responsive.
        Performs initial validation of all required inputs.
        """
        # Validate all required input fields
        if not all([
            self.reference_image_path.get(),
            self.reference_label_path.get(),
            self.target_folder_path.get(),
            self.output_image_folder.get(),
            self.output_label_folder.get()
        ]):
            messagebox.showerror("Error", "Please fill in all required fields (Reference Image, Reference Label, Target Images Folder, Output Image Folder, Output Label Folder).")
            return

        # Validate classes.txt path if YOLO output is selected or forced
        if (self.output_format.get() == "YOLO" or self.force_jpg_txt.get()):
            if not self.classes_file_path.get():
                messagebox.showerror("Error", "You have selected YOLO output or forced TXT format. Please select a 'classes.txt' file.")
                return
            else:
                self.load_classes_map(self.classes_file_path.get())
                if not self.class_name_to_id:
                    messagebox.showerror("Error", "Could not load class names from classes.txt. Processing aborted. Please check the file content.")
                    return
            
        # Inform user about file copying and output folders
        messagebox.showinfo("Important Information", 
                             "The tool will copy images to the specified output image folder and save new label files to the output label folder.\n\n"
                             "Your original input files will NOT be modified or deleted.\n\n"
                             "Please ensure your output folders are distinct from your input folders to avoid confusion.")

        # Create output directories if they don't exist
        try:
            os.makedirs(self.output_image_folder.get(), exist_ok=True)
            os.makedirs(self.output_label_folder.get(), exist_ok=True)
            self.log("Output directories ensured.")
        except OSError as e:
            messagebox.showerror("Error", f"Could not create output directories: {e}. Please check folder permissions or paths.")
            self.log(f"Error creating output directories: {e}")
            return
        
        # Start the heavy processing in a separate thread to prevent GUI freezing
        self.log("Starting auto-labeling process...")
        threading.Thread(target=self.process_images, daemon=True).start()
        
    def process_images(self):
        """
        Core function to process all images in the target folder.
        It loads reference annotations, performs template matching for each image,
        copies the image, and saves the generated labels.
        """
        try:
            self.progress_var.set("Loading reference annotations...")
            self.progress_bar['value'] = 0
            
            # Load ALL reference annotations
            ref_annotations = self.load_reference_annotation(self.reference_label_path.get())
            if not ref_annotations:
                self.log("Error: No reference annotations loaded. Processing aborted. Please check the reference label file and format.")
                self.progress_var.set("Processing aborted.")
                messagebox.showerror("Error", "No reference annotations loaded. Processing aborted.")
                return
                
            self.log(f"Successfully loaded {len(ref_annotations)} reference annotations.")
            
            # Get all image files in target folder
            target_folder = self.target_folder_path.get()
            image_extensions = {'.jpg', '.jpeg', '.png'} # Supported image extensions
            
            image_files = []
            for file_name in os.listdir(target_folder):
                full_path = os.path.join(target_folder, file_name)
                # Ensure it's a file and has a supported image extension
                if Path(full_path).is_file() and Path(file_name).suffix.lower() in image_extensions:
                    image_files.append(file_name)
                            
            if not image_files:
                self.log("No images found in the target folder. Please ensure it contains .jpg, .jpeg, or .png files.")
                self.progress_var.set("Processing completed (no images found).")
                messagebox.showinfo("Info", "No images found in the target folder to process. Please ensure it contains .jpg, .jpeg, or .png files.")
                return
                
            self.log(f"Found {len(image_files)} images to process in {target_folder}.")
            
            # Initialize counters for summary
            successful_images = 0
            failed_images = 0
            total_matches_found = 0
            
            self.progress_bar['maximum'] = len(image_files) # Set progress bar maximum value
            
            for i, image_file_name in enumerate(image_files):
                self.progress_var.set(f"Processing {i+1}/{len(image_files)}: {image_file_name}...")
                self.progress_bar['value'] = i + 1 # Update progress bar
                
                source_path = os.path.join(target_folder, image_file_name)
                
                found_annotations_for_current_image = []
                scores_for_current_image = []

                # Iterate through each reference annotation and try to find it in the current target image
                for ref_ann in ref_annotations:
                    annotation, score = self.find_and_match_template(
                        self.reference_image_path.get(), ref_ann, source_path
                    )
                    
                    if annotation:
                        found_annotations_for_current_image.append(annotation)
                        scores_for_current_image.append(score)
                        total_matches_found += 1
                
                # Copy image to output folder (and convert if force_jpg_txt is true)
                target_image_path = self.copy_and_convert_image(
                    source_path, self.output_image_folder.get(), 
                    self.force_jpg_txt.get()
                )
                
                if target_image_path:
                    # Determine the output label format
                    format_to_save = "YOLO" if self.force_jpg_txt.get() else self.output_format.get()
                    
                    # Save all found annotations (or an empty list if none found) for the current image
                    if self.save_annotation(target_image_path, found_annotations_for_current_image, 
                                            self.output_label_folder.get(), format_to_save):
                        successful_images += 1
                        if found_annotations_for_current_image:
                            scores_str = ", ".join([f"{s:.2f}" for s in scores_for_current_image])
                            self.log(f"✓ {image_file_name}: Found {len(found_annotations_for_current_image)} match(es) (scores: {scores_str})")
                        else:
                            self.log(f"✓ {image_file_name}: No strong match found, created empty label file.")
                    else:
                        failed_images += 1
                        self.log(f"✗ {image_file_name}: Failed to save annotation (even empty label file creation failed).")
                else:
                    failed_images += 1
                    self.log(f"✗ {image_file_name}: Failed to copy image to output folder (skipped processing for this image).")
                    
                self.root.update_idletasks() # Keep GUI responsive during the loop
                        
            self.progress_bar['value'] = len(image_files) # Ensure progress bar is full
            self.progress_var.set("Processing completed!")
            
            # Display final summary in log and a message box
            self.log(f"\n--- Processing Summary ---")
            self.log(f"Total Images Scanned: {len(image_files)}")
            self.log(f"Images Successfully Processed (with or without matches): {successful_images}")
            self.log(f"Images with Failures (e.g., failed to copy/save): {failed_images}")
            self.log(f"Total Individual Object Matches Found: {total_matches_found}")
            
            messagebox.showinfo("Processing Completed", 
                                  f"Auto-labeling process finished!\n\n"
                                  f"Images Scanned: {len(image_files)}\n"
                                  f"Successfully Processed: {successful_images}\n"
                                  f"Failed Images: {failed_images}\n"
                                  f"Total Object Matches Found: {total_matches_found}\n\n"
                                  f"Check the log for details on individual files.")
            
        except Exception as e:
            self.log(f"An unhandled error occurred during processing: {e}")
            messagebox.showerror("Fatal Error", f"Processing failed due to an unexpected error: {e}\nCheck the log for more details.")

def main():
    root = tk.Tk()
    app = AutoLabelingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
