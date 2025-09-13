from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
# ✅ Fix deprecated asscalar issue in newer numpy
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
from typing import List, Dict, Tuple
import colorsys
import os
import requests
from dotenv import load_dotenv
from typing import Any, Optional
import re
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000



# Load .env file
load_dotenv()

app = FastAPI()

# --- Google API Config ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

class JewelryItem:
    def __init__(self, title: str, url: str, rating: Optional[float], 
                 reviews_count: Optional[int], site: str, 
                 description: str, popularity_score: float):
        self.title = title
        self.url = url
        self.rating = rating
        self.reviews_count = reviews_count
        self.site = site
        self.description = description
        self.popularity_score = popularity_score

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for better perceptual color matching"""
    # Normalize RGB values
    rgb = np.array(rgb) / 255.0
    
    # Convert to XYZ
    def f(t):
        return np.where(t > 0.008856, np.power(t, 1/3), (7.787 * t) + (16/116))
    
    # sRGB to XYZ transformation matrix
    matrix = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])
    
    xyz = np.dot(matrix, rgb.reshape(-1, 1)).flatten()
    
    # Normalize by D65 illuminant
    xyz[0] /= 0.95047
    xyz[2] /= 1.08883
    
    fx, fy, fz = f(xyz)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return [L, a, b]

def color_distance_lab(color1, color2):
    """Calculate perceptual distance between colors in LAB space"""
    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)
    
    delta_e = np.sqrt(
        (lab1[0] - lab2[0]) ** 2 + 
        (lab1[1] - lab2[1]) ** 2 + 
        (lab1[2] - lab2[2]) ** 2
    )
    return delta_e

def closest_color_improved(requested_color):
    """Find closest color using perceptually uniform LAB color space"""
    min_distance = float("inf")
    closest_name = None

    # Extended color dictionary including CSS3 and additional common colors
    extended_colors = {
        **webcolors.CSS3_NAMES_TO_HEX,
        # Add some additional common colors not in CSS3
        'light_gray': '#D3D3D3',
        'dark_gray': '#A9A9A9',
        'charcoal': '#36454F',
        'cream': '#FFFDD0',
        'beige': '#F5F5DC',
        'tan': '#D2B48C',
        'coral': '#FF7F50',
        'salmon': '#FA8072',
        'peach': '#FFCBA4',
        'mint': '#98FB98',
        'lavender': '#E6E6FA',
        'rose': '#FF66CC'
    }

    for name, hex_val in extended_colors.items():
        try:
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
            distance = color_distance_lab(requested_color, [r_c, g_c, b_c])
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        except:
            continue
    
    return closest_name

def get_color_name_improved(rgb_tuple):
    """Get color name with improved matching"""
    try:
        # Try exact match first
        return webcolors.rgb_to_name(rgb_tuple, spec='css3')
    except ValueError:
        # Use improved closest color matching
        return closest_color_improved(rgb_tuple)

def preprocess_image(img):
    """Enhanced image preprocessing for better color extraction"""
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Enhance contrast slightly
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def filter_similar_colors(centers, labels, min_distance_threshold=30):
    """Remove very similar colors and merge them"""
    filtered_centers = []
    filtered_labels = labels.copy()
    color_mapping = {}
    
    for i, center in enumerate(centers):
        is_similar = False
        for j, filtered_center in enumerate(filtered_centers):
            if color_distance_lab(center, filtered_center) < min_distance_threshold:
                # Map this color to the existing similar color
                color_mapping[i] = j
                is_similar = True
                break
        
        if not is_similar:
            color_mapping[i] = len(filtered_centers)
            filtered_centers.append(center)
    
    # Update labels based on color mapping
    for i in range(len(filtered_labels)):
        old_label = filtered_labels[i]
        filtered_labels[i] = color_mapping[old_label]
    
    return np.array(filtered_centers), filtered_labels

def get_dominant_colors_improved(img, n_colors=6):
    """Extract dominant colors with improved clustering"""
    # Preprocess image
    processed_img = preprocess_image(img)
    
    # Convert to different color space for better clustering
    hsv = cv2.cvtColor(processed_img, cv2.COLOR_RGB2HSV)
    
    # Reshape for clustering
    pixels_rgb = processed_img.reshape((-1, 3))
    pixels_hsv = hsv.reshape((-1, 3))
    
    # Remove very dark and very light pixels (likely shadows/highlights)
    brightness_mask = (pixels_hsv[:, 2] > 30) & (pixels_hsv[:, 2] < 240)
    filtered_pixels_rgb = pixels_rgb[brightness_mask]
    
    if len(filtered_pixels_rgb) < 10:  # Fallback if too few pixels
        filtered_pixels_rgb = pixels_rgb
    
    # Use K-means clustering with better initialization
    kmeans = KMeans(
        n_clusters=min(n_colors, len(filtered_pixels_rgb)), 
        random_state=42,
        init='k-means++',
        n_init=10,
        max_iter=300
    ).fit(filtered_pixels_rgb)
    
    # Get labels for all pixels (not just filtered ones)
    all_labels = kmeans.predict(pixels_rgb)
    centers = kmeans.cluster_centers_
    
    # Filter out very similar colors
    filtered_centers, filtered_labels = filter_similar_colors(centers, all_labels)
    
    return filtered_centers, filtered_labels

def get_representative_positions(positions, max_positions=50):
    """Get representative positions to avoid overwhelming data"""
    if len(positions) <= max_positions:
        return positions
    
    # Use uniform sampling to get representative positions
    step = len(positions) // max_positions
    return positions[::step][:max_positions]

def calculate_color_percentage(labels, color_index, total_pixels):
    """Calculate what percentage of the image this color represents"""
    color_count = np.sum(labels == color_index)
    return round((color_count / total_pixels) * 100, 1)

@app.get("/")
def root():
    return {"message": "Enhanced Color Detection Backend is running ✅",
            "google_api_set": bool(GOOGLE_API_KEY),
        "google_cx_set": bool(GOOGLE_CX)
        }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        # Resize image (larger size for better accuracy, but not too large for performance)
        height, width = img.shape[:2]
        max_size = 400
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = img

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Get improved dominant colors
        centers, labels = get_dominant_colors_improved(rgb, n_colors=6)
        
        # Calculate total pixels
        total_pixels = resized.shape[0] * resized.shape[1]
        
        colors_data = []
        
        # Sort colors by frequency (most dominant first)
        label_counts = Counter(labels)
        sorted_colors = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        for color_index, count in sorted_colors:
            if color_index >= len(centers):
                continue
                
            center = centers[color_index]
            r, g, b = map(int, np.clip(center, 0, 255))
            
            # Calculate color properties
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            color_name = get_color_name_improved((r, g, b))
            percentage = calculate_color_percentage(labels, color_index, total_pixels)
            
            # Get positions where this color appears
            indices = np.where(labels == color_index)[0]
            positions = []
            for idx in indices:
                y, x = divmod(idx, resized.shape[1])
                positions.append({"x": int(x), "y": int(y)})
            
            # Get representative positions to avoid too much data
            representative_positions = get_representative_positions(positions)
            
            # Convert HSV for additional color information
            hsv_color = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_RGB2HSV)[0][0]
            hue, saturation, brightness = hsv_color
            
            colors_data.append({
                "hex": hex_color,
                "rgb": [r, g, b],
                "hsv": [int(hue * 2), int(saturation / 255 * 100), int(brightness / 255 * 100)],  # Convert to standard HSV ranges
                "colorName": color_name.replace('_', ' ').title(),
                "percentage": percentage,
                "positions": representative_positions,
                "totalPixels": len(positions)
            })
        
        return {
            "colors": colors_data,
            "imageInfo": {
                "width": resized.shape[1],
                "height": resized.shape[0],
                "totalPixels": total_pixels,
                "colorsFound": len(colors_data)
            }
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Optional: Add endpoint for color palette extraction with custom parameters
@app.post("/analyze-custom")
async def analyze_image_custom(
    file: UploadFile = File(...),
    n_colors: int = 5,
    max_positions: int = 100,
    min_percentage: float = 2.0
):
    """Custom analysis with configurable parameters"""
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        resized = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        centers, labels = get_dominant_colors_improved(rgb, n_colors=n_colors)
        total_pixels = resized.shape[0] * resized.shape[1]
        
        colors_data = []
        label_counts = Counter(labels)
        sorted_colors = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        for color_index, count in sorted_colors:
            if color_index >= len(centers):
                continue
                
            percentage = calculate_color_percentage(labels, color_index, total_pixels)
            
            # Skip colors below minimum percentage threshold
            if percentage < min_percentage:
                continue
            
            center = centers[color_index]
            r, g, b = map(int, np.clip(center, 0, 255))
            
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            color_name = get_color_name_improved((r, g, b))
            
            indices = np.where(labels == color_index)[0]
            positions = []
            for idx in indices:
                y, x = divmod(idx, resized.shape[1])
                positions.append({"x": int(x), "y": int(y)})
            
            representative_positions = get_representative_positions(positions, max_positions)
            
            colors_data.append({
                "hex": hex_color,
                "rgb": [r, g, b],
                "colorName": color_name.replace('_', ' ').title(),
                "percentage": percentage,
                "positions": representative_positions,
                "totalPixels": len(positions)
            })
        
        return {
            "colors": colors_data,
            "parameters": {
                "n_colors": n_colors,
                "max_positions": max_positions,
                "min_percentage": min_percentage
            }
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}



def extract_dominant_colors_lab(image, n_colors=3):
    """
    Extract dominant colors using KMeans in Lab color space
    for better perceptual accuracy.
    """
    # Convert to Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    pixels = lab_image.reshape((-1, 3))

    # Cluster with KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Convert cluster centers back to RGB
    rgb_centers = []
    for lab in centers:
        lab_color = LabColor(lab[0], lab[1], lab[2])
        rgb = convert_color(lab_color, sRGBColor)
        r, g, b = rgb.clamped_rgb_r * 255, rgb.clamped_rgb_g * 255, rgb.clamped_rgb_b * 255
        rgb_centers.append([r, g, b])

    return np.array(rgb_centers), labels


# 🎯 Improved color naming using ΔE comparison
def get_color_name_deltaE(rgb_tuple):
    """
    Match RGB to closest color name using CIEDE2000 (ΔE).
    """
    # Define a simple dictionary (can be expanded)
    color_dict = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Pink": (255, 192, 203),
        "Purple": (128, 0, 128),
        "Brown": (139, 69, 19),
        "Gray": (128, 128, 128),
        "Orange": (255, 165, 0),
    }

    input_lab = convert_color(sRGBColor(rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255), LabColor)
    min_diff, closest_name = float("inf"), "Unknown"

    for name, rgb in color_dict.items():
        ref_lab = convert_color(sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255), LabColor)
        diff = delta_e_cie2000(input_lab, ref_lab)
        if diff < min_diff:
            min_diff, closest_name = diff, name

    return closest_name
# ✅ Updated API
@app.post("/jewelry-search")
async def jewelry_search(
    file: UploadFile = File(...),
    jewelry_type: Optional[str] = Form(None),
    material: Optional[str] = Form(None),
    price_range: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    referenceImage: Optional[str] = Form(None)
):
    try:
        # Step 1: Extract dominant colors
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        centers, labels = extract_dominant_colors_lab(rgb, n_colors=3)

        total_pixels = rgb.shape[0] * rgb.shape[1]
        colors_data = []
        label_counts = Counter(labels)
        sorted_colors = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        for color_index, count in sorted_colors:
            if color_index >= len(centers):
                continue
            r, g, b = map(int, np.clip(centers[color_index], 0, 255))
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            color_name = get_color_name_deltaE((r, g, b))
            percentage = calculate_color_percentage(labels, color_index, total_pixels)

            colors_data.append({
                "hex": hex_color,
                "rgb": [r, g, b],
                "colorName": color_name,
                "percentage": percentage
            })

        if not colors_data:
            return {"error": "No dominant colors detected"}

        # Step 2: Pick main color
        main_color = colors_data[0]["colorName"]
        closet_colour = main_color.replace('_', ' ').title()
        print(f"[jewelry_search] Main color detected: {closet_colour}")
        print(f"[jewelry_search] Description: {description}")

        # Step 3: Build query string (kept identical to your logic)
        query_parts = []
        if description:
            query_parts.append(f"{description} jewelry for women")
            query_parts.append(f"matching with {closet_colour} outfits")
        else:
            query_parts.append(f"{closet_colour} saree matching jewelry items for women")

        if jewelry_type:
            query_parts.append(jewelry_type)
        if material:
            query_parts.append(material)
        if price_range:
            query_parts.append(price_range)
        if referenceImage:
            query_parts.append("inspired by this design")

        query = " ".join(query_parts)

        return {
            "dominantColors": colors_data,
            "query": query,
            "received": {
                "jewelry_type": jewelry_type,
                "material": material,
                "price_range": price_range,
                "description": description,
                "referenceImage": referenceImage,
            }
        }

    except Exception as e:
        return {"error": f"Jewelry search failed: {str(e)}"}

