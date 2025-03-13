import os
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from scipy.ndimage import gaussian_filter
import sys
import uuid
from PIL import Image
import io
import base64
from typing import Optional, List
import matplotlib.pyplot as plt

# Add the sam2 package to the path if it's in a separate directory
# sys.path.append("./path/to/sam2/directory")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI(title="SAM2 Segmentation API", 
              description="API for image segmentation using SAM2 model",
              version="1.0.0")

# Create output directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load SAM2 model at startup
@app.on_event("startup")
async def load_model():
    global sam2_model, predictor
    
    try:
        FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_sam2_2000.torch"
        sam2_checkpoint = "sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        
        # Make sure these files exist
        for file_path in [FINE_TUNED_MODEL_WEIGHTS, sam2_checkpoint, model_cfg]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required model file not found: {file_path}")
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location=torch.device('cpu')))
        print("SAM2 model loaded successfully")
    except Exception as e:
        print(f"Error loading SAM2 model: {str(e)}")
        raise e

def process_image(image_path, x1, y1, x2, y2):
    """Process an image with SAM2 model using the specified bounding box."""
    try:
        # Load and resize image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        image = cv2.resize(original_image, (512, 512))
        
        # Calculate modifications for the original image dimensions
        height_ratio = original_image.shape[0] / 512
        width_ratio = original_image.shape[1] / 512
        
        modifiedx1 = int(x1 * width_ratio)
        modifiedx2 = int(x2 * width_ratio)
        modifiedy1 = int(y1 * height_ratio)
        modifiedy2 = int(y2 * height_ratio)
        
        # Crop images
        cropped_image_og = original_image[modifiedy1:modifiedy2, modifiedx1:modifiedx2]
        cropped_image = image[y1:y2, x1:x2]
        
        # Get dimensions and resize
        original_height, original_width = cropped_image.shape[:2]
        cropped_image = cv2.resize(cropped_image, (256, 256))
        
        # Ensure image is 3-channel
        if cropped_image.ndim == 2:
            cropped_image = np.stack([cropped_image] * 3, axis=-1)
        
        # Use center point for prediction
        input_points = [[[128, 128]]]
        
        # Predict masks
        with torch.no_grad():
            predictor.set_image(cropped_image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=[[1]]
            )
        
        # Process masks
        sorted_masks = masks[np.argsort(scores)][::-1]
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
        
        for i in range(sorted_masks.shape[0]):
            mask = sorted_masks[i]
            if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
                continue
                
            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = i + 1
            occupancy_mask[mask_bool] = True
        
        # Process the segmentation map
        seg_mask = gaussian_filter(seg_map.astype(float), sigma=2)
        smoothed_mask = (seg_mask > 0.5).astype(np.uint8)
        segmentation_resized = cv2.resize(smoothed_mask, (original_width, original_height))
        
        # Create visualization images
        # 1. First visualization (original with outline)
        segmentation_full_size = np.zeros((512, 512), dtype=np.uint8)
        segmentation_full_size[y1:y2, x1:x2] = cv2.resize(smoothed_mask, (x2-x1, y2-y1)) * 255
        
        contours, _ = cv2.findContours(segmentation_full_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert to RGB for visualization
        image_vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_vis = image_vis.copy()
        
        # Draw yellow contours
        cv2.drawContours(overlay_vis, contours, -1, (255, 255, 0), 2)
        
        # 2. Second visualization (cropped with red overlay)
        cropped_image_og_rgb = cv2.cvtColor(cropped_image_og, cv2.COLOR_BGR2RGB)
        segmentation_for_overlay = cv2.resize(smoothed_mask, (cropped_image_og.shape[1], cropped_image_og.shape[0]))
        
        # Create a red mask (semi-transparent)
        red_mask = np.zeros_like(cropped_image_og_rgb)
        red_mask[segmentation_for_overlay == 1] = [255, 0, 0]  # Red color
        
        # Create overlay image with red mask
        overlay = cropped_image_og_rgb.copy()
        alpha = 0.25  # Transparency factor
        mask_bool = segmentation_for_overlay.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(cropped_image_og_rgb[mask_bool], 1-alpha, red_mask[mask_bool], alpha, 0)
        
        # Save the mask data for return
        mask_data = segmentation_for_overlay.astype(np.uint8) * 255
        
        # Create output images
        result_id = str(uuid.uuid4())
        vis_path = f"results/{result_id}_visualization.png"
        mask_path = f"results/{result_id}_mask.png"
        overlay_path = f"results/{result_id}_overlay.png"
        
        # Create a figure with two subplots
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Original with Segmentation Outline')
        plt.imshow(overlay_vis)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Cropped with Red Overlay')
        plt.imshow(overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close()
        
        # Save mask
        cv2.imwrite(mask_path, mask_data)
        
        # Save overlay
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_path, overlay_bgr)
        
        return {
            "success": True,
            "visualization_path": vis_path,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "segmentation_size": {
                "width": segmentation_for_overlay.shape[1],
                "height": segmentation_for_overlay.shape[0]
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/segment/")
async def segment_image(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...)
):
    """
    Segment an image using SAM2 with a bounding box.
    
    Parameters:
    - file: Image file to process (TIF format recommended)
    - x1, y1: Top-left coordinates of bounding box (in 512x512 coordinate space)
    - x2, y2: Bottom-right coordinates of bounding box (in 512x512 coordinate space)
    
    Returns:
    - Paths to visualization, mask, and overlay images
    - Segmentation size information
    """
    try:
        # Save uploaded file
        file_path = f"uploads/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process the image
        result = process_image(file_path, x1, y1, x2, y2)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{filename}")
async def get_result(filename: str):
    """Retrieve a result file by filename."""
    file_path = f"results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/health")
async def health_check():
    """Check if the API is running and model is loaded."""
    return {"status": "healthy", "model_loaded": "predictor" in globals()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)