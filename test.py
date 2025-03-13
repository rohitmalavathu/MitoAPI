import requests
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

# API endpoint
API_URL = "http://localhost:8000"

def segment_image(image_path, x1, y1, x2, y2):
    """
    Send an image to the segmentation API with bounding box coordinates.
    
    Args:
        image_path: Path to the image file
        x1, y1: Top-left coordinates of bounding box
        x2, y2: Bottom-right coordinates of bounding box
        
    Returns:
        API response JSON
    """
    # Prepare the files and data
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/tiff')}
        data = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        
        # Make the request
        response = requests.post(f"{API_URL}/segment/", files=files, data=data)
        
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def display_result(result_json):
    """
    Display the results from the API.
    
    Args:
        result_json: API response JSON containing paths to result images
    """
    if not result_json:
        return
    
    # Download visualization image
    vis_response = requests.get(f"{API_URL}/result/{os.path.basename(result_json['visualization_path'])}")
    vis_image = Image.open(io.BytesIO(vis_response.content))
    
    # Download mask image
    mask_response = requests.get(f"{API_URL}/result/{os.path.basename(result_json['mask_path'])}")
    mask_image = Image.open(io.BytesIO(mask_response.content))
    
    # Download overlay image
    overlay_response = requests.get(f"{API_URL}/result/{os.path.basename(result_json['overlay_path'])}")
    overlay_image = Image.open(io.BytesIO(overlay_response.content))
    
    # Display images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Visualization')
    plt.imshow(vis_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(mask_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(overlay_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "your_image.tif"
    
    # Example bounding box coordinates (in 512x512 coordinate space)
    x1, y1 = 100, 100
    x2, y2 = 200, 200
    
    # Call the API
    result = segment_image(image_path, x1, y1, x2, y2)
    
    # Display results
    if result:
        print(f"Segmentation completed successfully!")
        print(f"Mask size: {result['segmentation_size']['width']} x {result['segmentation_size']['height']}")
        display_result(result)