import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import logging
import matplotlib.pyplot as plt
import os

# 1. Setup logging so we can track what the module is doing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Loads the Depth Anything V2 Small model into memory.
        """
        logger.info(f"Loading depth model: {model_id}")
        
        # Load the processor (handles image resizing and normalization)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Load the model architecture and weights
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move the model to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model successfully loaded on {self.device}")

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Takes a PIL Image, runs it through the model, and returns a NumPy depth map.
        """
        # Preprocess: resize, normalize, and convert to PyTorch tensor
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Inference: disable gradient calculation to save memory
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Post-process: resize the depth map back to the original image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1], # (height, width)
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert from PyTorch tensor to NumPy array
        depth_map = prediction.squeeze().cpu().numpy()
        
        return depth_map

# This block only runs if we execute this file directly
if __name__ == "__main__":
    logger.info("Starting depth estimation test...")
    
    # 1. Define the path to your real image
    image_path = "datasets/test_image.jpg" 
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}! Please put an image there.")
        exit()
        
    # 2. Load the real image
    real_image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image with size: {real_image.size}")
    
    # 3. Initialize and predict
    estimator = DepthEstimator()
    depth_result = estimator.predict(real_image)
    
    logger.info(f"Success! Output shape: {depth_result.shape}")
    logger.info(f"Min depth: {depth_result.min():.4f}, Max depth: {depth_result.max():.4f}")
    
    # 4. Visualize the results using Matplotlib
    # We create a side-by-side plot: Original on left, Depth on right
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Original image
    axes[0].imshow(real_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Right: Depth map
    # We use 'inferno' colormap. 
    # In Depth Anything V2, BRIGHTER colors (yellow) = CLOSER, DARKER (purple) = FARTHER.
    im = axes[1].imshow(depth_result, cmap="inferno")
    axes[1].set_title("Depth Map (Brighter = Closer)")
    axes[1].axis("off")
    
    # Add a colorbar to show the scale
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Save the plot to the experiments folder
    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/depth_visualization.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"Visualization saved to {output_path}")