import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import logging
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

# Import constants from config
from backend.utils.config import ALPHA_MOVING, ALPHA_STATIC, MOTION_DIFF_THRESHOLD

logger = logging.getLogger(__name__)

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Loads the Depth Anything V2 Small model and initializes EMA state.
        """
        logger.info(f"Loading depth model: {model_id}")
        
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # --- EMA TEMPORAL SMOOTHING STATE ---
        self.smoothed_depth_map = None
        
        logger.info(f"Model successfully loaded on {self.device}")

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Runs inference and applies Adaptive Exponential Moving Average (EMA) 
        to smooth temporal noise between frames.
        """
        # 1. Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 2. Raw Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 3. Post-process (resize to original image dimensions)
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1], # (height, width)
            mode="bicubic",
            align_corners=False,
        )
        
        # 4. Convert to NumPy
        raw_depth_map = prediction.squeeze().cpu().numpy()
        
        # =====================================================================
        # 5. ADAPTIVE TEMPORAL SMOOTHING (EMA)
        # =====================================================================
        if self.smoothed_depth_map is None:
            # First frame: no previous map to blend with
            self.smoothed_depth_map = raw_depth_map
        else:
            # Calculate mean absolute difference to detect motion
            # If the scene is changing rapidly (user walking), diff will be high.
            diff = np.mean(np.abs(raw_depth_map - self.smoothed_depth_map))
            
            if diff > MOTION_DIFF_THRESHOLD:
                # High motion: Trust the new frame more
                alpha = ALPHA_MOVING
            else:
                # Stationary: Smooth heavily to eliminate flicker
                alpha = ALPHA_STATIC
                
            # EMA Formula: D_smooth = (alpha * D_new) + ((1 - alpha) * D_old)
            self.smoothed_depth_map = (alpha * raw_depth_map) + ((1.0 - alpha) * self.smoothed_depth_map)
            
        return self.smoothed_depth_map

# This block only runs if we execute this file directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting depth estimation test...")
    
    image_path = "datasets/test_image.jpg" 
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}! Please put an image there.")
        exit()
        
    real_image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image with size: {real_image.size}")
    
    estimator = DepthEstimator()
    
    # Run it twice to simulate a video feed and test the EMA state
    logger.info("Frame 1 (No EMA)")
    depth_result_1 = estimator.predict(real_image)
    
    logger.info("Frame 2 (EMA applied)")
    depth_result_2 = estimator.predict(real_image)
    
    logger.info(f"Success! Output shape: {depth_result_2.shape}")
    logger.info(f"Min depth: {depth_result_2.min():.4f}, Max depth: {depth_result_2.max():.4f}")
    
    # 4. Visualize the results using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Original image
    axes[0].imshow(real_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Right: Smoothed Depth map
    # 'inferno' colormap: Brighter (yellow) = closer, Darker (purple) = farther
    im = axes[1].imshow(depth_result_2, cmap="inferno")
    axes[1].set_title("Smoothed Depth Map (EMA)")
    axes[1].axis("off")
    
    # Add a colorbar to show the scale
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Save the plot to the experiments folder
    os.makedirs("experiments", exist_ok=True)
    output_path = "experiments/depth_visualization.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"Visualization saved to {output_path}")