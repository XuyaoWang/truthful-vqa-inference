import cv2
import numpy as np
import random
from typing import Tuple, Optional, List
from PIL import Image

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)

# Set the fixed random seed
set_seed(42)


def check_overlap(box: Tuple[int, int, int, int], existing_boxes: List[Tuple[int, int, int, int]]) -> bool:
    """
    Check if a box overlaps with any existing boxes.
    
    Args:
        box: (x, y, width, height) of the new box
        existing_boxes: List of existing boxes
    
    Returns:
        True if there is overlap, False otherwise
    """
    x, y, w, h = box
    for ex_x, ex_y, ex_w, ex_h in existing_boxes:
        # Check if boxes overlap
        if not (x + w <= ex_x or ex_x + ex_w <= x or y + h <= ex_y or ex_y + ex_h <= y):
            return True
    return False

def create_occlusion(
    image: Image.Image,
    level: str = 'low',  # 'low', 'medium', or 'high'
    box_size: float = 0.2,  # Size of each box relative to image size
) -> Image.Image:
    """
    Create occlusions on an image with three fixed levels.
    
    Args:
        image: Input image (PIL.Image)
        level: Occlusion level ('low', 'medium', or 'high')
        box_size: Size of each box relative to image size (0-1)
    
    Returns:
        Image with occlusions (PIL.Image)
    """
    set_seed(42)
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Create a copy of the input image
    result = image_np.copy()
    height, width = image_np.shape[:2]
    
    # Define level parameters
    level_params = {
        'low': {'num_boxes': 5, 'alpha': 0.5},    # 5 boxes, 50% opacity
        'medium': {'num_boxes': 3, 'alpha': 0.7},  # 3 boxes, 70% opacity
        'high': {'num_boxes': 2, 'alpha': 1.0}     # 2 boxes, 100% opacity
    }
    
    if level not in level_params:
        raise ValueError("Level must be 'low', 'medium', or 'high'")
    
    params = level_params[level]
    num_boxes = params['num_boxes']
    alpha = params['alpha']
    
    # Calculate box dimensions
    box_width = int(width * box_size)
    box_height = int(height * box_size)
    
    # Track existing boxes
    existing_boxes = []
    
    for _ in range(num_boxes):
        # Try to place a box without overlap
        box_placed = False
        for _ in range(100):  # Max 100 attempts to place each box
            # Calculate random position
            x = random.randint(0, width - box_width)
            y = random.randint(0, height - box_height)
            
            # Check for overlap
            if not check_overlap((x, y, box_width, box_height), existing_boxes):
                # Generate random color
                box_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                
                # Create a temporary image for the box
                temp = result.copy()
                cv2.rectangle(temp, (x, y), (x + box_width, y + box_height), box_color, -1)
                
                # Blend the box with the original image using alpha
                result = cv2.addWeighted(temp, alpha, result, 1 - alpha, 0)
                
                # Update tracking variables
                existing_boxes.append((x, y, box_width, box_height))
                box_placed = True
                break
        
        if not box_placed:
            print(f"Warning: Could not place box {_ + 1} for level {level}")
    
    # Convert back to PIL Image
    return Image.fromarray(result)