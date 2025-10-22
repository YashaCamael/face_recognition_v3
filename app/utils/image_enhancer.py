import cv2
import numpy as np
import logging

def is_low_light(img_array: np.ndarray, threshold: int = 70) -> bool:
    """
    Checks if an image is low-light by calculating its average brightness.
    
    Args:
        img_array: The input image (as a BGR NumPy array).
        threshold: The brightness value (0-255) to check against.
                   If the average brightness is < this, it's considered low-light.

    Returns:
        True if the image is low-light, False otherwise.
    """
    try:
        # Convert the image to Grayscale to get a single brightness channel
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Calculate the mean (average) brightness of all pixels
        average_brightness = gray.mean()
        
        if average_brightness < threshold:
            logging.info(f"Low-light detected. Average brightness: {average_brightness:.2f} < {threshold}")
            return True
        else:
            logging.info(f"Image is well-lit. Average brightness: {average_brightness:.2f} >= {threshold}")
            return False
            
    except Exception as e:
        logging.error(f"Error in is_low_light check: {e}")
        # Default to False (assume it's not low-light) if check fails
        return False
    
def preprocess_low_light(img_array: np.ndarray) -> np.ndarray:
    """
    Applies Option 1: Denoise -> Enhance (CLAHE) to an image.
    
    Args:
        img_array: A NumPy array representing the image (in BGR format).

    Returns:
        A NumPy array representing the preprocessed image (in BGR format).
    """
    try:
        # 1. --- Denoise the image (Non-Local Means) ---
        
        # --- TUNING PARAMETER 1: Denoising Strength (h) ---
        # This is the 'h' parameter (the first '10'). It's the most important
        # setting for noise reduction.
        #
        # * Increase this value (e.g., 12, 15, 20) if images are still
        #     very noisy or grainy.
        # * Decrease this value (e.g., 8, 5) if faces start to look
        #     too "waxy," "plasticky," or "unnatural" (i.e., too much
        #     detail is being blurred).
        #
        # The second '10' is hForColorComponents, which we can keep matched.
        # 7 and 21 are templateWindowSize and searchWindowSize (standard defaults).
        logging.info("Applying Denoising...")
        denoised_img = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

        # 2. --- Enhance the denoised image (CLAHE) ---
        # Convert to LAB color space to separate Lightness from Color
        logging.info("Applying CLAHE...")
        lab = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2LAB)
        
        # Split the L (Lightness), A, and B channels
        l, a, b = cv2.split(lab)
        
        # --- TUNING PARAMETER 2: Contrast Limit (clipLimit) ---
        # This controls how much the contrast is enhanced. It's a "limit"
        # to prevent the algorithm from over-amplifying any single area.
        #
        # * Increase this value (e.g., 3.0, 4.0) if faces are still
        #     too dark or details are not visible.
        # * Decrease this value (e.g., 1.5, 1.0) if the image looks
        #     too "harsh," "over-sharpened," or if noise artifacts
        #     (that denoising missed) are being amplified.
        
        # --- TUNING PARAMETER 3: Tile Grid Size (tileGridSize) ---
        # This defines the size of the "local regions" for enhancement.
        # (8, 8) is a standard default and works well for most cases.
        # You probably don't need to change this unless you have very
        # specific image sizes (e.g., tiny thumbnails).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE to *only* the Lightness channel
        enhanced_l = clahe.apply(l)
        
        # Merge the enhanced Lightness channel back with original A and B channels
        merged_lab = cv2.merge((enhanced_l, a, b))
        
        # Convert the LAB image back to BGR
        final_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        
        logging.info("Preprocessing complete.")
        return final_img

    except cv2.error as e:
        logging.error(f"OpenCV error during preprocessing: {e}")
        # If preprocessing fails, just return the original image
        return img_array
    except Exception as e:
        logging.error(f"Unexpected error during preprocessing: {e}")
        # If preprocessing fails, just return the original image
        return img_array