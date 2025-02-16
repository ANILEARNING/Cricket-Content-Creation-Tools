import streamlit as st
import cv2
import numpy as np
from skimage.restoration import inpaint
from skimage.filters import threshold_otsu
from PIL import Image
import io

def remove_watermark(image):
    # Convert to NumPy array
    image = np.array(image)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # **Step 1: Detect the Watermark**
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)  # Edge detection
    thresh = threshold_otsu(gray)  # Otsu's automatic thresholding
    watermark_mask = (gray > thresh).astype(np.uint8) * 255  # Binary mask

    # Combine edge detection & threshold mask
    combined_mask = cv2.bitwise_or(edges, watermark_mask)

    # **Step 2: Refine Mask with Morphology**
    kernel = np.ones((3,3), np.uint8)
    refined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    # **Step 3: Apply Advanced Inpainting (Fast Marching Method)**
    image_filled = inpaint.inpaint_biharmonic(image, refined_mask, multichannel=True)

    return (image_filled * 255).astype(np.uint8)  # Convert back to uint8

# **Streamlit App**
st.title("🖼️ Advanced Watermark Removal App")
st.write("Upload an image with a watermark, and this app will automatically detect and remove it.")

# **File Uploader**
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image
    if st.button("Remove Watermark"):
        with st.spinner("Processing... Please wait."):
            cleaned_image = remove_watermark(image)

            # Convert to PIL Image for saving
            cleaned_pil = Image.fromarray(cleaned_image)

            # Display result
            st.image(cleaned_pil, caption="Watermark Removed", use_column_width=True)

            # Save image for download
            img_byte_arr = io.BytesIO()
            cleaned_pil.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            # Download button
            st.download_button(
                label="📥 Download Cleaned Image",
                data=img_byte_arr,
                file_name="cleaned_image.png",
                mime="image/png"
            )
