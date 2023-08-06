import os
from PIL import Image
import numpy as np
from skimage.transform import rescale

# Define the dimensions for cropping (left, top, right, bottom)
crop_dimensions = (1629, 1449, 1629+1280, 1449+1792)

# Directory where the images are located
in_directory = 'promising_fragment'
out_directory = 'promising_cropped_7_2/surface_volume'


# Loop through the images
for filename in os.listdir(in_directory):
    if filename.endswith('.tif'):
        # Open the image
        with Image.open(os.path.join(in_directory, filename)) as img:
            # Rotate and crop the image
            img = np.rot90(img, k=-1, axes=(0, 1))

            cropped_img = Image.fromarray(img).crop(crop_dimensions)

            # # Convert cropped image to numpy array for scaling
            # numpy_image = np.array(cropped_img)
            
            # # Scale it up by 2x
            # scaled_image = rescale(numpy_image, 2, mode='reflect', multichannel=False)
            
            # # Convert back to PIL Image and convert mode
            # final_image = Image.fromarray(scaled_image.astype(np.uint16), 'I;16')
            
            # Get the filename without extension
            base_name = os.path.splitext(filename)[0]
            # Generate next image name
            # next_image_name = str(int(base_name) + 1).zfill(2)
            
            # Save the image twice with different names
            cropped_img.save(os.path.join(out_directory, f'{base_name}.tif'))
            # cropped_img.save(os.path.join(out_directory, f'{next_image_name}.tif'))
