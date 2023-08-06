import numpy as np
from PIL import Image
import glob

# Get a list of all images that contains 'raw' in filename
images_list = glob.glob('first_high_depth_inferences/*raw*.png')  # change the extension as per your files

# Create a list to store all the images
stack = []

for image in images_list:
    # Open image and convert to numpy array
    img = Image.open(image)
    
    print(image)

    # Check the size of the image
    width, height = img.size
    max_size = max(width, height)

    # If the max size is greater than 5000, resize the image
    if max_size > 5000:
        if width > height:
            new_width = 5000
            new_height = int((new_width / width) * height)
        else:
            new_height = 5000
            new_width = int((new_height / height) * width)

        img = img.resize((new_width, new_height))

    img = np.array(img)
    stack.append(img)

# Convert list to numpy array
stack = np.array(stack)

# Perform max operation on the 3D image stack
max_img = np.max(stack, axis=0)

# Convert numpy array to Image object
max_img = Image.fromarray(max_img)

# Save the image
max_img.save('max_img.png')
