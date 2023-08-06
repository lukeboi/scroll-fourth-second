import numpy as np
import pandas as pd
from PIL import Image

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# Load RLE CSV data
data = pd.read_csv("submission.csv")

# Decode the image
for i, row in data.iterrows():
    # img = rle_decode(row['Predicted'], shape=(2727, 6330))
    img = rle_decode(row['Predicted'], shape=(5454, 6330))
    # Do whatever you want with the image
    img_pil = Image.fromarray((img * 255).astype(np.uint8))  # Convert to 8-bit pixel values
    img_pil.save(f'image_{i}.png')