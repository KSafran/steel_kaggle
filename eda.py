from src.image_utils import id_to_mask
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_id = '0002cc93b.jpg'
img = mpimg.imread(f'data/train/{img_id}')
# This image has a class 1 defect
mask = id_to_mask(img_id)[:, :, 0]

plt.figure(1)
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(mask)
plt.show()
