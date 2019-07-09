# Data Augmentation
## Task
Combine bellow functions together to complete a data augmentation script.
- image crop
- color shift
- gamma correction
- histogram equalized
- rotation
- affine transform
- perspective transform 

## Usage
1. Using cv2 load image and change image channel from BGR to RGB.
2. Define data augmentation tools from DataAugmentation.py
```python
from DataAugmentation import DataAugmentation
data_augment = DataAugmentation()
```
3. Using data augmentation tools to finish these functions.