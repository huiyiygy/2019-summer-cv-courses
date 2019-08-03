# Classical Machine Learning I - Data Augmentation
## Task
Combine bellow functions together to complete a data augmentation script.
1. image crop
2. color shift
3. gamma correction
4. histogram equalized
5. rotation
6. affine transform
7. perspective transform

## Usage
1. Using cv2 load image and change image channel from BGR to RGB.
2. Define data augmentation tools from DataAugmentation.py
```python
from DataAugmentation import DataAugmentation
data_augment = DataAugmentation()
```
3. Using data augmentation tools to finish these functions.