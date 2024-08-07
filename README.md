## README

# Image Scaling and Rotation Using Nearest Neighbor and Bilinear Interpolation

This project demonstrates image processing techniques including scaling and rotating images using nearest neighbor and bilinear interpolation methods. The code is intended to run in Google Colab, utilizing OpenCV and NumPy for image manipulations.

## Requirements

- Google Colab environment
- OpenCV
- NumPy

## Usage

### Reading and Displaying the Image

The image `Eye.jpg` is read using OpenCV's `cv2.imread()` function and displayed using `cv2_imshow()`.

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

path = 'Eye.jpg'
image = cv2.imread(path)
cv2_imshow(image)
```

### Scaling the Image

Two methods are provided for scaling the image:

1. **Nearest Neighbor Scaling**

    The function `scale_nearest_neighbor` scales the image by a given scale factor using the nearest neighbor method.

    ```python
    def scale_nearest_neighbor(image, scale_factor):
        height, width, channels = image.shape
        scaled_height = int(height * scale_factor)
        scaled_width = int(width * scale_factor)
        scaled_image = np.zeros((scaled_height, scaled_width, channels), dtype=image.dtype)
        for i in range(scaled_height):
            for j in range(scaled_width):
                scaled_image[i, j] = image[int(i / scale_factor), int(j / scale_factor)]
        return scaled_image
    ```

2. **Bilinear Interpolation Scaling**

    The function `scale_bilinear_interpolation` scales the image by a given scale factor using bilinear interpolation.

    ```python
    def scale_bilinear_interpolation(image, scale_factor):
        height, width, channels = image.shape
        scaled_height = int(height * scale_factor)
        scaled_width = int(width * scale_factor)
        scaled_image = np.zeros((scaled_height, scaled_width, channels), dtype=image.dtype)
        for i in range(scaled_height):
            for j in range(scaled_width):
                x = j / scale_factor
                y = i / scale_factor

                x0 = int(x)
                x1 = min(x0 + 1, width - 1)
                y0 = int(y)
                y1 = min(y0 + 1, height - 1)

                a = x - x0
                b = y - y0

                for c in range(channels):
                    value = (1 - a) * (1 - b) * image[y0, x0, c] + a * (1 - b) * image[y0, x1, c] + (1 - a) * b * image[y1, x0, c] + a * b * image[y1, x1, c]
                    scaled_image[i, j, c] = int(value)
        return scaled_image
    ```

Both scaling methods are applied to the image by a factor of 1.5 and saved to disk.

```python
scaled_nn = scale_nearest_neighbor(image, 1.5)
scaled_bi = scale_bilinear_interpolation(image, 1.5)

def save_image(image, file_path):
    cv2.imwrite(file_path, image)

save_image(scaled_nn, 'scaled_nn.jpg')
save_image(scaled_bi, 'scaled_bi.jpg')
```

### Rotating the Image

Two methods are provided for rotating the image:

1. **Nearest Neighbor Rotation**

    The function `nearest_neighbor_rotation` rotates the image by a given angle using the nearest neighbor method.

    ```python
    def nearest_neighbor_rotation(image, angle):
        angle_rad = np.deg2rad(angle)
        height, width, channels = image.shape
        new_height = int(abs(width * np.sin(angle_rad)) + abs(height * np.cos(angle_rad)))
        new_width = int(abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad)))

        rotated_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        cx, cy = width // 2, height // 2
        ncx, ncy = new_width // 2, new_height // 2

        for i in range(new_height):
            for j in range(new_width):
                x = (j - ncx) * np.cos(angle_rad) + (i - ncy) * np.sin(angle_rad) + cx
                y = -(j - ncx) * np.sin(angle_rad) + (i - ncy) * np.cos(angle_rad) + cy

                if 0 <= x < width and 0 <= y < height:
                    rotated_image[i, j] = image[int(y), int(x)]

        return rotated_image
    ```

2. **Bilinear Interpolation Rotation**

    The function `bilinear_rotation` rotates the image by a given angle using bilinear interpolation.

    ```python
    def bilinear_rotation(image, angle):
        angle_rad = np.deg2rad(angle)
        height, width, channels = image.shape
        new_height = int(abs(width * np.sin(angle_rad)) + abs(height * np.cos(angle_rad)))
        new_width = int(abs(width * np.cos(angle_rad)) + abs(height * np.sin(angle_rad)))

        rotated_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
        cx, cy = width // 2, height // 2
        ncx, ncy = new_width // 2, new_height // 2

        for i in range(new_height):
            for j in range(new_width):
                x = (j - ncx) * np.cos(angle_rad) + (i - ncy) * np.sin(angle_rad) + cx
                y = -(j - ncx) * np.sin(angle_rad) + (i - ncy) * np.cos(angle_rad) + cy

                if 0 <= x < width and 0 <= y < height:
                    x0 = int(x)
                    x1 = min(x0 + 1, width - 1)
                    y0 = int(y)
                    y1 = min(y0 + 1, height - 1)

                    a = x - x0
                    b = y - y0

                    for c in range(channels):
                        value = (1 - a) * (1 - b) * image[y0, x0, c] + a * (1 - b) * image[y0, x1, c] + (1 - a) * b * image[y1, x0, c] + a * b * image[y1, x1, c]
                        rotated_image[i, j, c] = int(value)

        return rotated_image
    ```

Both rotation methods are applied to the image by 45 degrees and saved to disk.

```python
rotated_bi = bilinear_rotation(image, 45)
rotated_nn = nearest_neighbor_rotation(image, 45)

save_image(rotated_nn, 'rotated_nn.jpg')
save_image(rotated_bi, 'rotated_bi.jpg')
```

### Displaying the Results

The scaled and rotated images are displayed using `cv2_imshow()`.

```python
display(scaled_nn)
display(scaled_bi)
display(rotated_nn)
display(rotated_bi)
```

## Conclusion

This project provides implementations of scaling and rotating images using nearest neighbor and bilinear interpolation methods. The resulting images are displayed and saved for further use.
