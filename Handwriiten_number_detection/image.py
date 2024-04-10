import cv2
import numpy as np

# Create a blank white image
image = np.ones((28, 28), dtype=np.uint8) * 255

# Draw a digit (e.g., digit 3)
cv2.putText(image, "3", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

# Save the image
cv2.imwrite("digit_3.png", image)
