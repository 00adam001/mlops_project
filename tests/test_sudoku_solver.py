import os
import numpy as np
import cv2
import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# --- Test 1: Utility check for image resizing ---
def test_resize_image_to_28x28():
    dummy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    resized = cv2.resize(dummy, (28, 28))
    assert resized.shape == (28, 28)

# --- Test 2: Confirm test image can be loaded ---
def test_image_loading():
    image_path = "data/raw/s.jpg"
    assert os.path.exists(image_path), "Test image not found"
    img = cv2.imread(image_path)
    assert img is not None, "Failed to load image"
    assert len(img.shape) == 3  # RGB image check

# --- Test 3: Dummy model prediction without loading digit_model.h5 ---
def test_dummy_model_predict():
    dummy_model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation='softmax')
    ])
    dummy_input = np.random.rand(1, 28, 28)
    output = dummy_model.predict(dummy_input)
    assert output.shape == (1, 10)
    assert np.allclose(np.sum(output), 1.0, atol=1e-5)  # Softmax sanity check
