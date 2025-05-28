import os
from se_489_mlops_project.predict import process_sudoku


def test_real_sudoku_image():
    image_path = os.path.join("data", "raw", "s.jpg")
    result = process_sudoku(image_path)
    assert result.shape == (9, 9), f"Expected shape (9, 9), got {result.shape}"

