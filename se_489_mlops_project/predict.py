import argparse
import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from se_489_mlops_project.puzzle_utils import find_puzzle, extract_digit

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def norm(a):
    return (a / 9) - 0.5

def denorm(a):
    return (a + 0.5) * 9

def load_digit_model(model_path='models/digit_model.h5'):
    logging.info("Loading digit recognition model...")
    model = load_model(model_path)
    logging.info("Digit model loaded.")
    return model


def identify_number(image, digit_model):
    image_resized = cv2.resize(image, (28, 28))
    image_normalized = image_resized / 255.0
    image_final = image_normalized.reshape(1, 28, 28, 1)
    prediction = digit_model.predict(image_final, verbose=0)
    return np.argmax(prediction, axis=1)[0]

def inference_sudoku(board, model_path='models/sudoku_model.h5'):
    board_norm = norm(board)
    board_input = np.expand_dims(np.expand_dims(board_norm, axis=0), axis=-1)
    model = load_model(model_path)
    out = model.predict(board_input)
    pred = np.argmax(out, axis=2).reshape((9, 9)) + 1
    return pred

def solve_from_image(image_path, output_path=None):
    logging.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 600))
    (puzzleImage, warped) = find_puzzle(image, debug=False)

    digit_model = load_digit_model()
    board = np.zeros([9, 9], dtype='int')

    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    cellLocs = [[(j * stepX, i * stepY, (j + 1) * stepX, (i + 1) * stepY) for j in range(9)] for i in range(9)]

    for y in range(9):
        for x in range(9):
            (startX, startY, endX, endY) = cellLocs[y][x]
            cell = warped[startY:endY, startX:endX]
            digit_image = extract_digit(cell)
            if digit_image is None:
                continue
            digit = identify_number(digit_image, digit_model)
            board[y][x] = digit

    logging.info("Initial board extracted:")
    print(board)

    solved_board = inference_sudoku(board)
    logging.info("Solved board:")
    print(solved_board)

    for y in range(9):
        for x in range(9):
            (startX, startY, endX, endY) = cellLocs[y][x]
            textX = int((endX - startX) * 0.33) + startX
            textY = int((endY - startY) * -0.2) + endY
            cv2.rectangle(warped, (startX + 2, startY + 2), (endX - 2, endY - 2), (255, 255, 255), -1)
            digit = solved_board[y][x]
            color = (0, 0, 255) if board[y][x] == 0 else (0, 0, 0)
            cv2.putText(warped, str(digit), (textX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if output_path:
        logging.info(f"Saving result to {output_path}")
        cv2.imwrite(output_path, warped)

    plt.imshow(warped, cmap='gray')
    plt.title("Solved Sudoku")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a Sudoku from an image using a trained model.")
    parser.add_argument("--image", required=True, help="Path to input Sudoku image")
    parser.add_argument("--output", default=None, help="Path to save output visualization")
    args = parser.parse_args()
    solve_from_image(args.image, args.output)
