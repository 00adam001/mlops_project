# import argparse
# import os
# import cv2
# import numpy as np
# import logging
# import matplotlib.pyplot as plt
# import time
# import cProfile
# import pstats
# from rich.logging import RichHandler
# from tensorflow.keras.models import load_model
# from se_489_mlops_project.puzzle_utils import find_puzzle, extract_digit

# # Configure rich structured logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     datefmt="[%X]",
#     handlers=[RichHandler()]
# )
# logger = logging.getLogger("sudoku_logger")

# def norm(a): return (a / 9) - 0.5
# def denorm(a): return (a + 0.5) * 9

# def load_digit_model(model_path='models/digit_model.h5'):
#     logger.info("Loading digit recognition model...")
#     model = load_model(model_path)
#     logger.info("Digit model loaded.")
#     return model

# def identify_number(image, digit_model):
#     image_resized = cv2.resize(image, (28, 28))
#     image_normalized = image_resized / 255.0
#     image_final = image_normalized.reshape(1, 28, 28, 1)
#     prediction = digit_model.predict(image_final, verbose=0)
#     digit = np.argmax(prediction, axis=1)[0]
#     logger.debug(f"Predicted digit: {digit}")
#     return digit

# def inference_sudoku(board, model_path='models/sudoku_model.h5'):
#     logger.info("Solving the Sudoku board...")
#     board_norm = norm(board)
#     board_input = np.expand_dims(np.expand_dims(board_norm, axis=0), axis=-1)
#     model = load_model(model_path)
#     out = model.predict(board_input)
#     pred = np.argmax(out, axis=2).reshape((9, 9)) + 1
#     return pred

# def solve_from_image(image_path, output_path=None):
#     start_time = time.time()
#     logger.info(f"Loading image from {image_path}")

#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (600, 600))
#     (_, warped) = find_puzzle(image, debug=False)

#     digit_model = load_digit_model()
#     board = np.zeros([9, 9], dtype='int')
#     stepX = warped.shape[1] // 9
#     stepY = warped.shape[0] // 9
#     cellLocs = [[(j * stepX, i * stepY, (j + 1) * stepX, (i + 1) * stepY) for j in range(9)] for i in range(9)]

#     for y in range(9):
#         for x in range(9):
#             (startX, startY, endX, endY) = cellLocs[y][x]
#             cell = warped[startY:endY, startX:endX]
#             digit_image = extract_digit(cell)
#             if digit_image is None:
#                 continue
#             try:
#                 digit = identify_number(digit_image, digit_model)
#                 board[y][x] = digit
#                 logger.debug(f"Digit {digit} placed at ({y}, {x})")
#             except Exception as e:
#                 logger.exception(f"Failed to identify digit at ({y}, {x})")

#     logger.info("Initial board extracted:")
#     print(board)

#     solved_board = inference_sudoku(board)
#     logger.info("Solved board:")
#     print(solved_board)

#     for y in range(9):
#         for x in range(9):
#             (startX, startY, endX, endY) = cellLocs[y][x]
#             textX = int((endX - startX) * 0.33) + startX
#             textY = int((endY - startY) * -0.2) + endY
#             cv2.rectangle(warped, (startX + 2, startY + 2), (endX - 2, endY - 2), (255, 255, 255), -1)
#             digit = solved_board[y][x]
#             color = (0, 0, 255) if board[y][x] == 0 else (0, 0, 0)
#             cv2.putText(warped, str(digit), (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     if output_path:
#         logger.info(f"Saving result to {output_path}")
#         cv2.imwrite(output_path, warped)

#     elapsed = time.time() - start_time
#     logger.info(f"Total runtime: {elapsed:.2f} seconds")

#     plt.imshow(warped, cmap='gray')
#     plt.title("Solved Sudoku")
#     plt.show()

# # ✅ Profiling-enabled entrypoint
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Solve a Sudoku from an image using a trained model.")
#     parser.add_argument("--image", required=True, help="Path to input Sudoku image")
#     parser.add_argument("--output", default=None, help="Path to save output visualization")
#     args = parser.parse_args()

#     profiler = cProfile.Profile()
#     profiler.enable()

#     solve_from_image(args.image, args.output)

#     profiler.disable()
#     profiler.dump_stats("profile_output.prof")
#     print("✅ Profile data saved to profile_output.prof")

#     # Optional: view top 25 slowest
#     stats = pstats.Stats("profile_output.prof").sort_stats("cumtime")
#     stats.print_stats(25)

import argparse
import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
import cProfile
import mlflow
from prometheus_client import start_http_server, Counter, Summary, Gauge
from rich.logging import RichHandler
from tensorflow.keras.models import load_model
from se_489_mlops_project.puzzle_utils import find_puzzle, extract_digit

# ======================
# Setup
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("sudoku_logger")

# ======================
# Prometheus Metrics
# ======================
REQUEST_COUNT = Counter(
    "sudoku_requests_total",
    "Total number of Sudoku solve requests"
)
PREDICTION_TIME = Summary(
    "prediction_duration_seconds",
    "Time spent solving Sudoku"
)
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Currently processing requests"
)
ERROR_COUNT = Counter(
    "sudoku_errors_total",
    "Total number of processing errors"
)

# ======================
# Core Functions
# ======================
def norm(a):
    return (a / 9) - 0.5

def denorm(a):
    return (a + 0.5) * 9

def load_digit_model(model_path='models/digit_model.h5'):
    logger.info("Loading digit recognition model...")
    return load_model(model_path)

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
    return np.argmax(out, axis=2).reshape((9, 9)) + 1

# ======================
# Processing Function
# ======================
def process_sudoku(image_path, output_path=None):
    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        with PREDICTION_TIME.time():
            # Load and process image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (600, 600))
            puzzleImage, warped = find_puzzle(image, debug=False)

            # Extract digits
            digit_model = load_digit_model()
            board = np.zeros([9, 9], dtype='int')
            stepX = warped.shape[1] // 9
            stepY = warped.shape[0] // 9

            for y in range(9):
                for x in range(9):
                    startX, startY = x * stepX, y * stepY
                    endX, endY = (x + 1) * stepX, (y + 1) * stepY
                    cell = warped[startY:endY, startX:endX]
                    digit_image = extract_digit(cell)
                    
                    if digit_image is not None:
                        board[y][x] = identify_number(digit_image, digit_model)

            # Solve and render
            solved_board = inference_sudoku(board)
            
            if output_path:
                for y in range(9):
                    for x in range(9):
                        startX, startY = x * stepX, y * stepY
                        endX, endY = (x + 1) * stepX, (y + 1) * stepY
                        textX = int((endX - startX) * 0.33) + startX
                        textY = int((endY - startY) * -0.2) + endY
                        
                        cv2.rectangle(warped, (startX + 2, startY + 2), 
                                    (endX - 2, endY - 2), (255, 255, 255), -1)
                        digit = solved_board[y][x]
                        color = (0, 0, 255) if board[y][x] == 0 else (0, 0, 0)
                        cv2.putText(warped, str(digit), (textX, textY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                cv2.imwrite(output_path, warped)
                
            return solved_board
            
    except Exception as e:
        ERROR_COUNT.inc()
        logger.exception("Processing failed")
        raise
    finally:
        ACTIVE_REQUESTS.dec()
        mlflow.log_metric("processing_time_seconds", time.time() - start_time)

# ... [keep all your existing imports and setup code] ...

if __name__ == "__main__":
    # Start metrics server (runs in background)
    start_http_server(8000, addr="0.0.0.0")
    logger.info("Metrics server running at http://0.0.0.0:8000/metrics")

    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--interval", type=int, default=0,
                      help="Seconds between processing (0 for single run)")
    args = parser.parse_args()

    def process_once():
        with mlflow.start_run():
            mlflow.log_param("input_image", args.image)
            mlflow.log_param("output_path", args.output or "none")
            
            with cProfile.Profile() as profiler:
                result = process_sudoku(args.image, args.output)
                logger.info(f"Solved board:\n{result}")
                
            profiler.dump_stats("profile.prof")
            logger.info("Profile data saved to profile.prof")

    try:
        if args.interval > 0:
            # Continuous processing mode
            logger.info(f"Running in continuous mode (interval: {args.interval}s)")
            while True:
                process_once()
                time.sleep(args.interval)
        else:
            # Single-run mode
            process_once()
            
            # Keep server running for metrics
            logger.info("Processing complete. Metrics server remains active.")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        logger.info("Service terminated")