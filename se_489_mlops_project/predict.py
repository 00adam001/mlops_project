import argparse
import os
import cv2
import numpy as np
import logging
import time
import cProfile
import mlflow
from prometheus_client import start_http_server, Counter, Summary, Gauge
from rich.logging import RichHandler
from tensorflow.keras.models import load_model
from se_489_mlops_project.puzzle_utils import find_puzzle, extract_digit
from tensorflow.keras.optimizers import Adam

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
    model = load_model(model_path, compile=False)  # prevent loading outdated optimizer
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
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
            
    except Exception:
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
        
def solve_image(np_image):
    """Testing wrapper: converts in-memory image to temp file."""
    import uuid
    temp_id = str(uuid.uuid4())
    input_path = f"_temp_{temp_id}_input.jpg"
    output_path = f"_temp_{temp_id}_output.jpg"
    
    cv2.imwrite(input_path, np_image)
    try:
        result = process_sudoku(input_path, output_path)
        return result
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

