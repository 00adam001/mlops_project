from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from predict import process_sudoku
import logging
from rich.logging import RichHandler

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("sudoku_web")

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
        output_path = os.path.join(OUTPUT_FOLDER, 'output.jpg')
        file.save(input_path)

        # Process the image
        solved_board = process_sudoku(input_path, output_path)

        # Return the results
        return jsonify({
            'success': True,
            'solved_board': solved_board.tolist(),
            'output_image': '/output_image'
        })

    except Exception as e:
        logger.exception("Error processing image")
        return jsonify({'error': str(e)}), 500

@app.route('/output_image')
def get_output_image():
    output_path = os.path.join(OUTPUT_FOLDER, 'output.jpg')
    if os.path.exists(output_path):
        return send_file(output_path, mimetype='image/jpeg')
    return jsonify({'error': 'Output image not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080))) 