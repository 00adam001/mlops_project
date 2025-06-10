import json
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from se_489_mlops_project.data.data_preprocess import get_data

SAMPLE_SIZE = 1000  # Number of rows to use for evaluation

def evaluate_model():
    try:
        # Load the existing model
        model_path = 'models/sudoku_model.h5'
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Load a sample of test data from existing raw data
        print(f"Loading test data sample (first {SAMPLE_SIZE} rows)...")
        _, x_test, _, y_test = get_data('data/raw/sudoku-2.csv', nrows=SAMPLE_SIZE)
        print(f"Test data loaded. Shape: x_test: {x_test.shape}, y_test: {y_test.shape}")
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=2)  # shape: (n_samples, 81)
        y_true_classes = y_test.squeeze()           # shape: (n_samples, 81)
        
        # Per-board accuracy (all 81 digits correct)
        per_board_accuracy = float(np.mean(np.all(y_pred_classes == y_true_classes, axis=1)))
        # Per-digit accuracy
        per_digit_accuracy = float(np.mean(y_pred_classes == y_true_classes))
        
        # Flatten for classification report and confusion matrix
        y_pred_flat = y_pred_classes.flatten()
        y_true_flat = y_true_classes.flatten()
        
        print("Calculating metrics...")
        metrics = {
            'per_board_accuracy': per_board_accuracy,
            'per_digit_accuracy': per_digit_accuracy,
            'classification_report': classification_report(y_true_flat, y_pred_flat, output_dict=True)
        }
        
        # Save metrics
        print("Saving metrics...")
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create confusion matrix (for all digits)
        print("Creating confusion matrix...")
        cm = confusion_matrix(y_true_flat, y_pred_flat)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (All Digits)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        print(f"Evaluation completed! Per-board accuracy: {per_board_accuracy:.4f}, Per-digit accuracy: {per_digit_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model() 