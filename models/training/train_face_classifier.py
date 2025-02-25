import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.face_shape_classifier import FaceShapeClassifier
from models.feature_extractor import feature_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path):
    """
    Load dataset from CSV file.
    
    Args:
        dataset_path: Path to the dataset CSV file
        
    Returns:
        tuple: (features, labels)
    """
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load data
        data = pd.read_csv(dataset_path)
        
        # Check if dataset has the required columns
        required_columns = [
            'width_to_length_ratio', 'cheekbone_to_jaw_ratio', 
            'forehead_to_cheekbone_ratio', 'jaw_angle', 'face_shape'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Dataset missing required columns: {missing_columns}")
            return None, None
        
        # Extract features and labels
        features = data[[
            'width_to_length_ratio', 'cheekbone_to_jaw_ratio', 
            'forehead_to_cheekbone_ratio', 'jaw_angle'
        ]]
        
        labels = data['face_shape']
        
        logger.info(f"Loaded dataset with {len(data)} samples")
        logger.info(f"Class distribution: {labels.value_counts().to_dict()}")
        
        return features, labels
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None, None


def preprocess_data(features, labels):
    """
    Preprocess data for training.
    
    Args:
        features: Features DataFrame
        labels: Labels Series
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    try:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Data split into {len(X_train)} training and {len(X_test)} testing samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None, None


def train_model(X_train, y_train):
    """
    Train the model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        tuple: (best_model, best_params)
    """
    try:
        logger.info("Training model with hyperparameter tuning")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        # Create SVM classifier
        svc = SVC(probability=True, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            svc, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        return best_model, best_params
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, None


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    try:
        logger.info("Evaluating model")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_,
                    yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
        plt.savefig(plot_path)
        logger.info(f"Confusion matrix saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")


def save_model(model, scaler, output_dir):
    """
    Save the trained model and scaler.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        output_dir: Output directory
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'face_shape_model.pkl')
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'face_shape_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")


def main():
    """
    Main function to train the face shape classifier.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train face shape classifier')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, default='../trained_models', help='Output directory for model')
    
    args = parser.parse_args()
    
    # Load dataset
    features, labels = load_dataset(args.dataset)
    
    if features is None or labels is None:
        logger.error("Failed to load dataset")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, labels)
    
    if X_train is None:
        logger.error("Failed to preprocess data")
        return
    
    # Train model
    model, best_params = train_model(X_train, y_train)
    
    if model is None:
        logger.error("Failed to train model")
        return
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, scaler, args.output)
    
    # Create and test classifier
    classifier = FaceShapeClassifier(
        model_path=os.path.join(args.output, 'face_shape_model.pkl'),
        scaler_path=os.path.join(args.output, 'face_shape_scaler.pkl')
    )
    
    # Test classifier with sample features
    sample_features = {
        'width_to_length_ratio': 0.75,
        'cheekbone_to_jaw_ratio': 1.05,
        'forehead_to_cheekbone_ratio': 0.95,
        'jaw_angle': 130
    }
    
    face_shape, confidence, probabilities = classifier.predict(sample_features)
    
    logger.info(f"Sample prediction: {face_shape} with {confidence:.1f}% confidence")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()