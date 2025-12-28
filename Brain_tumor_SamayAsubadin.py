"""
Brain Tumor Classification using CNN
Author: Samay Asubadin

This script handles:
- Dataset loading
- CNN model definition
- Training
- Quantitative evaluation (classification report & confusion matrix)

Qualitative analysis and visualization are provided separately
in the accompanying Jupyter notebook.
"""

# ===============================
# Imports
# ===============================
import os
import numpy  as np
import tensorflow as tf

from   tensorflow.keras.preprocessing.image   import ImageDataGenerator
from   tensorflow.keras.models                import Sequential, load_model
from   tensorflow.keras.layers                import Conv2D, MaxPooling2D, Dense, Flatten, Dropout  
from   tensorflow.keras.callbacks             import EarlyStopping, ReduceLROnPlateau

from   sklearn.metrics             import classification_report, confusion_matrix

# ===============================
# Project Paths (PORTABLE)
# ===============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR   = os.path.join(DATASET_DIR, "Train")
TEST_DIR    = os.path.join(DATASET_DIR, "Test")

MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Parameters
# ===============================
IM_HEIGHT = 224
IM_WIDTH  = 224
BATCH_SIZE = 32
EPOCHS     = 30

# Expected class folders (used in analysis notebook)
CLASSES = ['glioma', 'meningioma', 'nontumor', 'pituitary']


# ===============================
# CNN Model
# ===============================
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===============================
# Training
# ===============================
def train_cnn_model(train_dir, model_path):
    print("\nTraining CNN Model...\n")

    data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = data_gen.flow_from_directory(
        train_dir,
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training'
    )

    validation_gen = data_gen.flow_from_directory(
        train_dir,  # Using the same train directory for validation subset
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation'
    )

    model = build_cnn_model((IM_HEIGHT, IM_WIDTH, 1), num_classes=len(train_gen.class_indices))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3)
    ]

    # Train the model
    training_history = model.fit(train_gen, validation_data=validation_gen, epochs=EPOCHS, callbacks=callbacks)
    np.save(os.path.join(MODEL_DIR, "training_history.npy"), training_history.history)

    # Save the model
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    return model

# ===============================
# Evaluation
# ===============================
def evaluate_cnn_model(model_path, test_dir):
    model = load_model(model_path)

    test_gen = ImageDataGenerator(rescale=1./255)
    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False,
    )

    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

    cm = confusion_matrix(y_true, y_pred)
    return cm

# ===============================
# Main Execution
# ===============================
def main():
    # Safety checks
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    model_path = os.path.join(MODEL_DIR, "brain_tumor_cnn_model.keras")

    print("Training the model...")
    train_cnn_model(TRAIN_DIR, model_path)

    print("Evaluating the model...")
    evaluate_cnn_model(model_path, TEST_DIR)

if __name__ == "__main__":
    main()