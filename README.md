# Brain Tumor Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify brain MRI images into four categories:
- Glioma
- Meningioma
- Pituitary tumor
- Non-tumor

## Model
- Framework: TensorFlow / Keras
- Input: 224×224 grayscale MRI images
- Architecture: 3 convolutional blocks + fully connected layers
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Dataset
The dataset is not included in this repository but the one used is from a the public opensource site Kaggle.
Expected structure:

data/
├── Train/
│   ├── glioma/
│   ├── meningioma/
│   ├── nontumor/
│   └── pituitary/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── nontumor/
    └── pituitary/

## How to Run
```bash
pip install -r requirements.txt
python Brain_tumor_SamayAsubadin.py
