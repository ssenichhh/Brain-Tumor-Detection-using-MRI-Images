# Brain Tumor Detection using MRI Images

This project aims to detect brain tumors using MRI images. The dataset used is the "Brain MRI Images for Brain Tumor Detection" from Kaggle. The model employed for this task is a fine-tuned VGG16, a popular deep learning model known for its performance in image classification tasks.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have Kaggle API credentials (`kaggle.json`) to download the dataset from Kaggle. 

## Dataset

The dataset can be downloaded from Kaggle:
- Dataset: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

### Structure

The dataset is organized into three categories: Training, Testing, and Validation. Each category contains two subdirectories: `YES` for images with tumors and `NO` for images without tumors.

## Preprocessing

### Steps

1. **Resizing**: All images are resized to 224x224 pixels to match the input size expected by VGG16.
2. **Cropping**: Images are cropped to focus on regions of interest by detecting contours.
3. **Data Augmentation**: Techniques such as rotation, width/height shift, shear, brightness adjustment, and flipping are applied to increase the diversity of the training data.

### Code
```python
def load_data(dir_path, img_size=(224,224)):
    # Load and resize images
    pass

def crop_imgs(set_name, add_pixels_value=0):
    # Crop images based on detected contours
    pass

def preprocess_imgs(set_name, img_size):
    # Resize and preprocess images for VGG16
    pass
```

## Model

### VGG16

VGG16 is a convolutional neural network model that is pre-trained on the ImageNet dataset. In this project, the base VGG16 model is fine-tuned to classify MRI images as either containing a brain tumor or not.

### Architecture

- **Base Model**: VGG16 without the top classification layer.
- **Custom Top Layers**: Added layers include Flatten, Dropout, and Dense layer with sigmoid activation for binary classification.

### Code
```python
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers, models

vgg16_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(weights=vgg16_weight_path, include_top=False, input_shape=(224, 224, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.layers[0].trainable = False  # Freeze the base model layers
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

## Training

### Data Generators

Data generators are created using `ImageDataGenerator` to handle data augmentation and preprocessing.

### Early Stopping

Early stopping is used to prevent overfitting by monitoring the validation accuracy and stopping training when it stops improving.

### Code
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, brightness_range=[0.5, 1.5], horizontal_flip=True, vertical_flip=True, preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory('TRAIN_CROP/', target_size=(224, 224), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory('VAL_CROP/', target_size=(224, 224), batch_size=16, class_mode='binary')

history = model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[EarlyStopping(monitor='val_accuracy', patience=6, mode='max')])
```

## Evaluation

### Accuracy and Loss

The model's performance is evaluated using accuracy and loss metrics on the validation set.

### Confusion Matrix

A confusion matrix is plotted to visualize the performance of the model.

### Code
```python
from sklearn.metrics import accuracy_score, confusion_matrix

predictions = (model.predict(val_generator) > 0.5).astype("int32")
accuracy = accuracy_score(val_generator.classes, predictions)
confusion_mtx = confusion_matrix(val_generator.classes, predictions)
```

## Visualization

### Sample Images

Sample images from the dataset are plotted to visualize the data.

### Augmented Images

Augmented images are displayed to show the effects of data augmentation.

### Code
```python
def plot_samples(X, y, labels_dict, n=50):
    # Plot sample images
    pass
```

## Usage

1. **Run the Jupyter Notebook**: Execute the notebook to train the model and evaluate its performance.
2. **Inference**: Use the trained model to make predictions on new MRI images by loading the model and running the inference.

## Acknowledgments

- This project uses the "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle.
- The VGG16 model is provided by Keras.

