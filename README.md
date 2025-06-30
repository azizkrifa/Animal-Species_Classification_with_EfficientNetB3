# 🐾 Animal-Species_Classification_with_EfficientNetB3

This project implements a deep learning pipeline for classifying 10 animal species from images using a pretrained **EfficientNetB3** model with data augmentation and balanced classes.

---

## 🧠 Overview ##

- Downloaded the `Animals10` dataset from `Kaggle`.

- Visualized random training images and **class distributions**.

- Performed `data augmentation` to **balance** class sizes.

- Used Keras `ImageDataGenerator` for loading train, validation, and test sets.

- Built a `transfer learning` model based on `EfficientNetB3` pretrained on `ImageNet`.

- Trained with `early stopping`, `model checkpointing`, and  `logging`.

- Evaluated the model on a test set with `accuracy`, `classification report` ,  and `confusion matrix`.

- Visualized random `test predictions` showing **correct** vs **incorrect** classification.

---

## 🚀 How to Run

#### 📥 1. Clone the Repository

``` bash
    git clone https://github.com/azizkrifa/Animal-Species_Classification_with_EfficientNetB3.git
    cd  Animal-Species_Classification_with_EfficientNetB3
```

#### 📦 2. Install Dependencies

``` bash 
   pip install -r requirements.txt
``` 

---

## 📁 Dataset ##

- Original dataset ( **26k images** ) downloaded from Kaggle via **kagglehub**: [alessiocorrado99/animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

- 10 different animals : `Butterfly`   `Cat`   `Chicken`   `Cow`   `Dog`   `Elephant`   `Horse`   `Sheep`   `Spider`   `Squirrel`.

- Data split into `train(70%)`, `val(20%)`, and `test(10%)` folders using **split-folders** library.

- Balanced the training set using **augmentation** to equalize image **counts per class**.

---

## ⚙️ Model Setup

- **Base model**: EfficientNetB3 (pretrained on ImageNet, `include_top=False`).
  
- **Input size**: 224×224×3 (**RGB**).
  
- **Architecture**:
  
  - GlobalAveragePooling2D
  - **Dense**(512, **ReLU**) + **BatchNormalization**.
  - **Dense**(256, **ReLU**) + **BatchNormalization**.
  - **Dense**(10, **Softmax**).
    
- **Optimizer**: Adam (learning rate = 1e-5).
  
- **Loss function**: Categorical Crossentropy.
  
- **Metrics**: Accuracy.

---

## 🏋️‍♂️ Training Details

- **Epochs**: 50 (11 effected).
  
- **Batch size**: 32.
  
- **Training data**: Augmented and balanced images.
  
- **Validation data**: Clean validation set.
  
- **Callbacks**:
  
  - **Checkpoints**: Best model saved based on validation accuracy (`ModelCheckpoint`).
  - **Early stopping**: Stops training if validation loss does not improve for 10 consecutive epochs (`EarlyStopping` with `restore_best_weights=True`).
  - **CSVLogger** (saves training history to CSV).

---

## 📈 Training Curves

Plots of training and validation `accuracy / loss` across **11** epochs :

   ![Sans titre](https://github.com/user-attachments/assets/2a138a4a-3126-4586-b15c-4c07895778ff)

---

## 📊 Evaluation Results

- **Test accuracy** printed after model evaluation.
- **Classification report** with `precision`, `recall`, and `F1-score` per class (High performance across all classes, with a **test accuracy of 97%** ).
- **Confusion matrix** heatmap to visualize class-wise performance.

➡️  All evaluation outputs and the final model are stored in the `Outputs/` folder.

----

## 🎯 Sample Predictions

 ![Sans titre](https://github.com/user-attachments/assets/5bc705da-cefd-4272-a024-6fb899f20ba5)





