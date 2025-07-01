import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import pandas as pd
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix



def display_sample_images(path): #Displays a sample of 16 images from the training dataset.

    # Path to the training dataset
    path = os.path.join(path, 'train')

    # Get all image paths with their class
    image_paths = []
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append((img_path, class_name))

    # Randomly select 16 images
    sampled = random.sample(image_paths, 16)

    # Plot them
    plt.figure(figsize=(12, 12))

    for i, (img_path, class_name) in enumerate(sampled):
        img = mpimg.imread(img_path)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_distribution(path): #Displays the distribution of images per class in the training dataset.

    # Count images per class
    train_counts = Counter([
        label for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
        for label in [folder] * len(os.listdir(os.path.join(path, folder)))
    ])

    # Create DataFrame
    df = pd.DataFrame(train_counts.items(), columns=['Class', 'Count'])
    df = df.sort_values(by='Count', ascending=False)

    # Plot using Seaborn
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x='Class', y='Count', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Image Count per Class – Train Set')
    plt.tight_layout()
    plt.show()


def load_dataset(path): # loads the training and validation datasets using Keras' ImageDataGenerator.

 
    train_path = os.path.join(path, 'train_augmented')  # Path to the training dataset
    val_path = os.path.join(path, 'val')  # Path to the validation dataset

    data_gen = ImageDataGenerator(rescale=1./255)

    train_data = data_gen.flow_from_directory(
        train_path, 
        target_size=(224,224), 
        batch_size=32, 
        class_mode='categorical')
    
    val_data = data_gen.flow_from_directory(
        val_path, 
        target_size=(224,224), 
        batch_size=32, 
        class_mode='categorical')
    
    return train_data, val_data  # Returns the training and validation datasets


def display_Training_History():

    # Load the saved history from file
    history = pd.read_csv("/Outputs/training_log.csv")

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def load_test_set(path): # loads the test dataset using Keras' ImageDataGenerator.

    test_path = os.path.join(path, 'test')  # Path to the test dataset

    data_gen = ImageDataGenerator(rescale=1./255)

    test_data = data_gen.flow_from_directory(
        test_path, 
        target_size=(224,224), 
        batch_size=32, 
        class_mode='categorical')   

    return test_data  # Returns the test dataset



def display_classfication_report_and_confusion_matrix(model, test_data):

    # Get true labels and predictions
    Y_true = test_data.classes
    Y_pred_probs = model.predict(test_data)
    Y_pred = np.argmax(Y_pred_probs, axis=1)

    # Get class labels
    class_labels = list(test_data.class_indices.keys())

    # Generate classification report
    report = classification_report(Y_true, Y_pred, target_names=class_labels)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(Y_true, Y_pred)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def dispaly_samples_predictions(model, test_path) :

    # Get class labels from the test directory
    class_labels = os.listdir(test_path)  
  
    italian_to_english = {
        'cane': 'dog',
        'cavallo': 'horse',
        'elefante': 'elephant',
        'farfalla': 'butterfly',
        'gallina': 'chicken',
        'gatto': 'cat',
        'mucca': 'cow',
        'pecora': 'sheep',
        'ragno': 'spider',
        'scoiattolo': 'squirrel'
    }
     
    # Convert Italian class labels to English
    class_labels_en = [italian_to_english[label] for label in class_labels] 


    # Get 6 random images
    random_images = []
    for _ in range(6):
        cls_dir = random.choice(class_labels)  # pick a random class
        img_name = random.choice(os.listdir(os.path.join(test_dir, cls_dir)))
        random_images.append((os.path.join(test_dir, cls_dir, img_name), cls_dir))

    # Plot
    plt.figure(figsize=(15, 8))

    for i, (img_path, true_label) in enumerate(random_images):

        # Load and preprocess
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred_prob = model.predict(img_array)
        pred_class = class_labels_en[np.argmax(pred_prob)]

        true_label_en = italian_to_english[true_label]

        # Plot
        plt.subplot(2, 3, i + 1)
        plt.imshow(load_img(img_path))
        plt.axis("off")

        plt.title(f"True: {true_label_en}\nPred: {pred_class}",
                color=("green" if true_label_en == pred_class else "red"))
        
    plt.tight_layout()
    plt.show()