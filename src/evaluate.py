from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model_path='outputs/model.h5', test_dir='data/split_animals10/test'):
    
    datagen = ImageDataGenerator(rescale=1./255) # Rescale pixel values to [0, 1]
    test_gen = datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')

    model = load_model(model_path)
    loss, accuracy = model.evaluate(test_gen)

    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}") # Print test accuracy and loss

    class_labels = list(test_gen.class_indices.keys()) # Get class labels from the test generator
    
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
    
    class_labels = [italian_to_english[label] for label in class_labels]
    
    images, true_labels = next(test_gen)
    pred_probs = model.predict(images)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    
    plt.figure(figsize=(12, 8))   # Plot 6 images with predictions and true labels
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[pred_classes[i]]}")
    plt.tight_layout()
    plt.show()
