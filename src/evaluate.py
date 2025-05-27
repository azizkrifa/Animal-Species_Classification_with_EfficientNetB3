from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model= load_model("model.h5")

loss, accuracy = model.evaluate(test_data)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


class_labels = list(test_data.class_indices.keys())

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

images, true_labels = next(test_data)
pred_probs = model.predict(images)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.argmax(true_labels, axis=1)

# Plot first 6 images with predictions
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.axis("off")
    plt.title(f"True: {class_labels[true_classes[i]]}\nPred: {class_labels[pred_classes[i]]}")
plt.tight_layout()
plt.show()
