from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data.split_data import split_data

def generate_data():   # Define paths for training and validation data

    split_data()  # Ensure the dataset is split into train, val, and test sets

    train_path = "data/split_animals10/train"  # Path to training data
    val_path = "data/split_animals10/val"      # Path to validation data

    train_gen = ImageDataGenerator(rescale=1./255)  #rescale pixel values to [0, 1]
    val_gen = ImageDataGenerator(rescale=1./255)    #rescale pixel values to [0, 1]

    train_data = train_gen.flow_from_directory(train_path, target_size=(150,150), batch_size=32, class_mode='categorical')
    val_data = val_gen.flow_from_directory(val_path, target_size=(150,150), batch_size=32, class_mode='categorical')

    return train_data, val_data  # Return the training and validation data generators