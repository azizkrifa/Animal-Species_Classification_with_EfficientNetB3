import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout as dropout
from src.model import build_model

train_path = "/content/drive/MyDrive/split_animals10/train"
val_path = "/content/drive/MyDrive/split_animals10/val"
test_path = "/content/drive/MyDrive/split_animals10/test"

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=(150,150), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=(150,150), batch_size=32, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_path, target_size=(150,150), batch_size=32, class_mode='categorical')

model = build_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,        # Stop if no improvement for 5 epochs
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop]
)
