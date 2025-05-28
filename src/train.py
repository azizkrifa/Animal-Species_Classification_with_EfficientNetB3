import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout as dropout
from src.model import build_model
from data.split_data import split_data

path=split_data()

train_path = path+"/split_animals10/train"
val_path = path+"/split_animals10/val"

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=(150,150), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=(150,150), batch_size=32, class_mode='categorical')

model = build_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,        # Stop if no improvement for 10 epochs
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stop]
)

visualize(history)

model.save('/output/model.h5')

