import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from src.model import build_model
from data import generate_data
from src import visualize , evaluate_model


train_data, val_data = generate_data()  # split the dataset into train and validation sets

model = build_model()    # build the model with architecture defined in src/model.py

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # compile the model with optimizer, loss function and metrics

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

model.save('/output/model.h5')  # save the model to the output directory

model_path = ""
data_path  = ""

evaluate_model(model_path, data_path)    #evaluate the model with the test data

visualize(history)    # visualize the training history(accuracy and loss)
