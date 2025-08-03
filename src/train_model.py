from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model

def train_model(train_gen, val_gen, input_shape, num_classes, class_weight=None, model_path='../models/best_model.h5'):
    """
    Trains the CNN model with the given data generators.

    Args:
        train_gen: Training data generator.
        val_gen: Validation data generator.
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        model_path (str): Where to save the best model.

    Returns:
        model (Sequential): Trained Keras model.
    """
    model = build_model(input_shape, num_classes)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)

    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight
    )

    return model, history 