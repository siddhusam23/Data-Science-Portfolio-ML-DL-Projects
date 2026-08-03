"""
Model architectures used in the study:
  - Custom CNN  (baseline, best-performing model per Grad-CAM analysis)
  - ResNet50    (transfer learning)
  - DenseNet121 (transfer learning)
  - EfficientNetB0 (transfer learning)

Each builder returns a compiled tf.keras.Model ready for training.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import config


def build_custom_cnn(input_shape=(*config.IMG_SIZE, config.CHANNELS),
                      num_classes=config.NUM_CLASSES):
    """
    Custom CNN architecture (CCNN) described in Chapter 5.1.8 of the report:
    stacked Conv2D + ReLU + MaxPooling blocks, batch normalization, dropout,
    and a softmax classification head. The final conv layer is explicitly
    named so Grad-CAM can target it.
    """
    model = models.Sequential(name="custom_cnn")
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Final conv block — this is the layer Grad-CAM will hook into.
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv2d_last"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def _build_transfer_model(base_model_fn, input_shape, num_classes, preprocess_fn=None):
    """Shared helper for building a transfer-learning classifier head on top of a
    frozen (initially) pretrained backbone."""
    base_model = base_model_fn(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False  # unfreeze later for fine-tuning if desired

    inputs = layers.Input(shape=input_shape)
    x = inputs
    if preprocess_fn is not None:
        x = preprocess_fn(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=base_model.name + "_classifier")
    return model


def build_resnet50(input_shape=(*config.IMG_SIZE_TRANSFER, config.CHANNELS),
                    num_classes=config.NUM_CLASSES):
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return _build_transfer_model(ResNet50, input_shape, num_classes, preprocess_input)


def build_densenet121(input_shape=(*config.IMG_SIZE_TRANSFER, config.CHANNELS),
                       num_classes=config.NUM_CLASSES):
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.applications.densenet import preprocess_input
    return _build_transfer_model(DenseNet121, input_shape, num_classes, preprocess_input)


def build_efficientnetb0(input_shape=(*config.IMG_SIZE_TRANSFER, config.CHANNELS),
                          num_classes=config.NUM_CLASSES):
    from tensorflow.keras.applications import EfficientNetB0
    # EfficientNet's preprocess_input is a no-op (scaling is built into the model)
    return _build_transfer_model(EfficientNetB0, input_shape, num_classes, preprocess_fn=None)


MODEL_BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "efficientnetb0": build_efficientnetb0,
}


def get_model(model_name: str = config.MODEL_NAME):
    """Factory function: returns a freshly built, compiled model by name."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose from {list(MODEL_BUILDERS)}"
        )
    model = MODEL_BUILDERS[model_name]()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = get_model(config.MODEL_NAME)
    m.summary()
