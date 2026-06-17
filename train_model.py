import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, models

DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# LOAD DATASET
# =========================
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

# preprocessing khusus ResNet
dataset = dataset.map(lambda x, y: (preprocess_input(x), y))

# =========================
# CLASS WEIGHT (ANTI BIAS)
# =========================
labels = np.concatenate([y.numpy() for x, y in dataset])

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

# =========================
# MODEL RESNET50
# =========================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN AWAL
# =========================
model.fit(dataset, epochs=10, class_weight=class_weights)

# =========================
# FINE TUNING
# =========================
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(dataset, epochs=5)

# =========================
# SAVE MODEL
# =========================
model.save("model_paru.h5")

print("✅ Model berhasil disimpan!")