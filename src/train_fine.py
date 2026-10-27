import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# PARAMETERS
# =============================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 12
EPOCHS_PHASE2 = 10

# =============================
# DATA GENERATORS
# =============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# =============================
# SAVE CLASS INDICES
# =============================

with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ class_indices.json saved")

# =============================
# CLASS WEIGHTS
# =============================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))
print("✅ Class weights computed")

# =============================
# MODEL BUILDING
# =============================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# =============================
# PHASE 1 TRAINING
# =============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    verbose=1
)

print("\n🚀 Phase 1: Training classifier head\n")

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# =============================
# PHASE 2: FINE-TUNING
# =============================

print("\n🔥 Phase 2: Fine-tuning last layers\n")

base_model.trainable = True

# Unfreeze last 30 layers only
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("✅ Fine-tuning completed")

# =============================
# PLOTS
# =============================

plt.figure()
plt.plot(history1.history['accuracy'] + history2.history['accuracy'])
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'])
plt.title('Model Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig(os.path.join(MODEL_DIR, "accuracy_plot.png"))

plt.figure()
plt.plot(history1.history['loss'] + history2.history['loss'])
plt.plot(history1.history['val_loss'] + history2.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train', 'Validation'])
plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))

print("✅ Training plots saved")

# =============================
# CONFUSION MATRIX
# =============================

val_generator.reset()
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))

print("✅ Confusion matrix saved")

print("🎉 FINAL TRAINING + FINE-TUNING COMPLETE")