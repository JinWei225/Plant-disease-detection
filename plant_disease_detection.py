import os
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. SETUP & DATA DOWNLOAD
print("Downloading dataset via kagglehub...")
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

# Directly point to the 'color' subfolder to get color images instead of grayscale images
DATA_DIR = os.path.join(path, "plantvillage dataset", "color")

# Check if the dataset is downloaded correctly
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Color folder not found at: {DATA_DIR}")

print(f"Data Directory set to: {DATA_DIR}")

# CONFIGURATION
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FE = 10  # Number of epochs training the classifier head
EPOCHS_FT = 10  # Number of epochs training part of the model and the classifier head
SEED = 42

# Setting the seed so that we can get the exact same training output across different devices
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Prepare data to be loaded into model easily
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input,
    # apply several data augmentation techniques to improve model's ability to generalise
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input,
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

NUM_CLASSES = train_gen.num_classes
CLASS_NAMES = list(train_gen.class_indices.keys())

print(f"\nNumber of classes: {NUM_CLASSES}")
print(f"Number of train samples: {train_gen.samples}")
print(f"Number of validation samples: {val_gen.samples}")


# VISUALISE SAMPLE IMAGES
sample_imgs, sample_labels = next(train_gen)


# Normalise images back to [0,1] range so that it fits the matplotlib's format to display image
def deprocess(img):
    img = img - img.min()
    img = img / img.max()
    return img


fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle(
    "Sample Training Images (after augmentation)", fontsize=14, fontweight="bold"
)
for i, ax in enumerate(axes.flat):
    ax.imshow(deprocess(sample_imgs[i]))
    label_idx = np.argmax(sample_labels[i])
    short_name = CLASS_NAMES[label_idx].replace("___", "\n").replace("__", " ")
    ax.set_title(short_name, fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.show()


# BUILD MODEL (Transfer Learning)
# We learn about finetuning a EfficientNet model from
# https://python.plainenglish.io/revolutionizing-pharmaceutical-waste-management-with-ai-a-solution-built-with-tensorflow-bc08b31a7dbc
# We get the potential smaller version of custom classifier head from this article: https://doi.org/10.48550/arXiv.2604.12305,
# since the classifier head is used in X-ray images in the article, we first try to use the same classifier head and the performance
# is good so we did not make much changes to the head to avoid affecting the model's performance and training result.
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
)
# Frozen for training only the classifier head to let the model be familiar with the leaf features first
base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.summary()


# 5. CALLBACKS


def get_callbacks(phase):
    return [
        callbacks.ModelCheckpoint(
            f"best_phase{phase}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# Training the classifier head first
print("\n[INFO] Phase 1: Training classifier head (backbone frozen)...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FE,
    callbacks=get_callbacks(1),
)

# 7. PHASE 2 — Fine-Tuning
print("\n[INFO] Phase 2: Fine-tuning top backbone layers...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable backbone layers: {trainable} / {len(base_model.layers)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FT,
    callbacks=get_callbacks(2),
)


# 8. SAVE FINAL MODEL

model.save("plant_disease_efficientnet.keras")
print("\n✅ Model saved → plant_disease_efficientnet.keras")


# 9. TRAINING CURVES

acc = history1.history["accuracy"] + history2.history["accuracy"]
val_a = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss = history1.history["loss"] + history2.history["loss"]
val_l = history1.history["val_loss"] + history2.history["val_loss"]
ep = range(1, len(acc) + 1)
split = len(history1.history["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

for ax, tm, vm, title in [
    (ax1, acc, val_a, "Accuracy"),
    (ax2, loss, val_l, "Loss"),
]:
    ax.plot(ep, tm, label=f"Train {title}", linewidth=2)
    ax.plot(ep, vm, label=f"Val {title}", linewidth=2, linestyle="--")
    ax.axvline(
        split + 0.5, color="gray", linestyle=":", linewidth=1.5, label="Fine-tune start"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 10. EVALUATION

val_gen.reset()
loss_val, acc_val = model.evaluate(val_gen, verbose=1)
print(f"\n✅ Validation Loss     : {loss_val:.4f}")
print(f"✅ Validation Accuracy : {acc_val * 100:.2f}%")

val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes

print("\n── Classification Report ──────────────────────────────────")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))


# 11. CONFUSION MATRIX

TOP_N = 20
top_classes_idx = [label[0] for label in Counter(y_true).most_common()[:TOP_N]]
top_classes_names = [CLASS_NAMES[i] for i in top_classes_idx] + ["_other"]

top_y_true_names = [CLASS_NAMES[y] if y in top_classes_idx else "_other" for y in y_true]
top_y_pred_names = [CLASS_NAMES[y] if y in top_classes_idx else "_other" for y in y_pred]

cm = confusion_matrix(top_y_true_names, top_y_pred_names, labels=top_classes_names)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    xticklabels=top_classes_names,
    yticklabels=top_classes_names,
    ax=ax,
    linewidths=0.3,
    linecolor="white",
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(
    f"Normalised Confusion Matrix — Top {TOP_N} classes + _other", fontsize=13, fontweight="bold"
)
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.show()


# 12. PREDICTION GRID

val_gen.reset()
imgs, true_labels = next(val_gen)
preds = model.predict(imgs[:12], verbose=0)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle(
    "Sample Predictions  (green = correct, red = wrong)", fontsize=13, fontweight="bold"
)
for i, ax in enumerate(axes.flat):
    ax.imshow(deprocess(imgs[i]))  # deprocess for display only
    true_name = CLASS_NAMES[np.argmax(true_labels[i])].split("___")[-1][:22]
    pred_name = CLASS_NAMES[np.argmax(preds[i])].split("___")[-1][:22]
    conf = np.max(preds[i]) * 100
    correct = np.argmax(true_labels[i]) == np.argmax(preds[i])
    ax.set_title(
        f"True : {true_name}\nPred : {pred_name} ({conf:.1f}%)",
        color="green" if correct else "red",
        fontsize=7,
    )
    ax.axis("off")
plt.tight_layout()
plt.show()


# 13. SINGLE IMAGE PREDICTION

# Upload any leaf image and run this cell to get a prediction.

model = load_model("plant_disease_efficientnet.keras")
img_path = "img/Apple Scab 2.jpg"

img = kimage.load_img(img_path, target_size=(224, 224))
x = kimage.img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)
probs = model.predict(x, verbose=0)[0]
top5 = probs.argsort()[-5:][::-1]

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title(f"{top5[0]}\n({probs[top5[0]]*100:.1f}%)", fontsize=10)
plt.show()

print("Top-5 predictions:")
for idx in top5:
    print(f"  {CLASS_NAMES[idx]:<50s}  {probs[idx]*100:6.2f}%")
