import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. CONFIG
# ============================================================

IMG_SIZE        = 224
BATCH_SIZE      = 16
EPOCHS          = 30    # increased from 25 → more time to learn blight differences
FINETUNE_EPOCHS = 15    # increased from 10 → deeper fine-tuning

DATASET_PATH      = 'dataset/train'
MODEL_SAVE_PHASE1 = 'model/tomato_best_phase1.h5'
MODEL_SAVE_FINAL  = 'model/tomato_disease_model.h5'
CLASS_INDICES_PATH = 'model/class_indices.pkl'

os.makedirs('model', exist_ok=True)

# ============================================================
# 2. CLASS NAMES — matched to class_indices.pkl exactly
#    {'early_blight': 0, 'Healthy': 1, 'late_blight': 2}
#    Dataset folder structure must be:
#    dataset/train/
#        early_blight/
#        Healthy/
#        late_blight/
# ============================================================

CLASSES = ['early_blight', 'Healthy', 'late_blight']

# ============================================================
# 3. DATASET CHECK — shows image counts and balance warning
# ============================================================

def check_dataset():
    print("\n📂 Dataset Check:")
    all_ok    = True
    counts    = {}

    for c in CLASSES:
        path = os.path.join(DATASET_PATH, c)
        if not os.path.exists(path):
            print(f"  ❌ Missing folder: {c}")
            all_ok = False
        else:
            count = len([
                f for f in os.listdir(path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            counts[c] = count
            status = "✅" if count >= 100 else "⚠️  (low count — add more images)"
            print(f"  {status} {c}: {count} images")

    # ── Balance check between blight classes ────────────────
    if 'early_blight' in counts and 'late_blight' in counts:
        eb = counts['early_blight']
        lb = counts['late_blight']
        ratio = max(eb, lb) / max(min(eb, lb), 1)
        if ratio > 1.2:
            print(
                f"\n  ⚠️  Blight imbalance detected! "
                f"early_blight={eb}, late_blight={lb} (ratio={ratio:.2f})\n"
                f"  → Download more images from:\n"
                f"    https://www.kaggle.com/datasets/emmarex/plantdisease\n"
                f"  → Target: equal counts for both blight classes"
            )
        else:
            print(f"\n  ✅ Blight classes are balanced (ratio={ratio:.2f})")

    return all_ok

# ============================================================
# 4. DATA GENERATORS
#    FIX 1: Added channel_shift_range — helps model learn
#            colour differences between early & late blight
#    FIX 2: Added fill_mode='reflect' — cleaner edge handling
# ============================================================

def prepare_data():
    # Train datagen — augmentation including colour shift
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=30.0,   # ← NEW: distinguishes blight colour tones
        fill_mode='reflect',         # ← NEW: cleaner edge pixels after rotation
        validation_split=0.2
    )

    # Val datagen — rescale ONLY, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42,
        classes=CLASSES     # enforces correct class order
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,      # keep order for evaluation
        seed=42,
        classes=CLASSES     # enforces correct class order
    )

    # Save class indices — overwrites pkl with correct mapping
    with open(CLASS_INDICES_PATH, 'wb') as f:
        pickle.dump(train_gen.class_indices, f)

    print("\n📋 Class Indices:", train_gen.class_indices)
    print(f"   Train batches : {len(train_gen)}")
    print(f"   Val batches   : {len(val_gen)}")

    return train_gen, val_gen

# ============================================================
# 5. MODEL
# ============================================================

def create_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False     # freeze for phase 1

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model, base_model

# ============================================================
# 6. CALLBACKS
# ============================================================

def get_callbacks(save_path):
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,              # slightly more patience for blight learning
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(model, val_gen):
    print("\n📊 Generating Confusion Matrix...")
    val_gen.reset()

    y_pred         = model.predict(val_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true         = val_gen.classes

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.title('Confusion Matrix — Early vs Late Blight')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png')
    plt.show()

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

    # ── Per-class accuracy printout ──────────────────────────
    print("\n📊 Per-class Accuracy:")
    for i, c in enumerate(CLASSES):
        correct = cm[i, i]
        total   = cm[i].sum()
        pct     = correct / max(total, 1) * 100
        print(f"   {c:20s}: {correct}/{total}  ({pct:.1f}%)")

# ============================================================
# 8. PLOT HISTORY
# ============================================================

def plot_history(history, title="Training History"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title(f'{title} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title(f'{title} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'model/{title.replace(" ", "_")}.png')
    plt.show()

# ============================================================
# 9. TRAIN
# ============================================================

def train():
    # Step 1: check dataset folders and balance
    if not check_dataset():
        print("\n❌ Fix dataset folders before training.")
        print("   Required folders:")
        for c in CLASSES:
            print(f"     dataset/train/{c}/")
        return

    # Step 2: load data
    train_gen, val_gen = prepare_data()

    # Step 3: class weights — reduces bias toward dominant class
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("\n⚖️  Class Weights:", class_weights)

    # Step 4: build model
    model, base_model = create_model(train_gen.num_classes)
    model.summary()

    # ── PHASE 1: train top layers only ──────────────────────
    print("\n🚀 Phase 1: Training top layers...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_SAVE_PHASE1)
    )

    plot_history(history1, "Phase 1 Training")

    # ── PHASE 2: fine-tune deeper layers ────────────────────
    # FIX 3: Freeze first 105 layers instead of 100
    #         → exposes last 50 MobileNetV2 layers for training
    #         → these layers learn texture/colour patterns
    #           critical for distinguishing early vs late blight
    print("\n🔧 Phase 2: Fine-tuning deeper layers...")
    base_model.trainable = True
    for layer in base_model.layers[:105]:   # was [:100] — now unfreezes 5 more
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Trainable MobileNetV2 layers: {trainable_count} / {len(base_model.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),   # lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINETUNE_EPOCHS,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_SAVE_FINAL)
    )

    plot_history(history2, "Phase 2 Fine-tuning")

    # ── FINAL EVALUATION ────────────────────────────────────
    print("\n📊 Final Evaluation:")
    loss, acc = model.evaluate(val_gen, steps=len(val_gen))
    print(f"   Val Loss     : {loss:.4f}")
    print(f"   Val Accuracy : {acc:.4f}")

    plot_confusion_matrix(model, val_gen)

    model.save(MODEL_SAVE_FINAL)
    print(f"\n✅ Training Complete!")
    print(f"   Model saved      : {MODEL_SAVE_FINAL}")
    print(f"   Class indices    : {CLASS_INDICES_PATH}")
    print(f"   Confusion matrix : model/confusion_matrix.png")

    # ── Quick blight confusion check ────────────────────────
    print("\n🔍 Blight Confusion Check:")
    print("   Open model/confusion_matrix.png and check:")
    print("   Row 'early_blight' → Column 'late_blight' should be LOW")
    print("   Row 'late_blight'  → Column 'early_blight' should be LOW")
    print("   If still high → add more diverse blight images and retrain.")

# ============================================================
# 10. TEST A SINGLE IMAGE
# ============================================================

def test_model():
    model = tf.keras.models.load_model(MODEL_SAVE_FINAL)

    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)

    class_names = sorted(class_indices, key=class_indices.get)
    print("Class order:", class_names)

    img_path = input("\nEnter test image path: ")

    from PIL import Image
    img       = Image.open(img_path).convert("RGB")
    img       = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]

    print("\n🔍 Prediction Scores:")
    for i, c in enumerate(class_names):
        bar = "█" * int(pred[i] * 30)
        print(f"  {c:20s}: {pred[i]:.4f} {bar}")

    top_idx    = np.argmax(pred)
    confidence = pred[top_idx]

    if confidence < 0.75:
        print(f"\n⚠️  Uncertain ({confidence:.1%}) — try a clearer image.")
    else:
        print(f"\n✅ Prediction : {class_names[top_idx]}")
        print(f"   Confidence  : {confidence:.1%}")

# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":
    train()
    print("\n" + "=" * 50)
    test_model()