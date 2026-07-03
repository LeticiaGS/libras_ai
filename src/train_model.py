import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# =========================
# CONFIGURAÇÕES
# =========================
DATASET_PATH = "data/raw"
LABELS_PATH = "data/labels.json"
MODEL_SAVE_PATH = "models/checkpoints/model.h5"

SEQUENCE_LENGTH = 20

EPOCHS = 50
BATCH_SIZE = 16
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Se você quiser reponderar classes raras:
USE_CLASS_WEIGHTS = False


# =========================
# CARREGAR LABELS FIXOS
# =========================
def load_labels(labels_path):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"❌ labels.json não encontrado em: {labels_path}\n"
            f"Crie pelo add_gestures.py antes de treinar."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # garante ids inteiros
    labels = {k: int(v) for k, v in labels.items()}

    return labels


labels_dict = load_labels(LABELS_PATH)
idx_to_label = {v: k for k, v in labels_dict.items()}

num_classes = len(labels_dict)

print(f"✅ labels.json carregado. Classes: {num_classes}")
print("📌 Classes:", labels_dict)


# =========================
# CARREGAR DADOS
# =========================
def load_dataset(dataset_path, labels_dict, seq_len):
    """
    Espera arquivos .npy com shape: (seq_len, num_features)

    - Ignora arquivos com shape antigo (ex.: (seq_len, 21, 3))
    - Ignora gestos que não existem em labels.json
    """

    X = []
    y = []

    ignored_shape = 0
    ignored_label = 0
    total_files = 0

    gesture_names = sorted(os.listdir(dataset_path))

    for gesture in gesture_names:
        gesture_folder = os.path.join(dataset_path, gesture)

        if not os.path.isdir(gesture_folder):
            continue

        # Só aceita gestos registrados em labels.json
        if gesture not in labels_dict:
            ignored_label += 1
            continue

        for file in os.listdir(gesture_folder):
            if not file.endswith(".npy"):
                continue

            total_files += 1
            file_path = os.path.join(gesture_folder, file)

            seq = np.load(file_path)

            # esperado: (seq_len, num_features)
            if len(seq.shape) != 2 or seq.shape[0] != seq_len:
                ignored_shape += 1
                print(f"[IGNORADO SHAPE] {file_path} shape {seq.shape}")
                continue

            X.append(seq.astype(np.float32))
            y.append(labels_dict[gesture])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n📦 Total arquivos lidos: {total_files}")
    print(f"✅ Sequências válidas: {len(X)}")
    print(f"⚠️ Ignorados por shape: {ignored_shape}")
    print(f"⚠️ Pastas ignoradas sem label: {ignored_label}")

    return X, y


print("\n📌 Carregando dataset...")
X, y = load_dataset(DATASET_PATH, labels_dict, SEQUENCE_LENGTH)

if len(X) == 0:
    print("❌ Nenhum dado válido encontrado.")
    print("➡️ Verifique se você gravou com o add_gestures.py novo (features).")
    exit()

# detecta automaticamente quantas features existem
NUM_FEATURES = X.shape[2]
print(f"\n✅ Dataset carregado: {len(X)} sequências")
print(f"✅ SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
print(f"✅ FEATURES por frame: {NUM_FEATURES}")


# =========================
# SPLIT TREINO / TESTE
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\n✅ Treino: {len(X_train)} | Teste: {len(X_test)}")


# =========================
# CLASS WEIGHTS (opcional, mas ajuda em gestos desbalanceados)
# =========================
class_weights = None
if USE_CLASS_WEIGHTS:
    # calcula pesos inversamente proporcionais à frequência
    unique, counts = np.unique(y_train, return_counts=True)
    total = np.sum(counts)

    class_weights = {}
    for cls, c in zip(unique, counts):
        # peso = total / (n_classes * count_classe)
        class_weights[int(cls)] = float(total / (len(unique) * c))

    print("\n⚖️ Class Weights ativado:")
    print(class_weights)


# =========================
# CRIAR MODELO
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(96, activation="relu"),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# CALLBACKS
# =========================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# =========================
# TREINAR
# =========================
print("\n🚀 Treinando modelo...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights
)


# =========================
# SALVAR MODELO FINAL
# =========================
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Modelo treinado e salvo em: {MODEL_SAVE_PATH}")


# =========================
# AVALIAÇÃO
# =========================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Acurácia no teste: {acc:.4f}")