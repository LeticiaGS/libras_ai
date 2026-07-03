import cv2
import numpy as np
import json
from collections import deque
import tensorflow as tf
import time

from hand_landmarks import HandLandmarks


# =========================
# CONFIGURAÇÕES
# =========================
SEQUENCE_LENGTH = 20

MODEL_PATH = "models/checkpoints/model.h5"
LABELS_PATH = "data/labels.json"


# =========================
# CONFIG DE TROCA (ANTI-GRUDAR / ANTI-FLICKER)
# =========================
SWITCH_CONFIRM_FRAMES = 2
SWITCH_CONFIDENCE = 0.70
DROP_CURRENT_CONF = 0.60
SWITCH_MARGIN = 0.25


# =========================
# RESET INTELIGENTE DO BUFFER
# =========================
CLEAR_BUFFER_CONF = 0.35
CLEAR_BUFFER_FRAMES = 5


# =========================
# OTIMIZAÇÃO DE INFERÊNCIA
# =========================
PREDICT_EVERY_N_FRAMES = 2


# =========================
# SUAVIZAÇÃO (MELHORA MUITO GESTOS PARECIDOS)
# =========================
SMOOTH_WINDOW = 5  # média das últimas 5 previsões


# =========================
# CARREGA LABELS
# =========================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

labels = {k: int(v) for k, v in labels.items()}

idx_to_label = {v: k for k, v in labels.items()}
label_to_idx = {k: v for k, v in labels.items()}


# =========================
# CARREGA MODELO
# =========================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Modelo carregado: {MODEL_PATH}")

# detecta automaticamente quantas features o modelo espera
MODEL_EXPECTED_FEATURES = model.input_shape[-1]
print(f"🧠 Modelo espera {MODEL_EXPECTED_FEATURES} features por frame")


# =========================
# INICIALIZA DETECTOR
# =========================
detector = HandLandmarks(max_hands=1)


# =========================
# BUFFER DE FRAMES (features)
# =========================
sequence = deque(maxlen=SEQUENCE_LENGTH)

# buffer para suavização das probabilidades
pred_buffer = deque(maxlen=SMOOTH_WINDOW)


# =========================
# ESTADOS
# =========================
current_gesture = "Desconhecido"
current_confidence = 0.0

candidate_gesture = None
candidate_count = 0

weak_current_count = 0


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Erro: não foi possível abrir a câmera.")
    exit()

print("✅ Câmera iniciada. Pressione 'q' para sair.")


# =========================
# CONTROLE DE FPS
# =========================
prev_time = time.time()
fps = 0.0
frame_count = 0


# =========================
# LOOP PRINCIPAL
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ✅ Agora usamos features (landmarks + distâncias)
    feats = detector.extract_features(frame, draw=True)

    # só adiciona se capturou mão
    if feats is not None:
        # segurança: garante que o vetor bate com o que o modelo espera
        if feats.shape[0] == MODEL_EXPECTED_FEATURES:
            sequence.append(feats)
        else:
            # se houver mismatch, não adiciona (evita quebrar)
            cv2.putText(frame, "Feature mismatch!", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # sem mão detectada: limpa estado para não manter gesto antigo na tela
        sequence.clear()
        pred_buffer.clear()
        current_gesture = "Desconhecido"
        current_confidence = 0.0
        candidate_gesture = None
        candidate_count = 0
        weak_current_count = 0

    # =========================
    # PREDIÇÃO
    # =========================
    frame_count += 1

    if feats is not None and len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N_FRAMES == 0:
        input_data = np.array(sequence, dtype=np.float32)  # (seq_len, features)
        input_data = input_data.reshape(1, SEQUENCE_LENGTH, MODEL_EXPECTED_FEATURES)

        predictions = model.predict(input_data, verbose=0)[0]  # (num_classes,)

        # adiciona ao buffer para suavização
        pred_buffer.append(predictions)

        # usa média das probabilidades das últimas previsões
        avg_pred = np.mean(np.array(pred_buffer), axis=0)

        predicted_index = int(np.argmax(avg_pred))
        predicted_conf = float(np.max(avg_pred))
        predicted_label = idx_to_label.get(predicted_index, "Desconhecido")

        # probabilidade do gesto atual
        if current_gesture in label_to_idx:
            current_prob = float(avg_pred[label_to_idx[current_gesture]])
        else:
            current_prob = 0.0

        # =========================
        # RESET INTELIGENTE DO BUFFER
        # =========================
        if current_prob < CLEAR_BUFFER_CONF:
            weak_current_count += 1
        else:
            weak_current_count = 0

        if weak_current_count >= CLEAR_BUFFER_FRAMES:
            sequence.clear()
            pred_buffer.clear()

            current_gesture = "Desconhecido"
            current_confidence = 0.0
            candidate_gesture = None
            candidate_count = 0
            weak_current_count = 0
            continue

        # =========================
        # LÓGICA DE TROCA (MANTIDA)
        # =========================
        if current_gesture == "Desconhecido":
            if predicted_conf >= SWITCH_CONFIDENCE:
                current_gesture = predicted_label
                current_confidence = predicted_conf

        else:
            if current_prob >= DROP_CURRENT_CONF:
                current_confidence = current_prob
                candidate_gesture = None
                candidate_count = 0

            else:
                if (
                    predicted_conf >= SWITCH_CONFIDENCE
                    and (predicted_conf - current_prob) >= SWITCH_MARGIN
                ):
                    if candidate_gesture == predicted_label:
                        candidate_count += 1
                    else:
                        candidate_gesture = predicted_label
                        candidate_count = 1

                    if candidate_count >= SWITCH_CONFIRM_FRAMES:
                        current_gesture = candidate_gesture
                        current_confidence = predicted_conf
                        candidate_gesture = None
                        candidate_count = 0
                else:
                    candidate_gesture = None
                    candidate_count = 0

    # =========================
    # FPS
    # =========================
    curr_time = time.time()
    dt = curr_time - prev_time
    prev_time = curr_time

    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # =========================
    # TEXTO NA TELA
    # =========================
    if current_gesture != "Desconhecido":
        text = f"{current_gesture} ({current_confidence:.2f})"
        color = (0, 255, 0)
        cv2.putText(frame, text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # dica de status do buffer
    cv2.putText(frame, f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("LIBRAS AI - Reconhecimento", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        

# =========================
# FINALIZA
# =========================
cap.release()
cv2.destroyAllWindows()
print("✅ Finalizado.")
