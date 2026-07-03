import os
import json
import cv2
import numpy as np

from hand_landmarks import HandLandmarks


# ==========================
# CONFIGURAÇÕES DO DATASET
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
LABELS_PATH = os.path.join(BASE_DIR, "data", "labels.json")


# ==========================
# CONFIG PADRÃO
# ==========================
DEFAULT_SEQ_LEN = 20
DEFAULT_NUM_SEQUENCES = 40


# ==========================
# CONTROLE DE QUALIDADE
# ==========================
MAX_FAIL_STREAK = 12          # quantos frames seguidos pode falhar antes de descartar a sequência
MIN_VALID_RATIO = 0.80        # mínimo de frames válidos por sequência (80%)
MAX_EXTRA_ATTEMPTS = 60       # limite de tentativas extras para completar uma sequência


# ==========================
# FUNÇÕES AUXILIARES
# ==========================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_labels():
    if not os.path.exists(LABELS_PATH):
        return {}

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels_dict):
    ensure_dir(os.path.dirname(LABELS_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=4, ensure_ascii=False)


def update_labels(gesture_name: str):
    labels = load_labels()

    if gesture_name in labels:
        return labels[gesture_name]

    new_id = 0 if len(labels) == 0 else max(labels.values()) + 1
    labels[gesture_name] = new_id
    save_labels(labels)
    return new_id


def get_next_seq_id(folder_path: str):
    if not os.path.exists(folder_path):
        return 1

    files = [f for f in os.listdir(folder_path) if f.startswith("seq_") and f.endswith(".npy")]
    if not files:
        return 1

    nums = []
    for f in files:
        try:
            num = int(f.replace("seq_", "").replace(".npy", ""))
            nums.append(num)
        except:
            pass

    return max(nums) + 1 if nums else 1


def clear_gesture_folder(folder_path: str):
    if not os.path.exists(folder_path):
        return

    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            os.remove(os.path.join(folder_path, file))


# ==========================
# CAPTURA DE GESTOS
# ==========================
def record_gesture(
    gesture_name: str,
    num_sequences=DEFAULT_NUM_SEQUENCES,
    seq_len=DEFAULT_SEQ_LEN,
    wait_frames=15,
    reset_folder=False
):
    """
    Captura sequências e salva como .npy

    AGORA:
    - captura features 1D (landmarks + distâncias)
    - salva shape: (seq_len, num_features)
    - não salva zeros: valida qualidade e descarta sequência ruim
    """

    gesture_name = gesture_name.strip().upper()

    gesture_folder = os.path.join(RAW_DIR, gesture_name)
    ensure_dir(gesture_folder)

    if reset_folder:
        clear_gesture_folder(gesture_folder)
        print("🧹 Pasta do gesto limpa. Gravando do zero...")

    gesture_id = update_labels(gesture_name)

    print(f"\n✅ Gesto: {gesture_name} | ID: {gesture_id}")
    print(f"📁 Salvando em: {gesture_folder}")
    print(f"🎞️ Frames por sequência: {seq_len}")
    print(f"📦 Sequências: {num_sequences}")
    print(f"🧠 Modo: FEATURES (landmarks + distâncias)")

    detector = HandLandmarks(max_hands=1, detection_conf=0.7, tracking_conf=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: não foi possível abrir a webcam.")
        return

    next_seq = get_next_seq_id(gesture_folder)

    print("\n📌 CONTROLES:")
    print("   [ESPAÇO] iniciar captura")
    print("   [Q] sair\n")

    started = False
    sequences_done = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro: não foi possível capturar frame.")
            break

        frame = cv2.flip(frame, 1)

        # Info na tela
        cv2.putText(frame, f"Gesture: {gesture_name} (ID {gesture_id})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Sequences: {sequences_done}/{num_sequences}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if not started:
            cv2.putText(frame, "Press SPACE to start", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Add Gesture - Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        # Sair
        if key == ord("q"):
            break

        # Iniciar captura
        if key == 32 and not started:
            started = True
            print("\n⏳ Preparando para capturar...")

            for i in range(wait_frames, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Starting in {i}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Add Gesture - Capture", frame)
                cv2.waitKey(60)

            print("✅ Captura iniciada!\n")

        # ==========================
        # GRAVAÇÃO DAS SEQUÊNCIAS
        # ==========================
        if started and sequences_done < num_sequences:
            seq_data = []
            valid_frames = 0
            fail_streak = 0
            attempts = 0

            while len(seq_data) < seq_len:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # ✅ extrai FEATURES (1D) em vez de landmarks (21,3)
                feats = detector.extract_features(frame, draw=True)

                attempts += 1

                if feats is None:
                    fail_streak += 1

                    cv2.putText(frame, f"Tracking lost ({fail_streak})", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Add Gesture - Capture", frame)
                    cv2.waitKey(1)

                    # se falhar demais, descarta a sequência
                    if fail_streak >= MAX_FAIL_STREAK:
                        break

                    # não conta frame falho
                    continue

                # ok
                fail_streak = 0
                valid_frames += 1
                seq_data.append(feats)

                # progresso
                cv2.putText(frame, f"Recording... {len(seq_data)}/{seq_len}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Add Gesture - Capture", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    started = False
                    break

                # evita loop infinito
                if attempts >= (seq_len + MAX_EXTRA_ATTEMPTS):
                    break

            # ==========================
            # VALIDAÇÃO DA SEQUÊNCIA
            # ==========================
            if not started:
                break

            if len(seq_data) < seq_len:
                print("⚠️ Sequência descartada: perda de tracking.")
                # pequena pausa antes de tentar de novo
                for i in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Retry in {i}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.imshow("Add Gesture - Capture", frame)
                    cv2.waitKey(300)
                continue

            valid_ratio = valid_frames / float(seq_len)

            if valid_ratio < MIN_VALID_RATIO:
                print(f"⚠️ Sequência descartada: poucos frames válidos ({valid_ratio:.0%}).")
                for i in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Retry in {i}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.imshow("Add Gesture - Capture", frame)
                    cv2.waitKey(300)
                continue

            # ==========================
            # SALVAR
            # ==========================
            seq_data = np.array(seq_data, dtype=np.float32)  # shape: (seq_len, num_features)

            seq_filename = f"seq_{next_seq:04d}.npy"
            save_path = os.path.join(gesture_folder, seq_filename)
            np.save(save_path, seq_data)

            print(f"✅ Salvo: {save_path} | shape: {seq_data.shape} | valid_ratio: {valid_ratio:.0%}")

            sequences_done += 1
            next_seq += 1

            # pausa curta entre sequências
            for i in range(6, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Next in {i}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                cv2.imshow("Add Gesture - Capture", frame)
                cv2.waitKey(120)

        if sequences_done >= num_sequences:
            print("\n🎉 Captura concluída!")
            print(f"✅ {num_sequences} sequências salvas para o gesto {gesture_name}")
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================
# EXECUÇÃO
# ==========================
if __name__ == "__main__":
    print("\n=== ADD GESTURES (CAPTURE) ===")
    gesture = input("Digite o nome do gesto (ex: OI, TCHAU): ").strip()

    if not gesture:
        print("❌ Nome inválido.")
        exit()

    reset = input("Quer apagar os dados antigos desse gesto antes de gravar? (s/n): ").strip().lower() == "s"

    try:
        num_sequences = int(input(f"Quantas sequências capturar? (padrão {DEFAULT_NUM_SEQUENCES}): ").strip() or str(DEFAULT_NUM_SEQUENCES))
        seq_len = int(input(f"Quantos frames por sequência? (padrão {DEFAULT_SEQ_LEN}): ").strip() or str(DEFAULT_SEQ_LEN))
    except:
        print(f"❌ Valor inválido. Usando padrão: {DEFAULT_NUM_SEQUENCES} seq / {DEFAULT_SEQ_LEN} frames.")
        num_sequences = DEFAULT_NUM_SEQUENCES
        seq_len = DEFAULT_SEQ_LEN

    record_gesture(
        gesture_name=gesture,
        num_sequences=num_sequences,
        seq_len=seq_len,
        reset_folder=reset
    )