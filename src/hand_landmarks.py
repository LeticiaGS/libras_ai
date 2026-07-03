import cv2
import mediapipe as mp
import numpy as np


class HandLandmarks:
    """
    Detector de mão + extração de landmarks normalizados + features extras (distâncias).
    Objetivo: melhorar separação de gestos parecidos (SIM/NÃO, TCHAU/LEGAL etc.).
    """

    # Índices MediaPipe (padrão)
    WRIST = 0

    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    THUMB_MCP = 2
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,  # mais rápido para realtime
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    # =========================
    # EXTRAÇÃO DE LANDMARKS
    # =========================
    def extract_landmarks(self, frame, draw=True):
        """
        Recebe frame BGR e retorna:
        - landmarks normalizados (shape: (21,3))
        - None se não detectar mão
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # desenha usando o mesmo results
        if draw:
            for hand_lms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS
                )

        hand = results.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        # =========================
        # NORMALIZAÇÃO ROBUSTA
        # =========================
        # centraliza pelo pulso
        base = landmarks[self.WRIST].copy()
        landmarks = landmarks - base

        # escala: normaliza pela maior distância no plano (x,y)
        max_dist = np.max(np.linalg.norm(landmarks[:, :2], axis=1))
        if max_dist > 0:
            landmarks = landmarks / max_dist

        return landmarks

    # =========================
    # FEATURES EXTRAS (DISTÂNCIAS)
    # =========================
    @staticmethod
    def _dist(a, b):
        """Distância Euclidiana 3D entre dois pontos (a e b shape (3,))."""
        return float(np.linalg.norm(a - b))

    def compute_distance_features(self, landmarks):
        """
        Recebe landmarks normalizados (21,3) e retorna um vetor 1D de distâncias úteis.

        Essas distâncias capturam:
        - polegar dobrado vs estendido (LEGAL vs TCHAU)
        - indicador levantado vs dobrado (NÃO vs outros)
        - abertura geral da mão (mão aberta/fechada)
        """

        if landmarks is None:
            return None

        # Pontos principais
        wrist = landmarks[self.WRIST]

        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        ring_tip = landmarks[self.RING_TIP]
        pinky_tip = landmarks[self.PINKY_TIP]

        thumb_mcp = landmarks[self.THUMB_MCP]
        index_mcp = landmarks[self.INDEX_MCP]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        ring_mcp = landmarks[self.RING_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        # =========================
        # DISTÂNCIAS MUITO DISCRIMINATIVAS
        # =========================

        features = []

        # 1) Distâncias do pulso até as pontas (grau de extensão do dedo)
        features.append(self._dist(wrist, thumb_tip))
        features.append(self._dist(wrist, index_tip))
        features.append(self._dist(wrist, middle_tip))
        features.append(self._dist(wrist, ring_tip))
        features.append(self._dist(wrist, pinky_tip))

        # 2) Distâncias entre pontas dos dedos (abertura da mão)
        features.append(self._dist(thumb_tip, index_tip))
        features.append(self._dist(index_tip, middle_tip))
        features.append(self._dist(middle_tip, ring_tip))
        features.append(self._dist(ring_tip, pinky_tip))

        # 3) Polegar vs dedos (LEGAL vs TCHAU / SIM etc.)
        features.append(self._dist(thumb_tip, middle_tip))
        features.append(self._dist(thumb_tip, index_tip))
        features.append(self._dist(thumb_tip, ring_tip))
        features.append(self._dist(thumb_tip, pinky_tip))

        # 4) Pontas x MCP (dedo dobrado vs estendido)
        # Quando o dedo dobra, a ponta se aproxima do MCP.
        features.append(self._dist(thumb_tip, thumb_mcp))
        features.append(self._dist(index_tip, index_mcp))
        features.append(self._dist(middle_tip, middle_mcp))
        features.append(self._dist(ring_tip, ring_mcp))
        features.append(self._dist(pinky_tip, pinky_mcp))

        return np.array(features, dtype=np.float32)

    # =========================
    # VETOR FINAL PARA MODELO
    # =========================
    def landmarks_to_feature_vector(self, landmarks):
        """
        Converte landmarks (21,3) em um vetor 1D:
        - 63 valores dos landmarks flatten
        - + distâncias extras
        """

        if landmarks is None:
            return None

        # 63 features (21*3)
        flat_landmarks = landmarks.reshape(-1).astype(np.float32)

        # distâncias extras
        dist_features = self.compute_distance_features(landmarks)
        if dist_features is None:
            return None

        # concatena
        feat = np.concatenate([flat_landmarks, dist_features], axis=0)
        return feat.astype(np.float32)

    def extract_features(self, frame, draw=True):
        """
        Função pronta para uso:
        - processa frame
        - extrai landmarks normalizados
        - gera vetor final 1D (landmarks + distâncias)
        Retorna:
            - features (shape: (63 + n_dist,))
            - ou None se não detectar mão
        """

        landmarks = self.extract_landmarks(frame, draw=draw)
        if landmarks is None:
            return None

        return self.landmarks_to_feature_vector(landmarks)


# =========================
# TESTE DA CÂMERA
# =========================
def test_camera():
    detector = HandLandmarks()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Extrai features com desenho dos landmarks
        feats = detector.extract_features(frame, draw=True)

        if feats is not None:
            cv2.putText(
                frame,
                f"Features OK - shape {feats.shape}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Hand Landmarks + Features - Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()