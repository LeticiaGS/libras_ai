# LIBRAS AI - Reconhecimento de Gestos em Tempo Real

Projeto de reconhecimento de gestos de Libras usando webcam, MediaPipe e TensorFlow.
Pipeline completo:
1) capturar sequencias de gestos,
2) treinar o modelo,
3) reconhecer gestos ao vivo.

---

## Destaques
- Pipeline fim a fim (coleta -> treino -> inferencia)
- Features robustas: landmarks normalizados + distancias entre dedos
- Modelo temporal (LSTM) para sequencias
- Inferencia em tempo real com suavizacao e anti-flicker

---

## Demo (recomendado)
- Grave um GIF/MP4 curto (5-10s) mostrando o reconhecimento em tempo real.
- Dica: use boa iluminacao e fundo neutro para melhor estabilidade.

---

## Requisitos
- Python 3.10 (recomendado)
- Webcam funcionando
- Linux/Windows/macOS com suporte a OpenCV

Dependencias principais:
- tensorflow
- mediapipe
- opencv-python
- numpy
- scikit-learn

## Versoes recomendadas
Este projeto usa `requirements.txt` sem pins. Para compatibilidade, a recomendacao geral e:
- Python 3.10
- TensorFlow 2.x (CPU)
- MediaPipe 0.10+
- OpenCV 4.x
- NumPy 1.x
- scikit-learn 1.x

Se houver erro de compatibilidade na sua maquina, fixe as versoes no `requirements.txt` de acordo com o seu SO/arquitetura.

---

## Instalar e rodar do zero (maquina local)
### 1) Clonar e instalar dependencias
```bash
git clone <URL_DO_REPOSITORIO>
cd libras_ai

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2) Capturar seus gestos (cria o dataset)
```bash
python src/add_gestures.py
```
Siga os prompts no terminal. Durante a captura:
- Pressione `ESPACO` para iniciar a gravacao
- Pressione `Q` para sair

Isso cria:
- `data/raw/<GESTO>/seq_XXXX.npy` (sequencias)
- `data/labels.json` (mapa gesto -> id)

### 3) Treinar o modelo
```bash
python src/train_model.py
```
O modelo sera salvo em `models/checkpoints/model.h5`.

### 4) Reconhecimento em tempo real
```bash
python src/main.py
```
Pressione `Q` para sair.

---

## Estrutura do projeto
```
.
├── data/
│   ├── labels.json          # mapa gesto -> id
│   └── raw/                 # sequencias capturadas (.npy)
├── models/
│   └── checkpoints/
│       └── model.h5         # modelo treinado
├── src/
│   ├── add_gestures.py      # captura de dados
│   ├── train_model.py       # treino do modelo
│   ├── main.py              # inferencia em tempo real
│   └── hand_landmarks.py    # extracao de landmarks/features
└── requirements.txt
```

---

## Como o modelo funciona (resumo)
- MediaPipe detecta a mao e extrai 21 landmarks (x, y, z)
- Os landmarks sao normalizados e combinados com distancias entre dedos
- Cada frame vira um vetor 1D de features
- Uma sequencia de 20 frames alimenta o modelo LSTM
- A previsao e suavizada para evitar flicker

---

## Dicas para melhorar resultados
- Colete mais sequencias por gesto (>= 50)
- Mantenha iluminacao consistente
- Use fundo neutro
- Evite gestos muito parecidos sem contexto
- Balanceie o numero de exemplos entre as classes

---

## Limitacoes atuais
- **Duas maos**: pipeline atual extrai landmarks de apenas uma mao
- **Gestos parecidos**: sinais com movimentos semelhantes podem confundir
- **Expressoes faciais**: nao captura informacao do rosto/postura

---

## Problemas comuns
- **Webcam nao abre**: verifique permissao de camera no SO
- **Erro TensorFlow/MediaPipe**: use Python 3.10 e fixe versoes no `requirements.txt`
- **Feature mismatch**: grave os dados com a versao atual do `add_gestures.py`

---

## Roadmap (ideias de evolucao)
- Suporte a duas maos
- Pose/face landmarks para contexto
- Export para TFLite
- Dataset publico e metricas mais completas

---

## Autores
- Anna Luiza Silva Dome
- Arthur Machado Garlati
- Arthur Miguel GOnzaga da Fonseca
- Letícia Gomes dos Santos
- Luciana Carolline Fernandes Luiz Gomes
- Mateus Carvalho Rodrigues da Silva
- Mel Borges Martins
