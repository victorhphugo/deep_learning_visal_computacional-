import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Inicializa o mixer de áudio
pygame.mixer.init()

# Carrega os arquivos de som
pygame.mixer.music.load("audio/coin.mp3")  # Alarme da boca
som_olho = pygame.mixer.Sound("audio/galo.mp3")  # Alarme dos olhos

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca =  [82, 87, 13, 14, 312, 317, 78, 308]

# Função EAR (Eye Aspect Ratio)
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR (Mouth Aspect Ratio)
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Limiares
ear_limiar = 0.27
mar_limiar = 0.5
dormindo = False  # Flag para controle dos olhos fechados
aberto_boca = False  # Flag para controle da boca aberta
tempo_olhos_fechados = 0.0  # Tempo que os olhos ficaram fechados
tempo_boca_aberta = 0.0  # Tempo que a boca ficou aberta
contagem_piscadas = 0  # Contagem de piscadas

# Inicializa a câmera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Estado do som
som_boca_tocando = False
som_olho_tocando = False

# Define o tamanho da janela
window_width = 800
window_height = 600

cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera', window_width, window_height)

# Variáveis de tempo para o aviso de rosto
rosto_detectado = False
tempo_inicial_aviso = None

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if saida_facemesh.multi_face_landmarks:
            rosto_detectado = True
            tempo_inicial_aviso = time.time()

            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar = calculo_mar(face, p_boca)

                # Verificação da condição da boca aberta
                if mar > mar_limiar:
                    if not aberto_boca:
                        tempo_inicial_boca_aberta = time.time()
                        aberto_boca = True
                    else:
                        tempo_boca_aberta = time.time() - tempo_inicial_boca_aberta
                    estado_boca = "aberta"
                else:
                    tempo_boca_aberta = 0.0
                    aberto_boca = False
                    estado_boca = "fechada"

                # Verificação da condição dos olhos fechados
                if ear < ear_limiar:
                    if not dormindo:
                        tempo_inicial_olhos_fechados = time.time()
                        dormindo = True
                    else:
                        tempo_olhos_fechados = time.time() - tempo_inicial_olhos_fechados
                    estado_olho = "fechados"
                else:
                    if dormindo:
                        contagem_piscadas += 1
                    tempo_olhos_fechados = 0.0
                    dormindo = False
                    estado_olho = "abertos"

                # Exibir estados e tempos na tela
                cv2.putText(frame, f"MAR - Boca: {estado_boca} - {round(tempo_boca_aberta, 2)}s", (1, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (180,50,0), 2)
                cv2.putText(frame, f"EAR - Olhos: {estado_olho} - {round(tempo_olhos_fechados, 2)}s", (1, 160), cv2.FONT_HERSHEY_DUPLEX, 0.9, (180, 50, 0), 2)
                cv2.putText(frame, f"Piscadas: {contagem_piscadas}", (1, 190), cv2.FONT_HERSHEY_DUPLEX, 0.9, (180,50,0), 2)

                # Alarme para boca aberta
                if tempo_boca_aberta >= 1.5 and not som_boca_tocando:
                    cv2.rectangle(frame, (30, 400), (610, 452), (0,0,240), -1)
                    cv2.putText(frame, f"Alerta de Sono: Você está bocejando! {round(tempo_boca_aberta, 3)}", (1, 110),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)
                    pygame.mixer.music.play(-1)
                    som_boca_tocando = True
                elif tempo_boca_aberta == 0.0 and som_boca_tocando:
                    pygame.mixer.music.stop()
                    som_boca_tocando = False

                # Alarme para olhos fechados
                if tempo_olhos_fechados >= 1.0 and not som_olho_tocando:
                    cv2.rectangle(frame, (30, 400), (610, 452), (0,0,255), -1)
                    cv2.putText(frame, f"Olhos fechados por muito tempo!", (80, 435),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.85, (58, 58, 55), 1)
                    som_olho.play()
                    som_olho_tocando = True
                elif tempo_olhos_fechados == 0.0 and som_olho_tocando:
                    som_olho.stop()
                    som_olho_tocando = False
        else:
            # Atualiza o estado e tempo quando nenhum rosto é detectado
            if rosto_detectado:
                tempo_inicial_aviso = time.time()
            rosto_detectado = False

        # Exibe aviso de rosto detectado ou não detectado por 5 segundos
        if tempo_inicial_aviso and (time.time() - tempo_inicial_aviso <= 5):
            mensagem = "Rosto detectado" if rosto_detectado else "Nenhum rosto detectado"
            cv2.putText(frame, mensagem, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if rosto_detectado else (0, 0, 255), 3)

        cv2.imshow('Camera', frame)

        # Se pressionar 'c', fecha a janela
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# Finaliza a captura da câmera e fecha a janela
cap.release()
cv2.destroyAllWindows()