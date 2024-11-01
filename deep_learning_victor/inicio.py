import cv2
import mediapipe as mp

# Capturar a câmera
cap = cv2.VideoCapture()

# Ferramenta de desenho do MediaPipe para desenhar os pontos faciais
mp_drawing = mp.solutions.drawing_utils

# Solução de Face Mesh do MediaPipe para detectar pontos faciais
mp_face_mesh = mp.solutions.face_mesh

# Inicializar a solução de Face Mesh com parâmetros de confiança
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # Captura o frame da câmera
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera')
            continue

        # Converte a imagem para RGB, pois o MediaPipe usa imagens em RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem e detecta pontos faciais
        resultado = face_mesh.process(frame_rgb)

        # Desenha os pontos do Face Mesh, se algum rosto for detectado
        if resultado.multi_face_landmarks:
            for face_landmarks in resultado.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

        # Exibe o frame com pontos faciais
        cv2.imshow('Camera', frame)

        # Encerra o loop ao pressionar a tecla 'c'
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# Fecha a captura
cap.release()
cv2.destroyAllWindows()
