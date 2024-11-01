import cv2
import mediapipe as mp

# Capturar a câmera / OPENCV (cap)
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

        # transformando bgr para rgb 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem e detecta pontos faciais
        saida_face_mesh = face_mesh.process(frame)

        #saida face_mesh 
        frame = face_mesh.process(frame, cv2.COLOR_RGB2BGR)

        # 1-vamos desenhar ? 
        # 2-mostrar edda detecção
        # 3-for (iteração) nas coordenadas da face
        # 4-face_landmarks - conjunto das coordenadas da face
        # 5- multi_Face_landmarks:x,y,z de cada ponto que MP encontrar no rosto 

        for face_landmarks in saida_face_mesh.multi_face_landmarks:
        #desenhando
        # 1 - frame
        # 2 -  face_landmark: os 
        # Desenha os pontos do Face Mesh, se algum rosto for detectado

            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    # Exibe o frame com pontos faciais
     #Encerra o loop ao pressionar a tecla 'c'
        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

# Fecha a captura
cap.release()
cv2.destroyAllWindows()
