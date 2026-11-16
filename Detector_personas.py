
import cv2
from ultralytics import YOLO
from datetime import datetime

# Cargar modelo YOLO
modelo = YOLO("yolov8n.pt")

# Activar cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

print("Sistema de detección de personas iniciado")
print("Presiona Q para salir")

while True:

    ret, frame = camara.read()

    if not ret:
        print("Error al capturar imagen")
        break

    # Ejecutar detección
    resultados = modelo(frame, verbose=False)

    contador = 0

    for resultado in resultados:

        for caja in resultado.boxes:

            clase = int(caja.cls[0])

            # Clase 0 = persona
            if clase == 0:

                confianza = float(caja.conf[0])

                if confianza >= 0.50:

                    contador += 1

                    x1, y1, x2, y2 = map(
                        int, caja.xyxy[0]
                    )

                    # Dibujar rectángulo
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Etiqueta
                    etiqueta = f"Persona {confianza:.2%}"

                    cv2.putText(
                        frame,
                        etiqueta,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

    # Mostrar contador
    cv2.putText(
        frame,
        f"Personas detectadas: {contador}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Mostrar fecha y hora
    fecha = datetime.now().strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    cv2.putText(
        frame,
        fecha,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Mostrar cámara
    cv2.imshow("Detector Inteligente de Personas", frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()

print("Sistema finalizado")
