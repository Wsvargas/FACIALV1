from flask import Flask, render_template, Response
import cv2
import mediapipe as mp


app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

dataPath = "E:/facialexp/Dataset_faces"

# Cargar modelo entrenado
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_model.xml")

@app.route('/camara.html')
def camera_page():
    return render_template('camara.html')

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)

                if xmin < 0 and ymin < 0:
                    continue

                face_image = frame[ymin: ymin + h, xmin: xmin + w]
                face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image_resized = cv2.resize(face_image_gray, (72, 72), interpolation=cv2.INTER_CUBIC)

                # Predecir usando el modelo entrenado
                result, confidence = face_mask.predict(face_image_resized)

                if confidence > 135:
                    grado_paralisis = "Grado 1"
                elif 120 <= confidence <= 135:
                    grado_paralisis = "Grado 2"
                else:
                    grado_paralisis = "Grado 3"

                print("Grado de parálisis:", grado_paralisis)

                if result < 150:
                    label = "P FACIAL DETECTADA" if result == 0 else "TODO MARCHA BIEN"
                    color = (0, 255, 0) if label == "P FACIAL DETECTADA" else (0, 0, 255)
                    cv2.putText(frame, "{}".format(label), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
