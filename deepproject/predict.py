import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis

# ✅ Load RetinaFace for Face Detection and ArcFace for Embeddings
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])  # Runs on GPU
app.prepare(ctx_id=0, det_size=(640, 640))  # Set detection input size


RTSP_URL = 0
cap = cv2.VideoCapture(RTSP_URL)

# ✅ Database of Known Faces (Simulated with Stored Embeddings)

folder_path="embeddings"
def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two face embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def recognize_face(face_embedding, threshold=0.5):
    """Recognize a face by comparing embeddings."""
    best_match = None
    highest_similarity = 0

    for file  in os.listdir(folder_path):
        # print(type(file ))
        known_embedding = np.load(os.path.join(folder_path, file))
        similarity = cosine_similarity(face_embedding, known_embedding)
        if similarity > highest_similarity and similarity > threshold:
            highest_similarity = similarity
            best_match = file[:-4]

    return best_match

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    faces = app.get(frame)  
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.normed_embedding  # Extract ArcFace embedding

        # ✅ Recognize the person
        person_name = recognize_face(embedding)
        # print(person_name)
        # ✅ Draw bounding box & name
        color = (0, 255, 0) if person_name else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, person_name if person_name else "Unknown", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
