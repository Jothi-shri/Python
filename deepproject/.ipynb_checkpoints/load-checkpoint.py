import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])  # Use GPU
app.prepare(ctx_id=0, det_size=(640, 640))


dataset_path = "dataset"  # Folder containing subfolders (each person's images)
output_folder = "embeddings"  # Folder to save embeddings

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):  # Skip non-folder files
        continue

    embeddings = []  # Store all embeddings of the person

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping {image_path} (Cannot open)")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"No face detected in {image_path}, skipping...")
            continue

        face_embedding = faces[0].normed_embedding
        print("face_embedding")
        embeddings.append(face_embedding)

    if len(embeddings) > 0:
        print("yes")
        avg_embedding = np.mean(embeddings, axis=0)
        save_path = os.path.join(output_folder, f"{person_name}.npy")
        np.save(save_path, avg_embedding)
        print(f"Saved embedding for {person_name} -> {save_path}")
    else:
        print(f"Skipping {person_name}, no valid faces found.")

print("✅ All embeddings saved successfully!")
