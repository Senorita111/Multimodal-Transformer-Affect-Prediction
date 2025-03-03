import os
import cv2
import numpy as np
import dlib
import pickle
import pandas as pd
import time

# Paths to data directories
DATA_DIR = "data/mithos/video/"
DATA_FILE = "data/mithos/Timestamps_All.csv"
OUTPUT_DIR = "data/processed/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_video(video_path, target_size=(224, 224)):
    """Extracts cropped face frames from video."""
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    target_frame_rate = 1
    frame_skip = max(1, round(frame_rate / target_frame_rate))  # Avoid division by zero
    
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                x, y = max(0, x), max(0, y)
                face_crop = cv2.resize(frame[y:y + h, x:x + w], target_size)
                frames.append(face_crop)
    
    cap.release()
    return np.array(frames, dtype=np.float32)

def process_mithos_data():
    """Processes all MITHOS dataset videos and saves them as pickle files."""
    all_videos, labels_p, labels_a, labels_d = [], [], [], []
    
    # Load labels from TSV file
    df = pd.read_csv(DATA_FILE, sep='\t')
    print(f"Loaded labels: {df.shape[0]} samples")
    
    for i in range(len(df)):
        participant = df.iloc[i]['Participant_Number'][-2:]
        timestamp = df.iloc[i]['Timestamp_No']
        
        video_file = os.path.join(DATA_DIR, f"user.video_MP000{participant}.mp4")
        
        if not os.path.exists(video_file):
            print(f"Missing video: {video_file}, skipping.")
            continue
        
        print(f"Processing video: {video_file}")
        start_time = time.time()
        
        face_frames = preprocess_video(video_file)
        if face_frames.shape[0] > 0:
            all_videos.extend(face_frames)
            labels_p.extend([df.iloc[i]['PleasureAverage']] * len(face_frames))
            labels_a.extend([df.iloc[i]['ArousalAverage']] * len(face_frames))
            labels_d.extend([df.iloc[i]['DominanceAverage']] * len(face_frames))
        
        print(f"Completed processing {video_file}, Time Taken: {time.time() - start_time:.2f} sec")
    
    # Convert to numpy arrays
    all_videos = np.array(all_videos, dtype=np.float32)
    labels_p = np.array(labels_p, dtype=np.float32)
    labels_a = np.array(labels_a, dtype=np.float32)
    labels_d = np.array(labels_d, dtype=np.float32)
    
    # Save processed data
    with open(os.path.join(OUTPUT_DIR, 'Cropped_Face_MITHOS.pkl'), 'wb') as f:
        pickle.dump(all_videos, f)
    with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_MITHOS_P.pkl'), 'wb') as f:
        pickle.dump(labels_p, f)
    with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_MITHOS_A.pkl'), 'wb') as f:
        pickle.dump(labels_a, f)
    with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_MITHOS_D.pkl'), 'wb') as f:
        pickle.dump(labels_d, f)
    
    print("MITHOS Data Preprocessing Complete!")

if __name__ == "__main__":
    process_mithos_data()