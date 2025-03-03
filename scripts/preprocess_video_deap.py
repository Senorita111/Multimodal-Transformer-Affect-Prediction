import os
import cv2
import numpy as np
import dlib
import pickle
import time

# Paths to data directories
DATA_DIR = "data/deap/face_video/"
LABELS_DIR = "data/deap/face_video_labels/"
OUTPUT_DIR = "data/processed/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_video(video_path, target_size=(224, 224)):
    """Extracts cropped face frames from video."""
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)
    
    frame_rate = 50
    target_frame_rate = 1
    frame_skip = round(frame_rate / target_frame_rate)
    
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

def process_deap_data():
    """Processes all DEAP dataset videos and saves them as pickle files."""
    all_videos, labels_p,labels_a,labels_d = [], [],[], []
    
    for participant_id in range(1, 23):  # Participants 1 to 23
        if participant_id in [3, 5, 11, 14]:  # Skip missing data
            continue
        
        participant_folder = os.path.join(DATA_DIR, f"s{participant_id:02d}")
        label_file = os.path.join(LABELS_DIR, f"s{participant_id:02d}.dat")
        
        if not os.path.exists(label_file):
            print(f"Missing labels for participant {participant_id}, skipping.")
            continue
        
        with open(label_file, 'rb') as file:
            labels_data = pickle.load(file, encoding='latin1')['labels']
        
        print(f"Processing participant {participant_id}...")
        start_time = time.time()
        
        for trial in range(1, 41):  # 40 trials per participant
            video_file = os.path.join(participant_folder, f"s{participant_id:02d}_trial{trial:02d}.avi")
            
            if not os.path.exists(video_file):
                print(f"Missing video: {video_file}")
                continue
            
            face_frames = preprocess_video(video_file)
            if face_frames.shape[0] > 0:
                all_videos.extend(face_frames)
                # all_labels.extend([labels_data[trial - 1][2]] * len(face_frames))  # Dominance labels
                labels_p.extend([labels_data[trial - 1][0]] * len(face_frames))  # Pleasure
                labels_a.extend([labels_data[trial - 1][1]] * len(face_frames))  # Arousal
                labels_d.extend([labels_data[trial - 1][2]] * len(face_frames))  # Dominance

        
        print(f"Completed {participant_id}, Time Taken: {time.time() - start_time:.2f} sec")
    
    # Convert to numpy arrays
    all_videos = np.array(all_videos, dtype=np.float32)
    # all_labels = np.array(all_labels, dtype=np.float32)
    labels_p = np.array(labels_p, dtype=np.float32)
    labels_a = np.array(labels_a, dtype=np.float32)
    labels_d = np.array(labels_d, dtype=np.float32)
    
    # Save processed data
    with open(os.path.join(OUTPUT_DIR, 'Cropped_Face_DEAP.pkl'), 'wb') as f:
        pickle.dump(all_videos, f)
    # with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_DEAP.pkl'), 'wb') as f:
    #     pickle.dump(all_labels, f)
    with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_DEAP_P.pkl'), 'wb') as f:
        pickle.dump(labels_p, f)
    with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_DEAP_A.pkl'), 'wb') as f:
        pickle.dump(labels_a, f)
    with open(os.path.join(OUTPUT_DIR, 'Cleaned_Labels_DEAP_D.pkl'), 'wb') as f:
        pickle.dump(labels_d, f)
    
    print("DEAP Data Preprocessing Complete!")

if __name__ == "__main__":
    process_deap_data()