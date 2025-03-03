import os
import cv2
import pandas as pd
import numpy as np
import dlib
import pickle
import librosa
import time

# Paths to data directories
VIDEO_DIR = "data/mithos/video/"
AUDIO_DIR = "data/mithos/audio/"
DATA_FILE = "data/mithos/Timestamps_All.csv"
OUTPUT_DIR = "data/processed/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_audio_segment(audio_file_path, start, end, max_length=400000):
    """Extracts and pads/truncates an audio segment."""
    audio, sr = librosa.load(audio_file_path, sr=16000)
    start_sample, end_sample = int(start * sr), int(end * sr)
    audio_segment = audio[start_sample:end_sample]
    
    if len(audio_segment) > max_length:
        audio_segment = audio_segment[:max_length]
    elif len(audio_segment) < max_length:
        audio_segment = np.pad(audio_segment, (0, max_length - len(audio_segment)), 'constant')
    
    return audio_segment.reshape(1, -1)

def crop_video_segment(video_file, start_timestamp, end_timestamp, target_size=(224, 224)):
    """Extracts cropped face frames from a video segment."""
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame = max(0, int(start_timestamp * fps)), min(int(end_timestamp * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames, interval = [], fps // 2
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
            gray, faces = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_crop = cv2.resize(frame[max(0, y):y + h, max(0, x):x + w], target_size)
                frames.append(face_crop)
    cap.release()
    return np.array(frames, dtype=np.float32)

def process_fusion_data():
    """Processes MITHOS dataset for both audio and video."""
    df, all_videos, all_audios, labels_p, labels_a, labels_d = pd.read_csv(DATA_FILE, sep='\t'), [], [], [], [], []
    
    for i in range(len(df)):
        participant, ts = df.iloc[i]['Participant_Number'][-2:], df.iloc[i]['Timestamp_No']
        video_file, audio_file = os.path.join(VIDEO_DIR, f"user.video_MP000{participant}.mp4"), os.path.join(AUDIO_DIR, f"Participant{participant}_TS_{ts}.wav")
        
        if not os.path.exists(video_file) or not os.path.exists(audio_file):
            print(f"Skipping {participant} - Missing files.")
            continue
        
        print(f"Processing participant {participant}...")
        start_time, end_time = df.iloc[i]['Start_Time_s'], df.iloc[i]['End_time_for_Overlaps']
        video_segment, audio_segment = crop_video_segment(video_file, start_time, end_time), crop_audio_segment(audio_file, start_time, end_time)
        
        if video_segment.shape[0] > 0:
            all_videos.append(video_segment)
            for _ in range(video_segment.shape[0] // 16):
                all_audios.append(audio_segment)
                labels_p.append(df.iloc[i]['PleasureAverage'])
                labels_a.append(df.iloc[i]['ArousalAverage'])
                labels_d.append(df.iloc[i]['DominanceAverage'])
        
        print(f"Completed processing {participant}.")
    
    # Convert and save
    all_videos, all_audios = np.concatenate(all_videos, axis=0), np.concatenate(all_audios, axis=0)
    labels_p, labels_a, labels_d = np.array(labels_p), np.array(labels_a), np.array(labels_d)
    
    with open(os.path.join(OUTPUT_DIR, 'Fusion_Video_MITHOS.pkl'), 'wb') as f: pickle.dump(all_videos, f)
    with open(os.path.join(OUTPUT_DIR, 'Fusion_Audio_MITHOS.pkl'), 'wb') as f: pickle.dump(all_audios, f)
    with open(os.path.join(OUTPUT_DIR, 'Fusion_Labels_P_MITHOS.pkl'), 'wb') as f: pickle.dump(labels_p, f)
    with open(os.path.join(OUTPUT_DIR, 'Fusion_Labels_A_MITHOS.pkl'), 'wb') as f: pickle.dump(labels_a, f)
    with open(os.path.join(OUTPUT_DIR, 'Fusion_Labels_D_MITHOS.pkl'), 'wb') as f: pickle.dump(labels_d, f)
    
    print("Fusion Data Preprocessing Complete!")

if __name__ == "__main__":
    process_fusion_data()
