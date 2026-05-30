import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/genre"

# 模式："global","middle","segment"
PREPROCESS_MODE = "segment" 

# segment 秒數
SEGMENT_DURATION = 10

# 儲存 feature
features = []


# Feature Extraction Function
def extract_features(y, sr, label):

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Tempo
    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )
    tempo = float(np.mean(tempo))

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )
    spectral_centroid_mean = np.mean(spectral_centroid)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # 合併 Feature
    feature_vector = list(mfcc_mean)

    feature_vector.append(tempo)
    feature_vector.append(spectral_centroid_mean)
    feature_vector.append(zcr_mean)
    feature_vector.append(rms_mean)

    # label
    feature_vector.append(label)
    return feature_vector

# 讀取 Dataset
for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue
    print(f"\nProcessing: {label}")

    # 每個 wav
    for filename in os.listdir(label_path):

        if not filename.endswith(".wav"):
            continue
        file_path = os.path.join(
            label_path,
            filename
        )

        try:

            # 載入音樂
            y, sr = librosa.load(
                file_path,
                duration=30
            )

            
            # Method 1 : Global Average
            if PREPROCESS_MODE == "global":
                feature_vector = extract_features(
                    y,
                    sr,
                    label
                )
                features.append(feature_vector)

            # Method 2 : Middle Only
            elif PREPROCESS_MODE == "middle":
                length = len(y)
                start = length // 4
                end = length * 3 // 4
                y_middle = y[start:end]
                feature_vector = extract_features(
                    y_middle,
                    sr,
                    label
                )
                features.append(feature_vector)

            
            # Method 3 : Segment-based
            elif PREPROCESS_MODE == "segment":
                samples_per_segment = (
                    SEGMENT_DURATION * sr
                )
                total_segments = len(y) // samples_per_segment
                for i in range(total_segments):
                    start = i * samples_per_segment
                    end = start + samples_per_segment
                    segment = y[start:end]

                    # segment 太短跳過
                    if len(segment) < samples_per_segment:
                        continue
                    feature_vector = extract_features(
                        segment,
                        sr,
                        label
                    )
                    features.append(feature_vector)

        except Exception as e:
            print(f"Error processing {file_path}")
            print(e)

# 欄位名稱
columns = []

for i in range(13):
    columns.append(
        f"mfcc_{i+1}"
    )

columns += [
    "tempo",
    "spectral_centroid",
    "zcr",
    "rms",
    "label"
]

# DataFrame
feature_df = pd.DataFrame(
    features,
    columns=columns
)

# 輸出 CSV
os.makedirs(
    "features",
    exist_ok=True
)

output_file = (
    f"features/{PREPROCESS_MODE}_features.csv"
)

feature_df.to_csv(
    output_file,
    index=False
)

print("\n========================")
print("Feature extraction complete!")
print("========================")
print(f"Mode: {PREPROCESS_MODE}")
print(f"Saved to: {output_file}")
print(feature_df.head())