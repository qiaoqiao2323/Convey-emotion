import librosa
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


# Utility Functions
class AudioUtil:
    def __init__(self):
        self.sig = None
        self.sr = None

    def open(self, audio_file):
        self.sig, self.sr = torchaudio.load(audio_file)
        return self.sig, self.sr

    def extract_audio_features(self):
        features = []
        audio_data = self.sig.numpy().squeeze()  # Convert to numpy array and squeeze to remove channel dimension if mono
        sr = self.sr

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))

        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
        rmse = np.mean(librosa.feature.rms(y=audio_data))

        pitches, _ = librosa.core.piptrack(y=audio_data, sr=sr)
        pitch = np.mean(pitches)

        features.extend(mfccs_mean)
        features.extend([spectral_centroid, spectral_bandwidth, zero_crossing_rate, rmse, pitch])
        return np.array(features)


class MultimodalDataset(Dataset):
    def __init__(self, df):
        self.df = df

        # Adjust participant IDs
        self.df['participant_id'] = self.df['participant_id'].apply(self.adjust_participant_id)

        # First, crop the data to the same size
        self.df['tactile_data'] = self.df['path'].apply(lambda x: self.pad_trunc_tactile(pd.read_csv(x).values, 450))

        # Calculate min and max tactile values for each participant after padding/truncating
        self.min_max_dict = self.calculate_min_max_tactile()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tactile_data = self.df.loc[idx, 'tactile_data']
        audio_file = self.df.loc[idx, 'path_wav']
        class_id = self.df.loc[idx, 'emotion_id']
        pn = self.df.loc[idx, 'participant_id']
        round_num = self.df.loc[idx, 'round']

        # Process Audio
        audio_util = AudioUtil()
        sig, sr = audio_util.open(audio_file)
        audio_features = audio_util.extract_audio_features()

        # Rescale tactile data
        tactile_data = self.rescale_tactile_data(tactile_data, pn)
        tactile_features = self.extract_tactile_features(tactile_data)

        return audio_features, tactile_features, class_id, pn, round_num

    def adjust_participant_id(self, pid):
        if pid > 90:
            return 90 + (pid - 90) * (70 / (self.df['participant_id'].max() - 90))
        return pid

    def calculate_min_max_tactile(self):
        min_max_dict = {}
        for pid in self.df['participant_id'].unique():
            tactile_values = np.vstack(self.df[self.df['participant_id'] == pid]['tactile_data'].values)
            min_max_dict[pid] = {
                'min': np.min(tactile_values),
                'max': np.max(tactile_values)
            }
        return min_max_dict

    def rescale_tactile_data(self, data, pid):
        min_val = self.min_max_dict[pid]['min']
        max_val = self.min_max_dict[pid]['max']
        data = (data - min_val) / (max_val - min_val) * 120
        return data

    def pad_trunc_tactile(self, data, max_len):
        if data.shape[0] > max_len:
            data = data[:max_len, :]
        elif data.shape[0] < max_len:
            pad_len = max_len - data.shape[0]
            pad = np.zeros((pad_len, data.shape[1]))
            data = np.vstack((data, pad))
        return data

    #save the

    @staticmethod
    def extract_tactile_features(tactile_data, threshold=5):
        features = []
        touch_durations = []
        num_touches = 0
        current_touch = []
        over_all_touches = []
        touch_start_time = None

        # Existing features
        mean_pressure = np.mean(tactile_data > threshold)
        max_pressure = np.max(tactile_data)
        pressure_variance = np.var(tactile_data)
        y, x = np.meshgrid(range(tactile_data.shape[1]), range(tactile_data.shape[0]))  # Correct shape
        center_of_pressure_x = np.sum(x * tactile_data) / np.sum(tactile_data)
        center_of_pressure_y = np.sum(y * tactile_data) / np.sum(tactile_data)
        gradient = np.gradient(tactile_data)
        pressure_gradient = np.mean(np.sqrt(gradient[0] ** 2 + gradient[1] ** 2))

        # Additional features
        median_force = np.median(tactile_data)
        iqr_force = np.percentile(tactile_data, 75) - np.percentile(tactile_data, 25)
        touch_duration = tactile_data.shape[0]
        contact_area = np.sum(tactile_data > threshold)  # Assuming non-zero pressure values indicate contact
        min_force = np.min(tactile_data)
        rate_of_pressure_change = np.mean(np.abs(np.diff(tactile_data, axis=0)))
        pressure_std = np.std(tactile_data)

        features.extend([
            mean_pressure, max_pressure, pressure_variance, center_of_pressure_x, center_of_pressure_y,
            pressure_gradient, median_force, iqr_force, touch_duration, contact_area,
            min_force, rate_of_pressure_change, pressure_std
        ])

        for t in range(tactile_data.shape[0]):
            touch_area = tactile_data[t] > threshold

            if np.any(touch_area):
                if touch_start_time is None:
                    touch_start_time = t
                    num_touches += 1
                current_touch.append(tactile_data[t])
                over_all_touches.append(tactile_data[t])
            else:
                if touch_start_time is not None:
                    duration = t - touch_start_time
                    touch_durations.append(duration)
                    current_touch_np = np.array(current_touch)
                    mean_force = np.mean(current_touch_np)
                    touch_area_count = np.count_nonzero(np.mean(current_touch_np, axis=0))
                    current_touch = []
                    touch_start_time = None

        if touch_start_time is not None:
            duration = tactile_data.shape[0] - touch_start_time
            touch_durations.append(duration)
            current_touch_np = np.array(current_touch)
            #mean_force = np.mean(current_touch_np)
            #touch_area_count = np.count_nonzero(np.mean(current_touch_np, axis=0))

        max_touch_duration = max(touch_durations) if touch_durations else 0
        min_touch_duration = min(touch_durations) if touch_durations else 0
        mean_duration = np.mean(touch_durations) if touch_durations else 0

        features.extend([
             num_touches, max_touch_duration, min_touch_duration, mean_duration
        ])

        # Debugging print statements
        print(f"Touch durations: {touch_durations}")
        print(f"Number of touches: {num_touches}")
        print(f"Max touch duration: {max_touch_duration}")
        print(f"Min touch duration: {min_touch_duration}")
        print(f"Features: {features}")

        return np.array(features)


def save_dataset_to_csv(dataset, file_name):
    data = []

    audio_feature_names = [f'mfcc_{i + 1}' for i in range(13)] + ['spectral_centroid', 'spectral_bandwidth',
                                                                  'zero_crossing_rate', 'rmse', 'pitch']
    tactile_feature_names = [
        'mean_pressure', 'max_pressure', 'pressure_variance', 'center_of_pressure_x', 'center_of_pressure_y',
        'pressure_gradient', 'median_force', 'iqr_force', 'touch_duration', 'contact_area',
        'min_force', 'rate_of_pressure_change', 'pressure_std',
        'num_touches', 'max_touch_duration', 'min_touch_duration', "mean_duration"
    ]

    for i in range(len(dataset)):
        audio_features, tactile_features, class_id, pn, round_num = dataset[i]
        audio_path = dataset.df.iloc[i]['path_wav']

        # Convert tensor to list
        audio_features = audio_features.tolist()
        tactile_features = tactile_features.tolist()
        print(f"tactile_features: {tactile_features}")

        # Create a dictionary for row data
        row_data = {
            'audio_path': audio_path,
            'emotion_id': class_id,
            'pn': pn,
            'round': round_num
        }

        # Add audio features to row data
        for name, value in zip(audio_feature_names, audio_features):
            row_data[name] = value

        # Add tactile features to row data
        for name, value in zip(tactile_feature_names, tactile_features):
            row_data[name] = value

        # Classify emotions into five classes based on arousal and valence
        if class_id in [1]:
            row_data['arousal_class'] = 0
            row_data['valence_class'] = 0
            row_data['quadrant'] = 0
        elif class_id in [7, 9]:
            row_data['arousal_class'] = 1
            row_data['valence_class'] = 1
            row_data['quadrant'] = 1
        elif class_id in [0, 5, 6]:
            row_data['arousal_class'] = 1
            row_data['valence_class'] = -1
            row_data['quadrant'] = 2
        elif class_id in [4, 8]:
            row_data['arousal_class'] = -1
            row_data['valence_class'] = -1
            row_data['quadrant'] = 3
        elif class_id in [2, 3]:
            row_data['arousal_class'] = -1
            row_data['valence_class'] = 1
            row_data['quadrant'] = 4

        data.append(row_data)
    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv(file_name, index=False)




data = pd.read_csv('D:/Data_collection_results/metadata.csv')

# Create a MultimodalDataset
df = pd.DataFrame(data)
print(df.head())
multi_data = MultimodalDataset(df)

save_dataset_to_csv(multi_data, 'All_features_test_1.csv')
print("Datasets saved successfully.")
