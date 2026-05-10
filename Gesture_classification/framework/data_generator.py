import numpy as np
import os, pickle
from framework.utilities import calculate_scalar, scale, create_folder
from sklearn.model_selection import train_test_split
import framework.config as config
 
class DataGeneratorgesture:
    def __init__(self, renormal=True, clip_length=1000, batch_size=32, test_size=0.115, val_size=0.1, seed=42,
                 fold_index=None, folds_num=10, use_all_training_ids=False, modality=None):
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.clip_length = clip_length
        self.modality = modality

        # Emotion and Gesture labels
        self.Gesture_list_name = ['Tickle', 'Poke', 'Rub', 'Pat', 'Tap', 'Hold']
        self.emotion_list_name = [
            'Happiness', 'Attention', 'Fear', 'Surprise', 'Confusion',
            'Sadness', 'Comfort', 'Calmimg', 'Anger', 'Disgust'
        ]

        # Paths
        data_space = config.data_space

        audio_root = os.path.join(data_space, 'Audio_features')
        tactile_root = os.path.join(data_space, 'Tactile_features')

        common_files = []
        with open(os.path.join(data_space, 'common_files.txt'), 'r') as f:
            for each in f.readlines():
                common_files.append(each.split('\n')[0])

        if not common_files:
            raise ValueError("No matching files between tactile and audio datasets.")

        print(f"Number of common files: {len(common_files)}")

        tactile_data = None
        audio_data = None

        if modality != 'audio':
            tactile_data = {f: np.load(os.path.join(tactile_root, f), allow_pickle=True) for f in common_files}

        if modality != 'tactile':
            audio_data = {f: np.load(os.path.join(audio_root, f), allow_pickle=True) for f in common_files}

        # Separate files into gestures and emotions
        gesture_files = [f for f in common_files if any(label in f for label in self.Gesture_list_name)]
        emotion_files = [f for f in common_files if any(label in f for label in self.emotion_list_name)]

        print(len(emotion_files))
        print(len(gesture_files))

        # Split into train, validation, and test sets
        QQ_training_IDs = [18, 4, 14, 23, 26, 21, 2, 7, 12, 28, 10, 15, 24, 6, 19, 16, 25, 11, 8, 20, 3, 1]
        QQ_training_IDs = list(map(str, QQ_training_IDs))

        print('QQ_training_IDs: ', QQ_training_IDs)

        if fold_index is None:
            QQ_training_IDs = set(QQ_training_IDs)
            if use_all_training_ids:
                gesture_train, gesture_val, gesture_test = self.split_files_train_all(gesture_files, QQ_training_IDs)
            else:
                gesture_train, gesture_val, gesture_test = self.split_files(gesture_files, QQ_training_IDs, test_size, seed)
        else:
            print('10 fold cross validation, fold_index: ', fold_index)
            gesture_train, gesture_val = self.split_files_10fold(gesture_files, QQ_training_IDs, fold_index, folds_num, seed)
            gesture_test = []

        self.training_ids = gesture_train
        self.Gesture_training = gesture_train
        self.Gesture_validation = gesture_val
        if fold_index is None:
            self.Gesture_test = gesture_test

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        if fold_index is None:
            self.test_random_state = np.random.RandomState(0)

        # Process features and labels separately for gestures and emotions
        print(len(gesture_train), len(gesture_val), len(gesture_test))
        # 303 65 108
        # 303 1294 1294
        # 296 72 108
        # 296 1294 1294

        if modality is None:
            self.gesture_training_t, self.gesture_training_a, self.gesture_training_label = self.process_features_and_labels(
                gesture_train, tactile_data, audio_data
            )
            print(self.gesture_training_t.shape, self.gesture_training_a.shape)

            self.Emotion_validation_t, self.Emotion_validation_a, self.Emotion_validation_label = self.process_features_and_labels(
                gesture_val, tactile_data, audio_data
            )
            if fold_index is None:
                self.Emotion_test_t, self.Emotion_test_a, self.Emotion_test_label = self.process_features_and_labels(
                    gesture_test, tactile_data, audio_data
                )

            print('Gesture training audio feature shape: ', self.gesture_training_a.shape)
            print('Emotion training tactile feature shape: ', self.gesture_training_t.shape)

        elif modality == 'audio':
            self.training_feature, self.training_label = self.process_single_features_and_labels(
                gesture_train, audio_data
            )
            self.validation_feature, self.validation_label = self.process_single_features_and_labels(
                gesture_val, audio_data
            )
            if fold_index is None:
                self.test_feature, self.test_label = self.process_single_features_and_labels(
                    gesture_test, audio_data
                )

            print('Gesture training audio feature shape: ', self.training_feature.shape)

        elif modality == 'tactile':
            self.training_feature, self.training_label = self.process_single_features_and_labels(
                gesture_train, tactile_data
            )
            self.validation_feature, self.validation_label = self.process_single_features_and_labels(
                gesture_val, tactile_data
            )
            if fold_index is None:
                self.test_feature, self.test_label = self.process_single_features_and_labels(
                    gesture_test, tactile_data
                )

            print('Gesture training tactile feature shape: ', self.training_feature.shape)

        else:
            raise ValueError('Unknown modality: ' + str(modality))

        # Normalize
        # self.normalize_features('Gesture', renormal)
        output_dir = os.path.join(os.getcwd(), '0_normalization_files')
        # print('output_dir', output_dir)
        create_folder(output_dir)
        if modality is None:
            normalization_t_a_file = os.path.join(output_dir, 'norm_tactile_audio.pickle')

            if renormal or not os.path.exists(normalization_t_a_file):
                print('normalize......')
                norm_pickle = {}
                (self.mean_audio, self.std_audio) = calculate_scalar(np.concatenate(self.gesture_training_a))
                norm_pickle['mean_a'] = self.mean_audio
                norm_pickle['std_a'] = self.std_audio

                (self.mean_tactile, self.std_tactile) = calculate_scalar(np.concatenate(self.gesture_training_t))
                norm_pickle['mean_t'] = self.mean_tactile
                norm_pickle['std_t'] = self.std_tactile

                self.save_pickle(norm_pickle, normalization_t_a_file)

            else:
                print('using: ', normalization_t_a_file)
                norm_pickle = self.load_pickle(normalization_t_a_file)
                self.mean_audio = norm_pickle['mean_a']
                self.std_audio = norm_pickle['std_a']

                self.mean_tactile = norm_pickle['mean_t']
                self.std_tactile = norm_pickle['std_t']

            print(self.mean_audio)
            print(self.std_audio)

            print(self.mean_tactile)
            print(self.std_tactile)

            print("norm: ", self.mean_audio.shape, self.std_audio.shape, self.mean_tactile.shape, self.std_tactile.shape)

        else:
            normalization_file = os.path.join(output_dir, 'norm_' + modality + '.pickle')

            if renormal or not os.path.exists(normalization_file):
                print('normalize......')
                norm_pickle = {}
                (self.mean_feature, self.std_feature) = calculate_scalar(np.concatenate(self.training_feature))
                norm_pickle['mean_feature'] = self.mean_feature
                norm_pickle['std_feature'] = self.std_feature

                self.save_pickle(norm_pickle, normalization_file)

            else:
                print('using: ', normalization_file)
                norm_pickle = self.load_pickle(normalization_file)
                self.mean_feature = norm_pickle['mean_feature']
                self.std_feature = norm_pickle['std_feature']

            print(self.mean_feature)
            print(self.std_feature)
            print("norm: ", self.mean_feature.shape, self.std_feature.shape)


    def split_files(self, files, training_ids, test_size, seed):
        training_ids = sorted(list(training_ids))
        file_ids = [f.split('_')[0] for f in files]
        train_ids, val_ids = train_test_split(
            training_ids, test_size=test_size, random_state=seed
        )
        test_ids = sorted(list(set([fid for fid in file_ids if fid not in training_ids])))

        train_files = [f for f in files if f.split('_')[0] in train_ids]
        val_files = [f for f in files if f.split('_')[0] in val_ids]
        test_files = [f for f in files if f.split('_')[0] in test_ids]
        # print(len(train_files)) #518
        # print(len(val_files)) #120
        # print(len(test_files)) #180

        return train_files, val_files, test_files

    def split_files_10fold(self, files, training_ids, fold_index, folds_num, seed):
        print('training_val_ids num: ', len(training_ids))
        print('training_val_ids: ', training_ids)

        if fold_index < 0 or fold_index >= folds_num:
            raise ValueError('fold_index is out of range.')

        shuffled_ids = list(np.random.RandomState(seed).permutation(training_ids))
        fold_sizes = [2] * folds_num
        for i in range(len(training_ids) - np.sum(fold_sizes)):
            fold_sizes[i] += 1

        all_folds = []
        start_index = 0
        for fold_size in fold_sizes:
            end_index = start_index + fold_size
            all_folds.append(shuffled_ids[start_index:end_index])
            start_index = end_index

        print('fold_sizes: ', fold_sizes)
        print('all_folds: ', all_folds)

        val_ids = all_folds[fold_index]
        train_ids = []
        for each_fold_index, each_fold_ids in enumerate(all_folds):
            if each_fold_index != fold_index:
                train_ids.extend(each_fold_ids)

        print('train_ids num: ', len(train_ids))
        print('train_ids: ', train_ids)
        print('val_ids num: ', len(val_ids))
        print('val_ids: ', val_ids)

        train_files = [f for f in files if f.split('_')[0] in train_ids]
        val_files = [f for f in files if f.split('_')[0] in val_ids]

        print(len(train_files))
        print(len(val_files))

        return train_files, val_files

    def split_files_train_all(self, files, training_ids):
        print('training_val_ids num: ', len(training_ids))
        print('training_val_ids: ', training_ids)

        file_ids = [f.split('_')[0] for f in files]
        test_ids = [fid for fid in file_ids if fid not in training_ids]

        print('train_ids num: ', len(training_ids))
        print('train_ids: ', training_ids)
        print('val_ids num: ', 0)
        print('val_ids: ', [])
        print('test_ids num: ', len(set(test_ids)))
        print('test_ids: ', set(test_ids))

        train_files = [f for f in files if f.split('_')[0] in training_ids]
        val_files = []
        test_files = [f for f in files if f.split('_')[0] in test_ids]

        print(len(train_files))
        print(len(val_files))
        print(len(test_files))

        return train_files, val_files, test_files

    def process_features_and_labels(self, file_list, tactile_data, audio_data):
        tactile_features = []
        audio_features = []
        labels = []

        print(len(file_list), len(tactile_data), len(audio_data))
        # 296 1294 1294

        for f in file_list:
            # print(f)
            # print(f, tactile_data[f].shape, audio_data[f].shape)
            # # 1_11---Rub__Round_3.npy
            # # 1_11---Rub__Round_3.npy (450, 25) (1001, 64)
            if not tactile_data[f].shape[0]==450 and tactile_data[f].shape[1]==25:
                print('f name: ', f)
                print(f, tactile_data[f].shape, audio_data[f].shape)

            tactile_features.append(tactile_data[f])
            audio_features.append(audio_data[f])
            
            label = self.extract_label(f, self.Gesture_list_name)
            labels.append(label)

        return np.array(tactile_features), np.array(audio_features),  np.array(labels)

    def process_single_features_and_labels(self, file_list, feature_data):
        features = []
        labels = []

        print(len(file_list), len(feature_data))

        for f in file_list:
            features.append(feature_data[f])

            label = self.extract_label(f, self.Gesture_list_name)
            labels.append(label)

        return np.array(features), np.array(labels)

    def clip_features(self, features, clip_length):
        return features[:, :int(clip_length * 100)]

    def pad_or_clip(self, feature, target_length):
        current_length = feature.shape[0]
        if current_length < target_length:
            pad_width = ((0, target_length - current_length), (0, 0))
            return np.pad(feature, pad_width, mode='constant')
        return feature[:target_length]

    def extract_label(self, filename, space_list):
        # for idx, label in enumerate(self.Gesture_list_name + self.emotion_list_name):
        for idx, label in enumerate(space_list):
            if label in filename:
                return idx
        raise ValueError(f"Unknown label in filename: {filename}")

    def save_pickle(self, data, file):
        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def generate_training(self):
        if self.modality is not None:
            audios_num = len(self.training_feature)

            audio_indexes = [i for i in range(audios_num)]

            self.random_state.shuffle(audio_indexes)

            iteration = 0
            pointer = 0

            while True:
                if pointer >= audios_num:
                    pointer = 0
                    self.random_state.shuffle(audio_indexes)

                batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
                pointer += self.batch_size

                iteration += 1

                batch_x = self.training_feature[batch_audio_indexes]
                batch_x = self.transform(batch_x, self.mean_feature, self.std_feature)

                batch_y = self.training_label[batch_audio_indexes]

                yield batch_x, batch_y

        audios_num = len(self.gesture_training_t)

        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x_a = self.gesture_training_a[batch_audio_indexes]
            batch_x_a = self.transform(batch_x_a, self.mean_audio, self.std_audio)

            batch_x_t = self.gesture_training_t[batch_audio_indexes]
            batch_x_t = self.transform(batch_x_t, self.mean_tactile, self.std_tactile)

            batch_y = self.gesture_training_label[batch_audio_indexes]

            # print(batch_y)

            yield batch_x_a, batch_x_t, batch_y


    def generate_validate(self, data_type, max_iteration=None):
        if self.modality is not None:
            audios_num = len(self.validation_feature)

            audio_indexes = [i for i in range(audios_num)]

            self.validate_random_state.shuffle(audio_indexes)

            print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

            iteration = 0
            pointer = 0

            while True:
                if iteration == max_iteration:
                    break

                if pointer >= audios_num:
                    break

                batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
                pointer += self.batch_size

                iteration += 1

                batch_x = self.validation_feature[batch_audio_indexes]
                batch_x = self.transform(batch_x, self.mean_feature, self.std_feature)

                batch_y = self.validation_label[batch_audio_indexes]

                yield batch_x, batch_y

            return

        # load
        # ------------------ validation --------------------------------------------------------------------------------

        audios_num = len(self.Emotion_validation_a)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x_a = self.Emotion_validation_a[batch_audio_indexes]
            batch_x_a = self.transform(batch_x_a, self.mean_audio, self.std_audio)

            batch_x_t = self.Emotion_validation_t[batch_audio_indexes]
            batch_x_t = self.transform(batch_x_t, self.mean_tactile, self.std_tactile)

            batch_y = self.Emotion_validation_label[batch_audio_indexes]

            # print(batch_y)

            yield batch_x_a, batch_x_t, batch_y


    def generate_testing(self, data_type, max_iteration=None):
        if self.modality is not None:
            audios_num = len(self.test_feature)

            audio_indexes = [i for i in range(audios_num)]

            self.validate_random_state.shuffle(audio_indexes)

            print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

            iteration = 0
            pointer = 0

            while True:
                if iteration == max_iteration:
                    break

                if pointer >= audios_num:
                    break

                batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
                pointer += self.batch_size

                iteration += 1

                batch_x = self.test_feature[batch_audio_indexes]
                batch_x = self.transform(batch_x, self.mean_feature, self.std_feature)

                batch_y = self.test_label[batch_audio_indexes]

                yield batch_x, batch_y

            return

        audios_num = len(self.Emotion_test_a)

        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x_a = self.Emotion_test_a[batch_audio_indexes]
            batch_x_a = self.transform(batch_x_a, self.mean_audio, self.std_audio)

            batch_x_t = self.Emotion_test_t[batch_audio_indexes]
            batch_x_t = self.transform(batch_x_t, self.mean_tactile, self.std_tactile)

            batch_y = self.Emotion_test_label[batch_audio_indexes]

            # print(batch_y)

            yield batch_x_a, batch_x_t, batch_y


    def transform(self, x, mean, std):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, mean, std)


