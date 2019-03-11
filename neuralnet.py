import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from datetime import timedelta, datetime
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from math import floor
import time
from sklearn.utils import shuffle
from sys import getsizeof

PICKLE_MODE = False
DATE_PICKLE_NAME = "datagrabberdate.pickle"
CHANNELS_MODE = "channels_last"

BATCH_SIZE = 16
MIN_RAD = 1
MAX_RAD = 140

'''
# manager that helps deal with data to in order to directly feed into NN
# takes in datetime objects to represent where to start looking for each season
# returns tuple of 3 numpy arrays, radiance input data, weather classifier data, temperature data
'''
class TFDataManager:
    def __init__(self, summer_date, fall_date, data_format, input_per_epoch):
        self.__file_already_processed_list = []
        self.__all_dates = [summer_date, fall_date]
        self.__data_format = data_format
        self.__input_per_epoch = input_per_epoch

    def get_numpy_arrays(self):
        data_files_in_directory = [f for f in listdir("NumpyDataFiles/") if isfile(join("NumpyDataFiles/", f))]

        data_collected = False
        first_data_append = True

        rad_features = None
        weather_labels = None

        count_index = 1
        while not data_collected and len(self.__file_already_processed_list) < len(data_files_in_directory):
            for i in range(len(self.__all_dates)):
                rad_file = str(self.__all_dates[i].year) + "-" + str(self.__all_dates[i].month) + "-" + str(
                    self.__all_dates[i].day) + "-" + str(self.__all_dates[i].hour) + "-rad_feature.npy"
                weather_file = str(self.__all_dates[i].year) + "-" + str(self.__all_dates[i].month) + "-" + str(
                    self.__all_dates[i].day) + "-" + str(self.__all_dates[i].hour) + "-weather_label.npy"

                # data file for this date does not exist
                if rad_file not in data_files_in_directory and weather_file not in data_files_in_directory:
                    # print("Current date doesn't exist (", self.__self.__all_dates[i], ")")
                    # print("Skipping...\n")
                    continue

                # check these files off so we don't process them twice
                self.__file_already_processed_list.append(rad_file)
                self.__file_already_processed_list.append(weather_file)

                # get the numpy arrays from these files
                rad_file_data = np.load("NumpyDataFiles/" + rad_file)
                weather_file_data = np.load("NumpyDataFiles/" + weather_file)

                # bad data, skip this iteration
                if rad_file_data.shape == (0,) or weather_file_data.shape == (0,):
                    # print("Bad file, skipping...\n")
                    continue

                print("Found valid file", rad_file, weather_file)
                print("Getting values now...")
                if first_data_append:
                    rad_features = rad_file_data
                    weather_labels = weather_file_data

                else:
                    rad_features = np.concatenate((rad_features, rad_file_data), 0)
                    weather_labels = np.concatenate((weather_labels, weather_file_data), 0)

                print("Sub Shape:", rad_file_data.shape)
                print("Total Shape", rad_features.shape)
                print("Index", count_index)
                print()

                count_index += 1
                first_data_append = False

            for i in range(len(self.__all_dates)):
                self.__all_dates[i] = self.__all_dates[i] + timedelta(hours=1)

            if rad_features is not None and weather_labels is not None and rad_features.shape[0] == weather_labels.shape[0] and rad_features.shape[0] > self.__input_per_epoch:
                # print("Breaking out of loop\n")
                data_collected = True

        if rad_features is None:
            return None, None  # haha

        print("Done getting data for this epoch, here are the shapes:")
        print(weather_labels.shape)
        print(rad_features.shape)

        # crop them to be divisible by batch size
        rad_features = TFDataManager.format_numpy_arrays(rad_features)
        weather_labels = TFDataManager.format_numpy_arrays(weather_labels)

        # shuffle the arrays, mainly for splitting up validation data
        rad_features, weather_labels = shuffle(rad_features, weather_labels)

        return rad_features, weather_labels
    
    @staticmethod
    def split_weather_data(weather_array):
        # transform the cloud + condition labels to one array
        cloud_labels = weather_array[:, 1]
        condition_labels = weather_array[:, 2]
        temp_labels = weather_array[:, 0]
        
        stack_clouds = np.stack(cloud_labels)
        stack_condition = np.stack(condition_labels)

        labels_classifier = np.concatenate((stack_clouds, stack_condition), 1)
        labels_temperature = np.stack(temp_labels)
        
        return labels_classifier, labels_temperature
    
    @staticmethod
    def augment_data(rad_array, class_array, temp_array):
        original_size = rad_array.shape[0]
        print("-----------BEGIN DATA AUGMENTATION-------------")
        print("Before augmentation", rad_array.shape)

        # flip images left-right
        flipped_features = np.flip(rad_array[0 : int(original_size / 2)], (2, 3))
        rad_features_augmented = np.concatenate((rad_array, flipped_features))
        class_labels_augmented = np.concatenate((class_array, class_array[0 : int(original_size / 2)]))
        temp_array_augmented = np.concatenate((temp_array, temp_array[0 : int(original_size / 2)]))
        print("Flip augment rad", rad_features_augmented.shape)
        print("Flip augment class + temp", class_labels_augmented.shape, temp_array_augmented.shape)

        # rotate images by 90 deg
        rotated_features = np.rot90(rad_array[int(original_size / 2):], axes=(2, 3))
        rad_features_augmented = np.concatenate((rad_features_augmented, rotated_features))
        class_labels_augmented = np.concatenate((class_labels_augmented, class_array[int(original_size / 2):]))
        temp_array_augmented = np.concatenate((temp_array_augmented, temp_array[int(original_size / 2):]))
        print("Rotate augment rad", rad_features_augmented.shape)
        print("Rotate augment class + temp", class_labels_augmented.shape, temp_array_augmented.shape)

        # add gaussian noise
        noise = TFDataManager.add_gaussian_noise(rad_features_augmented)
        rad_features_augmented = np.concatenate((rad_features_augmented, noise))
        class_labels_augmented = np.concatenate((class_labels_augmented, class_labels_augmented))
        temp_array_augmented = np.concatenate((temp_array_augmented, temp_array_augmented))
        print("Noise augment rad", rad_features_augmented.shape)
        print("Noise augment class + temp", class_labels_augmented.shape, temp_array_augmented.shape)

        print("-----------END DATA AUGMENTATION-------------")


        return rad_features_augmented, class_labels_augmented, temp_array_augmented
            
    # def get_data_loop(self):
    #     if PICKLE_MODE:
    #         print("Using Pickled Date")
    #         data_datetime = pickle.load(open(DATE_PICKLE_NAME, "rb"))
    #         data_date = (data_datetime.year, data_datetime.month, data_datetime.day, data_datetime.hour)
    #     else:
    #         data_date = (2017, 8, 8, 1)
    #         print("Using explicitly set date:", data_date)
    #
    #     data_retriever = DataManager(starting_date=data_date, channels=["C13", "C14", "C15", "C16"])
    #
    #     print(data_retriever.print_all_states())
    #
    #     while data_date[0] is not 2018 and data_date[1] is not 12:
    #         radiance_features, weather_labels = data_retriever.get_formatted_data()
    #
    #         if radiance_features is not None:
    #             radiance_features_nparray = np.array(radiance_features)
    #             weather_labels_nparray = np.array(weather_labels)
    #
    #             save_path = str.format("NumpyDataFiles/{0}-{1}-{2}-{3}", *data_date)
    #
    #             np.save(save_path + "-rad_feature", radiance_features_nparray)
    #             np.save(save_path + "-weather_label", weather_labels_nparray)
    #
    #         data_retriever.increment_date()
    #         data_datetime = data_retriever.get_current_date()
    #         data_date = (data_datetime.year, data_datetime.month, data_datetime.day, data_datetime.hour)
    #
    #         data_retriever.pickle_date()
    #         print("Sucesfully pickled current date")
    #         print("Done with 1 hour iteration... moving on to ", data_date)



    @staticmethod
    def add_gaussian_noise(array):
        return array + np.random.normal(0, 1, size=array.shape)

    @staticmethod
    def format_numpy_arrays(array):
        size_to_crop_to = floor(array.shape[0] / BATCH_SIZE) * BATCH_SIZE
        difference = abs(array.shape[0] - size_to_crop_to)

        return np.delete(array, np.s_[:difference], 0)
    
    @staticmethod
    def normalize_radiance_array(rad_array):
        for entry_index in range(rad_array.shape[0]):
            for channel_index in range(rad_array.shape[1]):
                # channel 13
                if channel_index == 0:
                    rad_array[entry_index, channel_index]  = (rad_array[entry_index, channel_index] - -0.4935) / (183.62 - -0.4935)

                # channel 14
                elif channel_index == 1:
                    rad_array[entry_index, channel_index]  = (rad_array[entry_index, channel_index] - -0.5154) / (198.71 - -0.5154)

                # channel 15
                elif channel_index == 2:
                    rad_array[entry_index, channel_index]  = (rad_array[entry_index, channel_index] - -0.5262) / (212.28 - -0.5262)

                # channel 16
                elif channel_index == 3:
                    rad_array[entry_index, channel_index]  = (rad_array[entry_index, channel_index] - -1.5726) / (170.19 - -1.5726)

        return rad_array

    @staticmethod
    def split_validation_data(rad_array, classify_array, temperature_array, validate_percent):
        assert validate_percent < 1
        validation_index = floor(rad_array.shape[0] * validate_percent)

        rad_validate = rad_array[0:validation_index]
        classify_validate = classify_array[0:validation_index]
        temperature_validate = temperature_array[0:validation_index]

        rad_array = np.delete(rad_array, np.s_[0:validation_index], 0)
        classify_array = np.delete(classify_array, np.s_[0:validation_index], 0)
        temperature_array = np.delete(temperature_array, np.s_[0:validation_index], 0)

        return rad_array, classify_array, temperature_array, rad_validate, classify_validate, temperature_validate

class NeuralNet:
    
    def __init__(self, width, height, data_format):
        self.__width = width
        self.__height = height
        self.__data_format = data_format

    def create_model(self):
        if self.__data_format == "channels_last":
            inputs = tf.keras.Input(shape=(self.__height, self.__width, 4))
        else:
            inputs = tf.keras.Input(shape=(4, self.__height, self.__width))

        cloud_classifier_branch = self.__create_cloud_classifier_branch(inputs)
        condition_classifier_branch = self.__create_condition_classifier_branch(inputs)
        temperature_branch = self.__create_temperature_branch(inputs)

        tfmodel = tf.keras.Model(inputs=inputs, outputs=[cloud_classifier_branch, condition_classifier_branch, temperature_branch])

        losses = {"cloud_classifier": "categorical_crossentropy",
                  "condition_classifier": "binary_crossentropy",
                  "temperature_output": "categorical_crossentropy"}

        metrics = {"cloud_classifier": "accuracy",
                   "condition_classifier": "accuracy",
                   "temperature_output": "accuracy"}

        tfmodel.compile(optimizer=tf.keras.optimizers.Adam(), metrics=metrics, loss=losses)

        return tfmodel

    def __create_condition_classifier_branch(self, inputs):
        print(inputs.shape)
        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        print(x.shape)

        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        print(x.shape)

        x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        print(x.shape)

        x = layers.Flatten()(x)
        x = layers.Dense(320, activation="relu", bias_regularizer=regularizers.l2(.01))(x)
        print(x.shape)
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(4, activation="sigmoid", name="condition_classifier")(x)

        print(x.shape)
        return x

    def __create_cloud_classifier_branch(self, inputs):
        print(inputs.shape)
        x = layers.Conv2D(filters=12, kernel_size=5, activation="relu", data_format=self.__data_format)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        print(x.shape)

        # x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPool2D(pool_size=2)(x)
        # print(x.shape)
        #
        # x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPool2D(pool_size=2)(x)
        # print(x.shape)

        x = layers.Flatten()(x)
        x = layers.Dense(50, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        print(x.shape)

        x = layers.Dense(6, activation="softmax", name="cloud_classifier")(x)
        print(x.shape)

        return x

    def __create_temperature_branch(self, inputs):
        x = layers.Conv2D(filters=128, kernel_size=5, activation="relu", data_format=self.__data_format)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Conv2D(filters=64, kernel_size=2, activation="relu", data_format=self.__data_format)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Conv2D(filters=32, kernel_size=2, activation="relu", data_format=self.__data_format)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(300, activation="relu", bias_regularizer=regularizers.l2(.05))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(201, activation="softmax", name="temperature_output")(x)

        return x

class MainDriver:
    def train(self):
        summer_time = datetime(year=2017, month=8, day=1, hour=0)
        fall_time = datetime(year=2017, month=10, day=1, hour=0)

        net = NeuralNet(100, 100, CHANNELS_MODE)
        data_manager = TFDataManager(summer_date=summer_time, fall_date=fall_time, data_format=CHANNELS_MODE, input_per_epoch=1000)

        model = net.create_model()

        cp_callback = tf.keras.callbacks.ModelCheckpoint("checkpoints/cp.cpkt", verbose=1)

        forever_loop = True
        while forever_loop:
            rad_features, weather_labels = data_manager.get_numpy_arrays()

            print("Rad Features, Weather Labels:", rad_features.shape, weather_labels.shape)
            if rad_features is None:
                print("Waiting...")
                # time.sleep(5)
                break

            # split weather data
            class_label, temp_label = TFDataManager.split_weather_data(weather_labels)

            print("Class, temp (shape)", class_label.shape, temp_label.shape)

            # split up validation data ONLY on non-augmented data
            rad_features, class_label, temp_label, rad_validate, class_validate, temp_validate = TFDataManager.split_validation_data(rad_array=rad_features, classify_array=class_label,
                                                                                                                                     temperature_array=temp_label, validate_percent=0.2)
            print("Validation Size:", rad_validate.shape, class_validate.shape, temp_validate.shape)

            # augment the data
            rad_features, class_label, temp_label = TFDataManager.augment_data(rad_features, class_label, temp_label)

            # normalize data between 0-1
            rad_features = TFDataManager.normalize_radiance_array(rad_features)

            # transpose if needed
            rad_features = rad_features.transpose([0, 2, 3, 1])
            rad_validate = rad_validate.transpose([0, 2, 3, 1])

            print(rad_features.shape, class_label.shape, temp_label.shape)
            print(rad_features.nbytes / 1000000, class_label.nbytes / 1000000, temp_label.nbytes / 1000000)

            # split up the weather data, both labels + validation labels
            clouds_label = class_label[:, 0:6]
            conditions_label = class_label[:, 6:10]

            clouds_validate = class_validate[:, 0:6]
            conditions_validate = class_validate[:, 6:10]

            outputs = {"cloud_classifier": clouds_label,
                       "condition_classifier": conditions_label,
                       "temperature_output": temp_label}

            weather_validation_set = {"cloud_classifier": clouds_validate,
                       "condition_classifier": conditions_validate,
                       "temperature_output": temp_validate}

            model.fit(rad_features, outputs, batch_size=BATCH_SIZE, epochs=20, validation_data=(rad_validate, weather_validation_set)
                      , verbose=1, callbacks=[cp_callback])

            model.save("model.hd5")
            forever_loop = True

if __name__ == "__main__":
    main = MainDriver()
    main.train()

