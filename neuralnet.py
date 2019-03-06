import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from datagrabber import DataManager
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, datetime
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from math import floor

PICKLE_MODE = False
DATE_PICKLE_NAME = "datagrabberdate.pickle"

BATCH_SIZE = 8

summer_date = datetime(year=2017, month=8, day=1, hour=0)
fall_date = datetime(year=2017, month=10, day=1, hour=0)
all_dates = [summer_date, fall_date]

file_already_processed_list = []

MIN_RAD = 1
MAX_RAD = 140


def get_data_loop():
    if PICKLE_MODE:
        print("Using Pickled Date")
        data_datetime = pickle.load(open(DATE_PICKLE_NAME, "rb"))
        data_date = (data_datetime.year, data_datetime.month, data_datetime.day, data_datetime.hour)
    else:
        data_date = (2017, 8, 8, 1)
        print("Using explicitly set date:", data_date)

    data_retriever = DataManager(starting_date=data_date, channels=["C13", "C14", "C15", "C16"])

    print(data_retriever.print_all_states())

    while data_date[0] is not 2018 and data_date[1] is not 12:
        radiance_features, weather_labels = data_retriever.get_formatted_data()

        if radiance_features is not None:
            radiance_features_nparray = np.array(radiance_features)
            weather_labels_nparray = np.array(weather_labels)

            save_path = str.format("NumpyDataFiles/{0}-{1}-{2}-{3}", *data_date)

            np.save(save_path + "-rad_feature", radiance_features_nparray)
            np.save(save_path + "-weather_label", weather_labels_nparray)

        data_retriever.increment_date()
        data_datetime = data_retriever.get_current_date()
        data_date = (data_datetime.year, data_datetime.month, data_datetime.day, data_datetime.hour)

        data_retriever.pickle_date()
        print("Sucesfully pickled current date")
        print("Done with 1 hour iteration... moving on to ", data_date)


def format_numpy_arrays(array):
    size_to_crop_to = floor(array.shape[0] / BATCH_SIZE) * BATCH_SIZE
    difference = abs(array.shape[0] - size_to_crop_to)

    return np.delete(array, np.s_[:difference], 0)


def normalize_radiance_array(rad_array):
    for entry_index in range(rad_array.shape[0]):
        for channel_index in range(rad_array.shape[1]):
            scaler = MinMaxScaler()

            scaler.fit([[MIN_RAD], [MAX_RAD]])
            rad_array[entry_index, channel_index] = scaler.transform(rad_array[entry_index, channel_index])

    return rad_array


def grab_numpy_arrays():
    global all_dates

    data_files_in_directory = [f for f in listdir("NumpyDataFiles/") if isfile(join("NumpyDataFiles/", f))]

    data_collected = False
    first_data_append = True

    features = None
    labels = None

    while not data_collected and len(file_already_processed_list) < len(data_files_in_directory):
        for i in range(len(all_dates)):
            rad_file = str(all_dates[i].year) + "-" + str(all_dates[i].month) + "-" + str(all_dates[i].day) + "-" + str(all_dates[i].hour) + "-rad_feature.npy"
            weather_file = str(all_dates[i].year) + "-" + str(all_dates[i].month) + "-" + str(all_dates[i].day) + "-" + str(all_dates[i].hour) + "-weather_label.npy"

            # data file for this date does not exist
            if rad_file not in data_files_in_directory and weather_file not in data_files_in_directory:
                # print("Current date doesn't exist (", all_dates[i], ")")
                # print("Skipping...\n")
                continue

            # check these files off so we don't process them twice
            file_already_processed_list.append(rad_file)
            file_already_processed_list.append(weather_file)

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
                features = rad_file_data
                labels = weather_file_data

            else:
                features = np.concatenate((features, rad_file_data), 0)
                labels = np.concatenate((labels, weather_file_data), 0)

            print("Sub Shape:", rad_file_data.shape)
            print("Total Shape", features.shape)
            print()

            first_data_append = False

        for i in range(len(all_dates)):
            all_dates[i] = all_dates[i] + timedelta(hours=1)

        if features is not None and labels is not None and features.shape[0] == labels.shape[0] and features.shape[0] > 300:
            # print("Breaking out of loop\n")
            data_collected = True

    print("Done getting data for this epoch, here are the shapes:")
    print(labels.shape)
    print(features.shape)

    return features, labels


def create_temperature_branch(inputs):
    x = layers.Conv2D(filters=12, kernel_size=2, activation="relu", data_format="channels_first")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=1, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="sigmoid")(x)
    x = layers.Dense(200, activation="softmax", name="temperature_output")(x)

    return x


def create_classifier_branch(inputs, name):
    print (inputs.shape)
    x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format="channels_first")(inputs)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)


    x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format="channels_first")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)

    x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", data_format="channels_first")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)

    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="sigmoid", bias_regularizer=regularizers.l2(.01))(x)
    print(x.shape)

    x = layers.Dense(10, activation="sigmoid", name=name)(x)

    print(x.shape)
    return x


def create_model(width, height):
    inputs = tf.keras.Input(shape=(4, height, width))

    cloud_branch = create_classifier_branch(inputs, name="cloud_branch")
    weather_classify_branch = create_classifier_branch(inputs, name="weather_condition")
    temperature_branch = create_temperature_branch(inputs)

    tfmodel = tf.keras.Model(inputs=inputs, outputs=[cloud_branch, weather_classify_branch, temperature_branch])

    return tfmodel


if __name__ == "__main__":
    features, labels = grab_numpy_arrays()
    features = format_numpy_arrays(features)
    labels = format_numpy_arrays(labels)

    features = normalize_radiance_array(features)

    print(features.shape)

    model = create_model(100, 100)
    losses = {"cloud_branch": "categorical_crossentropy",
              "temperature_output": "categorical_crossentropy",
              "weather_condition": "categorical_crossentropy"}

    model.compile(optimizer=tf.train.AdamOptimizer(), metrics=["accuracy"], loss=losses)

    labels_classifier = labels[:, 1:3]
    cloud_labels = labels_classifier[:, 0]
    labels_temperature = labels[:, 0]
    print(labels_classifier[:][1].shape)
    # model.fit(features, {"classifier_output": labels_classifier, "temperature_output": labels_temperature}, batch_size=BATCH_SIZE, epochs=1)



