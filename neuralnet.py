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
import matplotlib.pyplot as plt

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
    def __init__(self, summer_date, fall_date, winter_date, spring_date, data_format, input_per_epoch):
        self.__file_already_processed_list = []
        self.__all_dates = [summer_date, fall_date, winter_date, spring_date]
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

            if rad_features is not None and weather_labels is not None and rad_features.shape[0] == \
                    weather_labels.shape[0] and rad_features.shape[0] > self.__input_per_epoch:
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
        print("Before augmentation", class_array.shape)

        rad_of_NCD = []
        class_of_NCD = []
        temp_of_NCD = []

        for i in range(len(class_array)):
            if class_array[i][0] == 0 or np.where(class_array[i][6:10] == 1)[0] > 0:
                rad_of_NCD.append(rad_array[i])
                class_of_NCD.append(class_array[i])
                temp_of_NCD.append(temp_array[i])

        rad_of_NCD = np.array(rad_of_NCD)
        class_of_NCD = np.array(class_of_NCD)
        temp_of_NCD = np.array(temp_of_NCD)

        print("Size of NCD array: ", rad_of_NCD.shape)

        # rot 1
        rad_of_NCD_augmented = np.concatenate((rad_of_NCD, np.rot90(rad_of_NCD, 2, axes=(2, 3))))
        class_of_NCD_augmented = np.concatenate((class_of_NCD, class_of_NCD))
        temp_of_NCD_augmented = np.concatenate((temp_of_NCD, temp_of_NCD))

        rad_of_NCD_augmented = np.concatenate((rad_of_NCD_augmented, np.rot90(rad_of_NCD, 3, axes=(2, 3))))
        class_of_NCD_augmented = np.concatenate((class_of_NCD_augmented, class_of_NCD))
        temp_of_NCD_augmented = np.concatenate((temp_of_NCD_augmented, temp_of_NCD))

        print("rad aug shape", rad_of_NCD_augmented.shape)


        rad_noise = np.concatenate((rad_of_NCD_augmented, TFDataManager.add_gaussian_noise(rad_of_NCD_augmented)))
        rad_noise = np.concatenate((rad_noise, TFDataManager.add_gaussian_noise(rad_of_NCD_augmented, std=0.5)))
        rad_noise = np.concatenate((rad_noise, TFDataManager.add_gaussian_noise(rad_of_NCD_augmented, std=0.3)))

        print("rad noise shape", rad_noise.shape)
        print("class shape before repeat", class_of_NCD_augmented.shape)
        rad_of_NCD_augmented = np.concatenate((rad_of_NCD_augmented, rad_noise))

        class_of_NCD_augmented = np.concatenate((class_of_NCD_augmented, np.repeat(class_of_NCD_augmented, 4, 0)))
        temp_of_NCD_augmented = np.concatenate((temp_of_NCD_augmented, np.repeat(temp_of_NCD_augmented, 4, 0)))

        print(rad_of_NCD_augmented.shape)
        print(class_of_NCD_augmented.shape)
        print(temp_of_NCD_augmented.shape)


        '''
        End of NCD stuff
        '''

        # flip images left-right
        flipped_features = np.flip(rad_array, (2, 3))
        rad_features_augmented = np.concatenate((rad_array, flipped_features))
        class_labels_augmented = np.concatenate((class_array, class_array))
        temp_array_augmented = np.concatenate((temp_array, temp_array))
        print("Flip augment rad", rad_features_augmented.shape)
        print("Flip augment class + temp", class_labels_augmented.shape, temp_array_augmented.shape)

        # rotate images by 90 deg
        rotated_features = np.rot90(rad_array, axes=(2, 3))
        rad_features_augmented = np.concatenate((rad_features_augmented, rotated_features))
        class_labels_augmented = np.concatenate((class_labels_augmented, class_array))
        temp_array_augmented = np.concatenate((temp_array_augmented, temp_array))
        print("Rotate augment rad", rad_features_augmented.shape)
        print("Rotate augment class + temp", class_labels_augmented.shape, temp_array_augmented.shape)

        # add gaussian noise
        noise = TFDataManager.add_gaussian_noise(rad_features_augmented)
        rad_features_augmented = np.concatenate((rad_features_augmented, noise))
        class_labels_augmented = np.concatenate((class_labels_augmented, class_labels_augmented))
        temp_array_augmented = np.concatenate((temp_array_augmented, temp_array_augmented))

        # combine NCD data augment and normal data augment
        rad_features_augmented = np.concatenate((rad_features_augmented, rad_of_NCD_augmented))
        class_labels_augmented = np.concatenate((class_labels_augmented, class_of_NCD_augmented))
        temp_array_augmented = np.concatenate((temp_array_augmented, temp_of_NCD_augmented))

        print("Noise augment rad", rad_features_augmented.shape)
        print("Noise augment class + temp", class_labels_augmented.shape, temp_array_augmented.shape)

        print("-----------END DATA AUGMENTATION--------------")

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
    def add_gaussian_noise(array, std=1.0):
        return array + np.random.normal(0, std, size=array.shape)

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
                    rad_array[entry_index, channel_index] = (rad_array[entry_index, channel_index] - -0.4935) / (
                                183.62 - -0.4935)

                # channel 14
                elif channel_index == 1:
                    rad_array[entry_index, channel_index] = (rad_array[entry_index, channel_index] - -0.5154) / (
                                198.71 - -0.5154)

                # channel 15
                elif channel_index == 2:
                    rad_array[entry_index, channel_index] = (rad_array[entry_index, channel_index] - -0.5262) / (
                                212.28 - -0.5262)

                # channel 16
                elif channel_index == 3:
                    rad_array[entry_index, channel_index] = (rad_array[entry_index, channel_index] - -1.5726) / (
                                170.19 - -1.5726)

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

        tfmodel = tf.keras.Model(inputs=inputs,
                                 outputs=[cloud_classifier_branch, condition_classifier_branch, temperature_branch])

        losses = {"cloud": "categorical_crossentropy",
                  "condition": "binary_crossentropy",
                  "temp": "categorical_crossentropy"}

        metrics = {"cloud": "accuracy",
                   "condition": "accuracy",
                   "temp": "accuracy"}

        tfmodel.compile(optimizer=tf.keras.optimizers.Adam(), metrics=metrics, loss=losses)

        return tfmodel

    def __create_condition_classifier_branch(self, inputs):
        leaky_RELU = tf.keras.layers.LeakyReLU(alpha=0.01)

        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(inputs)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(320, activation="relu", bias_regularizer=regularizers.l2(.01))(x)
        
        x = layers.Dropout(0.1)(x)

        x = layers.Dense(4, activation="sigmoid", name="condition")(x)

        return x

    def __create_cloud_classifier_branch(self, inputs):
        leaky_RELU = tf.keras.layers.LeakyReLU(alpha=0.01)

        print(inputs.shape)
        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(inputs)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        # x = layers.Dropout(0.2)(x)
        print(x.shape)

        x = layers.Conv2D(filters=32, kernel_size=2, activation="relu", data_format=self.__data_format)(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        # x = layers.Dropout(0.2)(x)
        print(x.shape)
        #
        # x = layers.Conv2D(filters=32, kernel_size=5, activation=leaky_RELU, data_format=self.__data_format)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPool2D(pool_size=2)(x)
        # print(x.shape)

        x = layers.Flatten()(x)
        x = layers.Dense(100)(x)
        
        # x = layers.Dropout(0.4)(x)
        print(x.shape)

        x = layers.Dense(6, activation="softmax", name="cloud")(x)
        print(x.shape)

        return x

    def __create_temperature_branch(self, inputs):
        leaky_RELU = tf.keras.layers.LeakyReLU(alpha=0.01)
        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(inputs)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        # x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(filters=64, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)
        # x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", data_format=self.__data_format)(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D(pool_size=2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(300, activation="relu", bias_regularizer=regularizers.l2(.05))(x)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(201, activation="softmax", name="temp")(x)

        return x


class MainDriver:
    def train(self):
        summer_time = datetime(year=2017, month=8, day=1, hour=0)
        fall_time = datetime(year=2017, month=10, day=1, hour=0)
        winter_time = datetime(year=2018, month=2, day=1, hour=0)
        spring_time = datetime(year=2018, month=6, day=1, hour=0)

        net = NeuralNet(100, 100, CHANNELS_MODE)
        data_manager = TFDataManager(summer_date=summer_time, fall_date=fall_time, winter_date=winter_time,
                                     spring_date=spring_time, data_format=CHANNELS_MODE, input_per_epoch=350)

        model = net.create_model()

        cp_callback = tf.keras.callbacks.ModelCheckpoint("checkpoints/cp.cpkt", verbose=1)

        forever_loop = True
        while forever_loop:
            rad_features, weather_labels = data_manager.get_numpy_arrays()

            if rad_features is None:
                print("Waiting...")
                # time.sleep(5)
                break

            print("Rad Features, Weather Labels:", rad_features.shape, weather_labels.shape)

            # split weather data
            class_label, temp_label = TFDataManager.split_weather_data(weather_labels)

            print("Class, temp (shape)", class_label.shape, temp_label.shape)



            # split up validation data ONLY on non-augmented data
            rad_features, class_label, temp_label, rad_validate, class_validate, temp_validate = TFDataManager.split_validation_data(
                rad_array=rad_features, classify_array=class_label,
                temperature_array=temp_label, validate_percent=0.3)
            print("Validation Size:", rad_validate.shape, class_validate.shape, temp_validate.shape)

            #TODO see if this fixes validation
            # augment the data
            rad_features, class_label, temp_label = TFDataManager.augment_data(rad_features, class_label, temp_label)

            # normalize data between 0-1
            rad_features = TFDataManager.normalize_radiance_array(rad_features)

            # transpose
            rad_features = rad_features.transpose([0, 2, 3, 1])
            rad_validate = rad_validate.transpose([0, 2, 3, 1])

            print(rad_features.shape, class_label.shape, temp_label.shape)
            print(rad_features.nbytes / 1000000, class_label.nbytes / 1000000, temp_label.nbytes / 1000000)

            # split up the weather data, both labels + validation labels
            clouds_label = class_label[:, 0:6]
            conditions_label = class_label[:, 6:10]

            clouds_validate = class_validate[:, 0:6]
            conditions_validate = class_validate[:, 6:10]

            outputs = {"cloud": clouds_label,
                       "condition": conditions_label,
                       "temp": temp_label}

            weather_validation_set = {"cloud": clouds_validate,
                                      "condition": conditions_validate,
                                      "temp": temp_validate}

            history = model.fit(rad_features, outputs, batch_size=BATCH_SIZE, epochs=5,
                                validation_data=(rad_validate, weather_validation_set)
                                , verbose=1, callbacks=[cp_callback])

            print("Iteration complete, saving model...")
            model.save("model.hd5")

            print(history.history.keys())

            self.save_graphs(history)

            forever_loop = True

    def save_graphs(self, history):
        # clouds
        # Plot training & validation accuracy values
        plt.plot(history.history['cloud_acc'])
        plt.plot(history.history['val_cloud_acc'])
        plt.title('Cloud Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig("cloud_accuracy_plot.png")
        plt.clf()

        # Plot training & validation loss values
        plt.plot(history.history['cloud_loss'])
        plt.plot(history.history['val_cloud_loss'])
        plt.title('Cloud Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig("cloud_loss_plot.png")
        plt.clf()

        # conditions
        # Plot training & validation accuracy values
        plt.plot(history.history['condition_acc'])
        plt.plot(history.history['val_condition_acc'])
        plt.title('Condition Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig("condition_accuracy_plot.png")
        plt.clf()

        # Plot training & validation loss values
        plt.plot(history.history['condition_loss'])
        plt.plot(history.history['val_condition_loss'])
        plt.title('Condition Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig("condition_loss_plot.png")
        plt.clf()

        # temperature
        # Plot training & validation accuracy values
        plt.plot(history.history['temp_acc'])
        plt.plot(history.history['val_temp_acc'])
        plt.title('Temperature Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig("temperature_accuracy_plot.png")
        plt.clf()

        # Plot training & validation loss values
        plt.plot(history.history['temp_loss'])
        plt.plot(history.history['val_temp_loss'])
        plt.title('Temperature Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        plt.savefig("temperature_loss_plot.png")
        plt.clf()

    def interpret_data(self):
        model = tf.keras.models.load_model("model.hd5")
        rad = np.load("rad-eval-to_columbus.npy")
        rad2 = np.load("rad-eval-to_finish.npy")

        rad = np.concatenate((rad, rad2))
        rad = rad.transpose([0, 2, 3, 1])

        predictions = model.predict(rad)

        for i in range(len(predictions)):
            print("Index:", i)
            print(predictions[i].shape)

            if i == 0:
                print("Cloud Conditions:")
                for cloud in predictions[i]:
                    print(cloud)

            elif i == 1:
                print("Weather Conditions:")
                for weather in predictions[i]:
                    print(weather)

            elif i == 2:
                print("Temperature:")
                for temp in predictions[i]:
                    print(temp)
                    temp_array = np.where(temp == 1)[0]
                    if len(temp_array) > 0:
                        temp_index = temp_array[0]
                        actual_temp = temp_index - 70
                        print(actual_temp)

    def plot_model(self):
        model = tf.keras.models.load_model("model_saves/model_with_NCD_aug.hd5")
        tf.keras.utils.plot_model(model, to_file="model_plot.png")

if __name__ == "__main__":
    # only if running with GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    # supress sci-notation
    np.set_printoptions(suppress=True)

    main = MainDriver()
    # main.interpret_data()
    # main.train()
    main.plot_model()


