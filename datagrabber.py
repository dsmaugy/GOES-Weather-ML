import csv
from os import listdir
from os.path import isfile, join
from datetime import timedelta, datetime
from netCDF4 import Dataset
import time
import timezonefinder
import pytz
import dateutil.parser
import numpy as np
import gcstools
import pickle

NO_DOWNLOAD_MODE = False
DATE_PICKLE_NAME = "datagrabberdate.pickle"
PICKLE_MODE = False


class CsvDataGrabber:

    # initialize a data grabber object
    # state is the file name
    def __init__(self, state, starting_date=(2017, 8, 1, 1)):
        self.__state = state
        self.__CSV_PATH = "WeatherData/" + str(state)
        self.__file_is_active = False
        self.__csv_reader = csv.DictReader  # placeholder value, probably very unsafe
        self.__current_date = starting_date
        self.__timezone_dict = {}

    def __read_csv(self, csv_reader, input_year, input_month, input_day, input_hour):
        i = 0
        start_time = time.time()
        tf = timezonefinder.TimezoneFinder()

        for row in csv_reader:
            lat = float(row["LATITUDE"])
            lng = float(row["LONGITUDE"])


            # append lat + long coords to dictionary with timezone to save processing time
            if (lat, lng) in self.__timezone_dict:
                timezone_str = self.__timezone_dict[(lat, lng)]
            else:
                timezone_str = tf.timezone_at(lat=lat, lng=lng)
                self.__timezone_dict[(lat, lng)] = timezone_str

            timezone = pytz.timezone(timezone_str)

            row_date = dateutil.parser.parse(row["DATE"])
            # row_date = datetime.strptime(row["DATE"], "%Y-%m-%d %H:%M")

            # make the datetime object aware with timezone
            row_date = timezone.localize(row_date)

            # remove effects of DST
            if row_date.dst().seconds > 0:
                updated_row_date = row_date + timedelta(seconds=row_date.dst().seconds)
            else:
                updated_row_date = row_date # dumb utctimetuple doesn't update if we just replace row_date

            year = updated_row_date.utctimetuple().tm_year
            month = updated_row_date.utctimetuple().tm_mon
            day = updated_row_date.utctimetuple().tm_mday
            hour = updated_row_date.utctimetuple().tm_hour
            minute = updated_row_date.utctimetuple().tm_min

            if year == input_year:
                if month == input_month:
                    if day == input_day:
                        if hour == input_hour and minute == 0:

                            if len(row["HOURLYSKYCONDITIONS"]) > 0 and len(row["HOURLYDRYBULBTEMPF"]) > 0 and str.isnumeric(row["HOURLYDRYBULBTEMPF"]):
                                yield row
                        elif hour + 1 == input_hour:
                            if minute > 50:

                                if len(row["HOURLYSKYCONDITIONS"]) > 0 and len(row["HOURLYDRYBULBTEMPF"]) > 0 and str.isnumeric(row["HOURLYDRYBULBTEMPF"]):
                                    yield row

                        elif hour >= input_hour:
                            break


                    # same as the code block above but special case when we looking at hour == 0
                    elif day + 1 == input_day:
                        if input_hour == 0:
                            if hour == 23:
                                if minute > 50:
                                    if len(row["HOURLYSKYCONDITIONS"]) > 0 and len(row["HOURLYDRYBULBTEMPF"]) > 0 and str.isnumeric(row["HOURLYDRYBULBTEMPF"]):
                                        yield row



            # print("Local Time:", row_date, row_date.tzname(), "Iteration:", i)
            # print("UTC:", row_date.utctimetuple())
            # print()

            i += 1

        end_time = time.time()

        print("Time Elapsed:", end_time - start_time)
        print("Found Entries:")

    def __initialize_csv(self):
        csv_file = open(self.__CSV_PATH)
        return csv.DictReader(csv_file, delimiter=",")

    def close_file(self):
        self.csv_file.close()

    def find_row_by_time(self, input_year, input_month, input_day, input_hour):
        print("Finding Entry in: %s for %s" % (self.__state, self.__current_date))

        if not self.__file_is_active:
            self.__csv_reader = self.__initialize_csv()
            self.__file_is_active = True

        # find the rows for each date in this specified state
        matching_rows = []
        for row in self.__read_csv(self.__csv_reader, input_year, input_month, input_day, input_hour):
            matching_rows.append(row)

        # remove any duplicate data from the same station within the 10 minute grace period
        if len(matching_rows) > 5:
            indexes_to_pop = []  # stores the list of indexes that we can remove



            for i in range(0, len(matching_rows) - 1):
                name_to_check = matching_rows[i]["STATION"]

                # loop over the sublist from name_to_check to see if any dupes are found
                for j in range(i + 1, len(matching_rows)):
                    if matching_rows[j]["STATION"] == name_to_check:  # this means that the station has recorded another data entry within the 10 minute grace period
                        date_to_check = dateutil.parser.parse(matching_rows[i]["DATE"])
                        date_examined = dateutil.parser.parse(matching_rows[j]["DATE"])

                        if date_to_check.time() < date_examined.time():  # examined time is closer to the hour, get rid of station[i]
                            if i not in indexes_to_pop:
                                indexes_to_pop.append(i)
                        else:
                            if j not in indexes_to_pop:
                                indexes_to_pop.append(j)
            datetime
            # get rid of the duplicates
            for pop_index in sorted(indexes_to_pop, reverse=True):
                # if pop_index < len(matching_rows): # bruh
                matching_rows.pop(pop_index)

        for sanitized_row in matching_rows:
            print(sanitized_row["STATION_NAME"], sanitized_row["DATE"])


        return matching_rows

    def find_row_by_set_time(self):
        return self.find_row_by_time(*self.__current_date)

    def update_time(self, new_date):
        self.__current_date = new_date


class RadianceDataGrabber:

    def __init__(self, time_to_grab):
        self.time = time_to_grab

    def find_rad_by_set_time(self, channel):
        if not NO_DOWNLOAD_MODE:

            try:
                ncs_file_id = gcstools.get_objectId_at(self.time, product="ABI-L1b-RadC", channel=channel)
                rad_file = gcstools.copy_fromgcs(gcstools.GOES_PUBLIC_BUCKET, ncs_file_id, "SatFiles/SatFile-" + channel)
                print("Downloaded", rad_file)

            except:
                return None
            else:
                return rad_file
        else:
            rad_file = "SatFiles/SatFile-" + channel
            print("Found", rad_file)
            return rad_file


class DataManager:

    def __init__(self, starting_date, channels=("C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16")):
        self.__csv_states = []
        self.__current_date = datetime(year=starting_date[0], month=starting_date[1], day=starting_date[2], hour=starting_date[3])
        self.__channels = channels

        print("No Download Mode is:", NO_DOWNLOAD_MODE)
        self.__load_all_states()

    def __load_all_states(self):
        files = [f for f in listdir("WeatherData/") if isfile(join("WeatherData/", f))]

        for state in files:
            self.__csv_states.append(CsvDataGrabber(state))

        return self.__csv_states

    # returns unpadded arrays
    def get_formatted_data(self):
        # these lists hold our respective radiance and weather data for this time iteration
        radiance_feature_input = []
        weather_label_output = []

        # get all the radiance channels we're going to be using
        rad_retriever = RadianceDataGrabber(self.__current_date)
        channel_files = []
        for channel in self.__channels:
            rad_file = rad_retriever.find_rad_by_set_time(channel)

            # skip this if the file is missing
            if rad_file == None:
                return None, None

            channel_files.append(rad_file)

        # loops through every STATE in the same TIME
        for s in self.__csv_states:
            # update the time to check in all of the CsvDataGrabbers
            s.update_time((self.__current_date.year, self.__current_date.month, self.__current_date.day, self.__current_date.hour))
            station_entries = s.find_row_by_set_time()

            if len(station_entries) == 0:
                print("Empty Dataset, skipping this set time")
                continue

            state_start_time = time.time()

            # loops through every STATION in the SAME STATE
            for entry in station_entries:
                valid_data = True

                channel_rad_data = []

                # loops through every CHANNEL in the SAME STATION
                for file in channel_files:
                    print("Getting Radiance Channel " + file + " from " + entry["STATION_NAME"] + "...", end="")

                    with Dataset(file) as nc:
                        rad = nc.variables["Rad"][:]
                        dqf = nc.variables["DQF"][:]

                        rad, dqf = gcstools.crop_image(nc, rad, clat=float(entry["LATITUDE"]), clon=float(entry["LONGITUDE"]), dqf=dqf)

                        # make sure that at least 95% of the pixels are good
                        bad_vals_count = np.count_nonzero(dqf > 0)
                        if bad_vals_count > 500:
                            valid_data = False
                            break

                        channel_rad_data.append(rad)

                        print("Done")

                if not valid_data:
                    continue

                try:
                    actual_temp = int(entry["HOURLYDRYBULBTEMPF"])
                    sky_condition = entry["HOURLYSKYCONDITIONS"]
                except:
                    continue  #  another data check, probably not needed


                print("Getting Weather Values from " + entry["STATION_NAME"] + "...", end="")
                # temp label is array of size 201 where -70 degrees F is index 0 and 130 degrees F is index 200, corresponding temp is marked with 1
                temperature_label = np.zeros(201)
                temp_index = actual_temp + 70
                temperature_label[temp_index] = 1

                filtered_sky_conditions = []
                for word in sky_condition.split():
                    if "CLR" in word:
                        filtered_sky_conditions.append("CLR")
                    elif "FEW" in word:
                        filtered_sky_conditions.append("FEW")
                    elif "SCT" in word:
                        filtered_sky_conditions.append("SCT")
                    elif "BKN" in word:
                        filtered_sky_conditions.append("BKN")
                    elif "OVC" in word:
                        filtered_sky_conditions.append("OVC")
                    elif "VV" in word:
                        filtered_sky_conditions.append("VV")

                if len(filtered_sky_conditions) == 0:
                    continue  # data is weird, skip this one

                # sky label is of size 6 where each index corresponds to the cloud conditions below
                sky_condition_label = np.zeros(6)
                sky_condition_to_check = filtered_sky_conditions[-1]

                if sky_condition_to_check == "CLR":         # clear
                    sky_condition_label[0] = 1
                elif sky_condition_to_check == "FEW":       # few clouds
                    sky_condition_label[1] = 1
                elif sky_condition_to_check == "SCT":       # scattered clouds
                    sky_condition_label[2] = 1
                elif sky_condition_to_check == "BKN":       # broken clouds
                    sky_condition_label[3] = 1
                elif sky_condition_to_check == "OVC":       # overcast
                    sky_condition_label[4] = 1
                elif sky_condition_to_check == "VV":        # obscured sky
                    sky_condition_label[5] = 1

                # weather condition label is of size 4 where each index corresponds to the weather condition below
                # note: more than 1 condition can be present
                weather_condition_label = np.zeros(4)
                weather_conditions = entry["HOURLYPRSENTWEATHERTYPE"]
                if "DZ" in weather_conditions or "RA" in weather_conditions or "SH" in weather_conditions:  # rain
                    weather_condition_label[0] = 1
                if "SN" in weather_conditions:  # snow
                    weather_condition_label[1] = 1
                if "BR" in weather_conditions or "FG" in weather_conditions or "HZ" in weather_conditions:  # fog / mist
                    weather_condition_label[2] = 1
                if "TS" in weather_conditions:  # thunderstorms
                    weather_condition_label[3] = 1

                # this list represents all the weather labels for just ONE station
                total_weather_labels = [temperature_label, sky_condition_label, weather_condition_label]

                radiance_feature_input.append(channel_rad_data)
                weather_label_output.append(total_weather_labels)

                print("Done")

            state_end_time = time.time()

            print("Time Elapsed for Data Grabbing:", state_end_time - state_start_time)
            print("-----------------------------")

        return radiance_feature_input, weather_label_output


    def increment_date(self):
        self.__current_date = self.__current_date + timedelta(hours=1)

    def get_all_states(self):
        return self.__csv_states

    def get_current_date(self):
        return self.__current_date

    def pickle_date(self):
        pickle.dump(self.__current_date, open(DATE_PICKLE_NAME, "wb"))

    # for debug purposes
    def print_all_states(self):
        print([f for f in listdir("WeatherData/") if isfile(join("WeatherData/", f))])



if __name__ == "__main__":
    if PICKLE_MODE:
        print("Using Pickled Date")
        data_datetime = pickle.load(open(DATE_PICKLE_NAME, "rb"))
        data_date = (data_datetime.year, data_datetime.month, data_datetime.day, data_datetime.hour)
    else:
        data_date = (2017, 8, 8, 1)
        print("Using explicitly set date:", data_date)

    data_retriever = DataManager(starting_date=data_date, channels=["C13", "C14", "C15", "C16"])

    print(data_retriever.print_all_states())

    while data_date[0] is not 2018:
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

# ["C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16"]