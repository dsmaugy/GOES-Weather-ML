import csv
from os import listdir
from os.path import isfile, join
from datetime import timedelta, datetime
import time
import timezonefinder
import pytz
import dateutil.parser
import numpy as np
import gcstools
import matplotlib.pyplot as plt

NO_DOWNLOAD_MODE = True

class CsvDataGrabber:

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
                row_date = row_date + timedelta(seconds=row_date.dst().seconds)

            year = row_date.utctimetuple().tm_year
            month = row_date.utctimetuple().tm_mon
            day = row_date.utctimetuple().tm_mday
            hour = row_date.utctimetuple().tm_hour
            minute = row_date.utctimetuple().tm_min

            if year == input_year:
                if month == input_month:
                    if day == input_day:
                        if hour == input_hour and minute == 0:

                            if len(row["HOURLYSKYCONDITIONS"]) > 0 and len(row["HOURLYDRYBULBTEMPF"]) > 0:
                                yield row
                        elif hour + 1 == input_hour:
                            if minute > 50:

                                if len(row["HOURLYSKYCONDITIONS"]) > 0 and len(row["HOURLYDRYBULBTEMPF"]) > 0:
                                    yield row

                        elif hour >= input_hour:
                            # print("-------------------------------------------")
                            # print("- No more data found for given time stamp -")
                            # print("-------------------------------------------")
                            # print()
                            break



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

            # get rid of the duplicates
            for pop_index in sorted(indexes_to_pop, reverse=True):
                # if pop_index < len(matching_rows): # bruh
                matching_rows.pop(pop_index)

        for sanitized_row in matching_rows:
            print(sanitized_row["STATION_NAME"], sanitized_row["DATE"])

        print("-----------------------------")

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
            ncs_file_id = gcstools.get_objectId_at(self.time, product="ABI-L1b-RadC", channel=channel)
            rad_file = gcstools.copy_fromgcs(gcstools.GOES_PUBLIC_BUCKET, ncs_file_id, "SatFiles/SatFile-" + channel)
            print("Downloaded", rad_file)

            return rad_file
        else:
            rad_file = "SatFiles/SatFile-" + channel
            print("Found", rad_file)
            return rad_file


class DataManager:

    def __init__(self, starting_date=(2017, 8, 1, 1), channels=("C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16")):
        self.__csv_states = []
        self.__current_date = datetime(year=starting_date[0], month=starting_date[1], day=starting_date[2], hour=starting_date[3])
        self.__channels = channels

    def load_all_states(self):
        files = [f for f in listdir("WeatherData/") if isfile(join("WeatherData/", f))]

        for state in files:
            self.__csv_states.append(CsvDataGrabber(state))

        return self.__csv_states

    def get_formatted_data(self):
        from netCDF4 import Dataset
        import gcstools

        rad_retriever = RadianceDataGrabber(self.__current_date)
        channel_files = []

        # get all the channels we're going to be using
        for channel in self.__channels:
            rad_file = rad_retriever.find_rad_by_set_time(channel)
            channel_files.append(rad_file)

        # loops through every STATE in the same TIME
        for s in self.__csv_states:
            # update the time to check in all of the CsvDataGrabbers
            s.update_time((self.__current_date.year, self.__current_date.month, self.__current_date.day, self.__current_date.hour))
            station_entries = s.find_row_by_set_time()

            # loops through every STATION in the SAME STATE
            for entry in station_entries:
                valid_data = True

                station_rad_data = []
                station_weather_data = []

                # loops through every CHANNEL in the SAME STATION
                for file in channel_files:

                    channel_rad_data = []
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

                if not valid_data:
                    continue


                # temp label is array of size 201 where -70 degrees F is index 0 and 130 degrees F is index 200, corresponding temp is marked with 1
                temperature_label = np.zeros(201)
                actual_temp = int(entry["HOURLYDRYBULBTEMPF"])
                temp_index = actual_temp + 70
                temperature_label[temp_index] = 1

                sky_condition = entry["HOURLYSKYCONDITIONS"]



    def increment_date(self):
        self.__current_date = self.__current_date + timedelta(hours=1)

    def get_all_states(self):
        return self.__csv_states

if __name__ == "__main__":
    data_retriever = DataManager(starting_date=(2017, 8, 4, 5), channels=["C07"])

    data_retriever.load_all_states()

    data_retriever.get_formatted_data()

# ("C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16")


# TODO Use drybulb temp
