import csv
from os import listdir
from os.path import isfile, join
from datetime import timedelta, datetime
import time
import timezonefinder
import pytz
import dateutil.parser


class CsvDataGrabber:

    def __init__(self, state, starting_date=(2017, 7, 10, 0)):
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

            one_hour_ahead_check = row_date + timedelta(hours=1)

            if year == input_year:
                if month == input_month:
                    if day == input_day:
                        if hour == input_hour and minute == 0:
                            yield row
                        elif one_hour_ahead_check.utctimetuple().tm_hour == input_hour:
                            if minute > 50:
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

    def find_rad_by_set_time(self, channel):
        import gcstools
        current_datetime = datetime(year=self.__current_date[0], month=self.__current_date[1], day=self.__current_date[2],
                                    hour=self.__current_date[3])

        ncs_file_id = gcstools.get_objectId_at(current_datetime, product="ABI-L1b-RadC", channel=channel)
        gcstools.copy_fromgcs(gcstools.GOES_PUBLIC_BUCKET, ncs_file_id, "SatFile-" + channel)


    # adds 1 hour to the date
    def increment_date(self):
        current_datetime = datetime(year=self.__current_date[0], month=self.__current_date[1], day=self.__current_date[2],
                                    hour=self.__current_date[3])
        current_datetime = current_datetime + timedelta(hours=1)

        self.__current_date = (current_datetime.year, current_datetime.month, current_datetime.day, current_datetime.hour)

    @staticmethod
    def get_all_states():
        files = [f for f in listdir("WeatherData/") if isfile(join("WeatherData/", f))]
        csv_states = []

        for state in files:
            csv_states.append(CsvDataGrabber(state))

        return csv_states


states = CsvDataGrabber.get_all_states()

for s in states:
    s.find_row_by_set_time()
    s.increment_date()

for s in states:
    s.find_row_by_set_time()
    s.increment_date()

for s in states:
    s.find_row_by_set_time()
    s.increment_date()

# TODO Use drybulb temp
