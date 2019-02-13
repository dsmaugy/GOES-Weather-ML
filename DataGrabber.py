import csv
from os import listdir
from os.path import isfile, join
from datetime import datetime
import time
import timezonefinder
import pytz
import dateutil.parser


# with open("WeatherData/newyork.csv") as csv_file:
#     csv_reader = csv.DictReader(csv_file, delimiter=",")

class CsvDataGrabber:

    def __init__(self):
        self.CSV_PATH = "WeatherData/"
        self.csv_file = None # placeholder value, probably very bad practice

    def __read_csv(self, csv_reader, input_month, input_day, input_hour):
        i = 0
        start_time = time.time()
        tf = timezonefinder.TimezoneFinder()

        for row in csv_reader:
            lat = float(row["LATITUDE"])
            lng = float(row["LONGITUDE"])

            timezone_str = tf.timezone_at(lat=lat, lng=lng)
            timezone = pytz.timezone(timezone_str)

            row_date = dateutil.parser.parse(row["DATE"])
            # row_date = datetime.strptime(row["DATE"], "%Y-%m-%d %H:%M")

            row_date = timezone.localize(row_date)

            hour = row_date.utctimetuple().tm_hour
            minute = row_date.utctimetuple().tm_min
            month = row_date.utctimetuple().tm_mon
            day = row_date.utctimetuple().tm_mday

            if month == input_month:
                if day == input_day:
                    if hour == input_hour and minute == 0:
                        yield row
                    elif hour + 1 == input_hour:
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

    def close_file(self):
        self.csv_file.close()

    def cycle_states(self, input_month, input_day, input_hour):
        files = [f for f in listdir(self.CSV_PATH) if isfile(join(self.CSV_PATH, f))]
        all_rows = []

        for state in files:
            # this way the generator expression is saved
            file_name = "WeatherData/" + state
            self.csv_file = open(file_name)
            csv_reader = csv.DictReader(self.csv_file, delimiter=",")

            print()
            print("-----------------------------")
            print("Opened", state)

            # find the rows for each date in this specified state
            matching_rows = []
            for row in self.__read_csv(csv_reader, input_month, input_day, input_hour):
                matching_rows.append(row)

            # remove any duplicate data from the same station within the 10 minute grace period
            if len(matching_rows) > 5:
                indexes_to_pop = [] # stores the list of indexes that we can remove

                for i in range(0, len(matching_rows)-1):
                    name_to_check = matching_rows[i]["STATION"]

                    # loop over the sublist from name_to_check to see if any dupes are found
                    for j in range(i+1, len(matching_rows)):
                        if matching_rows[j]["STATION"] == name_to_check: # this means that the station has recorded another data entry within the 10 minute grace period
                            date_to_check = dateutil.parser.parse(matching_rows[i]["DATE"])
                            date_examined = dateutil.parser.parse(matching_rows[j]["DATE"])

                            if date_to_check.time() < date_examined.time(): # examined time is closer to the hour, get rid of station[i]
                                indexes_to_pop.append(i)
                            else:
                                indexes_to_pop.append(j)

                # get rid of the duplicates
                for pop_index in sorted(indexes_to_pop, reverse=True):
                    matching_rows.pop(pop_index)

            for sanitized_row in matching_rows:
                print(sanitized_row["STATION_NAME"], row["DATE"])

            print("-----------------------------")

            all_rows = all_rows + matching_rows

        return all_rows



data_grabber = CsvDataGrabber()

# date_to_find = (1, 1, 7)
# data = data_grabber.cycle_states(*date_to_find)

date_to_find = (1, 2, 21)
data = data_grabber.cycle_states(*date_to_find)

print("DONE DONE DONE DONE DONE")

date_to_find = (1, 2, 22)
data = data_grabber.cycle_states(*date_to_find)


# for entry in data:
#     print("Name:", entry["STATION_NAME"])
#     print("Dry Bulb Temp:", entry["HOURLYDRYBULBTEMPF"])
#     print("Weather Conditions:", entry["HOURLYPRSENTWEATHERTYPE"])
#     print()


# TODO Use drybulb temp
# TODO Get the iterator save working by passing the state to each DataGrabber object
