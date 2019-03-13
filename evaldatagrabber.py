from datagrabber import RadianceDataGrabber
import numpy as np
import gcstools
from datetime import datetime
from netCDF4 import Dataset

loop = True

rad_labels = []
weather_labels = []

while loop:
    channels = ["C13", "C14", "C15", "C16"]
    year, month, day, hour = [int(x) for x in input("year, month, day, hour (in UTC): ").split()]
    latitude, longitude = [float(x) for x in input("lat, lon: ").split()]

    date = datetime(year=year, month=month, day=day, hour=hour)
    rad_grabber = RadianceDataGrabber(date)

    rad_files = []
    for channel in channels:
        rad_files.append(rad_grabber.find_rad_by_set_time(channel))

    channel_rad_data = []
    for file in rad_files:
        with Dataset(file) as nc:
            rad = nc.variables["Rad"][:]
            rad = gcstools.crop_image(nc, rad, clat=latitude, clon=longitude)

            channel_rad_data.append(rad)

    temp = int(input("temp: "))
    cloud = input("cloud: ")
    weather_conditions = input("conditions: ").split()

    print(weather_conditions)

    # temperature stuff
    temperature_label = np.zeros(201)
    temp_index = temp + 70
    temperature_label[temp_index] = 1

    # cloud stuff
    sky_condition_label = np.zeros(6)

    if cloud == "CLR":  # clear
        sky_condition_label[0] = 1
    elif cloud == "FEW":  # few clouds
        sky_condition_label[1] = 1
    elif cloud == "SCT":  # scattered clouds
        sky_condition_label[2] = 1
    elif cloud == "BKN":  # broken clouds
        sky_condition_label[3] = 1
    elif cloud == "OVC":  # overcast
        sky_condition_label[4] = 1
    elif cloud == "VV":  # obscured sky
        sky_condition_label[5] = 1

    # weather conditon stuff
    weather_condition_label = np.zeros(4)
    if "DZ" in weather_conditions or "RA" in weather_conditions or "SH" in weather_conditions:  # rain
        weather_condition_label[0] = 1
    if "SN" in weather_conditions:  # snow
        weather_condition_label[1] = 1
    if "BR" in weather_conditions or "FG" in weather_conditions or "HZ" in weather_conditions:  # fog / mist
        weather_condition_label[2] = 1
    if "TS" in weather_conditions:  # thunderstorms
        weather_condition_label[3] = 1

    ask_to_loop = input("continue? (y/n):")

    total_weather_labels = [temperature_label, sky_condition_label, weather_condition_label]
    rad_labels.append(channel_rad_data)
    weather_labels.append(total_weather_labels)

    if ask_to_loop == "n":
        loop = False


rad_np_array = np.array(rad_labels)
weather_np_array = np.array(weather_labels)

np.save("rad-eval.npy", rad_np_array)
np.save("weather-eval.npy", weather_np_array)

print("Saved eval data")

