from datetime import datetime
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib, logging
import logging
import numpy as np
from pyproj import Proj
import pyresample as pr
import os.path
import logging
import google.cloud.storage as gcs

GOES_PUBLIC_BUCKET = "gcp-public-data-goes-16"


def list_gcs(bucket, gcs_prefix, gcs_patterns):
    bucket = gcs.Client().get_bucket(bucket)
    blobs = bucket.list_blobs(prefix=gcs_prefix, delimiter='/')
    result = []
    if gcs_patterns == None or len(gcs_patterns) == 0:
        for b in blobs:
            result.append(b)
    else:
        for b in blobs:
            match = True
            for pattern in gcs_patterns:
                if not pattern in b.path:
                    match = False
            if match:
                result.append(b)
    return result


def get_objectId_at(dt, product='ABI-L1b-RadF', channel='C14'):
    # get first 11-micron band (C14) at this hour
    # See: https://www.goes-r.gov/education/ABI-bands-quick-info.html
    logging.info('Looking for data collected on {}'.format(dt))
    dayno = dt.timetuple().tm_yday
    gcs_prefix = '{}/{}/{:03d}/{:02d}/'.format(product, dt.year, dayno, dt.hour)
    gcs_patterns = [channel,
                    's{}{:03d}{:02d}'.format(dt.year, dayno, dt.hour)]
    blobs = list_gcs(GOES_PUBLIC_BUCKET, gcs_prefix, gcs_patterns)
    if len(blobs) > 0:
        objectId = blobs[0].path.replace('%2F', '/').replace('/b/{}/o/'.format(GOES_PUBLIC_BUCKET), '')
        logging.info('Found {} for {}'.format(objectId, dt))
        return objectId
    else:
        logging.error('No matching files found for gs://{}/{}* containing {}'.format(GOES_PUBLIC_BUCKET, gcs_prefix,
                                                                                     gcs_patterns))
        return None


def copy_fromgcs(bucket, objectId, name):
    bucket = gcs.Client().get_bucket(bucket)
    blob = bucket.blob(objectId)
    basename = os.path.basename(objectId)
    logging.info('Downloading {}'.format(basename))
    blob.download_to_filename(name)
    return name


def crop_image(nc, data, clat, clon, dqf=None):
   # output grid centered on clat, clon in equal-lat-lon
   lats = np.arange(clat-0.5,clat+0.5,0.01) # approx 1km resolution, 2000km extent     0.5 for good zoom in
   lons = np.arange(clon-0.5,clon+0.5,0.01) # approx 1km resolution, 2000km extent

   lons, lats = np.meshgrid(lons, lats)
   new_grid = pr.geometry.GridDefinition(lons=lons, lats=lats)

   # Subsatellite_Longitude is where the GEO satellite is
   lon_0 = nc.variables['nominal_satellite_subpoint_lon'][0]
   ht_0 = nc.variables['nominal_satellite_height'][0] * 1000 # meters
   x = nc.variables['x'][:] * ht_0 #/ 1000.0
   y = nc.variables['y'][:] * ht_0 #/ 1000.0

   nx = len(x)
   ny = len(y)
   max_x = x.max(); min_x = x.min(); max_y = y.max(); min_y = y.min()
   half_x = (max_x - min_x) / nx / 2.
   half_y = (max_y - min_y) / ny / 2.
   extents = (min_x - half_x, min_y - half_y, max_x + half_x, max_y + half_y)
   old_grid = pr.geometry.AreaDefinition('geos','goes_conus','geos',
       {'proj':'geos', 'h':str(ht_0), 'lon_0':str(lon_0) ,'a':'6378169.0', 'b':'6356584.0'},
       nx, ny, extents)

   # now do remapping
   logging.info('Remapping from {}'.format(old_grid))

   if dqf is not None:
       return pr.kd_tree.resample_nearest(old_grid, data, new_grid, radius_of_influence=50000), pr.kd_tree.resample_nearest(old_grid, dqf, new_grid, radius_of_influence=50000, nprocs=4)
   else:
       return pr.kd_tree.resample_nearest(old_grid, data, new_grid, radius_of_influence=50000, nprocs=4)



def plot_image(ncfilename, outfile, clat, clon):
    matplotlib.use('Agg')  # headless display

    with Dataset(ncfilename, 'r') as nc:
        rad = nc.variables['Rad'][:]
        # See http://www.goes-r.gov/products/ATBDs/baseline/Imagery_v2.0_no_color.pdf
        # ref = (rad * np.pi * 0.3) / 663.274497
        # ref = np.minimum(np.maximum(ref, 0.0), 1.0)

        ref = rad
        # crop to area of interest
        ref = crop_image(nc, ref, clat, clon)

        ref = np.flipud(ref)

        # do gamma correction to stretch the values
        # ref = np.sqrt(ref)

        # plotting to jpg file
        fig = plt.figure()
        plt.imsave(outfile, ref, cmap='gist_ncar_r')  # or 'Greys_r' without color / gist_ncar_r
        plt.close('all')
        logging.info('Created {}'.format(outfile))
        return outfile
    return None


if __name__ == "__main__":

    date = datetime(2018, 4, 21, 19)

    # sat_file = get_objectId_at(date, product="ABI-L1b-RadC", channel="C12") # gets it at

    # print(sat_file)

    # copy_fromgcs(GOES_PUBLIC_BUCKET, sat_file, "TestSatFiles/RadC-C12-4-21-19-BOFLY.nc")

    # plot_image("SatFiles/SatFile-C01", "testing_bad_data.png", 27.9506, -82.4572) # 42.3601 -71.0589 for boston

    # copy_fromgcs(GOES_PUBLIC_BUCKET, get_objectId_at(datetime.today(), product="ABI-L2-CMIPC", channel=""), "TestSatFiles")

# TODO: find out what channel to use for clouds
# TODO: it starts at the beginning of the hour
# TODO: "{:02d}".format(1)
