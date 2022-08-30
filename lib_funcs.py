import os
import boto3
import json
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import satsearch
import rasterio as rio
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from pprint import pprint
from pyproj import Transformer #  coordinates-to-pixels transformation
from rasterio.features import bounds
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_geometry(json_file):
    '''to get the geometry of the lake from json file'''
    try:
        geojson = json.load(open(json_file))
        geometry = geojson["features"][0]["geometry"]
        
        return geometry
    except:
        print("json file not found")

def get_sentinel_search_results(geometry):
    ''' get search results for the waterbody from the repository'''
    try:
        # The data repositpory has images from 2017 January
        today = datetime.datetime.today().date()
        timeRange = '2017-01-01/' + str(today)
        print("Results between 2017 Jan 1 and today (",today.strftime("%Y %b %d"),") found")
        
        # server for searching in the satellite
        SentinelSearch = satsearch.Search.search( 
            url = "https://earth-search.aws.element84.com/v0",
            intersects = geometry,
            datetime = timeRange,
            collections = ['sentinel-s2-l2a-cogs'])

        Sentinel_items = SentinelSearch.items()
        print("Found " , len(Sentinel_items) , "items")
        return Sentinel_items
    except:
        print("Search in sentinel returned no results")
        return None


def getSubset(aws_session, geotiff_file, bbox, date, kind='band', save=False):
    ''' to get the satellite data from the repository for the specific region  '''
    try:
        with rio.Env(aws_session):
            with rio.open(geotiff_file) as geo_fp:

                Transf = Transformer.from_crs("epsg:4326", geo_fp.crs) 
                lat_north, lon_west = Transf.transform(bbox[3], bbox[0])
                lat_south, lon_east = Transf.transform(bbox[1], bbox[2]) 
                x_top, y_top = geo_fp.index( lat_north, lon_west )
                x_bottom, y_bottom = geo_fp.index( lat_south, lon_east )

                # Define window in RasterIO
                window = rio.windows.Window.from_slices( ( x_top, x_bottom ), ( y_top, y_bottom ) )                
                # Actual HTTP range request

                if kind=='visual':
                    # only for the visual data as it has 3 channels
                    if save:
                        # loop through the raster's bands to fill the empty array
                        img = np.stack([geo_fp.read(i, window=window) for i in range(1,4)],
                                       axis=-1)
                        im = Image.fromarray(img)
                        im.save('temp/lake' + '_' + date + '_visual.png')
                        return img
                else:
                    # all the other bands
                    subset = geo_fp.read(1, window=window)
                    if save:
                        im = Image.fromarray(img)
                        im.save('temp/lake' + '_' + date + '_' + kind + '.png')
                    return subset
    except:
        print("Data not found for ", date)
        ar = np.zeros(2)
        return ar


def get_stats(item, date, np_scl):
    ''' to get the amount of clouds in each image based on scene classification data'''
    try:
        area = np_scl.shape[0]*np_scl.shape[1]
        image = np.zeros((np_scl.shape), dtype= np.uint8)

        water = np_scl == 6
        high_clouds = np_scl == 8
        low_clouds = np_scl == 9
        cloud_shadows = np_scl == 3
        cirrus = np_scl == 10
        snow = np_scl == 11
        stats ={'entry':item,
                'date':date,
                'water':np.sum(water*100)/area,
                'high_clouds':np.sum(high_clouds*100)/area,
                'low_clouds':np.sum(low_clouds*100)/area,
                'cloud_shadows':np.sum(cloud_shadows*100)/area,
                'cirrus':np.sum(cirrus*100)/area,
                'cloudy':np.sum(cirrus + high_clouds + low_clouds)*100/area}
        # print(stats)
        return stats
    except:
        print("Scene classification data for ", date, ": ", item, " not found")

def get_ndmi(vnir, swir22, date, binspace=0.2):
    ''' To get the ndmi data given particular Band8A and Band11 information'''
    filename = 'temp/lake_' + date + '_ndmi.png'
    ndmi = (vnir.astype(float)-swir22.astype(float))/(vnir.astype(float)+swir22.astype(float))
    l = ndmi.shape[0]
    w = ndmi.shape[1]
    wd = w/l
    plt.figure(figsize=(8*wd,8))

    Nbins = int(2/binspace)
    plt.imshow(ndmi)
    plt.axis('off')
    # ax.set_aspect('auto')
    # plt.gca().set_aspect('auto')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    # bins = np.zeros(Nbins)
    # binkeys = []
    df_entry = {'date':date}

    for i in range(Nbins):
        value1 = -1.0 + i*binspace
        value2 = -1.0 + (i+1.0)*binspace

        binval = np.sum((ndmi>value1) & (ndmi<value2))
        # bins[i] = binval
        binkey = 'bin' + str(i+1)
        # binkeys.append(binkey)
        df_entry[binkey] = binval
    
    ndmi_pos = ndmi[ndmi > 0]
    df_entry['sum'] = ndmi.sum()
    df_entry['positive_sum'] = ndmi_pos.sum()

    return df_entry


def get_results(aws_session, json_file):
    '''the main function to do the calculations and save the results to csv file'''
    geometry = get_geometry(json_file)
    Sentinel_items = get_sentinel_search_results(geometry)
    
    if Sentinel_items is None:
        print("No data found for the location in the repository")
        return

    temp_exists = os.path.exists("temp/")
    if not temp_exists:
        os.makedirs("temp")

    bbox = bounds(geometry)

    lake_stats = pd.DataFrame()
    ndmi_stats = pd.DataFrame()
    i = 0
    print(Sentinel_items)
    for i, item in tqdm(enumerate(Sentinel_items) , total=len(Sentinel_items)):
        # if i == 0:
        #     print(item.assets)
        time1 = datetime.datetime.now()
        scl = item.assets['SCL']['href']
    #     tci = item.assets['visual']['href']
        vnir = item.assets['B8A']['href']
        swir22 = item.assets['B11']['href']
        date = item.properties['datetime'][0:10]
        time2 = datetime.datetime.now()

        # print("Sentinel item number " + str(i+1) + "/" + str(len(Sentinel_items)) +  " // :"  + date)
        time3 = datetime.datetime.now()
        scl = getSubset(aws_session, scl, bbox, date, save=False)
    #     tci = getSubset(aws_session, tci, bbox, date, kind='visual', save=True)
        vnir = getSubset(aws_session, vnir, bbox, date, save=False)
        swir22 = getSubset(aws_session, swir22, bbox, date, save=False)
        time4 = datetime.datetime.now()

        if len(scl.shape) != 1:
            stats = get_stats(item, date, scl)
            cloudiness = stats['cloudy']
            stats = pd.DataFrame([stats])
            lake_stats = pd.concat([lake_stats, stats])
            if cloudiness < 20:
                ndmi = get_ndmi(vnir, swir22, date, binspace=0.2)
                ndmi = pd.DataFrame([ndmi])
                ndmi_stats = pd.concat([ndmi_stats, ndmi])
        time5 = datetime.datetime.now()
        i =+ 1
#         print("Fetch time: ", time4-time3, " ; analysis time: ", time5-time4)
    ndmi_stats.to_csv('lake_ndmi_stats.csv', index=False)
