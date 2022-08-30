from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from pyproj import Transformer #  coordinates-to-pixels transformation
from rasterio.features import bounds
import rasterio as rio


import satsearch
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from lib_funcs import get_geometry
import datetime
import warnings
warnings.filterwarnings("ignore")

months_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = range(2017,datetime.date.today().year+1)

def get_sublist(year, toord):
    '''to convert the dates to the day of the year'''
    new = toord - datetime.datetime(year,1,1).toordinal()+1
    return new


def clean_data(df):
    '''cleans the data frame and adds additional columns required for processing'''
    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['toordinal'] = df['date'].map(datetime.datetime.toordinal)
    df = df.sort_values('date',ascending=True)
    df['monthly'] = df['date'].dt.strftime('%m-%d')
    df['yearly'] = df['date'].dt.strftime('%Y')
    df['yearly'] = pd.to_numeric(df['yearly'])
    df['absolute_dates'] = df.apply(lambda x: get_sublist(x.yearly, x.toordinal), axis=1)
    
    months = []
    print("Number of data points per month over the years")
    summer = 0
    monsoon = 0
    
    for m in range(1,13):
        month = df[df['date'].dt.month == m]
        if m < 6 :
            summer += len(month)
        else:
            monsoon += len(month)
        months.append(month)
        print(m, " Month ", months_list[m-1], " :", len(month))
    if summer > monsoon:
        print("It can be seen that in the second half of the year there are less data points comparatively due to monsoon clouds")
    else :
        print("This is unusual!! Usually summer has more data points than monsoon due to the absence of clouds.")
    
    return df

    
def get_summer_trends(df):
    '''gets the trends of the summer every year'''
    plt.figure(figsize=(20,10))
    matplotlib.rcParams.update({'font.size': 32})
    
    # selects the first 150 days of the year
    cond = df['absolute_dates'] < 150
    df_cond = df[cond]
    colors = ['#747FE3', '#8EE35D', '#E37346']
    palette= sns.color_palette('dark', n_colors=len(years))
    sns.lineplot(x='absolute_dates', y='sum', data=df_cond, palette=palette, hue='yearly')

    plt.xlim([0, 150])
    plt.xlabel('Day of the year')
    plt.ylabel('Sum of NDMI over the region')
    plt.legend(title='Year')
    plt.tight_layout()
    
    
def get_inference(aws_session, geotiff_file, bbox, date, title=''):
    '''given a date, corresponding visual tiff file reference and title
    shows the correspoding figure'''
    matplotlib.rcParams.update({'font.size': 14})
    try:
        with rio.Env(aws_session):
            with rio.open(geotiff_file) as geo_fp:
                #gets the coordinates from pixel values
                Transf = Transformer.from_crs("epsg:4326", geo_fp.crs) 
                lat_north, lon_west = Transf.transform(bbox[3], bbox[0])
                lat_south, lon_east = Transf.transform(bbox[1], bbox[2]) 
                x_top, y_top = geo_fp.index( lat_north, lon_west )
                x_bottom, y_bottom = geo_fp.index( lat_south, lon_east )
                # Define window in RasterIO
                window = rio.windows.Window.from_slices( ( x_top, x_bottom ), ( y_top, y_bottom ) )                
                # joins RGB channels
                img = np.stack([geo_fp.read(i, window=window) for i in range(1,4)],
                                   axis=-1)
                plt.figure(figsize=(10,10))
                plt.imshow(img)
                plt.title(title + " Appearance on " + str(date))
    except:
        print('The file could not be found in the repository')


def get_visual(aws_session, date, title, geometry):
    '''searches for the data on the particular date'''
    timeRange = str(date)
    SentinelSearch = satsearch.Search.search( 
        url = "https://earth-search.aws.element84.com/v0",
        intersects = geometry,
        datetime = timeRange,
        collections = ['sentinel-s2-l2a-cogs'])
    
    Sentinel_items = SentinelSearch.items()
    print(len(Sentinel_items), " item(s) found")
    bbox = bounds(geometry)
    for item in Sentinel_items:
        # gets the visual spectrum band from the satellite data
        visual = item.assets['visual']['href']
        get_inference(aws_session, visual, bbox, date, title)


def get_similar(aws_session, pred, df, json_file):
    '''given a particular predicted value, gets the closest example'''
    loc = np.argmin(abs(df['sum'] - pred))
    date = df['date'].iloc[loc].date()
    geometry = get_geometry(json_file)
    title = 'Prediction value : ' + str(pred)
    get_visual(aws_session, date, title, geometry)


def get_minmax(aws_session, df, geojson):
    '''gets the instances of max and min values for NDMI'''
    try:
        minloc = np.argmin(df['sum'])
        maxloc = np.argmax(df['sum'])
        mindate = df['date'].iloc[minloc].date()
        maxdate = df['date'].iloc[maxloc].date()
        # print(mindate, maxdate)
        # print(minloc, maxloc)
        geometry = get_geometry(geojson)

        labels = ['Minimum NDMI value: the driest date', 'Maximum NDMI value: the wettest date'] 
        for i, date in enumerate([mindate, maxdate]):
            get_visual(aws_session, date, labels[i], geometry)
    except:
        print('Error fetching the file')
        

def do_yearly_fitting(df, deg=1):
    '''show the annual trends for the previous years AND
    do fitting for linear regression'''
    matplotlib.rcParams.update({'font.size': 12})
    n = len(years)
    fig, ax = plt.subplots(n-1, 1, figsize=(6,20))

    for i, year in enumerate(years):
        print("Year ", year, "-", year+1)
        
        # converts dates into continuous ordinal numbers 
        start = datetime.datetime(year, 10, 1).toordinal()
        end = datetime.datetime(year+1, 6, 1).toordinal()
        data = df.loc[(df['toordinal'] > start) & (df['toordinal'] < end)]
        
        # if there are 10 data points
        if len(data) > 10:
            # split into inpiut and output elements
            X, y = data['toordinal'], data['sum']
            poly = PolynomialFeatures(degree=deg)
            if deg == 1:
                X = np.array(X).reshape(-1,1)
            else:
                X = poly.fit_transform(X)

            # split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

            # identify outliers in the training dataset
            lof = LocalOutlierFactor(contamination=0.1)
            yhat = lof.fit_predict(X_train)

            # select all rows that are not outliers
            mask = yhat != -1
            X_train, y_train = X_train[mask, :], y_train[mask]

            # fit the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            # # evaluate the model
            # X_test = np.array(X_test).reshape(-1, 1)
            yhat = model.predict(X_test)

            ax[i].plot_date(data['toordinal'], data['sum'])
            ax[i].set_xlim([start, end])
            ax[i].set_title('Year ' + str(year) + '-' + str(year+1))
            ax[i].set_xlabel('date')
            # converts the axis ticks from ordinal values to date format
            new_labels = [datetime.date.fromordinal(int(item)) for item in ax[i].get_xticks()]
            ax[i].set_xticklabels(new_labels)

            for label in ax[i].get_xticklabels():
                label.set_rotation(75)

            ax[i].plot(X, model.predict(X),color='k')
            # # evaluate predictions
            mae = mean_absolute_error(y_test, yhat)
            # print('MAE: %.3f' % mae)
            print("Slope: ",model.coef_, " ; Intercept: ", model.intercept_)

            plt.tight_layout()
        else:
            print("Not enough data points")
            

def do_prediction(aws_session, df, date, deg=1):
    today = datetime.datetime.today()
#     if date < today:
#         print("Give a future date")
#         return
        
#     if date.month > 5:
#         print("Please give a date before June 1st")
#         return
    
    matplotlib.rcParams.update({'font.size': 12})
    n = len(years)
    fig, ax = plt.subplots(1, 1, figsize=(6,5))

    date = today
    year = date.year
    start = datetime.datetime(date.year - 1, 10, 1).toordinal()
    end = date.toordinal()
    print("Year ", year, "-", year+1)

    data = df.loc[(df['toordinal'] > start) & (df['toordinal'] < end)]

    if len(data) > 10:

        # split into inpiut and output elements
        X, y = data['toordinal'], data['sum']
        poly = PolynomialFeatures(degree=deg)
        if deg == 1:
            X = np.array(X).reshape(-1,1)
        else:
            X = poly.fit_transform(X)

        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        # identify outliers in the training dataset
        lof = LocalOutlierFactor(contamination=0.1)
        yhat = lof.fit_predict(X_train)

        # select all rows that are not outliers
        mask = yhat != -1
        X_train, y_train = X_train[mask, :], y_train[mask]

        # fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # # evaluate the model
        # X_test = np.array(X_test).reshape(-1, 1)
        yhat = model.predict(X_test)

        ax.plot_date(data['toordinal'], data['sum'])
        ax.set_xlim([start, end])
        ax.set_title('Year ' + str(year) + '-' + str(year+1))
        ax.set_xlabel('date')
        new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]
        ax.set_xticklabels(new_labels)

        for label in ax.get_xticklabels():
            label.set_rotation(75)

        ax.plot(X, model.predict(X),color='k')
        date_toordinal = date.toordinal()
        prediction = model.predict(date_toordinal)
        ax.scatter(date_toordinal, prediction)
        plt.text(date_toordinal,prediction,"Prediction")

        plt.tight_layout()
        
        get_similar(aws_session, predction, df, geojson)
    else:
        print("Not enough data points")