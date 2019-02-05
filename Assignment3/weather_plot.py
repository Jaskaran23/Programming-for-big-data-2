import sys
#assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
import elevation_grid as eg
from datetime import datetime

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('colour prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3' # make sure we have Spark 2.3+



#from colour_tools import colour_schema, rgb2lab_query, plot_predictions


def main(inputs,test_input,model_file):
    tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
    ])

    data = spark.read.csv(inputs, schema=tmax_schema)
   
    data = data.withColumn('year',functions.year(data["date"]))
    


    data3 = data.filter((data['year'] < 2000))
    data31 = data3.groupBy('station').agg(functions.avg('tmax').alias('avgtmax'))
    data32 = data31.select(data31.station.alias('station1'), data31.avgtmax)
    data33 = data32.join(data3, data32.station1 == data3.station)
    quater31 = data33.select('station', 'date', 'latitude', 'longitude', 'elevation', 'avgtmax', 'year')
    # quater11.show()
    quater3 = quater31.dropDuplicates(['station'])  # this is done

    data4 = data.filter((data['year'] >= 2000) )
    data5 = data4.groupBy('station').agg(functions.avg('tmax').alias('avgtmax'))
    print(data5.count())
    data6 = data5.select(data5.station.alias('station1'),data5.avgtmax)
    data7 = data6.join(data4, data6.station1 == data4.station)
    #data7.show()
    quater41 = data7.select('station','date','latitude','longitude','elevation','avgtmax','year')
    quater4 = quater41.dropDuplicates(['station']) #this is done

    
    quater2df = quater3.toPandas()
    quater3df = quater4.toPandas()
    
    f1 = plt.figure(figsize = (12,6))
    m1 = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    m1.drawmapboundary(fill_color='lightblue', linewidth=0)
    m1.drawcoastlines()
    m1.drawcountries()

    
    # Draw coastlines, and the edges of the map.

    # Convert latitude and longitude to x and y coordinates
    x, y = m1(list(quater2df["longitude"].astype(float)), list(quater2df["latitude"].astype(float)))
    # Use matplotlib to draw the points onto the map.
    tmax1 = list(quater2df['avgtmax'])

    m1.scatter(x,y,c=(tmax1) ,s=(tmax1), alpha=0.4,cmap='coolwarm')
    
    plt.colorbar(label=r'$( tmax1)$')
    plt.clim(-50,50 )
    f1.suptitle('Maximum temperature distribution before 2000',fontsize=14)
    plt.show()
    

    #for second plot
    f2 = plt.figure(figsize = (12,6))
    m2 = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    m2.drawmapboundary(fill_color='lightblue', linewidth=0)
    m2.drawcoastlines()
    m2.drawcountries()
    x1, y1 = m2(list(quater3df["longitude"].astype(float)), list(quater3df["latitude"].astype(float)))
    # Use matplotlib to draw the points onto the map.
    tmax11 = list(quater3df['avgtmax'])
    #m.scatter(x,y,c=(tmax1) ,s=tmax1,marker='o',cmap='viridis')
    # Show the plot.
    #plt.show()
    m2.scatter(x1,y1,c=(tmax11) ,s=(tmax11), alpha=0.4,cmap='coolwarm')
    
   

    plt.colorbar(label=r'$( tmax11)$')
    plt.clim(-50,50 )
    f2.suptitle('Maximum temperature distribution after 2000',fontsize=14)
    plt.show()



    # code for task b part 2 begins here
    predictions = partb2(test_input,model_file)
    #predictions.show()

    predictions = predictions.withColumn('diff',predictions.tmax - predictions.prediction)
    predictions = predictions.withColumn('year',functions.year(predictions["date"]))

    #pdata3 = predictions.filter((predictions['year'] < 2000))
    pdata31 = predictions.groupBy('station').agg(functions.max('diff').alias('diff1'))
    #print(data31.count())
    pdata32 = pdata31.select(pdata31.station.alias('station1'), pdata31.diff1)
    pdata33 = pdata32.join(predictions, pdata32.station1 == predictions.station)
    pquater31 = pdata33.select('station', 'date', 'latitude', 'longitude', 'elevation','year','diff1')
    pquater3 = pquater31.dropDuplicates(['station'])  # this is done

    
    
    pquater2df = pquater3.toPandas()
    
    plt.figure(figsize = (12,6))

    m3 = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    m3.drawmapboundary(fill_color='lightblue', linewidth=0)
    m3.drawcoastlines()
    m3.drawcountries()

    

    # Convert latitude and longitude to x and y coordinates
    x3, y3 = m3(list(pquater2df["longitude"].astype(float)), list(pquater2df["latitude"].astype(float)))
    # Use matplotlib to draw the points onto the map.
    diff1 = list(pquater2df['diff1'])

    m3.scatter(x3,y3,c=(diff1) ,s=(10*diff1), alpha=0.5,cmap='viridis')
    
    plt.colorbar(label=r'$( diff1)$')
    plt.clim(-15,10 )
    plt.title('Predicted temperature difference : Regression ERROR',fontsize=14)
    plt.show()


    #part b 1
    predictions1=create_testdata(model_file)
    lat1s = predictions1.select("latitude").rdd.flatMap(lambda n: n).collect()
    lon1s = predictions1.select("longitude").rdd.flatMap(lambda n: n).collect()
    temp_list = predictions1.select("prediction").rdd.flatMap(lambda n: n).collect()
   
    
    plt.figure(figsize=(12,6))
    m4= Basemap(projection='robin', lon_0= 0, resolution='c')
    m4.drawcoastlines()
    m4.drawmapboundary(fill_color='black')
    lons, lat = m4(lon1s, lat1s)
    m4.scatter(lons, lat, marker='.', c=temp_list,cmap='jet')
    plt.title(" Temperature predictions")
    plt.show()
    
    '''
    #for second plot
    f4 = plt.figure(figsize = (12,6))
    m4 = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    m4.drawmapboundary(fill_color='lightblue', linewidth=0)
    m4.drawcoastlines()
    m4.drawcountries()
    x4, y4 = m4(list(pquater3df["longitude"].astype(float)), list(pquater3df["latitude"].astype(float)))
    # Use matplotlib to draw the points onto the map.
    diff2 = list(pquater3df['diff1'])
    m4.scatter(x4,y4,c=(diff2) ,s=(10*diff2), alpha=0.5,cmap='viridis')
    
   

    plt.colorbar(label=r'$( diff2)$')
    plt.clim(-15,10 )
    f4.suptitle('Predicted temperature difference after 2000',fontsize=14)

    plt.show()

    '''
   
def partb2(test_input,model_file):

    tmax_schema1 = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
    ])
  
    test_tmax = spark.read.csv(test_input, schema=tmax_schema1)

    model = PipelineModel.load(model_file)
    
    # use the model to make predictions
    predictions = model.transform(test_tmax)
    return predictions

def create_testdata(model_file):
    temp_schema = types.StructType([
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('date', types.DateType()),
    types.StructField('temp', types.FloatType()),
    ])
  
   
    
    ls =[]
    lats, lons = (np.arange(-90,90,.5),np.arange(-180,180,.5))
    
    for l in lats:
        for ln in lons:
            ele = eg.get_elevation(l,ln)
            ls.append([float(l),float(ln),float(ele),datetime.strptime('2016-11-12','%Y-%m-%d'),0.0])

    df = spark.createDataFrame(ls,schema=temp_schema)
   
    model = PipelineModel.load(model_file)
    predictions1 = model.transform(df)
    return predictions1
   
    
   
    
if __name__ == '__main__':
    inputs = sys.argv[1]
    test_input = sys.argv[2]
    model_file = sys.argv[3]
    main(inputs,test_input,model_file)



    # command to run : spark-submit final.py tmax-2 tmax-test weather-model

   
