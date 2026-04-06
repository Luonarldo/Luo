import geopandas as gpd
import pandas as pd
import numpy as np
import os, glob

in_csv =  r'../gsam-data-output/bj0921/result.csv'
out_shp = r'../gsam-data-output/bj0921/result.gpkg'

df = pd.read_csv(in_csv)
df['idd'] = df['fileBaseName'].str.split('_', expand=True)[0]
df['lon'] = df['fileBaseName'].str.split('_', expand=True)[1]
df['lat'] = df['fileBaseName'].str.split('_', expand=True)[2]
df['date'] = df['fileBaseName'].str.split('_', expand=True)[3]

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326")
gdf.to_file(out_shp, layer='result', driver='GPKG')        
gdf.to_file(out_shp.replace('.gpkg', '.gdb'), layer='result', driver='OpenFileGDB')
gdf.drop(columns=['geometry']).to_excel(out_shp.replace('.gpkg', '.xlsx'))