from geopy.distance import great_circle, geodesic
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import itertools

# --------------------------------------------------- Data ----------------------------------------------------------- #
'''
cities = gpd.read_file('../data/us-major-cities/USA_Major_Cities.shp')
og_crs = cities.crs

cols_interest = ['NAME',  # name of city/town
                 'CLASS',  # city, town, etc. classification
                 'ST',  # state code
                 'POPULATION',  # population
                 'geometry'  # point
                 ]

cities = cities[cols_interest]
# filters the columns
cities = cities[(cities['ST'] != 'AK') & (cities['ST'] != 'HI')]
# removes Alaska and Hawaii


big_cities: gpd.GeoDataFrame = cities[cities['POPULATION'] > 500000]

states = gpd.read_file('../data/us-states/States_shapefile.shp').to_crs(epsg=3395)
states: gpd.GeoDataFrame = states[(states['State_Code'] != 'AK') & (states['State_Code'] != 'HI')]
# removes Alaska and Hawaii

cities_for_plot: gpd.GeoDataFrame = gpd.GeoDataFrame(big_cities).to_crs(epsg=3395)  # type: ignore
# cities_for_plot.plot(ax=states.plot(cmap='Pastel2', figsize=(50,50)), marker='o', color='black', markersize=15)

big_cities_cross = big_cities.merge(big_cities, how='cross', suffixes=('_dep', '_arr'))
idx = big_cities_cross['NAME_dep'] != big_cities_cross['NAME_arr']
big_cities_cross = big_cities_cross[idx]
big_cities_cross = big_cities_cross.reset_index().drop(columns='index')

# assume annual passenger volume (in each direction) is 5% of the combined population of the two cities
passenger_vol = 0.05 * (big_cities_cross['POPULATION_dep'] + big_cities_cross['POPULATION_arr'])
big_cities_cross['annual_passengers'] = passenger_vol
# todo: this is a rudimentary way of removing duped pairs: (drop rows that have exact same passenger vol)
big_cities_cross = big_cities_cross[~passenger_vol.duplicated()]
# intercity distances
big_cities_cross['distance_km'] = big_cities_cross.apply(
    lambda row: great_circle_two_points(row['geometry_dep'], row['geometry_arr']).km,
    axis='columns'
)

# print(big_cities_cross)
'''

# --------------------------------------------------- Methods/Functionality ------------------------------------------ #

# returns the geopy great_circle between two shapely Points (note reversal of x<->y)
def great_circle_two_points(pt1: Point, pt2: Point):
    return great_circle((pt1.y, pt1.x), (pt2.y, pt2.x))


