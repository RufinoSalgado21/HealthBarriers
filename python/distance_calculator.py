# Importing the geodesic module from the library
import os
from python import tools
from geopy.distance import geodesic
import pandas as pd
from geopy.geocoders import GoogleV3
import geopy.distance
import googlemaps
from googleplaces import GooglePlaces, types, lang

API = tools.read_file('files','key.txt')

def find_nearest_hospital(starting_loc):
    print('Searching for nearest hospital...')
    finder = GooglePlaces(api_key=API)
    geolocator = GoogleV3(api_key=API)
    hospital = ''
    starting_loc = geolocator.geocode(starting_loc)
    loc_1_lat = starting_loc.latitude
    loc_1_long = starting_loc.longitude
    location_1 = (loc_1_lat, loc_1_long)

    result = finder.nearby_search(
        lat_lng={'lat': loc_1_lat, 'lng': loc_1_long},
        radius = 5000,
        types = [types.TYPE_HOSPITAL]
    )

    r = result.places[0]
    hospital = [r.name, r.geo_location['lat'], r.geo_location['lng']]

    return hospital

'''
Returns the straight distance betwwen 2 given locations
represented by lists of 2 values, their longitudes and
lattitudes.
'''
def calculate_distance(location_1, location_2):
    print('Calculating distance between points...')
    starting = (location_1[1],location_1[2])
    ending = (float(location_2[1]),float(location_2[2]))

    distance = geodesic(starting,ending).km
    return distance

#Returns latitude and longitude for a given location.
def find_lat_lng(location):
    print('Returning latitude and longitude of zip code...')
    geolocator = GoogleV3(api_key=API)

    location_1 = geolocator.geocode(location)
    loc_1_lat = location_1.latitude
    loc_1_long = location_1.longitude

    return [location,loc_1_lat,loc_1_long]

def main():
    hospital = find_nearest_hospital('60641')
    zipcode = find_lat_lng('60641')
    print(calculate_distance(zipcode,hospital),'km')


if __name__ == '__main__':
    main()
