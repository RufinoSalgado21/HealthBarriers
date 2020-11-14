# Importing the geodesic module from the library
import os
import json
from python import tools
from geopy.distance import geodesic
import pandas as pd
from geopy.geocoders import GoogleV3
import geopy.distance
import googlemaps
import urllib
from googleplaces import GooglePlaces, types, lang

def read_api(directory, file):
    path = os.environ['PYTHONPATH'] + os.path.sep + directory + os.path.sep + file
    file = open(path)
    txt = file.read()
    file.close()
    return txt

def find_nearest_hospital(API,starting_loc):
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
def calculate_distance(API,location_1, location_2):
    print('Calculating distance between points...')
    starting = (location_1[1],location_1[2])
    ending = (float(location_2[1]),float(location_2[2]))

    distance = geodesic(starting,ending).km
    return distance

#Returns latitude and longitude for a given location.
def find_lat_lng(API,location):
    print('Returning latitude and longitude of zip code...')
    geolocator = GoogleV3(api_key=API)

    location_1 = geolocator.geocode(location)
    loc_1_lat = location_1.latitude
    loc_1_long = location_1.longitude

    return [location,loc_1_lat,loc_1_long]

def calc_travel_time(API,location_1, location_2,mode):
    gmap = googlemaps.Client(key=API)
    l1 = (location_1[1],location_1[2])
    l2 = (location_2[1],location_2[2])
    drive_time = gmap.distance_matrix(l1,l2,mode=mode)
    print(drive_time)
    return drive_time['rows'][0]['elements'][0]['duration']['text']

def getResponse(url):
    operUrl = urllib.request.urlopen(url)
    if(operUrl.getcode()==200):
       data = operUrl.read()
    else:
       print("Error receiving data", operUrl.getcode())
    return data

def travel_info(API, location_1, location_2):
    loc1 = find_lat_lng(location_1)
    loc2 = find_lat_lng(location_2)
    direct_distance = calculate_distance(API,loc1, loc2)
    drive_travel_time = calc_travel_time(API, loc1, loc2, 'driving')
    transit_travel_time = calc_travel_time(API, loc1, loc2, 'transit')
    info = [direct_distance,drive_travel_time,transit_travel_time]

def main():
    API = read_api('files', 'key.txt')
    '''
    hospital = find_nearest_hospital(API,'1512 W 19th St Chicago IL 60608')
    zipcode = find_lat_lng(API,'1512 W 19th St Chicago IL 60608')
    print(calculate_distance(API,zipcode,hospital),'km')
    drive_time = calc_travel_time(API,zipcode, hospital,mode='driving')
    transit_time = calc_travel_time(API,zipcode, hospital,mode='transit')
    print(drive_time)
    print(transit_time)
    '''



if __name__ == '__main__':
    main()
