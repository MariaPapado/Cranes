
from pimsys.regions.RegionsDb import RegionsDb
import orbital_vault as ov
from shapely import wkb, geometry, ops
import json
from CustomerDatabase import CustomerDatabase
from datetime import datetime
import requests
import pyproj
import pandas as pd
import numpy as np
import cv2
from shapely.affinity import translate
from geopy.distance import distance

def download_from_geoserve(image, region, width, height, auth=None):

    arguments = {"layer_name": image["wms_layer_name"], "bbox": "%s,%s,%s,%s" % (region[0], region[1], region[2], region[3]), "width": width, "height": height,}

    if "image_url" in image.keys() and image["image_url"] is not None:
        image_url = image["image_url"]
        if not image_url[-1] == "?":
            image_url = image_url + "?"
        arguments["bbox"] = "%s,%s,%s,%s" % (region[1], region[0], region[3], region[2],)
        url = (image_url + "&VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&CRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments)

    try:
        resp = requests.get(url, auth=auth)
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]
    except:
        try:
            resp = requests.get(url)
            image = np.asarray(bytearray(resp.content), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = image[:, :, [2, 1, 0]]
        except:
            print(resp)
            print(url)
            return None
    
    return image

def download_from_mapserver(image, region, width, height, auth=None):
    url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_"
    image_url = url + f'{image["wms_layer_name"]}.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX={region[0]},{region[1]},{region[2]},{region[3]}&WIDTH={width}&HEIGHT={height}&FORMAT=image/png&LAYERS={image["wms_layer_name"]}'
    resp = requests.get(image_url, auth=auth)
    try:
        image = np.asarray(bytearray(resp.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[:, :, [2, 1, 0]]
    except:
        print(image_url)
        return None
    
    return image

def download_image(wms_image, indicator_window, width, height, creds_mapserver, creds_geoserve):
    if wms_image['downloader'] == 'geoserve' or 'geoserve' in wms_image['wms_layer_name']:
        print('Downloading from geoserve')
        print(creds_geoserve)
        if creds_geoserve is not None:
            image = download_from_geoserve(wms_image, indicator_window.bounds, width, height, auth=(creds_geoserve['user'], creds_geoserve['password']))
        else:
            image = download_from_geoserve(wms_image, indicator_window.bounds, width, height)
    
    else:
        print('Downloading from mapserver')
        image = download_from_mapserver(wms_image, indicator_window.bounds, width, height, auth=(creds_mapserver['username'], creds_mapserver['password']))
    return image


# get unix timestamp
def get_utc_timestamp(x: datetime):
    return int((x - datetime(1970, 1, 1)).total_seconds())

# convert from wgs (lat, lon) coordinates to utm (cartesian) coordinates
def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return int(float(epsg_code))


def get_settings(project_name: str):
    db_creds = ov.get_customerdb_credentials()
    db = CustomerDatabase(
        username=db_creds.get("username"),
        password=db_creds.get("password"),
    )

    project = db.get_project_by_name(project_name=project_name)
    creds_client = ov._read_secret(path="internal/credentials/client/admin", env="production")

    return {
        "settings_client": {'host': project.get("servers")[-1].get("domain"),
                            'port': project.get("servers")[-1].get("port"),
                            'user': creds_client.get('user'),
                            'password': creds_client.get('password')},

        "settings_db": ov.get_sarccdb_credentials(),
        "customer": project.get("name"),
        "project": project
    }

def move_small_square_inside(big_square, small_square):
    # Get the bounds of the big square
    big_minx, big_miny, big_maxx, big_maxy = big_square.bounds
    
    # Get the bounds of the small square
    small_minx, small_miny, small_maxx, small_maxy = small_square.bounds
    
    # Calculate the translation needed to keep the small square inside the big square
    dx = 0
    dy = 0
    
    if small_minx < big_minx:
        dx = big_minx - small_minx
    elif small_maxx > big_maxx:
        dx = big_maxx - small_maxx
    
    if small_miny < big_miny:
        dy = big_miny - small_miny
    elif small_maxy > big_maxy:
        dy = big_maxy - small_maxy
    
    # Move the small square by the calculated translation
    small_square_moved = translate(small_square, xoff=dx, yoff=dy)
    
    return small_square_moved
