import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import folium
from folium import IFrame



# docker run -t -i -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/new-york-latest.osrm

def TSP_route(origin_point, destination_points):

    url = "http://localhost:5000/trip/v1/driving/" + str(origin_point[1]) + "," + str(origin_point[0]) + ";"

    for i in range(len(destination_points)):
        if i < len(destination_points) - 1:
            url += str(destination_points[i][1]) + "," + str(destination_points[i][0]) + ";"
        else:
            url += str(destination_points[i][1]) + "," + str(
                destination_points[i][0]) + "?roundtrip=false&source=first&annotations=true&geometries=geojson&overview=full"

    session = requests.Session()
    retry = Retry(connect=500000000000000000, backoff_factor=0.2)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    r = session.get(url)
    res = r.json()

    route = []
    route_time = []
    time = []
    time_permutation = []
    time_order = []
    distance = 0

    if res['code'] == 'Ok':
        route = res["trips"][0]["geometry"]["coordinates"]

        distance = res['trips'][0]['distance']

        waypoints = res['waypoints']
        for i in range(len(waypoints)):
            time_permutation.append(waypoints[i]['waypoint_index'])

        legs = res['trips'][0]['legs']
        for i in range(len(legs)):
            route_time.extend(legs[i]['annotation']["duration"])
            if i == 0:
                time.append(int(legs[i]['duration'] / 60))
            else:
                time.append(int(legs[i]['duration'] / 60) + time[i - 1])

        for i in range(len(time)):
            time_order.append(time[time_permutation[i + 1] - 1])

    else:
        route_time = [0]
        time_order = [0]

    return (route,route_time,time_order, distance) # newly added distance

def update_loc(step,route,route_t):
    for i in range(len(route_t)):
        if step < sum(route_t[:i]):
            route = route[i:]
            route_t = route_t[i:]
            break

    loc = (route[0][1],route[0][0])

    return loc,route,route_t


def get_map(route,originpoint, destinationpoints, real_dests, destination_zones, onboard_remaining_time,detour_time, m):#real_dests
    #inverse route to get right lon&lat

    # print(destinationpoints)

    if len(destinationpoints) > 0:

        route_map = []
        for i in range(len(route)):
            route_map.append([route[i][1], route[i][0]])

        folium.PolyLine(
            route_map,
            weight=8,
            color='blue',
            opacity=0.6
        ).add_to(m)

        folium.Marker(
            location=originpoint,
            icon=folium.Icon(icon='play', color='green')
        ).add_to(m)



        for i in range(len(destinationpoints)):
            destination_zone = int(destination_zones[i])
            remaining_time = int(onboard_remaining_time[i])
            detour = int(detour_time[i])
            text = '''
                    <div style="width:200px;color:black">
                    passenger {0} dropoff zone is {1} with remaining time and detour time as {2} and {3} mins
                    </div>'''.format(i, destination_zone, remaining_time, detour)

            iframe = IFrame(text, width=210, height=120)
            popup = folium.Popup(iframe, max_width=500)

            folium.Marker(
                location=destinationpoints[i],
                icon=folium.Icon(icon='stop', color='red'),
                popup=popup,
            ).add_to(m)
            # print(destinationpoints[i])
            # print(real_dests[i])

            folium.PolyLine(
                [destinationpoints[i], real_dests[i]],
                weight=8,
                color='purple',
                opacity=0.6
            ).add_to(m)

            folium.Marker(
                location=real_dests[i],
                icon=folium.Icon(icon='home', color='purple')
                , popup='real_'+str(i)
            ).add_to(m)

    else:

        folium.Marker(
            location=originpoint,
            icon=folium.Icon(icon='play', color='green')
        ).add_to(m)

    return m