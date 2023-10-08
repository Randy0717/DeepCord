##Import necessary libraries

import csv
from collections import defaultdict
from Dijkstra import Dijkstra
from math import radians, cos, sin, asin, sqrt
import folium
from osrm_router import *
def time_str_to_float(t):
    parts = t.split(':')
    hours, mins, secs = map(float, parts)
    return hours * 60 * 60 + mins * 60 + secs
assert time_str_to_float('01:23:00') == 1 * 60 * 60 + 23 * 60



#公式计算两点间距离（m）

def geodistance(lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

# print(geodistance(120.12802999999997,30.28708,120.12802999999997,30.28708))
# 返回 446.721 千米

## Note: Transit information could be found on https://developers.google.com/transit/gtfs/reference
## Open-souce code reference found at: https://github.com/nate-parrott/subway

## According to Frequency of transit company
## Generate tranist edge graph and transit station look-up table (ID --- lat, lon)
def generate_transit_edge(Frequency):
    stations_by_id = {}
    station_ids_by_stop_id = {}

    # Get corresponding SUBWAY STATIONS information
    # Stored in DICT stations_by_id
    for stop in csv.DictReader(
            open('google_transit\stops.txt')):
        if stop['parent_station']:
            station_ids_by_stop_id[stop['stop_id']] = stop['parent_station']
        else:
            # 字典里面套字典
            stations_by_id[stop['stop_id']] = {
                'name': stop['stop_name'],
                'lat': float(stop['stop_lat']),
                'lon': float(stop['stop_lon'])
            }

    # Get corresponding SUBWAY ROUTE information
    # Stored in DIC routes_by_id
    routes_by_id = {}
    for route in csv.DictReader(open('google_transit/routes.txt')):
        routes_by_id[route['route_id']] = route

    # Get history weekday service schedule information
    # Stored in stop_times_by_trip_id and stop_times_by_line_and_station respectively
    weekday_service_ids = set(['A20171105WKD', 'B20171105WKD'])
    trips_by_id = {}

    for trip in csv.DictReader(open('google_transit/trips.txt')):
        if trip['service_id'] in weekday_service_ids:
            trips_by_id[trip['trip_id']] = trip

    # 作用是当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
    stop_times_by_line_and_station = defaultdict(list)
    stop_times_by_trip_id = defaultdict(list)

    for stop_time in csv.DictReader(open('google_transit/stop_times.txt')):
        trip_id = stop_time['trip_id']
        if trip_id in trips_by_id:
            stop_id = stop_time['stop_id']
            station_id = station_ids_by_stop_id.get(stop_id, stop_id)
            d = {
                "order": int(stop_time['stop_sequence']),
                "station_id": station_id,
                "time": time_str_to_float(stop_time['departure_time'])
            }
            stop_times_by_trip_id[trip_id].append(d)

            trip = trips_by_id[trip_id]
            line = routes_by_id[trip['route_id']]['route_short_name']
            stop_name = stations_by_id[station_id]['name']
            stop_times_by_line_and_station[(line, station_id)].append(d['time'])

    # Define Class TrainRun to sort schedule and calculate running time between each station of each route
    class TrainRun(object):
        def __init__(self, trip_id, stop_times):
            stop_times = list(sorted(stop_times, key=lambda x: x['order']))
            self.trip = trips_by_id[trip_id]
            self.stop_times = stop_times
            self.station_sequence = [stop_time['station_id'] for stop_time in stop_times]
            self.times_for_stations = {stop_time['station_id']: stop_time['time'] for stop_time in stop_times}
            self.midpoint_time = stop_times[int(len(stop_times) / 2)]['time']
            self.route = routes_by_id[self.trip['route_id']]
            self.line = self.route['route_short_name']

        def times_between_stops(self):
            times = {}
            for prev_station, next_station in zip(self.station_sequence[:-1], self.station_sequence[1:]):
                prev_time = self.times_for_stations[prev_station]
                next_time = self.times_for_stations[next_station]
                duration = next_time - prev_time
                times[(prev_station, next_station)] = duration
                times[(next_station, prev_station)] = duration
                # prev_name = stations_by_id[prev_station]['name']
                # next_name = stations_by_id[next_station]['name']
                # print("Time between {} and {}: {} mins".format(prev_name, next_name, duration / 60))
            return times

        def hashable_orderless_route(self):
            seq = self.station_sequence
            if seq[0] > seq[1]: seq = list(reversed(seq))
            return tuple(seq)

    # def best run to select the line that is closest to 9 am, so as to sort the route information
    # Information stored in DICT runs_by_line and route_set_line respectively
    def select_best_run(runs, time=9 * 60 * 60):
        return min(runs, key=lambda run: abs(run.midpoint_time - time))

    runs_by_trip_id = {id: TrainRun(id, times) for id, times in stop_times_by_trip_id.items()}
    runs_by_line = defaultdict(list)
    route_set_by_line = defaultdict(set)
    for run in runs_by_trip_id.values():
        runs_by_line[run.line].append(run)
        route_set_by_line[run.line].add(run.hashable_orderless_route())

    runs_by_line = {line: select_best_run(runs) for line, runs in runs_by_line.items()}

    # Find only the stations that fall on lines
    # information stored in stations_on_lines
    stations_on_lines = set()
    for run in runs_by_line.values():
        for id in run.station_sequence:
            stations_on_lines.add(id)
    stations_on_lines = {id: stations_by_id[id] for id in stations_on_lines}

    # STEP1: compute a routing graph:
    # the node for being at a station is represented by a station_id
    # the node for being on a train at a station in a given direction is referenced by "station_id+LINE" or "station_id-LINE", depending on direction
    edges = defaultdict(
        list)  # keys are the source nodes; values are a list of ({to_node: node id, time: time in seconds})
    for line, run in runs_by_line.items():
        times_between_stops = run.times_between_stops()
        # print(line, max(times_between_stops.values()))

        for (station_seq, direction) in [(run.station_sequence, '+'), (list(reversed(run.station_sequence)), '-')]:
            # add edges for boarding and exiting the train:
            for station_id in station_seq:
                # boarding:
                from_node = station_id
                to_node = station_id + direction + line
                time = Frequency  # arrival_frequencies[station_id][line] / 2 ## on average, it'll take half the 'time between trains' to board one or transfer
                edges[from_node].append({"to_node": to_node, "time": time})
                # leaving is free:
                edges[to_node].append({"to_node": from_node, "time": 0})

            # add edges for moving between stations:
            for prev_station, next_station in zip(station_seq[:-1], station_seq[1:]):
                time_between = times_between_stops[(prev_station, next_station)]
                from_node = prev_station + direction + line
                to_node = next_station + direction + line
                edges[from_node].append({"to_node": to_node, "time": time_between})

    # add edges for transfers:
    for xfer in csv.DictReader(open('google_transit/transfers.txt')):
        from_id = xfer['from_stop_id']
        to_id = xfer['to_stop_id']
        time = float(xfer['min_transfer_time'])
        for (a, b) in [(from_id, to_id), (to_id, from_id)]:
            edges[a].append({"to_node": b, "time": time})

    # STEP2:
    # Convert the edges into graph form required by the Dijkstra(shortest path finding algorithm)
    G = {}
    for station_seq in edges:
        test = {}
        for i in range(len(edges[station_seq])):
            test[edges[station_seq][i]['to_node']] = edges[station_seq][i]['time']
        G[station_seq] = test

    return G, stations_on_lines

def ETA_Transit(Olat,Olon,Dlat,Dlon,G_transit,stations_on_lines):
    Odistance=[]
    Ddistance=[]
    temp= 3 # number of stations to be considered
    # if temp=1 then it means the user prefer walking distance to be min
    # if temp=3 then it means the user prefer walking distance and transit time together to be jointly considered

    for i in stations_on_lines:
        test1=stations_on_lines[i]
        test1['numb']=i
        test1['dis_to_O']= geodistance(float(test1['lon']),float(test1['lat']),Olon,Olat)
        Odistance.append(test1)

        test2=stations_on_lines[i]
        test2['numb']=i
        test2['dis_to_D']= geodistance(float(test2['lon']),float(test2['lat']),Dlon,Dlat)
        Ddistance.append(test2)

    Odistance=sorted(Odistance, key=lambda x: x['dis_to_O'])
    #print(Odistance)
    O_closest_station=Odistance[0:temp]
    # print(O_closest_station)

    Ddistance=sorted(Ddistance, key=lambda x: x['dis_to_D'])
    #print(Ddistance)
    D_closest_station=Ddistance[0:temp]
    # print(D_closest_station)

    transit=[]
    for i in range(temp):
        for j in range(temp):
            candidate={}
            if str(O_closest_station[i]['numb']) == str(D_closest_station[j]['numb']):
                candidate['path'] = 'just walk'
                candidate['ETA'] = int(geodistance(Olon,Olat,Dlon,Dlat)/3.6 * 60)
            else:
                dijk = Dijkstra(G_transit, str(O_closest_station[i]['numb']), str(D_closest_station[j]['numb']))
                candidate['path'], ETA = dijk.shortest_path()
                ETA = int(O_closest_station[i]['dis_to_O']/3.6 * 60 + ETA / 60 + D_closest_station[j]['dis_to_D'] / 3.6 * 60)
                candidate['ETA']= ETA
            transit.append(candidate)

    transit = sorted(transit, key=lambda x: x['ETA'])

    choice = transit[0]
    # print(choice['path'])

    return choice['ETA'],choice['path']

def transit_map(Stations,path): #choice['path']
    #inverse route to get right lon&lat

    if len(path)>0:
        stat_path = []
        for i in range(len(path)):
            if path[i] in Stations.keys():
                stat_path.append((Stations[path[i]]['lat'], Stations[path[i]]['lon']))

        m = folium.Map(location=[stat_path[0][0],
                                 stat_path[0][1]],
                       zoom_start=13)

        folium.PolyLine(
            stat_path,
            weight=8,
            color='purple',
            opacity=0.6
        ).add_to(m)

        folium.Marker(
            location=stat_path[-1],
            icon=folium.Icon(icon='play', color='red')
        ).add_to(m)

    return m


# Test

# G, Stations = generate_transit_edge(200)
#
# # print(G)
# # print(Stations)
#
# # Some place in New Jersy
# Olat= 40.775198830118
# Olon= -74.0308845523486
#
# # Empire state building
# Dlat= 40.75092371841445
# Dlon= -73.98588965367681

# # 修道院博物馆
# Olat= 40.865491
# Olon= -73.927271
#
# # 洛克公园
# Dlat= 40.83091544907362
# Dlon= -73.93658983312503
#
# time, path = ETA_Transit(Olat,Olon,Dlat,Dlon,G, Stations)
#
# print(time)
# #
# # print(time)
# print(path)
#
# m = transit_map(Stations, path)
# m.save("test" + ".html")
