import pandas as pd
import numpy as np
import ast
import folium

from joblib import Parallel, delayed
from osrm_router import update_loc, get_map
from utility import geodistance
from ActorCritic import BSACAgent

VACANT_SEATS = 3
df = pd.read_csv('csv/demand_new.csv')
Capacity = 2000
actor_Capacity = 2000

class Vehicles():
    def __init__(self, num_vehicles):
        ## Vehicle information initialization
        # X: num_vehicles by 13 size, where only the previous 11 is what we will extract for its own state
        # 0 --- current zones
        # 1 --- vacant seats
        # 2-4 --- remaining travel time on car of each passenger
        # 5-7 --- drop-off destinations of each passenger
        # 8-10 --- total additional time occurred due to pooling+transit of each passenger
        # 11 --- Vehicle state 0: available 1: picking up 2: full
        # 12 --- remaining pickup time

        self.zone_lookup = pd.read_csv("csv/zone_table.csv")

        X = np.zeros((num_vehicles, 13))

        # Initialize Vehicles' zone information and vacant seats
        # random_integers = np.random.randint(0, len(self.zone_lookup), size=(num_vehicles, 1))
        # random_integers = random_integers.flatten()
        # X[:, 0] = random_integers
        X[:, 0] = df[:num_vehicles]['pzone']
        X[:, 1] = VACANT_SEATS

        # Vehicles' route and route time recorder X_travel & X_travel_time
        X_travel_route = []
        X_travel_time = []
        X_experience = []
        X_obsorders = []
        # X_gps = []

        S = []
        # each passenger's real destinations on board
        X_real_dest = np.zeros((num_vehicles, VACANT_SEATS * 2))

        for i in range(num_vehicles):
            X_travel_route.append([])
            X_travel_time.append([])
            X_experience.append([])
            X_obsorders.append([])
            S.append([])
            # loc = ast.literal_eval(self.zone_lookup.loc[X[i,0], '(lat,lon)'])
            # X_gps.append([loc[0],loc[1]])

        self.number = num_vehicles
        self.zone_lookup = self.zone_lookup
        self.X = X
        self.X_travel_route = X_travel_route
        self.X_travel_time = X_travel_time
        self.X_real_dest = X_real_dest
        self.X_experience = X_experience
        self.X_observe_orders = X_obsorders
        self.S = S

        # Policy Parameter
        self.A2CPolicy = BSACAgent()
        self.num_of_train_steps = 0
        self.training_steps = []
        self.M = []
        self.actorm = []

    def reset(self, num_vehicles):
        X = np.zeros((num_vehicles, 13))

        # Initialize Vehicles' zone information and vacant seats
        # random_integers = np.random.randint(0, len(self.zone_lookup), size=(num_vehicles, 1))
        # random_integers = random_integers.flatten()
        # X[:, 0] = random_integers
        X[:, 0] = df[:num_vehicles]['pzone']
        X[:, 1] = VACANT_SEATS

        # Vehicles' route and route time recorder X_travel & X_travel_time
        X_travel_route = []
        X_travel_time = []
        X_experience = []
        X_obsorders = []
        # X_gps = []

        S = []
        # each passenger's real destinations on board
        X_real_dest = np.zeros((num_vehicles, VACANT_SEATS * 2))

        for i in range(num_vehicles):
            X_travel_route.append([])
            X_travel_time.append([])
            X_experience.append([])
            X_obsorders.append([])
            S.append([])
            # loc = ast.literal_eval(self.zone_lookup.loc[X[i,0], '(lat,lon)'])
            # X_gps.append([loc[0],loc[1]])

        self.number = num_vehicles
        self.zone_lookup = self.zone_lookup
        self.X = X
        self.X_travel_route = X_travel_route
        self.X_travel_time = X_travel_time
        self.X_real_dest = X_real_dest
        self.X_experience = X_experience
        self.X_observe_orders = X_obsorders
        self.S = S

    def observe(self, demand_current, current_time):
        if self.number >= 200:
            results = Parallel(n_jobs=-1, backend='loky')(
                delayed(vehicle_observe)(x, self.zone_lookup, demand_current, geodistance, current_time)
                for x in self.X)
        else:
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(vehicle_observe)(x, self.zone_lookup, demand_current, geodistance, current_time)
                for x in self.X)
        self.S = [result[0] for result in results]
        self.X_observe_orders = [result[1] for result in results]

    def query(self, alpha):
        for i in range(self.number):
            s = self.S[i]
            if s is not None:
                self.X_experience[i].append(s)

        for i in range(self.number):
            if len(self.X_experience[i]) == 5:
                self.M = buffer(self.X_experience[i], self.M, CAPACITY=Capacity)
                print(self.X_experience[i])
                self.actorm = buffer(self.X_experience[i], self.actorm, CAPACITY=actor_Capacity)
                self.X_experience[i] = [self.S[i]]

        # if len(self.M) >= Capacity:
        #     self.learn(alpha)

    def decide(self):
        # decision_table = Parallel(n_jobs=-1)(
        #         delayed(vehicle_decide)(s,self.DQN_training, exploration_rate) for s in self.S)
        decision_table = []
        for s in self.S:
            if s is not None:
                decision = [s]
                decision.extend(self.A2CPolicy.get_action(s))
                decision_table.append(decision)
            else:
                decision_table.append(None)

        return decision_table

    def update(self, feedback_table, x_table, new_route_table, new_route_time_table,
               current_time, episode_time, test_time):
        for i in range(self.number):
            feedback = feedback_table[i]
            x = x_table[i]
            new_route = new_route_table[i]
            new_route_time = new_route_time_table[i]

            if feedback is not None:
                # experience load and put into buffer
                self.X_experience[i].append(feedback[1])
                self.X_experience[i].append(feedback[2])
                self.X_experience[i].append(feedback[3])


            if feedback is None or feedback[1] == 0 or new_route is None:
                continue

            # next time information, new_route, new_route_time loading from feedback
            self.X[i] = x
            self.X_travel_route[i] = new_route
            self.X_travel_time[i] = new_route_time

        for n in range(self.number):
            x = self.X[n]
            occupied_seats = 3 - int(x[1])
            # if still picking up
            if x[11] == 1:
                if x[12] >= 1:
                    x[12] = x[12] - 1
                    # if remaining picking up time is 0:
                    if x[12] <= 0:
                        # check whether is full
                        if x[1] == 0:
                            x[11] = 2
                        else:
                            x[11] = 0

            else:
                if x[1] < 3:  # if with passenger, update its next location
                    loc, route, route_t = update_loc(60, self.X_travel_route[n], self.X_travel_time[n])
                    zone_distance = []
                    for j in range(len(self.zone_lookup)):
                        zone_loc = ast.literal_eval(self.zone_lookup.loc[j, '(lat,lon)'])
                        dist = geodistance(loc[1], loc[0], zone_loc[1], zone_loc[0])
                        zone_distance.append(dist)
                    # sorted() to sort the distance
                    smallest = sorted(enumerate(zone_distance), key=lambda x: x[1])[:1]
                    smallest_index = [i for i, _ in smallest]
                    # print(smallest_index)
                    x[0] = smallest_index[0]

                    self.X_travel_route[n] = route
                    self.X_travel_time[n] = route_t
                    x[2: 2 + occupied_seats] -= 1

                    # if passenger arrives at its drop-off place:
                    x = dropoff(x)

            self.X[n] = x

    def draw_map(self, current_time, TEST_VEHICLE, time_interval=3):
        if current_time % time_interval == 0:
            # get information
            route = self.X_travel_route[TEST_VEHICLE]
            x = self.X[TEST_VEHICLE]
            loc = ast.literal_eval(self.zone_lookup.loc[x[0], '(lat,lon)'])
            occupied_seats = 3 - int(x[1])
            destination_points = []
            if occupied_seats > 0:
                for i in range(occupied_seats):
                    destination_points.append(ast.literal_eval(self.zone_lookup.loc[x[5 + i], '(lat,lon)']))
            # draw the map
            map = folium.Map(location=[40.81179592602443, -73.96498583811469], zoom_start=13)
            map = get_map(route, loc, destination_points, map)
            map.save("Validation/CP+TR_Parallel/" + "_" + str(current_time) + ".html")

    def learn(self, alpha):
        if len(self.M) == Capacity:
            self.A2CPolicy.update(self.M, self.actorm, alpha)
        self.training_steps = self.A2CPolicy.training_steps

    def save(self, policy_path):
        self.A2CPolicy.save(policy_path)

    def load(self, load_path):
        self.A2CPolicy.load(load_path)


# 1 vehicle observation:
def vehicle_observe(x, zone_lookup, demand_current, geodistance_function, current_time, unique_rid_lists):
    if x[11] == 0:
        loc = ast.literal_eval(zone_lookup.loc[x[0], '(lat,lon)'])

        # get the 5 nearest orders
        rv_distance = []
        for j in range(len(demand_current)):
            if j in unique_rid_lists:
                dist = 100000
            else:
                dist = geodistance_function(loc[1], loc[0], demand_current.loc[j, 'plon'],
                                            demand_current.loc[j, 'plat'])
            rv_distance.append(dist)
        # sorted() to sort the distance
        smallest_five = sorted(enumerate(rv_distance), key=lambda x: x[1])[:5]
        smallest_five_indices = [i for i, _ in smallest_five]

        # get his own state s
        s = list(x)[:11]
        for j in range(len(smallest_five_indices)):
            s.extend([float(demand_current.loc[smallest_five_indices[j], 'pzone']),
                      demand_current.loc[smallest_five_indices[j], 'dzone']])

        s.append(current_time-480)

    else:
        s = None
        smallest_five_indices = None

    return s, smallest_five_indices

# 3 vehicle drop_off:
def dropoff(x):
    occupied_seats = 3 - int(x[1])
    passenger = []
    get_off = []

    for i in range(occupied_seats):
        passenger.append(i)
        if x[2 + i] <= 0:
            get_off.append(i)
            if x[11] == 2:
                x[11] = 0
    #
    # print(passenger)
    # print(get_off)
    x[1] += len(get_off)
    for i in range(len(get_off)):
        passenger.remove(get_off[i])

    # print(passenger)
    occupied_seats = 3 - int(x[1])

    temp = np.zeros(11)
    for i in range(occupied_seats):
        temp[2 + i] = x[2 + passenger[i]]
        temp[5 + i] = x[5 + passenger[i]]
        temp[8 + i] = x[8 + passenger[i]]

    x[2:11] = temp[2:]
    return x

# 4 vehicle puts his current experience into replay buffer:
def buffer(X_experience, M, CAPACITY=Capacity):
    if len(M) < CAPACITY:
        M.append(X_experience)
    else:
        M.remove(M[0])
        M.append(X_experience)

    return M
