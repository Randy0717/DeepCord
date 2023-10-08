from osrm_router import TSP_route
from transit import *
from utility import geodistance, zone_order_map
from joblib import Parallel, delayed
import ast

mode = 1
if mode == 1:
    action_per_order = 57
else:
    action_per_order = 1

beta0 = 30
beta1 = 40
beta2 = 5
beta3 = 2
beta4 = 10
threshold = 15
penalty = 15


class Central():
    def __init__(self):
        G, Stations = generate_transit_edge(200)
        self.Transit_G = G
        self.Transit_Stations = Stations
        self.Contradiction = 0
        self.Pickup = 0
        self.Total_Reward = 0
        self.Total_Detour = []

    def reset(self):
        self.Contradiction = 0
        self.Pickup = 0
        self.Total_Reward = 0
        self.Total_Detour = []

    def Transit(self, Olat, Olon, Dlat, Dlon):
        time, path = ETA_Transit(Olat, Olon, Dlat, Dlon, self.Transit_G, self.Transit_Stations)

        return time, path

    def OSRM(self, origin_point, destination_points):
        route, route_t, t = TSP_route(origin_point, destination_points)[:-1]

        return route, route_t, t

    def assign(self, decision_table, observe_table):
        Assignment_table = []
        for i in range (len(decision_table)):
            decision = decision_table[i]
            orders = observe_table[i]
            if decision is None or decision[1] == 0:
                Assignment_table.append(None)
            else:
                index = zone_order_map(decision[1])[0]
                Assignment_table.append([orders[index], decision[2], 0])

        r_id_dict = {}
        unique_r_ids = set()
        contradiction_changes = 0

        for row in Assignment_table:
            if row is not None:
                r_id = row[0]
                unique_r_ids.add(r_id)
                q_value = row[1]
                if r_id in r_id_dict:
                    r_id_dict[r_id].append(row)
                else:
                    r_id_dict[r_id] = [row]

        for r_id, rows in r_id_dict.items():
            if len(rows) > 1:
                # Find the row with the highest q-value
                max_q_value_row = max(rows, key=lambda row: row[1])

                # Set contradiction to 1 for all other rows
                for row in rows:
                    if row is not max_q_value_row:
                        row[2] = 1
                        self. Contradiction += 1

        self.Pickup += len(unique_r_ids)

        return Assignment_table, unique_r_ids

    def feedback(self, decision_table, Assignment_table, demand_current, zone_lookup):
        feedback_table = []
        x_table = []
        new_route_table = []
        new_route_time_table = []

        results = Parallel(n_jobs=-1)(
            delayed(excute)(decision, assignment, demand_current, zone_lookup, self.OSRM, self.Transit, geodistance)
            for decision, assignment in zip(decision_table, Assignment_table))

        for i in range(len(results)):
            result = results[i]
            feedback_table.append(result[0])
            x_table.append(result[1])
            new_route_table.append(result[2])
            new_route_time_table.append(result[3])
            if Assignment_table[i] is not None:
                if Assignment_table[i][-1] == 0:
                    self.Total_Reward += result[0][2]
                    self.Total_Detour.append(result[1][7 + 3-int(result[1][1])])

        return feedback_table, x_table, new_route_table, new_route_time_table


def excute(decision, assignment, demand_current, zone_lookup, TSP_route, ETA_Transit, geodistance):
    # output reward , new_route, new_route_time
    if decision is not None:
        s, a = decision[:2]
        if a == 0:
            reward = 0
            x = s[:11]
            x.append(0)
            x.append(0)
            new_route = None
            new_route_time = None
            feedback = [s, a, reward, 1]

        else:
            x_loc = ast.literal_eval(zone_lookup.loc[s[0], '(lat,lon)'])
            occupied_seats = int(3 - s[1])
            r_id = assignment[0]
            contradiction = assignment[2]
            plat, plon, dlat, dlon = demand_current.loc[r_id, 'plat'], demand_current.loc[r_id, 'plon'], \
                demand_current.loc[r_id, 'dlat'], demand_current.loc[r_id, 'dlon']
            zone = zone_order_map(a)[1]
            oloc = ast.literal_eval(zone_lookup.loc[zone, '(lat,lon)'])

            # 0. direct_distance
            direct_distance = geodistance(plat, plon, dlat, dlon)
            direct_time = int((direct_distance*1.3/40)*60)

            # 1. pickup
            # pickup_route, pickup_route_t, pickup_time = TSP_route(x_loc, [(plat, plon)])
            pickup_time = [int((geodistance(x_loc[1],x_loc[0],plon,plat)*1.3/40)*60)]

            # 2. carpooling
            destination_points = []
            for i in range(occupied_seats):
                onboard_loc = ast.literal_eval(zone_lookup.loc[s[5+i], '(lat,lon)'])
                destination_points.append(onboard_loc)
            destination_points.append(oloc)
            new_route, new_route_time, new_time = TSP_route((plat, plon), destination_points)

            if len(new_route) == 0:
                a = 0
                reward = 0
                x = s[:11]
                x.append(0)
                x.append(0)
                feedback = [s, a, reward, 1]
                new_route = None
                new_route_time = None

            else:
                # 3. Transit
                transit, path = ETA_Transit(oloc[0], oloc[1], dlat, dlon)

                # 4. direct transfer
                # direct_route, direct_route_time, direct_time = TSP_route((plat, plon), [destination_points[-1]])
                # direct_time = direct_time[0]

                # 5. reward calculation
                original_total_travel_time = sum(s[2:2+occupied_seats]) + direct_time
                total_travel_time = sum(new_time) + transit + pickup_time[0] * occupied_seats
                add_time = total_travel_time - original_total_travel_time

                if add_time > threshold:
                    reward = beta0 + beta1 * direct_distance - beta2 * pickup_time[0] - beta3 * threshold - beta4 * (
                                add_time - threshold)
                else:
                    reward = beta0 + beta1 * direct_distance - beta2 * pickup_time[0] - beta3 * add_time

                x = s[:11]
                if contradiction == 1:
                    reward = 0
                    x.append(0)
                    x.append(0)
                    new_route = None
                    new_route_time = None
                    pickup_time = [0]

                elif contradiction == 0 and reward <= -30:
                    reward = -30
                    x.append(0)
                    x.append(0)
                    new_route = None
                    new_route_time = None
                    pickup_time = [0]

                else:
                    x[1] -= 1
                    index, zone = zone_order_map(a)
                    x[0] = s[11 + 2*index]
                    x[8: 8 + occupied_seats] = list(np.array(x[8: 8 + occupied_seats]) + np.array(new_time[:-1]) - np.array(x[2:2 + occupied_seats]))
                    x[8 + occupied_seats] = transit + new_time[-1] - direct_time
                    x[2: 2 + occupied_seats + 1] = new_time
                    x[5 + occupied_seats] = zone
                    x.append(1)
                    x.append(pickup_time[0] + 1)

                feedback = [s, a, reward, pickup_time[0]+1]

    else:
        feedback = None
        new_route = None
        new_route_time = None
        x = None

    return feedback, x, new_route, new_route_time












