from Demand import Demand
from Vehicles_AC import *
from Central_AC import Central
from utility import time_record, training_curve, training_record
import time
import random

EPS_START = time.time()

# Hyperparameter settings
DEMAND_PATH = 'csv\demand_new.csv'
EPISODE_TIME = 480
TEST_TIME = 30
TEST_VEHICLE = 0
NUM_VEHICLES = 200
NUM_EPISODES = 0
alpha = 0.05
delta = 0.995
MODE = "CP+TR"

# SAVE PATH
Load_Path = 'Save\DQN_CP+TR_SeqAC_200CAR_larger entrophy.pt'
POLICY_PATH = 'Save\DQN_CP+TR_SeqAC_200CAR_larger entrophy_TEST.pt'
training_curve_path = 'Training plot/training_DQN_CP+TR_AC_200CAR_larger entrophy_TEST.png'
training_record_path = 'recording/recording_SeqAC_200CAR_larger entrophy_TEST.pkl'

# Simulation System Initialization
Initial_start = time.time()
# DEMAND INITIALIZATION
Demand = Demand(DEMAND_PATH)
Demand.initialization(EPISODE_TIME)
# VEHICLE INITIALIZATION
Vehicles = Vehicles(NUM_VEHICLES)
Vehicles.load(load_path=Load_Path)
# CENTRAL INITIALIZATION
Central = Central(mode=MODE)
Initial_end = time.time()
print("Initialization takes {0} seconds".format(Initial_end - Initial_start))

# evaluation benchmarks
contradiction_rate_eps = []
total_reward_eps = []
average_detour_eps = []
training_loss_steps = []
training_actor_loss_steps = []
training_entrophy_loss_steps = []
training_ratio_clip_steps = []

for eps in range(NUM_EPISODES + 1):
    print("\n -------- episode {0} starts --------".format(eps + 1))
    Demand.initialization(EPISODE_TIME)
    Central.reset()
    Vehicles.reset(NUM_VEHICLES)
    # if eps >= 700:
    alpha = max(alpha * delta, 0.05)

    for delta_t in range(TEST_TIME):
        Demand.update()
        # print("\ncurrent time is {0}, exploration rate is {1} ".format(Demand.current_time, alpha))
        start = time.time()
        # print("{0}'s beginning information is {1}".format(TEST_VEHICLE, Vehicles.X[TEST_VEHICLE]))
        # print("{0}'s observed orders is {1} ".format(TEST_VEHICLE, Vehicles.X_observe_orders[TEST_VEHICLE]))

        decision_table = []
        unique_r_ids = []
        for n in range(Vehicles.number):
            if len(Demand.current_demand) == len(unique_r_ids):
                print("orders have all been taken up! by {0} vehicles".format(n + 1))
                break
            x = Vehicles.X[n]
            s, Vehicles.X_observe_orders[n] = vehicle_observe(x, Vehicles.zone_lookup, Demand.current_demand,
                                                              geodistance, Demand.current_time, unique_r_ids)

            if s is not None:
                Vehicles.X_experience[n].append(s)
                if len(Vehicles.X_experience[n]) == 5:
                    Vehicles.M = buffer(Vehicles.X_experience[n], Vehicles.M)
                    Vehicles.actorm = buffer(Vehicles.X_experience[n], Vehicles.actorm, CAPACITY=actor_Capacity)
                    Vehicles.X_experience[n] = [s]

            if s is not None:
                decision = [s]
                decision.extend(Vehicles.A2CPolicy.get_action(s))
                decision_table.append(decision)
            else:
                decision = None
                decision_table.append(None)

            # print(decision_table)
            if decision is not None and decision[1] > 0:

                action = decision[1]

                if action % 57 == 0:
                    index = action // 57 - 1
                else:
                    index = action // 57

                r_id = Vehicles.X_observe_orders[n][index]
                unique_r_ids.append(r_id)

        end1 = time.time()
        # print(end1 - start)

        # if decision_table[TEST_VEHICLE] is not None:
        #     print("vehicle {0}'s decision is {1} ".format(TEST_VEHICLE, decision_table[TEST_VEHICLE][1]))
        end2 = time.time()

        Assignment_table, unique_r_ids = Central.assign(decision_table, Vehicles.X_observe_orders)
        # if Assignment_table[TEST_VEHICLE] is not None:
        #     if Assignment_table[TEST_VEHICLE][2] == 1:
        #         print("vehicle {0}'s decision is contradictory ".format(TEST_VEHICLE))
        #     else:
        #         print("vehicle {0}'s decision is acceptable ".format(TEST_VEHICLE))
        end3 = time.time()

        feedback_table, x_table, new_route_table, new_route_time_table \
            = Central.feedback(decision_table, Assignment_table, Demand.current_demand, Vehicles.zone_lookup)
        end4 = time.time()

        if feedback_table[TEST_VEHICLE] is not None:
            print(" vehicle {0}'s reward is {1} ".format(TEST_VEHICLE, feedback_table[TEST_VEHICLE][2]))

        Vehicles.update(feedback_table, x_table, new_route_table, \
                        new_route_time_table, Demand.current_time, EPISODE_TIME, TEST_TIME)
        end5 = time.time()

        # print("vehicle {0}'s updated information is {1}".format(TEST_VEHICLE, Vehicles.X[TEST_VEHICLE]))
        # Vehicles.draw_map(Demand.current_time, TEST_VEHICLE)

        Vehicles.learn(alpha)

        Demand.pickup(unique_r_ids)
        end6 = time.time()

        # #time record of this minute
        # time_record(start, end1, end2, end3, end4, end5, end6)

    if (Central.Pickup + Central.Contradiction) > 0:
        contradict_rate = Central.Contradiction / (Central.Pickup + Central.Contradiction)
        average_detour = sum(Central.Total_Detour) / len(Central.Total_Detour)

        contradiction_rate_eps.append(contradict_rate)
        total_reward_eps.append(Central.Total_Reward)
        average_detour_eps.append(average_detour)
        training_loss_steps = Vehicles.training_steps
        training_actor_loss_steps = Vehicles.A2CPolicy.actor_eps
        training_entrophy_loss_steps = Vehicles.A2CPolicy.entropy_eps
        training_ratio_clip_steps = Vehicles.A2CPolicy.ratio

        if eps % 20 == 0:
            print('\n validation episode: ', eps + 1)
            print('No. orders being picked up:', Central.Pickup)
            print("Contradict rate is:", contradict_rate)
            print('Total reward is:', Central.Total_Reward)
            print('Average detour:', average_detour)
            # training_curve(training_loss_steps,
            #                training_ratio_clip_steps, total_reward_eps,
            #                average_detour_eps, training_curve_path)
            # training_record(training_loss_steps,
            #                 training_ratio_clip_steps, total_reward_eps,
            #                 average_detour_eps, training_record_path)
            # Vehicles.save(POLICY_PATH)

# Vehicles.save(POLICY_PATH)
EPS_END = time.time()
print("\n whole training takes {0} hours".format(int(EPS_END - EPS_START) / 3600))
