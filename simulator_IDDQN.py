from Demand import Demand
from Vehicles import Vehicles
from Central_IDDQN import Central
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
EPSILON = 1
exploration_rate = EPSILON
EPSILON_FINAL = 0.00005
EPSILON_DECAY_Rate = 0.995

# SAVE PATH
Load_Path = 'Save/DQN_CP+TR_IDDQN_200car_soft_update_gradclip.pt'
POLICY_PATH = 'Save/DQN_CP+TR_IDDQN_200car_soft_update_gradclip_Test.pt'
training_curve_path = 'Training plot/training_IDDQN_200car_soft_update_gradclip_Test.png'
training_record_path = 'recording/recording_IDDQN_200car_soft_update_gradclip_Test.pkl'

# Simulation System Initialization
Initial_start = time.time()
# DEMAND INITIALIZATION
Demand = Demand(DEMAND_PATH)
Demand.initialization(EPISODE_TIME)
# VEHICLE INITIALIZATION
Vehicles = Vehicles(NUM_VEHICLES)
Vehicles.load(load_path=Load_Path)
# CENTRAL INITIALIZATION
Central = Central()
Initial_end = time.time()
print("Initialization takes {0} seconds".format(Initial_end-Initial_start))

# evaluation benchmarks
contradiction_rate_eps = []
total_reward_eps = []
average_detour_eps = []
training_loss_steps = []


for eps in range(NUM_EPISODES+1):
    print("\n -------- episode {0} starts --------".format(eps+1))
    Demand.initialization(EPISODE_TIME)
    Central.reset()
    Vehicles.reset(NUM_VEHICLES)
    exploration_rate = max(exploration_rate * EPSILON_DECAY_Rate, EPSILON_FINAL)

    for delta_t in range(TEST_TIME):
        Demand.update()

        # print("\ncurrent time is {0}, exploration rate is {1} ".format(Demand.current_time,exploration_rate))

        start = time.time()

        Vehicles.observe(Demand.current_demand, Demand.current_time)
        Vehicles.query()

        # print("{0}'s beginning information is {1}".format(TEST_VEHICLE, Vehicles.X[TEST_VEHICLE]))
        # print("{0}'s observed orders is {1} ".format(TEST_VEHICLE, Vehicles.X_observe_orders[TEST_VEHICLE]))
        end1 = time.time()
        # print(end1 - start)

        if eps % 50 == 0:
            # validation
            decision_table = Vehicles.decide(exploration_rate=0)
        else:
            # training
            decision_table = Vehicles.decide(exploration_rate=exploration_rate)

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

        feedback_table, x_table, new_route_table, new_route_time_table\
            = Central.feedback(decision_table, Assignment_table, Demand.current_demand, Vehicles.zone_lookup)
        end4 = time.time()

        # if feedback_table[TEST_VEHICLE] is not None:
        #     print(" vehicle {0}'s reward is {1} ".format(TEST_VEHICLE, feedback_table[TEST_VEHICLE][2]))

        Vehicles.update(feedback_table, x_table, new_route_table,\
                        new_route_time_table, Demand.current_time, EPISODE_TIME, TEST_TIME)
        end5 = time.time()
        # print("vehicle {0}'s updated information is {1}".format(TEST_VEHICLE, Vehicles.X[TEST_VEHICLE]))
        # Vehicles.draw_map(Demand.current_time, TEST_VEHICLE)
        if eps % 50 != 0 and delta_t % 2 == 0:
            Vehicles.learn()
        Demand.pickup(unique_r_ids)
        end6 = time.time()

        #time record of this minute
        # time_record(start, end1, end2, end3, end4, end5, end6)

    if (Central.Pickup + Central.Contradiction) > 0:
        contradict_rate = Central.Contradiction/(Central.Pickup + Central.Contradiction)
        average_detour = sum(Central.Total_Detour)/len(Central.Total_Detour)

    if eps % 50 != 0:
        contradiction_rate_eps.append(contradict_rate)
        total_reward_eps.append(Central.Total_Reward)
        average_detour_eps.append(average_detour)
        training_loss_steps = Vehicles.training_steps

    else:
        print('\n validation episode: ', eps+1)
        print('No. orders being picked up:', Central.Pickup)
        print("Contradict rate is:", contradict_rate)
        print('Total reward is:', Central.Total_Reward)
        print('Average detour:', average_detour)
        # training_curve(training_loss_steps, total_reward_eps, contradiction_rate_eps,
        #                average_detour_eps, training_curve_path)
        # training_record(training_loss_steps, total_reward_eps, contradiction_rate_eps,
        #                 average_detour_eps, training_record_path)
        # Vehicles.save(POLICY_PATH)

# Vehicles.save(POLICY_PATH)
# EPS_END = time.time()
# print("\n whole training takes {0} hours".format(int(EPS_END-EPS_START)/3600))
