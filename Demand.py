import pandas as pd


class Demand():
    def __init__(self, demand_path):
        self.demand = pd.read_csv(demand_path)
        # self.demand = pd.read_csv(demand_path).sample(frac=0.1, random_state=42)
        # print(len(self.demand))
        self.current_demand = self.demand.loc[self.demand['minute'] == 0].reset_index(drop=True)
        self.episode_time = 0
        self.current_time = 0
        self.num_lost_demand = 0

    def initialization(self, episode_time):
        self.current_demand = self.demand.loc[self.demand['minute'] == episode_time].reset_index(drop=True)
        self.episode_time = episode_time
        self.current_time = episode_time
        self.num_lost_demand = 0

    def update(self):
        self.current_time += 1
        self.current_demand = pd.concat(
            [self.current_demand, self.demand.loc[self.demand['minute'] == self.current_time]])
        self.current_demand = self.current_demand.reset_index(drop=True)

        # drop those orders that are not taken over 5 minutes
        if self.current_time >= 5 + self.episode_time:
            self.num_lost_demand += len(self.current_demand[self.current_demand['minute'] <= (self.current_time - 5)])
            self.current_demand = self.current_demand.drop(
                index=self.current_demand[self.current_demand['minute'] <= (self.current_time - 5)].index).reset_index(
                drop=True)

    def pickup(self,unique_r_ids):
        # Convert the set to a list
        unique_r_ids_list = list(unique_r_ids)

        # Drop rows whose index is in unique_r_ids_list
        self.current_demand = self.current_demand.drop(unique_r_ids_list)

        # Reset index
        self.current_demand = self.current_demand.reset_index(drop=True)


