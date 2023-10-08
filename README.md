# A.Demand Preparations
The trip requests data has already been prepared for you in csv/demand_new.csv, to run this program, u need to download csv directory for demand_new.csv and zone_table.csv from https://hkustconnect-my.sharepoint.com/:f:/g/personal/yhucm_connect_ust_hk/Eiounx6mHOJGinmOgtjt5l0Bfdt0uss9fkCCs6Fn2BKuRQ?e=0Wwvqz

The environment for this study is based on the public dataset of taxi trips provided by Manhattan, New York City. We extracted 30 minutes' worth of trip order requirements during the peak hours (specifically from 8:00 am to 10:00 am) in May 2016, and divided Manhattan into 57 zones. This zoning was informed by the distribution of orders and a resolution of 800m x 800m was used. The visualization is shown in Figure Demand Visualization.png

# B.OSRM Implementation 
The route guidance, driving route time estimation, driving route update, and driving route visualization in the simulation model are provide by osrm_router.py and Dijkstra.py, by solving Travel Sallings Man problem.

However, to implement and test the codes, you have to follow the official gudiance of OSRM to use docker to create local server. The links are provided in our paper.

After completing the OSRM guidance, you may run the codes by running test_osrm_2.py, which will print out the route total time for each destination and also show the visualization of the route

# C. Transit ETA model implementation
The metro system guidance and time estimation of each trip is provided by transit.py based on tranisit schedule extracted from the link given in our paper and stored in google_transit directory. 

To download the google_transit directory, please use the link: https://hkustconnect-my.sharepoint.com/:f:/g/personal/yhucm_connect_ust_hk/ElG__hGtG3ZLp9lo2NMY3qkB-Xdj_3VBEDT-rn-2fCau3w?e=RMlmKF


# D. DeepCord Structure
The architecture of the DeepCord Simulation Framework can be better understood through ![Alt text](/DeepCord Simulation1.png "DeepCord Simulation"). More precisely, in the case of a specific vehicle agent, upon observing multiple orders and making a decision, the central router and Transit Simulator provide feedback. This feedback takes the form of a reward to the vehicle agent, and estimated travel time(ETA) and cost to the designated user. Subsequently, the vehicle agent forwards the transition tuples to the Sequential Independent Deep Reinforcement Learner(SeqIDRL)'s experience replay buffer, thus facilitating further training.

The SeqIDDQN is choosen as the backbone structures for our DeepCord project for both the Pure CP and CP+TR cases.

However, IDDQN and SeqIPPO are also implemented as our benchmarks.

You may check the details by reading through the lines.

# E. Training Simulation Framework
After preparing the stuffs above, you may run the simulation framework in simulator_xxx.py, which use parallel computing and will save neural network parameters in Save directory, and save training plot in Training plot directory during the training process. 

Also you may read the past training record by running read2.py in the recording file. Moreover, you may change some hyperparameters and the save path.

However, before running, you need to first keep your local OSRM server activated. Also, you may need to change the input file path according to your own computer's settings.

# F. Validation
You may also directly do the validation by running simulator_xxx.py to read the pre-trained parameter DQN_CP+TR_SeqDDQNCP_200car.pt for cp cases or DQN_CP+TR_SeqDDQN_200car.pt for CP+TR case to observe performances from our DeepCord directly. 

The codes will print out the total rewards at the end of the episode and save some visualizations in the Validation directory.

Still, before running, you need to first keep your local OSRM server activated. Also, you may need to change the input file path according to your own computer's settings.

# Final Warning Note:
Please quote our codes or paper 'DeepCord: Scalable Deep Multiagent Reinforcement Learning for the Coordination of Ride Sharing and Public Transit' if u use for publication or commercial purposes.

Thanks.
