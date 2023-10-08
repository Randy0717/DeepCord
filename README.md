# A.Demand Preparations
The trip requests data has already been prepared for you in csv/demand_new.csv, which is extracted from the given link in Link.txt

you may visualize the trip records by running code demand_zone visualization.py, which will take csv/demand_central_oneday1.csv as an input and output the demand_visualization.png

# B.OSRM Implementation 
The route guidance, driving route time estimation, driving route update, and driving route visualization in the simulation model are provide by osrm_router.py and Dijkstra.py, by solving Travel Sallings Man problem.

However, to implement and test the codes, you have to follow the official gudiance of OSRM to use docker to create local server. The links are provided in the Link.txt.

After completing the OSRM guidance, you may run the codes by running test_osrm_2.py, which will print out the route total time for each destination and also show the visualization of the route

# C. Transit ETA model implementation
The metro system guidance and time estimation of each trip is provided by transit.py based on tranisit schedule extracted from the link given in Link.txt and stored in google_transit directory

You can test it by denoting the grey codes down below in transit.py and run the program file directly to check the results.

# D. IDDQN backbone structure
The IDDQN backbone structures for DeepCord and Pure CP case are respectively settlled in DQN_CP.py and DQN_v11.py.

You may check the details by reading through the lines.

# E. Training Simulation Framework
After preparing the stuffs above, you may run the simulation framework in simulator_CP+TR_parallel.py and simulator_CP_parallel.py, which use parallel computing and will save IDDQN parameters in Save directory, and save training plot in Training plot directory during the training process. Also you may read the past training record by running read.py in the recording file. Moreover, you may change some hyperparameters and the save path.

However, before running, you need to first keep your local OSRM server activated. Also, you may need to change the input file path according to your own computer's settings.

# F. Validation
You may also directly do the validation by running simulator_CP+TR_parallel_validation.py to read the pre-trained parameter DQN_CP+TR_0510.pt for DeepCord or running simulator_CP_parallel_validation.py to read DQN_CP_0509.pt for CP case. The codes will print out the total rewards at the end of the episode and save some visualizations that may need further improvement in the Validation directory.

Still, before running, you need to first keep your local OSRM server activated. Also, you may need to change the input file path according to your own computer's settings.

# Final Warning Note:
Please do not share the codes outside with others except for canvas purpose because I am still improving this project and would like to publish a paper. 

Thanks.