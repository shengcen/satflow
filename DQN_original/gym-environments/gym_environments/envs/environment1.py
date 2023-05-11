# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu
import math

from read_tles import read_tles
from distance_tools import create_basic_ground_station_for_satellite_shadow,distance_m_between_satellites, \
    create_basic_ground_station_for_satellite_shadow, straight_distance_m_between_ground_stations




import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pylab
import json 
import gc
import matplotlib.pyplot as plt
import datetime
import time



MIN_DISTANCE = 3000000
MAX_DISTANCE = 11000000


POWER_TER_LIMIT = 50
TRANS_GAIN = 18*18
KTAU = 1.38*10**(-23) * 580
BANDWIDTH = 250000000
BANDWIDTH_FREQ = 15 * 1000000
# print(BANDWIDTH)
LIGHT_SPEED = 3 * 100000000
FREQ = 23.28 * 1000000000
MY_FILE = "sat0.txt"
MY_LAMBDA = 1
MY_MU = 600
PROFIT = 1/(10 ** 10)



NORM = 1000

N =  2 * 5
sat_per_orbit = 5
orbit_num = 2
# N = 72*22
# sat_per_orbit = 22
# orbit_num = 72

# N = 36*20
# sat_per_orbit = 20
# orbit_num = 36

ALPHA = 0.8
BETA = 0.2

PERIOD = 100


my_tle = read_tles(MY_FILE)


def create_sat0_graph(final_linkage):
    # sat_per_orbit = 6
    # orbit_num = 4
    node_list = [i for i in range(sat_per_orbit * orbit_num)]
    Gbase = nx.DiGraph()
    Gbase.add_nodes_from(node_list)
    edge_list = []

    for i in range(orbit_num):
        for j in range(sat_per_orbit-1):
            edge_list.append((sat_per_orbit*i+j,sat_per_orbit*i+j+1))
            edge_list.append((sat_per_orbit * i + j+1, sat_per_orbit * i + j))
        edge_list.append(((i+1)*sat_per_orbit-1,i*sat_per_orbit))
        edge_list.append(( i * sat_per_orbit,(i + 1) * sat_per_orbit - 1))

    ## inter-plane
    # inter_plane_link = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2,3,4,5]
    for i in range(orbit_num*sat_per_orbit):
        # edge_list.append((i,inter_plane_link[i]))
        # edge_list.append((inter_plane_link[i], i))
        edge_list.append((i, final_linkage[i]))
        edge_list.append((final_linkage[i], i))

    Gbase.add_edges_from(edge_list)

    return Gbase





def create_geant2_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12,19), (12,21),
         (14, 15), (15, 16), (16, 17), (17,18), (18,21), (19, 23), (21,22), (22, 23)])

    return Gbase

def create_nsfnet_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    return Gbase

def create_small_top():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 0),
         (6, 7), (6, 8), (7, 8), (8, 0), (8, 6), (3, 2), (5, 3)])

    return Gbase

def create_gbn_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    Gbase.add_edges_from(
        [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
         (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
         (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

    return Gbase






# my_tle = read_tles("../../../sat0.txt")
# sat0 = my_tle['satellites'][0]
# sat1 = my_tle['satellites'][1]
# sat2 = my_tle['satellites'][2]
# sat3 = my_tle['satellites'][3]
#
# sat15 = my_tle['satellites'][15]
# sat30= my_tle['satellites'][30]

# print(distance_m_between_satellites(my_tle['satellites'][0], my_tle['satellites'][1], '2000/1/1 00:00:00', '2000/1/1 00:00:00'))
# print(distance_m_between_satellites(sat0, sat30, '2000/1/1 00:10:00', '2000/1/1 00:10:00'))
# print(distance_m_between_satellites(sat0, sat30, '2000/1/1 00:20:00', '2000/1/1 00:20:00'))
# print(distance_m_between_satellites(sat0, sat30, '2000/1/1 00:30:00', '2000/1/1 00:30:00'))
# print(distance_m_between_satellites(sat0, sat30, '2000/1/1 00:40:00', '2000/1/1 00:40:00'))
# print(distance_m_between_satellites(sat0, sat30, '2000/1/1 00:50:00', '2000/1/1 00:50:00'))




def time_transfer(time_slot):
    start = '2000/01/01 00:00:00'
    # 先将字符串转化为时间格式
    a = datetime.datetime.strptime(start, "%Y/%m/%d %H:%M:%S")
    c = a + datetime.timedelta(seconds=time_slot)
    timeArray = time.strptime(str(c), "%Y-%m-%d %H:%M:%S")
    ts = time.strftime("%Y/%m/%d %H:%M:%S", timeArray)
    return str(ts)

def generate_nx_graph(topology, time_slot_init,final_linkage):
    """
    Generate graphs for training with the same topology.
    """



    if topology == 0:
        # G = create_nsfnet_graph()
        G = create_sat0_graph(final_linkage)
    elif topology == 1:
        G = create_geant2_graph()
    elif topology == 2:
        G = create_small_top()
    else:
        G = create_gbn_graph()

    # nx.draw(G, with_labels=True)
    # plt.show()
    # plt.clf()

    # Node id counter
    incId = 1
    # Put all distance weights into edge attributes.

    ## UPDATE: ORDERED VERSION
    # for i, j in G.edges():
    for i, j in G.edges():
        # print(i)
        # print(j)
        G.get_edge_data(i, j)['edgeId'] = incId
        G.get_edge_data(i, j)['betweenness'] = 0
        G.get_edge_data(i, j)['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        dist = distance_m_between_satellites(my_tle['satellites'][i], my_tle['satellites'][j], time_transfer(time_slot_init),
                                             time_transfer(time_slot_init))
        capacity = BANDWIDTH * math.log2(1+(POWER_TER_LIMIT*TRANS_GAIN)/(KTAU*BANDWIDTH_FREQ*(4*math.pi*dist*FREQ/LIGHT_SPEED)**2))

        # print(capacity)
        #
        # exit(0)


        # G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)["capacity"] = capacity
        G.get_edge_data(i, j)['bw_allocated'] = 0
        incId = incId + 1

    return G


# generate_nx_graph(0,10)



def compute_link_betweenness(g, k, time_slot_init):




    n = len(g.nodes())
    betw = []
    ## UPDATE: ORDERED VERSION
    # for i, j in G.edges():
    for i, j in g.edges():
        # we add a very small number to avoid division by zero
        # b_link = g.get_edge_data(i, j)['numsp'] / ((2.0 * n * (n - 1) * k) + 0.00000001)

        ## TEST
        b_link = 0
        b_link = distance_m_between_satellites(my_tle['satellites'][i], my_tle['satellites'][j], time_transfer(time_slot_init),
                                      time_transfer(time_slot_init))

        g.get_edge_data(i, j)['betweenness'] = b_link

        betw.append(b_link)


    mu_bet = np.mean(betw)
    std_bet = np.std(betw)
    return mu_bet, std_bet


def compute_energy(currentPath, demand, time_slot):

    energy_flow = 0
    ## ADD
    for m in range(len(currentPath) - 1):
        energy_flow += (2 ** (demand/BANDWIDTH) - 1) * KTAU * BANDWIDTH_FREQ * (4*math.pi*distance_m_between_satellites(my_tle['satellites'][currentPath[m]],
                                                                    my_tle['satellites'][currentPath[m + 1]],
                                                                    time_transfer(time_slot),
                                                                    time_transfer(time_slot))*FREQ/LIGHT_SPEED)**2/TRANS_GAIN

    return energy_flow






class Env1(gym.Env):
    """
    Description:
    The self.graph_state stores the relevant features for the GNN model

    self.graph_state[:][0] = CAPACITY
    self.graph_state[:][1] = BW_ALLOCATED
  """
    def __init__(self):
        self.graph = None
        self.initial_state = None
        self.source = None
        self.destination = None
        self.demand = None
        self.graph_state = None
        self.diameter = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.first = None
        self.firstTrueSize = None
        self.second = None
        self.between_feature = None

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        self.max_demand = 0
        # self.K = 10
        self.K = 4
        self.listofDemands = None
        self.nodes = None
        self.ordered_edges = None
        self.edgesDict = None
        self.numNodes = None
        self.numEdges = None

        self.state = None
        self.episode_over = True
        self.reward = 0
        self.allPaths = dict()
        self.energy_flow_min_theory = 0
        self.path_list = None

        self.sor_list = []
        self.dst_list = []
        self.demand_list = []
        self.time_list = []

        self.energy_flow_list = None
        self.pf_list = None
        self.energy_cnt = 0
        self.demand_cnt = 0
        self.unfull = 0
        self.max_demand = 0
        self.min_demand = 0


    def map_sat(self,lat, long,  time_slot):
        ## return sat index


        min_dist = float("inf")
        min_sat = 0
        for i in range(N):
            # curr_sat_lat = \
            # create_basic_ground_station_for_satellite_shadow(my_tle['satellites'][i], time_transfer(time_slot),
            #                                                  time_transfer(time_slot))["latitude_degrees_str"]
            # curr_sat_long = \
            #     create_basic_ground_station_for_satellite_shadow(my_tle['satellites'][i], time_transfer(time_slot),
            #                                                      time_transfer(time_slot))["longitude_degrees_str"]

            sat_gs = create_basic_ground_station_for_satellite_shadow(my_tle['satellites'][i], time_transfer(time_slot),time_transfer(time_slot))
            flow_gs = {
                "gid": -1,
                "name": "Shadow of sat0",
                "latitude_degrees_str": lat,
                "longitude_degrees_str": long,
                "elevation_m_float": 0,
            }
            curr_dist = straight_distance_m_between_ground_stations(sat_gs, flow_gs)
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_sat = i

        return min_sat
            # print("lat")
            # print(curr_sat_lat)
            # print("long")
            # print(curr_sat_long)




    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # def num_shortest_path(self, topology):
    def num_shortest_path(self, n1, n2, demand):
        #
        # print("sor dst:")
        # print(n1)
        # print(n2)


        start_time = time.time()
        baodi_path = 0
        candidate_path = []
        self.diameter = nx.diameter(self.graph)

        # Iterate over all node1,node2 pairs from the graph
        # for n1 in self.graph:
        #     for n2 in self.graph:
        if (n1 != n2):
            # Check if we added the element of the matrix
            if str(n1)+':'+str(n2) not in self.allPaths:
                self.allPaths[str(n1)+':'+str(n2)] = []



            # We take all the paths from n1 to n2 and we order them according to the path length
            # self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))

            [self.allPaths[str(n1) + ':' + str(n2)].append(p) for p in nx.all_shortest_paths(self.graph, source=n1, target=n2)]

            # print("candidate length: ")
            # print(len(self.allPaths[str(n1)+':'+str(n2)]))

            path = 0
            while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                currentPath = self.allPaths[str(n1)+':'+str(n2)][path]

                not_allocate = False

                i = 0
                j = 1
                # currentPath = self.allPaths[str(source) + ':' + str(destination)][action]

                # Once we pick the action, we decrease the total edge capacity from the edges
                # from the allocated path (action path)
                while (j < len(currentPath)):
                    # self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                    #     0] -= demand
                    if self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] < demand:
                        # FINISH IF LINKS CAPACITY <0
                        # return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination
                        # self.reward = 0
                        # self.unfull += 1
                        not_allocate = True
                        break
                    i = i + 1
                    j = j + 1


                # i = 0
                # j = 1

                # Iterate over pairs of nodes increase the number of sp
                # while (j < len(currentPath)):
                #     self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] = \
                #         self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] + 1
                #     i = i + 1
                #     j = j + 1

                if not not_allocate:
                    candidate_path.append(currentPath)
                baodi_path = currentPath


                path = path + 1

            # Remove paths not needed
            # del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
            del self.allPaths[str(n1)+':'+str(n2)]
            gc.collect()

            if len(candidate_path) == 0:
                candidate_path.append(baodi_path)

            end_time = time.time()

            # print("time: ")
            # print(end_time - start_time)




            return candidate_path


    def _first_second_between(self):
        self.first = list()
        self.second = list()

        # For each edge we iterate over all neighbour edges
        for i, j in self.ordered_edges:
            neighbour_edges = self.graph.edges(i)

            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) +':'+ str(j)])
                    self.second.append(self.edgesDict[str(m) +':'+ str(n)])

            neighbour_edges = self.graph.edges(j)
            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) +':'+ str(j)])
                    self.second.append(self.edgesDict[str(m) +':'+ str(n)])


    def generate_environment(self, topology, listofdemands, time_slot_init, final_linkage):
        # The nx graph will only be used to convert graph from edges to nodes

        self.sor_list = []
        self.dst_list = []
        self.demand_list = []
        self.time_list = []

        source, destination, demand = 0,0,0

        index_cnt = 0


        ## Northern Virginia: 20
        ## Seattle: 7
        ## California Bay Area: 7
        ## Northeastern Oregon: 4
        ## Dublin: 6
        ## Luxembourg:3
        ## Frankfurt: 4
        ## Beijing: 6
        ## Ningxia: 1
        ## Tokyo: 7
        ## Osaka: 2
        ## Singapore: 4
        ## Sydney: 8
        ## Sao Paolo: 4
        ## Rio: 1


        for time_slot in range(time_slot_init, time_slot_init + PERIOD):



            # if time_slot % 6 == 0:
            #     source = self.map_sat("40", "115", time_slot)
            #     destination = self.map_sat("40", "-74", time_slot)
            # elif time_slot % 6 == 1:
            #     source = self.map_sat("-20", "85", time_slot)
            #     destination = self.map_sat("60", "145", time_slot)
            # elif time_slot % 6 == 2:
            #     source = self.map_sat("10", "135", time_slot)
            #     destination = self.map_sat("-40", "5", time_slot)
            # elif time_slot % 6 == 3:
            #     source = self.map_sat("60", "-23", time_slot)
            #     destination = self.map_sat("2", "-25", time_slot)
            # elif time_slot % 6 == 4:
            #     source = self.map_sat("80", "-125", time_slot)
            #     destination = self.map_sat("10", "45", time_slot)
            # elif time_slot % 6 == 5:
            #     source = self.map_sat("-60", "75", time_slot)
            #     destination = self.map_sat("-32", "155", time_slot)
            # else:
            #     source = self.map_sat("25", "5", time_slot)
            #     destination = self.map_sat("-11", "-65", time_slot)


            ## Northern Virginia: 20
            ## Seattle: 7
            ## California Bay Area: 7
            ## Northeastern Oregon: 4
            ## Dublin: 6
            ## Luxembourg:3
            ## Frankfurt: 4
            ## Beijing: 6
            ## Ningxia: 1
            ## Tokyo: 7
            ## Osaka: 2
            ## Singapore: 4
            ## Sydney: 8
            ## Sao Paolo: 4
            ## Rio: 1

            random.seed(time_slot)
            draw_a = random.randint(0,83)
            if draw_a <= 19:
                source = self.map_sat("43", "-119", time_slot)
            elif draw_a <= 26:
                source = self.map_sat("47", "-122", time_slot)
            elif draw_a <= 33:
                source = self.map_sat("33", "-117", time_slot)
            elif draw_a <= 37:
                source = self.map_sat("46", "-117", time_slot)
            elif draw_a <= 43:
                source = self.map_sat("53", "-6", time_slot)
            elif draw_a <= 46:
                source = self.map_sat("49", "6", time_slot)
            elif draw_a <= 50:
                source = self.map_sat("50", "8", time_slot)
            elif draw_a <= 56:
                source = self.map_sat("40", "117", time_slot)
            elif draw_a <= 57:
                source = self.map_sat("38", "105", time_slot)
            elif draw_a <= 64:
                source = self.map_sat("35", "139", time_slot)
            elif draw_a <= 66:
                source = self.map_sat("34", "135", time_slot)
            elif draw_a <= 70:
                source = self.map_sat("1", "104", time_slot)
            elif draw_a <= 78:
                source = self.map_sat("-33", "151", time_slot)
            elif draw_a <= 82:
                source = self.map_sat("-23", "-46", time_slot)
            else:
                source = self.map_sat("-22", "-43", time_slot)

            random.seed(time_slot + 3)

            draw_a = random.randint(0, 83)
            if draw_a <= 19:
                destination = self.map_sat("43", "-119", time_slot)
            elif draw_a <= 26:
                destination = self.map_sat("47", "-122", time_slot)
            elif draw_a <= 33:
                destination = self.map_sat("33", "-117", time_slot)
            elif draw_a <= 37:
                destination = self.map_sat("46", "-117", time_slot)
            elif draw_a <= 43:
                destination = self.map_sat("53", "-6", time_slot)
            elif draw_a <= 46:
                destination = self.map_sat("49", "6", time_slot)
            elif draw_a <= 50:
                destination = self.map_sat("50", "8", time_slot)
            elif draw_a <= 56:
                destination = self.map_sat("40", "117", time_slot)
            elif draw_a <= 57:
                destination = self.map_sat("38", "105", time_slot)
            elif draw_a <= 64:
                destination = self.map_sat("35", "139", time_slot)
            elif draw_a <= 66:
                destination = self.map_sat("34", "135", time_slot)
            elif draw_a <= 70:
                destination = self.map_sat("1", "104", time_slot)
            elif draw_a <= 78:
                destination = self.map_sat("-33", "151", time_slot)
            elif draw_a <= 82:
                destination = self.map_sat("-23", "-46", time_slot)
            else:
                destination = self.map_sat("-22", "-43", time_slot)

            if source == destination:
                destination = (source + N//3) % N







            demand = listofdemands[time_slot % 3]

            self.sor_list.append(source)
            self.dst_list.append(destination)
            self.demand_list.append(demand)
            self.time_list.append(PERIOD-index_cnt)
            index_cnt += 1


        self.graph = generate_nx_graph(topology, time_slot_init, final_linkage)

        self.listofDemands = listofdemands

        self.max_demand = np.amax(self.listofDemands)
        self.min_demand = np.amin(self.listofDemands)

        # Compute number of shortest paths per link. This will be used for the betweenness
        # self.num_shortest_path(topology)



        ### ADD IF
        # if time_slot_init == 0:
        #     self.num_shortest_path()

        # Compute the betweenness value for each link
        self.mu_bet, self.std_bet = compute_link_betweenness(self.graph, self.K, time_slot_init)

        self.edgesDict = dict()

        ## UPDATE  ORDERED EDGES

        some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
        # some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.ordered_edges()]

        self.ordered_edges = sorted(some_edges_1)

        self.numNodes = len(self.graph.nodes())

        self.numEdges = len(self.graph.edges())
        # self.numEdges = len(self.graph.ordered_edges())

        self.graph_state = np.zeros((self.numEdges, 2))
        self.between_feature = np.zeros(self.numEdges)

        position = 0
        for edge in self.ordered_edges:
            i = edge[0]
            j = edge[1]
            self.edgesDict[str(i)+':'+str(j)] = position
            self.edgesDict[str(j)+':'+str(i)] = position


            ## TEST
            # betweenness = (self.graph.get_edge_data(i, j)['betweenness'] - self.mu_bet) / self.std_bet
            betweenness = 0


            self.graph.get_edge_data(i, j)['betweenness'] = betweenness
            self.graph_state[position][0] = self.graph.get_edge_data(i, j)["capacity"]
            self.between_feature[position] = self.graph.get_edge_data(i, j)['betweenness']
            position = position + 1

        self.initial_state = np.copy(self.graph_state)

        self._first_second_between()

        self.firstTrueSize = len(self.first)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))


    ## NEW
    # graph state as the time changes
    def quant_change(self, time_slot):

        # self.allPaths = dict()
        position = 0
        for i, j in self.graph.edges():
            self.graph.get_edge_data(i, j)['betweenness'] = (distance_m_between_satellites(my_tle['satellites'][i], my_tle['satellites'][j],
                                                 time_transfer(time_slot),
                                                 time_transfer(time_slot)) - MIN_DISTANCE)/(MAX_DISTANCE - MIN_DISTANCE)
            self.between_feature[position] = self.graph.get_edge_data(i, j)['betweenness']

            # print(self.between_feature[position])


            dist = distance_m_between_satellites(my_tle['satellites'][i], my_tle['satellites'][j],
                                                 time_transfer(time_slot),
                                                 time_transfer(time_slot))
            capacity = BANDWIDTH * math.log2(1 + (POWER_TER_LIMIT * TRANS_GAIN) / (
                        KTAU * BANDWIDTH_FREQ * (4 * math.pi * dist * FREQ / LIGHT_SPEED) ** 2))

            last_capacity = 0
            last_dist = 0
            if time_slot >= 1:
                last_dist = distance_m_between_satellites(my_tle['satellites'][i], my_tle['satellites'][j],
                                                 time_transfer(time_slot-1),
                                                 time_transfer(time_slot-1))
                # last_capacity = BANDWIDTH * math.log2(1 + (POWER_TER_LIMIT * TRANS_GAIN) / (
                #         KTAU * BANDWIDTH_FREQ * (4 * math.pi * last_dist * FREQ / LIGHT_SPEED) ** 2))

            # print(capacity)
            #
            # exit(0)

            # G.get_edge_data(i, j)["capacity"] = float(200)
            if time_slot >= 1:
                # self.graph.get_edge_data(i, j)["capacity"] = self.graph.get_edge_data(i, j)["capacity"] - last_capacity + capacity
                self.graph.get_edge_data(i, j)["capacity"] = \
                    self.graph_state[self.edgesDict[str(i) + ':' + str(j)]][0]
            else:
                self.graph.get_edge_data(i, j)["capacity"] = capacity

            position = position + 1

        # exit(0)
        # self.num_shortest_path()




    def make_step(self, state, action, demand, source, destination, time_slot, path_list):

        seed_list = [i for i in range(100)]




        self.graph_state = np.copy(state)
        self.episode_over = True
        self.reward = 0
        not_allocate = False

        i = 0
        j = 1
        # currentPath = self.allPaths[str(source) +':'+ str(destination)][action]
        # currentPath = self.allPaths[str(source) + ':' + str(destination)][action]
        currentPath = path_list[action]
        energy_flow = 0






        # Once we pick the action, we decrease the total edge capacity from the edges
        # from the allocated path (action path)
        while (j < len(currentPath)):
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
            if self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] < 0:
                # FINISH IF LINKS CAPACITY <0
                # return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination
                # self.reward = 0
                self.unfull += 1
                not_allocate = True
                break
            i = i + 1
            j = j + 1

        if not not_allocate:

            self.path_list.append(currentPath)
            # self.demand_list.append(demand)




        # Leave the bw_allocated back to 0
        self.graph_state[:,1] = 0

        if not not_allocate:
            energy_flow = 0

            ## ADD
            energy_flow = compute_energy(currentPath, demand, time_slot)
            self.energy_flow_list.append(energy_flow)
            self.pf_list.append(PROFIT * demand * distance_m_between_satellites(my_tle['satellites'][source], my_tle['satellites'][destination],
                                                     time_transfer(time_slot),
                                                     time_transfer(time_slot)) )
            # self.reward = (PROFIT * demand * distance_m_between_satellites(my_tle['satellites'][source], my_tle['satellites'][destination],
            #                                          time_transfer(time_slot),
            #                                          time_transfer(time_slot)) - energy_flow) / NORM

            # curr_profit = PROFIT * demand * distance_m_between_satellites(my_tle['satellites'][source],
            #                                                               my_tle['satellites'][destination],
            #                                                               time_transfer(time_slot),
            #                                                               time_transfer(time_slot))
            #
            # min_profit = PROFIT * self.min_demand * distance_m_between_satellites(my_tle['satellites'][0],
            #                                                                       my_tle['satellites'][1],
            #                                                                       time_transfer(time_slot),
            #                                                                       time_transfer(time_slot))
            #
            # max_profit = PROFIT * self.max_demand * distance_m_between_satellites(my_tle['satellites'][0],
            #                                                                       my_tle['satellites'][
            #                                                                           int(orbit_num * sat_per_orbit / 2)],
            #                                                                       time_transfer(time_slot),
            #                                                                       time_transfer(time_slot))
            #
            # min_energy = 0
            # max_energy = compute_energy([0, 1], self.max_demand, time_slot) * int(sat_per_orbit / 2 + orbit_num / 2)
            # self.reward = ALPHA * (curr_profit - min_profit) / (max_profit - min_profit) + \
            #               BETA * (1 - (energy_flow - min_energy) / (max_energy - min_energy))

            # self.energy_flow_list.append(energy_flow*(-1))


        else:
            # self.reward = 0
            self.reward = 0 + BETA * (1 - 0)



        for m in range(len(self.energy_flow_list)):
            self.energy_cnt += self.energy_flow_list[m]

        # for m in range(len(self.demand_list)):
        #     self.demand_cnt += self.demand_list[m]

        # if time_slot % 1 == 0:
        #     least_energy_path = sorted(self.allPaths[str(source) + ':' + str(destination)],
        #                                                     key=lambda item: (compute_energy(item,demand, time_slot), item))[0]
        #
        #     ## given sor and dst and curr time slot
        #     self.energy_flow_min_theory = compute_energy(least_energy_path, demand, time_slot)

        # Reward is the allocated demand or 0 otherwise (end of episode)
        # We normalize the demand to don't have extremely large values

        current_energy = 0
        current_profit = 0


        for m in range(len(self.energy_flow_list)):
            current_energy += self.energy_flow_list[m]

        for m in range(len(self.pf_list)):
            current_profit += self.pf_list[m]

        # self.reward = (current_profit - current_energy)/NORM
        # self.reward = - 1 * current_energy
        self.reward = - 1 * energy_flow * ((PERIOD - time_slot)%(PERIOD+1)) / 50

        # print("profit: "+str(current_profit))
        # print("energy: "+str(current_energy))


        # if not not_allocate:
        #     self.reward = (MY_LAMBDA*demand/(5*MY_MU*energy_flow))/(MY_LAMBDA*demand/(5*MY_MU*self.energy_flow_min_theory))
        # else:
        #     self.reward = 0


        # self.reward = demand - 5*energy_flow

        # print(str(demand)+"  "+str(5*energy_flow))
        # print("my_reward: "+str(self.reward))


        self.episode_over = False

        random.seed(seed_list[time_slot%100])
        np.random.seed(seed_list[time_slot%100])

        # seed = int(time.time())
        # random.seed(seed)
        # np.random.seed(seed)

        # self.demand = random.choice(self.listofDemands)
        # self.source = random.choice(self.nodes)
        # self.destination = (self.source + 13) % N


        if not (time_slot % 30 ==0):
            self.demand = self.demand_list[0]
            self.source = self.sor_list[0]
            self.destination = self.dst_list[0]

            self.demand_list.pop(0)
            self.sor_list.pop(0)
            self.dst_list.pop(0)

        # print(self.source)

        #
        # if time_slot % 3 == 0:
        #     self.source = self.map_sat("40", "115", time_slot)
        #     self.destination = self.map_sat("40", "-74", time_slot)
        # elif time_slot % 3 == 1:
        #     self.source = self.map_sat("-20", "85", time_slot)
        #     self.destination = self.map_sat("60", "145", time_slot)
        # else:
        #     self.source = self.map_sat("10", "10", time_slot)
        #     self.destination = self.map_sat("-60", "95", time_slot)
        #
        # if self.source == self.destination:
        #     self.destination = (self.destination+1)%N
        #
        # self.demand = self.listofDemands[time_slot % 3]


        # self.source = self.nodes[time_slot % (orbit_num*sat_per_orbit)]

        # We pick a pair of SOURCE,DESTINATION different nodes
        ## flow 出现
        # while True:
        #     self.destination = random.choice(self.nodes)
        #     if self.destination != self.source:
        #         break




        ## flow 消失
        '''
        i = 0
        j = 1



        if len(self.demand_list) >= N:
            # a = random.randint(0,N-1)
            a = 0
            end_path = self.path_list[a]
            end_demand = self.demand_list[a]
            while (j < len(end_path)):
                self.graph_state[self.edgesDict[str(end_path[i]) + ':' + str(end_path[j])]][0] += end_demand
                i = i + 1
                j = j + 1
            del self.path_list[a]
            del self.demand_list[a]
            del self.energy_flow_list[a]
            del self.pf_list[a]
            '''


        return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination



    def reset(self, listofdemands,time_slot):
        """
        Reset environment and setup for new episode. Generate new demand and pair source, destination.

        Returns:
            initial state of reset environment, a new demand and a source and destination node
        """

        self.graph_state = np.copy(self.initial_state)
        # self.demand = random.choice(self.listofDemands)
        # self.source = random.choice(self.nodes)
        #
        # # We pick a pair of SOURCE,DESTINATION different nodes
        # while True:
        #     self.destination = random.choice(self.nodes)
        #     if self.destination != self.source:
        #         break

        self.demand = self.demand_list[0]
        self.source = self.sor_list[0]
        self.destination = self.dst_list[0]

        self.demand_list.pop(0)
        self.sor_list.pop(0)
        self.dst_list.pop(0)

        # if time_slot % 3 == 0:
        #     self.source = self.map_sat("40", "115", time_slot)
        #     self.destination = self.map_sat("40", "-74", time_slot)
        # elif time_slot % 3 == 1:
        #     self.source = self.map_sat("-20", "85", time_slot)
        #     self.destination = self.map_sat("60", "145", time_slot)
        # else:
        #     self.source = self.map_sat("10", "10", time_slot)
        #     self.destination = self.map_sat("-60", "95", time_slot)
        #

        #
        # self.source = self.nodes[0]
        # self.destination = self.nodes[10]


        # self.sor_list = []
        # self.dst_list = []
        # self.demand_list = []
        # self.time_list = []

        self.path_list = []
        self.energy_flow_list = []
        self.pf_list = []
        self.unfull = 0
        self.energy_cnt = 0
        self.demand_cnt = 0

        return self.graph_state, self.demand, self.source, self.destination
    
    def eval_sap_reset(self, demand, source, destination):
        """
        Reset environment and setup for new episode. This function is used in the "evaluate_DQN.py" script.
        """
        self.graph_state = np.copy(self.initial_state)
        self.demand = demand
        self.source = source
        self.destination = destination

        return self.graph_state