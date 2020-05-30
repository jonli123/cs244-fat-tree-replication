import matplotlib.pyplot as plt
import numpy as np

'''
$7,000 for each 48-port GigE switch at the edge
and $700,000 for 128-port 10 GigE switches in the aggregation and
core layers.
'''

edge_switch_cost = 7000
switch_128_cost = 700000

ratios = ['1','3','7']

num_hosts = range(0,20001)


def num_leaves(layers, branching):
    return branching ** layers


k = np.array([4,8,12,16,24,28,32,48])
fat_tree_hosts = (np.power(k,3))//4

core_switches = np.power(k/2,2)
aggregate_switches = np.power(k,2)/2
edge_switches = np.power(k,2)/2


print(fat_tree_hosts)

cost = 3000* (core_switches + (edge_switches+aggregate_switches))

print(cost)
