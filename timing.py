import networkx as nx
from collections import deque
import random

#generate a fat tree


class Graph:
	def __init__(self):
		self.vertices = {}
		self.edges = {}
		self.ticks = 0
		self.k = -1
		self.X = -1
		self.Y = -1
		self.Z = -1
		self.END = -1

	def add_vertices(self, vertex_list):
		for v in vertex_list:
			self.vertices[v] = (Node(v))

	def get_neighbors(self, vertex):
		return self.vertices[vertex].neighbors


	def is_path_free(self, path):
		for i in range(len(path) - 1):
			e1 = (path[i], path[i+1])
			e2 = (path[i+1], path[i])
			if e1 in self.edges:
				e = self.edges[e1]
				if e.reserved:
					return False
			else:
				e = self.edges[e2]
				if e.reserved:
					return False
		return True

	def reserve_path(self, path):
		#v1 = self.vertices[path[0]]
		#v2 = self.vertices[path[len(path) - 1]]
		for i in range(len(path) - 1):
			e1 = (path[i], path[i+1])
			e2 = (path[i+1], path[i])
			if e1 in self.edges:
				e = self.edges[e1]
				#print(path)
				e.set_reserved()
			else:
				e = self.edges[e2]
				#print(path)
				e.set_reserved()
			#TODO need to readd to map?

	
	def pair_shortest_paths(self, v1, v2):
		#run BFS until we've found the other node, finish that layer of BFS then return all paths
		#visited = set()
		queue = deque()
		v1_list = [v1]
		queue.append(v1_list)

		answers = []
		stop = False
		i = 0
		while len(queue) > 0:
			i += 1
			path = queue.popleft()
			x = path[len(path) - 1]
			for y in self.get_neighbors(x):
				if y == v2:
					temp = path.copy()
					temp.append(y)
					answers.append(temp)

				stop |= y == v2
				if not stop:
					temp = path.copy()
					temp.append(y)
					queue.append(temp)
		return answers


	def get_pod(self, vertex):
		return ((vertex - self.Z) // (self.k // 2)) // (self.k // 2) 


	def get_switch_from_server(self, vertex, pod):
		z_index = (vertex - self.Z) // (self.k // 2)
		return self.Y + z_index


	def get_switch_from_core(self, core, pod):
		half_k = self.k // 2
		core_index = core // half_k
		return self.X + (pod * half_k) + core_index

	def get_core_path(self, core, v1, v2):
		pod1 = self.get_pod(v1)
		pod2 = self.get_pod(v2)
		top_switch_1 = self.get_switch_from_core(core, pod1)
		top_switch_2 = self.get_switch_from_core(core, pod2)
		bottom_switch_1 = self.get_switch_from_server(v1, pod1)
		bottom_switch_2 = self.get_switch_from_server(v2, pod2)
		return [v1, bottom_switch_1, top_switch_1, core, top_switch_2, bottom_switch_2, v2]

	def get_path_linear_cores(self, v1, v2):
		assert(v1 != v2)
		if (self.get_pod(v1) == self.get_pod(v2)):
			return None
		for i in range (0, (self.k ** 2) // 4):
			path = self.get_core_path(i, v1, v2)
			if self.is_path_free(path):
				#print(path)
				self.reserve_path(path)
				return path
		return None


	def add_edge(self, e):
		self.edges[e] =  Edge(e)
		v1 = self.vertices.get(e[0])
		v1.add_neighbor(e[1])
		v2 = self.vertices.get(e[1])
		v2.add_neighbor(e[0])
		#TODO: update the maps?

	def add_edges(self, edge_list):
		for e in edge_list:
			self.add_edge(e)

	def exists_edge(self, e):
		return Edge(e) in self.edges or Edge((e[1], e[0])) in self.edges

	def exists_vertex(self, a):
		return Node(a) in self.vertices

	def edge_count(self):
		return len(self.edges)


	def print_graph(self):
		print("Vertices: \n")
		for x in self.vertices:
			print(str(x) + "\n")

		print("Edges: \n")
		for y in self.edges:
			print(str(y) + "\n")

	def tick(self):
		for v in self.vertices:
			v.tick()

	def print_degree(self):
		for v, x in self.vertices.items():
			print("Vertex: {}. Degree: {}".format(x.name, x.degree()))


class Node:
	BUFFER_SIZE = 200
	def __init__(self, label):
		self.name = label
		self.neighbors = set()
		self.queue = deque()
		self.dropped_set = set()

	def __eq__(self, other):
		return self.name == other.name
	def __str__(self):
		return str(self.name)
	def __hash__(self):
		return hash(self.name)

	def add_neighbor(self, n):
		self.neighbors.add(n)

	def degree(self):
		return len(self.neighbors)

	#TODO: ticking
	def enqueue(self, packet):
		if len(self.queue) > BUFFER_SIZE:
			self.queue.append(packet)
		else:
			dropped_set.add(packet)

	def tick(self):
		#TODO: define a policy to initiate insertion
		#go through queue and enqueue into neighbors any appropriately timed packets.
		return False




class Edge:
	def __init__(self, e):
		self.first = e[0]
		self.second = e[1]
		self.rate = 1
		self.reserved = False
	def __eq__(self, other):
		return self.first == other.first and self.second == other.second
	def __str__(self):
		return "{} , {}".format(self.first, self.second)
	def __hash__(self):
		return hash((self.first, self.second))
	def set_reserved(self):
		if self.reserved:
			print("First: {}. Second: {}".format(self.first, self.second))
		assert(not self.reserved)
		self.reserved = True


def server_to_edge(i, k):
	assert(False)
	return 0




def fat_tree_host_start(k):
	return (k ** 2) // 4 + (k ** 2)

def fat_tree_host_end(k):
	return (k ** 2) // 4 + (k ** 2) + (k ** 3) // 4




#TODO may need to infer port numbers, or add them in to the model
def build_fat_tree(k):
	g = Graph()
	g.k = k



	#Core nodes range from 0, ..., k^2 / 4 - 1 = X - 1
	#Top level aggregation nodes go from X to X + k^2 / 2 - 1 = Y - 1			(intervals of k/2, k total intervals)
	#Bottom level aggregation nodes go from Y to Y + k^2 /2 - 1 = Z - 1			(intervals of k/2, k total intervals)
	#Servers go from Z to Z + k^3 / 4 - 1										(intervals of k/2, k^2/2 total intervals)



	#Add core nodes (k^2 / 4) = X
	#Add first layer of aggregation (k^2 / 2)
	#Add second layer of aggregation (k^2 / 2)
	#Add servers (k^3 / 4)
	#Aggregate is  k^2 + ((k^2 + k^3) / 4)

	#total_servers = k**2 + (k ** 3 + k ** 2) / 4

	
	X = (k**2 ) // 4
	Y = X + (k ** 2) // 2
	Z = Y + (k ** 2) // 2
	END = Z + (k ** 3) // 4 #not inclusive

	g.X = X
	g.Y = Y
	g.Z = Z
	g.END = END
	g.add_vertices(range(END))


	#Add connections from servers to edge
	temp = Y
	for i in range(Z, END, k // 2):
		g.add_edges([(temp, server) for server in range(i, i + k // 2)])
		temp += 1


	#First to second layer connections
	for i in range(Y, Z):
		#Find the pod it is in. Connect it to entire top level of pod.
		index = (i - Y) // (k // 2)
		pod = (index * (k // 2)) + X
		g.add_edges([(i, first_agg) for first_agg in range(pod, pod + k // 2)])


	#Core connections
	#First k/2 core nodes connect to first node in each pod. Next k/2 goes to second node, etc. 
	for i in range(0, X, k//2):
		#print ("Core step")
		core_nodes = range(i, i + k//2)
		#print(core_nodes)
		#print ("J values")
		for j in range(X + (i // (k // 2)), Y, k // 2):
			#print(j)
			g.add_edges([(j, core_node) for core_node in core_nodes])

	return g




def test_random(graph, k):
	hosts_in_use = set()
	start = fat_tree_host_start(k)
	end = fat_tree_host_end(k)
	hosts_not_in_use = set(range(start, end))
	z  = 0
	for i in range(start, end):
		print("Starting {}".format(z))
		z += 1
		if i not in hosts_not_in_use:
			continue
		hosts_not_in_use.remove(i)
		target = random.choice(tuple(hosts_not_in_use))
		paths = graph.pair_shortest_paths(i, target)
		success = False
		for path in paths:
			if graph.is_path_free(path):
				graph.reserve_path(path)
				success = True
				break
		if success:
			hosts_not_in_use.remove(target)
		else:
			hosts_not_in_use.add(i)
	total_hosts = end - start
	print ("A total of {} hosts connected out of {} total hosts.".format(total_hosts - len(hosts_not_in_use), total_hosts))

				








#g4 = build_fat_tree(4)
#g16 = build_fat_tree(16)
#g24 = build_fat_tree(24)
#g32 = build_fat_tree(32)
#g48 = build_fat_tree(48)
#print ("Done building")






def timer(k):
	from timeit import default_timer as timer
	g = build_fat_tree(k)
	completed = 0
	time = 0.0

	for iterations in range(100):
		#if iterations % 100 == 0:
			#print("Iterations done: {}".format(iterations))
		v1 = random.choice(range(fat_tree_host_start(k), fat_tree_host_end(k)))
		v2 = random.choice(range(fat_tree_host_start(k), fat_tree_host_end(k)))
		if v1 == v2:
			continue

		start = timer()
		g.get_path_linear_cores(v1, v2)
		end = timer()
		time += end - start
		completed += 1

	print ("For k = {}, average time: {}".format(k, time / completed))


#timer(4)
#timer(16)
#timer(24)
#timer(32)
#timer(48)
timer(64)
timer(80)
timer(96)
#timer(128)