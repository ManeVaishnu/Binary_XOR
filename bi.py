import time

# Class definition for node to be added to graph
class AdjacentNode:
    def __init__(self, vertex, weight):
        self.vertex = vertex
        self.weight = weight
        self.next = None

# Dijkstra's algorithm implementation
class Dijkstra:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [None] * self.vertices

    def add_edge(self, src, dest, weight):
        node = AdjacentNode(dest, weight)
        node.next = self.graph[src]
        self.graph[src] = node

    def dijkstra(self, src):
        dist = [float('inf')] * self.vertices
        dist[src] = 0

        visited = [False] * self.vertices

        for _ in range(self.vertices):
            u = self.min_distance(dist, visited)
            visited[u] = True

            node = self.graph[u]
            while node:
                v = node.vertex
                if not visited[v] and dist[u] + node.weight < dist[v]:
                    dist[v] = dist[u] + node.weight
                node = node.next

        return dist

    def min_distance(self, dist, visited):
        min_dist = float('inf')
        min_index = -1

        for v in range(self.vertices):
            if dist[v] < min_dist and not visited[v]:
                min_dist = dist[v]
                min_index = v

        return min_index

if __name__ == '__main__':
    # Take input from the user
    vertices = int(input("Enter the number of vertices: "))
    edges = int(input("Enter the number of edges: "))

    src = int(input("Enter the source node: "))
    dest = int(input("Enter the destination node: "))

    # Create a graph
    graph = Dijkstra(vertices)

    print("Enter edge details (source, destination, weight):")
    for _ in range(edges):
        source, destination, weight = map(int, input().split())
        graph.add_edge(source, destination, weight)

    start_time = time.time()
    distances = graph.dijkstra(src)
    computation_time = time.time() - start_time

    # Print the shortest distance from the source node to each node
    print("Shortest distances from node", src, ":")
    for i in range(vertices):
        print("To node", i, ":", distances[i])

    # Print the shortest path from the source to the destination
    print("Shortest path from node", src, "to node", dest, ":")
    path = []
    node = dest
    while node != src:
        path.append(node)
        min_neighbor = None
        min_weight = float('inf')
        neighbor = graph.graph[node]
        while neighbor:
            if distances[neighbor.vertex] < min_weight:
                min_neighbor = neighbor.vertex
                min_weight = distances[neighbor.vertex]
            neighbor = neighbor.next
        node = min_neighbor
    path.append(src)
    path.reverse()
    print("Path:", path)

    # Print computation time
    print("Computation Time (s):", computation_time)
