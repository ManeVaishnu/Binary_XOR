class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist, pred, src, dest):
        print("\nShortest distance from {} to {} is: {}".format(src, dest, dist[dest]))
        print("Shortest path from {} to {} is: {}".format(src, dest, self.get_path(pred, src, dest)))

    def minDistance(self, dist, sptSet):
        min_dist = float('inf')
        min_index = -1
        for v in range(self.V):
            if dist[v] < min_dist and not sptSet[v]:
                min_dist = dist[v]
                min_index = v
        return min_index

    def get_path(self, pred, src, dest):
        path = []
        current = dest
        while current != src:
            path.insert(0, current)
            current = pred[current]
        path.insert(0, src)
        return ' -> '.join(map(str, path))

    def dijkstra(self, src, dest):
        dist = [float('inf')] * self.V
        pred = [-1] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for _ in range(self.V):
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                        not sptSet[v] and
                        dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
                    pred[v] = u

        self.printSolution(dist, pred, src, dest)

def take_input():
    vertices = int(input("Enter the number of vertices: "))
    edges = int(input("Enter the number of edges: "))

    graph = Graph(vertices)

    print("Enter edges in the format 'start_node end_node weight':")
    for _ in range(edges):
        start, end, weight = map(int, input().split())
        graph.graph[start][end] = weight
        graph.graph[end][start] = weight 

    return graph

if __name__ == "__main__":
    graph = take_input()
    source = int(input("Enter source vertex: "))
    destination = int(input("Enter destination vertex: "))
    graph.dijkstra(source, destination)
