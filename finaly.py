# Python3 program for Bidirectional BFS 
# Search to check path between two vertices 

# Class definition for node to 
# be added to graph 
class AdjacentNode: 
	
	def __init__(self, vertex): 
		
		self.vertex = vertex 
		self.next = None

# BidirectionalSearch implementation 
class BidirectionalSearch:
    
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[] for _ in range(vertices)]
        
        self.src_visited = [False] * self.vertices
        self.dest_visited = [False] * self.vertices
        
        self.src_parent = [None] * self.vertices
        self.dest_parent = [None] * self.vertices
    
    def add_edge(self, src, dest, weight):
        self.graph[src].append((dest, weight))
        self.graph[dest].append((src, weight))
        
    def bfs(self, direction='forward'):
        if direction == 'forward':
            current = self.src_queue.pop(0)
            for neighbor, weight in self.graph[current]:
                if not self.src_visited[neighbor]:
                    self.src_queue.append(neighbor)
                    self.src_visited[neighbor] = True
                    self.src_parent[neighbor] = current
        else:
            current = self.dest_queue.pop(0)
            for neighbor, weight in self.graph[current]:
                if not self.dest_visited[neighbor]:
                    self.dest_queue.append(neighbor)
                    self.dest_visited[neighbor] = True
                    self.dest_parent[neighbor] = current
    
    def is_intersecting(self):
        for i in range(self.vertices):
            if self.src_visited[i] and self.dest_visited[i]:
                return i
        return -1
    
    def print_path(self, intersecting_node, src, dest):
        path = []
        current = intersecting_node
        while current != src:
            path.append(current)
            current = self.src_parent[current]
        path.append(src)
        path = path[::-1]
        current = self.dest_parent[intersecting_node]
        while current != dest:
            path.append(current)
            current = self.dest_parent[current]
        path.append(dest)
        print("*****Path*****")
        print(' '.join(map(str, path)))
    
    def bidirectional_search(self, src, dest):
        self.src_queue = [src]
        self.src_visited[src] = True
        self.dest_queue = [dest]
        self.dest_visited[dest] = True
        
        while self.src_queue and self.dest_queue:
            self.bfs(direction='forward')
            self.bfs(direction='backward')
            intersecting_node = self.is_intersecting()
            if intersecting_node != -1:
                print(f"Path exists between {src} and {dest}")
                print(f"Intersection at: {intersecting_node}")
                self.print_path(intersecting_node, src, dest)
                return
        print(f"Path does not exist between {src} and {dest}")


if __name__ == '__main__': 
    # Take input from the user
    n = int(input("Enter the number of vertices in the graph: "))
    m = int(input("Enter the number of edges in the graph: "))

    # Create an instance of the BidirectionalSearch class
    graph = BidirectionalSearch(n)

    print("Enter the edges (start node, end node, weight):")
    for _ in range(m):
        start, end, weight = map(int, input().split())
        graph.add_edge(start, end, weight)

    src = int(input("Enter the source vertex: "))
    dest = int(input("Enter the destination vertex: "))

    # Perform bidirectional search
    out = graph.bidirectional_search(src, dest)

    if out == -1: 
        print(f"Path does not exist between {src} and {dest}") 
