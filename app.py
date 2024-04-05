import heapq
import math
import time

def heuristic(node, end_node, positions):
    # Euclidean distance as the heuristic
    node_pos = positions.get(node)
    end_node_pos = positions.get(end_node)
    if node_pos is not None and end_node_pos is not None:
        return math.sqrt((end_node_pos[0] - node_pos[0])**2 + (end_node_pos[1] - node_pos[1])**2)
    return float('inf') 

def a_star_algorithm(start_node, end_node, graph, positions):
    start_time = time.time()
    total_distance = 0
    nodes_visited = 0
    open_set = set([start_node])
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start_node] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start_node] = heuristic(start_node, end_node, positions)

    open_heap = []
    heapq.heappush(open_heap, (f_score[start_node], start_node))

    while open_set:
        current = heapq.heappop(open_heap)[1]
        if current == end_node:
            path = reconstruct_path(came_from, current)
            computation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            return path, total_distance, nodes_visited, computation_time

        open_set.remove(current)
        for neighbor in graph[current]:
            nodes_visited += 1
            tentative_g_score = g_score[current] + graph[current][neighbor]
            if tentative_g_score < g_score[neighbor]:
                total_distance += graph[current][neighbor]
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end_node, positions)
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None, None, None, None  # If no path is found

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path

def bidirectional_dijkstra(graph, start, goal):
    def dijkstra_partial(graph, start, other_visited):
        distances = {vertex: float('infinity') for vertex in graph}
        distances[start] = 0
        pq = [(0, start)]
        visited = {}
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            if current_vertex in other_visited:
                return visited, current_vertex

            visited[current_vertex] = current_distance
            for neighbor, weight in graph[current_vertex].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return visited, None

    visited_start, visited_goal = {}, {}
    while True:
        visited_start, middle = dijkstra_partial(graph, start, visited_goal)
        if middle:
            break
        visited_goal, middle = dijkstra_partial(graph, goal, visited_start)
        if middle:
            break

    path = [middle]
    while path[-1] != start:
        for neighbor in graph[path[-1]]:
            if neighbor in visited_start and visited_start[neighbor] + graph[path[-1]][neighbor] == visited_start[path[-1]]:
                path.append(neighbor)
                break
    path.reverse()

    next_node = middle
    while next_node != goal:
        for neighbor in graph[next_node]:
            if neighbor in visited_goal and visited_goal[neighbor] + graph[next_node][neighbor] == visited_goal[next_node]:
                next_node = neighbor
                path.append(next_node)
                break

    return path



if __name__ == "__main__":
    # Take input from the user
    N = int(input("Enter the number of nodes: "))
    M = int(input("Enter the number of edges: "))

    graph = {}
    positions = {}
    print("Enter edge details (start node, end node, weight):")
    for _ in range(M):
        start, end, weight = input().split()
        weight = int(weight)
        if start not in graph:
            graph[start] = {}
        graph[start][end] = weight
        if end not in graph:
            graph[end] = {}
        graph[end][start] = weight

    print("Enter positions of nodes (node, x, y):")
    for _ in range(N):
        node, x, y = input().split()
        positions[node] = (int(x), int(y))

    start_node = input("Enter the source node: ")
    end_node = input("Enter the destination node: ")

    # Run A* algorithm
    path_astar, total_distance_astar, _, _ = a_star_algorithm(start_node, end_node, graph, positions)
    print("A* Algorithm:")
    print("Path:", path_astar)
    print("Total Distance:", total_distance_astar)

    print()

    # Run Bidirectional Dijkstra's algorithm
    path_bidirectional_dijkstra = bidirectional_dijkstra(graph, start_node, end_node)
    total_distance_bidirectional = sum(graph[u][v] for u, v in zip(path_bidirectional_dijkstra, path_bidirectional_dijkstra[1:]))
    print("Bidirectional Dijkstra's Algorithm:")
    print("Path:", path_bidirectional_dijkstra)
    print("Total Distance:", total_distance_bidirectional)

