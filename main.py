import streamlit as st
from typing import List
import numpy as np
from sklearn.mixture import GaussianMixture
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

geolocator = Nominatim(user_agent="travel_route_optimizer")

# get coordinates using OpenStreetMap Nominatim geocoding
def get_coordinates(destination):
    location = geolocator.geocode(destination)
    if location:
        return {"lat": location.latitude, "lon": location.longitude}
    else:
        return None

# Function to cluster destinations using Gaussian Mixture Model (GMM)
def cluster_destinations_gmm(destinations: List[str], n_clusters: int):
    """
    Cluster the given destinations using Gaussian Mixture Model.
    :param destinations: List of destination names.
    :param n_clusters: Number of clusters to form.
    :return: Cluster labels for each destination.
    """
    destinations_coords = [get_coordinates(dest) for dest in destinations if get_coordinates(dest)]
    data = np.array([[d["lat"], d["lon"]] for d in destinations_coords])

    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(data)

    return labels

# Function to calculate distance between two points using Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)*2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers
    r = 6371.0
    return c * r

# Function to solve TSP using OR-Tools
def solve_tsp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)
    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))

    return route

# Streamlit app code
def main():
    st.title("Travel Route Optimizer")

    # Get user input for destinations
    destinations_input = st.text_input("Enter destinations (comma-separated):")
    destinations = [dest.strip() for dest in destinations_input.split(",") if dest.strip()]

    # Cluster destinations based on user input
    if destinations:
        n_clusters = st.slider("Number of Clusters:", min_value=1, max_value=10, value=3)
        cluster_labels = cluster_destinations_gmm(destinations, n_clusters)

        # Display map with clusters and city names
        display_map_with_clusters(destinations, cluster_labels)

# Function to display map with clustered destinations and city names
def display_map_with_clusters(destinations: List[str], cluster_labels: np.ndarray):
    # Fetch coordinates for the first destination for map centering
    center_coords = get_coordinates(destinations[0])

    if center_coords:
        m = folium.Map(location=[center_coords["lat"], center_coords["lon"]], zoom_start=8)
    else:
        m = folium.Map(location=[35.0, 135.0], zoom_start=6)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    # Count cities in each cluster
    cluster_counts = {label: 0 for label in set(cluster_labels)}
    for label in cluster_labels:
        cluster_counts[label] += 1

    max_cluster = max(cluster_counts, key=cluster_counts.get)
    max_cluster_cities = [destinations[i] for i, label in enumerate(cluster_labels) if label == max_cluster]

    for dest, label in zip(destinations, cluster_labels):
        # Fetch coordinates for each destination
        dest_coords = get_coordinates(dest)
        if dest_coords:
            color = colors[label % len(colors)]
            folium.Marker(
                location=[dest_coords["lat"], dest_coords["lon"]],
                popup=f"{dest} (Cluster {label+1})",
                icon=folium.Icon(color=color)
            ).add_to(m)

    folium_static(m)

    # Display city names based on clusters
    st.subheader("Cities by Clusters:")
    for cluster_num in range(max(cluster_labels) + 1):
        cluster_cities = [destinations[i] for i, label in enumerate(cluster_labels) if label == cluster_num]
        st.write(f"Cluster {cluster_num + 1}: {', '.join(cluster_cities)}")

    # Display cluster with maximum cities
    st.subheader(f"Cluster with Maximum Cities: {max_cluster + 1}")
    st.write(f"Cities in Cluster {max_cluster + 1}: {', '.join(max_cluster_cities)}")

    # Calculate distance matrix for cities in max cluster
    max_cluster_coords = [get_coordinates(city) for city in max_cluster_cities]
    distance_matrix = np.zeros((len(max_cluster_cities), len(max_cluster_cities)))
    for i in range(len(max_cluster_cities)):
        for j in range(len(max_cluster_cities)):
            distance_matrix[i][j] = haversine(
                max_cluster_coords[i]["lon"], max_cluster_coords[i]["lat"],
                max_cluster_coords[j]["lon"], max_cluster_coords[j]["lat"]
            )

    #TSP
    tsp_route_indices = solve_tsp(distance_matrix)
    tsp_route = [max_cluster_cities[i] for i in tsp_route_indices]

    #route
    st.subheader("Optimal Route for Cluster with Maximum Cities:")
    st.write(" -> ".join(tsp_route))

if _name_ == "_main_":
    main()