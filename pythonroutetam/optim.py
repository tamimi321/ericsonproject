import osmnx as ox
import networkx as nx
import folium

def find_route(graph, origin_point, destination_point):
    """
    Finds the shortest route between two geographic points.

    :param graph: OSMnx graph.
    :param origin_point: (lat, lon) of the origin.
    :param destination_point: (lat, lon) of the destination.
    :return: Shortest path node list and distance.
    """
    origin_node = ox.distance.nearest_nodes(graph, origin_point[1], origin_point[0])
    destination_node = ox.distance.nearest_nodes(graph, destination_point[1], destination_point[0])
    
    try:
        path = nx.shortest_path(graph, origin_node, destination_node, weight="length")
        distance = nx.shortest_path_length(graph, origin_node, destination_node, weight="length")
        return path, distance
    except nx.NetworkXNoPath:
        return None, float('inf')

def plot_route_folium(graph, path):
    """
    Plots the route on an interactive Folium map.

    :param graph: OSMnx graph.
    :param path: List of node IDs in the route.
    :return: Folium map object.
    """
    # Get the coordinates of the route
    route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in path]
    
    # Create a Folium map centered on the starting point
    m = folium.Map(location=route_coords[0], zoom_start=13)
    
    # Add the route to the map
    folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)
    
    # Add markers for the start and end points
    folium.Marker(route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(route_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
    
    return m

# Define the location and download the road network graph
location_name = "Jakarta, Indonesia"
graph = ox.graph_from_place(location_name, network_type="drive")

# Define origin and destination points (latitude, longitude)
origin = (-6.200000, 106.816666)  # Example: Jakarta city center
destination = (-6.300000, 106.816666)  # Example: A location south of Jakarta

# Find the route
path, distance = find_route(graph, origin, destination)

if path:
    print(f"Shortest path nodes: {path}")
    print(f"Total distance: {distance:.2f} meters")
    
    # Create and save the interactive map
    route_map = plot_route_folium(graph, path)
    route_map.save("route_map.html")
    print("Route map saved as 'route_map.html'.")
else:
    print(f"No path found between {origin} and {destination}.")
