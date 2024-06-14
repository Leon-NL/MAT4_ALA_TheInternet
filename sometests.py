import networkx as nx
import matplotlib.pyplot as plt

# Function to dynamically add directed edges to a graph
def add_connections(graph, node, connected_nodes):
    for cn in connected_nodes:
        graph.add_edge(cn, node)

# Create a directed graph
G = nx.DiGraph()

# Add nodes and initial edges (example data)
G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 5), (4, 5), (4, 6)])

# Add connections dynamically (example: connect node 7 to nodes 1, 3, and 5)
add_connections(G, 7, [1, 3, 5])

# Calculate the degree of each node
degrees = dict(G.degree())
# Determine the scaling factor for node sizes
scaling_factor = 900

# Create node sizes based on the degree (number of connections)
node_sizes = [degrees[node] * scaling_factor for node in G.nodes()]

# Draw the graph with adjusted layout to minimize edge overlaps and spread out the nodes
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, k=1.5, seed=42)  # Increase the value of k for more spread out nodes
nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", edge_color="gray", font_size=10, font_weight="bold", arrows=True)

# Display the plot
plt.show()
