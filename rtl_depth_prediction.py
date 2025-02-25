import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import json
import matplotlib.pyplot as plt

# --- Feature Extraction ---
def extract_features(G, gate_types):
    """ Extracts graph-based features like centrality, fan-in/fan-out, and gate types."""
    if not nx.is_connected(G.to_undirected()):
        print("⚠️ Graph is disconnected. Using degree centrality instead.")
        centrality = nx.degree_centrality(G)
    else:
        centrality = nx.betweenness_centrality(G)  
    
    fan_in = {node: len(list(G.predecessors(node))) for node in G.nodes()}
    fan_out = {node: len(list(G.successors(node))) for node in G.nodes()}
    
    gate_type_encoding = {"AND": 1, "OR": 2, "NOT": 3, "NAND": 4, "NOR": 5, "XOR": 6, "XNOR": 7, "INPUT": 0}
    
    features = []
    for node in G.nodes():
        gate_type = gate_types.get(node, "INPUT")  # Default to "INPUT" if missing
        features.append([centrality.get(node, 0), fan_in.get(node, 0), fan_out.get(node, 0), gate_type_encoding.get(gate_type, 0)])
    
    return torch.tensor(features, dtype=torch.float)

# --- Hybrid GNN + Heuristics Model ---
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# --- Compute MAE and RMSE ---
def compute_metrics(y_true, y_pred):
    """ Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). """
    mae = torch.mean(torch.abs(y_true - y_pred))
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return mae.item(), rmse.item()

# --- Training the Model ---
def train_model(G, depth_dict, gate_types, setup_time, hold_time, clock_period):
    features = extract_features(G, gate_types)
    
    # Convert node names to indices
    node_to_index = {node: i for i, node in enumerate(G.nodes())}
    edges = [(node_to_index[src], node_to_index[dst]) for src, dst in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    y = torch.tensor([depth_dict.get(node, 0) for node in G.nodes()], dtype=torch.float).view(-1, 1)
    
    data = Data(x=features, edge_index=edge_index)
    model = GNNModel(in_channels=4, hidden_channels=16, out_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        # Calculate MAE and RMSE for the current epoch
        mae, rmse = compute_metrics(y, out)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    
    print("✅ Model training complete!")
    
    # Output predicted depths and check for timing violations
    predicted_depths = out.detach().numpy().flatten()
    for node, depth in zip(G.nodes(), predicted_depths):
        print(f"Gate {node}: Predicted Combinational Depth = {depth:.2f}")
        if depth * setup_time > clock_period:
            print(f"⚠️ Timing Violation at {node}: Depth {depth:.2f} exceeds clock period {clock_period}")
    
    # Final MAE and RMSE after training
    final_mae, final_rmse = compute_metrics(y, out)
    print(f"\nFinal Metrics: MAE = {final_mae:.4f}, RMSE = {final_rmse:.4f}")
    
    # Save graph visualization
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold', edge_color='gray')
    plt.title("RTL Netlist Graph Visualization")
    plt.savefig("rtl_netlist_graph.png")  # Save the plot to a file
    plt.close()  # Close the plot
    
    print("Graph saved as 'rtl_netlist_graph.png'.")
    
    return model

# --- Main Execution ---
if __name__ == "__main__":
    rtl_file = input("Enter RTL file path: ")
    depth_report = input("Enter combinational depth report file path: ")
    
    setup_time = float(input("Enter setup time constraint: "))
    hold_time = float(input("Enter hold time constraint: "))
    clock_period = float(input("Enter clock period constraint: "))
    
    G = nx.DiGraph()
    depth_dict = {}
    gate_types = {}
    
    # Load RTL Netlist (Example: parsing from JSON format)
    with open(rtl_file, 'r') as f:
        netlist = json.load(f)
        for gate in netlist:
            G.add_node(gate["name"], type=gate["type"])
            gate_types[gate["name"]] = gate["type"]
            for connection in gate["connections"]:
                if connection not in G.nodes():  # Ensure input nodes exist
                    G.add_node(connection, type="INPUT")
                G.add_edge(connection, gate["name"])
    
    # Load Depth Report
    with open(depth_report, 'r') as f:
        depth_dict = json.load(f)
    
    model = train_model(G, depth_dict, gate_types, setup_time, hold_time, clock_period)
    print("✅ Prediction model ready to use!")
