import networkx as nx
import torch
import pytest
import torch_geometric.utils as pyg_utils
from rtl_depth_prediction import extract_features, compute_metrics, GNNModel

@pytest.mark.parametrize("num_nodes, num_edges", [(50, 100), (100, 200)])
def test_large_graph(num_nodes, num_edges):
    """Test feature extraction for large RTL graphs."""
    G = nx.gnm_random_graph(num_nodes, num_edges, directed=True)
    gate_types = {node: "AND" for node in G.nodes()}  # Default all gates to "AND"
    
    features = extract_features(G, gate_types)

    assert features.shape[0] == num_nodes  # Ensure correct number of nodes
    assert isinstance(features, torch.Tensor)

def test_full_pipeline():
    """Integration test for feature extraction, GNN forward pass, and prediction."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    gate_types = {"A": "AND", "B": "OR", "C": "NAND", "D": "XOR"}

    # Extract features
    features = extract_features(G, gate_types)

    # Convert NetworkX graph to PyTorch Geometric format
    data = pyg_utils.from_networkx(G)  # Returns Data object
    edge_index = data.edge_index.long()  # Extract edge index

    # Ensure feature extraction is valid
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == len(G.nodes())

    # Initialize model dynamically based on extracted features
    model = GNNModel(in_channels=features.shape[1], hidden_channels=16, out_channels=1)

    # Run model inference without gradients
    with torch.no_grad():
        output = model(features, edge_index)

    # Check output shape
    assert output.shape == (features.shape[0], 1)

if __name__ == "__main__":
    pytest.main()
