# rtl-depth-prediction
# RTL Netlist Depth Prediction using Graph Neural Networks

## 📌 Project Overview
This project utilizes *Graph Neural Networks (GNNs)* to predict the *combinational depth* of gates in an *RTL (Register Transfer Level) netlist. The system extracts **graph-based features* and applies *hybrid GNN + heuristic* techniques to analyze and predict the circuit timing.

## 🚀 Features
- *Graph-Based Feature Extraction*: Computes centrality, fan-in, fan-out, and gate type encoding.
- *Hybrid GNN Model: Implements a two-layer **Graph Convolutional Network (GCN)*.
- *Training and Evaluation: Uses **MSE Loss* optimization with *Adam optimizer*.
- *Timing Analysis: Predicts gate depths and checks for **timing violations*.
- *Visualization: Generates a **graph representation* of the RTL netlist.

## 📂 File Structure

├── rtl_depth_prediction.py  # Main script for feature extraction & training
├── test_rtl_depth.py        # Unit and integration tests
├── rtl_netlist_graph.png    # Generated visualization of RTL graph
└── README.md                # Project documentation


## 🛠 Dependencies
Ensure you have the following installed before running the project:

bash
pip install networkx torch torch-geometric numpy matplotlib pytest


## 🎯 Usage
### 1️⃣ Running the Model
Execute the script and provide the required file paths and constraints:
bash
python rtl_depth_prediction.py

*Inputs Required:*
- RTL netlist file (JSON format)
- Combinational depth report file (JSON format)
- Setup time, hold time, and clock period constraints

### 2️⃣ Running Tests
To validate feature extraction and GNN performance:
bash
pytest test_rtl_depth.py


## 📊 Results & Output
- *Training Logs*: Displays loss, MAE, RMSE for every 10 epochs.
- *Predicted Gate Depths*: Outputs computed depth values per gate.
- *Timing Violations*: Alerts if any gate violates the clock period.
- *Graph Visualization*: Saves the RTL netlist graph as rtl_netlist_graph.png.

## 🏆 Performance Metrics
- *Mean Absolute Error (MAE)*: Measures absolute prediction accuracy.
- *Root Mean Square Error (RMSE)*: Evaluates the deviation from true values.

## 📌 Future Improvements
- Implement *attention-based GNNs* (e.g., GAT) for better feature learning.
- Introduce *hyperparameter tuning* for improved model accuracy.
- Extend support for *multi-clock domain circuits*.