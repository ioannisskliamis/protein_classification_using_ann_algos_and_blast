import subprocess
import re
from collections import defaultdict
import numpy as np

# Funciton to compile and run ivfflat algorithm
def run_ivfflat(args):
    # Compile
    subprocess.run(["make"], check=True)
    if args.type == 'mnist':
        # Execution command
        cmd = [
            "./search",
            "-d", args.d,
            "-q", args.d,
            "-o", f"{args.i}/ivfflat.txt",
            "-type", args.type,
            "-ivfflat",
            "-graph", # Custom parameter to skip brute force and metrics
            "-kclusters", "22", # Optimal params for highest recall kclusters=22, nprobes=5(default)
            "-N", str(args.knn),
            "-seed", str(args.seed)
        ]
    elif args.type == 'sift':
        # Execution command
        cmd = [
            "./search",
            "-d", args.d,
            "-q", args.d,
            "-o", f"{args.i}/ivfflat.txt",
            "-type", args.type,
            "-ivfflat",
            "-graph", # Custom parameter to skip brute force and metrics
            "-kclusters", "100", # Optimal params for highest recall and fastest build time kclusters=100, nprobes=5(default)
            "-N", str(args.knn),
            "-seed", str(args.seed)
        ]
    elif args.type == 'protein':
        # Execution command
        cmd = [
            "./search",
            "-d", args.d,
            "-q", args.d,
            "-o", f"{args.i}/ivfflat.txt",
            "-type", args.type,
            "-ivfflat",
            "-graph", # Custom parameter to skip brute force and metrics
            "-kclusters", "22", # Optimal params for highest recall kclusters=22, nprobes=5(default)
            "-N", str(args.knn),
            "-seed", str(args.seed)
        ]

    subprocess.run(cmd, check=True)


# Function to parse results of ivfflat
def parse_knn_file(path):
    neighbors = defaultdict(list)
    current = None

    with open(path) as f:
        for line in f:
            q = re.match(r"Query:\s+(\d+)", line)
            if q:
                current = int(q.group(1))
                continue
            
            n = re.match(r"Nearest neighbor-\d+:\s+(\d+)", line)
            if n:
                neighbors[current].append(int(n.group(1))) # Append the neighbor_id

    return neighbors


# Function to build the weighted undirected graph
def build_weighted_graph(neighbors):
    graph = defaultdict(dict)

    # Add all edges with weight 1
    for i, nbrs in neighbors.items():
        for j in nbrs:
            graph[i][j] = 1 # Temporarily
            graph[j] # Ensure it exists
    
    # Update mutual to weight 2
    for i, nbrs in neighbors.items():
        for j in nbrs:
            if i in neighbors.get(j, []): # mutual
                graph[i][j] = 2
                graph[j][i] = 2
            else:
                # Keep weight 1 if non mutual
                if graph[j].get(i, 0) < 1:
                    graph[j][i] = 1
                    # graph[i][j] is already set to 1
    
    return graph


# Function to create kahip format
def create_kahip_format(graph):
    # Initialization
    sorted_nodes = sorted(graph.keys())

    # KaHIP lists
    xadj = [0]
    adjncy = []
    adjcwgt = []

    current_degree_sum = 0

    # Iterate through nodes
    for node_id in sorted_nodes:
        # Get neighbors and weights of current node
        neighbors_dict = graph[node_id]

        # Sort by ID
        sorted_neighbors = sorted(neighbors_dict.items(), key=lambda item: item[0])

        for neighbor_id, weight in sorted_neighbors:
            adjncy.append(neighbor_id)
            adjcwgt.append(weight)
        
        current_degree_sum += len(sorted_neighbors)
        xadj.append(current_degree_sum)

    # All nodes usually have weight 1
    vwgt = [1] * len(sorted_nodes)

    # Convert to numpy arrays
    vwgt = np.array(vwgt, dtype=np.int32)
    xadj = np.array(xadj, dtype=np.int32)
    adjcwgt = np.array(adjcwgt, dtype=np.int32)
    adjncy = np.array(adjncy, dtype=np.int32)

    return {"xadj": xadj, "adjncy": adjncy, "adjcwgt": adjcwgt, "vwgt": vwgt, "num_nodes": len(sorted_nodes)}