import sys
import kahip
import json
from args_parser import parser
from graph_builder import run_ivfflat, parse_knn_file, build_weighted_graph, create_kahip_format
from feedforward_nn import FeedForwardNN
from set_seed import set_seed
from dataset import TrainDataset,TestDataset
from read_dataset import read_mnist, read_sift, read_embeddings
from set_seed import set_seed
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from train import train

def main():
    # Parse and validate input
    args = parser()

    # Read dataset based on its type
    if args.type == 'mnist':
        images = read_mnist(args.d)
    elif args.type == 'sift':
        images = read_sift(args.d)
    elif args.type == 'protein':
        images = read_embeddings(args.d)

    #Set seed
    set_seed(args.seed)

    # Run 1st project executable
    run_ivfflat(args)
    
    knn_results = parse_knn_file(f"{args.i}/ivfflat.txt")
    weighted_graph = build_weighted_graph(knn_results)
    kahip_format = create_kahip_format(weighted_graph)
    # Load kahip
    edgecut, blocks = kahip.kaffpa(
        kahip_format["vwgt"],
        kahip_format["xadj"],
        kahip_format["adjcwgt"],
        kahip_format["adjncy"],
        args.m,
        args.imbalance,
        False,
        args.seed,
        args.kahip_mode
    )
    # Initialization of inverted index
    inverted_index = {r: [] for r in range(args.m)}

    for node_id, partition_label in enumerate(blocks):
        # Add point to corresponding label
        inverted_index[partition_label].append(node_id)

    # Save inverted index
    filename = f"{args.i}/inverted_file.json"
    try:
        with open(filename, 'w') as f:
            json.dump(inverted_index, f, indent=4)
    except Exception as e:
        print("Failed saving inverted file")

    # Check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create Dataset and DataLoaders
    dataset = TrainDataset(images, blocks)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))], generator = g)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, generator = g)
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)

    # Create Neural Network
    model = FeedForwardNN(input_size = images[0].size, classes = args.m, layers = args.layers, nodes = args.nodes).to(device)

    # Saving the parameters of the model
    model_params = {
        "layers": args.layers,
        "classes": args.m,
        "nodes": args.nodes,
        "input": images[0].size
    }

    filename = f"{args.i}/model_params.json"
    try:
        with open(filename, 'w') as f:
            json.dump(model_params, f, indent=4)
    except Exception as e:
        print("Failed saving parameters of model")

    # Optimizer and Loss Function
    optim = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    train(model, optim, loss_fn, train_dataloader, val_dataloader, device, images, blocks, args.lr, args.epochs, args.batch_size, args.i)


    return 0

if __name__ == '__main__':
    sys.exit(main())