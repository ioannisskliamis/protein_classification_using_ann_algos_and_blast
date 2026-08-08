import numpy as np
import torch
from feedforward_nn import FeedForwardNN
from dataset import TrainDataset,TestDataset
from torch.utils.data import Dataset, DataLoader
from read_dataset import read_mnist, read_sift, read_embeddings, read_blast_tsv, read_query_id_map, read_protein_id_map, read_blast_identities
from args_parser import parser
import sys
import json
import time

def main():
	args = parser()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Read model parameters
	with open(f"{args.i}/model_params.json", "r") as f:
		model_params = json.load(f)

	layers = model_params["layers"]
	classes = model_params["classes"]
	nodes = model_params["nodes"]
	input_size = model_params["input"]

	# Create model using saved parameters
	model = FeedForwardNN(input_size, classes, layers, nodes).to(device)

	# Load weights of trained model
	state = torch.load(f"{args.i}/model.pth", map_location = device)
	model.load_state_dict(state)

	if args.type == 'mnist':
		# Read input images
		input_images = read_mnist(args.d)
		# Read query images and create Dataset
		query_images = read_mnist(args.q)
	elif args.type == 'sift':
		# Read input images
		input_images = read_sift(args.d)
		# Read query images and create Dataset
		query_images = read_sift(args.q)
	elif args.type == 'protein':
		# Read input images
		input_images = read_embeddings(args.d)
		# Read query images and create Dataset
		query_images = read_embeddings(args.q)
	
	query_dataset = TestDataset(query_images)
	test_loader = DataLoader(query_dataset, shuffle = False)

	# Evaluate model using query images set
	model.eval()
	predictions = []

	with torch.no_grad():
		for image in test_loader:
			image = image.to(device)
			outputs = model(image)
			preds = torch.softmax(outputs, dim = 1)
			predictions.append(preds)

	predictions = torch.cat(predictions, dim = 0)
	print(predictions.shape)

	# Find top T parts according to Neural Network
	T = args.T
	N = args.N
	top_T_probs, top_T_parts = torch.topk(predictions, T, dim=1)
	print(top_T_probs)
	print(top_T_parts)

	file_path = f"{args.i}/inverted_file.json"
	with open(file_path, "r") as f:
		inverted_index = json.load(f)

	output = args.o
	method_name = "Neural LSH"
	with open(output, "w") as f:
		f.write(f"{method_name}\n")

	approx_time = 0.0
	true_time = 0.0
	total_AF = 0.0
	total_recall = 0.0

	protein_id_index = read_protein_id_map("protein_ids.txt")
	query_id_index = read_query_id_map("targets.fasta")

	protein_ids = [None] * len(protein_id_index)

	for pid, idx in protein_id_index.items():
		protein_ids[idx] = pid

	query_ids = [None] * len(query_id_index)

	for pid, idx in query_id_index.items():
		query_ids[idx] = pid

	blast_hits = read_blast_tsv("blast_results.tsv", protein_id_index, query_id_index, args.N)
	blast_identities = read_blast_identities("blast_results.tsv", protein_id_index, query_id_index)

  	# For every query image check the images to find euclidean distance
	for id in range(len(query_images)):
		start = time.time()
		hits_per_query = 0

		top_T = top_T_parts[id]
		top_T = top_T.cpu().numpy()
		images_to_check = []
		cand_images = []

  		# Read parts of partition
		for t in top_T:
			part_string = str(t)
			candidates = inverted_index[part_string]
			for cand in candidates:
				images_to_check.append(cand)
		for cand in images_to_check:
			cand_images.append(input_images[cand])

		# Find distances
		if args.type == 'mnist':
			dists = np.linalg.norm(cand_images - query_images[id], axis = (1, 2))
		elif args.type in ('sift', 'protein'):
			dists = np.linalg.norm(cand_images - query_images[id], axis = 1)
		nei_dict = dict(zip(images_to_check, dists))

		dists = np.array(list(nei_dict.values()))
		ids = np.array(list(nei_dict.keys()))

		# Get top N Neighbors
		top_nei_ids = np.argpartition(dists, N)[: N]

		# Sort them
		top_nei_sorted = top_nei_ids[np.argsort(dists[top_nei_ids])]

		# Get ids and distances
		top_N = ids[top_nei_sorted]
		top_N_dists = dists[top_nei_sorted]

		end = time.time()
		time_elapsed = end - start
		approx_time += time_elapsed

		# Perform range search on candidate points
		if args.range:
			range_neighbors_ids = ids[dists <= args.R]

		start1 = time.time()

		# Finding true neighbors
		if args.type == 'mnist':
			true_dists = np.linalg.norm(input_images - query_images[id], axis = (1, 2))
		elif args.type in ('sift', 'protein'):
			true_dists = np.linalg.norm(input_images - query_images[id], axis = 1)
		true_nei_dict = dict(zip(range(len(true_dists)), true_dists))

		dists = np.array(list(true_nei_dict.values()))
		true_ids = np.array(list(true_nei_dict.keys()))

		# Get top N Neighbors
		true_nei_ids = np.argpartition(true_dists, N)[: N]

		# Sort them
		true_nei_sorted = true_nei_ids[np.argsort(true_dists[true_nei_ids])]

		# Get ids and distances
		true_N = true_ids[true_nei_sorted]
		true_N_dists = true_dists[true_nei_sorted]

		end1 = time.time()
		time_elapsed_true = end1 - start1
		true_time += time_elapsed_true
		
		total_AF += top_N_dists[0] / true_N_dists[0]

		blast_ids = blast_hits.get(id, [])

		if len(blast_ids) == 0:
			total_recall += 0.0
			continue

		blast_set = set(blast_ids[:min(N, len(blast_ids))])

		hits_per_query = 0
		for k in range(min(N, len(top_N))):
			if top_N[k] in blast_set:
				hits_per_query += 1

		query_recall = hits_per_query / min(N, len(blast_set))
		total_recall += query_recall

		with open(output, "a") as f:
			f.write(f"Query: {query_ids[id]}\n")
			f.write(f"Query Recall: {query_recall}\n")
			f.write(f"Query Time: {time_elapsed}\n")
			for i in range(len(top_N)):
				neighbor_idx = top_N[i]
				neighbor_id = protein_ids[neighbor_idx]
				identity = blast_identities.get(id, {}).get(neighbor_idx, 0.0)
				if neighbor_idx in blast_set:
					inN = "Yes"
				else:
					inN = "No"
				f.write(f"Nearest neighbor-{i+1}: {neighbor_id}\n")
				f.write(f"distanceApproximate: {top_N_dists[i]}\n")
				f.write(f"Blast Identity: {identity}\n")
				f.write(f"In BLAST top N: {inN}\n")
			f.write(f"R-near neighbors:\n")
			if args.range and len(range_neighbors_ids) > 0:
				for rid in range_neighbors_ids:
					f.write(f"{rid}\n")

	average_AF = total_AF / len(query_images)
	recall_at_N = total_recall / len(query_images)
	qps = len(query_images) / approx_time
	total_approx_time = approx_time / len(query_images)
	total_true_time = true_time / len(query_images)

	with open(output, "a") as f:
		f.write(f"Average AF: {average_AF}\n")
		f.write(f"Recall@N: {recall_at_N}\n")
		f.write(f"QPS: {qps}\n")
		f.write(f"tApproximateAverage: {total_approx_time}\n")
		f.write(f"tTrueAverage: {total_true_time}")

	return 0


if __name__ == '__main__':
    sys.exit(main())