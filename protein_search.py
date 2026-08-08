import sys
import numpy as np
import torch
import esm
import subprocess
from args_parser import parser
from read_dataset import read_fasta
from algo_runner import algo_runner
from print_results import load_all_results, print_report
from time import time

def main():
    
    # Parse and validate input
    args = parser()

    # Read query file
    queries = read_fasta(args.q)

    # Load pre-trained esm model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    query_embeddings = []
    query_ids = []

    model.eval()
    # Generate embeddings
    with torch.no_grad():
        for pid, seq in queries:
            # Truncate long sequences
            if len(seq) > 1022:
                seq = seq[:1022]

            data = [(pid, seq)]
            # Convert sequences to tokens
            labels, strs, tokens = batch_converter(data)
            tokens = tokens.to(device)

            out = model(tokens, repr_layers=[6])

            # Extract representations from the last layer
            token_embeddings = out["representations"][6]

            # Mean pooling over sequence length to get one vector per protein
            embedding = token_embeddings.mean(dim=1)

            # Store as numpy arrays
            query_embeddings.append(embedding.squeeze(0).cpu().numpy())
            query_ids.append(pid)

    # Save query embeddings
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    query_embeddings.tofile("queries.bin") # Shape (Q, 320)

    # Create db
    subprocess.run(["makeblastdb", "-in", "swissprot_50k.fasta", "-dbtype", "prot", "-out", "swissprot_db"])

    start = time()
    # Create results file
    subprocess.run(["blastp", "-db", "swissprot_db", "-query", "targets.fasta", "-outfmt", "6", "-out", "blast_results.tsv"])
    blast_total_time = time() - start

    # Execute ANN
    algo_runner(args.d, "queries.bin", args.method)
    
    # Parse ANN output
    all_queries = load_all_results()
    report = print_report(all_queries, 20, blast_total_time) # For 20 neighbors per query

    # write results.txt
    with open(args.o, "w") as f:
        f.write(report)

if __name__ == '__main__':
    sys.exit(main())