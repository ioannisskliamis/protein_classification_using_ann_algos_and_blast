import sys
import numpy as np
import torch
import esm
from read_dataset import read_fasta
from args_parser import parser


def main():

    # Parse and validate input
    args = parser()

    # Load pre-trained esm model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Read .fasta file and return a list of (protein_id, sequence)
    proteins = read_fasta(args.i)

    embeddings = []
    ids = []

    model.eval()
    # Generate embeddings
    with torch.no_grad():
        for pid, seq in proteins:
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
            embeddings.append(embedding.squeeze(0).cpu().numpy())
            ids.append(pid)

    embeddings = np.array(embeddings, dtype=np.float32) # Shape (N, 320)
    embeddings.tofile(args.o)

    # Save protein IDs
    with open("protein_ids.txt", "w") as f:
        for pid in ids:
            f.write(pid + "\n")

    print(embeddings.shape)


if __name__ == '__main__':
    sys.exit(main())