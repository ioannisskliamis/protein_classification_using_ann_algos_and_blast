import numpy as np
import os
import struct
from collections import defaultdict

def read_mnist(path):
	with open(path, 'rb') as f:
		data = f.read()

	magic = int.from_bytes(data[0:4], 'big')
	dims = magic & 0xFF
	shape = tuple(int.from_bytes(data[4+i*4:8+i*4], 'big') for i in range(dims))

	image_array = np.frombuffer(data[4+4*dims:], dtype=np.uint8).astype(np.float32)
	image_array = image_array.reshape(shape)
	image_array = image_array / 255.0

	return image_array


def read_sift(path):
	data = np.fromfile(path, dtype='<i4')

	dims = data[0]

	data = data.reshape(-1, dims + 1)

	# Drop the first int
	data = data[:, 1:].view('<f4')

	#Normalize dataset
	norms = np.linalg.norm(data, axis = 1, keepdims = True)
	data = data / norms

	return data


def read_fasta(path):
	data = []
	
	with open(path) as f:
		seq_id = None
		seq = ""
		for line in f:
			line = line.strip()
			if line.startswith(">"):
				# Save previous protein before starting a new one
				if seq_id is not None:
					data.append((seq_id, seq))
				seq_id = line[1:] # Remove the ">" from the header
				seq = ""
			else:
				# Append sequence content line
				seq += line
		
		# Add the last protein
		data.append((seq_id, seq))
	return data


def read_embeddings(path):
	return np.fromfile(path, dtype=np.float32).reshape(-1, 320)


def read_protein_id_map(filename):
	protein_id_index = {}

	with open(filename) as f:

		for idx, line in enumerate(f):
			line = line.strip()

			if line:
				protein_id_index[line] = idx # Map ID to index

	return protein_id_index


def read_query_id_map(fasta_file):
	query_id_index = {}
	idx = 0

	with open(fasta_file) as f:

		for line in f:
			line = line.strip()

			if line.startswith(">"):
				query_id_index[line[1:]] = idx # Map header to index
				idx += 1

	return query_id_index


def read_blast_tsv(filename, protein_id_index, query_id_index, N):
	blast_hits = defaultdict(list)
	threshold = 0.01

	with open(filename) as f:

		for line in f:
			
			if not line.strip():
				continue

			cols = line.split()
			qseqid = cols[0] # Query sequence ID
			sseqid = cols[1] # Subject sequence ID
			evalue = float(cols[10])

			# Filter based on e-value
			if evalue > threshold:
				continue

			# Skip IDs not found in maps
			if qseqid not in query_id_index:
				continue
			if sseqid not in protein_id_index:
				continue

			qidx = query_id_index[qseqid]
			pidx = protein_id_index[sseqid]

			# Store up to N hits
			if len(blast_hits[qidx]) < N:
				blast_hits[qidx].append(pidx)

	return blast_hits


def read_blast_identities(filename, protein_id_index, query_id_index):
	blast_identities = defaultdict(dict)

	with open(filename) as f:

		for line in f:

			if not line.strip():
				continue

			cols = line.split()
			qseqid = cols[0]
			sseqid = cols[1]
			evalue = float(cols[10])
			pident = float(cols[2]) # Identity %

			# Skip IDs not found in maps
			if qseqid not in query_id_index:
				continue
			if sseqid not in protein_id_index:
				continue

			qidx = query_id_index[qseqid]
			pidx = protein_id_index[sseqid]

			blast_identities[qidx][pidx] = pident

	return blast_identities