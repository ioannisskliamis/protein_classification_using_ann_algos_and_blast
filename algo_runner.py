import subprocess

# Optimal parameters
K_LSH = 6 ; L_LSH = 12 ; W_LSH = 5.0

K_PROJ = 14 ; M_CUBE = 10000 ; PROBES = 14 ; W_CUBE = 4.0

K_CLUSTERS_FLAT = 200 ; N_PROBE_FLAT = 20

K_CLUSTERS_PQ = 200 ; N_PROBE_PQ = 20 ; M_PQ = 80 ; N_BITS = 8

N_GRAPH = 20 ; LAYERS = 7 ; NODES = 512 ; LR = 0.001 ; EPOCHS = 30 ; BATCH_SIZE = 128 ; M_NEURAL = 100 ; T_NEURAL = 10

INDEX_PATH = "nlsh_index"

N = 20



def algo_runner(input, query, method):
    commands_to_run = [['make']]

    if method in ['lsh', 'all']:
        commands_to_run.append([
            "./search", "-d", input, "-q", query, "-o", "lsh_results.txt", "-k", str(K_LSH),
            "-L", str(L_LSH), "-w", str(W_LSH), "-N", str(N), "-type", "protein", "-range", "false", "-lsh"
        ])
    
    if method in ['hypercube', 'all']:
        commands_to_run.append([
            "./search", "-d", input, "-q", query, "-o", "hypercube_results.txt", "-kproj", str(K_PROJ),
            "-M", str(M_CUBE), "-w", str(W_CUBE), "-probes", str(PROBES), "-N", str(N), "-type", "protein", "-range", "false", "-hypercube"
        ])

    if method in ['ivfflat', 'all']:
        commands_to_run.append([
            "./search", "-d", input, "-q", query, "-o", "ivfflat_results.txt", "-kclusters", str(K_CLUSTERS_FLAT),
            "-nprobe", str(N_PROBE_FLAT), "-N", str(N), "-type", "protein", "-range", "false", "-ivfflat"
        ])

    if method in ['ivfpq', 'all']:
        commands_to_run.append([
            "./search", "-d", input, "-q", query, "-o", "ivfpq_results.txt", "-kclusters", str(K_CLUSTERS_PQ),
            "-nprobe", str(N_PROBE_PQ), "-M", str(M_PQ), "-nbits", str(N_BITS), "-N", str(N), "-type", "protein", "-range", "false", "-ivfpq"
        ])

    if method in ['neural', 'all']:
        commands_to_run.append([
            "python3", "./nlsh_build.py", "-d", input, "-i", INDEX_PATH, "--kn", str(N_GRAPH), "--layers", str(LAYERS), "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE), "--lr", str(LR), "-m", str(M_NEURAL), "-type", "protein"
        ])

        commands_to_run.append([
            "python3", "./nlsh_search.py", "-d", input, "-q", query, "-i", INDEX_PATH, "-o", "nlsh_results.txt", "-N", str(N), "-T", str(T_NEURAL), 
            "-type", "protein", "-range", "false"
        ])

    for cmd in commands_to_run:
        subprocess.run(cmd, check=True)
