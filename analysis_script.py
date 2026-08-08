import pandas as pd
import matplotlib.pyplot as plt

df_lsh1 = pd.read_csv("experiments/lsh_experiments_results1.csv")
df_lsh2 = pd.read_csv("experiments/lsh_experiments_results2.csv")
df_lsh = pd.concat([df_lsh1, df_lsh2], ignore_index=True)
df_lsh["Algorithm"] = "Euclidean_LSH"

df_cube = pd.read_csv("experiments/hypercube_experiments_results.csv")
df_cube = df_cube[df_cube["QPS"] <= 2000]
df_cube["Algorithm"] = "Hypercube"

df_ivfflat = pd.read_csv("experiments/ivfflat_experiments_results.csv")
df_ivfflat["Algorithm"] = "IVFFlat"

df_ivfpq = pd.read_csv("experiments/ivfpq_experiments_results.csv")
df_ivfpq["Algorithm"] = "IVFPQ"

df_nlsh = pd.read_csv("experiments/neural_experiments_results.csv")
df_nlsh["Algorithm"] = "Neural_LSH"

df = pd.concat([
    df_lsh,
    df_cube,
    df_ivfflat,
    df_ivfpq,
    df_nlsh
], ignore_index=True)

algorithms = df["Algorithm"].unique()

for alg in algorithms:
    subset = df[df["Algorithm"] == alg]

    plt.figure(figsize=(8,6))
    plt.scatter(subset["QPS"], subset["Recall@N"])
    plt.title(f"Recall@N vs QPS - {alg}")
    plt.xlabel("QPS")
    plt.ylabel("Recall@N")
    plt.grid(True)
    plt.savefig(f"images/{alg}_recall_vs_qps.png")