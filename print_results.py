import glob
from tabulate import tabulate

ALGORITHMS = {
    "lsh": "Euclidean LSH",
    "hypercube": "Hypercube",
    "ivfflat": "IVF-Flat",
    "ivfpq": "IVF-PQ",
    "nlsh": "Neural LSH"
}

def parse_algorithm_output(filename):
    method_key = filename.split("_")[0]
    method_name = ALGORITHMS.get(method_key, method_key)

    queries = {}
    current = None
    neighbor = {}
    rank = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Query:"):
                query_id = line.split(":")[1].strip()

                # Store previous query
                if current:
                    queries[current["query"]] = current
                
                current = {
                    "method": method_name,
                    "query": query_id,
                    "time_per_query": None,
                    "QPS": None,
                    "recall_at_n": None,
                    "neighbors": []
                }
                rank = 0
                continue
            
            if current is None:
                continue

            # Parse elements
            if line.startswith("Query Recall:"):
                current["recall_at_n"] = float(line.split(":")[1])

            if line.startswith("Query Time:"):
                time = float(line.split(":")[1])
                current["time_per_query"] = time
                current["QPS"] = 1.0 / time

            if line.startswith("Nearest neighbor"):
                rank += 1
                nid = line.split(":")[1].strip()
                neighbor = {"rank": rank, "id": nid}
            
            if line.startswith("distanceApproximate"):
                neighbor["l2"] = float(line.split(":")[1])

            if line.startswith("Blast Identity"):
                neighbor["blast_id"] = float(line.split(":")[1])

            if line.startswith("In BLAST top N"):
                neighbor["in_blast_top_n"] = line.split(":")[1].strip()
                current["neighbors"].append(neighbor)
                neighbor = {}
        
        # Save last query
        if current:
            queries[current["query"]] = current

    return queries


def load_all_results():
    all_queries = {}

    for file in glob.glob("*_results.txt"):
        method_results = parse_algorithm_output(file)

        for query_id, result in method_results.items():
            if query_id not in all_queries:
                all_queries[query_id] = {}
            all_queries[query_id][result["method"]] = result

    return all_queries


def print_report(all_queries, N, blast_total_time):
    output_lines = []

    for query_id, methods in all_queries.items():
        output_lines.append("\n============================================================================================================")
        output_lines.append(f"Query Protein: {query_id}")
        output_lines.append(f"N = {N}")

        # Summary table
        summary_table = []

        for method, q in methods.items():
            summary_table.append([
                method,
                f"{q['time_per_query']:.4f}",
                f"{q['QPS']:.2f}",
                f"{q['recall_at_n']:.3f}"
            ])

        # Add BLAST reference row
        num_queries = len(all_queries)
        blast_time_per_query = blast_total_time / num_queries
        blast_qps = num_queries / blast_total_time

        summary_table.append([
            "BLAST (Ref)",
            f"{blast_time_per_query:.4f}",
            f"{blast_qps:.2f}",
            "1.00"
        ])
        
        output_lines.append(tabulate(
            summary_table,
            headers=["Method", "Time/query (s)", "QPS", "Recall@N vs BLAST Top-N"],
            tablefmt="github"
        ))

        for method, q in methods.items():
            output_lines.append(f"\nMethod: {method}")

            neighbor_table = []
            for q in q["neighbors"][:N]:
                neighbor_table.append([
                    q['rank'],
                    q['id'],
                    f"{q['l2']:.4f}",
                    f"{q['blast_id']}%",
                    q['in_blast_top_n'],
                    "--"
                ])
            
            output_lines.append(tabulate(
                neighbor_table,
                headers=["Rank", "Neighbor ID", "L2 Dist", "BLAST Identity", "In BLAST Top-N?", "Bio comment"],
                tablefmt="github"
            ))
    
    return "\n".join(output_lines)