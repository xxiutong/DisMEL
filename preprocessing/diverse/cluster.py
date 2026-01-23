import json
import numpy as np
from tqdm import tqdm
import metis
import random


# =========================================================
# 1. Load train data
# =========================================================
def load_train_data(train_json_path):
    print(f"Loading train data: {train_json_path}")
    data = json.load(open(train_json_path, "r", encoding="utf-8"))
    print(f"Loaded {len(data)} samples")
    return data


# =========================================================
# 2. Load N×N rank matrix
# =========================================================
def load_rank_matrix(rank_path):
    print(f"Loading rank matrix: {rank_path}")
    rank_mat = np.load(rank_path)
    print(f"Rank matrix shape: {rank_mat.shape}")
    return rank_mat


# =========================================================
# 3. Build qid -> indices map
# =========================================================
def build_qid_index_map(train_data):
    qid2indices = {}
    for idx, item in enumerate(train_data):
        qid = item["answer"]
        qid2indices.setdefault(qid, []).append(idx)
    return qid2indices


# =========================================================
# 4. Find top-K hard negative qids for each sample
# =========================================================
def find_topk_hardneg_qids(train_data, rank_mat, K=5):
    """
    """
    N = len(train_data)
    neg_qids_list = [[] for _ in range(N)]

    print(f"Finding top-{K} hard negative qids from rank matrix...")

    for i in tqdm(range(N)):
        pos_qid = train_data[i]["answer"]
        rank_list = rank_mat[i]

        for ptr in range(len(rank_list)):
            cand_idx = int(rank_list[ptr])
            cand_qid = train_data[cand_idx]["answer"]

            if cand_qid != pos_qid:
                neg_qids_list[i].append(cand_qid)
                if len(neg_qids_list[i]) >= K:
                    break

    return neg_qids_list


# =========================================================
# 5. Build multi-hard-negative adjacency graph
# =========================================================
def build_hardneg_adjacency_multi(train_data, neg_qids_list, qid2indices):
    """
    Edge: i <-> j
    where answer(j) in neg_qids_list[i]
    """
    n = len(train_data)
    adjacency = [[] for _ in range(n)]

    print("Building multi-hard-negative adjacency graph...")

    for i, neg_qids in tqdm(enumerate(neg_qids_list), total=n):
        for neg_qid in neg_qids:
            candidates = qid2indices.get(neg_qid, [])
            if len(candidates) == 0:
                continue

            j = random.choice(candidates)

            if j != i:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


# =========================================================
# 6. METIS partition
# =========================================================
def metis_partition(adjacency_list, n_clusters):
    print(f"Running METIS partition into {n_clusters} clusters...")
    edgecut, parts = metis.part_graph(adjacency_list, n_clusters)
    print(f"METIS finished. edgecut = {edgecut}")
    return parts


# =========================================================
# 7. Fix cluster sizes to cluster_size
# =========================================================
def fix_cluster_sizes(parts, cluster_size):
    clusters = {}
    for idx, cid in enumerate(parts):
        clusters.setdefault(cid, []).append(idx)

    cluster_lists = sorted(clusters.values(), key=len, reverse=True)

    final_clusters = []
    remaining = []

    for cluster in cluster_lists:
        while len(cluster) > cluster_size:
            remaining.append(cluster.pop())

        if len(cluster) < cluster_size:
            need = cluster_size - len(cluster)
            take = min(need, len(remaining))
            for _ in range(take):
                cluster.append(remaining.pop())
            need -= take

            if need > 0:
                base = cluster.copy()
                for k in range(need):
                    cluster.append(base[k % len(base)])

        final_clusters.append(cluster)

    if remaining:
        idx = 0
        while idx < len(remaining):
            chunk = remaining[idx: idx + cluster_size]
            idx += cluster_size
            if len(chunk) < cluster_size:
                base = chunk.copy()
                for k in range(cluster_size - len(chunk)):
                    chunk.append(base[k % len(base)])
            final_clusters.append(chunk)

    assert all(len(c) == cluster_size for c in final_clusters)
    print(f"Total clusters = {len(final_clusters)}")
    return final_clusters


# =========================================================
# 8. Reorder JSON
# =========================================================
def make_ordered_indices(final_clusters):
    return [idx for cluster in final_clusters for idx in cluster]


def reorder_json(train_json_path, ordered_indices, out_path):
    data = json.load(open(train_json_path, "r", encoding="utf-8"))
    new_data = [data[i] for i in ordered_indices]
    json.dump(
        new_data,
        open(out_path, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"Saved reordered dataset → {out_path}")


# =========================================================
# 9. Main
# =========================================================
if __name__ == "__main__":

    # ---------- paths ----------
    train_json = "/data/zyt/datas4/WikiDiverse/WikiDiverse_train_with_boxes.json"
    rank_path = "/data/zyt/datas4/WikiDiverse/mel_pred_rank.npy"
    out_path = "/data/zyt/datas4/WikiDiverse/WikiDiverse_train_with_boxes_sort3_32.json"

    # ---------- params ----------
    cluster_size = 32
    TOPK_NEG = 3

    # ---------- pipeline ----------
    train_data = load_train_data(train_json)
    rank_mat = load_rank_matrix(rank_path)

    qid2indices = build_qid_index_map(train_data)

    neg_qids_list = find_topk_hardneg_qids(
        train_data, rank_mat, K=TOPK_NEG
    )

    adjacency_list = build_hardneg_adjacency_multi(
        train_data, neg_qids_list, qid2indices
    )

    n_clusters = len(train_data) // cluster_size
    parts = metis_partition(adjacency_list, n_clusters)

    final_clusters = fix_cluster_sizes(parts, cluster_size)

    ordered_indices = make_ordered_indices(final_clusters)
    reorder_json(train_json, ordered_indices, out_path)
