import os
import json
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from codes.utils.dataset import _load_json_file
from codes.utils.functions import setup_parser


def fill_negative_from_rank(args):

    print("🟦 Loading train file ...")
    train_path = args.data.train_file
    train_data = _load_json_file(train_path)
    N = len(train_data)
    print(f"→ train size = {N}")

    print("🟦 Loading rank matrix ...")
    rank_mat = np.load(args.data.rank_matrix)
    assert rank_mat.shape == (N, N)
    print(f"→ rank_mat shape = {rank_mat.shape}")

    neg_rank_k = getattr(args.data, "neg_rank_k", 0)
    print(f"🟩 Using neg_rank_k = {neg_rank_k}")

    new_train = []

    print("🟩 Filling negatives ...")
    for i, sample in enumerate(tqdm(train_data)):
        new_sample = sample.copy()

        rank_list = rank_mat[i]

        if int(rank_list[0]) != i:
            raise RuntimeError(
                f"❌ rank_mat[{i}][0] != {i}, got {rank_list[0]}"
            )

        pos_qid = sample["answer"]

        ptr = 1 + neg_rank_k
        neg_qid = None

        while ptr < N:
            cand_train_idx = int(rank_list[ptr])
            cand_qid = train_data[cand_train_idx]["answer"]

            if cand_qid != pos_qid:
                neg_qid = cand_qid
                break
            # else:
            #     print('....')

            ptr += 1

        if neg_qid is None:
            raise RuntimeError(
                f"❌ No valid negative found for sample {i} (qid={pos_qid})"
            )

        new_sample["negative"] = neg_qid
        new_train.append(new_sample)

    out_path = train_path.replace(".json", "_with_neg.json")
    print(f"🟦 Saving enhanced train file → {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_train, f, ensure_ascii=False, indent=2)

    print("🎉 Done.")




if __name__ == "__main__":
    args = setup_parser()
    args.data.rank_matrix = "/data/zyt/datas4/WikiMEL/mel_pred_rank.npy"
    args.data.neg_rank_k = 0
    fill_negative_from_rank(args)
