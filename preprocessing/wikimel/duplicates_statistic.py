import json
from collections import defaultdict
from tqdm import tqdm

from codes.utils.dataset import _load_json_file
from codes.utils.functions import setup_parser


def analyze_duplicate_answers_by_sample_id(args):
    print("🟦 Loading train file ...")
    train_data = _load_json_file(args.data.train_file)
    print(f"→ train size = {len(train_data)}")

    # qid -> [sample_id, sample_id, ...]
    qid2sample_ids = defaultdict(list)

    print("🟩 Scanning answers ...")
    for sample in tqdm(train_data):
        qid = sample.get("answer")
        sample_id = sample.get("id")   #
        qid2sample_ids[qid].append(sample_id)

    duplicates = {
        qid: ids
        for qid, ids in qid2sample_ids.items()
        if len(ids) > 1
    }

    print("\n================ RESULT ================")
    print(f"🔴 Total duplicated answers: {len(duplicates)}")

    for i, (qid, ids) in enumerate(duplicates.items()):
        print(f"\n[{i}] qid = {qid}")
        print(f"    count = {len(ids)}")
        print(f"    sample_ids = {ids}")
        if i >= 9:
            print("\n... (only showing first 10)")
            break

    out_path = args.data.train_file.replace(".json", "_answer_duplicate_sample_ids.json")
    print(f"\n🟦 Saving duplicate report → {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_duplicated_qids": len(duplicates),
                "duplicates": duplicates
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("🎉 Done.")


if __name__ == "__main__":
    args = setup_parser()
    analyze_duplicate_answers_by_sample_id(args)
