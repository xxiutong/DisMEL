import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from urllib.parse import unquote
from transformers import CLIPProcessor
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from codes.model.lightning_dismel import LightningForDisMEL
from codes.utils.dataset import _load_json_file, boxes_to_patch_mask_vit_b32
from codes.utils.functions import setup_parser


class TrainedModelForRankMatrix:

    def __init__(self, args, checkpoint_path):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"load model: {checkpoint_path}")
        self.model = LightningForDisMEL.load_from_checkpoint(
            checkpoint_path, args=self.args, map_location="cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(self.args.pretrained_model)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        print(f"load entities: {args.data.entity}")
        self.entities = sorted(_load_json_file(args.data.entity), key=lambda x: x["id"])
        self.entity_dict = {e["id"]: e for e in self.entities}

        print(f"load train mentions: {args.data.train_file}")
        self.mentions = _load_json_file(args.data.train_file)

        self.mention_boxes = []
        for mention in self.mentions:
            boxes = mention.pop('boxes', [])
            mask = boxes_to_patch_mask_vit_b32(boxes, 384, 384)
            self.mention_boxes.append(mask)
        self.mention_boxes = torch.stack(self.mention_boxes)

        with open(self.args.data.qid2id, "r", encoding="utf-8") as f:
            self.qid2id = json.loads(f.readline())

        print("init ok")

    def load_image_or_black(self, path):
        try:
            img = Image.open(path).resize((224, 224), Image.Resampling.LANCZOS)
            pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
        except:
            pixel_values = torch.zeros((3, 224, 224))

        return pixel_values

    def encode_mention_batch(self, batch):
        texts, images = [], []

        for m in batch:
            text = unquote(m["mentions"]) + " [SEP] " + m["sentence"]
            texts.append(text)

            if m.get("imgPath") and m["imgPath"] != "":
                img_name = m["imgPath"].split("/")[-1].split(".")[0] + ".jpg"
                img_path = os.path.join(self.args.data.mention_img_folder, img_name)
            else:
                img_path = None

            images.append(self.load_image_or_black(img_path))

        pixel_values = torch.stack(images)
        tok = self.tokenizer(texts, padding="max_length", truncation=True, max_length=40, return_tensors="pt")

        with torch.no_grad():
            txt, im, txt_seq, img_patch = self.model.encoder(
                input_ids=tok["input_ids"].to(self.device),
                attention_mask=tok["attention_mask"].to(self.device),
                pixel_values=pixel_values.to(self.device)
            )

        return txt.cpu(), im.cpu(), txt_seq.cpu(), img_patch.cpu()

    def encode_entity_batch(self, ents):
        texts, images = [], []

        for e in ents:
            text = unquote(e["entity_name"]) + " [SEP] " + e["desc"]
            texts.append(text)

            if e.get("image_list") and len(e["image_list"]) > 0:
                img_path = os.path.join(self.args.data.kb_img_folder, e["image_list"][0])
            else:
                img_path = None

            images.append(self.load_image_or_black(img_path))
        pixel_values = torch.stack(images)
        tok = self.tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

        with torch.no_grad():
            txt, im, txt_seq, img_patch = self.model.encoder(
                input_ids=tok["input_ids"].to(self.device),
                attention_mask=tok["attention_mask"].to(self.device),
                pixel_values=pixel_values.to(self.device)
            )

        return txt.cpu(), im.cpu(), txt_seq.cpu(), img_patch.cpu()

    def encode_all_mentions(self):
        print("👉 encode all mentions ...")

        M = len(self.mentions)
        mt, mi, mts, mip = [], [], [], []
        batch_size = 512

        for i in tqdm(range(0, M, batch_size)):
            batch = self.mentions[i:i + batch_size]
            t, im, ts, ip = self.encode_mention_batch(batch)
            mt.append(t)
            mi.append(im)
            mts.append(ts)
            mip.append(ip)

        return (
            torch.cat(mt, dim=0),
            torch.cat(mi, dim=0),
            torch.cat(mts, dim=0),
            torch.cat(mip, dim=0)
        )

    # ==========================================
    # ==========================================
    def encode_all_entities(self):
        print("👉 encode train entities ...")
        ents = []
        for m in self.mentions:
            qid = m["answer"]
            if qid == 'nil':
                print('nil')
            entity_idx = self.qid2id[qid]
            ents.append(self.entity_dict[entity_idx])

        M = len(ents)
        et, ei, ets, eip = [], [], [], []
        batch_size = 512

        for i in tqdm(range(0, M, batch_size)):
            batch = ents[i:i + batch_size]
            t, im, ts, ip = self.encode_entity_batch(batch)
            et.append(t)
            ei.append(im)
            ets.append(ts)
            eip.append(ip)

        return (
            torch.cat(et, dim=0),
            torch.cat(ei, dim=0),
            torch.cat(ets, dim=0),
            torch.cat(eip, dim=0)
        )

    # ==========================================
    # ==========================================
    def compute_rank_matrix(self, mt, mi, mts, mip, et, ei, ets, eip):
        N = mt.size(0)
        print(f"generate NN：{N} × {N}")

        batch_mention = 128
        chunk_entity = 128
        rank_mat = np.zeros((N, N), dtype=np.int32)

        for i in tqdm(range(0, N, batch_mention)):
            end_i = min(i + batch_mention, N)

            mt_i = mt[i:end_i].to(self.device)
            mi_i = mi[i:end_i].to(self.device)
            mts_i = mts[i:end_i].to(self.device)
            mip_i = mip[i:end_i].to(self.device)
            box_mask = self.mention_boxes[i:end_i].to(self.device)

            sims_all = []

            for j in range(0, N, chunk_entity):
                end_j = min(j + chunk_entity, N)

                et_j = et[j:end_j].to(self.device)
                ei_j = ei[j:end_j].to(self.device)
                ets_j = ets[j:end_j].to(self.device)
                eip_j = eip[j:end_j].to(self.device)

                with torch.no_grad():
                    sims, _ = self.model.matcher(
                        et_j, ets_j, mt_i, mts_i,
                        ei_j, eip_j, mi_i, mip_i, box_mask
                    )  # sims: [chunk_i, chunk_j]

                sims_all.append(sims.cpu())

            sims_all = torch.cat(sims_all, dim=1)  # [chunk_i, N]

            row_count = end_i - i
            local_rows = torch.arange(row_count)
            global_cols = torch.arange(i, end_i)  #

            sims_all[local_rows, global_cols] = 1e9

            sorted_idx = torch.argsort(sims_all, dim=-1, descending=True)
            rank_mat[i:end_i] = sorted_idx.numpy()

        return rank_mat

    # ==========================================
    # ==========================================
    def preprocess(self):
        print("===== start =====")

        # 1) encode
        mt, mi, mts, mip = self.encode_all_mentions()
        et, ei, ets, eip = self.encode_all_entities()

        # 2) rank matrix
        rank_mat = self.compute_rank_matrix(mt, mi, mts, mip, et, ei, ets, eip)

        out_dir = os.path.dirname(self.args.data.train_file)
        out_path = os.path.join(out_dir, "mel_pred_rank.npy")

        np.save(out_path, rank_mat)
        print(f"NN save to: {out_path}")


# ============================================================
# ============================================================
def main():
    args = setup_parser()

    checkpoint_path = "/data/zyt/MMOR/runs/WikiDiverse/version_13/checkpoints/epoch=9-step=890.ckpt"
    processor = TrainedModelForRankMatrix(args, checkpoint_path)
    processor.preprocess()


if __name__ == "__main__":
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    main()
