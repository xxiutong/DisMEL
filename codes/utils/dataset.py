import os
import copy
import json
import os.path
import random
import pickle

import torch
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from urllib.parse import unquote
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_json_file(filepath):
    data = []
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
            data.extend(d)
    elif isinstance(filepath, list):
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                data.extend(d)
    return data


class DataModuleForDisMEL(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModuleForDisMEL, self).__init__()
        self.args = args
        self.tokenizer = CLIPProcessor.from_pretrained(self.args.pretrained_model).tokenizer
        self.image_processor = CLIPProcessor.from_pretrained(self.args.pretrained_model).feature_extractor
        with open(self.args.data.qid2id, 'r', encoding='utf-8') as f:
            self.qid2id = json.loads(f.readline())
        self.raw_kb_entity = sorted(_load_json_file(self.args.data.entity), key=lambda x: x['id'])
        self.kb_entity = self.setup_dataset_for_entity(self.args.data.entity, self.raw_kb_entity)
        self.kb_id2entity = {raw_ent['id']: ent for raw_ent, ent in zip(self.raw_kb_entity, self.kb_entity)}

        self.train_data = self.setup_dataset_for_mention(self.args.data.train_file,
                                                         _load_json_file(self.args.data.train_file))
        self.val_data = self.setup_dataset_for_mention(self.args.data.dev_file,
                                                       _load_json_file(self.args.data.dev_file))
        self.test_data = self.setup_dataset_for_mention(self.args.data.test_file,
                                                        _load_json_file(self.args.data.test_file))

    def setup_dataset_for_entity(self, path, data):
        # prepare entity information
        # pkl_path = path[0:path.rfind('.')] + '.pkl'
        # if os.path.exists(pkl_path):
        #     with open(pkl_path, 'rb') as file:
        #         input_data = pickle.load(file)
        #     return input_data

        input_data = []
        for sample_dict in tqdm(data, desc='PreProcessing'):
            sample_type = sample_dict['type']
            if sample_type == 'entity':
                entity, desc = unquote(sample_dict.pop('entity_name')), sample_dict.pop('desc')
                input_text = entity + ' [SEP] ' + desc  # concat entity and sentence
                input_dict = self.tokenizer(input_text, padding='max_length', max_length=self.args.data.text_max_length,
                                            truncation=True)
            input_dict['img_list'] = sample_dict['image_list']
            input_dict['sample_type'] = 0 if sample_type == 'entity' else 1
            if 'answer' in sample_dict.keys():
                input_dict['answer'] = self.qid2id[sample_dict['answer']]
            input_data.append(input_dict)

        # with open(pkl_path, 'wb') as file:
        #     pickle.dump(input_data, file)

        return input_data

    def setup_dataset_for_mention(self, path, data):
        # prepare mention information
        # pkl_path = path[0:path.rfind('.')] + '.pkl'
        # if os.path.exists(pkl_path):
        #     with open(pkl_path, 'rb') as file:
        #         input_data = pickle.load(file)
        #     return input_data

        input_data = []
        for sample_dict in tqdm(data, desc='PreProcessing'):
            sample_type = 1
            entity, mention, text, desc = unquote(sample_dict.pop('entities')), unquote(
                sample_dict.pop('mentions')), sample_dict.pop('sentence'), sample_dict.pop('desc')
            input_text = mention + ' [SEP] ' + text + ' [SEP] ' + desc # concat entity and text
            # input_text = mention + ' [SEP] ' + text  # concat entity and text
            input_dict = self.tokenizer(input_text, padding='max_length', max_length=self.args.data.text_max_length,
                                        truncation=True)

            input_dict['img_list'] = [sample_dict['imgPath']] if sample_dict['imgPath'] != '' else []
            input_dict['sample_type'] = sample_type
            if 'answer' in sample_dict.keys():
                input_dict['answer'] = self.qid2id[sample_dict['answer']]
            if 'negative' in sample_dict.keys():
                input_dict['negative'] = self.qid2id[sample_dict['negative']]
            if sample_dict['answer'] == 'nil':  # ignore the sample without ground truth
                continue

            input_dict['boxes'] = sample_dict.get('boxes', [])
            input_data.append(input_dict)

        # with open(pkl_path, 'wb') as file:
        #     pickle.dump(input_data, file)

        return input_data

    def choose_image(self, sample_type, img_list, is_eval=False):
        if len(img_list):
            img_name = random.choice(img_list)
            # when evaluation, we choose the first image
            if is_eval:
                img_name = img_list[0]
            if sample_type == 1:
                img_name = img_name.split('/')[-1].split('.')[0] + '.jpg'  # we already convert all image to jpg format
            try:
                img_path = os.path.join(
                    self.args.data.kb_img_folder if sample_type == 0 else self.args.data.mention_img_folder,
                    img_name)
                img = Image.open(img_path).resize((224, 224), Image.Resampling.LANCZOS)
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
            except:
                pixel_values = torch.zeros((3, 224, 224))
        else:
            pixel_values = torch.zeros((3, 224, 224))
        return pixel_values

    def train_collator(self, samples):
        cls_idx, img_list, sample_type, input_dict_list = [], [], [], []
        pixel_values, gt_ent_id = [], []
        negative_ids = []

        mention_boxes = []
        mention_patch_masks = []
        # collect the metadata that need to further process
        for sample_idx, sample in enumerate(samples):
            img_list.append(sample.pop('img_list'))  # mention image list
            sample_type.append(sample.pop('sample_type'))  # input type: 0 for mention and 1 for entity
            input_dict_list.append(sample)  # mention input dict (input_tokens, token_type_ids, attention_mask)
            gt_ent_id.append(sample.pop('answer'))  # ground truth entity id of mentions
            if 'negative' in sample:
                neg = sample.pop('negative')
                negative_ids.append(neg)
            mention_boxes.append(sample.pop('boxes', []))
        ###
        # Now we process mention information
        # choose an image
        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx]))
            boxes = mention_boxes[idx]  # list of dict
            mask = boxes_to_patch_mask_vit_b32(
                boxes,
                img_w=384,
                img_h=384
            )
            mention_patch_masks.append(mask)
        # pad textual input
        input_dict = self.tokenizer.pad(input_dict_list,
                                        padding='max_length',
                                        max_length=self.args.data.text_max_length,
                                        return_tensors='pt')
        # concat all images
        pixel_values = torch.stack(pixel_values)
        input_dict['pixel_values'] = pixel_values
        input_dict['mention_patch_mask'] = torch.stack(mention_patch_masks)
        ###
        # now we process entity information
        # fetch the entities' metadata
        # ent_info_list = [copy.deepcopy(self.kb_id2entity[idx]) for idx in gt_ent_id]
        input_dict['answer'] = torch.tensor(gt_ent_id, dtype=torch.long)
        ent_info_list = []
        for ent_id in gt_ent_id:
            ent_info_list.append(copy.deepcopy(self.kb_id2entity[ent_id]))
        for ent_id in negative_ids:
            ent_info_list.append(copy.deepcopy(self.kb_id2entity[ent_id]))
        ent_img_list, ent_type, ent_input_dict_list, ent_pixel_values = [], [], [], []
        for ent_dict in ent_info_list:
            ent_img_list.append(ent_dict.pop('img_list'))  # entity image list
            ent_type.append(ent_dict.pop('sample_type'))  # input type: 0 for mention and 1 for entity
            ent_input_dict_list.append(ent_dict)  # entity input dict (input_tokens, token_type_ids, attention_mask)
        # choose an image
        for idx, _ in enumerate(ent_input_dict_list):
            ent_pixel_values.append(self.choose_image(ent_type[idx], ent_img_list[idx]))
        # some of the entities do not have image, so we use bool flags to tag them
        # ent_empty_img_flag = torch.tensor([True if not len(_) else False for _ in ent_img_list], dtype=torch.bool)
        # pad textual input
        ent_input_dict = self.tokenizer.pad(ent_input_dict_list,
                                            padding='max_length',
                                            max_length=self.args.data.text_max_length,
                                            return_tensors='pt')
        # concat all image
        ent_pixel_values = torch.stack(ent_pixel_values)
        ent_input_dict['pixel_values'] = ent_pixel_values
        # ent_input_dict['empty_img_flag'] = ent_empty_img_flag

        # for the entity information, we use prefix 'ent_' to tag them
        for k, v in ent_input_dict.items():
            input_dict[f'ent_{k}'] = v
        return input_dict

    def eval_collator(self, samples):
        # eval collator is similar to train collator, but only include mention information
        cls_idx, img_list, sample_type, input_dict_list = [], [], [], []
        pixel_values, gt_ent_id = [], []
        mention_boxes = []
        mention_patch_masks = []

        for sample_idx, sample in enumerate(samples):
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            gt_ent_id.append(sample.pop('answer'))
            input_dict_list.append(sample)
            mention_boxes.append(sample.pop('boxes', []))
        for idx, _ in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))
            boxes = mention_boxes[idx]  # list of dict
            mask = boxes_to_patch_mask_vit_b32(
                boxes,
                img_w=384,
                img_h=384
            )
            mention_patch_masks.append(mask)

        input_dict = self.tokenizer.pad(input_dict_list,
                                        padding='max_length',
                                        max_length=self.args.data.text_max_length,
                                        return_tensors='pt')
        input_dict['mention_patch_mask'] = torch.stack(mention_patch_masks)
        input_dict['pixel_values'] = torch.stack(pixel_values)
        input_dict['answer'] = torch.tensor(gt_ent_id, dtype=torch.long)
        return input_dict

    def entity_collator(self, samples):
        # entity collator is similar to train collator, but only include entity information
        pixel_values, img_list, sample_type, input_dict_list = [], [], [], []
        for sample_idx, sample in enumerate(samples):
            img_list.append(sample.pop('img_list'))
            sample_type.append(sample.pop('sample_type'))
            input_dict_list.append(sample)
        for idx, input_dict in enumerate(input_dict_list):
            pixel_values.append(self.choose_image(sample_type[idx], img_list[idx], is_eval=True))

        input_dict = self.tokenizer.pad(input_dict_list,
                                        padding='max_length',
                                        max_length=self.args.data.text_max_length,
                                        return_tensors='pt')
        input_dict['pixel_values'] = torch.stack(pixel_values)

        return input_dict

    def entity_dataloader(self):
        return DataLoader(self.kb_entity,
                          batch_size=self.args.data.embed_update_batch_size,
                          num_workers=self.args.data.num_workers,
                          shuffle=False,
                          collate_fn=self.entity_collator)

    # def train_dataloader(self):
    #     return DataLoader(self.train_data,
    #                       batch_size=self.args.data.batch_size,
    #                       num_workers=self.args.data.num_workers,
    #                       shuffle=True,
    #                       collate_fn=self.train_collator)
    def train_dataloader(self):
        if self.args.data.cluster_size == 0:
            return DataLoader(self.train_data,
                              batch_size=self.args.data.batch_size,
                              num_workers=self.args.data.num_workers,
                              shuffle=True,
                              collate_fn=self.train_collator)

        else:
            dataset_len = len(self.train_data)

            batch_sampler = MultiClusterBatchSampler(
                dataset_len=dataset_len,
                batch_size=self.args.data.batch_size,
                cluster_size=self.args.data.cluster_size,
                shuffle=True,
            )

        return DataLoader(
            self.train_data,
            batch_sampler=batch_sampler,
            num_workers=self.args.data.num_workers,
            collate_fn=self.train_collator,
        )

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.args.data.eval_batch_size,
                          num_workers=self.args.data.num_workers,
                          shuffle=False,
                          collate_fn=self.eval_collator)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.args.data.eval_batch_size,
                          num_workers=self.args.data.num_workers,
                          shuffle=False,
                          collate_fn=self.eval_collator)


class MultiClusterBatchSampler(torch.utils.data.Sampler):
    """
    """

    def __init__(self, dataset_len, batch_size, cluster_size, shuffle=True):
        self.dataset_len = dataset_len
        self.cluster_size = cluster_size
        self.batch_size = batch_size
        self.clusters_per_batch = batch_size // cluster_size
        self.shuffle = shuffle

        self.clusters = []
        for start in range(0, dataset_len, cluster_size):
            end = min(start + cluster_size, dataset_len)
            self.clusters.append(list(range(start, end)))

        assert all(len(c) == cluster_size for c in self.clusters), \
            f"Some clusters are not size={cluster_size}"

    def __iter__(self):
        cluster_ids = list(range(len(self.clusters)))
        if self.shuffle:
            random.shuffle(cluster_ids)

        for i in range(0, len(cluster_ids), self.clusters_per_batch):
            selected = cluster_ids[i:i + self.clusters_per_batch]

            batch_indices = []
            for cid in selected:
                batch_indices.extend(self.clusters[cid])

            yield batch_indices

    def __len__(self):
        return len(self.clusters) // self.clusters_per_batch


def boxes_to_patch_mask_vit_b32(
        boxes,  # list of dicts: {"box":[x1,y1,x2,y2], ...}
        img_w,
        img_h,
        device="cpu"
):
    """
    Returns:
        mask: Tensor [50]  (CLS + 49 patches)
    """
    G = 7
    P = 49

    # ---- patch centers ----
    ys, xs = torch.meshgrid(
        torch.arange(G),
        torch.arange(G),
        indexing="ij"
    )
    xs = (xs + 0.5) / G
    ys = (ys + 0.5) / G
    patch_centers = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # [49,2]

    # ---- patch mask (no CLS yet) ----
    patch_mask = torch.zeros(P, dtype=torch.float32)

    if len(boxes) == 0:
        patch_mask[:] = 1.0
    else:
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            x1 /= img_w;
            x2 /= img_w
            y1 /= img_h;
            y2 /= img_h

            inside = (
                    (patch_centers[:, 0] >= x1) &
                    (patch_centers[:, 0] <= x2) &
                    (patch_centers[:, 1] >= y1) &
                    (patch_centers[:, 1] <= y2)
            )
            patch_mask[inside] = 1.0

    # ---- prepend CLS = 0 ----
    cls = torch.zeros(1, dtype=torch.float32)
    mask = torch.cat([cls, patch_mask], dim=0)  # [50]

    return mask
