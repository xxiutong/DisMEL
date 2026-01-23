import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel
from codes.model.moe import MoE
from codes.utils.utils import MultiHeadModule, MatchModule


class DisMELEncoder(nn.Module):
    def __init__(self, args):
        super(DisMELEncoder, self).__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.pretrained_model)

        self.text_tokens_fc = nn.Linear(
            self.args.model.input_hidden_dim, self.args.model.input_hidden_dim
        )
        self.image_tokens_fc = nn.Linear(
            self.args.model.input_image_hidden_dim, self.args.model.input_hidden_dim
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            pixel_values=None,
    ):
        clip_output = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        text_seq_tokens = clip_output.text_model_output[0]  # [batch_size, 40, 512]
        image_patch_tokens = clip_output.vision_model_output[0]  # [batch_size, 50, 768]

        text_seq_tokens = self.text_tokens_fc(text_seq_tokens)  # [batch_size, 40, 512]
        image_patch_tokens = self.image_tokens_fc(
            image_patch_tokens
        )  # [batch_size, 50, 512]

        text_cls = clip_output.text_embeds  # [batch_size, 512]
        image_cls = clip_output.image_embeds  # [batch_size, 512]

        return text_cls, image_cls, text_seq_tokens, image_patch_tokens


class TextUnit(nn.Module):
    def __init__(self, args):
        super(TextUnit, self).__init__()
        self.args = args
        self.fc_query = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.TGLU_hidden_dim)

        self.fc_cls = nn.Linear(self.args.model.input_hidden_dim, self.args.model.TGLU_hidden_dim)

    def forward(self,
                entity_text_cls,
                entity_text_tokens,
                mention_text_cls,
                mention_text_tokens):
        entity_cls_fc = self.fc_cls(entity_text_cls)
        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)

        query = self.fc_query(entity_text_tokens)  # [num_entity, max_seq_len, dim]
        key = self.fc_key(mention_text_tokens)  # [batch_size, max_sqe_len, dim]
        value = self.fc_value(mention_text_tokens)  # [batch_size, max_sqe_len, dim]

        query = query.unsqueeze(dim=1)  # [num_entity, 1, max_seq_len, dim]
        key = key.unsqueeze(dim=0)  # [1, batch_size, max_sqe_len, dim]
        value = value.unsqueeze(dim=0)  # [1, batch_size, max_sqe_len, dim]

        attention_scores = torch.matmul(query,
                                        key.transpose(-1, -2))  # [num_entity, batach_size, max_seq_len, max_seq_len]

        attention_scores = attention_scores / math.sqrt(self.args.model.TGLU_hidden_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [num_entity, batch_size, max_seq_len, max_seq_len]

        context = torch.matmul(attention_probs, value)  # [num_entity, batch_size, max_seq_len, dim]
        context = torch.mean(context, dim=-2)  # [num_entity, batch_size, dim]
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)  # [num_entity, batch_size]
        g2l_matching_score = g2l_matching_score.transpose(0, 1)  # [batch_size, num_entity]
        g2g_matching_score = torch.matmul(mention_text_cls, entity_text_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score


class VisionUnit(nn.Module):
    def __init__(self, args):
        super(VisionUnit, self).__init__()
        self.args = args

        self.fc_query = nn.Linear(self.args.model.input_hidden_dim, self.args.model.IDLU_hidden_dim)
        self.fc_key = nn.Linear(self.args.model.input_hidden_dim, self.args.model.IDLU_hidden_dim)
        self.fc_value = nn.Linear(self.args.model.input_hidden_dim, self.args.model.IDLU_hidden_dim)
        self.layer_norm = nn.LayerNorm(self.args.model.IDLU_hidden_dim)
        self.fc_cls = nn.Linear(self.args.model.input_hidden_dim, self.args.model.IDLU_hidden_dim)
        self.rho = self.args.model.rho

    def forward(self,
                entity_cls,
                entity_tokens,
                mention_image_cls,
                mention_image_tokens, mention_patch_mask):
        """
        :param entity_image_cls:        [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_image_cls:       [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """

        entity_cls_fc = self.fc_cls(entity_cls)
        entity_cls_fc = entity_cls_fc.unsqueeze(dim=1)

        query = self.fc_query(entity_tokens)  # [num_entity, num_patch, dim]
        key = self.fc_key(mention_image_tokens)  # [batch_size, num_patch, dim]
        value = self.fc_value(mention_image_tokens)  # [batch_size, num_patch, dim]

        query = query.unsqueeze(dim=1)  # [num_entity, 1, num_patch, dim]
        key = key.unsqueeze(dim=0)  # [1, batch_size, num_patch, dim]
        value = value.unsqueeze(dim=0)  # [1, batch_size, num_patch, dim]

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [num_entity, batch_size, num_patch, num_patch]

        attention_scores = attention_scores / math.sqrt(self.args.model.IDLU_hidden_dim)
        # =========================================================
        # NEW: box-guided patch prior → attention bias
        # =========================================================
        # mention_patch_mask: [B, P] → [1, B, 1, P]
        mask = mention_patch_mask.unsqueeze(0).unsqueeze(2)

        attention_scores[:, :, 0:1, :] += self.rho * mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        # attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [num_entity, batch_size, num_patch, num_patch]

        context = torch.matmul(attention_probs, value)  # [num_entity, batch_size, num_patch, dim]
        context = torch.mean(context, dim=-2)  # [num_entity, batch_size, dim]
        context = self.layer_norm(context)

        g2l_matching_score = torch.sum(entity_cls_fc * context, dim=-1)  # [num_entity, batch_size]
        g2l_matching_score = g2l_matching_score.transpose(0, 1)  # [batch_size, num_entity]
        g2g_matching_score = torch.matmul(mention_image_cls, entity_cls.transpose(-1, -2))

        matching_score = (g2l_matching_score + g2g_matching_score) / 2
        return matching_score


class CrossUnit(nn.Module):
    def __init__(self, args):
        super(CrossUnit, self).__init__()
        self.args = args
        self.text_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMFU_hidden_dim)
        self.image_fc = nn.Linear(self.args.model.input_hidden_dim, self.args.model.CMFU_hidden_dim)
        self.gate_fc = nn.Linear(self.args.model.CMFU_hidden_dim, 1)
        self.gate_act = nn.Tanh()
        self.gate_layer_norm = nn.LayerNorm(self.args.model.CMFU_hidden_dim)

        self.match_module = MatchModule(self.args.model.CMFU_hidden_dim)
        self.multi_head_module = MultiHeadModule(4, 1)
        self.moe = MoE(in_dim=args.model.CMFU_hidden_dim, hidden_dim=args.model.CMFU_hidden_dim)

    def forward(self, entity_text_cls, entity_image_tokens,
                mention_text_cls, mention_image_tokens):
        """
        :param entity_text_cls:         [num_entity, dim]
        :param entity_image_tokens:     [num_entity, num_patch, dim]
        :param mention_text_cls:        [batch_size, dim]
        :param mention_image_tokens:    [batch_size, num_patch, dim]
        :return:
        """
        entity_text_cls = self.text_fc(entity_text_cls)
        entity_text_cls_ori = entity_text_cls
        mention_text_cls = self.text_fc(mention_text_cls)
        mention_text_cls_ori = mention_text_cls

        entity_image_tokens = self.image_fc(entity_image_tokens)
        mention_image_tokens = self.image_fc(mention_image_tokens)

        entity_text_cls = self.match_module([entity_text_cls_ori.unsqueeze(dim=1), entity_image_tokens]).squeeze()
        entity_image_tokens = self.match_module([entity_image_tokens, entity_text_cls_ori.unsqueeze(dim=1)])
        entity_image_tokens = self.multi_head_module(entity_image_tokens)
        entity_context = self.moe(entity_text_cls, entity_image_tokens)
        entity_gate_score = self.gate_act(self.gate_fc(entity_text_cls_ori))
        entity_context = self.gate_layer_norm((entity_text_cls_ori * entity_gate_score) + entity_context)

        mention_text_cls = self.match_module([mention_text_cls_ori.unsqueeze(dim=1), mention_image_tokens]).squeeze()
        mention_image_tokens = self.match_module([mention_image_tokens, mention_text_cls_ori.unsqueeze(dim=1)])
        mention_image_tokens = self.multi_head_module(mention_image_tokens)
        mention_context = self.moe(mention_text_cls, mention_image_tokens)
        mention_gate_score = self.gate_act(self.gate_fc(mention_text_cls_ori))
        mention_context = self.gate_layer_norm((mention_text_cls_ori * mention_gate_score) + mention_context)

        score = torch.matmul(mention_context, entity_context.transpose(-1, -2))

        return score


class DisMELMatcher(nn.Module):
    def __init__(self, args):
        super(DisMELMatcher, self).__init__()
        self.args = args
        self.text_module = TextUnit(self.args)
        self.vision_module = VisionUnit(self.args)
        self.cross_module = CrossUnit(self.args)

        self.text_tokens_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)
        self.image_tokens_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)
        self.text_cls_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)
        self.image_cls_layernorm = nn.LayerNorm(self.args.model.input_hidden_dim)


    def forward(self,
                entity_text_cls, entity_text_tokens,
                mention_text_cls, mention_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens, mention_patch_mask):
        """

        :param entity_text_cls:     [num_entity, dim]
        :param entity_text_tokens:  [num_entity, max_seq_len, dim]
        :param mention_text_cls:    [batch_size, dim]
        :param mention_text_tokens: [batch_size, max_sqe_len, dim]
        :param entity_image_cls:    [num_entity, dim]
        :param mention_image_cls:   [batch_size, dim]
        :param entity_image_tokens: [num_entity, num_patch, dim]
        :param mention_image_tokens:[num_entity, num_patch, dim]
        :return:
        """
        entity_text_cls = self.text_cls_layernorm(entity_text_cls)
        mention_text_cls = self.text_cls_layernorm(mention_text_cls)
        entity_text_tokens = self.text_tokens_layernorm(entity_text_tokens)
        mention_text_tokens = self.text_tokens_layernorm(mention_text_tokens)
        entity_image_cls = self.image_cls_layernorm(entity_image_cls)
        mention_image_cls = self.image_cls_layernorm(mention_image_cls)
        entity_image_tokens = self.image_tokens_layernorm(entity_image_tokens)
        mention_image_tokens = self.image_tokens_layernorm(mention_image_tokens)

        text_matching_score = self.text_module(entity_text_cls, entity_text_tokens,
                                               mention_text_cls, mention_text_tokens)
        image_matching_score = self.vision_module(entity_image_cls, entity_image_tokens,
                                                  mention_image_cls, mention_image_tokens, mention_patch_mask)
        cross_matching_score = self.cross_module(entity_text_cls, entity_image_tokens,
                                                 mention_text_cls,
                                                 mention_image_tokens)

        image_score = image_matching_score
        cross_score = cross_matching_score

        score = (text_matching_score + image_score + cross_score) / 3
        return score, (text_matching_score, image_score, cross_score)
