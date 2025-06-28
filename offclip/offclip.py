import os
import torch
import numpy as np
import random
import pandas as pd
from . import builder
from typing import List

np.random.seed(10)
random.seed(10)


_MODELS = {
    "OFFCLIP_resnet50": "",
    "OFFCLIP_vit_b_16": "./checkpoint/paper/offclip/offclip_best_model.ckpt"
}



_FEATURE_DIM = {"OFFCLIP_resnet50": 2048, "OFFCLIP_vit_b_16": 768 }


def available_models() -> List[str]:
    """Returns the names of available offclip models"""
    return list(_MODELS.keys())

def load_offclip_validation(cfg, state_dict):
    """Load a offclip model for validation purpose"""

    model = builder.build_offclip_model(cfg)
    new_ckpt = torch.load(state_dict, map_location = 'cpu')
    model.load_state_dict(new_ckpt['model_state_dict'], strict=False) 

    return model

def get_dqn_similarities(offclip_model, imgs, txts, similarity_type="both"):
    """Load a offclip pretrained classification model

    Parameters
    ----------
    offclip_model : str
        offclip model, load via offclip.load_models()
    imgs:
        processed images using offclip_model.process_img
    txts:
        processed text using offclip_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use offclip_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use offclip_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        offclip_model.eval()
        label_img_emb_l, label_img_emb_g = offclip_model.image_encoder_forward(imgs)
        query_emb_l, query_emb_g, _ = offclip_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

        cls_bs = []
        it2_SimR_bs = []
        t2i_SimR_bs = []
        bs = label_img_emb_g.size(0)
        for i in range(bs):
            label_img_emb_l_ = label_img_emb_l[i:i+1].view(label_img_emb_l[i:i+1].size(0), label_img_emb_l[i:i+1].size(1), -1) 

            label_img_emb_g_ = label_img_emb_g[i:i+1]

            label_img_emb_l_ = label_img_emb_l_.permute(0, 2, 1) #patch_num b dim

            query_emb_l_ = query_emb_l.view(query_emb_l.size(0), query_emb_l.size(1), -1) 

            query_emb_l_ = query_emb_l_.permute(0, 2, 1) #patch_num b dim # [97, 512, 768]

            i2t_cls, atten_i2t = offclip_model.fusion_module(torch.cat([label_img_emb_g_.unsqueeze(1) , label_img_emb_l_], dim=1), query_emb_g, return_atten=True)

            i2t_cls = i2t_cls.squeeze(-1)  ## use text as query, use image as k, v, so image batch size have not distrubed the result

            t2i_cls, atten_t2i = offclip_model.fusion_module(torch.cat([query_emb_g.unsqueeze(1) , query_emb_l_], dim=1), label_img_emb_g_, return_atten=True)

            t2i_cls = t2i_cls.squeeze(-1).transpose(1,0) 

            cls = (i2t_cls + t2i_cls) / 2

            cls_bs.append(cls)

        cls = torch.cat(cls_bs, dim=0)

        return cls.detach().cpu().numpy()

def dqn_shot_classification(offclip_model, imgs, cls_txt_mapping):
    """Load a offclip_model pretrained classification model

    Parameters
    ----------
    offclip_model : str
        offclip model, load via offclip.load_models()
    imgs:
        processed images using offclip_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    # class_similarities = []
    caption_ids = []
    attention_mask = []
    token_type_ids = []
    for cls_name, txts in cls_txt_mapping.items():
        caption_ids.append(txts["caption_ids"])
        attention_mask.append(txts["attention_mask"])
        token_type_ids.append(txts["token_type_ids"])

    caption_ids = torch.cat(caption_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    text_batch = {"caption_ids": caption_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}

    cls_similarity = get_dqn_similarities(
        offclip_model, imgs, text_batch, similarity_type="both"
    )

    class_similarities = pd.DataFrame(
        cls_similarity, columns=cls_txt_mapping.keys()
    )
    return class_similarities



