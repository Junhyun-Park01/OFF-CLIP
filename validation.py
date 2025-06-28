# Torch and Model Imports
import torch
import offclip

# Data Handling
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Utilities
import os
import time
from utils import *

import argparse
from omegaconf import OmegaConf

# Metrics and Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import expit

'''
Code based by CARZero (https://github.com/laihaoran/CARZero)
Modified evaluation function for classification. Every classes including 'No Finding' can be evaluated.
Classifiction:
    evaluation metric: AUC, ACC, Confidence matrix
    dataset: VinDr-CXR, OpenI, Padchest, chexpert
'''

def obtain_simr(image_path, text_path, model):
    df = pd.read_csv(image_path)
    with open(text_path, 'r') as f:
        cls_prompts = json.load(f)

    # load model
    
    model = model.to(device)
    
    image_list = []
    for img_p in df['Path'].tolist():
        if os.path.exists(img_p):
            image_list.append(img_p)
        else:
            continue

    # process input images and class prompts 
    ## batchsize
    bs = 1024
    bs_image_list = split_list(image_list, bs) # if error accurs check space after Path in csv.
    processed_txt = model.process_class_prompts(cls_prompts, device)

    print(f'Total {len(image_list)} images')
    print(f'calculating similarities...')
    for i, img in enumerate(bs_image_list): # batch size length image list.
        processed_imgs = model.process_img(img, False, device)
        # zero-shot classification images
        similarities = offclip.dqn_shot_classification(
            model, processed_imgs, processed_txt
            )    
        if i == 0:
            similar = similarities
        else:
            similar = pd.concat([similar, similarities], axis=0)
        print(f'{((i+1)/len(bs_image_list))*100:.2f}% of data inferenced.')
    return similar

def vindr_cxr_obtain_simr(image_root, csv_path, model):
    
    model = model.to(device)

    df = pd.read_csv(csv_path)
    
    cls_prompts = ["There is aortic enlargement", "There is atelectasis", "There is calcification", "There is cardiomegaly", "There is consolidation",
                   "There is edema", "There is emphysema", "There is enlarged PA", "There is infiltration", "There is lung opacity", 
                   "There is nodule/mass", "There is pleural effusion", "There is pleural thickening", "There is pneumothorax", 
                   "There is lung tumor", "There is pneumonia", "There is no finding"] # 17 in total
    
    key = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Edema", "Emphysema", "Enlarged PA", "Infiltration",
             "Lung Opacity", "Nodule/Mass", "Pleural effusion",  "Pleural thickening", "Pneumothorax", "Lung tumor", "Pneumonia", "No finding"]
    
    df_filtered = df[~(df[key] == 0).all(axis=1)]
    print(df_filtered)

    label_prompt_dict = dict(zip(key, cls_prompts))
    print(label_prompt_dict)


    img_list = split_list(df_filtered["image_id"].tolist(), 1024)
    processed_txt = model.process_class_prompts(label_prompt_dict, device)

    out_texts = ""
    for i, img in enumerate(img_list):
        img_elem_new = []
        for image_name in img:  # Ensure `img` is a list of filenames
            img_elem_new.append(image_root + "/"+ image_name + ".dicom")
        
        processed_imgs = model.process_img(img_elem_new, True, device)
        similarities = offclip.dqn_shot_classification(model, processed_imgs, processed_txt)

        probs = expit(similarities)  # Convert logits to probabilities
        print(probs)

        if i == 0:
            similar = probs
        else:
            similar = pd.concat([similar, probs], axis=0)

    label = df_filtered[key].values
    probs_df = similar[key].values

    auc, threshold = compute_auc_threshold(label, probs_df, n_class=len(key))
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(probs_df, label)
    
    print(f"Total AUC: {macro_auc}")
    print(f'per_auc: {per_auc}')

    for disease, auc in zip(key, per_auc):
        print(f"{disease}: {auc}")

    return similar, out_texts

def tripple_openi_rusult_merge(predict_csv, label_file_path):
    pathologies = [
        # NIH
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        ## "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
        # ---------
        "Fracture",
        "Opacity",
        "Lesion",
        # ---------
        "Calcified Granuloma",
        "Granuloma",
        # ---------
        "No_Finding",
    ] # 19 in total


    mapping = dict()
    mapping["Pleural_Thickening"] = ["pleural thickening"]
    mapping["Infiltration"] = ["Infiltrate"]
    mapping["Atelectasis"] = ["Atelectases"]

    # Load data
    csv = pd.read_csv(label_file_path)
    csv = csv.replace(np.nan, "-1")

    gt = []
    for pathology in pathologies:
        mask = csv["labels_automatic"].str.contains(pathology.lower())
        if pathology in mapping:
            for syn in mapping[pathology]:
                # print("mapping", syn)
                mask |= csv["labels_automatic"].str.contains(syn.lower())
        gt.append(mask.values)

    gt = np.asarray(gt).T
    gt = gt.astype(np.float32)

    # Rename pathologies
    pathologies = np.char.replace(pathologies, "Opacity", "Lung Opacity")
    pathologies = np.char.replace(pathologies, "Lesion", "Lung Lesion")

    ## Rename by myself
    pathologies = np.char.replace(pathologies, "Pleural_Thickening", "pleural thickening")
    pathologies = np.char.replace(pathologies, "Infiltration", "Infiltrate")
    pathologies = np.char.replace(pathologies, "Atelectasis", "Atelectases")
    gt[np.where(np.sum(gt, axis=1) == 0), -1] = 1
    
    # label = gt[:, :-1]
    label = gt

    predict = pd.read_csv(predict_csv).values
    
    aucs, thresholds = compute_auc_threshold(label, predict, n_class=len(pathologies))
    predict_multi_binary = (predict > np.array(thresholds)).astype(int)

    out_texts = "OpenI results\n"
    auc_texts = "-AUCs-\n"
    acc_texts = "-ACCs-\n"
    cm_texts = "-Confusion Matrix-\n"

    total_accs = []
    i=0
    for disease, auc in zip(pathologies, aucs):
        auc_texts += f"{disease}: {auc:.3f}\n"
        acc_i = accuracy_score(label[:, i], predict_multi_binary[:, i])
        acc_texts += f"{disease}: {acc_i:.3f}\n"
        total_accs.append(acc_i)
        cm_texts += f"{disease}:\n{confusion_matrix(label[:, i], predict_multi_binary[:, i])}\n"
        i+=1

    out_texts += f'Average AUC: {sum(aucs) / len(aucs):.3f}\n'
    out_texts += f'Average ACC: {sum(total_accs) / len(total_accs):.3f}\n'
    out_texts += auc_texts + acc_texts + cm_texts

    return out_texts

def triple_Chexpert_all_result(predict_csv, label_file_path):
    
    key = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

    df_test = pd.read_csv(label_file_path)
    predict = pd.read_csv(predict_csv).values
    print(df_test)
    print(predict)
    label = df_test[key].values
    # predict_multi_binary = np.zeros((predict.shape[0] , predict.shape[1]))
    out_texts = "Chexpert all results\n"
    auc_texts = "-AUCs-\n"
    acc_texts = "-ACCs-\n"
    cm_texts = "-Confusion Matrix-\n"

    aucs, thresholds = compute_auc_threshold(label, predict, n_class=len(key))
    predict_multi_binary = (predict > np.array(thresholds)).astype(int)
        
    ## For debugging ##
    # pd.DataFrame(thresholds).to_csv('thresholds.csv')
    # pd.DataFrame(predict_multi_binary).to_csv('predict_multi_binary.csv')
    # pd.DataFrame(label).to_csv('label.csv')
    
    total_accs = []
    for disease, auc in zip(key, aucs):
        auc_texts += f"{disease}: {auc:.3f}\n"
        acc_i = accuracy_score(label[:, key.index(disease)], predict_multi_binary[:, key.index(disease)])
        acc_texts += f"{disease}: {acc_i:.3f}\n"
        total_accs.append(acc_i)
        cm_texts += f"{disease}:\n{confusion_matrix(label[:, key.index(disease)], predict_multi_binary[:, key.index(disease)])}\n"

    out_texts += f'Average AUC: {sum(aucs) / len(aucs):.3f}\n'
    out_texts += f'Average ACC: {sum(total_accs) / len(total_accs):.3f}\n'
    out_texts += auc_texts + acc_texts + cm_texts
    return out_texts

def tripple_padchest_result_merge(predict_csv, label_file_path, img_file_path):
    test_query = ['COPD signs', 'Chilaiditi sign', 'NSG tube', 'abnormal foreign body', 'abscess', 'adenopathy', 'air bronchogram', 'air fluid level', 
    'air trapping', 'alveolar pattern', 'aortic aneurysm', 'aortic atheromatosis', 'aortic button enlargement', 'aortic elongation', 'aortic endoprosthesis', 
    'apical pleural thickening', 'artificial aortic heart valve', 'artificial heart valve', 'artificial mitral heart valve', 'asbestosis signs', 
    'ascendent aortic elongation', 'atelectasis', 'atelectasis basal', 'atypical pneumonia', 'axial hyperostosis', 'azygoesophageal recess shift', 
    'azygos lobe', 'blastic bone lesion', 'bone cement', 'bone metastasis', 'breast mass', 'bronchiectasis', 'bronchovascular markings', 'bullas', 
    'calcified adenopathy', 'calcified densities', 'calcified fibroadenoma', 'calcified granuloma', 'calcified mediastinal adenopathy', 
    'calcified pleural plaques', 'calcified pleural thickening', 'callus rib fracture', 'cardiomegaly', 'catheter', 'cavitation', 
    'central vascular redistribution', 'central venous catheter', 'central venous catheter via jugular vein', 'central venous catheter via subclavian vein', 
    'central venous catheter via umbilical vein', 'cervical rib', 'chest drain tube', 'chronic changes', 'clavicle fracture', 'consolidation', 
    'costochondral junction hypertrophy', 'costophrenic angle blunting', 'cyst', 'dai', 'descendent aortic elongation', 'dextrocardia', 
    'diaphragmatic eventration', 'double J stent', 'dual chamber device', 'electrical device', 'emphysema', 'empyema', 'end on vessel', 
    'endoprosthesis', 'endotracheal tube', 'esophagic dilatation', 'exclude', 'external foreign body', 'fibrotic band', 'fissure thickening', 
    'flattened diaphragm', 'fracture', 'gastrostomy tube', 'goiter', 'granuloma', 'ground glass pattern', 'gynecomastia', 'heart insufficiency', 
    'heart valve calcified', 'hemidiaphragm elevation', 'hiatal hernia', 'hilar congestion', 'hilar enlargement', 'humeral fracture', 'humeral prosthesis', 
    'hydropneumothorax', 'hyperinflated lung', 'hypoexpansion', 'hypoexpansion basal', 'increased density', 'infiltrates', 'interstitial pattern', 
    'kerley lines', 'kyphosis', 'laminar atelectasis', 'lepidic adenocarcinoma', 'lipomatosis', 'lobar atelectasis', 'loculated fissural effusion', 
    'loculated pleural effusion', 'lung metastasis', 'lung vascular paucity', 'lymphangitis carcinomatosa', 'lytic bone lesion', 'major fissure thickening', 
    'mammary prosthesis', 'mass', 'mastectomy', 'mediastinal enlargement', 'mediastinal mass', 'mediastinal shift', 'mediastinic lipomatosis', 'metal', 
    'miliary opacities', 'minor fissure thickening', 'multiple nodules', 'nephrostomy tube', 'nipple shadow', 'nodule', 'non axial articular degenerative changes', 
    'obesity', 'osteopenia', 'osteoporosis', 'osteosynthesis material', 'pacemaker', 'pectum carinatum', 'pectum excavatum', 'pericardial effusion', 
    'pleural effusion', 'pleural mass', 'pleural plaques', 'pleural thickening', 'pneumomediastinum', 'pneumonia', 'pneumoperitoneo', 'pneumothorax', 
    'post radiotherapy changes', 'prosthesis', 'pseudonodule', 'pulmonary artery enlargement', 'pulmonary artery hypertension', 'pulmonary edema', 
    'pulmonary fibrosis', 'pulmonary hypertension', 'pulmonary mass', 'pulmonary venous hypertension', 'reservoir central venous catheter', 
    'respiratory distress', 'reticular interstitial pattern', 'reticulonodular interstitial pattern', 'rib fracture', 'right sided aortic arch', 
    'round atelectasis', 'sclerotic bone lesion', 'scoliosis', 'segmental atelectasis', 'single chamber device', 'soft tissue mass', 
    'sternoclavicular junction hypertrophy', 'sternotomy', 'subacromial space narrowing', 'subcutaneous emphysema', 'suboptimal study', 
    'superior mediastinal enlargement', 'supra aortic elongation', 'surgery', 'surgery breast', 'surgery heart', 'surgery humeral', 'surgery lung', 
    'surgery neck', 'suture material', 'thoracic cage deformation', 'total atelectasis', 'tracheal shift', 'tracheostomy tube', 'tuberculosis', 
    'tuberculosis sequelae', 'unchanged', 'vascular hilar enlargement', 'vascular redistribution', 'ventriculoperitoneal drain tube', 
    'vertebral anterior compression', 'vertebral compression', 'vertebral degenerative changes', 'vertebral fracture', 'volume loss', 'normal']
    
    with open(label_file_path, "r") as file:
        data = json.load(file) 
    
    label = []
    df = pd.read_csv(img_file_path)
    img_path = df['Path'].tolist()
    
    key = [img_p.split('/')[-1] for img_p in img_path]
    
    print(PARENT_DIR)
    for k in key:
        label += data[k]
    
    unique_label = list(set(label))
    # print(unique_label) 
    sorted_strings = sorted(unique_label, key=lambda x: (x, label.index(x)))
    # print(sorted_strings)
    sorted_strings.remove('normal')
    sorted_strings.append('normal')
    # print(sorted_strings)
    labels = [data[k] for k in key]
    # print(labels[0])
    mlb = MultiLabelBinarizer(classes=sorted_strings)
    encoded_labels = mlb.fit_transform(labels)
    # print(encoded_labels[0])
    # print(len(encoded_labels[0]))
    
    predict = pd.read_csv(predict_csv).values
    
    test_query_index = [sorted_strings.index(i) for i in test_query]
    print([sorted_strings[i] for i in test_query_index])

    out_texts = "Padchest results\n"
    auc_texts = "-AUCs-\n"
    acc_texts = "-ACCs-\n"
    cm_texts = "-Confusion Matrix-\n"

    print(encoded_labels.shape)
    print(predict.shape)

    aucs, thresholds = compute_auc_threshold(encoded_labels, predict, n_class=encoded_labels.shape[-1])
    predict_multi_binary = (predict > np.array(thresholds)).astype(int)
    
    macro_auc, micro_auc, _, _ = eval_auc(predict, encoded_labels)
    out_texts += f"Total AUC: {macro_auc}\n"
    out_texts += f"Micro AUC: {micro_auc}\n"
    
    micro_prc, macro_prc = calculate_micro_macro_auprc(encoded_labels, predict)
    out_texts += f"Micro AUPRC: {micro_prc:.3f}, Macro AUPRC: {macro_prc:.3f}\n"

    total_accs = []
    for i in test_query_index:
        macro_auc, _, _, _ = eval_auc(predict[:, i], encoded_labels[:, i])
        auc_texts += f"{sorted_strings[i]} AUC: {macro_auc:.3f}\n"
        acc_i = accuracy_score(encoded_labels[:, i], predict_multi_binary[:, i])
        total_accs.append(acc_i)
        acc_texts += f"{sorted_strings[i]} ACC: {acc_i:.3f}\n"
        cm_i = confusion_matrix(encoded_labels[:, i], predict_multi_binary[:, i])
        cm_texts += f"{sorted_strings[i]}:\n{cm_i}\n"

    out_texts += f'Average ACC: {sum(total_accs) / len(total_accs):.3f}\n'
        
    out_texts += auc_texts + acc_texts + cm_texts

    return out_texts


if __name__ == '__main__':    

    ### args ###
    parser = argparse.ArgumentParser(description="Argument parser for OFF-CLIP model validation")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--save_name", type=str, help="Name of the result file(experiment name; model name).")
    parser.add_argument(
        "-c",
        "--config",
        metavar="base_config.yaml",
        help="paths to base config",
        required=True
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.config)

    ### paths ###
    result_file_name =  args.save_name
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)

    ### Load Model ###
    print(f"loading {args.weight_path} ...")
    origin_ckpt_path = f"{PARENT_DIR}/checkpoint/paper/Carzero/CARZero_best_model.ckpt" 
    trained_ckpt_path = args.weight_path

    images = [ 
            f'{PARENT_DIR}/dataset/open-I/openi_multi_label_image.csv',
            f'{PARENT_DIR}/dataset/Chexpert/chexlocalize/CheXpert/chexpert_all_test_image.csv',
            f'{PARENT_DIR}/dataset/padchest/padchest_multi_label_image_new.csv',
            ]
    

    texts = [ 
            f'{PARENT_DIR}/dataset/open-I/openi_multi_label_text.json',
            f'{PARENT_DIR}/dataset/Chexpert/chexlocalize/CheXpert/chexpert_all_test_text.json',
            f'{PARENT_DIR}/dataset/padchest/padchest_multi_label_text.json',
            ]
    
    os.makedirs(f'{PARENT_DIR}/output/offclip/', exist_ok = True)
    save_csvs = [    
                f'{PARENT_DIR}/output/offclip/'+result_file_name+'_open_I_similarities.csv',
                f'{PARENT_DIR}/output/offclip/'+result_file_name+'_chexpert_similarities.csv',
                f'{PARENT_DIR}/output/offclip/'+result_file_name+'_padchest_similarities.csv'
                ]
    
    test_csvs = [
                f'{PARENT_DIR}/dataset/open-I/custom.csv',
                f'{PARENT_DIR}/dataset/Chexpert/chexlocalize/CheXpert/test_labels.csv',
                f'{PARENT_DIR}/dataset/padchest/manual_image.json'
                ]
    
    experiment_result_text = f'Date: {time.ctime()}\nloaded check point: {trained_ckpt_path}\n\n'
    
    model = offclip.load_offclip_validation(cfg, trained_ckpt_path)
    model.eval()
    for i, (img, txt, savecsv) in  enumerate(zip(images, texts, save_csvs)):
        start = time.time()
        print(f'img: {img}')
        print(f'txt: {txt}')
        similarities = obtain_simr(img, txt, model) # model 을 inference 하는 부분.
        similarities.to_csv(savecsv, index=False)
        print(f'similarities calculated for {time.time() - start:.3f}, saved to {savecsv}')

    ### vindr ###
    vindr_cxr_img_root = f'{PARENT_DIR}/dataset/vindr-cxr/files/vindr-cxr/1.0.0/test'
    vindr_cxr_test_csv = f'{PARENT_DIR}/dataset/vindr-cxr/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv'
    vindr_save = f'{PARENT_DIR}/output/offclip/vindr-cxr/'+result_file_name+'_vindr_similarities.csv'

    model = offclip.load_offclip_validation(cfg, trained_ckpt_path)
    similar, vindr_result_txt = vindr_cxr_obtain_simr(vindr_cxr_img_root, vindr_cxr_test_csv, model)

    experiment_result_text += tripple_openi_rusult_merge(save_csvs[0], test_csvs[0])
    experiment_result_text += triple_Chexpert_all_result(save_csvs[1], test_csvs[1])
    experiment_result_text += tripple_padchest_result_merge(save_csvs[2], test_csvs[2], images[2])
    experiment_result_text += vindr_result_txt

    with open(f'{PARENT_DIR}/output/offclip/'+result_file_name+f'_validation_result.txt', 'w') as f:
        f.write(experiment_result_text)
    