# an example: 
# os.system('CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=18149 evaluation_everything.py model_family=llama2-7b split=forget01 model_path=iclr/llama2-7b/gd_att__1e-05_forget01_8_0.0_2_0.1/checkpoint-25 ps_type=similar')
# note: add a property of 'ps_type' in config/eval_everything.yaml, taking values from exact, perturb, and similar
import pdb, os, hydra
import logging
import random,time
import numpy as np
import sklearn.metrics as sk

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoConfig
from utils import get_model_identifiers_from_yaml
from tqdm import tqdm
from torch.utils.data import Subset

import safetensors
import json 
import math
recall_level_default = 0.95
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def verify_and_report_dataset(data_path, split, filename="dataset_samples.json"):
    """Load, compare, save, and print the first, middle, and last entries of a dataset."""
    # Load the dataset
    dataset = load_dataset(data_path, split)["train"]
    current_samples = {
        "first": dataset[0],
        "middle": dataset[len(dataset) // 2],
        "last": dataset[-1]
    }

    try:
        # Try to load the previous samples
        with open(filename, 'r') as file:
            previous_samples = json.load(file)
    except FileNotFoundError:
        # If no file exists, save the current samples and exit
        with open(filename, 'w') as file:
            json.dump(current_samples, file, indent=4)
        print("No previous samples found. Current samples saved.")
        return

    # Compare the current samples with the previously saved samples and print details
    consistent = True
    for key in ["first", "middle", "last"]:
        print(f"Checking {key} entry:")
        print(f"Current: {current_samples[key]}")
        print(f"Previous: {previous_samples[key]}")
        if current_samples[key] != previous_samples[key]:
            print("Warning: Entries do not match.")
            consistent = False

    if consistent:
        print("All entries match. Dataset order is consistent with previous load.")
    else:
        # Update the file with the new samples if inconsistencies are found
        with open(filename, 'w') as file:
            json.dump(current_samples, file, indent=4)
        print("Inconsistencies found. Updated the samples with current dataset entries.")


def extract_indices_by_ratio(filename, ge_type, ratio=50, order="top"):
    """
    Extract the forget_idx and retain_idx lists from a specified percentage of entries
    based on the rank of ge_type ('ge_u' or 'ge_r').

    :param filename: Path to the JSON file containing step details.
    :param ge_type: Specify 'ge_u' or 'ge_r' to select based on ge_u_rank or ge_r_rank.
    :param ratio: The percentage of entries to return (default 50%).
    :param order: Select from 'top' or 'bottom' entries based on ranking.
    :return: Tuple containing lists of forget_idx and retain_idx.
    """
    # Load the data from JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    # Determine the rank key based on ge_type
    rank_key = f"{ge_type}_rank"

    # Extract entries and sort them by the specified ge_type rank
    sorted_entries = sorted(data.values(), key=lambda x: x[rank_key], reverse=(order == "bottom"))

    print("len:",len(sorted_entries))
    # Calculate the cutoff index based on the specified ratio
    #cutoff_index = int(len(sorted_entries) * (ratio / 100))
    cutoff_index = math.ceil(len(sorted_entries) * (ratio / 100))
    print("cutoff_index:",cutoff_index)

    # Extract the specified ratio of entries
    selected_entries = sorted_entries[:cutoff_index]

    # Extract forget_idx and retain_idx from these entries
    forget_idx_list = [entry['forget_idx'] for entry in selected_entries]
    retain_idx_list = [entry['retain_idx'] for entry in selected_entries]

    print("Selected Forget indices:",forget_idx_list)
    print("Selected Retain indices:", retain_idx_list)

    return (forget_idx_list, retain_idx_list)

def sample_data(dataset, num_samples=50):
    """Randomly samples the specified number of samples from the dataset or returns the original dataset if it contains fewer items than requested."""
    if len(dataset) <= num_samples:
        return dataset  # Return the original dataset if it's smaller than the requested sample size
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    return subset

def model_mix(model,before,after,update_ratio):
    for name,parameter in model.named_parameters():
        parameter.data=update_ratio*before[name[:]].cuda()+(1-update_ratio)*after[name[:]].cuda()
    return model    


@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):

    # setting the log #######
    log_file_path = f'./logs/{log_file_directory}.log'

    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    
    logger = logging.getLogger(log_file_directory)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

    logger.info('split: %s' % cfg.split)
    logger.info('model_path: %s' % cfg.model_path)

    #######################


    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'

    max_length = 500
    batch_size = cfg.batch_size

    model = None
    config = AutoConfig.from_pretrained(model_id)

    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                except:
                    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                except:
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    
    
    root_path = "/data/weight"
    if model_id=='microsoft/phi-1_5':
        before_ckpt=safetensors.torch.load_file(root_path+'/ft_epoch5_lr2e-05_phi_full_wd0.0/checkpoint-625/model.safetensors')
        after_ckpt=safetensors.torch.load_file(cfg.model_path+'/model.safetensors')
    if model_id=='NousResearch/Llama-2-7b-chat-hf':
        before_ckpt_1=safetensors.torch.load_file(root_path+'/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00001-of-00003.safetensors')
        before_ckpt_2=safetensors.torch.load_file(root_path+'/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00002-of-00003.safetensors')
        before_ckpt_3=safetensors.torch.load_file(root_path+'/ft_epoch5_lr1e-05_llama2-7b_full_wd0.0/checkpoint-625/model-00003-of-00003.safetensors')
        before_ckpt={**before_ckpt_1,**before_ckpt_2,**before_ckpt_3}
        after_ckpt1=safetensors.torch.load_file(cfg.model_path+'/model-00001-of-00003.safetensors')
        after_ckpt2=safetensors.torch.load_file(cfg.model_path+'/model-00002-of-00003.safetensors')
        after_ckpt3=safetensors.torch.load_file(cfg.model_path+'/model-00003-of-00003.safetensors')
        after_ckpt={**after_ckpt1,**after_ckpt2,**after_ckpt3}
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    
    if cfg.split=='forget10':
        retain_name='retain90'
    elif  cfg.split=='forget05':
        retain_name='retain95'
    elif  cfg.split=='forget01':
        retain_name='retain99'

    
    def string2token(strings):
        tks = [tokenizer.encode(_, add_special_tokens=True, return_tensors='pt').to(model.device)[0] for _ in strings]
        tk_lens = [_.size(0) for _ in tks]
        return {'token': tks, 'length': tk_lens}
    def token2string(tokens):
        strs = [tokenizer.decode(_, skip_special_tokens=True) for _ in tokens]
        return strs
    def lcs(s1,s2):
        a = [[None for i in range(len(s2))] for j in range(len(s1))]
        def _lcs(s1, s2, s1Index, s2Index, arr):
            if s1Index ==-1 or s2Index == -1:
                return 0
            if(arr[s1Index][s2Index] != None):
                return arr[s1Index][s2Index]
            if s1[s1Index] == s2 [s2Index]:
                result = 1+ _lcs(s1, s2, s1Index -1, s2Index -1, arr)
            else:
                result= max(_lcs(s1, s2, s1Index -1, s2Index, arr), _lcs(s1, s2, s1Index, s2Index -1, arr))
            arr[s1Index][s2Index] = result
            return result 
        return _lcs(s1, s2, len(s1)-1, len(s2)-1, a)

    from tqdm import tqdm

    def processing(loader, model):
            ps_list = []
            for idx, s in tqdm(enumerate(loader), desc="Processing items", total=len(loader)):
                if cfg.ps_type == 'perturb':
                    ques, anws = s['paraphrased_question'], s['answer']
                else:
                    ques, anws = s['question'], s['answer'] 
                fuls = [f"### Question: {que}\n ### Answer: {ans}" for que, ans in zip(ques, anws)]
                _ques_tks_and_lens, _fuls_tks_and_lens = string2token(ques), string2token(fuls)
                ques_tks, ques_tks_lens = _ques_tks_and_lens['token'], _ques_tks_and_lens['length']
                fuls_tks, fuls_tks_lens = _fuls_tks_and_lens['token'], _fuls_tks_and_lens['length']
                left_bar, right_bar = [_ for _ in ques_tks_lens], [_ for _ in fuls_tks_lens]
                
                for _num_attempt_ in range(max([b - a for a, b in zip(left_bar, right_bar)])):
                    mid_bar = [(a + b) // 2 for a, b in zip(left_bar, right_bar)]
                    if _num_attempt_ != 0: 
                        if sum([int(l==r) for l, r in zip(mid_bar, old_mid_bar)]) == len(old_mid_bar): break
                    can_strings = token2string([tk[:cur] for cur, tk in zip(mid_bar, fuls_tks)])
                    inputs = tokenizer.batch_encode_plus(can_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
                    preds_tks = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=200, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    _preds_tks_and_lens = string2token(tokenizer.batch_decode(preds_tks, skip_special_tokens=True))
                    pred_tks, pred_tks_lens = _preds_tks_and_lens['token'], _preds_tks_and_lens['length']
                    
                    pred_tks_ = [pred_tks[idx][mid_bar[idx]:len(fuls_tks[idx])] for idx in range(len(fuls_tks))]
                    fuls_tks_ = [fuls_tks[idx][mid_bar[idx]:] for idx in range(len(fuls_tks))]
                    if cfg.ps_type == 'similar':
                        match = [lcs(p,f) >= 0.5 * len(p) for p, f in zip(pred_tks_, fuls_tks_)]
                    else:
                        match = [sum([int(a == b) for a, b in zip(p, f)]) == len(p) for p, f in zip(pred_tks_, fuls_tks_)]
                    left_bar  = [left if match else mid  for match,  left, mid in zip(match,  left_bar, mid_bar)]
                    right_bar = [mid if match else right for match, right, mid in zip(match, right_bar, mid_bar)]
                    old_mid_bar = mid_bar
                ps_list += [1- (m-l)/(r-l) for l, m, r in zip(ques_tks_lens, right_bar, fuls_tks_lens)]
            return ps_list

    
    
    for ps_type in ['perturb', 'exact']:
        cfg.ps_type = ps_type
        logger.info('ps_type: %s' % cfg.ps_type)

        # getting data ################
        if cfg.ps_type == 'perturb':
            retain_eval_data=load_dataset('locuslab/TOFU','retain_perturbed')['train']
            forget_data=load_dataset('locuslab/TOFU',cfg.split+'_perturbed')['train']
        else:
            retain_eval_data=load_dataset('locuslab/TOFU',retain_name)['train'].train_test_split(train_size=400,shuffle=False)['train']
            forget_data=load_dataset('locuslab/TOFU',cfg.split)['train']

        verify_and_report_dataset('locuslab/TOFU',cfg.split)

        if "bottom" in cfg.model_path or "top" in cfg.model_path:
            parts = cfg.model_path.split("/")[-2].split("_")
            ge_type = "ge_"+parts[-3]
            ratio = int(parts[-2])  # Select the top or bottom 30%
            order = parts[-1] # Choose "top" or "bottom"
            ranking_table = "updated_step_details.json"
            forget_idx, _ = extract_indices_by_ratio(ranking_table, ge_type, ratio, order)
            # retain_eval_data = retain_eval_data.select(forget_select_indices)
            forget_data = forget_data.select(forget_idx)

        # sampling
        retain_eval_data = sample_data(retain_eval_data, 50)
        forget_data = sample_data(forget_data, 50)
    
        retain_eval_loader=torch.utils.data.DataLoader(retain_eval_data,batch_size=50)
        forget_loader=torch.utils.data.DataLoader(forget_data,batch_size=50)
        ################################

        model=model_mix(model,before_ckpt,after_ckpt,0)
        ps_forget_u = processing(forget_loader, model)
        ps_forget_u = sum(ps_forget_u) / len(ps_forget_u)
        ps_retain_u = processing(retain_eval_loader, model)
        ps_retain_u = sum(ps_retain_u) / len(ps_retain_u)
        logger.info('unlearned model: ps retain %.4f forget %.4f | retain bar %.4f' 
                % (ps_retain_u, ps_forget_u, ps_retain_u * cfg.ps_p))


        model=model_mix(model,before_ckpt,after_ckpt,1)
        ps_forget_o = processing(forget_loader, model)
        ps_forget_o = sum(ps_forget_o) / len(ps_forget_o)
        ps_retain_o = processing(retain_eval_loader, model)
        ps_retain_o = sum(ps_retain_o) / len(ps_retain_o)
        logger.info('original model: ps retain %.4f forget %.4f | retain bar %.4f' 
                % (ps_retain_o, ps_forget_o, ps_retain_o * cfg.ps_p))

        logger.info('\n' + '~' * 80)




if __name__ == "__main__":
    main()