import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
import pdb
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

flag1, flag2, flag3, flag4 = 0, 0, 0,0 
objective_gradient_dict = {}
unlearn_gradient_dict = {}
retain_gradient_dict = {}
risk_dict = {}
weight_dict = {}

def risk_report(risks):
    global flag3
    flag3 += 1
    for key in risks:
        if key not in risk_dict:
            risk_dict[key]=risks[key]
        else: 
            risk_dict[key]=.6*risk_dict[key]+.4*risks[key]
    if flag3 %4==3: 
        print('risk | ' + (''.join(['%s: %.4f ' % (key, risk_dict[key]) for key in risk_dict])))

def weight_report(weights):
    global flag4
    flag4 += 1
    for key in weights:
        if key not in risk_dict:
            weight_dict[key]=weights[key]
        else: 
            weight_dict[key]=.6*weight_dict[key]+.4*weights[key]

    if flag4 %4== 3: 
        print('weight | ' + (''.join(['%s: %.4f ' % (key, weight_dict[key]) for key in weight_dict])))
        
def objective_gradient_fcn(model, risk):
    risk.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name not in objective_gradient_dict:
                objective_gradient_dict[name] = param.grad.view(-1).cpu() if param.grad is not None else torch.zeros_like(param).cpu()
            else: 
                cur_grads = param.grad.view(-1).cpu() if param.grad is not None else torch.zeros_like(param).cpu()
                objective_gradient_dict[name] = .6*objective_gradient_dict[name]+.4*cur_grads
    model.zero_grad()


def unlearn_gradient_fcn(model, risk):
    risk.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name not in unlearn_gradient_dict:
                unlearn_gradient_dict[name] = param.grad.view(-1).cpu()
            else: 
                unlearn_gradient_dict[name] = .6*unlearn_gradient_dict[name]+.4* param.grad.view(-1).cpu()
    model.zero_grad()

def retain_gradient_fcn(model, risk):
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    risk.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in retain_gradient_dict:
                retain_gradient_dict[name] = param.grad.view(-1).cpu()
            else: 
                retain_gradient_dict[name] = .6*retain_gradient_dict[name]+.4* param.grad.view(-1).cpu()
    model.zero_grad()

def gradient_unlearn_effect():
    global flag1
    flag1 += 1
    if flag1 % 4==3: 
        total_ge = 0
        ll_ge = {}
        for name in objective_gradient_dict:
            layer_ge = (objective_gradient_dict[name].cuda() * unlearn_gradient_dict[name].cuda()).sum()
            if 'model.embed_tokens' in name:
                key = 'embed'
            if 'module.model.layers.' in name:
                if int(name.split('.')[3]) < 10:
                    key = 'module1'
                elif int(name.split('.')[3]) < 20:
                    key = 'module2'
                else: 
                    key = 'module3'
            elif 'lm_head' in name:
                key = 'lm'
            total_ge += layer_ge.item()
            if key not in ll_ge:
                ll_ge[key] = layer_ge.item()
            else: 
                ll_ge[key] += layer_ge.item()
        print('ge u |', total_ge, ll_ge)

def gradient_retain_effect():
    global flag2
    flag2 += 1
    if flag2 %4 == 3: 
        total_ge = 0
        ll_ge = {}
        for name in objective_gradient_dict:
            layer_ge = (objective_gradient_dict[name].cuda() * retain_gradient_dict[name].cuda()).sum()
            if 'model.embed_tokens' in name:
                key = 'embed'
            if 'module.model.layers.' in name:
                if int(name.split('.')[3]) < 10:
                    key = 'module1'
                elif int(name.split('.')[3]) < 20:
                    key = 'module2'
                else: 
                    key = 'module3'
            elif 'lm_head' in name:
                key = 'lm'
            total_ge += layer_ge.item()
            
            if key not in ll_ge:
                ll_ge[key] = layer_ge.item()
            else: 
                ll_ge[key] += layer_ge.item()
        
        print('ge r |', total_ge, ll_ge)
        
        
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):# the first ti
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.org_ckpt=kwargs.pop('ckpt_org')
        self.hyper_param=kwargs.pop('hyper_param')
        self.max_steps=kwargs.pop('max_steps')
        self.count_step=0
        self.rmu_noise=torch.rand((1,1,4096)).cuda()
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if 'KL' in self.loss_type or 'npo' in self.loss_type:
            try:
                self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
            except: 
                self.oracle_model = self.oracle_model.cuda()

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    

    def compute_loss(self, model, inputs, return_outputs=False):
        

        if self.loss_type == 'npo':
            self.beta = self.hyper_param
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            
            # objective gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_loss_oracle = get_batch_loss(forget_outputs_oracle.logits, labels)
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            npo_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 
            objective_gradient_fcn(model, npo_loss)
            
            # retain gradient
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # reporter
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': npo_loss.item()})
            
            labels = labels.to(outputs.logits.device)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            npo_weights = (2 / (1 + (neg_log_ratios.exp() ** self.beta))).mean().item()
            probs = (- CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))).exp().detach()# .clamp(max=.9,min=.1)
            ce_weights  = (1 / (probs + 1e-20)).mean().item()
            weight_report({'npo': npo_weights, 'ce': ce_weights})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
        
        
        if self.loss_type == 'ins_npo':
            
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            # objective gradient
            self.beta = self.hyper_param
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            with torch.no_grad():
                outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                shift_logits_oracle = outputs_oracle.logits[..., :-1, :].contiguous()
            ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ce_oracle = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits_oracle.view(-1, shift_logits_oracle.size(-1)), shift_labels.view(-1))
            probs, probs_oracle = (-ce).exp().detach(), (-ce_oracle).exp().detach()
            npo_weights = 2 * probs ** self.beta / (probs ** self.beta + probs_oracle ** self.beta + 1e-5)
            loss = -(npo_weights * ce)[shift_labels.view(-1)!=-100].mean()
            objective_gradient_fcn(model, loss)
            
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # reporter
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': loss.item()})
            weight_report({'npo': npo_weights.mean().item()})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
        if self.loss_type == 'w_ins_npo':
            self.beta = 0.1 
            self.alpha = self.hyper_param
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            # objective gradient
            self.beta = self.hyper_param
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            with torch.no_grad():
                outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                shift_logits_oracle = outputs_oracle.logits[..., :-1, :].contiguous()
            ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ce_oracle = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits_oracle.view(-1, shift_logits_oracle.size(-1)), shift_labels.view(-1))
            probs, probs_oracle = (-ce).exp().detach(), (-ce_oracle).exp().detach()
            npo_weights = 2 * probs ** (self.beta + self.alpha) / (probs ** self.beta + probs_oracle ** self.beta + 1e-5)
            loss = -(npo_weights * ce)[shift_labels.view(-1)!=-100].mean()
            objective_gradient_fcn(model, loss)
            
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # reporter
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': loss.item()})
            weight_report({'npo': npo_weights.mean().item()})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
            
        if self.loss_type == "ga":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            # objective gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            ga_loss = - outputs.loss
            objective_gradient_fcn(model, ga_loss)
            
            # retain gradient
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # reporter
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': ga_loss.item()})
            
            labels = labels.to(outputs.logits.device)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            probs = (- CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))).exp().detach()# .clamp(max=.9,min=.1)
            ce_weights  = (1 / (probs + 1e-20)).mean().item()
            weight_report({'ce': ce_weights})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
            
        if self.loss_type == "wga":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            # objective gradient
            self.beta = self.hyper_param
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            labels = labels.to(outputs.logits.device)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            weights = (- lm_loss).exp().detach() ** self.beta
            ga_loss = -(weights * lm_loss)[shift_labels.view(-1)!=-100].mean()
            objective_gradient_fcn(model, ga_loss)
            
            # retain gradient
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # reporter
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': ga_loss.item()})
            weight_report({'wga': weights.mean()})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
            
        elif 'rmu_' in self.loss_type:
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            emb_tar = self.rmu_noise * self.hyper_param
            emb_idx = int(self.loss_type.split('_')[-1]) # 32 21 10
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            
            # retain gradient
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # objective gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask, output_hidden_states=True)
            emb_dif = ((outputs.hidden_states[emb_idx][..., :-1, :] - emb_tar) ** 2).mean()
            # emb_dif = ((outputs.hidden_states[emb_idx][..., :-1, :] - emb_tar).abs()).mean(-1)
            forget_loss = emb_dif[labels[..., 1:] != -100].mean() + 0 * outputs.loss
            objective_gradient_fcn(model, forget_loss)
            
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': forget_loss.item()})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
        
        elif 'rmu_mask_' in self.loss_type:
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            # emb_tar = self.rmu_noise * self.hyper_param
            emb_mask = (self.rmu_noise > .5).float()
            emb_idx = int(self.loss_type.split('_')[-1]) # 32 21 10
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            
            # retain gradient
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # objective gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask, output_hidden_states=True)
            emb_dif = ((outputs.hidden_states[emb_idx][..., :-1, :] - outputs.hidden_states[emb_idx][..., :-1, :].detach() * emb_mask) ** 2).mean(-1) 
            forget_loss = emb_dif[labels[..., 1:] != -100].mean()
            objective_gradient_fcn(model, forget_loss)
            
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': forget_loss.item()})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
                    
        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            # objective gradient
            outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            idk_loss = outputs.loss
            objective_gradient_fcn(model, idk_loss)
            risk_report({'unlearn': idk_loss.item()})
            
            # retain gradient
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)

            gradient_retain_effect()
            
            outputs_idk = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            loss = outputs_idk.loss * 0
        
        if self.loss_type == 'npo_temp':
            self.beta = self.hyper_param
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            # unlearn gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            unlearn_metric = outputs.loss
            unlearn_gradient_fcn(model, unlearn_metric)
            
            
            # objective gradient
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_loss_oracle = get_batch_loss(forget_outputs_oracle.logits, labels)
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            npo_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
            npo_loss = - outputs.loss
            objective_gradient_fcn(model, npo_loss)
            
            # retain gradient
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_metric = retain_outputs.loss
            retain_gradient_fcn(model, retain_metric)
            
            # reporter
            gradient_unlearn_effect()
            gradient_retain_effect()
            risk_report({'unlearn': npo_loss.item()})
            
            labels = labels.to(outputs.logits.device)
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            npo_weights = (2 / (1 + (neg_log_ratios.exp() ** self.beta))).mean().item()
            probs = (- CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))).exp().detach()# .clamp(max=.9,min=.1)
            ce_weights  = (1 / (probs + 1e-20)).mean().item()
            weight_report({'npo': npo_weights, 'ce': ce_weights})
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * 0
            
        return (loss, outputs) if return_outputs else loss
        
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                # print('dataset condition: ', len(eval_dataloader.dataset), self.accelerator.local_process_index)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 
                # if 'eval_log' not in eval_task:
                #     normalize_gt = True

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            
                #wait for all process to finish
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)

                            #delete old files use shutil

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                # aggregated_eval_logs = interleave_eval_result_dict(aggregated_eval_logs, forget_rate, large_bsz=eval_cfg.batch_size, num_processes=world_size)
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)

def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


