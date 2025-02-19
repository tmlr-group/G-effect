# Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond

TBD


Before running the codes, we need to have some pretrained models, which can be achieved folloing the guidelines of the original TOFU repo. We also need to change the model_path and save_dir (icml/ by default) for the yaml files in config/ (e.g., config/forget.yaml and config/forget_ge.yaml)

run basher1.py for the original unlearning methods with different checkpoints, automatically saving into the folder named icml.

run basher2.py for the computation of g-effects for the save checkpoints, the results can be save to a particular file. 

For example,

python basher1.py ga --model=llama --cuda_id=3 --setting=forget05 --hyper=2 

python basher2.py ga --model=llama --cuda_id=3 --setting=forget05 --hyper=2 > ga_ge_log.txt

the current support methods are 

ga, npo, ins_npo (tnpo), w_ins_npo (wtnpo), wga, rmu_\[particular layer to be perturbed, e.g., rmu_32, rmu_21, rmu_10\],  idk (po)

the current supoort models are 

llama, phi

the current support settings are

forget01, forget05, and forget10

