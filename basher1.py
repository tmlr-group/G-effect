import os, random, argparse, time
parser = argparse.ArgumentParser(description='DAL training procedure on the CIFAR benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('loss', type=str,
                    help='npo npo2 npov2 - v5')
parser.add_argument('--setting', type=str,
                    help='forget01 forget05 forget10')
parser.add_argument('--model', type=str,
                    help='phi llama')
parser.add_argument('--cuda_id', type=int,
                    help='0~7')
parser.add_argument('--hyper', type=int)
args = parser.parse_args()

if args.setting == 'forget10': 
    save_steps = 5
    epoch = 5
elif args.setting == 'forget05': 
    save_steps = 5
    epoch = 5
elif args.setting == 'forget01': 
    epoch = 5
    save_steps = 5
else: 
    raise RuntimeError()
if args.model == 'phi':
    lr = 2e-5
elif args.model == 'llama':
    lr = 1e-5
else: raise RuntimeError()

for param in [args.hyper]:
    if args.model == 'phi':
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_id} torchrun --nproc_per_node=1 --master_port={random.randint(0,60000)} forget2.py --config-name=forget.yaml split={args.setting} model_family=phi       lr={lr} forget_loss={args.loss} save_steps={save_steps}  hyper_param={param} num_epochs={epoch}')
    elif args.model == 'llama':
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_id} torchrun --nproc_per_node=1 --master_port={random.randint(0,60000)} forget2.py --config-name=forget.yaml split={args.setting} model_family=llama2-7b lr={lr} forget_loss={args.loss} save_steps={save_steps}  hyper_param={param} num_epochs={epoch}')
    time.sleep(1)


