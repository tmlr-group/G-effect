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
    save_steps = 1000
    epoch = 1
elif args.setting == 'forget05': 
    save_steps = 1000
    epoch = 1
elif args.setting == 'forget01': 
    epoch = 1
    save_steps = 1000
else: 
    raise RuntimeError()
if args.model == 'phi':
    lr = 2e-5
    lr_str = '2e-05'
    model = 'phi'
elif args.model == 'llama':
    lr = 1e-5
    lr_str = '1e-05'
    model = 'llama2-7b'
else: raise RuntimeError()

param = args.hyper 
if args.model == 'phi':
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_id} torchrun --nproc_per_node=1 --master_port={random.randint(0,60000)} forget2_ge.py --config-name=forget_ge.yaml split={args.setting} model_family=phi       lr={lr} forget_loss={args.loss} save_steps={save_steps}  hyper_param={param} num_epochs={epoch}')
elif args.model == 'llama':
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_id} torchrun --nproc_per_node=1 --master_port={random.randint(0,60000)} forget2_ge.py --config-name=forget_ge.yaml split={args.setting} model_family=llama2-7b lr=1e-5 forget_loss={args.loss} save_steps={save_steps}  hyper_param={param} num_epochs={epoch}')
time.sleep(1)
cap = 62 if args.setting=='forget05' else 130
for iteration in range(5,cap,5):
    if args.loss == 'idk':
        path = f'icml/{model}/{args.loss}_{lr_str}_{args.setting}_5_0.0_{param}/checkpoint-' + ('%d' % iteration)
    else: 
        path = f'icml/{model}/{args.loss}_{lr_str}_{args.setting}_5_0.0_{param}/checkpoint-' + ('%d' % iteration)
    if args.model == 'phi':
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_id} torchrun --nproc_per_node=1 --master_port={random.randint(0,60000)} forget2_ge.py --config-name=forget_ge.yaml split={args.setting} model_family=phi       lr={lr} forget_loss={args.loss} save_steps={save_steps}  hyper_param={param} num_epochs={epoch} model_path_cur={path}')
    elif args.model == 'llama':
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_id} torchrun --nproc_per_node=1 --master_port={random.randint(0,60000)} forget2_ge.py --config-name=forget_ge.yaml split={args.setting} model_family=llama2-7b lr=1e-5 forget_loss={args.loss} save_steps={save_steps}  hyper_param={param} num_epochs={epoch} model_path_cur={path}')
    time.sleep(1)


