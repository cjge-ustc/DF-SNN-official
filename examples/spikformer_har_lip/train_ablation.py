import argparse
import json
import math
from pathlib import Path
import sys
import torch
torch.set_num_threads(1)
import train_transformer as training
from ablation_models import make_model, effective_masses
from aligned_glmulti import make_model as aligned_model

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--implementation', choices=['wrapper', 'native'], required=True)
parser.add_argument('--policy-json', required=True)
choice, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
args = training.parse_args()
args.implementation = 'cupy'
policy=json.loads(choice.policy_json)
import aligned_glmulti
aligned_glmulti.HISTORY_RHO=policy['history_rho']
args.coefficient_policy=policy
assert args.epochs == 500 and args.cooldown_epochs == 0
assert len(args.alpha)==3 and args.cupy
assert args.step_size == .1 and args.time_interval == .1 and args.batch_size == 32
assert not args.learn_alpha and args.coefficient_constraint == 'softplus'
beta = list(args.alpha)
masses = list(args.multi_coefficient)
assert len(beta) == len(masses) and all(0 < x <= 1 for x in beta)
training.alpha, training.multi_coefficient = beta, masses
training.args = args
def factory(name, **kwargs):
    net = aligned_model('cupy',beta,args.learn_coefficient,masses,batch=args.batch_size,
        T=args.T,fixed_gain=False,num_classes=args.num_classes).to(args.device)
    coefficient_params = [p for n,p in net.named_parameters() if 'multi_coefficient' in n]
    assert len(coefficient_params) == int(args.learn_coefficient)
    assert torch.allclose(effective_masses(net), torch.tensor(masses,device=args.device), atol=1e-7)
    net.beta_regularization=lambda: regularization(net)
    print('ACTUAL_BACKEND=cupy COEFFICIENT_HISTORY_GRAD=True state_history_rho=',policy['history_rho'],'beta',beta,'initial_masses',masses,
        'learn',args.learn_coefficient,'history=full',flush=True)
    return net
training.create_model = factory
def regularization(net):
    m=effective_masses(net).float();m0=m.new_tensor(policy['masses'])
    p=m/m.sum();p0=m0/m0.sum()
    return (policy['anchor']*(m.sum()/m0.sum()-1).square()
        +policy['balance']*(p0*(p0.log()-p.clamp_min(1e-12).log())).sum())
original=training.train_one_epoch
def train_epoch(model,criterion,optimizer,data_loader,device,epoch,*rest,**kwargs):
    net=model.module if hasattr(model,'module') else model
    enabled=policy['learn'] and policy['learn_start']<=epoch<policy['learn_end']
    lr=policy['coefficient_lr'] if enabled else 0.
    if enabled and epoch>=policy['decay_start']:
        fraction=(epoch-policy['decay_start'])/(policy['learn_end']-policy['decay_start'])
        lr*=.5*(1+math.cos(math.pi*fraction))
    for n,p in net.named_parameters():
        if 'multi_coefficient' in n:p.requires_grad_(enabled)
    for group in optimizer.param_groups:
        if group.get('shared_coefficient'):group['lr']=lr
    before=effective_masses(net).detach().clone()
    result=original(model,criterion,optimizer,data_loader,device,epoch,*rest,**kwargs)
    after=effective_masses(net).detach();proportions=after/after.sum()
    record=dict(epoch=epoch+1,learning=enabled,coefficient_lr=lr,masses=after.cpu().tolist(),
        proportions=proportions.cpu().tolist(),max_share=float(proportions.max()),
        update=(after-before).cpu().tolist(),regularization=float(regularization(net).detach()))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'coefficient_policy.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print('COEFFICIENT_POLICY',json.dumps(record),flush=True)
    return result
training.train_one_epoch=train_epoch
training.main(args)
