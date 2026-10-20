import os
import random
import numpy as np
import torch


def rng_state(train_gen, test_gen):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(), train=train_gen.get_state(), test=test_gen.get_state())


def restore_rng(state, train_gen, test_gen):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])
    train_gen.set_state(state['train']); test_gen.set_state(state['test'])


def load_model(net, state):
    canonical = lambda key: '.'.join(part for part in key.split('.') if part != '_orig_mod')
    source = {canonical(key): value for key, value in state.items()}
    target = {canonical(key): key for key in net.state_dict()}
    if len(source) != len(state) or len(target) != len(net.state_dict()) or source.keys() != target.keys():
        raise RuntimeError('Checkpoint key mismatch after strict compile-prefix normalization')
    net.load_state_dict({target[key]: value for key, value in source.items()}, strict=True)
