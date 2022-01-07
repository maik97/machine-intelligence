from torch import optim

def get_optim(name, params, lr, *args, **kwargs):

    if name == 'Adadelta':
        return optim.Adadelta(params, lr, *args, **kwargs)
    elif name == 'Adagrad':
        return optim.Adagrad(params, lr, *args, **kwargs)
    elif name == 'Adam':
        return optim.Adam(params, lr, *args, **kwargs)
    elif name == 'AdamW':
        return optim.AdamW(params, lr, *args, **kwargs)
    elif name == 'SparseAdam':
        return optim.SparseAdam(params, lr, *args, **kwargs)
    elif name == 'Adamax':
        return optim.Adamax(params, lr, *args, **kwargs)
    elif name == 'ASGD':
        return optim.ASGD(params, lr, *args, **kwargs)
    elif name == 'LBFGS':
        return optim.LBFGS(params, lr, *args, **kwargs)
    elif name == 'NAdam':
        return optim.NAdam(params, lr, *args, **kwargs)
    elif name == 'RAdam':
        return optim.RAdam(params, lr, *args, **kwargs)
    elif name == 'RMSprop':
        return optim.RMSprop(params, lr, *args, **kwargs)
    elif name == 'Rprop':
        return optim.Rprop(params, lr, *args, **kwargs)
    elif name == 'SGD':
        return optim.SGD(params, lr, *args, **kwargs)
    else:
        raise ValueError("Optimizer not found:", name)
