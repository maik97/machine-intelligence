from torch import optim
from wacky.backend import WackyValueError


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
        raise WackyValueError(
            name,
            ('Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax',
             'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD'),
            parameter='name',
            optional=False
        )
