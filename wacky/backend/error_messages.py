
def raise_type_error(var, allowed_types, var_name=None):
    if var_name is None:
        raise TypeError("Must be types:", allowed_types, "not", type(var))
    else:
        raise TypeError(var_name,"must be types:", allowed_types, "not", type(var))

def check_type(var, allowed_types, var_name=None, allow_none=False):
    if allow_none and var is None:
        pass
    elif not isinstance(var, allowed_types):
        raise_type_error(var, allowed_types, var_name)


