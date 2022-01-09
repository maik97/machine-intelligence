
def standardize_tensor(vals, eps=1e-08):
    return (vals - vals.mean()) / (vals.std() + eps)
