from collections import UserDict
import torch as th


def all_equal(iterable):
    iterator = iter(iterable)

    try:
        first_item = next(iterator)
    except StopIteration:
        return True

    for x in iterator:
        if x != first_item:
            return False
    return True


class MemoryDict(UserDict):

    def __init__(self, *args, **kwargs):
        self.stacked = False
        super(MemoryDict, self).__init__(*args, **kwargs)

    def __getitem__(self, y):
        if y not in self.keys():
            self.__setitem__(key=y, value=[])
        return super(MemoryDict, self).__getitem__(y)

    def __setitem__(self, key, value):

        if not isinstance(value, list) and not self.stacked:
            error_msg = ("Setting value at key '"+str(key)+"' failed:"
                        "\n Expected type "+str(list)+" got "+str(type(value))+" instead"
                        "\n\n While not stacked, a value assigned to a key of MemoryDict"+
                        "\n must be type list, not "+ str(type(value))+"."
                        "\n Call the stack_tensors() method first, if you are trying to store a tensor"+
                        "\n or using some of the function wrapper based on memory.")
            raise TypeError(error_msg)

        elif not isinstance(value, th.Tensor) and self.stacked:
            error_msg = ("Setting value at key " + str(key) +
                        "\n While stacked, a value assigned to a key of MemoryDict"+
                        "\n must be type torch.Tensor, not " + str(type(value))+ " Call the clear() method first,"+
                        "\n if you are trying to store a list of tensors")
            raise TypeError(error_msg)
        else:
            super(MemoryDict, self).__setitem__(key, value)

    def stack_tensors(self):
        self.stacked = True
        for key in self.keys():
            self[key] = th.stack(self[key])

    def clear(self) -> None:
        self.stacked = False
        super(MemoryDict, self).clear()

    def batch(self, batch_size):
        if not self.stacked:
            self.stack_tensors()

        splitted = self.split(batch_size, copy_dict=True)

        if splitted.global_keys_len is not None:
            sub_memories = []
            for i in range(splitted.global_keys_len):
                sub_mem = MemoryDict()
                for key in splitted.keys():
                    sub_mem[key].append(splitted[key][i])
                sub_memories.append(sub_mem)
            return sub_memories

        else:
            raise Exception("Number of batches not equal:", str(splitted.keys_len_dict))

    def split(self, split_size_or_sections, copy_dict=True):
        splitted = self.copy() if copy_dict else self
        for key in splitted.keys():
            splitted[key] = th.split(splitted[key], split_size_or_sections)
        return splitted

    def compare_len(self, key_a, key_b):
        return len(self[key_a]) == len(self[key_b])

    @property
    def keys_len_dict(self):
        return {key: len(self[key]) for key in self.keys()}

    @property
    def keys_len_list(self):
        return [len(self[key]) for key in self.keys()]

    @property
    def global_keys_len(self):
        if all_equal(self.keys_len_list):
            return self.keys_len_list[0]
        else:
            return None

