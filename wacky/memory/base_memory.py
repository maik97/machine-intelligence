from collections import UserDict, UserList
from collections.abc import Iterable
import torch as th
import numpy as np


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


class TensorList(UserList):

    def __init__(self, initlist=None):
        super(TensorList, self).__init__(initlist)

        if initlist is not None:
            for i in range(len(self)):
                if not isinstance(self[i], th.Tensor):
                    self[i] = th.tensor(self[i], dtype=th.float)

    def append(self, item: th.Tensor) -> None:
        if not isinstance(item, th.Tensor):
            item = th.tensor(item, dtype=th.float)
        super(TensorList, self).append(item)


class MemoryDict(UserDict):

    def __init__(self, *args, **kwargs):
        self.stacked = False
        super(MemoryDict, self).__init__(*args, **kwargs)

    def __getitem__(self, y):
        if y not in self.keys() and not self.stacked:
            self.__setitem__(key=y, value=TensorList())
        return super(MemoryDict, self).__getitem__(y)

    def __setitem__(self, key, value):

        if not self.stacked:
            e_1 = None
            e_2 = None

            if not isinstance(value , Iterable):
                try:
                    value = TensorList(value)
                except Exception as e:
                    e_1 = f"\n Converting value at key '{key}' to Iterable failed:\n {e}"

            if not isinstance(value, (TensorList, th.Tensor)):
                try:
                    value = TensorList(value)
                except Exception as e:
                    e_2 = f"\n Converting value at key '{key}' to TensorList failed:\n {e}"

            if not isinstance(value, TensorList):
                e_3 = (f"\n Setting value at key '{key}' failed: Invalid type {type(value)}"
                        f"\n While not stacked, assigned values to keys of {MemoryDict} must be convertable to {list},"
                        f"\n Call the stack_tensors() method first, if you are trying to store a tensor"
                        f" or using some of the function wrapper based on memory.")
                if e_2 is not None:
                    e_3 = e_2 + "\n" + e_3
                if e_1 is not None:
                    e_3 = e_1 + "\n" + e_3
                raise TypeError(e_3)

        elif not isinstance(value, th.Tensor):
            raise TypeError(f"\n Setting value at key '{key}' failed: Expected type {th.Tensor} got {type(value)} instead."
                            f"\n While stacked, assigned values to keys of {MemoryDict} must be type {th.Tensor}."
                            f"\n Call the clear() method first, if you are trying to store a list of tensors.")

        super(MemoryDict, self).__setitem__(key, value)

    def stack_tensors(self):
        self.stacked = True
        for key in self.keys():
            self[key] = th.stack(self[key])

    def clear(self) -> None:
        self.stacked = False
        super(MemoryDict, self).clear()

    def batch(self, batch_size):
        splitted = self.split(batch_size, copy_dict=True)

        if splitted.global_keys_len is not None:
            sub_memories = []
            for i in range(splitted.global_keys_len):
                sub_mem = MemoryDict()
                sub_mem.stacked = True
                for key in splitted.keys():
                    sub_mem[key].append(splitted[key][i])
                sub_memories.append(sub_mem)
            return sub_memories

        else:
            raise Exception("Number of batches not equal:", str(splitted.keys_len_dict))

    def split(self, split_size_or_sections, copy_dict=True):
        splitted = self.copy() if copy_dict else self

        if not splitted.stacked:
            splitted.stack_tensors()

        for key in splitted.keys():
            splitted[key] = th.split(splitted[key], split_size_or_sections)
        return splitted

    def compare_len(self, keys: list):
        return all_equal(len(self[key]) for key in keys)

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


test = np.load('dicom_data_analysis_all.npy')
print(test.shape)
last_row = np.asarray(test[-1], dtype=np.float64)
print(last_row)
test = np.delete(test, np.argwhere(last_row < 3.5), axis=1)
print(test.shape)
print(test[-1])

test = np.array([], dtype=np.float64)
print(np.append(test, '1')) # ['1']
print(np.append(test, float('1'))) # [1.]

