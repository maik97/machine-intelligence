'''

'''
import abc
class WackyBase(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError()
