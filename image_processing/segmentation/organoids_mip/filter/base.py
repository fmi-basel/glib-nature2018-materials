import abc, six

@six.add_metaclass(abc.ABCMeta)
class Filter():
    @abc.abstractmethod
    def filter(self, label, parameter):
        pass