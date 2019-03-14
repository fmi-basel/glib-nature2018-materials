import abc, six

@six.add_metaclass(abc.ABCMeta)
class Task():
    @abc.abstractmethod
    def run(self, data):
        pass