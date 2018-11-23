import abc, six

@six.add_metaclass(abc.ABCMeta)
class Writer():
    @abc.abstractmethod
    def write(self, data_frame, target_path):
        pass