import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Workflow():
    @abc.abstractmethod
    def run(self, data):
        pass


@six.add_metaclass(abc.ABCMeta)
class PipelineWorkflow(Workflow):
    tasks = []

    def __init__(self, tasks=[]):
        self.tasks = tasks

    def run(self, data):
        for task in self.tasks:
            data = task.run(data)
        return data


@six.add_metaclass(abc.ABCMeta)
class TreeWorkflow(Workflow):
    tasks = []
    merge_func = None

    def __init__(self, tasks=[], merge_func=lambda lst: None):
        self.tasks = tasks
        self.merge_func = merge_func

    def _run_recursion(self, in_data, tasks):
        """
        data flow:
                   in_data -> current_task -> out_data
                                  ^ |
           intermediate_out_data  | | intermediate_in_data
                                  | v
                               next_task
        """
        if len(tasks) == 0:
            return in_data

        task = tasks[0]
        intermediate_in_data_list = [data for data in task.run(in_data)]

        intermediate_out_data_list = []
        for intermediate_in_data in intermediate_in_data_list:
            intermediate_out_data_list.append(
                self._run_recursion(intermediate_in_data, tasks[1:])
            )

        out_data = self.merge_func(intermediate_out_data_list)
        return out_data

    def run(self, data):
        return self._run_recursion(data, self.tasks)
