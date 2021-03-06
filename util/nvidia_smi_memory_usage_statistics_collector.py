import subprocess
import util.timing
import time
import statistics
import threading

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"


# See: https://stackoverflow.com/questions/19846332/python-threading-inside-a-class
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class GpuMemoryUsageStatistics:

    def __init__(self, gpu_id: int):
        self.memory_usage_measurements_list = list()
        self.gpu_id = gpu_id

    @staticmethod
    def create_gpu_memory_usage_statistics(gpu_id: int):
        return GpuMemoryUsageStatistics(gpu_id)

    def add_memory_usage_statistic(self, memory_usage_in_mb: int):
        # print("GpuMemoryUsageStatistics - gpu_id: " + str(self.gpu_id) + " -- add_memory_usage_statistic")
        self.memory_usage_measurements_list.append(memory_usage_in_mb)
        # print("len(self.memory_usage_measurements_list): " + str(len(self.memory_usage_measurements_list)))

    def get_max_memory_usage(self):
        return max(self.memory_usage_measurements_list)

    def get_min_memory_usage(self):
        return min(self.memory_usage_measurements_list)

    def get_mean_memory_usage(self):
        # print("GpuMemoryUsageStatistics - gpu_id: " + str(self.gpu_id) + " -- get_mean_memory_usage")
        return sum(self.memory_usage_measurements_list) / float(len(self.memory_usage_measurements_list))

    def get_stdev_memory_usage(self):
        return statistics.stdev(self.memory_usage_measurements_list)


class MemoryStatisticsCollectorTestThread():

    def __init__(self):
        return

    def test_memory_statistics_collector(self):
        nvidia_smi_memory_statistics_collector = NvidiaSmiMemoryStatisticsCollector. \
        create_nvidai_smi_memory_statistics_collector(1, list([2, 3]))
        handle = nvidia_smi_memory_statistics_collector.collect_statistics_threaded()
        time.sleep(20)
        nvidia_smi_memory_statistics_collector.stop_collecting()
        handle.join()
        nvidia_smi_memory_statistics_collector.print_statistics()
        # nvidia_smi_memory_statistics_collector


class NvidiaSmiMemoryStatisticsCollector:
    """
    We want to collect statistics of the gpu memory usage during model training and evaluation.
    Pytorch itself has functions to measure GPU memory usage. However, these functions omit the memory
    allocated by the CUDA driver when it loads pytorch.
    See the discussion at:
    https://discuss.pytorch.org/t/memory-cached-and-memory-allocated-does-not-nvidia-smi-result/28420


    As such these functions underestimate the
    actual complete volatile memory usage. To get the full memory usage, nvidia-smi is probably
    still the most reliable source, hence we will query that.
    """

    def __init__(self, approximate_seconds_between_measurements: int, gpus_memory_usage_statistics: dict):
        self.approximate_seconds_between_measurements = approximate_seconds_between_measurements
        self.gpus_memory_usage_statistics = gpus_memory_usage_statistics
        self.perform_collection = True

    @staticmethod
    def create_nvidai_smi_memory_statistics_collector(approximate_seconds_between_measurements: int,
                                                      used_gpu_indices: list):
        number_of_gpus = NvidiaSmiMemoryStatisticsCollector.get_number_of_gpus()
        print(" create_nvidai_smi_memory_statistics_collector - number of gpus: " + str(number_of_gpus))
        gpus_memory_usage_statistics = dict()
        for used_gpu_index in used_gpu_indices:
            gpus_memory_usage_statistics[used_gpu_index] = \
                GpuMemoryUsageStatistics.create_gpu_memory_usage_statistics(used_gpu_index)
        return NvidiaSmiMemoryStatisticsCollector(approximate_seconds_between_measurements,
                                                  gpus_memory_usage_statistics)

    @staticmethod
    def get_number_of_gpus():
        map = NvidiaSmiMemoryStatisticsCollector.get_gpu_memory_map()
        # print("get_number_of_gpus - map: " + str(map))
        return len(map)

    @staticmethod
    def get_gpu_memory_map():
        """
        See: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3

        Get the current gpu usage.

        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        # result = subprocess.check_output(
        #    [
        #        'nvidia-smi', '--query-gpu=memory.used',
        #        '--format=csv,nounits,noheader'
        #    ], encoding='utf-8')

        # For python 3.5
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
        # In python 3.5 the "encoding" argument is not available, in order to get text
        # therefore manual decoding is necessary, see:
        # https://docs.python.org/3/library/subprocess.html#subprocess.check_output
        # Convert byte stream to text
        result = result.decode('utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    def update_statistics(self, gpu_memory_map: dict):
        for gpu_id in gpu_memory_map.keys():

            # Only update the statistic for the gpus we're interested in
            if gpu_id in self.gpus_memory_usage_statistics.keys():
                # print("update_statistics - gpu_id: " + str(gpu_id))
                memory_usage_in_mb = gpu_memory_map[gpu_id]
                # print("memory_usage_in_mb: " + str(memory_usage_in_mb))
                self.gpus_memory_usage_statistics[gpu_id].add_memory_usage_statistic(memory_usage_in_mb)

    @threaded
    def collect_statistics_threaded(self):
        number_of_collected_readings = 0
        last_time = util.timing.date_time_now()
        gpu_memory_map = NvidiaSmiMemoryStatisticsCollector.get_gpu_memory_map()
        self.update_statistics(gpu_memory_map)
        number_of_collected_readings += 1

        while self.perform_collection:
            while util.timing.seconds_since(last_time) < self.approximate_seconds_between_measurements:
                time.sleep(1)
            last_time = util.timing.date_time_now()
            gpu_memory_map = NvidiaSmiMemoryStatisticsCollector.get_gpu_memory_map()
            self.update_statistics(gpu_memory_map)
            number_of_collected_readings += 1
            # print(">>> collect_statistics_threaded - Number of collected readings: "
            #      + str(number_of_collected_readings))

    def stop_collecting(self):
        self.perform_collection = False

    def print_statistics(self):

        for gpu_id in self.gpus_memory_usage_statistics.keys():
            gpu_memory_statistics = self.gpus_memory_usage_statistics[gpu_id]
            print("gpu_id: " + str(gpu_id))
            print("Average memory usage: " + str(gpu_memory_statistics.get_mean_memory_usage()))
            print("Standard deviation memory usage: " + str(gpu_memory_statistics.get_stdev_memory_usage))
            print("Min memory usage: " + str(gpu_memory_statistics.get_min_memory_usage()))
            print("Max memory usage: " + str(gpu_memory_statistics.get_max_memory_usage()))


def main():
    test_thread = MemoryStatisticsCollectorTestThread()
    test_thread.test_memory_statistics_collector()


if __name__ == "__main__":
    main()
