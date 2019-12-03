from os import listdir
from os.path import isfile, join, abspath
import os.path

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"


class FileUtils:

    @staticmethod
    def get_all_files_in_directory(folder_path: str,
                                   get_absolute_paths: bool = True):
        """S
        :param folder_path:
        :param get_absolute_paths
        :return:
        """
        # See: https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

        if get_absolute_paths:
            result = list([])
            for file in only_files:
                result.append(abspath(file))
            return result

        return only_files


    @staticmethod
    def file_exist_and_is_not_empty(path: str):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        else:
            return False
