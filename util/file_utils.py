from os import listdir
from os.path import isfile, join, abspath


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




