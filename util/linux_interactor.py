import subprocess
import os
import sys

__author__ = "Gideon Maillette de Buy Wenniger"
__copyright__ = "Copyright 2019, Gideon Maillette de Buy Wenniger"
__credits__ = ["Gideon Maillette de Buy Wenniger"]
__license__ = "Apache License 2.0"

# import subprocess
# p = subprocess.Popen([command, argument1,...], cwd=working_directory)
# p.wait()


#
class LinuxInteractor:

    # This method writes output of the process provided in the argument to the console only
    @staticmethod
    def write_to_console_only(process):
        for line in iter(process.stdout.readline, b''):
            # sys.stdout.write(c)
            ## See: https://stackoverflow.com/questions/
            # 40904979/the-print-of-string-constant-is-always-attached-with-b-intensorflow
            # The .decode() method is needed to remove preceding b' characters
            # Write the output to the terminal
            print(line.decode())

        for line in iter(process.stderr.readline, b''):
            print(line.decode())

    # This method writes output of the process provided in the argument to the console as
    # well as to the file specified by command_output_file_path
    @staticmethod
    def write_to_console_and_file(process, command_output_file_path):
        with open(command_output_file_path, 'w') as f:
            for line in iter(process.stdout.readline, b''):
                # sys.stdout.write(c)
                ## See: https://stackoverflow.com/questions/
                # 40904979/the-print-of-string-constant-is-always-attached-with-b-intensorflow
                # The .decode() method is needed to remove preceding b' characters
                # Write the output to the terminal
                print(line.decode())
                # Write the output to the output file
                f.write(line.decode())
                f.flush()

            for line in iter(process.stderr.readline, b''):
                print(line.decode())

    # This method writes output of the process provided in the argument to the console as
    # well as to the file specified by command_output_file_path
    @staticmethod
    def write_to_console_and_array(process):
        result = list([])

        for line in iter(process.stdout.readline, b''):
            print(line.decode())
            # Add the output line to the result array
            result.append(line.decode())

        for line in iter(process.stderr.readline, b''):
            print(line.decode())

        return result

    @staticmethod
    def get_environment_extended_with_usr_sbin():
        my_env = os.environ.copy()
        my_env["PATH"] = "/usr/sbin:/sbin:" + my_env["PATH"]
        return my_env

    # Execute command_string_or_command_arguments_list, which can either be a single command-argument
    # string or a list containing the command followed by its argument.
    # It is recommended to pass the command as a sequence
    # See also: https://docs.python.org/2/library/subprocess.html#subprocess.Popen
    # Why to avoid shell= True: https://stackoverflow.com/questions/3172470/actual-meaning-of-shell-true-in-subprocess
    @staticmethod
    def create_command_execution_process(command_string_or_command_arguments_list, working_directory, stdin_file_path):
        print("working_directory: " + working_directory)

        my_env = LinuxInteractor.get_environment_extended_with_usr_sbin()

        # This doesn't works
        # print("working directory: " + working_directory)
        # p = subprocess.Popen([command_string], cwd=working_directory,env=my_env)
        # p.wait()

        # See: https://stackoverflow.com/questions/22417010/subprocess-popen-stdin-read-file
        if stdin_file_path is not None:
            with open(stdin_file_path, 'rb', 0) as a:
                process = subprocess.Popen(command_string_or_command_arguments_list, stdout=subprocess.PIPE, stdin=a,
                                           stderr=subprocess.PIPE, cwd=working_directory, env=my_env)
        else:
            process = subprocess.Popen(command_string_or_command_arguments_list, stdout=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       stderr=subprocess.PIPE, cwd=working_directory, env=my_env)
        return process

    # This method executes a command string, and optionally writes the result not only to the console,
    #  but also to a file. This is suitable even in cases were the amount of output is relatively
    #  large.
    ## See: https://stackoverflow.com/questions/13744473/command-line-execution-in-different-folder
    # https://stackoverflow.com/questions/2231227/python-subprocess-popen-with-a-modified-environment
    @staticmethod
    def execute_external_command_and_show_output(command_string_or_command_arguments_list, working_directory,
                                                 command_output_file_path=None,
                                                 stdin_file_path=None):

        process = LinuxInteractor.create_command_execution_process(command_string_or_command_arguments_list,
                                                                   working_directory, stdin_file_path)

        # See: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command

        if command_output_file_path is None:
            LinuxInteractor.write_to_console_only(process)
        else:
            LinuxInteractor.write_to_console_and_file(process, command_output_file_path)

        print("execute external command finished")

    # This method executes a command and returns the resulting output as a list. This is only
    # suitable for commands that don't produce too much output, since if they do this will cause
    # memory problems. For cases were much output is expected the method
    # execute_external_command_and_show_output is a more suitable alternative.
    ## See: https://stackoverflow.com/questions/13744473/command-line-execution-in-different-folder
    # https://stackoverflow.com/questions/2231227/python-subprocess-popen-with-a-modified-environment
    @staticmethod
    def execute_external_command_and_return_output_as_list(command_string_or_command_arguments_list, working_directory,
                                                           stdin_file_path=None):

        process = LinuxInteractor.create_command_execution_process(command_string_or_command_arguments_list,
                                                                   working_directory, stdin_file_path)

        # See: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command
        result = LinuxInteractor.write_to_console_and_array(process)

        print("execute external command finished")
        return result

    @staticmethod
    def get_bash_format_command_for_command_argument_list(command_and_arguments_list):
        command_bash_format = command_and_arguments_list[0]
        for i in range(1, len(command_and_arguments_list)):
            command_bash_format += " " + command_and_arguments_list[i]
        return command_bash_format