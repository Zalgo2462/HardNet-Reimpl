from typing import Dict, Any

from Logger import Logger


class LoggerConsole(Logger):

    def log(self, log_data):
        # type: (LoggerConsole, Dict[str, Any])->None
        """
        Base implementation of log function for Loggers.
        :param log_data: dictionary of key-value pairs to log
        """
        items = list(log_data.items())

        for i in range(len(items) - 1):
            LoggerConsole.log_kvp(items[i])
            print(" | ", end="")

        LoggerConsole.log_kvp(items[-1])
        print("")

    @staticmethod
    def log_kvp(kvp):
        print("{}: {}".format(kvp[0], kvp[1]), end="")
