class MotlException(Exception):
    """Base exception class for the cryoCAT motl module.

    Parameters
    ----------
    *args : object
        Optional message string as the first positional argument.
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Unspecified error in the emmotl module.'


class UserInputError(MotlException):
    """Exception raised when the user provides invalid input.

    Inherits from :class:`MotlException`.
    """

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Incorrect input provided by the user.'


class ProcessError(MotlException):
    """Exception raised when an internal processing step fails.

    Inherits from :class:`MotlException`.
    """

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Failed to finish a part of app process.'
