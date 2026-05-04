class MotlException(Exception):

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

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Incorrect input provided by the user.'


class ProcessError(MotlException):

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Failed to finish a part of app process.'
