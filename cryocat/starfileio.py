from enum import Enum
import pandas as pd
from os import path
import warnings


class TokenType(Enum):
    LITERAL = 0
    NEWLINE = 1
    COMMENT = 2
    LOOP = 3
    PROPERTY = 4


class Token:
    def __init__(self, token_type: TokenType, value, location):
        self.token_type = token_type
        self.value = value
        self.location = (location[0] + 1, location[1] + 1)

    @staticmethod
    def tokenize(text):
        """This function tokenizes a text into several tokens.

        Parameters
        ----------
        text :
            a given text

        Returns
        -------
        type
            list of tokens

        """
        tokens = list()

        # Split the text into several lines
        lines = text.split("\n")
        for line_number, line in enumerate(lines):
            # The first index of a non-space-or-hash sequence of characters. None means there is no sequence found
            first = None
            for index, char in enumerate(line):
                if not char.isspace() and char != "#":
                    # Set the first index of the sequence if it is None
                    if first is None:
                        first = index
                    continue
                elif first is not None:
                    # If a space or # and the sequence are found, classifies the sequence as
                    #   LOOP if it is 'loop_'
                    #   PROPERTY if it starts with '_'
                    #   LITERAL otherwise

                    if line[first] == "_":
                        tokens.append(Token(TokenType.PROPERTY, line[first:index], (line_number, first)))
                    elif line[first:index] == "loop_":
                        tokens.append(Token(TokenType.LOOP, line[first:index], (line_number, first)))
                    else:
                        tokens.append(Token(TokenType.LITERAL, line[first:index], (line_number, first)))

                    # Set that there is no sequence found
                    first = None
                if char == "#":
                    # Anything after the # character is a comment

                    tokens.append(Token(TokenType.COMMENT, line[index + 1 :].strip(), (line_number, index)))
                    break
                elif not char.isspace():
                    raise IOError(f"Got unexpected {char} at (Line {line_number}, Column {index}).")
            if first is not None:
                # Classifies the sequence if there is an end of line

                if line[first] == "_":
                    tokens.append(Token(TokenType.PROPERTY, line[first:], (line_number, first)))
                elif line[first:] == "loop_":
                    tokens.append(Token(TokenType.LOOP, line[first:], (line_number, first)))
                else:
                    tokens.append(Token(TokenType.LITERAL, line[first:], (line_number, first)))

            # Add a NEWLINE token
            tokens.append(Token(TokenType.NEWLINE, None, (line_number, 0)))

        return tokens[::-1]

    @staticmethod
    def parse_newline_or_comments(tokens):
        """This function takes a token queue and dequeues any NEWLINE token and COMMENT token while storing the comments from
        the COMMENT tokens.

        Parameters
        ----------
        tokens :
            a queue of tokens

        Returns
        -------
        type
            list of comments retrieves from the dequeued COMMENT tokens

        """
        comments = []
        while True:
            comment_token = Token.check_then_consume(tokens, TokenType.COMMENT)
            if comment_token is not None:
                comments.append(comment_token.value)
            elif not Token.check_then_consume(tokens, TokenType.NEWLINE):
                break
        return comments

    @staticmethod
    def parse_specifier(tokens):
        """This function takes a token queue, gets comments, and consumes (matches) a specifier as a LITERAL token.

        Parameters
        ----------
        tokens :
            a queue of tokens

        Returns
        -------
        type
            a tuple of comments and the parsed specifier

        """
        comments = Token.parse_newline_or_comments(tokens)
        specifier = Token.consume(tokens, TokenType.LITERAL)
        return comments, specifier.value

    @staticmethod
    def parse_columns(tokens):
        """This function takes a token queue, gets comments, consumes (matches) the `loop_` keyword as a LOOP token
        following by a NEWLINE token, and parses the column names

        Parameters
        ----------
        tokens :
            a queue of tokens

        Returns
        -------
        type
            a tuple of comments and column names

        """
        comments = Token.parse_newline_or_comments(tokens)
        columns = []
        Token.consume(tokens, TokenType.LOOP)
        Token.consume(tokens, TokenType.NEWLINE)
        while Token.check(tokens, TokenType.PROPERTY):
            column = Token.parse_column(tokens)
            columns.append(column)
        return comments, columns

    @staticmethod
    def parse_column(tokens):
        """This function takes a token queue, consumes a column name token as a PROPERTY token, and tries to consume
        a COMMENT token to retrieve the comment if existed.

        The PROPERTY token captures anything starting with "_", therefore the column name be the value of the token
        without the "_".

        Parameters
        ----------
        tokens :
            a token queue

        Returns
        -------
        type
            a tuple of comments and the column name

        """
        column = Token.consume(tokens, TokenType.PROPERTY)
        Token.check_then_consume(tokens, TokenType.COMMENT)
        Token.consume(tokens, TokenType.NEWLINE)
        return column.value[1:]

    @staticmethod
    def parse_rows(tokens, columns):
        """This function takes a token queue, gets comments, tries to consume LITERAL tokens as a rows which matches
        the number of columns before getting a new line, and converts the rows to a Pandas DataFrame.

        Parameters
        ----------
        tokens :
            a queue of tokens
        columns :
            a list of column names

        Returns
        -------
        type
            a tuple of comments and Pandas DataFrames

        """
        comments = Token.parse_newline_or_comments(tokens)
        end = False
        rows = []
        while not end:
            data = []
            for i in range(len(columns)):
                token = Token.check_then_consume(tokens, TokenType.LITERAL)
                if token is None:
                    end = True
                    break
                else:
                    data.append(token.value)
            else:
                Token.consume(tokens, TokenType.NEWLINE)
                rows.append(data)
        return comments, pd.DataFrame(rows, columns=columns)

    @staticmethod
    def check(tokens, token_type):
        """This function checks if the first token from the given token queue matches a given token type.

        Parameters
        ----------
        tokens :
            a queue of tokens
        token_type :
            a token type to be matched

        Returns
        -------
        type
            a boolean value indicating the match

        """

        if len(tokens) == 0:
            raise IOError(f"Expected {token_type} but there are not enough token.")
        if tokens[-1].token_type == token_type:
            return True
        return False

    @staticmethod
    def consume(tokens, token_type):
        """This function consumes the first token from the given token queue. If the token type of the first
        token does not match the token type to be matched, this function will raise a parsing error.

        Parameters
        ----------
        tokens :
            a queue of tokens
        token_type :
            a token type to be matched

        Returns
        -------
        type
            the first token

        """
        if len(tokens) == 0:
            raise IOError(f"Expected {token_type} but there are enough token.")
        if tokens[-1].token_type == token_type:
            return tokens.pop()
        else:
            raise IOError(f"Expected {token_type} but got {tokens[0].token_type} at {tokens[0].location}.")

    @staticmethod
    def check_then_consume(tokens, token_type):
        """This function checks the first token from the given token queue and consumes it if matched. Otherwise,
        it returns a None

        Parameters
        ----------
        tokens :
            a queue of tokens
        token_type :
            a token type to be matched

        Returns
        -------
        type
            the first token or None

        """
        if len(tokens) > 0 and tokens[-1].token_type == token_type:
            return Token.consume(tokens, token_type)
        return None

    @staticmethod
    def lookahead(tokens, token_type_target, ignores):
        """This function looks for a token type while ignoring token types from the ignores list

        Parameters
        ----------
        tokens :
            a queue of tokens
        token_type_target :
            a token type to be found
        ignores :
            a list of token types to be ignored

        Returns
        -------
        type
            a boolean value indicating a found token

        """
        ignores = set(ignores)
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].token_type == token_type_target:
                return True
            elif tokens[i].token_type in ignores:
                continue
            else:
                break
        return False


class Starfile:
    def __init__(self, file_path=None, frames=None, specifiers=None, comments=None):
        """
        This function reads a starfile with a *.star extension into a tuple of a list of Pandas DataFrame, a list of Data
            Specifier, and a list of comments

            It reads the file and extracts the lists from the parsing function.

        Parameters
        ----------
        path :
            the path to the starfile to be read

        Returns
        -------
        type
            a tuples of a list of Pandas DataFrames, list of specifiers, and list of comments

        """

        if file_path and path.isfile(file_path):
            self.frames, self.specifiers, self.comments = self.read(file_path)
        else:
            self.frames = frames
            self.specifiers = specifiers
            self.comments = comments

    @staticmethod
    def remove_lines(file_path, lines_to_remove, output_file=None, data_specifier=None, number_columns=True):

        frames, specifiers, comments = Starfile.read(file_path)

        if data_specifier is None:
            spec_id = 0
        else:
            spec_id = Starfile.get_specifier_id(specifiers, data_specifier)
            if spec_id is None:
                warnings.warn(f"The data specifier {data_specifier} was not found in the file. No lines were removed.")
                return

        # Convert row numbers to index labels
        rows_to_remove_labels = frames[spec_id].index[lines_to_remove]
        frames[spec_id] = frames[spec_id].drop(rows_to_remove_labels)
        frames[spec_id].reset_index(drop=True, inplace=True)

        if output_file is not None:
            Starfile.write(frames, output_file, specifiers=specifiers, comments=comments, number_columns=number_columns)
        else:
            return frames, specifiers, comments

    @staticmethod
    def read(file_path, data_id=None):
        """This function parses a starfile into a tuple of a list of Pandas DataFrame, a list of Data Specifier, and a list of
        comments.

        It tokenizes the file and if it finds a specifier, it starts parsing in the following order:
            1. Specifier
            2. Columns      (as column names)
            3. Rows         (as a Pandas Dataframe together with the Columns)

        Parameters
        ----------
        raw_starfile :
            the starfile to be parsed
        file_path :


        Returns
        -------
        type
            a tuples of a list of Pandas DataFrames, list of specifiers, and list of comments

        """

        with open(file_path, mode="r") as file:
            raw_starfile = file.read()

        tokens = Token.tokenize(raw_starfile)
        frames = []
        comments = []
        specifiers = []
        while Token.lookahead(tokens, TokenType.LITERAL, [TokenType.NEWLINE, TokenType.COMMENT]):
            specifier_comments, specifier = Token.parse_specifier(tokens)
            column_comments, columns = Token.parse_columns(tokens)
            rows_comments, data = Token.parse_rows(tokens, columns)
            comments.append(specifier_comments + column_comments + rows_comments)
            specifiers.append(specifier)
            frames.append(data)
        Token.parse_newline_or_comments(tokens)
        if len(tokens) > 0:
            raise IOError(f"Expected a specifier or an end of token but got {tokens[0].token_type}")

        for i, f in enumerate(frames):
            frames[i] = f.apply(pd.to_numeric, errors="ignore")

        if data_id is not None:
            return frames[data_id], specifiers[data_id], comments[data_id]
        else:
            return frames, specifiers, comments

    @staticmethod
    def get_specifier_id(speficiers, specifier_id):
        if specifier_id in speficiers:
            return speficiers.index(specifier_id)
        else:
            return None

    @staticmethod
    def get_frame_and_comments(file_path, specifier):
        frames, specifiers, comments = Starfile.read(file_path)

        spec_id = Starfile.get_specifier_id(specifiers, specifier)

        if spec_id is None:
            raise ValueError(f"There is no entry with specifier {specifier}.")

        return frames[spec_id], comments[spec_id]

    @staticmethod
    def write(frames, path, specifiers=None, comments=None, number_columns=True, float_precision=6):
        if specifiers is None:
            specifiers = ["data"] * len(frames)
        if comments is None:
            comments = (None,) * len(frames)

        if len(frames) != len(specifiers) or len(frames) != len(comments) or len(specifiers) != len(comments):
            raise ValueError(
                f"Invalid size of the lists found. "
                f"The sizes are (frames: {len(frames)}), "
                f"(specifiers: {len(specifiers)}), "
                f"and (comments: {len(comments)})."
            )

        for i, f in enumerate(frames):
            frames[i] = f.round(float_precision)

        with open(path, "w") as file:

            def write_with_number(name, number):
                file.write(f"_{name} #{number}\n")

            def write_without_number(name, _):
                file.write(f"_{name}\n")

            def format_value(value):
                return "{:<10}".format(str(value))

            for frame, specifier, comment in zip(frames, specifiers, comments):
                frame = frame.applymap(format_value)
                stopgap = "stopgap" in specifier
                write_function = write_without_number if not number_columns or stopgap else write_with_number
                if comment is not None:
                    for c in comment:
                        file.write(f"\n# {c}")
                    file.write("\n")
                file.write(f"\n{specifier}\n\n")
                file.write("loop_\n")
                for index, column in enumerate(frame.columns, 1):
                    write_function(column, index)
                if stopgap:
                    file.write("\n")

                for row in frame.itertuples(index=False):
                    file.write("\t".join(map(str, row)) + "\n")
                # formatted_row = "\t".join("{:<10}".format(str(value)) for value in row)
                # file.write(formatted_row + "\n")
                file.write("\n")
