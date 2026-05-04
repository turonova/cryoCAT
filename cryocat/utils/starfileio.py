import os
import tempfile
from enum import Enum
import pandas as pd
from os import path
import warnings


def _try_numeric(col):
    """Convert a Series to numeric, returning the original Series if any value cannot be converted."""
    result = pd.to_numeric(col, errors="coerce")
    return result if result.notna().all() else col


class TokenType(Enum):
    LITERAL = 0
    NEWLINE = 1
    COMMENT = 2
    LOOP = 3
    PROPERTY = 4


class Token:
    """Lexical token used during STAR file parsing.

    Each token carries a type (from TokenType), its string value, and the
    (1-based) line/column location in the original file.
    """

    def __init__(self, token_type: TokenType, value, location):
        self.token_type = token_type
        self.value = value
        self.location = (location[0] + 1, location[1] + 1)

    @staticmethod
    def tokenize(text):
        """Tokenize raw STAR file text into a reversed list of Token objects.

        The returned list is ordered so that ``list.pop()`` yields tokens in
        reading order (i.e. the first token in the file is at index ``-1``).

        Parameters
        ----------
        text : str
            Raw text content of a STAR file.

        Returns
        -------
        list of Token
            Tokens in reverse reading order.
        """
        tokens = list()

        lines = text.split("\n")
        for line_number, line in enumerate(lines):
            first = None
            for index, char in enumerate(line):
                if not char.isspace() and char != "#":
                    if first is None:
                        first = index
                    continue
                elif first is not None:
                    if line[first] == "_":
                        tokens.append(Token(TokenType.PROPERTY, line[first:index], (line_number, first)))
                    elif line[first:index] == "loop_":
                        tokens.append(Token(TokenType.LOOP, line[first:index], (line_number, first)))
                    else:
                        tokens.append(Token(TokenType.LITERAL, line[first:index], (line_number, first)))
                    first = None
                if char == "#":
                    tokens.append(Token(TokenType.COMMENT, line[index + 1 :].strip(), (line_number, index)))
                    break
                elif not char.isspace():
                    raise IOError(f"Got unexpected {char} at (Line {line_number}, Column {index}).")
            if first is not None:
                if line[first] == "_":
                    tokens.append(Token(TokenType.PROPERTY, line[first:], (line_number, first)))
                elif line[first:] == "loop_":
                    tokens.append(Token(TokenType.LOOP, line[first:], (line_number, first)))
                else:
                    tokens.append(Token(TokenType.LITERAL, line[first:], (line_number, first)))

            tokens.append(Token(TokenType.NEWLINE, None, (line_number, 0)))

        return tokens[::-1]

    @staticmethod
    def parse_newline_or_comments(tokens):
        """Consume NEWLINE and COMMENT tokens from the front of the queue.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).

        Returns
        -------
        list of str
            Comment strings extracted from consumed COMMENT tokens.
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
        """Consume leading whitespace/comments and then a LITERAL token as a data specifier.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).

        Returns
        -------
        comments : list of str
            Any comments found before the specifier.
        specifier : str
            The specifier value.
        """
        comments = Token.parse_newline_or_comments(tokens)
        specifier = Token.consume(tokens, TokenType.LITERAL)
        return comments, specifier.value

    @staticmethod
    def parse_columns(tokens):
        """Consume the ``loop_`` keyword and subsequent PROPERTY tokens as column names.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).

        Returns
        -------
        comments : list of str
            Any comments found before ``loop_``.
        columns : list of str
            Column names (PROPERTY values with the leading ``_`` stripped).
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
        """Consume a single PROPERTY token (optionally followed by a COMMENT) as a column name.

        The leading ``_`` of the PROPERTY value is stripped before returning.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).

        Returns
        -------
        str
            Column name without the leading ``_``.
        """
        column = Token.consume(tokens, TokenType.PROPERTY)
        Token.check_then_consume(tokens, TokenType.COMMENT)
        Token.consume(tokens, TokenType.NEWLINE)
        return column.value[1:]

    @staticmethod
    def parse_rows(tokens, columns):
        """Consume LITERAL tokens as row data and build a DataFrame.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).
        columns : list of str
            Column names for the resulting DataFrame.

        Returns
        -------
        comments : list of str
            Any comments found before the row data.
        data : pandas.DataFrame
            Parsed rows as a DataFrame.
        """
        comments = Token.parse_newline_or_comments(tokens)
        end = False
        rows = []
        while not end:
            data = []
            for _ in range(len(columns)):
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
        """Return True if the next token in the queue matches ``token_type``.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).
        token_type : TokenType
            Expected token type.

        Returns
        -------
        bool
        """
        if len(tokens) == 0:
            raise IOError(f"Expected {token_type} but there are not enough tokens.")
        return tokens[-1].token_type == token_type

    @staticmethod
    def consume(tokens, token_type):
        """Remove and return the next token, raising IOError if the type does not match.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).
        token_type : TokenType
            Expected token type.

        Returns
        -------
        Token
            The consumed token.

        Raises
        ------
        IOError
            If the queue is empty or the next token has a different type.
        """
        if len(tokens) == 0:
            raise IOError(f"Expected {token_type} but there are not enough tokens.")
        if tokens[-1].token_type == token_type:
            return tokens.pop()
        raise IOError(f"Expected {token_type} but got {tokens[-1].token_type} at {tokens[-1].location}.")

    @staticmethod
    def check_then_consume(tokens, token_type):
        """Consume the next token if it matches ``token_type``, otherwise return None.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).
        token_type : TokenType
            Token type to match.

        Returns
        -------
        Token or None
            The consumed token, or None if the type did not match.
        """
        if len(tokens) > 0 and tokens[-1].token_type == token_type:
            return Token.consume(tokens, token_type)
        return None

    @staticmethod
    def lookahead(tokens, token_type_target, ignores):
        """Scan ahead through the queue for ``token_type_target``, skipping ``ignores``.

        Parameters
        ----------
        tokens : list of Token
            Token queue (front at index ``-1``).
        token_type_target : TokenType
            Token type to search for.
        ignores : list of TokenType
            Token types that should be skipped during the scan.

        Returns
        -------
        bool
            True if ``token_type_target`` is found before any non-ignored token.
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
    """Read, write, and manipulate RELION/STOPGAP STAR files.

    A STAR file contains one or more data blocks, each identified by a specifier
    (e.g. ``data_particles``). Each block is parsed into a :class:`pandas.DataFrame`.

    Attributes
    ----------
    frames : list of pandas.DataFrame
        One DataFrame per data block.
    specifiers : list of str
        Data-block specifier strings (e.g. ``"data_particles"``).
    comments : list of list of str
        Comment lines associated with each data block.
    """

    def __init__(self, file_path=None, frames=None, specifiers=None, comments=None):
        """Initialise a Starfile, optionally reading from *file_path*.

        If *file_path* points to an existing file it is read immediately and
        ``frames``, ``specifiers``, and ``comments`` are populated from it.
        Otherwise the provided values (or ``None``) are stored directly.

        Parameters
        ----------
        file_path : str, optional
            Path to a ``.star`` file to read. Defaults to None.
        frames : list of pandas.DataFrame, optional
            Pre-built data frames. Used only when *file_path* is not given.
        specifiers : list of str, optional
            Specifier strings matching *frames*. Used only when *file_path* is not given.
        comments : list of list of str, optional
            Comments matching *frames*. Used only when *file_path* is not given.
        """
        if file_path and path.isfile(file_path):
            self.frames, self.specifiers, self.comments = self.read(file_path)
        else:
            self.frames = frames
            self.specifiers = specifiers
            self.comments = comments

    @staticmethod
    def remove_lines(file_path, lines_to_remove, output_path=None, data_specifier=None, number_columns=True):
        """Remove rows from a data block in a STAR file.

        Parameters
        ----------
        file_path : str
            Path to the input STAR file.
        lines_to_remove : array-like of int
            Integer row indices (0-based) to remove from the target data block.
        output_path : str, optional
            If given, the modified STAR file is written to this path. If None,
            the modified data structures are returned instead. Defaults to None.
        data_specifier : str, optional
            Specifier of the data block to modify. If None, the first block is used.
            Defaults to None.
        number_columns : bool, default=True
            Whether to write column indices in the output file. Defaults to True.

        Returns
        -------
        tuple or None
            ``(frames, specifiers, comments)`` when *output_path* is None, otherwise None.
        """
        frames, specifiers, comments = Starfile.read(file_path)

        if data_specifier is None:
            spec_id = 0
        else:
            spec_id = Starfile.get_specifier_id(specifiers, data_specifier)
            if spec_id is None:
                warnings.warn(f"The data specifier {data_specifier} was not found in the file. No lines were removed.")
                return

        rows_to_remove_labels = frames[spec_id].index[lines_to_remove]
        frames[spec_id] = frames[spec_id].drop(rows_to_remove_labels)
        frames[spec_id].reset_index(drop=True, inplace=True)

        if output_path is not None:
            Starfile.write(frames, output_path, specifiers=specifiers, comments=comments, number_columns=number_columns)
        else:
            return frames, specifiers, comments

    @staticmethod
    def read(file_path, data_id=None):
        """Parse a STAR file into DataFrames, specifiers, and comments.

        Columns that contain entirely numeric data are automatically cast to
        numeric types.

        Parameters
        ----------
        file_path : str
            Path to the ``.star`` file to read.
        data_id : int, optional
            If given, return only the data block at this index rather than all
            blocks. Defaults to None.

        Returns
        -------
        frames : list of pandas.DataFrame, or pandas.DataFrame
            All data blocks (or a single block when *data_id* is given).
        specifiers : list of str, or str
            Specifier strings (or a single specifier when *data_id* is given).
        comments : list of list of str, or list of str
            Comments (or a single comment list when *data_id* is given).
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
            frames[i] = f.apply(_try_numeric)

        if data_id is not None:
            return frames[data_id], specifiers[data_id], comments[data_id]
        return frames, specifiers, comments

    @staticmethod
    def get_specifier_id(specifiers, specifier_id):
        """Return the list index of a specifier string, or None if not found.

        Parameters
        ----------
        specifiers : list of str
            List of specifier strings from a parsed STAR file.
        specifier_id : str
            Specifier to search for.

        Returns
        -------
        int or None
            Index of *specifier_id* in *specifiers*, or None if absent.
        """
        if specifier_id in specifiers:
            return specifiers.index(specifier_id)
        return None

    @staticmethod
    def get_frame_and_comments(file_path, specifier):
        """Read a single data block and its comments from a STAR file.

        Parameters
        ----------
        file_path : str
            Path to the ``.star`` file.
        specifier : str
            Specifier of the data block to retrieve (e.g. ``"data_particles"``).

        Returns
        -------
        frame : pandas.DataFrame
            The data block.
        comments : list of str
            Comments associated with the data block.

        Raises
        ------
        ValueError
            If *specifier* is not found in the file.
        """
        frames, specifiers, comments = Starfile.read(file_path)
        spec_id = Starfile.get_specifier_id(specifiers, specifier)
        if spec_id is None:
            raise ValueError(f"There is no entry with specifier {specifier}.")
        return frames[spec_id], comments[spec_id]

    @staticmethod
    def write(frames, output_path, specifiers=None, comments=None, number_columns=True, float_precision=6):
        """Write data blocks to a STAR file.

        Parameters
        ----------
        frames : list of pandas.DataFrame
            Data blocks to write.
        output_path : str
            Path of the output ``.star`` file.
        specifiers : list of str, optional
            Specifier string for each data block. Defaults to ``["data"] * len(frames)``.
        comments : list of list of str or None, optional
            Comments to write before each data block. Defaults to no comments.
        number_columns : bool, default=True
            If True, append ``#<index>`` after each column header line. Forced off
            for blocks whose specifier contains ``"stopgap"``. Defaults to True.
        float_precision : int, default=6
            Number of decimal places used when rounding float values. Defaults to 6.

        Raises
        ------
        ValueError
            If the lengths of *frames*, *specifiers*, and *comments* do not match.
        """
        if specifiers is None:
            specifiers = ["data"] * len(frames)
        if comments is None:
            comments = (None,) * len(frames)

        if len(frames) != len(specifiers) or len(frames) != len(comments):
            raise ValueError(
                f"Invalid size of the lists found. "
                f"The sizes are (frames: {len(frames)}), "
                f"(specifiers: {len(specifiers)}), "
                f"and (comments: {len(comments)})."
            )

        frames = [f.round(float_precision) for f in frames]

        with open(output_path, "w") as file:
            for frame, specifier, comment in zip(frames, specifiers, comments):
                stopgap = "stopgap" in specifier
                numbered = number_columns and not stopgap

                if comment is not None:
                    for c in comment:
                        file.write(f"\n# {c}")
                    file.write("\n")

                file.write(f"\n{specifier}\n\n")
                file.write("loop_\n")
                for index, column in enumerate(frame.columns, 1):
                    file.write(f"_{column} #{index}\n" if numbered else f"_{column}\n")
                if stopgap:
                    file.write("\n")

                for row in frame.itertuples(index=False):
                    file.write("\t".join(f"{str(v):<10}" for v in row) + "\n")
                file.write("\n")

    @staticmethod
    def fix_relion5_star(input_path):
        """Convert a RELION 5 STAR file with key-value ``data_general`` blocks to loop format.

        RELION 5 sometimes writes ``data_general`` sections as bare ``_key value``
        pairs rather than as a ``loop_`` table. This function rewrites such sections
        into the loop-based format expected by the rest of the STAR parser so that
        the file can be read by :meth:`read`.

        If the file already contains a ``loop_`` inside ``data_general`` (i.e. it is
        already in the correct format), the original path is returned unchanged and
        no temporary file is created.

        Parameters
        ----------
        input_path : str
            Path to the input RELION 5 STAR file.

        Returns
        -------
        str
            Path to the (possibly temporary) fixed file, or *input_path* if no
            changes were necessary.
        """
        with open(input_path) as f:
            lines = f.readlines()

        fixed = []
        in_data_general = False
        headers = []
        values = []
        was_modified = False
        has_loop = False

        for line in lines:
            l = line.strip()
            if l.startswith("data_general"):
                in_data_general = True
                was_modified = True
                fixed.append(l)
                continue
            if in_data_general:
                if l.startswith("loop_"):
                    has_loop = True
                    break
                if l.startswith("data_") and not l.startswith("data_general"):
                    if headers:
                        fixed.append("loop_")
                        fixed.extend(headers)
                        fixed.extend(values)
                        fixed.append("")
                        was_modified = True
                    in_data_general = False
                    fixed.append(l)
                    continue
                elif l.startswith("_rln"):
                    parts = l.split(None, 1)
                    headers.append(parts[0])
                    values.append(parts[1].strip() if len(parts) > 1 else "")
                elif l == "":
                    continue
                else:
                    fixed.append(line.rstrip())
            else:
                fixed.append(line.rstrip())

        if in_data_general and headers and not has_loop:
            fixed.append("loop_")
            fixed.extend(headers)
            fixed.extend(values)
            fixed.append("")
            was_modified = True

        if was_modified and not has_loop:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".star", prefix="fixed_", text=True)
            os.close(tmp_fd)
            with open(tmp_path, "w") as out:
                out.write("\n".join(fixed) + "\n")
            return tmp_path
        return input_path
