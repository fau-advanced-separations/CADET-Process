import functools as ft
import io
import os.path as op
import pickletools
import sqlite3
from typing import Any, Union

import dill as pickle
import diskcache
from diskcache.core import MODE_BINARY, MODE_PICKLE, MODE_RAW, MODE_TEXT, UNKNOWN

__all__ = ["DillDisk"]


class DillDisk(diskcache.Disk):
    """Cache key and value serialization for SQLite database and files."""

    def put(self, key: Any) -> None:
        """
        Convert `key` to fields key and raw for Cache table.

        :param key: key to convert
        :return: (database key, raw boolean) pair
        """
        # pylint: disable=unidiomatic-typecheck
        type_key = type(key)

        if type_key is bytes:
            return sqlite3.Binary(key), True
        elif (
            (type_key is str)
            or (type_key is int and -9223372036854775808 <= key <= 9223372036854775807)
            or (type_key is float)
        ):
            return key, True
        else:
            data = pickle.dumps(key, protocol=self.pickle_protocol)
            result = pickletools.optimize(data)
            return sqlite3.Binary(result), False

    def get(self, key: Any, raw: Any) -> Any:
        """
        Convert fields `key` and `raw` from Cache table to key.

        :param key: database key to convert
        :param bool raw: flag indicating raw database storage
        :return: corresponding Python key
        """
        # pylint: disable=no-self-use,unidiomatic-typecheck
        if raw:
            return bytes(key) if type(key) is sqlite3.Binary else key
        else:
            return pickle.load(io.BytesIO(key))

    def store(
        self,
        value: Any,
        read: bool,
        key: Any = UNKNOWN,
    ) -> tuple[int, int, Union[str, None], Union[Any, sqlite3.Binary]]:
        """
        Convert `value` to fields size, mode, filename, and value for Cache table.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :param key: key for item (default UNKNOWN)
        :return: (size, mode, filename, value) tuple for Cache table
        """
        # pylint: disable=unidiomatic-typecheck
        type_value = type(value)
        min_file_size = self.min_file_size

        if (
            (type_value is str and len(value) < min_file_size)
            or (type_value is int and -9223372036854775808 <= value <= 9223372036854775807)
            or (type_value is float)
        ):
            return 0, MODE_RAW, None, value
        elif type_value is bytes:
            if len(value) < min_file_size:
                return 0, MODE_RAW, None, sqlite3.Binary(value)
            else:
                filename, full_path = self.filename(key, value)
                self._write(full_path, io.BytesIO(value), "xb")
                return len(value), MODE_BINARY, filename, None
        elif type_value is str:
            filename, full_path = self.filename(key, value)
            self._write(full_path, io.StringIO(value), "x", "UTF-8")
            size = op.getsize(full_path)
            return size, MODE_TEXT, filename, None
        elif read:
            reader = ft.partial(value.read, 2**22)
            filename, full_path = self.filename(key, value)
            iterator = iter(reader, b"")
            size = self._write(full_path, iterator, "xb")
            return size, MODE_BINARY, filename, None
        else:
            result = pickle.dumps(value, protocol=self.pickle_protocol)

            if len(result) < min_file_size:
                return 0, MODE_PICKLE, None, sqlite3.Binary(result)
            else:
                filename, full_path = self.filename(key, value)
                self._write(full_path, io.BytesIO(result), "xb")
                return len(result), MODE_PICKLE, filename, None

    def fetch(self, mode: int, filename: str, value: Any, read: bool) -> Any:
        """
        Convert fields `mode`, `filename`, and `value` from Cache table to value.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        :raises: IOError if the value cannot be read
        """
        # pylint: disable=no-self-use,unidiomatic-typecheck,consider-using-with
        if mode == MODE_RAW:
            return bytes(value) if type(value) is sqlite3.Binary else value
        elif mode == MODE_BINARY:
            if read:
                return open(op.join(self._directory, filename), "rb")
            else:
                with open(op.join(self._directory, filename), "rb") as reader:
                    return reader.read()
        elif mode == MODE_TEXT:
            full_path = op.join(self._directory, filename)
            with open(full_path, "r", encoding="UTF-8") as reader:
                return reader.read()
        elif mode == MODE_PICKLE:
            if value is None:
                with open(op.join(self._directory, filename), "rb") as reader:
                    return pickle.load(reader)
            else:
                return pickle.load(io.BytesIO(value))
