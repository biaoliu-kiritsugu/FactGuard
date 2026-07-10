import jsonlines
import os
from typing import Callable


class DedupReader:
    def __init__(self, filename, key, *, filter_func=None) -> None:
        """
        DedupReader is a class that reads a file and keeps track of the keys in the file.
        It can be used to filter out examples that have the same key as an example already seen.
        :param filename: the file to read
        :param key: a string or callable that returns the key for an example
        :param filter: a callable that returns True if an example should be filtered out
        """
        self.keys = set()
        if isinstance(key, str):

            def key_func(ex):
                val = ex[key]
                if isinstance(val, list):
                    val = tuple(val)
                return val
        elif isinstance(key, Callable):
            key_func = key
        else:
            raise TypeError(f"key must be a str or callable, got {type(key)}")

        self.key_func = key_func
        if filter_func:
            self.filter_func = filter_func
        else:

            def filter_func(x):
                return False

            self.filter_func = filter_func

        if filename and os.path.exists(filename):
            self.add_file(filename)

    def add_example(self, example):
        self.keys.add(self.key_func(example))

    def add_key(self, key):
        self.keys.add(key)

    def add_file(self, filename):
        with jsonlines.open(filename, "r") as reader:
            for example in reader:
                if not self.filter_func(example):
                    self.add_example(example)

    def not_contains(self, example):
        key = self.key_func(example)
        return key not in self.keys

    def contains_key(self, key):
        return key in self.keys

    def __contains__(self, example):
        key = self.key_func(example)
        return key in self.keys

    def dedup(self, example):
        return filter(self.not_contains, example)

    def __len__(self):
        return len(self.keys)
