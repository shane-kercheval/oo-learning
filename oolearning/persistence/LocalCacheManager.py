import hashlib
import os.path
import pickle
import string

from typing import Callable

from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase


class LocalCacheManager(PersistenceManagerBase):
    """
    # TODO: document
    can either set the key in the constructor (perhaps for only retrieving a single object, or can set the
    key when calling get_object (perhaps when retrieving multiple objects from the same directory)
    """
    def __init__(self,
                 cache_directory: str,
                 sub_directory: str=None,
                 key: str=None,
                 key_prefix: str=None,
                 create_dir_if_not_exist: bool=True):
        """
        # TODO document
        :param cache_directory:
        :param key:
        :type key_prefix: object
        :param create_dir_if_not_exist:
        :return:
        """
        super().__init__()
        # if we aren't going to create the directory, it should exist
        if not create_dir_if_not_exist and not os.path.exists(cache_directory):
            raise NotADirectoryError()

        if key_prefix is not None and len(key_prefix) > 100:
            raise ValueError('prefix must be <= 100')

        self._cache_directory = os.path.join(cache_directory, sub_directory) if sub_directory else cache_directory  # noqa
        self._key = key
        self._key_prefix = key_prefix
        self._cache_path = None if key is None else self._create_cached_path(key)

    @property
    def key(self):
        return self._key

    @property
    def key_prefix(self):
        return self._key_prefix

    @property
    def cache_path(self):
        return self._cache_path

    def set_key(self, key: str):
        self._cache_path = self._create_cached_path(key=key)
        self._key = key

    def set_key_prefix(self, prefix: str):
        """
        NOTE: setting the prefix invalidates the current cache_path, so if the key was previously set, it must
            be reset.
        :param prefix: string of max length of 100; all future keys used are prefixed with this string
        """
        if len(prefix) > 100:
            raise ValueError('prefix must be <= 100')

        self._cache_path = None  # setting the prefix invalidates whatever the current cache_path is
        self._key_prefix = prefix

    def get_object(self, fetch_function: Callable[[], object], key: str=None) -> object:
        """
        # TODO document
        :param fetch_function:
        :param key:
        :return:
        """
        # we either have to have the `_cache_path` built up from `key` being passed into constructor or a
        # previous call to `get_object`; or we have to have the `key` passed in
        assert self._cache_path is not None or key is not None

        if key is not None:  # create the path (either for first time or replace existing)
            self._cache_path = self._create_cached_path(key=key)

        # if the cached object exists, then return the cache; else fetch, store, and return the object
        if os.path.isfile(self._cache_path):
            with open(self._cache_path, 'rb') as saved_object:
                return pickle.load(saved_object)
        else:
            # fetch
            fetched_object = fetch_function()
            assert fetched_object is not None
            # save; note, either the directory exists, or it doesn't exist but it is ok to create it
            # (otherwise we would have raised a NotADirectoryError in the constructor)
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, 'xb') as output:
                pickle.dump(fetched_object, output, pickle.HIGHEST_PROTOCOL)

            return fetched_object

    def _create_cached_path(self, key):
        """
        # TODO document

        :param key:
        :return: directory + final file name based off of the key.
            NOTE: if the final file name is > 255 characters, which is the max length in most popular
            operating systems, then the filename is converted to a hash.
        """
        assert key is not None

        cache_path = key if self._key_prefix is None else self._key_prefix + key

        # there might be invalid characters in the cache_path i.e. file name
        valid_chars = "-_.()k%s%s" % (string.ascii_letters, string.digits)
        cache_path = ''.join(c for c in cache_path if c in valid_chars)

        cache_path = cache_path + '.pkl'  # if the length isn't too long, this is the final file name

        if len(cache_path) > 255:  # 255 is max filename length in most popular systems
            # noinspection PyTypeChecker
            prefix = '' if self._key_prefix is None else ''.join(c for c in self._key_prefix if c in valid_chars)  # noqa
            # only hash the key
            cache_path = prefix + hashlib.sha224(key.encode('utf-8')).hexdigest() + '.pkl'

        cache_path = os.path.join(self._cache_directory, cache_path)
        return cache_path if cache_path[0] != '/' else cache_path[1:]

    def set_sub_structure(self, sub_structure: str):
        """
        :param sub_structure: string of max length of 100;
        """
        if len(sub_structure) > 100:
            raise ValueError('sub_structure must be <= 100')

        self._cache_directory = os.path.join(self._cache_directory, sub_structure)
        self._cache_path = None if self._key is None else self._create_cached_path(self._key)
