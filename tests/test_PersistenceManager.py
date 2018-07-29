import os
import shutil

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic
class PersistenceManagerTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_LocalCacheObject_with_constructor(self):

        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')

        ######################################################################################################
        # at this point, the expected_directory should not exist
        ######################################################################################################
        self.assertRaises(NotADirectoryError, lambda: LocalCacheManager(expected_directory,
                                                                        create_dir_if_not_exist=False))
        self.assertRaises(NotADirectoryError, lambda: LocalCacheManager(expected_directory,
                                                                        key='test_key_constructor',
                                                                        create_dir_if_not_exist=False))
        ######################################################################################################
        # test using key in constructor
        ######################################################################################################
        expected_key = 'test_key_constructor'
        expected_object = 'test expected_object'
        persistence_object = LocalCacheManager(cache_directory=expected_directory, key=expected_key)
        assert persistence_object.cache_path == os.path.join(expected_directory, expected_key+'.pkl')
        # neither the directory or file should exist yet for this test
        assert os.path.isdir(expected_directory) is False
        assert os.path.isfile(persistence_object.cache_path) is False

        fetched_object = persistence_object.get_object(fetch_function=lambda: expected_object)
        assert fetched_object == expected_object

        # should have created both the directory and the file
        assert os.path.isfile(persistence_object.cache_path)

        # now, since the object is cached, we can ensure that we should be grabbing the cached item by
        # changing the fetch_function, and ensuring that `get_object` still returns the old (cached) value,
        # not 'JUNK' as would be returned if we 'fetched' via fetch_function
        fetched_object = persistence_object.get_object(fetch_function=lambda: 'JUNK')
        assert fetched_object == expected_object
        assert os.path.isfile(persistence_object.cache_path)

        ######################################################################################################
        # test using key in get_object after using key in constructor
        ######################################################################################################
        previous_path = persistence_object.cache_path
        previous_key = expected_key
        previous_object = expected_object
        # test with a new key and new object
        expected_key = 'test_key_after_constructor'
        expected_object = 'test expected_object using key in get_object after constructor'

        expected_new_path = os.path.join(expected_directory, expected_key+'.pkl')
        assert os.path.isfile(expected_new_path) is False
        fetched_object = persistence_object.get_object(fetch_function=lambda: expected_object, key=expected_key)  # noqa
        assert fetched_object == expected_object
        # now we should have a new path
        assert persistence_object.cache_path == expected_new_path
        # should have created the new file
        assert os.path.isfile(expected_new_path)
        # but also should have retained the old file
        assert os.path.isfile(previous_path)
        # so if we reload our previously cached item, we should get it, which again we can test by passing in
        # junk to our `fetch_function` and ensuring we didn't get it, but instead got the previous object
        fetched_object = persistence_object.get_object(fetch_function=lambda: 'JUNK', key=previous_key)
        assert fetched_object == previous_object

        shutil.rmtree(expected_directory)

    def test_LocalCacheObject_without_constructor(self):
        ######################################################################################################
        # test using key in get_object without using key in constructor
        ######################################################################################################
        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')

        expected_key = 'test_key_in_get_object'
        expected_object = 'test expected_object using key in get_object without using key in constructor'
        persistence_object = LocalCacheManager(cache_directory=expected_directory)
        # this time, _cache_path should be None because we haven't set it
        assert persistence_object.cache_path is None
        # neither the directory or file should exist yet for this test
        assert os.path.isdir(expected_directory) is False

        # we should get an AssertionError if we are trying to use `get_object` and never passed in a key
        self.assertRaises(AssertionError, lambda: persistence_object.get_object(fetch_function=lambda: expected_object))  # noqa

        # now we can try with a key
        fetched_object = persistence_object.get_object(fetch_function=lambda: expected_object, key=expected_key)  # noqa
        assert fetched_object == expected_object

        # should have created both the directory and the file
        assert os.path.isfile(persistence_object.cache_path)

        fetched_object = persistence_object.get_object(fetch_function=lambda: 'JUNK')
        assert fetched_object == expected_object
        assert os.path.isfile(persistence_object.cache_path)

        shutil.rmtree(expected_directory)

    def test_LocalCacheObject_set_key(self):
        ######################################################################################################
        # test using key in `set_key` without using key in constructor
        # objects like Resamplers will use a CacheManager, but won't know the key at the time of creation
        # (i.e. constructor) and will also not know or be able to pass the key when creating/retrieving the
        # object (i.e. `train()` located in ModelWrapperBase), so have to have a way to set the key during,
        # for example, each iteration of the resampling.
        ######################################################################################################
        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')

        expected_key = 'test_key_in_set_key'
        expected_object = 'test expected_object using key set in the `set_key` method'
        persistence_object = LocalCacheManager(cache_directory=expected_directory)

        # neither the directory or file should exist yet for this test
        assert os.path.isdir(expected_directory) is False

        # we should get an AssertionError if we are trying to use `get_object` and never passed in a key,
        # and no key was set in `set_key`, yet
        self.assertRaises(AssertionError, lambda: persistence_object.get_object(fetch_function=lambda: expected_object))  # noqa

        # now we can try with a key
        assert persistence_object.cache_path is None
        persistence_object.set_key(key=expected_key)
        assert persistence_object.key == expected_key
        assert persistence_object.cache_path == os.path.join(expected_directory, expected_key + '.pkl')

        assert os.path.isfile(persistence_object.cache_path) is False
        fetched_object = persistence_object.get_object(fetch_function=lambda: expected_object)
        assert fetched_object == expected_object
        # should have created both the directory and the file
        assert os.path.isfile(persistence_object.cache_path)

        # test it goes to cache
        fetched_object = persistence_object.get_object(fetch_function=lambda: 'JUNK')
        assert fetched_object == expected_object
        assert os.path.isfile(persistence_object.cache_path)

        # test that using set_key again with different key works
        previous_path = persistence_object.cache_path
        previous_key = expected_key
        previous_object = expected_object
        # test with a new key and new object
        expected_key = 'test_set_key_after_set_key'
        expected_object = 'test expected_object using key in set_key second time'

        assert persistence_object.cache_path == previous_path
        persistence_object.set_key(key=expected_key)
        expected_new_path = os.path.join(expected_directory, expected_key + '.pkl')
        assert persistence_object.cache_path == expected_new_path

        assert os.path.isfile(expected_new_path) is False
        fetched_object = persistence_object.get_object(fetch_function=lambda: expected_object)
        assert fetched_object == expected_object
        # path should be the same
        assert persistence_object.cache_path == expected_new_path
        # should have created the new file
        assert os.path.isfile(expected_new_path)
        # but also should have retained the old file
        assert os.path.isfile(previous_path)
        # so if we reload our previously cached item, we should get it, which again we can test by passing in
        # junk to our `fetch_function` and ensuring we didn't get it, but instead got the previous object
        fetched_object = persistence_object.get_object(fetch_function=lambda: 'JUNK', key=previous_key)
        assert fetched_object == previous_object

        shutil.rmtree(expected_directory)

    def test_LocalCacheObject_invalid_keys(self):
        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')

        invalid_key = 'invalid key because it has spaces'
        valid_key = ''.join([x for x in invalid_key if x != ' '])

        persistence_object = LocalCacheManager(cache_directory=expected_directory, key=invalid_key)
        assert persistence_object.cache_path == os.path.join(expected_directory, valid_key + '.pkl')

    def test_LocalCacheObject_long_keys(self):
        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')

        # try valid key
        valid_key = 'a'*(255 - len('.pkl'))
        persistence_object = LocalCacheManager(cache_directory=expected_directory, key=valid_key)
        assert persistence_object.cache_path == os.path.join(expected_directory, valid_key + '.pkl')

        # file too long with prefix only converts key, not prefix
        invalid_key = 'a'*(256 - len('.pkl'))
        # raw string; that way we can check that we get the same hash value across sessions
        valid_key = '2e5fc80386a5c7df9425d72af5f7c8f89eb0f50563f2869ec3bfb108.pkl'
        persistence_object = LocalCacheManager(cache_directory=expected_directory, key=invalid_key)
        assert persistence_object.cache_path == os.path.join(expected_directory, valid_key)

    def test_LocalCacheObject_key_prefix(self):
        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')
        expected_key = 'expected_key'
        expected_prefix = 'prefix_'

        # if we set the key (thereby setting the path) before we call `set_key_prefix` the cache_path should
        # be set back to NULL
        persistence_object = LocalCacheManager(cache_directory=expected_directory, key=expected_key)
        assert persistence_object.cache_path == os.path.join(expected_directory, expected_key + '.pkl')
        persistence_object.set_key_prefix(prefix=expected_prefix)
        assert persistence_object.cache_path is None
        persistence_object.set_key(key=expected_key)
        assert persistence_object.cache_path == os.path.join(expected_directory,
                                                              expected_prefix + expected_key + '.pkl')
        # test invalid prefix
        invalid_prefix = 'invalid prefix_'
        invalid_key = 'invalid key because it has spaces'
        valid_key = ''.join([x for x in invalid_prefix+invalid_key if x != ' '])

        persistence_object = LocalCacheManager(cache_directory=expected_directory,
                                               key=invalid_key,
                                               key_prefix=invalid_prefix)
        assert persistence_object.cache_path == os.path.join(expected_directory, valid_key + '.pkl')

        # test long key with prefix added (but wouldn't have been long without prefix)
        # make sure what I think is valid to actually be valid
        key_limit = 'a' * (255 - len('.pkl'))
        persistence_object = LocalCacheManager(cache_directory=expected_directory, key=key_limit)
        assert persistence_object.cache_path == os.path.join(expected_directory, key_limit + '.pkl')

        # prefix pushes the filename over the limit; however, the key_prefix should never be changed/hashed
        # noinspection SpellCheckingInspection
        # raw string; that way we can check that we get the same hash value across sessions
        expected_key = 'invalidprefix_198712a23db7ae113f047178daeb9c8b15854383ae98410e20374f72.pkl'
        # we are passing in an invalid prefix to test that the prefix still gets converted, but not hashed
        persistence_object = LocalCacheManager(cache_directory=expected_directory,
                                               key=key_limit,
                                               key_prefix=invalid_prefix)
        assert persistence_object.cache_path == os.path.join(expected_directory, expected_key)

    def test_LocalCacheObject_invalid_key_prefix_length(self):
        expected_directory = TestHelper.ensure_test_directory('data/test_Local_CacheObject_directory')

        valid_key_prefix = 'a'*100

        persistence_object = LocalCacheManager(cache_directory=expected_directory, key='key', key_prefix=valid_key_prefix)  # noqa
        assert persistence_object.key_prefix == valid_key_prefix
        persistence_object.set_key_prefix('b'*100)
        assert persistence_object.key_prefix == 'b'*100

        invalid_key_prefix = 'a' * 101
        self.assertRaises(ValueError, lambda: LocalCacheManager(cache_directory=expected_directory, key='key', key_prefix=invalid_key_prefix))  # noqa
        persistence_object = LocalCacheManager(cache_directory=expected_directory, key='key', key_prefix=valid_key_prefix)  # noqa
        self.assertRaises(ValueError, lambda: persistence_object.set_key_prefix(prefix=invalid_key_prefix))
