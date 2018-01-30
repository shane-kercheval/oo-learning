import time
import unittest

SLOW_TEST_THRESHOLD = 1


class TimerTestCase(unittest.TestCase):
    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        if elapsed > SLOW_TEST_THRESHOLD:
            print('{} - WARNING: SLOW ({}s)'.format(self.id(), round(elapsed, 2)))
        else:
            print(self.id())
