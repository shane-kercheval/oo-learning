from oolearning import *
from oolearning.model_wrappers.ModelExceptions import AlreadyExecutedError, NotExecutedError
from tests.TimerTestCase import TimerTestCase


class TempObject:
    def __init__(self, value: int):
        self._value = value

    @property
    def value(self):
        return self._value

    def execute(self):
        self._value += 1


class MockSingleUseObject(SingleUseObjectMixin):
    def __init__(self, temp_object: TempObject,
                 already_executed_exception_class=None,
                 not_executed_exception_class=None):
        super().__init__(already_executed_exception_class=already_executed_exception_class,
                         not_executed_exception_class=not_executed_exception_class)
        self._temp_object = temp_object

    @property
    def temp_object(self):
        return self._temp_object

    def additional_cloning_checks(self):
        assert self._temp_object.value != 1000

    def _execute(self):
        self._temp_object.execute()


# noinspection PyMethodMayBeStatic,SpellCheckingInspection,PyTypeChecker
class SingleUseObjectTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_SingleUseObject_mock(self):
        # SUOs can
        #   only be use once
        #   can only clone before using
        #   use their own exception types
        starting_value = 1
        suo = MockSingleUseObject(TempObject(value=starting_value))
        assert suo.has_executed is False
        assert suo.temp_object.value == starting_value
        suo.ensure_has_not_executed()
        self.assertRaises(NotExecutedError,
                          lambda: suo.ensure_has_executed())

        # clone and ensure the clone is a different object (and sub-objects are different)
        suo_clone = suo.clone()
        assert suo is not suo_clone
        assert suo.temp_object is not suo_clone.temp_object
        assert suo.temp_object.value == suo_clone.temp_object.value

        suo_clone.execute()

        assert suo_clone.has_executed
        assert suo_clone.temp_object.value == starting_value + 1  # check that _execute() is called
        suo_clone.ensure_has_executed()
        self.assertRaises(AlreadyExecutedError,
                          lambda: suo_clone.ensure_has_not_executed())

        # ensure original object did not change
        assert not suo.has_executed
        assert suo.temp_object.value == starting_value
        assert suo.clone() is not None  # we should still be able to clone because this object hasn't executed
        suo.ensure_has_not_executed()
        self.assertRaises(NotExecutedError,
                          lambda: suo.ensure_has_executed())
        # ensure we cannot clone or execute again
        self.assertRaises(AlreadyExecutedError,
                          lambda: suo_clone.clone())

        self.assertRaises(AlreadyExecutedError,
                          lambda: suo_clone.execute())

    def test_SingleUseObject_mock_custom_exception_additional_checks(self):
        # SUOs can
        #   only be use once
        #   can only clone before using
        #   use their own exception types
        starting_value = 1000
        suo = MockSingleUseObject(TempObject(value=starting_value))
        assert suo.has_executed is False
        assert suo.temp_object.value == starting_value
        suo.ensure_has_not_executed()
        self.assertRaises(NotExecutedError,
                          lambda: suo.ensure_has_executed())
        # we should get an assertion error because the additional_cloning_checks in the Mock object
        # doesn't allow for values of 1000
        self.assertRaises(AssertionError,
                          lambda: suo.clone())

        starting_value = 1
        suo = MockSingleUseObject(TempObject(value=starting_value),
                                  already_executed_exception_class=ModelAlreadyFittedError,
                                  not_executed_exception_class=ModelNotFittedError)
        assert not suo.has_executed
        assert suo.temp_object.value == starting_value
        suo_clone = suo.clone()
        assert suo is not suo_clone
        assert suo.temp_object is not suo_clone.temp_object
        assert suo.temp_object.value == suo_clone.temp_object.value

        suo_clone.execute()
        assert suo_clone.has_executed
        assert suo_clone.temp_object.value == starting_value + 1
        suo_clone.ensure_has_executed()
        self.assertRaises(ModelAlreadyFittedError,
                          lambda: suo_clone.ensure_has_not_executed())
        # ensure we cannot clone or execute again
        self.assertRaises(ModelAlreadyFittedError,
                          lambda: suo_clone.clone())

        self.assertRaises(ModelAlreadyFittedError,
                          lambda: suo_clone.execute())

        # ensure original object did not change
        assert not suo.has_executed
        assert suo.temp_object.value == starting_value
        assert suo.clone() is not None  # we should still be able to clone because this object hasn't executed
        suo.ensure_has_not_executed()
        # ensure we cannot clone or execute again
        self.assertRaises(ModelNotFittedError,
                          lambda: suo.ensure_has_executed())
