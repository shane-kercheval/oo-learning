from oolearning import *
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


class MockCloneable(Cloneable):
    pass


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

    def test_CloneableFactory_MockCloneable(self):
        ##########################
        # test single object MockCloneable
        ##########################
        original = MockCloneable()
        factory = CloneableFactory(cloneable=original)
        new1 = factory.get()
        new2 = factory.get()

        assert isinstance(original, MockCloneable)
        assert isinstance(new1, MockCloneable)
        assert isinstance(new2, MockCloneable)
        assert original is original
        assert original is not new1
        assert original is not new2
        assert new1 is not new2

        ##########################
        # test single object MockSingleUseObject
        ##########################
        original = MockSingleUseObject(TempObject(value=1))
        factory = CloneableFactory(cloneable=original)
        new1 = factory.get()
        new2 = factory.get()

        assert isinstance(original, MockSingleUseObject)
        assert isinstance(new1, MockSingleUseObject)
        assert isinstance(new2, MockSingleUseObject)
        assert original is original
        assert original is not new1
        assert original is not new2
        assert new1 is not new2

        # test MockSingleUseObject works as expected after generating
        new1.execute()

        assert new1.has_executed
        assert new1.temp_object.value == 2  # check that _execute() is called
        new1.ensure_has_executed()

        # ensure original object did not change
        assert not original.has_executed
        assert original.temp_object.value == 1
        assert original.clone() is not None  # we should still be able to clone because object hasn't executed
        original.ensure_has_not_executed()
        self.assertRaises(NotExecutedError,
                          lambda: original.ensure_has_executed())

        ##########################
        # test list of objects (MockCloneable & MockSingleUseObject)
        ##########################
        original1 = MockCloneable()
        original2 = MockSingleUseObject(TempObject(value=1))

        factory = CloneableFactory(cloneable=[original1, original2])
        new1 = factory.get()
        new2 = factory.get()

        assert len(new1) == 2
        assert len(new2) == 2

        assert isinstance(new1[0], MockCloneable)
        assert isinstance(new1[1], MockSingleUseObject)
        assert isinstance(new2[0], MockCloneable)
        assert isinstance(new2[1], MockSingleUseObject)

        assert all([x is not original1 and x is not original2 for x in new1])
        assert all([x is not original1 and x is not original2 for x in new2])
