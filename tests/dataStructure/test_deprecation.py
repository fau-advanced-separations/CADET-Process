import unittest

# Import the functions to test
from CADETProcess.dataStructure import deprecated_alias, rename_kwargs


class TestDeprecatedAlias(unittest.TestCase):
    def test_basic_functionality(self):
        @deprecated_alias(old_arg="new_arg")
        def example_func(new_arg):
            return new_arg

        # Test with new argument name
        self.assertEqual(example_func(new_arg=42), 42)

        # Test with old argument name
        with self.assertWarns(DeprecationWarning) as warning_context:
            result = example_func(old_arg=42)
            self.assertEqual(result, 42)

        # Verify warning message
        warning_message = str(warning_context.warning)
        self.assertIn("`old_arg` is deprecated", warning_message)
        self.assertIn("use `new_arg` instead", warning_message)

    def test_argument_conflict(self):
        """Test error when both old and new argument names are used."""

        @deprecated_alias(old_arg="new_arg")
        def example_func(new_arg):
            return new_arg

        # Should raise TypeError when both old and new names are used
        with self.assertRaises(TypeError) as context:
            example_func(old_arg=1, new_arg=2)

        self.assertIn(
            "received both old_arg and new_arg as arguments", str(context.exception)
        )

    def test_rename_kwargs_direct(self):
        """Test rename_kwargs function directly."""
        kwargs = {"old_arg": 42}
        aliases = {"old_arg": "new_arg"}

        with self.assertWarns(DeprecationWarning):
            rename_kwargs("test_func", kwargs, aliases)

        self.assertIn("new_arg", kwargs)
        self.assertNotIn("old_arg", kwargs)
        self.assertEqual(kwargs["new_arg"], 42)

    def test_positional_args(self):
        """Test that positional arguments work normally with the decorator."""

        @deprecated_alias(old_kwarg="new_kwarg")
        def example_func(pos_arg, new_kwarg=None):
            return f"{pos_arg}-{new_kwarg}"

        # Test with positional argument
        self.assertEqual(example_func("test", new_kwarg=42), "test-42")

        # Test with old keyword argument
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(example_func("test", old_kwarg=42), "test-42")

    def test_no_arguments(self):
        """Test function with no arguments still works with decorator."""

        @deprecated_alias(old_arg="new_arg")
        def example_func():
            return "success"

        self.assertEqual(example_func(), "success")


if __name__ == "__main__":
    unittest.main()
