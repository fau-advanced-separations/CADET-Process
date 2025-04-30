import copy

import pytest
from CADETProcess.dataStructure.nested_dict import (
    check_nested,
    generate_nested_dict,
    get_leaves,
    get_nested_attribute,
    get_nested_list_value,
    get_nested_value,
    insert_path,
    set_nested_attribute,
    set_nested_list_value,
    set_nested_value,
    update_dict_recursively,
)


@pytest.fixture
def sample_dict():
    """Fixture providing a sample nested dictionary."""
    return {
        "a": {
            "b": {
                "c": 42,
            },
        },
        "x": {
            "y": {
                "z": 99,
            },
        },
    }


@pytest.fixture
def sample_list():
    """Fixture providing a sample nested list."""
    return [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


# --- TESTS FOR NESTED DICTIONARY FUNCTIONS --- #


def test_check_nested(sample_dict):
    assert check_nested(sample_dict, "a.b.c") is True
    assert check_nested(sample_dict, "a.b") is False
    assert check_nested(sample_dict, "x.y.z") is True
    assert check_nested(sample_dict, "missing.path") is False


def test_generate_nested_dict():
    expected = {"a": {"b": {"c": 10}}}
    assert generate_nested_dict("a.b.c", 10) == expected


def test_insert_path(sample_dict):
    insert_path(sample_dict, "a.b.d", 100)
    assert sample_dict["a"]["b"]["d"] == 100

    insert_path(sample_dict, "new.path.here", 50)
    assert sample_dict["new"]["path"]["here"] == 50


def test_get_leaves(sample_dict):
    leaves = set(get_leaves(sample_dict))
    assert leaves == {"a.b.c", "x.y.z"}


def test_set_nested_value(sample_dict):
    set_nested_value(sample_dict, "a.b.c", 77)
    assert sample_dict["a"]["b"]["c"] == 77

    set_nested_value(sample_dict, "x.y.z", "test")
    assert sample_dict["x"]["y"]["z"] == "test"


def test_get_nested_value(sample_dict):
    assert get_nested_value(sample_dict, "a.b.c") == 42
    assert get_nested_value(sample_dict, "x.y.z") == 99

    with pytest.raises(KeyError):
        get_nested_value(sample_dict, "missing.path")


# --- TESTS FOR DICTIONARY UPDATING --- #


def test_update_dict_recursively(sample_dict):
    target = {"a": {"b": 1}, "c": 3}
    updates = {"a": {"b": 2, "d": 4}, "c": 30, "e": 5}

    updated = update_dict_recursively(copy.deepcopy(target), updates)
    assert updated == {"a": {"b": 2, "d": 4}, "c": 30, "e": 5}

    updated_existing = update_dict_recursively(
        copy.deepcopy(target), updates, only_existing_keys=True
    )
    assert updated_existing == {"a": {"b": 2}, "c": 30}


# --- TESTS FOR OBJECT ATTRIBUTE FUNCTIONS --- #


class SampleObject:
    def __init__(self):
        self.a = SampleSubObject()


class SampleSubObject:
    def __init__(self):
        self.b = SampleInnerObject()


class SampleInnerObject:
    def __init__(self):
        self.c = 42


@pytest.fixture
def sample_obj():
    return SampleObject()


def test_get_nested_attribute(sample_obj):
    assert get_nested_attribute(sample_obj, "a.b.c") == 42

    with pytest.raises(AttributeError):
        get_nested_attribute(sample_obj, "a.b.d")


def test_set_nested_attribute(sample_obj):
    set_nested_attribute(sample_obj, "a.b.c", 99)
    assert sample_obj.a.b.c == 99

    with pytest.raises(AttributeError):
        set_nested_attribute(sample_obj, "a.c.b", 50)


# --- TESTS FOR NESTED LIST FUNCTIONS --- #


def test_get_nested_list_value(sample_list):
    assert get_nested_list_value(sample_list, (0, 1, 1)) == 4
    assert get_nested_list_value(sample_list, (1, 0, 1)) == 6

    with pytest.raises(IndexError):
        get_nested_list_value(sample_list, (3, 2, 1))


def test_set_nested_list_value(sample_list):
    set_nested_list_value(sample_list, (0, 1, 1), 99)
    assert sample_list[0][1][1] == 99

    with pytest.raises(IndexError):
        set_nested_list_value(sample_list, (3, 2, 1), 100)


if __name__ == "__main__":
    pytest.main([__file__])
