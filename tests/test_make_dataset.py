from delivery_time.data.make_dataset import dummy_sum


def test_dummy_sum():
    assert dummy_sum(2, 3) == 5
