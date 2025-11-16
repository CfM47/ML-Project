def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_add_two_numbers() -> None:
    """Test the addition of two numbers."""
    assert add_two_numbers(2, 3) == 5
