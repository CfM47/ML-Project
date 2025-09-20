from src.core.example import add_two_numbers


def test_add_two_numbers() -> None:
    """Test the addition of two numbers."""
    assert add_two_numbers(2, 3) == 5
