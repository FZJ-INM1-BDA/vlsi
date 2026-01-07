import numpy as np
import pytest
from vlsi import SpatialIndex

validate_pos = SpatialIndex.validate_pos

# Parametrized test cases for valid inputs
@pytest.mark.parametrize(
    "input_pos, expected_shape",
    [
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), (2, 3)),
        (np.array([[1, 2, 3]]), (1, 3)),
        ([[9, 10, 11], [12, 13, 14]], (2, 3)),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=int), (2, 3)),
    ]
)
def test_valid_inputs(input_pos, expected_shape):
    """Test that valid inputs are correctly processed and cast to uint32."""
    result = validate_pos(input_pos)
    assert result.shape == expected_shape
    assert result.dtype == np.uint32

@pytest.mark.parametrize(
    "input_pos",
    [
        np.array([1, 2, 3]),                            # 1D array
        np.array([[[1, 2]], [[3, 4]], [[5, 6]]]),     # 3D array
        np.array([[1, 2], [3, 4]]),                    # Wrong column size
        np.array([[]]),                                # Empty array
    ]
)
def test_invalid_shapes(input_pos):
    """Test that invalid shapes raise AssertionError."""
    with pytest.raises(AssertionError) as exc_info:
        validate_pos(input_pos)
    # Verify the error messages are informative
    error_msg = str(exc_info.value)
    assert "len(pos.shape) to be 2" in error_msg or "expecting Nx3 array" in error_msg

def test_large_values():
    """Test handling of large values near uint32 limit."""
    max_value = 2**31 - 1
    input_pos = np.array([[max_value, max_value - 1, max_value - 2]])
    result = validate_pos(input_pos)

    assert result.shape == (1, 3)
    assert result.dtype == np.uint32
    assert np.array_equal(result, input_pos)

def test_float_values():
    """Test conversion of float values to uint32."""
    input_pos = np.array([[1.5, 2.7, 3.8]], dtype=np.float64)
    result = validate_pos(input_pos)

    # Test that float values were properly cast to uint32 (truncated or floored)
    assert result.shape == (1, 3)
    assert result.dtype == np.uint32
    assert result[0, 0] == 1  # Should be 1 after truncation