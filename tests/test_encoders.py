import pytest
import numpy as np

from arc25.encoders import create_grid_encoder

@pytest.mark.parametrize("encoder_name", [
    'MinimalGridEncoder()',
    "GridWithSeparationEncoder('|')",
    'GridCodeBlockEncoder(MinimalGridEncoder())',
    'GridCodeBlockEncoder(GridWithSeparationEncoder("|"))',
    'GridCodeBlockEncoder(RepeatNumberEncoder(3))',
    'GridCodeBlockEncoder(RepeatNumberEncoder(2))',
    'GridShapeEncoder(MinimalGridEncoder())',
    'GridShapeEncoder(RepeatNumberEncoder(3))',
    'GridCodeBlockEncoder(RowNumberEncoder(MinimalGridEncoder()))',
    'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))',
    'GridShapeEncoder(ReplaceNumberEncoder(MinimalGridEncoder()))',
    'GridShapeEncoder(RowNumberEncoder(ReplaceNumberEncoder(MinimalGridEncoder())))'
])
def test_grid_encoder_is_reversible(encoder_name):
    grid_encoder = create_grid_encoder(encoder_name)
    sample_grid = np.eye(3, dtype=int).tolist()
    sample_grid = np.reshape(np.arange(9), (3, 3)).tolist()
    assert sample_grid == grid_encoder.to_grid(grid_encoder.to_text(sample_grid))
    print(grid_encoder.to_text(sample_grid) + '\n')