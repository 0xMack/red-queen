from games.rendering import render_grid_ascii


def test_render_grid_ascii_places_labeled_cells_and_fills_empty():
    cells = {(0, 0): "head", (1, 0): "food"}
    symbols = {"head": "@", "food": "*"}

    result = render_grid_ascii(width=2, height=2, cells=cells, symbols=symbols)

    assert result == "@*\n.."
