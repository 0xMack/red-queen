from games.checkers import Checkers


def test_reset_places_12_pieces_per_side_on_the_correct_rows():
    board = Checkers().board

    assert len(board) == 24
    player_0 = [pos for pos, (owner, _king) in board.items() if owner == 0]
    player_1 = [pos for pos, (owner, _king) in board.items() if owner == 1]
    assert len(player_0) == 12
    assert len(player_1) == 12
    assert all(y in (0, 1, 2) for _x, y in player_0)
    assert all(y in (5, 6, 7) for _x, y in player_1)
    assert not any(is_king for _owner, is_king in board.values())


def test_opening_position_has_exactly_7_legal_moves():
    # A well-known fact about the standard checkers opening position -- a strong correctness
    # signal if this matches, not a number picked to make the test pass.
    checkers = Checkers()

    moves = checkers.legal_moves()

    assert len(moves) == 7
    assert all(len(m) == 2 for m in moves)  # no captures possible from the opening position


def test_capture_is_mandatory_over_an_available_simple_move():
    checkers = Checkers()
    checkers.board = {(2, 2): (0, False), (3, 3): (1, False), (0, 0): (0, False)}
    checkers.current_player_index = 0

    moves = checkers.legal_moves()

    # (0, 0) also has simple moves available, but the capture at (2,2)->(4,4) must be the only
    # legal move once any capture exists anywhere for the player to move.
    assert moves == [((2, 2), (4, 4))]


def test_capturing_a_piece_removes_it_from_the_board():
    checkers = Checkers()
    checkers.board = {(2, 2): (0, False), (3, 3): (1, False)}
    checkers.current_player_index = 0

    _observation, rewards, done = checkers.step(((2, 2), (4, 4)))

    assert (3, 3) not in checkers.board
    assert checkers.board[(4, 4)] == (0, False)
    # player 1 had exactly one piece, now captured -- no legal moves left, so player 0 wins.
    assert done is True
    assert checkers.winner() == 0
    assert rewards == {0: 1.0, 1: -1.0}


def test_a_double_jump_chain_is_a_single_move_that_captures_both_pieces():
    checkers = Checkers()
    checkers.board = {(0, 0): (0, False), (1, 1): (1, False), (3, 3): (1, False)}
    checkers.current_player_index = 0

    moves = checkers.legal_moves()
    assert moves == [((0, 0), (2, 2), (4, 4))]

    checkers.step(moves[0])

    assert (1, 1) not in checkers.board
    assert (3, 3) not in checkers.board
    assert checkers.board[(4, 4)] == (0, False)


def test_a_man_reaching_the_far_row_is_crowned():
    checkers = Checkers()
    checkers.board = {(2, 6): (0, False)}
    checkers.current_player_index = 0

    checkers.step(((2, 6), (3, 7)))

    assert checkers.board[(3, 7)] == (0, True)


def test_a_king_can_move_and_capture_backward():
    checkers = Checkers()
    checkers.board = {(4, 4): (0, True), (5, 3): (1, False)}
    checkers.current_player_index = 0

    moves = checkers.legal_moves()

    # (5,3) is backward-diagonal from a player-0 piece -- only a king can capture it.
    assert ((4, 4), (6, 2)) in moves


def test_a_player_with_no_legal_moves_loses():
    checkers = Checkers()
    # player 1's man at (0,0) is boxed in: both its forward-diagonal squares are blocked, and
    # player 0's other piece has a simple move available so the turn can actually pass to player 1.
    checkers.board = {(1, 1): (0, False), (2, 0): (0, False), (0, 0): (1, False)}
    checkers.current_player_index = 0

    moves = checkers.legal_moves()
    _observation, rewards, done = checkers.step(moves[0])

    assert checkers.current_player() == 1
    assert checkers.legal_moves() == []
    assert done is True
    assert checkers.winner() == 0
    assert rewards == {0: 1.0, 1: -1.0}


def test_moves_without_capture_limit_ends_the_game_in_a_draw():
    checkers = Checkers(max_moves_without_capture=3)
    checkers.board = {(1, 1): (0, False), (6, 6): (1, False)}
    checkers.current_player_index = 0

    done = False
    rewards = {}
    for _ in range(10):
        moves = checkers.legal_moves()
        _observation, rewards, done = checkers.step(moves[0])
        if done:
            break

    assert done is True
    assert checkers.winner() is None
    assert rewards == {0: 0.0, 1: 0.0}


def test_render_state_labels_pieces_by_owner_and_type():
    checkers = Checkers()
    checkers.board = {(2, 2): (0, False), (5, 5): (1, True)}
    checkers.current_player_index = 1

    state = checkers.render_state()

    assert state["width"] == 8
    assert state["height"] == 8
    assert state["cells"][(2, 2)] == "red_man"
    assert state["cells"][(5, 5)] == "black_king"
    assert state["current_player"] == 1
    assert state["winner"] is None


def test_simulate_does_not_mutate_the_real_environment():
    checkers = Checkers()
    board_before = dict(checkers.board)
    move = checkers.legal_moves()[0]

    result = checkers.simulate(move)

    assert checkers.board == board_before
    assert checkers.current_player() == 0
    assert len(result) == 32


def test_observation_length_is_fixed_regardless_of_piece_count():
    checkers = Checkers()
    checkers.board = {(2, 2): (0, False)}  # far fewer than the initial 24 pieces

    assert len(checkers._observation()) == 32
