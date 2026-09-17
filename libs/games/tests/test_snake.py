from games.snake import Snake, benchmark_environments


def test_reset_returns_a_flattened_grid_of_the_expected_length():
    snake = Snake(width=5, height=4, seed=0)

    observation = snake.reset()

    assert len(observation) == 5 * 4
    assert observation.count(1.0) == 2  # 2 body segments (3-segment snake minus the head)
    assert observation.count(2.0) == 1  # the head
    assert observation.count(3.0) == 1  # the food


def test_reset_is_deterministic_for_a_given_seed():
    a = Snake(width=8, height=8, seed=42)
    b = Snake(width=8, height=8, seed=42)

    assert a.food == b.food

    a.step(0)
    b.step(0)
    assert a.food == b.food
    assert a.body == b.body


def test_turning_right_then_moving_straight_changes_heading():
    snake = Snake(width=10, height=10, seed=0)
    head_before = snake.body[0]

    snake.step(1)  # turn right: RIGHT -> DOWN
    head_after_turn = snake.body[0]

    assert head_after_turn == (head_before[0], head_before[1] + 1)  # moved down, not right

    snake.step(0)  # go straight while heading DOWN
    head_after_straight = snake.body[0]

    assert head_after_straight == (head_after_turn[0], head_after_turn[1] + 1)


def test_wall_collision_ends_the_episode():
    snake = Snake(width=10, height=10, seed=0)
    # drive straight off the right edge
    done = False
    for _ in range(20):
        _observation, reward, done = snake.step(0)
        if done:
            break

    assert done is True
    assert snake.alive is False
    assert reward == -1.0


def test_eating_food_grows_the_snake_and_increases_score():
    snake = Snake(width=10, height=10, seed=0)
    snake.food = (snake.body[0][0] + 1, snake.body[0][1])  # one cell ahead
    length_before = len(snake.body)

    _observation, reward, done = snake.step(0)

    assert snake.score == 1
    assert len(snake.body) == length_before + 1
    assert reward == 1.0
    assert done is False


def test_moving_without_eating_does_not_grow_the_snake():
    snake = Snake(width=10, height=10, seed=0)
    snake.food = (0, 0)  # far away, won't be eaten this step
    length_before = len(snake.body)

    _observation, reward, done = snake.step(0)

    assert len(snake.body) == length_before
    assert reward == -0.02  # moving right while food is far to the left -> farther away
    assert done is False


def test_moving_toward_food_gives_a_small_positive_shaping_reward():
    snake = Snake(width=10, height=10, seed=0)
    snake.food = (snake.body[0][0] + 5, snake.body[0][1])  # straight ahead, far away

    _observation, reward, done = snake.step(0)  # straight, toward the food -> closer

    assert reward == 0.01
    assert done is False


def test_moving_into_the_vacated_tail_cell_is_legal():
    snake = Snake(width=5, height=5, seed=0)
    snake.body = [(2, 2), (2, 1), (1, 1), (1, 2)]  # head=(2,2) ... tail=(1,2)
    snake._direction_index = 2  # LEFT
    snake.food = (4, 4)  # irrelevant to this move

    _observation, _reward, done = snake.step(0)  # straight (LEFT) -> head moves onto old tail cell

    assert done is False
    assert snake.alive is True
    assert snake.body[0] == (1, 2)


def test_moving_into_a_non_tail_body_segment_is_a_collision():
    snake = Snake(width=5, height=5, seed=0)
    snake.body = [(2, 2), (2, 1), (1, 1), (1, 2), (1, 3)]  # (1,2) is not the tail here
    snake._direction_index = 2  # LEFT -> next head (1,2), a body segment that does not vacate
    snake.food = (4, 4)

    _observation, reward, done = snake.step(0)

    assert done is True
    assert snake.alive is False
    assert reward == -1.0


def test_starvation_ends_the_episode_without_eating():
    snake = Snake(width=10, height=10, seed=0, max_steps_without_food=3)
    snake.food = (0, 0)  # unreachable within the starvation limit while moving straight

    done = False
    for _ in range(5):
        _observation, _reward, done = snake.step(0)
        if done:
            break

    assert done is True
    assert snake.alive is False


def test_render_state_labels_head_body_and_food_distinctly():
    snake = Snake(width=10, height=10, seed=0)

    state = snake.render_state()

    assert state["width"] == 10
    assert state["height"] == 10
    assert state["cells"][snake.body[0]] == "head"
    assert state["cells"][snake.body[1]] == "body"
    assert state["cells"][snake.food] == "food"
    assert state["score"] == 0
    assert state["alive"] is True


def test_benchmark_environments_use_distinct_seeds():
    envs = benchmark_environments()

    assert len(envs) == 5
    assert len({e.seed for e in envs}) == 5
