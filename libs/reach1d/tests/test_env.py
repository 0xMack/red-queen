from reach1d import ReachTarget1D, benchmark_environments


def test_reset_returns_position_relative_to_target():
    env = ReachTarget1D(target=5.0, start_position=1.0, start_velocity=2.0)

    assert env.reset() == (1.0 - 5.0, 2.0)


def test_step_matches_hand_traced_physics():
    env = ReachTarget1D(
        target=5.0, start_position=0.0, start_velocity=0.0, dt=0.1, max_acceleration=1.0, damping=0.98
    )
    env.reset()

    (relative_position, velocity), reward, done = env.step(action=1.0)

    expected_velocity = (0.0 + 1.0 * 0.1) * 0.98
    expected_position = 0.0 + expected_velocity * 0.1
    assert velocity == expected_velocity
    assert relative_position == expected_position - 5.0
    assert reward == -abs(expected_position - 5.0)
    assert done is False


def test_action_is_clamped_to_unit_range():
    env = ReachTarget1D(start_position=0.0, start_velocity=0.0, dt=0.1, max_acceleration=1.0, damping=1.0)
    env.reset()
    (_relative_position, velocity_from_huge_action), _reward, _done = env.step(action=1000.0)

    env.reset()
    (_relative_position, velocity_from_one), _reward, _done = env.step(action=1.0)

    assert velocity_from_huge_action == velocity_from_one


def test_reward_improves_as_position_approaches_target():
    env = ReachTarget1D(target=5.0, start_position=0.0, start_velocity=0.0)
    env.reset()

    _obs, first_reward, _done = env.step(action=1.0)
    _obs, second_reward, _done = env.step(action=1.0)

    # still approaching the target from below -> distance shrinks -> reward improves
    assert second_reward > first_reward


def test_reset_after_stepping_restores_the_original_start_state():
    env = ReachTarget1D(target=5.0, start_position=0.0, start_velocity=0.0)
    env.reset()
    env.step(action=1.0)
    env.step(action=1.0)

    assert env.reset() == (0.0 - 5.0, 0.0)


def test_two_scenarios_with_the_same_start_position_are_distinguishable_from_the_observation():
    # the whole reason the observation is relative-to-target rather than raw position: two
    # episodes starting at the same position but with different targets must not look identical
    # at reset, or no single policy could ever solve both.
    reach_right = ReachTarget1D(target=5.0, start_position=0.0, start_velocity=0.0)
    reach_left = ReachTarget1D(target=-5.0, start_position=0.0, start_velocity=0.0)

    assert reach_right.reset() != reach_left.reset()


def test_benchmark_environments_are_a_fixed_set_of_distinct_scenarios():
    envs = benchmark_environments()

    assert len(envs) == 5
    starts = {(e.start_position, e.target) for e in envs}
    assert len(starts) == len(envs)  # every scenario is actually distinct
