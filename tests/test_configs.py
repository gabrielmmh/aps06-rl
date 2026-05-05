from aps06.configs import (
    ConfigName,
    GridSize,
    PHASE_TIMESTEPS,
    PHASE_MAX_STEPS,
    PHASE_OBSTACLES,
    SEEDS,
    PPO_HYPERPARAMS,
    RECURRENT_HYPERPARAMS,
    CURRICULUM_CHAIN,
    get_phase_n_envs,
)


def test_seeds_are_three():
    assert SEEDS == (0, 1, 2)


def test_grid_sizes_have_timesteps():
    for size in (5, 10, 20):
        assert size in PHASE_TIMESTEPS


def test_grid_sizes_have_obstacles():
    assert PHASE_OBSTACLES == {5: 3, 10: 12, 20: 48}


def test_grid_sizes_have_max_steps():
    assert PHASE_MAX_STEPS == {5: 200, 10: 500, 20: 1000}


def test_n_envs_for_recurrent_is_two_in_all_grids():
    for size in (5, 10, 20):
        assert get_phase_n_envs("curriculum_recurrent", size) == 2


def test_n_envs_for_ppo_is_four_for_small_grids_two_for_large():
    assert get_phase_n_envs("baseline", 5) == 4
    assert get_phase_n_envs("baseline", 10) == 4
    assert get_phase_n_envs("baseline", 20) == 2


def test_config_names_complete():
    expected = {"baseline", "curriculum", "curriculum_enriched", "curriculum_recurrent"}
    assert set(ConfigName.__args__) == expected


def test_grid_size_args():
    assert set(GridSize.__args__) == {5, 10, 20}


def test_curriculum_chain_links_phases():
    assert len(CURRICULUM_CHAIN) == 3
    sizes = [s for s, _ in CURRICULUM_CHAIN]
    assert sizes == [5, 10, 20]
    assert CURRICULUM_CHAIN[0][1] is None
    for i in range(1, len(CURRICULUM_CHAIN)):
        prev_size = CURRICULUM_CHAIN[i - 1][0]
        init_from = CURRICULUM_CHAIN[i][1]
        assert init_from == prev_size


def test_ppo_hyperparams_use_cpu():
    assert PPO_HYPERPARAMS["device"] == "cpu"
    assert PPO_HYPERPARAMS["ent_coef"] == 0.05


def test_recurrent_hyperparams_use_cpu_and_lstm():
    assert RECURRENT_HYPERPARAMS["device"] == "cpu"
    pkw = RECURRENT_HYPERPARAMS["policy_kwargs"]
    assert 32 <= pkw["lstm_hidden_size"] <= 256
    assert pkw["n_lstm_layers"] >= 1
