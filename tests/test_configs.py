from broom.configs import (
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
    expected = {
        "baseline",
        "curriculum",
        "curriculum_enriched",
        "curriculum_recurrent",
        "curriculum_recurrent_v2",
        "mapcnn_bc_pbrs",
        "maskable_v3",
        "maskable_bc_kl",
        "maskable_frontier_pbrs",
    }
    assert set(ConfigName.__args__) == expected


def test_mapcnn_bc_pbrs_hyperparams_use_cuda_long_horizon_and_long_rollout():
    from broom.configs import MAPCNN_BC_PBRS_HYPERPARAMS, BC_WARMSTART_PATH, PBRS_GAMMA
    assert MAPCNN_BC_PBRS_HYPERPARAMS["device"] == "cuda"
    assert MAPCNN_BC_PBRS_HYPERPARAMS["gamma"] >= 0.99
    assert MAPCNN_BC_PBRS_HYPERPARAMS["n_steps"] >= 1024
    assert BC_WARMSTART_PATH.endswith(".zip")
    assert 0.9 <= PBRS_GAMMA <= 1.0


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


def test_recurrent_v2_hyperparams_use_cuda_larger_lstm_and_longer_rollouts():
    from broom.configs import RECURRENT_V2_HYPERPARAMS

    assert RECURRENT_V2_HYPERPARAMS["device"] == "cuda"
    assert RECURRENT_V2_HYPERPARAMS["n_steps"] == 512
    pkw = RECURRENT_V2_HYPERPARAMS["policy_kwargs"]
    assert pkw["lstm_hidden_size"] == 256
    assert pkw["n_lstm_layers"] == 1


def test_v2_uses_full_n_envs_unlike_v1():
    # v1 caps at 2 envs everywhere because LSTM hidden state lives in CPU RAM.
    # v2 frees that constraint by moving the LSTM to GPU, so it returns to
    # the MLP pattern: 4 envs in 5x5/10x10, 2 in 20x20.
    assert get_phase_n_envs("curriculum_recurrent_v2", 5) == 4
    assert get_phase_n_envs("curriculum_recurrent_v2", 10) == 4
    assert get_phase_n_envs("curriculum_recurrent_v2", 20) == 2


def test_maskable_v3_hyperparams_use_cuda_long_horizon_and_entropy_schedule():
    from broom.configs import MASKABLE_V3_HYPERPARAMS, _maskable_v3_entropy_schedule

    assert MASKABLE_V3_HYPERPARAMS["device"] == "cuda"
    assert MASKABLE_V3_HYPERPARAMS["gamma"] >= 0.99
    assert MASKABLE_V3_HYPERPARAMS["n_steps"] >= 1024
    pkw = MASKABLE_V3_HYPERPARAMS["policy_kwargs"]
    assert pkw["net_arch"] == [256, 256]

    # Schedule decays from ~0.02 at start to ~0.001 at end
    assert _maskable_v3_entropy_schedule(1.0) == 0.02
    assert _maskable_v3_entropy_schedule(0.0) == 0.001
    # Monotone decreasing
    mid = _maskable_v3_entropy_schedule(0.5)
    assert 0.001 < mid < 0.02


def test_maskable_bc_kl_hyperparams_use_smaller_lr_and_kl_constants():
    from broom.configs import (
        BC_V3_WARMSTART_PATH,
        KL_LAMBDA_DECAY_TIMESTEPS,
        KL_LAMBDA_FINAL,
        KL_LAMBDA_INITIAL,
        MASKABLE_BC_KL_HYPERPARAMS,
    )

    assert MASKABLE_BC_KL_HYPERPARAMS["device"] == "cuda"
    assert MASKABLE_BC_KL_HYPERPARAMS["gamma"] >= 0.99
    # Smaller LR than maskable_v3 since we're fine-tuning a BC warm-started policy
    assert MASKABLE_BC_KL_HYPERPARAMS["learning_rate"] < 3e-4
    assert MASKABLE_BC_KL_HYPERPARAMS["learning_rate"] >= 1e-5
    assert BC_V3_WARMSTART_PATH.endswith(".zip")
    # KL anchor schedule sane values
    assert 0.5 <= KL_LAMBDA_INITIAL <= 1.5
    assert 0.0 <= KL_LAMBDA_FINAL <= 0.2
    assert KL_LAMBDA_DECAY_TIMESTEPS >= 1_000_000


def test_kl_lambda_schedule_decays_linearly():
    from broom.maskable_bc_kl import make_kl_lambda_schedule

    schedule = make_kl_lambda_schedule(initial=1.0, final=0.05, decay_over_timesteps=1_000_000)
    assert schedule(0) == 1.0
    assert schedule(1_000_000) == 0.05
    # Half-way: linear interp = 0.525
    assert abs(schedule(500_000) - 0.525) < 0.01
    # Beyond schedule: clamped at final
    assert schedule(2_000_000) == 0.05


def test_maskable_frontier_pbrs_hyperparams_use_long_horizon_calibration():
    from broom.configs import (
        CONFIG_MAX_STEPS_OVERRIDE,
        MASKABLE_FRONTIER_PBRS_HYPERPARAMS,
        get_max_steps,
    )

    assert MASKABLE_FRONTIER_PBRS_HYPERPARAMS["device"] == "cuda"
    # gamma 0.995 (not 0.999) — middle ground for ~200-step effective horizon
    assert MASKABLE_FRONTIER_PBRS_HYPERPARAMS["gamma"] == 0.995
    # n_steps 2048 (longer rollouts than maskable_bc_kl's 1024)
    assert MASKABLE_FRONTIER_PBRS_HYPERPARAMS["n_steps"] == 2048
    # learning_rate 5e-5 (smaller than maskable_v3's 3e-4 to reduce drift)
    assert MASKABLE_FRONTIER_PBRS_HYPERPARAMS["learning_rate"] == 5e-5
    # clip_range 0.1 (tighter than default 0.2 — prevents large updates)
    assert MASKABLE_FRONTIER_PBRS_HYPERPARAMS["clip_range"] == 0.1
    # Net arch [256, 256]
    assert MASKABLE_FRONTIER_PBRS_HYPERPARAMS["policy_kwargs"]["net_arch"] == [256, 256]


def test_max_steps_override_for_frontier_pbrs_on_20x20():
    from broom.configs import get_max_steps

    # Other configs unchanged
    assert get_max_steps("baseline", 20) == 1000
    assert get_max_steps("maskable_bc_kl", 20) == 1000
    # frontier_pbrs gets 1500 on 20x20 (more time to close)
    assert get_max_steps("maskable_frontier_pbrs", 20) == 1500
    # frontier_pbrs unchanged on smaller grids
    assert get_max_steps("maskable_frontier_pbrs", 5) == 200
    assert get_max_steps("maskable_frontier_pbrs", 10) == 500
