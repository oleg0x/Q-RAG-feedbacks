from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf


def top_level_config_names():
    return sorted(
        path.stem
        for path in Path("configs").glob("*.yaml")
        if path.name.startswith(("training", "testing"))
    )


def test_all_top_level_configs_compose_and_resolve():
    with initialize(version_base="1.3", config_path="../configs"):
        for name in top_level_config_names():
            cfg = compose(config_name=name)
            OmegaConf.to_container(cfg, resolve=True)
            if name.startswith("training"):
                assert cfg.algo.action_model._target_
                assert cfg.envs.env._target_
                assert cfg.envs.test_env._target_


def test_base_training_defaults_are_preserved():
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="training")
    assert cfg.envs.task == "hotpotqa_2wiki_q0_s1"
    assert cfg.feedback.type == "candidate_beta"
    assert cfg.algo.model.model_name == "Alibaba-NLP/gte-multilingual-base"
    assert cfg.get("eval_strategy", "random_with_replacement") == "random_with_replacement"


def test_qicl_configs_do_not_reference_foreign_home():
    for path in Path("configs").glob("*.yaml"):
        assert "/home/o.inozemcev" not in path.read_text(encoding="utf-8")
