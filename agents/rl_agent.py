from pathlib import Path

try:
    import torch

    from src.config import load_train_config
    from src.features import encode_turn
    from src.opponents import actions_from_indices
    from src.ppo import sample_actions
    from src.train import build_policy, resolve_device
except Exception:
    torch = None

from agents import agent_a


_POLICY = None
_CFG = None
_DEVICE = None


def _load_policy():
    global _POLICY, _CFG, _DEVICE
    if _POLICY is not None or torch is None:
        return

    _CFG = load_train_config()
    _DEVICE = resolve_device("cpu")
    _POLICY = build_policy(_CFG, _DEVICE)
    checkpoint = Path(_CFG.save_dir) / _CFG.run_name / "ckpt_last.pt"
    if not checkpoint.exists():
        _POLICY = None
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _POLICY.load_state_dict(payload["policy"])
    _POLICY.eval()


def act(obs):
    _load_policy()
    if _POLICY is None:
        return agent_a.act(obs)

    batch = encode_turn(obs, _CFG.env)
    if batch.self_features.shape[0] == 0:
        return []

    with torch.inference_mode():
        outputs = _POLICY(
            torch.from_numpy(batch.self_features).to(_DEVICE),
            torch.from_numpy(batch.candidate_features).to(_DEVICE),
            torch.from_numpy(batch.global_features).to(_DEVICE),
            torch.from_numpy(batch.candidate_mask).to(_DEVICE).bool(),
        )
        sampled = sample_actions(outputs, deterministic=True)

    return actions_from_indices(batch, sampled.target_index.cpu().tolist())
