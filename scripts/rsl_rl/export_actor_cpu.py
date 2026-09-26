import argparse
from pathlib import Path

import torch
import torch.nn as nn


class ExportedActor(nn.Module):
    def __init__(self, state_dict, eps):
        super().__init__()
        self.register_buffer("mean", state_dict["actor_obs_normalizer._mean"].clone())
        self.register_buffer("std", state_dict["actor_obs_normalizer._std"].clone())
        self.eps = eps
        layers = []
        index = 0
        while f"actor.{index}.weight" in state_dict:
            weight = state_dict[f"actor.{index}.weight"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(state_dict[f"actor.{index}.bias"])
            layers.append(linear)
            if f"actor.{index + 2}.weight" in state_dict:
                layers.append(nn.ELU())
            index += 2
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net((obs - self.mean) / (self.std + self.eps))


def main():
    parser = argparse.ArgumentParser(
        description="Export an rsl_rl actor (observation normalizer + ELU MLP) to ONNX without starting Isaac."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-2, help="rsl_rl EmpiricalNormalization eps")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    actor = ExportedActor(state_dict, args.eps).eval()
    observation_size = state_dict["actor.0.weight"].shape[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor, torch.zeros(1, observation_size), str(args.output),
        input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False,
    )
    print(f"{args.output}  (observation {observation_size}, iteration {checkpoint.get('iter')})")


if __name__ == "__main__":
    main()
