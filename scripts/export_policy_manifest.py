import argparse
import os
import subprocess
from pathlib import Path

from robonex_common.joints import POLICY_JOINT_ORDER
from robonex_common.limits import action_normalization
from robonex_common.policy import PolicyContract, sha256_file


DESCRIPTION_REPO_NAMES = ("robonex-description", "robonex_description")


def find_repo(name, configured=None):
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.is_dir():
            return root
        raise FileNotFoundError(root)
    names = (name,) if isinstance(name, str) else tuple(name)
    for anchor in (Path.cwd().resolve(), Path(__file__).resolve()):
        for parent in (anchor, *anchor.parents):
            for candidate_name in names:
                candidate = parent / candidate_name
                if candidate.is_dir():
                    return candidate
    raise FileNotFoundError(names[0])


def git_commit(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--description-root", type=Path)
    parser.add_argument("--common-root", type=Path)
    parser.add_argument("--description-model", default="mujoco/scene.xml")
    args = parser.parse_args()

    policy = args.policy.expanduser().resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    output = args.output.expanduser().resolve() if args.output else policy.with_name("policy_manifest.json")
    description_root = find_repo(
        DESCRIPTION_REPO_NAMES,
        args.description_root or os.environ.get("ROBONEX_DESCRIPTION_ROOT"),
    )
    common_root = find_repo("robonex-common", args.common_root or os.environ.get("ROBONEX_COMMON_ROOT"))
    training_root = Path(__file__).resolve().parents[1]
    offsets, scales, clips = action_normalization(0.01)

    contract = PolicyContract(
        schema_version=1,
        task="RoboNex-Walking-v0",
        policy_file=os.path.relpath(policy, output.parent),
        policy_sha256=sha256_file(policy),
        description_model=args.description_model,
        joint_order=POLICY_JOINT_ORDER,
        observation_terms=("joint_pos_rel:12", "joint_vel_rel:12", "imu_ang_vel:3", "projected_gravity:3", "last_action:12"),
        action_offsets=tuple(offsets[name] for name in POLICY_JOINT_ORDER),
        action_scales=tuple(scales[name] for name in POLICY_JOINT_ORDER),
        target_clips=tuple(clips[name] for name in POLICY_JOINT_ORDER),
        runner_action_clip=3.0,
        observation_size=42,
        action_size=12,
        policy_hz=50.0,
        description_commit=git_commit(description_root),
        common_commit=git_commit(common_root),
        training_commit=git_commit(training_root),
    )
    contract.save(output)
    print(output)


if __name__ == "__main__":
    main()
