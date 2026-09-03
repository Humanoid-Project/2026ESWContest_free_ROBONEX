import argparse
import os
from pathlib import Path

from robonex_common.joints import POLICY_JOINT_ORDER
from robonex_common.limits import action_normalization
from robonex_common.paths import COMMON_REPO_NAMES, DESCRIPTION_REPO_NAMES, git_commit, resolve_repo
from robonex_common.policy import PolicyContract, sha256_file

TASK = "RoboNex-Walking-v0"


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
    description_root = resolve_repo(
        DESCRIPTION_REPO_NAMES,
        env_var="ROBONEX_DESCRIPTION_ROOT",
        explicit=args.description_root,
        anchors=(__file__,),
    )
    common_root = resolve_repo(
        COMMON_REPO_NAMES,
        env_var="ROBONEX_COMMON_ROOT",
        explicit=args.common_root,
        anchors=(__file__,),
    )
    training_root = Path(__file__).resolve().parents[1]
    offsets, scales, clips = action_normalization(0.01)

    contract = PolicyContract(
        schema_version=1,
        task=TASK,
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
