import argparse
import os
from pathlib import Path

from robonex_common.joints import POLICY_JOINT_ORDER
from robonex_common.limits import RUNNER_ACTION_CLIP, action_normalization
from robonex_common.runtime import (
    OBSERVATION_HISTORY_LENGTH,
    OBSERVATION_SIZE,
    OBSERVATION_TERM_SIZES,
)
from robonex_common.paths import COMMON_REPO_NAMES, DESCRIPTION_REPO_NAMES, git_commit, resolve_repo
from robonex_common.policy import (
    PolicyContract,
    mujoco_bundle_sha256,
    python_source_sha256,
    sha256_file,
)

TASK = "RoboNex-Walking-v0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--description-root", type=Path)
    parser.add_argument("--common-root", type=Path)
    parser.add_argument("--description-model", default="mujoco/robot/scene.xml")
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
        schema_version=2,
        task=TASK,
        policy_file=os.path.relpath(policy, output.parent),
        policy_sha256=sha256_file(policy),
        description_sha256=mujoco_bundle_sha256(description_root, args.description_model),
        common_sha256=python_source_sha256(common_root, ("src/robonex_common",)),
        training_sha256=python_source_sha256(training_root, ("source", "scripts")),
        description_model=args.description_model,
        joint_order=POLICY_JOINT_ORDER,
        observation_terms=tuple(
            f"{name}:{size}x{OBSERVATION_HISTORY_LENGTH}" for name, size in OBSERVATION_TERM_SIZES
        ),
        action_offsets=tuple(offsets[name] for name in POLICY_JOINT_ORDER),
        action_scales=tuple(scales[name] for name in POLICY_JOINT_ORDER),
        target_clips=tuple(clips[name] for name in POLICY_JOINT_ORDER),
        runner_action_clip=RUNNER_ACTION_CLIP,
        observation_size=OBSERVATION_SIZE,
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
