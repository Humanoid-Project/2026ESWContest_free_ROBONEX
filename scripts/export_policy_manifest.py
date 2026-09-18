import argparse
import os
from pathlib import Path

from robonex_common.joints import POLICY_JOINT_ORDER
from robonex_common.limits import RUNNER_ACTION_CLIP, action_normalization
from robonex_common.runtime import (
    OBSERVATION_HISTORY_LENGTH,
    ACTION_SIZE,
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

RECEIPT_VERSION = 1


DEFAULT_POLICY_HZ = 50.0


def policy_hz_from_saved_env(saved_env):
    """Policy rate from a run's saved `params/env.yaml`, or None if it cannot be read.

    `1 / (sim.dt * decimation)` is the rate the checkpoint was actually trained at, which is what
    the manifest should carry. Importing the task config instead would pull in `isaaclab.sim` and
    force a Kit launch on every export, so the YAML is read directly.
    """
    if saved_env is None or not saved_env.is_file():
        return None
    try:
        import yaml

        cfg = yaml.unsafe_load(saved_env.read_text())
        dt = float(cfg["sim"]["dt"])
        decimation = int(cfg["decimation"])
    except Exception:
        return None
    if dt <= 0 or decimation <= 0:
        return None
    return 1.0 / (dt * decimation)


def write_receipt(manifest_path, policy, checkpoint, training_root, contract):
    """Record what this manifest was actually built from, beside it.

    The manifest itself is reconstructed entirely from the *current* checkout, so exporting
    an old checkpoint stamps it with today's geometry and nothing downstream can tell. This
    sidecar does not change the schema-2 contract; it states the provenance the manifest
    cannot carry, including the case where the checkpoint was not supplied.
    """
    import json
    from datetime import datetime, timezone

    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": sha256_file(manifest_path),
        "policy_sha256": contract.policy_sha256,
        "training_commit": contract.training_commit,
        "training_source_sha256": contract.training_sha256,
    }
    if checkpoint is None:
        receipt["checkpoint"] = None
        receipt["provenance"] = (
            "UNVERIFIED: no --checkpoint given. Every geometry field in the manifest comes "
            "from the current checkout, not from the run that produced this policy."
        )
    else:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run_dir = checkpoint.parent
        saved_env = run_dir / "params" / "env.yaml"
        receipt["checkpoint"] = str(checkpoint)
        receipt["checkpoint_sha256"] = sha256_file(checkpoint)
        receipt["run_dir"] = str(run_dir)
        receipt["policy_hz_source"] = "checkpoint params/env.yaml"
        receipt["saved_env_yaml_sha256"] = (
            sha256_file(saved_env) if saved_env.is_file() else None
        )
        receipt["provenance"] = (
            "checkpoint recorded; the manifest's geometry is still read from the current "
            "checkout and is not verified against the saved config"
            if saved_env.is_file()
            else "checkpoint recorded, but the run saved no params/env.yaml to compare against"
        )
    path = manifest_path.with_name("export_receipt.json")
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if checkpoint is None:
        print("WARNING: no --checkpoint given; %s records this manifest as UNVERIFIED" % path)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--description-root", type=Path)
    parser.add_argument("--common-root", type=Path)
    parser.add_argument("--description-model", default="mujoco/robot/scene.xml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="The .pt this ONNX was exported from. Recorded in the sidecar receipt so the "
             "manifest can be traced back to a training artifact; without it the receipt "
             "says so explicitly",
    )
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

    # Prefer the rate the checkpoint was trained at over a literal. policy_hz drives the deploy
    # control period and the gait clock, so a stale value changes what the robot does with no
    # shape mismatch to catch it.
    policy_hz = DEFAULT_POLICY_HZ
    if args.checkpoint is not None:
        derived = policy_hz_from_saved_env(
            args.checkpoint.expanduser().resolve().parent / "params" / "env.yaml"
        )
        if derived is not None:
            policy_hz = derived
        else:
            print("WARNING: could not read sim.dt/decimation from the checkpoint's params/env.yaml; "
                  f"falling back to {DEFAULT_POLICY_HZ} Hz")

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
        action_size=ACTION_SIZE,
        policy_hz=policy_hz,
        description_commit=git_commit(description_root),
        common_commit=git_commit(common_root),
        training_commit=git_commit(training_root),
    )
    contract.save(output)
    write_receipt(output, policy, args.checkpoint, training_root, contract)
    print(output)


if __name__ == "__main__":
    main()
