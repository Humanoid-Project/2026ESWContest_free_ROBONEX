"""Judge an eval_commands json against the walking acceptance criteria."""
import json, sys

# Retention floor, measured on the fwd_max cell. Pilot model_399 reference in comments.
FLOOR = {
    "swing_peak_m":        (">=", 0.035),   # pilot 0.0539
    "single_stance_frac":  (">=", 0.78),    # pilot 0.895
    "duty_l":              ("<=", 0.62),    # pilot 0.568
    "touchdown_hz_l":      ("in", (1.1, 2.2)),   # pilot 1.25
    "failed_env_fraction": ("<=", 0.05),    # fraction of envs that failed at least once
    "clip_consistent":     ("==", True),
}

# Capability per command cell. got_* is used rather than err_*: a walking gait carries a
# large periodic oscillation, so mean-absolute error fails a correct gait and passes a
# planted-foot slide.
CAPABILITY = {
    "stop":        {"touchdown_hz_l": ("<=", 0.3), "failed_env_fraction": ("<=", 0.02)},
    "back":        {"got_vx": ("<=", -0.10), "failed_env_fraction": ("<=", 0.05)},
    "turn_walk_l": {"got_wz": (">=", 0.05), "failed_env_fraction": ("<=", 0.05)},
    "turn_walk_r": {"got_wz": ("<=", -0.05), "failed_env_fraction": ("<=", 0.05)},
    "strafe_l":    {"got_vy": (">=", 0.05), "failed_env_fraction": ("<=", 0.05)},
    "strafe_r":    {"got_vy": ("<=", -0.05), "failed_env_fraction": ("<=", 0.05)},
}


def check(value, rule):
    op, th = rule
    if op == "==": return value == th, "==%s" % th
    if op == ">=": return value >= th, ">=%.4g" % th
    if op == "<=": return value <= th, "<=%.4g" % th
    return th[0] <= value <= th[1], "%.4g-%.4g" % th


def main(path):
    data = json.load(open(path))
    rows, ok = [], True
    for metric, rule in FLOOR.items():
        good, desc = check(data["fwd_max"][metric], rule)
        ok &= good
        rows.append(("retention", "fwd_max", metric, float(data["fwd_max"][metric]), desc, good))
    for cell, checks in CAPABILITY.items():
        if cell not in data:
            rows.append(("capability", cell, "-", float("nan"), "cell missing", False))
            ok = False
            continue
        for metric, rule in checks.items():
            good, desc = check(data[cell][metric], rule)
            ok &= good
            rows.append(("capability", cell, metric, float(data[cell][metric]), desc, good))
    width = max(len(r[2]) for r in rows)
    for kind, cell, metric, value, desc, good in rows:
        print("  %-10s %-12s %-*s %9.4f  %-10s %s"
              % (kind, cell, width, metric, value, desc, "PASS" if good else "FAIL"))
    print("\nVERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
