import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "scripts/export_policy_manifest.py"
DESCRIPTION = Path(os.environ.get("ROBONEX_DESCRIPTION_ROOT", ROOT.parent / "robonex-description"))


def export_module():
    spec = importlib.util.spec_from_file_location("export_policy_manifest_scene", EXPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportSceneTests(unittest.TestCase):
    def setUp(self):
        self.export = export_module()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def saved_env(self, text):
        path = Path(self.tmp.name) / f"env{len(list(Path(self.tmp.name).iterdir()))}.yaml"
        path.write_text(text)
        return path

    def run_env(self, run_dir, model):
        path = Path(self.tmp.name) / run_dir / "params" / "env.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(f"usd_path: /home/x/robonex-description/{model}/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd\n")
        return path

    def test_the_trained_model_maps_to_its_free_base_scene(self):
        before_rename = self.run_env("2026-10-27_14-09-53_W156s44_v2_3_edu_ds60_s44", "ver2-3")
        ver2 = self.run_env("2026-11-03_01-00-00_W161s43_v2_edu_x_s43", "ver2")
        self.assertEqual(self.export.trained_description_model(before_rename), "ver2/mujoco/robot/edu/scene.xml")
        self.assertEqual(self.export.trained_description_model(ver2), "ver2/mujoco/robot/edu/scene.xml")
        self.assertIsNone(self.export.trained_description_model(self.saved_env("usd_path: other\n")))

    def test_removed_models_are_refused(self):
        old_ver2 = self.run_env("2026-09-30_22-49-43_W96_v2_edu_friction", "ver2")
        ver2_2 = self.run_env("2026-10-08_01-09-02_W113s43_v2_2_edu_base_s43", "ver2-2")
        with self.assertRaisesRegex(SystemExit, "old ver2"):
            self.export.trained_description_model(old_ver2)
        with self.assertRaisesRegex(SystemExit, "ver2-2"):
            self.export.trained_description_model(ver2_2)

    def test_an_unknown_model_needs_an_explicit_description_model(self):
        self.assertIn("pass --description-model explicitly", EXPORT.read_text())

    def test_no_default_names_a_fixed_base_scene(self):
        scenes = list(self.export.DESCRIPTION_MODELS.values()) + list(self.export.SCENE_BY_TRAINED_MODEL.values())
        for scene in scenes:
            self.assertTrue(scene.endswith("/scene.xml"), scene)

    def test_every_default_scene_has_the_free_root_joint(self):
        if not DESCRIPTION.is_dir():
            self.skipTest("robonex-description is not next to this checkout")
        scenes = set(self.export.DESCRIPTION_MODELS.values()) | set(self.export.SCENE_BY_TRAINED_MODEL.values())
        for scene in scenes:
            self.assertIn('<include file="robonex.xml"/>', (DESCRIPTION / scene).read_text(), scene)
            robot = (DESCRIPTION / scene).parent / "robonex.xml"
            self.assertIn('<freejoint name="root"/>', robot.read_text(), scene)

    def test_a_description_model_from_another_model_warns(self):
        warning = self.export.description_model_warning(
            "ver1/mujoco/robot/scene.xml", "ver2/mujoco/robot/edu/scene.xml")
        self.assertIn("ver2/mujoco/robot/edu/scene.xml", warning)
        self.assertIsNone(self.export.description_model_warning(
            "ver2/mujoco/robot/edu/scene.xml", "ver2/mujoco/robot/edu/scene.xml"))
        self.assertIsNone(self.export.description_model_warning(None, "ver2/mujoco/robot/edu/scene.xml"))
        self.assertIsNone(self.export.description_model_warning("ver2/mujoco/robot/edu/scene.xml", None))

    def test_the_warning_goes_to_stderr(self):
        self.assertIn("print(warning, file=sys.stderr)", EXPORT.read_text())


if __name__ == "__main__":
    unittest.main()
