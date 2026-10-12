import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "scripts/export_policy_manifest.py"
DESCRIPTION = ROOT.parent / "robonex-description"


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

    def test_the_trained_model_maps_to_its_free_base_scene(self):
        ver23 = self.saved_env("usd_path: /home/x/robonex-description/ver2-3/isaac/edu/closed_loop_mesh/robonex_edu_closed_loop_mesh.usd\n")
        ver22 = self.saved_env("usd_path: /home/x/robonex-description/ver2-2/usd/edu/robonex.usd\n")
        ver2 = self.saved_env("usd_path: /home/x/robonex-description/ver2/usd/edu/robonex.usd\n")
        self.assertEqual(self.export.trained_description_model(ver23), "ver2-3/mujoco/robot/edu/scene.xml")
        self.assertEqual(self.export.trained_description_model(ver22), "ver2-2/mujoco/robot/edu/scene.xml")
        self.assertEqual(self.export.trained_description_model(ver2), "ver2/mujoco/robot/edu/scene.xml")
        self.assertIsNone(self.export.trained_description_model(self.saved_env("usd_path: other\n")))

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
            "ver2/mujoco/robot/edu/scene.xml", "ver2-2/mujoco/robot/edu/scene.xml")
        self.assertIn("ver2-2/mujoco/robot/edu/scene.xml", warning)
        self.assertIsNone(self.export.description_model_warning(
            "ver2-2/mujoco/robot/edu/scene.xml", "ver2-2/mujoco/robot/edu/scene.xml"))
        self.assertIsNone(self.export.description_model_warning(None, "ver2-2/mujoco/robot/edu/scene.xml"))
        self.assertIsNone(self.export.description_model_warning("ver2/mujoco/robot/edu/scene.xml", None))

    def test_the_warning_goes_to_stderr(self):
        self.assertIn("print(warning, file=sys.stderr)", EXPORT.read_text())


if __name__ == "__main__":
    unittest.main()
