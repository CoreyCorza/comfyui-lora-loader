import importlib
import sys
import types
import unittest

import torch


def _install_comfy_stubs():
    class DummyNodeOutput(tuple):
        def __new__(cls, *values):
            return super().__new__(cls, values)

    comfy = types.ModuleType("comfy")
    comfy.sd = types.ModuleType("comfy.sd")
    comfy.utils = types.ModuleType("comfy.utils")

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = types.SimpleNamespace(ComfyNode=object, NodeOutput=DummyNodeOutput, Schema=object)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda *_args, **_kwargs: []

    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.sd", comfy.sd)
    sys.modules.setdefault("comfy.utils", comfy.utils)
    sys.modules.setdefault("comfy_api", comfy_api)
    sys.modules.setdefault("comfy_api.latest", latest)
    sys.modules.setdefault("folder_paths", folder_paths)


class FakeLoRAAdapter:
    name = "lora"

    def __init__(self, loaded_keys, weights):
        self.loaded_keys = loaded_keys
        self.weights = weights


class FakeModelPatcher:
    def __init__(self, patches):
        self.patches = patches
        self.patches_uuid = "original-uuid"

    def clone(self):
        clone = FakeModelPatcher({key: value[:] for key, value in self.patches.items()})
        clone.patches_uuid = self.patches_uuid
        return clone


class AppliedLoRACleanerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_comfy_stubs()
        cls.lora_loader = importlib.import_module("lora_loader")

    def test_cleans_standard_lora_adapters_attached_to_model_patches(self):
        up = torch.diag(torch.tensor([4.0, 1.0]))
        down = torch.eye(2)
        adapter = FakeLoRAAdapter({"up", "down"}, (up, down, 2.0, None, None, None))
        model = FakeModelPatcher(
            {
                "diffusion_model.block.weight": [
                    (0.75, adapter, 1.0, None, None),
                ],
            }
        )

        cleaned = self.lora_loader._clean_model_lora_patches(
            model,
            keep_energy=100.0,
            max_rank=1,
            tame_layers=0.0,
            star_rescale=False,
        )

        self.assertIsNot(cleaned, model)
        self.assertIs(cleaned.patches["diffusion_model.block.weight"][0][1].__class__, FakeLoRAAdapter)
        self.assertEqual(model.patches_uuid, "original-uuid")
        self.assertNotEqual(cleaned.patches_uuid, model.patches_uuid)

        strength, cleaned_adapter, strength_model, offset, function = cleaned.patches[
            "diffusion_model.block.weight"
        ][0]
        self.assertEqual((strength, strength_model, offset, function), (0.75, 1.0, None, None))
        cleaned_up, cleaned_down, cleaned_alpha, mid, dora_scale, reshape = cleaned_adapter.weights
        self.assertEqual(cleaned_up.shape, (2, 1))
        self.assertEqual(cleaned_down.shape, (1, 2))
        self.assertEqual(cleaned_alpha, 1.0)
        self.assertIsNone(mid)
        self.assertIsNone(dora_scale)
        self.assertIsNone(reshape)

        original_up, original_down, original_alpha, *_ = adapter.weights
        self.assertEqual(original_up.shape, (2, 2))
        self.assertEqual(original_down.shape, (2, 2))
        self.assertEqual(original_alpha, 2.0)

    def test_leaves_uncleanable_lora_adapters_unchanged(self):
        up = torch.diag(torch.tensor([4.0, 1.0]))
        down = torch.eye(2)
        adapter = FakeLoRAAdapter({"up", "down"}, (up, down, 2.0, torch.eye(2), None, None))
        model = FakeModelPatcher(
            {
                "diffusion_model.block.weight": [
                    (1.0, adapter, 1.0, None, None),
                ],
            }
        )

        cleaned = self.lora_loader._clean_model_lora_patches(
            model,
            keep_energy=100.0,
            max_rank=1,
            tame_layers=0.0,
            star_rescale=False,
        )

        self.assertIs(cleaned, model)
        self.assertIs(model.patches["diffusion_model.block.weight"][0][1], adapter)

    def test_registers_model_only_applied_patch_cleaner_node(self):
        self.assertIn("CorzaCleanAppliedLoRAs", self.lora_loader.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            self.lora_loader.NODE_DISPLAY_NAME_MAPPINGS["CorzaCleanAppliedLoRAs"],
            "Corza Clean Applied LoRAs",
        )

        up = torch.diag(torch.tensor([4.0, 1.0]))
        down = torch.eye(2)
        adapter = FakeLoRAAdapter({"up", "down"}, (up, down, 2.0, None, None, None))
        model = FakeModelPatcher(
            {
                "diffusion_model.block.weight": [
                    (1.0, adapter, 1.0, None, None),
                ],
            }
        )

        node_cls = self.lora_loader.NODE_CLASS_MAPPINGS["CorzaCleanAppliedLoRAs"]
        output = node_cls.execute(
            model,
            keep_energy=100.0,
            max_rank=1,
            tame_layers=0.0,
            star_rescale=False,
        )

        self.assertEqual(len(output), 1)
        self.assertIsNot(output[0], model)


if __name__ == "__main__":
    unittest.main()
