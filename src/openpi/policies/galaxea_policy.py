import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_galaxea_example() -> dict:
    """Creates a random input example for the Galaxea R1 Lite policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "head_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "wrist_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "wrist_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt": "stack the blocks",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GalaxeaInputs(transforms.DataTransformFn):
    """Inputs for the Galaxea R1 Lite (dual 6-DoF arms + grippers).

    Expected inputs:
    - images: dict with keys "head_left", "wrist_left", "wrist_right"
    - state: [14] = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    - actions: [action_horizon, 14] (training only)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": _parse_image(images["head_left"]),
                "left_wrist_0_rgb": _parse_image(images["wrist_left"]),
                "right_wrist_0_rgb": _parse_image(images["wrist_right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GalaxeaOutputs(transforms.DataTransformFn):
    """Outputs for the Galaxea R1 Lite policy: strip the padding back to 14 action dims."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :14])}
