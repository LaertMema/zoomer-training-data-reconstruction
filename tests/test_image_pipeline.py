from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from image_pipeline import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUN_RASTER_MAGIC,
    center_crop,
    preprocess_bgr_for_resnet,
    write_ras_bgr,
)
from scene_prior_retrieval import load_prompts  # noqa: E402


class ImagePipelineTests(unittest.TestCase):
    def test_center_crop_uses_image_center(self) -> None:
        image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        cropped = center_crop(image, 4)
        np.testing.assert_array_equal(cropped, image[1:5, 2:6])

    def test_preprocess_has_expected_shape_dtype_and_channel_order(self) -> None:
        image_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        image_bgr[..., 2] = 255

        result = preprocess_bgr_for_resnet(image_bgr)

        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, np.dtype("<f4"))
        expected = (np.array([1.0, 0.0, 0.0], dtype=np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(result[:, 0, 0], expected, rtol=1e-6, atol=1e-6)

    def test_sun_raster_header_matches_payload(self) -> None:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.ras"
            write_ras_bgr(path, image)
            payload = path.read_bytes()

        header = struct.unpack(">8I", payload[:32])
        self.assertEqual(header[0], SUN_RASTER_MAGIC)
        self.assertEqual(header[1:4], (3, 2, 24))
        self.assertEqual(header[4], image.nbytes)
        self.assertEqual(payload[32:], image.tobytes())

    def test_prompt_loader_ignores_comments_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text("# group\nfirst prompt\n\n  # another group\nsecond prompt\n")
            self.assertEqual(load_prompts(path), ["first prompt", "second prompt"])


if __name__ == "__main__":
    unittest.main()
