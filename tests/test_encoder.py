"""Fast shape/mask/graph-interface tests; no downloaded checkpoint needed."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from encoder import StaticEncoder, attention_mask, output_length


def tiny_tower():
    tower = nn.Module()
    tower.config = SimpleNamespace(d_model=8, encoder_attention_heads=2)
    tower.conv2d1 = nn.Conv2d(1, 4, 3, 2, 1)
    tower.conv2d2 = nn.Conv2d(4, 4, 3, 2, 1)
    tower.conv2d3 = nn.Conv2d(4, 4, 3, 2, 1)
    tower.conv_out = nn.Linear(4*16, 8, bias=False)
    tower.positional_embedding = nn.Module()
    tower.positional_embedding.register_buffer("positional_embedding", torch.randn(13, 8))
    layer = nn.Module()
    layer.self_attn_layer_norm = nn.LayerNorm(8)
    layer.self_attn = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        setattr(layer.self_attn, name, nn.Linear(8, 8))
    layer.self_attn.scaling = 0.5
    layer.final_layer_norm = nn.LayerNorm(8)
    layer.fc1, layer.fc2 = nn.Linear(8, 16), nn.Linear(16, 8)
    layer.activation_fn = nn.GELU()
    tower.layers = nn.ModuleList([layer])
    tower.ln_post = nn.LayerNorm(8)
    tower.proj1, tower.act, tower.proj2 = nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 12)
    return tower.eval()


class EncoderContracts(unittest.TestCase):
    def test_lengths(self):
        for frames, expected in [(1, 1), (99, 13), (100, 13), (101, 14), (137, 18),
                                 (200, 26), (317, 42), (400, 52), (1437, 187), (1500, 195)]:
            self.assertEqual(output_length(frames), expected)
        with self.assertRaises(ValueError):
            output_length(0)

    def test_mask_blocks_and_padding(self):
        mask = attention_mask(1500, 1437)[0, 0]
        self.assertEqual(mask.shape, (195, 195))
        self.assertEqual(mask[103, 103], 0)
        self.assertEqual(mask[103, 104], -10000)
        self.assertEqual(mask[104, 103], -10000)
        self.assertEqual(mask[104, 186], 0)
        self.assertEqual(mask[104, 187], -10000)
        sparse = attention_mask(1500, 100)[0, 0]
        self.assertTrue(np.isfinite(sparse).all())
        self.assertTrue(np.all(np.any(sparse == 0, axis=1)))

    def test_reject_unsupported_valid_lengths(self):
        for frames, valid in [(200, 99), (200, 201), (201, 200)]:
            with self.assertRaises(ValueError):
                attention_mask(frames, valid)

    def test_output_shape_and_window_independence(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        tower = tiny_tower()
        original_attention = tower.layers[0].self_attn
        model = StaticEncoder(tower, 1500)
        features = torch.randn(1, 128, 1500)
        mask = torch.from_numpy(attention_mask(1500))
        changed = features.clone()
        changed[:, :, 800:] += 10
        with torch.inference_mode():
            first = model(features, mask)
            second = model(changed, mask)
        self.assertEqual(first.shape, (1, 195, 12))
        self.assertTrue(torch.equal(first[:, :104], second[:, :104]))
        self.assertGreater(torch.max(torch.abs(first[:, 104:]-second[:, 104:])).item(), 1e-5)
        self.assertIs(tower.layers[0].self_attn, original_attention)


if __name__ == "__main__":
    unittest.main()
