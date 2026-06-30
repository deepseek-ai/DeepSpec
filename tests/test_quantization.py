import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    CacheDataset,
    CacheCollator,
    build_target_cache_manifest,
    write_target_cache_manifest,
    finalize_target_cache_index,
    LocalCacheWriteSummary,
)

# Mock classes for integration test
class MockConfigObj:
    def __init__(self):
        self.model_type = "qwen3"
        self.hidden_size = 64

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>\n"

    def __call__(self, text, **kwargs):
        class MockEncoding:
            def __init__(self, seq_len):
                self.input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.int32)
                self.attention_mask = torch.ones(1, seq_len, dtype=torch.uint8)
        return MockEncoding(128)

    def encode(self, text, **kwargs):
        length = max(1, len(text) // 2)
        return list(range(length))

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfigObj()
        self.embed_tokens = torch.nn.Embedding(1000, 64)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(2)])
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        
        class MockOutput:
            def __init__(self, last_hidden):
                self.last_hidden_state = last_hidden
        return MockOutput(x.to(dtype=torch.bfloat16))


class TestTargetCacheQuantization(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.seq_len = 128
        self.hidden_size = 64
        self.num_layers = 2
        
        # Generate some mock target data
        torch.manual_seed(42)
        self.mock_hidden_states = torch.randn(self.seq_len, self.num_layers * self.hidden_size) * 0.5
        self.mock_last_hidden_states = torch.randn(self.seq_len, self.hidden_size) * 0.5
        
        self.mock_input_ids = torch.randint(0, 1000, (self.seq_len,), dtype=torch.int32)
        self.mock_attention_mask = torch.ones(self.seq_len, dtype=torch.uint8)
        self.mock_loss_mask = torch.ones(self.seq_len, dtype=torch.uint8)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write_mock_cache(self, output_dir: str, dtype: str):
        os.makedirs(output_dir, exist_ok=True)
        rank_dir = os.path.join(output_dir, "_tmp", "rank_0")
        os.makedirs(rank_dir, exist_ok=True)

        writer = AsyncTargetCacheWriter(
            rank_dir=rank_dir,
            max_shard_bytes=10 * 1024 * 1024, # 10MB
            hidden_dtype=dtype,
        )
        writer.write_sample(
            input_ids=self.mock_input_ids,
            attention_mask=self.mock_attention_mask,
            loss_mask=self.mock_loss_mask,
            target_hidden_states=self.mock_hidden_states,
            target_last_hidden_states=self.mock_last_hidden_states,
        )
        writer.close()

        summary = LocalCacheWriteSummary(
            global_rank=0,
            source_sample_start=0,
            source_sample_end=1,
            num_local_samples=1,
            num_local_shards=1,
            local_shard_files=list(writer.local_shard_files),
        )
        
        src_shard = os.path.join(rank_dir, writer.local_shard_files[0])
        dst_shard = os.path.join(output_dir, "shard-00000.bin")
        shutil.copy(src_shard, dst_shard)

        src_idx = os.path.join(rank_dir, "samples.local.idx")
        dst_idx = os.path.join(output_dir, "samples.idx")
        shutil.copy(src_idx, dst_idx)

        shards_metadata = [{"shard_id": 0, "file_name": "shard-00000.bin"}]
        manifest = build_target_cache_manifest(
            num_samples=1,
            shards=shards_metadata,
            target_layer_ids=[0, 1],
            hidden_size=self.hidden_size,
            hidden_dtype=dtype,
            extra_fields={"target_model_name_or_path": "mock-model"},
        )
        write_target_cache_manifest(output_dir=output_dir, manifest=manifest)
        shutil.rmtree(os.path.join(output_dir, "_tmp"))

    def test_bfloat16_correctness(self):
        cache_path = os.path.join(self.tmp_dir, "bf16_cache")
        self._write_mock_cache(cache_path, "bfloat16")
        
        # Verify sizes
        shard_path = os.path.join(cache_path, "shard-00000.bin")
        bf16_size = os.path.getsize(shard_path)
        
        expected_size = (
            self.seq_len * 4 +
            self.seq_len * 1 +
            self.seq_len * 1 +
            self.seq_len * self.num_layers * self.hidden_size * 2 +
            self.seq_len * self.hidden_size * 2
        )
        self.assertEqual(bf16_size, expected_size)

        # Read back and compare
        dataset = CacheDataset(cache_path)
        sample = dataset[0]
        self.assertEqual(sample["target_hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(sample["target_last_hidden_states"].dtype, torch.bfloat16)
        
        self.assertTrue(torch.allclose(sample["target_hidden_states"], self.mock_hidden_states.to(torch.bfloat16)))
        self.assertTrue(torch.allclose(sample["target_last_hidden_states"], self.mock_last_hidden_states.to(torch.bfloat16)))
        dataset.close()

    def test_float8_e4m3fn_quantization(self):
        cache_path = os.path.join(self.tmp_dir, "fp8_e4m3fn_cache")
        self._write_mock_cache(cache_path, "float8_e4m3fn")
        
        # Verify sizes (should be reduced)
        shard_path = os.path.join(cache_path, "shard-00000.bin")
        fp8_size = os.path.getsize(shard_path)
        
        expected_size = (
            self.seq_len * 4 +
            self.seq_len * 1 +
            self.seq_len * 1 +
            self.seq_len * self.num_layers * self.hidden_size * 1 +
            self.seq_len * self.hidden_size * 1
        )
        self.assertEqual(fp8_size, expected_size)

        # Read back and compare
        dataset = CacheDataset(cache_path)
        sample = dataset[0]
        self.assertEqual(sample["target_hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(sample["target_last_hidden_states"].dtype, torch.bfloat16)
        
        diff_states = (sample["target_hidden_states"] - self.mock_hidden_states.to(torch.bfloat16)).abs().mean().item()
        diff_last = (sample["target_last_hidden_states"] - self.mock_last_hidden_states.to(torch.bfloat16)).abs().mean().item()
        
        print(f"FP8 E4M3FN target_hidden_states MAE: {diff_states}")
        print(f"FP8 E4M3FN target_last_hidden_states MAE: {diff_last}")
        
        self.assertLess(diff_states, 0.05)
        self.assertLess(diff_last, 0.05)
        dataset.close()

    def test_float8_e5m2_quantization(self):
        cache_path = os.path.join(self.tmp_dir, "fp8_e5m2_cache")
        self._write_mock_cache(cache_path, "float8_e5m2")
        
        # Verify sizes (should be reduced)
        shard_path = os.path.join(cache_path, "shard-00000.bin")
        fp8_size = os.path.getsize(shard_path)
        
        expected_size = (
            self.seq_len * 4 +
            self.seq_len * 1 +
            self.seq_len * 1 +
            self.seq_len * self.num_layers * self.hidden_size * 1 +
            self.seq_len * self.hidden_size * 1
        )
        self.assertEqual(fp8_size, expected_size)

        # Read back and compare
        dataset = CacheDataset(cache_path)
        sample = dataset[0]
        self.assertEqual(sample["target_hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(sample["target_last_hidden_states"].dtype, torch.bfloat16)
        
        diff_states = (sample["target_hidden_states"] - self.mock_hidden_states.to(torch.bfloat16)).abs().mean().item()
        diff_last = (sample["target_last_hidden_states"] - self.mock_last_hidden_states.to(torch.bfloat16)).abs().mean().item()
        
        print(f"FP8 E5M2 target_hidden_states MAE: {diff_states}")
        print(f"FP8 E5M2 target_last_hidden_states MAE: {diff_last}")
        
        self.assertLess(diff_states, 0.1)
        self.assertLess(diff_last, 0.1)
        dataset.close()

    def test_cache_collator_and_dataset(self):
        cache_path = os.path.join(self.tmp_dir, "collator_cache")
        self._write_mock_cache(cache_path, "float8_e4m3fn")
        
        dataset = CacheDataset(cache_path)
        collator = CacheCollator()
        
        # Test collating a single sample list
        batch = collator([dataset[0], dataset[0]])
        
        self.assertIn("input_ids", batch)
        self.assertIn("loss_mask", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("target_hidden_states", batch)
        self.assertIn("target_last_hidden_states", batch)
        
        # Verify shapes
        self.assertEqual(batch["input_ids"].shape, (2, self.seq_len))
        self.assertEqual(batch["target_hidden_states"].shape, (2, self.seq_len, self.num_layers * self.hidden_size))
        self.assertEqual(batch["target_last_hidden_states"].shape, (2, self.seq_len, self.hidden_size))
        
        # Verify datatypes after collation
        self.assertEqual(batch["target_hidden_states"].dtype, torch.bfloat16)
        self.assertEqual(batch["target_last_hidden_states"].dtype, torch.bfloat16)
        dataset.close()

    @patch("scripts.data.prepare_target_cache.dist.destroy_process_group")
    @patch("scripts.data.prepare_target_cache.dist.barrier")
    @patch("scripts.data.prepare_target_cache.dist.broadcast_object_list")
    @patch("scripts.data.prepare_target_cache.dist.get_world_size", return_value=1)
    @patch("scripts.data.prepare_target_cache.dist.get_rank", return_value=0)
    @patch("deepspec.utils.distributed.is_global_main_process", return_value=True)
    @patch("deepspec.utils.distributed.is_local_main_process", return_value=True)
    @patch("scripts.data.prepare_target_cache.init_dist")
    @patch("scripts.data.prepare_target_cache.AutoTokenizer.from_pretrained")
    @patch("scripts.data.prepare_target_cache.AutoModel.from_pretrained")
    def test_prepare_target_cache_script_integration(self, mock_from_pretrained_model, mock_from_pretrained_tokenizer, mock_init_dist, mock_is_local, mock_is_global, mock_get_rank, mock_get_world_size, mock_broadcast, mock_barrier, mock_destroy):
        # Configure mocks
        mock_init_dist.return_value = (torch.device("cpu"), 0, 1)
        mock_from_pretrained_tokenizer.return_value = MockTokenizer()
        mock_from_pretrained_model.return_value = MockModel()

        # Create temporary mock config and dataset files
        config_path = os.path.join(self.tmp_dir, "mock_config.py")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("""
import os
model = dict(
    target_model_name_or_path="mock-model",
    target_layer_ids=[0, 1],
)
seed = 42
data = dict(
    chat_template="qwen",
    max_length=128,
)
""")

        dataset_path = os.path.join(self.tmp_dir, "mock_dataset.jsonl")
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"conversations": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}) + "\n")

        output_dir = os.path.join(self.tmp_dir, "integration_output_cache")

        # Mock argv parameters (set min-loss-tokens to 0)
        test_argv = [
            "prepare_target_cache.py",
            "--config", config_path,
            "--train-data-path", dataset_path,
            "--output-dir", output_dir,
            "--hidden-dtype", "float8_e4m3fn",
            "--local-batch-size", "1",
            "--min-loss-tokens", "0",
        ]

        from scripts.data.prepare_target_cache import main as prepare_cache_main
        
        with patch.object(sys, "argv", test_argv):
            # Run the preparation script's main
            prepare_cache_main(0)

        # Verify output exists and is readable
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "samples.idx")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "shard-00000.bin")))

        # Check manifest contents
        with open(os.path.join(output_dir, "manifest.json"), "r") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["hidden_dtype"], "float8_e4m3fn")
        self.assertEqual(manifest["num_samples"], 1)

        # Load using dataset and verify it works
        dataset = CacheDataset(output_dir)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertEqual(sample["target_hidden_states"].shape, (128, 2 * 64))
        self.assertEqual(sample["target_hidden_states"].dtype, torch.bfloat16)
        dataset.close()

if __name__ == '__main__':
    unittest.main()
