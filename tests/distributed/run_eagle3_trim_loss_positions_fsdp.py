from __future__ import annotations

import copy
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import deepspec.modeling.eagle3.loss as loss_module
from deepspec.modeling.eagle3.loss import compute_eagle3_loss
from deepspec.modeling.eagle3.qwen3.modeling import Qwen3Eagle3Model
from deepspec.trainer.base_trainer import _build_fsdp_kwargs


def tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
    )
    config.target_layer_ids = [0, 1, 2, 3, 4]
    config.ttt_length = 3
    config.step_loss_decay = 0.8
    config.trim_loss_positions = False
    config._attn_implementation = "flex_attention"
    return config


def build_model(device: torch.device) -> Qwen3Eagle3Model:
    model = Qwen3Eagle3Model(tiny_config()).to(
        device=device, dtype=torch.bfloat16
    )
    model.embed_tokens.requires_grad_(False)
    model.lm_head.requires_grad_(False)
    model.train()
    return model


def build_batch(device: torch.device, rank: int):
    generator = torch.Generator(device=device).manual_seed(100 + rank)
    batch_size, seq_len, hidden_size = 1, 16, 32
    loss_mask = torch.zeros(
        batch_size, seq_len, dtype=torch.long, device=device
    )
    if rank == 0:
        loss_mask[0, [0, 2, 5, 7, 13]] = 1
    else:
        # Steps 1 and 2 have no valid rows on this rank. The compact path must
        # still execute a zero-masked dummy LM-head row without diverging in FSDP.
        loss_mask[0, 0] = 1
    return {
        "input_ids": torch.randint(
            0,
            97,
            (batch_size, seq_len),
            generator=generator,
            device=device,
        ),
        "attention_mask": torch.ones(
            batch_size, seq_len, dtype=torch.long, device=device
        ),
        "loss_mask": loss_mask,
        "target_hidden_states": torch.randn(
            batch_size,
            seq_len,
            5 * hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        "target_last_hidden_states": torch.randn(
            batch_size,
            seq_len,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
    }


def run_arm(
    *,
    initial_state: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    world_size: int,
    trim_loss_positions: bool,
):
    model = build_model(device)
    model.load_state_dict(initial_state)
    fsdp_model = FSDP(
        model,
        **_build_fsdp_kwargs(
            sharding_strategy_name="no_shard",
            precision_dtype=torch.bfloat16,
            world_size=world_size,
        ),
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in fsdp_model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = compute_eagle3_loss(
        model=fsdp_model,
        batch=batch,
        ttt_length=3,
        step_loss_decay=0.8,
        trim_loss_positions=trim_loss_positions,
    )
    loss.backward()
    optimizer.step()
    dist.barrier()

    with FSDP.summon_full_params(fsdp_model, recurse=True, writeback=False):
        parameters = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in fsdp_model.module.named_parameters()
            if parameter.requires_grad
        }
        for parameter in fsdp_model.module.parameters():
            if not parameter.requires_grad:
                continue
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(
                parameter,
                reference,
                rtol=5e-3,
                atol=5e-3,
            )

    loss_value = float(loss.detach().float().item())
    del optimizer, fsdp_model, model
    torch.cuda.empty_cache()
    dist.barrier()
    return loss_value, parameters


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    loss_module.add_metric = lambda *args, **kwargs: None

    torch.manual_seed(0)
    template = build_model(device)
    with torch.no_grad():
        template.lm_head.weight.mul_(16.0)
    initial_state = copy.deepcopy(template.state_dict())
    del template
    batch = build_batch(device, rank)

    full_loss, full_parameters = run_arm(
        initial_state=initial_state,
        batch=batch,
        device=device,
        world_size=world_size,
        trim_loss_positions=False,
    )
    trim_loss, trim_parameters = run_arm(
        initial_state=initial_state,
        batch=batch,
        device=device,
        world_size=world_size,
        trim_loss_positions=True,
    )

    loss_tolerance = 5e-3 * max(abs(full_loss), abs(trim_loss)) + 1e-4
    assert abs(full_loss - trim_loss) <= loss_tolerance
    assert full_parameters.keys() == trim_parameters.keys()
    max_parameter_diff = 0.0
    for name in full_parameters:
        max_parameter_diff = max(
            max_parameter_diff,
            float((full_parameters[name] - trim_parameters[name]).abs().max().item()),
        )
        torch.testing.assert_close(
            full_parameters[name],
            trim_parameters[name],
            rtol=5e-3,
            atol=5e-3,
            msg=lambda message, name=name: f"parameter {name}: {message}",
        )

    if rank == 0:
        print(
            "FSDP_EQUIV_OK "
            f"max_parameter_diff={max_parameter_diff:.8g} "
            "rank_consistent=true",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
