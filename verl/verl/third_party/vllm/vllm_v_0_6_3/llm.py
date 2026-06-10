# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer
from vllm import LLM
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.utils import Counter

from .arg_utils import EngineArgs
from .llm_engine_sp import LLMEngine


def _patch_vllm_custom_ops():
    import vllm._custom_ops as ops

    def _rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                  epsilon: float) -> None:
        x = input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        out.copy_((x * weight).to(input.dtype))

    def _fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                            weight: torch.Tensor, epsilon: float) -> None:
        x = input.to(torch.float32) + residual.to(torch.float32)
        residual.copy_(x.to(input.dtype))
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
        input.copy_((x * weight).to(input.dtype))

    def _silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        out.copy_(torch.nn.functional.silu(gate) * up)

    def _gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        out.copy_(torch.nn.functional.gelu(gate, approximate='none') * up)

    def _gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        out.copy_(torch.nn.functional.gelu(gate, approximate='tanh') * up)

    def _gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
        out.copy_(torch.nn.functional.gelu(x, approximate='none'))

    def _gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
        out.copy_(torch.nn.functional.gelu(x, approximate='tanh'))

    def _gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
        out.copy_(x * torch.sigmoid(1.702 * x))

    def _rotary_embedding(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None:
        rotary_dim = cos_sin_cache.shape[1]
        num_tokens = query.shape[0]
        query_3d = query.view(num_tokens, -1, head_size)
        key_3d = key.view(num_tokens, -1, head_size)

        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2).to(query.dtype)
        sin = sin.unsqueeze(-2).to(query.dtype)

        q_rot = query_3d[..., :rotary_dim]
        q_pass = query_3d[..., rotary_dim:]
        k_rot = key_3d[..., :rotary_dim]
        k_pass = key_3d[..., rotary_dim:]

        if is_neox:
            q1, q2 = q_rot.chunk(2, dim=-1)
            k1, k2 = k_rot.chunk(2, dim=-1)
        else:
            q1 = q_rot[..., ::2]
            q2 = q_rot[..., 1::2]
            k1 = k_rot[..., ::2]
            k2 = k_rot[..., 1::2]

        q1_new = q1 * cos - q2 * sin
        q2_new = q2 * cos + q1 * sin
        k1_new = k1 * cos - k2 * sin
        k2_new = k2 * cos + k1 * sin

        if is_neox:
            q_rot_new = torch.cat((q1_new, q2_new), dim=-1)
            k_rot_new = torch.cat((k1_new, k2_new), dim=-1)
        else:
            q_rot_new = torch.stack((q1_new, q2_new), dim=-1).flatten(-2)
            k_rot_new = torch.stack((k1_new, k2_new), dim=-1).flatten(-2)

        if rotary_dim < head_size:
            query_3d.copy_(torch.cat((q_rot_new, q_pass), dim=-1))
            key_3d.copy_(torch.cat((k_rot_new, k_pass), dim=-1))
        else:
            query_3d.copy_(q_rot_new)
            key_3d.copy_(k_rot_new)

    def _batched_rotary_embedding(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        rot_dim: int,
        cos_sin_cache_offsets: torch.Tensor,
    ) -> None:
        num_tokens = query.shape[0]
        query_3d = query.view(num_tokens, -1, head_size)
        key_3d = key.view(num_tokens, -1, head_size)

        cos_sin = cos_sin_cache[cos_sin_cache_offsets]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2).to(query.dtype)
        sin = sin.unsqueeze(-2).to(query.dtype)

        q_rot = query_3d[..., :rot_dim]
        q_pass = query_3d[..., rot_dim:]
        k_rot = key_3d[..., :rot_dim]
        k_pass = key_3d[..., rot_dim:]

        if is_neox:
            q1, q2 = q_rot.chunk(2, dim=-1)
            k1, k2 = k_rot.chunk(2, dim=-1)
        else:
            q1 = q_rot[..., ::2]
            q2 = q_rot[..., 1::2]
            k1 = k_rot[..., ::2]
            k2 = k_rot[..., 1::2]

        q1_new = q1 * cos - q2 * sin
        q2_new = q2 * cos + q1 * sin
        k1_new = k1 * cos - k2 * sin
        k2_new = k2 * cos + k1 * sin

        if is_neox:
            q_rot_new = torch.cat((q1_new, q2_new), dim=-1)
            k_rot_new = torch.cat((k1_new, k2_new), dim=-1)
        else:
            q_rot_new = torch.stack((q1_new, q2_new), dim=-1).flatten(-2)
            k_rot_new = torch.stack((k1_new, k2_new), dim=-1).flatten(-2)

        if rot_dim < head_size:
            query_3d.copy_(torch.cat((q_rot_new, q_pass), dim=-1))
            key_3d.copy_(torch.cat((k_rot_new, k_pass), dim=-1))
        else:
            query_3d.copy_(q_rot_new)
            key_3d.copy_(k_rot_new)

    def _reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[-1]
        x = key_cache.shape[-1]
        head_size_x = key_cache.shape[2]

        key_reshaped = key.view(-1, num_kv_heads, head_size_x, x)

        safe_slots = slot_mapping.clamp(min=0).to(torch.int64)
        block_indices = safe_slots // block_size
        offsets = safe_slots % block_size

        key_cache[block_indices, :, :, offsets, :] = key_reshaped
        value_cache[block_indices, :, :, offsets] = value

    def _reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        num_kv_heads = key.shape[1]
        head_size = key.shape[2]
        block_size = key_cache.shape[1] // (num_kv_heads * head_size)

        key_cache_3d = key_cache.view(key_cache.shape[0], block_size,
                                       num_kv_heads, head_size)
        key_cache_flat = key_cache_3d.reshape(-1, num_kv_heads, head_size)
        value_cache_3d = value_cache.view(value_cache.shape[0], block_size,
                                           num_kv_heads, head_size)
        value_cache_flat = value_cache_3d.reshape(-1, num_kv_heads, head_size)

        safe_slots = slot_mapping.clamp(min=0).to(torch.int64)
        key_cache_flat[safe_slots] = key
        value_cache_flat[safe_slots] = value

    def _copy_blocks(
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
        block_mapping: torch.Tensor,
    ) -> None:
        for key_cache, value_cache in zip(key_caches, value_caches):
            for mapping in block_mapping:
                src_idx = mapping[0].item()
                dst_idx = mapping[1].item()
                key_cache[dst_idx].copy_(key_cache[src_idx])
                value_cache[dst_idx].copy_(value_cache[src_idx])

    def _swap_blocks(
        src: torch.Tensor,
        dst: torch.Tensor,
        block_mapping: torch.Tensor,
    ) -> None:
        for mapping in block_mapping:
            src_idx = mapping[0].item()
            dst_idx = mapping[1].item()
            dst[dst_idx].copy_(src[src_idx])

    def _convert_fp8(
        output: torch.Tensor,
        input: torch.Tensor,
        scale: float = 1.0,
        kv_dtype: str = "fp8",
    ) -> None:
        output.copy_(input)

    def _paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        num_seqs = query.shape[0]
        num_heads = query.shape[1]
        head_size = query.shape[2]
        num_queries_per_kv = num_heads // num_kv_heads

        x = key_cache.shape[-1]
        key_cache_std = key_cache.permute(0, 1, 3, 2, 4).reshape(
            key_cache.shape[0], num_kv_heads, block_size, head_size)
        value_cache_std = value_cache.permute(0, 1, 3, 2)

        max_num_blocks = block_tables.shape[1]
        block_table_flat = block_tables.clamp(min=0)

        k_all = key_cache_std[block_table_flat]
        v_all = value_cache_std[block_table_flat]

        k_all = k_all.permute(0, 2, 1, 3, 4).reshape(
            num_seqs, max_num_blocks * block_size, num_kv_heads, head_size)
        v_all = v_all.permute(0, 2, 1, 3, 4).reshape(
            num_seqs, max_num_blocks * block_size, num_kv_heads, head_size)

        if num_queries_per_kv > 1:
            k_all = k_all.unsqueeze(3).expand(
                -1, -1, num_kv_heads, num_queries_per_kv, -1).reshape(
                    num_seqs, max_num_blocks * block_size, num_heads, head_size)
            v_all = v_all.unsqueeze(3).expand(
                -1, -1, num_kv_heads, num_queries_per_kv, -1).reshape(
                    num_seqs, max_num_blocks * block_size, num_heads, head_size)

        seq_lens_clamped = seq_lens.clamp(min=1)
        positions = torch.arange(max_num_blocks * block_size, device=query.device, dtype=torch.long)
        position_mask = positions.unsqueeze(0) < seq_lens_clamped.unsqueeze(1)

        q_expanded = query.unsqueeze(2)
        k_transposed = k_all.permute(0, 2, 3, 1)
        attn_weights = torch.matmul(q_expanded, k_transposed) * scale

        neg_inf = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(~position_mask.unsqueeze(1).unsqueeze(2), neg_inf)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(~position_mask.unsqueeze(1).unsqueeze(2), 0.0)

        v_transposed = v_all.permute(0, 2, 1, 3)
        attn_out = torch.matmul(attn_weights, v_transposed)
        out.copy_(attn_out.squeeze(2))

    def _paged_attention_v2(
        out: torch.Tensor,
        exp_sum: torch.Tensor,
        max_logits: torch.Tensor,
        tmp_out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
        alibi_slopes: Optional[torch.Tensor],
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> None:
        _paged_attention_v1(
            out, query, key_cache, value_cache, num_kv_heads, scale,
            block_tables, seq_lens, block_size, max_seq_len, alibi_slopes,
            kv_cache_dtype, k_scale, v_scale, tp_rank,
            blocksparse_local_blocks, blocksparse_vert_stride,
            blocksparse_block_size, blocksparse_head_sliding_step)

    ops.rms_norm = _rms_norm
    ops.fused_add_rms_norm = _fused_add_rms_norm
    ops.silu_and_mul = _silu_and_mul
    ops.gelu_and_mul = _gelu_and_mul
    ops.gelu_tanh_and_mul = _gelu_tanh_and_mul
    ops.gelu_new = _gelu_new
    ops.gelu_fast = _gelu_fast
    ops.gelu_quick = _gelu_quick
    ops.rotary_embedding = _rotary_embedding
    ops.batched_rotary_embedding = _batched_rotary_embedding
    ops.reshape_and_cache = _reshape_and_cache
    ops.reshape_and_cache_flash = _reshape_and_cache_flash
    ops.copy_blocks = _copy_blocks
    ops.swap_blocks = _swap_blocks
    ops.convert_fp8 = _convert_fp8
    ops.paged_attention_v1 = _paged_attention_v1
    ops.paged_attention_v2 = _paged_attention_v2


_patch_vllm_custom_ops()


class LLM(LLM):
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: A HuggingFace Transformers model instance.
        tokenizer: A HuggingFace Transformers tokenizer instance.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],  # model itself or its parameter dict
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_format="auto",
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = ("image_token_id", "image_feature_size", "image_input_shape", "image_input_type")
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError("There is no need to pass vision-related arguments anymore.")
        engine_args = EngineArgs(
            model_hf_config=model_hf_config,
            # tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            load_format=load_format,
            **kwargs,
        )
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be"
                "one of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            )
        self.llm_engine = LLMEngine.from_engine_args(model, tokenizer, engine_args)  # TODO: check usagecontext
        self.request_counter = Counter()

    def init_cache_engine(self):
        self.llm_engine.init_cache_engine()

    def free_cache_engine(self):
        self.llm_engine.free_cache_engine()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        outputs = super()._run_engine(use_tqdm=use_tqdm)
        return self._post_process_outputs(outputs)

    # # NOTE(shengguangming): add for verl
    # # TODO(sgm): we can optimize it by making the dataloader yield List[int] without padding.
    # def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
    #     # remove the left padding in the prompt token_id
    #     pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    #     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    #     token_ids = prompt_token_ids[non_pad_index:].tolist()
    #     return token_ids

    # NOTE(shengguangming): add for verl
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                # TODO(shengguangming): can be optimzied by rewrite the Sampler._get_logprobs() logits
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[id].logprob)
                    logprobs.append(torch.tensor(logprob))

        pad_token_id = (self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None
                        else self.llm_engine.tokenizer.eos_token_id)
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        return output_token_ids, logprobs

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.llm_engine.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.llm_engine.offload_model_weights()
