# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.models.mistral._component_builders import (
    mistral,
    lora_mistral,
    mistral_classifier,
    lora_mistral_classifier,
)
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._prompt_templates import _get_prompt_template

from torchtune.modules import TransformerDecoder
from torchtune.models.mistral._tokenizer import MistralTokenizer, MistralHFTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial


"""
Model builders build specific instantiations using component builders. For example
the ``mistral_7b`` model builder uses the ``mistral`` component builder.
"""


def mistral_7b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b parameter values
    from https://mistral.ai/news/announcing-mistral-7b/


    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model
    """
    return mistral(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def mistral_nemo_12b() -> TransformerDecoder:
    """
    Builder for creating a Mistral Nemo 12B model initialized w/ the default 12b parameter values
    from https://huggingface.co/mistralai/Mistral-Nemo-Base-2407


    Returns:
        TransformerDecoder: Instantiation of Mistral Nemo 12B model
    """
    return mistral(
        vocab_size=131072,
        num_layers=40,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=14336,
        max_seq_len=1024000,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1_000_000,
        # Output hidden states for debug purpose
        # output_hidden_states=[0, 1, 2],
    )


def mistral_8b() -> TransformerDecoder:
    """
    Builder for creating a Ministral 8B model initialized w/ the default 8b parameter values
    from https://huggingface.co/mistralai/Ministral-8B-Instruct-2410


    Returns:
        TransformerDecoder: Instantiation of Ministral 8B model
    """
    return mistral(
        vocab_size=131072,
        num_layers=36,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000_000,
        # Output hidden states for debug purpose
        # output_hidden_states=[0, 1, 2],
    )


def mistral_tokenizer(path: str, max_seq_len: Optional[int] = None, prompt_template: Optional[_TemplateType] = "torchtune.models.mistral.MistralChatTemplate") -> MistralTokenizer:
    """
    Tokenizer for Mistral models.

    Args:
        path (str): path to the tokenizer
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags. Default is :class:`~torchtune.models.mistral.MistralChatTemplate`.

    Returns:
        MistralTokenizer: Instantiation of the Mistral tokenizer
    """
    return MistralTokenizer(path=path, max_seq_len=max_seq_len, prompt_template=_get_prompt_template(prompt_template) if prompt_template is not None else None)


def mistral_hf_tokenizer(path: str, max_seq_len: Optional[int] = None, prompt_template: Optional[_TemplateType] = None) -> MistralTokenizer:
    """
    Tokenizer for Mistral models.

    Args:
        path (str): path to the tokenizer
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags. Default is :class:`~torchtune.models.mistral.MistralChatTemplate`.

    Returns:
        MistralTokenizer: Instantiation of the Mistral tokenizer
    """
    return MistralHFTokenizer(path=path, max_seq_len=max_seq_len, prompt_template=_get_prompt_template(prompt_template) if prompt_template is not None else None)


def lora_mistral_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model with LoRA enabled.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model with LoRA applied
    """
    return lora_mistral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

    
def lora_mistral_nemo_12b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Mistral Nemo 12b model with LoRA enabled.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral Nemo 12B model with LoRA applied
    """
    return lora_mistral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=131072,
        num_layers=40,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=14336,
        max_seq_len=1024000,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_mistral_7b = partial(lora_mistral_7b, quantize_base=True)

qlora_mistral_7b.__doc__ = """
Builder for creating a Mistral model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_mistral_7b` for full API arguments.
"""


def mistral_reward_7b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b
    parameter values from:
    https://huggingface.co/Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback
    where the output layer is a classification layer projecting to a single class for reward modelling.

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B classifier model
    """
    return mistral_classifier(
        num_classes=1,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_mistral_reward_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Mistral reward 7B model with LoRA enabled.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model with LoRA applied
    """
    return lora_mistral_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_mistral_reward_7b = partial(lora_mistral_reward_7b, quantize_base=True)

qlora_mistral_reward_7b.__doc__ = """
Builder for creating a Mistral reward 7B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_mistral_reward_7b` for full API arguments.
"""

if __name__ == "__main__":
    import torch
    from torchtune.training.checkpointing import FullModelHFCheckpointer
    checkpoint_dir = "/cephfs/GPT/usr/pangwenjie/her/haigpt/idea_tune/torchtune/recipes/output/mistral_nemo_12b_ugc_char_sft_low_mem_1025/hf2"
    checkpoint_files = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]
    print("Load checkpoint...")
    checkpoint = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=checkpoint_files,
        model_type="MISTRAL",
        output_dir="./test")
    checkpoint_dict = checkpoint.load_checkpoint()

    model = mistral_nemo_12b()
    for name, param in model.named_parameters():
        # print(name, param.shape)
        if name not in checkpoint_dict["model"]:
            print("Param: {name} shape: {param.shape} not in checkpoint")
    model.load_state_dict(checkpoint_dict["model"])
    model.to(torch.bfloat16)
    model = torch.nn.DataParallel(model)
    model.to("cuda")
    dtypes = set(param.dtype for param in model.parameters())
    print("Load checkpoint done. Model dtype", model.dtype, dtypes)

    input_tokens = torch.tensor([[1, 3, 3263, 4, 1267, 7801, 1584, 1636, 1063, 2, 3, 1503, 19464, 4, 1267]])
    output = model(input_tokens)
    if isinstance(output, list):
        hidden = output[:-1]
        torch.save(hidden, "./test/hidden.pt")
        print("Hidden size", [h.shape for h in hidden])
        output = output[-1]
    print("Output size", output.shape)
    torch.save(output, "./test/output.pt")
