# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

from torchtune.data._messages import OpenAIToMessages, ShareGPTToMessages, HaiGPTToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer


def chat_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_column: str,
    conversation_style: str,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Configure a custom dataset with conversations between user and model assistant.

    This builder function can be used to configure a custom chat dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset is expected to contain a single column with the conversations:

    .. code-block:: text

        |  conversations                         |
        |----------------------------------------|
        | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |

    This will be converted to:

    .. code-block:: python

        messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

    This list of messages is then tokenized for model training.

    You may have a different structure for your conversations, such as different role names or
    different keys in the json structure. You can use the ``conversation_style`` parameter
    to choose from standard formats such as "sharegpt" (see :class:`~torchtune.data.ShareGPTToMessages`)
    or "openai" (see :class:`~torchtune.data.OpenAIToMessages`). If your dataset is not in one of these
    formats, we recommend creating a custom message transform and using it in a custom dataset
    builder function similar to :class:`~torchtune.datasets.chat_dataset`.

    If your column names are different, use the ``conversation_column`` parameter to point
    towards the column with the conversations.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default.

    - If ``train_on_input`` is True, the prompt is used during training and
      contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        conversation_column (str): name of column containing the conversations.
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the :class:`~torchtune.data.Message` structure.
            Supported styles are: "sharegpt", "openai"
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:

    ::

        my_dataset.json
        [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "What time is it in London?",
                    },
                    {
                        "from": "gpt",
                        "value": "It is 10:00 AM in London.",
                    },
                ],
            },
            {
                "conversations": [
                    ...
                ],
            },
            ...,
        ]

    ::

        >>> from torchtune.datasets import chat_dataset
        >>> dataset = chat_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     data_files="my_dataset.json",
        ...     conversation_column="conversations",
        ...     conversation_style="sharegpt",
        ...     train_on_input=False,
        ...     packed=False,
        ...     split="train",
        ... )
        >>> tokens = dataset[0]["tokens"]
        >>> tokenizer.decode(tokens)
        "What time is it in London?It is 10:00 AM in London."

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.chat_dataset
          source: json
          data_files: my_dataset.json
          conversation_column: conversations
          conversation_style: sharegpt
          train_on_input: False
          packed: False
          split: train

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
            new_system_prompt=new_system_prompt,
        )
    elif conversation_style == "openai":
        message_transform = OpenAIToMessages(
            train_on_input=train_on_input,
            column_map={"messages": conversation_column},
            new_system_prompt=new_system_prompt,
        )
    elif conversation_style == "haigpt":
        message_transform = HaiGPTToMessages(
            train_on_input=train_on_input,
            column_map={"dialogue": conversation_column},
            new_system_prompt=new_system_prompt,
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        split=split,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds

    
if __name__ == "__main__":
    from torchtune.models.mistral import mistral_hf_tokenizer

    mistral_path = "/cephfs/SHARE/project/huggingface.co/hub/models--mistralai--Mistral-Nemo-Base-2407/snapshots/d613c787305d2300f41ad94abaec338411fbbecf"
    tokenizer = mistral_hf_tokenizer(mistral_path)

    dataset = chat_dataset(
        tokenizer=tokenizer,
        source="json",
        conversation_column="dialogue",
        conversation_style="haigpt",
        train_on_input=False,
        packed=False,
        split="train",
        data_files="/cephfs/GPT/usr/pangwenjie/her/haigpt/idea/data/ugc_model_dataset_0619.jsonl",
    )

    print(f"Total number of samples: {len(dataset)}")

    for k, v in dataset[0].items():
        print(f"{k} {len(v)}: {v}")

    print(tokenizer.decode(dataset[0]["tokens"]))


    # Calculate the avg number of tokens in the dataset "tokens" field
    total_tokens = []
    for i in range(5000):
        total_tokens.append(len(dataset[i]["tokens"]))
    avg_tokens = sum(total_tokens) / len(total_tokens)
    print(f"Average number of tokens in the dataset: {avg_tokens}")
    print(f"Top 10 max tokens in the dataset: {sorted(total_tokens, reverse=True)[:10]}")
