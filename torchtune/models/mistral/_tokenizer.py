# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, List, Mapping, Optional, Tuple

from transformers import AutoTokenizer
from torchtune.data import Message, PromptTemplate, truncate
from torchtune.models.mistral._prompt_template import MistralChatTemplate
from torchtune.modules.tokenizers import (
    ModelTokenizer,
    SentencePieceBaseTokenizer,
    tokenize_messages_no_special_tokens,
)
from torchtune.modules.transforms import Transform

WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class MistralTokenizer(ModelTokenizer, Transform):
    """
    Mistral's implementation of the SentencePiece tokenizer

    Args:
        path (str): Path to pretrained tokenizer file.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
            Default is :class:`~torchtune.models.mistral.MistralChatTemplate`.

    Examples:
        >>> tokenizer = MistralTokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = MistralChatTemplate(),
    ):
        self._spm_model = SentencePieceBaseTokenizer(path)

        # Original tokenizer has no pad_id, which causes indexing errors when batch training
        self._spm_model.pad_id = 0

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

    @property
    def eos_id(self):
        return self._spm_model.eos_id

    @property
    def bos_id(self):
        return self._spm_model.bos_id

    @property
    def pad_id(self):
        return self._spm_model.pad_id

    @property
    def vocab_size(self):
        return self._spm_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        """
        Encode a string into a list of token IDs

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS special token (Beginning of Sentence) to the input, defaults to True.
            add_eos (bool): Whether to append EOS special token (End of Sentence) to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Sentencepiece normally prepends
                whitespace to any tokenized text, which can cause differences where
                encode(s1) + encode(s2) != encode(s1 + s2) due to leading whitespace
                added to s2. Default: False
        Returns:
            List[int]: The encoded token IDs.
        """
        return self._spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            trim_leading_whitespace=trim_leading_whitespace,
        )

    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        """Decode token IDs to strings.

        Args:
            token_ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return self._spm_model.decode(token_ids)

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note:
            sentencepiece has problems where in general
            encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
            We can get around this by prepending s2 with a known token and slicing the
            beginning off the tokenized s2.

        Example:
            >>> tokenizer = MistralTokenizer(tokenizer_path, max_seq_len)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
            ]

            >>> # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


            >>> # Same result as encoding the full string in one go
            >>> tokenizer.encode(''.join([message.content for message in messages]))
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.
            add_eos (bool): Whether to append EOS after assistant message, default to True

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )
        return tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=templated_messages,
            bos_id=self.bos_id,
            eos_id=self.eos_id if add_eos else None,
        )

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
            inference (bool): Whether the template is being used for inference or not.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample


class MistralHFTokenizer(ModelTokenizer, Transform):
    """
    Mistral's implementation of the HuggingFace tokenizer

    Args:
        path (str): Path to pretrained tokenizer file.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
            Default is :class:`~torchtune.models.mistral.MistralChatTemplate`.

    Examples:
        >>> tokenizer = MistralHFTokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self._hf_model = AutoTokenizer.from_pretrained(path)

        self.special_tokens = {
            "<s>": 1,
            "</s>": 2,
            "[INST]": 3,
            "[/INST]": 4,
            "<pad>": 10,
        }

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<s>"]
        self.eos_id = self.special_tokens["</s>"]
        self.pad_id = self.special_tokens["<pad>"]

        # Encode extra special tokens
        self.start_header_id = self.special_tokens["[INST]"]
        self.end_header_id = self.special_tokens["[/INST]"]
        self.eot_id = self.special_tokens["</s>"]

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id]

        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

        # Regex for removing special tokens from the decoded string
        self._special_token_regex = re.compile(r"<\|.*?\|>")
        self._special_token_header_regex = re.compile(
            r"<\|start_header_id\|>.*?<\|end_header_id\|>\n\n"
        )

    @property
    def vocab_size(self):
        return self._hf_model.vocab_size

    def _remove_special_tokens(self, text: str) -> str:
        """
        Remove special tokens from the decoded string.
        """
        # First remove the headers, then the remaining special tokens
        return self._special_token_regex.sub(
            "", self._special_token_header_regex.sub("", text)
        )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        """
        Encode a string into a list of token IDs

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS special token (Beginning of Sentence) to the input, defaults to True.
            add_eos (bool): Whether to append EOS special token (End of Sentence) to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Sentencepiece normally prepends
                whitespace to any tokenized text, which can cause differences where
                encode(s1) + encode(s2) != encode(s1 + s2) due to leading whitespace
                added to s2. Default: False
        Returns:
            List[int]: The encoded token IDs.
        """
        if trim_leading_whitespace:
            text = text.lstrip()
        if add_bos:
            text = self._hf_model.bos_token + text
        if add_eos:
            text = text + self._hf_model.eos_token
        return self._hf_model.encode(text, add_special_tokens=False)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to strings.

        Args:
            token_ids (List[int]): The input token IDs to be decoded.
            skip_special_tokens (bool): Whether to remove special tokens from the decoded string.

        Returns:
            str: The decoded text.
        """
        # We will remove special tokens manually via regex on the decoded string.
        # This is because removing all special tokens does not remove the role and
        # whitespace added from the special tokens, i.e., the "user" and "\n\n" in
        # "<|start_header_id|>user<|end_header_id|>\n\n"
        decoded_string = self._hf_model.decode(token_ids=token_ids)
        return (
            self._remove_special_tokens(decoded_string)
            if skip_special_tokens
            else decoded_string
        )

    def _tokenize_header(self, message: Message) -> List[int]:
        """
        Tokenize header start, message role, and header end as list of ids
        """
        return (
            [self.start_header_id]
            + self.encode(message.role.strip(), add_bos=False, add_eos=False)
            + [self.end_header_id]
            + self.encode("\n\n", add_bos=False, add_eos=False)
        )

    def _tokenize_end(self, message: Message) -> List[int]:
        """
        Add eot or eom id at the end of the message.
        """
        return [self.eot_id]

    def _tokenize_body(self, message: Message) -> List[int]:
        """
        Tokenize message content as list of ids
        """
        tokenized_body = []
        for item in message.content:
            if item["type"] == "text":
                tokenized_body += self.encode(
                    item["content"].strip(), add_bos=False, add_eos=False
                )
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")

        return tokenized_body

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to prepend a tokenized header to the message. Default is True.
            add_end_tokens (bool): Whether to append eot or eom id at the end of the message. Default is True.

        Returns:
            List[int]: The list of token ids.
        """
        tokenized_header = self._tokenize_header(message) if add_start_tokens else []
        tokenized_body = self._tokenize_body(message)
        tokenized_end = self._tokenize_end(message) if add_end_tokens else []

        tokenized_message = tokenized_header + tokenized_body + tokenized_end
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_eos: bool = False,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note:
            sentencepiece has problems where in general
            encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
            We can get around this by prepending s2 with a known token and slicing the
            beginning off the tokenized s2.

        Example:
            >>> tokenizer = MistralTokenizer(tokenizer_path, max_seq_len)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
            ]

            >>> # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


            >>> # Same result as encoding the full string in one go
            >>> tokenizer.encode(''.join([message.content for message in messages]))
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.
            add_eos (bool): Whether to append EOS after assistant message, default to True

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]

        for message in templated_messages:
            tokenized_message = self.tokenize_message(message)

            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if self.max_seq_len and len(tokens) >= self.max_seq_len:
                break

        if add_eos:
            tokens = tokens + [self.eos_id]
            mask = mask + [True]

        if self.max_seq_len:
            tokens = truncate(
                tokens, self.max_seq_len, self.eos_id if add_eos else None
            )
            mask = truncate(mask, self.max_seq_len, True if add_eos else None)

        return tokens, mask

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
            inference (bool): Whether the template is being used for inference or not.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample

        
if __name__ == "__main__":
    tokenizer = MistralHFTokenizer("/cephfs/SHARE/project/huggingface.co/hub/models--mistralai--Mistral-Nemo-Base-2407/snapshots/d613c787305d2300f41ad94abaec338411fbbecf")
    tokenized_text = tokenizer.encode("[INST]user[/INST]\n\nHow are you?</s>", add_bos=True, add_eos=False)
    print(tokenized_text)

    message = Message(role="user", content="How are you?", masked=True)
    tokenized_message = tokenizer.tokenize_message(message)
    print(tokenized_message)

    
    messages = [
        Message(role="user", content="How are you?", masked=False),
    ]
    tokenized_messages = tokenizer.tokenize_messages(messages)
    print(tokenized_messages)

    messages = [
        Message(role="system", content="system message.", masked=True),
        Message(role="user", content="user prompt.", masked=True),
        Message(role="assistant", content="assistant response.", masked=False),
    ]
    tokenized_messages = tokenizer.tokenize_messages(messages)
    print(tokenized_messages)
    
