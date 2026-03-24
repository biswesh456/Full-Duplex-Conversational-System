import numpy as np
import uuid

from training.step1.preprocessing.schema import PackedSample, RawSample
from tokenization.multimodal_tokenizer import MultimodalTokenizer


IGNORE_INDEX = -100


class SequenceBuilder:
    def __init__(self, mm_tokenizer: MultimodalTokenizer, max_length: int) -> None:
        self.mm_tokenizer = mm_tokenizer
        self.max_length = max_length
        self.assistant_prefix_ids = self._assistant_prefix_ids()

    def _speech_segment(self, speech_codes: np.ndarray) -> list[int]:
        return self.mm_tokenizer.speech_ids(speech_codes)

    def _require_text(self, value: str | None, field_name: str, task: str) -> str:
        if value is None:
            raise ValueError(f"{task} sample missing required text field: {field_name}")
        return value

    def _require_speech(self, value: np.ndarray | None, field_name: str, task: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{task} sample missing required speech field: {field_name}")
        return value

    def _require_tokens(self, value: np.ndarray | None, field_name: str, task: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{task} sample missing required token field: {field_name}")
        return value

    def _chat_prefix_ids(self, system_text: str, user_text: str) -> list[int]:
        hf_tok = self.mm_tokenizer.hf_tokenizer
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return list(
            hf_tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

    def _chat_user_content_wrapper_ids(
        self,
        system_text: str,
    ) -> tuple[list[int], list[int]]:
        """
        Returns:
        before_user_content_ids, after_user_content_ids

        This lets us insert speech tokens inside the user turn, before <|im_end|>.
        """
        hf_tok = self.mm_tokenizer.hf_tokenizer

        placeholder = f"__USER_CONTENT_PLACEHOLDER_{uuid.uuid4().hex}__"

        rendered = hf_tok.apply_chat_template(
            [
                {"role": "system", "content": system_text},
                {"role": "user", "content": placeholder},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        if placeholder not in rendered:
            raise ValueError("Could not find placeholder in rendered chat template")

        before_text, after_text = rendered.split(placeholder, 1)

        before_ids = hf_tok.encode(before_text, add_special_tokens=False)
        after_ids = hf_tok.encode(after_text, add_special_tokens=False)

        return before_ids, after_ids

    def _assistant_prefix_ids(self) -> list[int]:
        hf_tok = self.mm_tokenizer.hf_tokenizer
        messages = [
            {"role": "system", "content": "dummy"},
            {"role": "user", "content": "dummy"}
        ]

        ids_without_assistant_header = list(
            hf_tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )

        ids_with_assistant_header = list(
            hf_tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        if ids_with_assistant_header[: len(ids_without_assistant_header)] != ids_without_assistant_header:
            print(ids_without_assistant_header, '\n', ids_with_assistant_header)
            raise ValueError("Chat template prefix mismatch while extracting assistant prefix")

        return ids_with_assistant_header[len(ids_without_assistant_header):]

    def _truncate_parts(
        self,
        chat_prefix_ids: list[int],
        prompt_suffix_ids: list[int],
        target_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Keep the chat prefix intact.
        Then fit as much of prompt_suffix_ids as possible.
        Then all of the target_ids.
        """
        prefix_len = len(chat_prefix_ids)
        target_len = len(target_ids)

        if prefix_len >= self.max_length:
            raise ValueError(
                f"Chat prefix length {prefix_len} already exceeds max_length={self.max_length}"
            )

        if prefix_len + target_len > self.max_length:
            raise ValueError(
                f"Chat prefix + target length ({prefix_len + target_len}) exceeds max_length={self.max_length}"
            )

        remaining = self.max_length - prefix_len - target_len

        if remaining > 0:
            kept_prompt_suffix = prompt_suffix_ids[-remaining:]
        else:
            kept_prompt_suffix = []

        input_ids = chat_prefix_ids + kept_prompt_suffix + target_ids
        labels = [IGNORE_INDEX] * (len(chat_prefix_ids) + len(kept_prompt_suffix)) + target_ids

        return input_ids, labels

    def build(self, sample: RawSample) -> PackedSample:
        eos = [self.mm_tokenizer.eos_token_id]

        if sample.task == "asr":
            input_speech = self._require_speech(sample.input_speech, "input_speech", sample.task)
            target_text = self._require_text(sample.target_text, "target_text", sample.task)

            system_text = "<text> Transcribe the speech into text."
            user_before_ids, user_after_ids = self._chat_user_content_wrapper_ids(system_text=system_text)
            chat_prefix_ids = user_before_ids + self.mm_tokenizer.text_ids("<speech> ")

            prompt_ids_suffix = self._speech_segment(input_speech)
            prompt_ids_suffix += user_after_ids
            prompt_ids_suffix += self.assistant_prefix_ids
            prompt_ids = chat_prefix_ids + prompt_ids_suffix

            target_ids = self.mm_tokenizer.text_ids("<text> ") + self.mm_tokenizer.text_ids(target_text) + eos

        elif sample.task == "tts":
            input_text = self._require_text(sample.input_text, "input_text", sample.task)
            target_speech = self._require_speech(sample.target_speech, "target_speech", sample.task)

            system_text = "<text> Synthesize speech for the given text."
            user_text = f"<text> {input_text}"
            chat_prefix_ids = self._chat_prefix_ids(system_text=system_text, user_text=user_text)
            prompt_ids_suffix = self.assistant_prefix_ids
            prompt_ids = chat_prefix_ids + prompt_ids_suffix

            target_ids = self.mm_tokenizer.text_ids("<speech> ") + self._speech_segment(target_speech) + eos

        elif sample.task == "speech_to_text_translation":
            input_speech = self._require_speech(sample.input_speech, "input_speech", sample.task)
            target_text = self._require_text(sample.target_text, "target_text", sample.task)

            system_text = "<text> Translate the speech from English to German in text."
            user_before_ids, user_after_ids = self._chat_user_content_wrapper_ids(system_text=system_text)

            chat_prefix_ids = user_before_ids + self.mm_tokenizer.text_ids("<speech> ") 
            prompt_ids_suffix = self._speech_segment(input_speech)
            prompt_ids_suffix += user_after_ids
            prompt_ids_suffix += self.assistant_prefix_ids
            prompt_ids = chat_prefix_ids + prompt_ids_suffix

            target_ids = self.mm_tokenizer.text_ids("<text> ") + self.mm_tokenizer.text_ids(target_text) + eos

        elif sample.task == "spoken_extract_qa":
            input_speech = self._require_speech(sample.input_speech, "input_speech", sample.task)
            question_text = self._require_text(sample.question_text, "question_text", sample.task)
            target_text = self._require_text(sample.target_text, "target_text", sample.task)

            system_text = "<text> Answer the question using the spoken passage."
            user_before_ids, user_after_ids = self._chat_user_content_wrapper_ids(system_text=system_text)

            chat_prefix_ids = user_before_ids + self.mm_tokenizer.text_ids("<speech> ")
            prompt_ids_suffix = self._speech_segment(input_speech)
            prompt_ids_suffix += self.mm_tokenizer.text_ids(f"<text> {question_text}")
            prompt_ids_suffix += user_after_ids
            prompt_ids_suffix += self.assistant_prefix_ids
            prompt_ids = chat_prefix_ids + prompt_ids_suffix

            target_ids = self.mm_tokenizer.text_ids("<text> ") + self.mm_tokenizer.text_ids(target_text) + eos

        elif sample.task == "text_dialog_sft":
            input_text_token = self._require_tokens(sample.input_text_token, "input_text_token", sample.task)
            target_text_token = self._require_tokens(sample.target_text_token, "target_text_token", sample.task)

            chat_prefix_ids = []
            prompt_ids = input_text_token.tolist()
            prompt_ids_suffix = prompt_ids
            target_ids = target_text_token.tolist() + eos

        else:
            raise ValueError(f"Unsupported task: {sample.task}")

        full_input_ids = prompt_ids + target_ids
        full_labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        if len(full_input_ids) > self.max_length:
            full_input_ids, full_labels = self._truncate_parts(chat_prefix_ids, prompt_ids_suffix, target_ids)

        if not any(x != IGNORE_INDEX for x in full_labels):
            raise ValueError(f"No target tokens remain after truncation for sample {sample.sample_id}")

        attention_mask = [1] * len(full_input_ids)

        meta = dict(sample.meta or {})
        meta.update(
            {
                "packed_num_tokens": len(full_input_ids),
                "packed_num_target_tokens": sum(1 for x in full_labels if x != IGNORE_INDEX),
            }
        )

        return PackedSample(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            split=sample.split,
            task=sample.task,
            input_ids=np.asarray(full_input_ids, dtype=np.int32),
            labels=np.asarray(full_labels, dtype=np.int32),
            attention_mask=np.asarray(attention_mask, dtype=np.int8),
            meta=meta,
        )