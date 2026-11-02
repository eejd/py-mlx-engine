"""Build a JSON schema logits processor for the current Outlines API."""

import json

from outlines.backends.outlines_core import (
    OutlinesCoreBackend,
    OutlinesCoreLogitsProcessor,
)
from outlines_core import Index
from outlines_core.json_schema import build_regex_from_schema


def json_logits_processor(schema, tokenizer):
    vocabulary = OutlinesCoreBackend.create_outlines_core_vocabulary(
        tokenizer.get_vocab(),
        tokenizer.eos_token_id,
        tokenizer.eos_token,
        lambda token: tokenizer.convert_tokens_to_string([token]),
    )
    if not isinstance(schema, str):
        schema = json.dumps(schema)
    return OutlinesCoreLogitsProcessor(
        Index(build_regex_from_schema(schema), vocabulary), "mlx"
    )
