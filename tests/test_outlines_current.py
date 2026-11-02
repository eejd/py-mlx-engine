import mlx.core as mx

from mlx_engine.utils.outlines_transformer_tokenizer import json_logits_processor


class CharacterTokenizer:
    eos_token_id = 4
    eos_token = "<eos>"

    def get_vocab(self):
        return {"t": 0, "r": 1, "u": 2, "e": 3, "<eos>": 4, "x": 5}

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def test_json_schema_logits_follow_current_outlines_guide():
    processor = json_logits_processor({"type": "boolean"}, CharacterTokenizer())
    token_ids = [0, 1, 2, 3]
    for index, expected in enumerate(token_ids):
        result = processor(mx.array(token_ids[:index], dtype=mx.int32), mx.zeros((6,)))
        mx.eval(result)
        assert bool(mx.isfinite(result[expected]).item())
        assert bool(mx.isinf(result[5]).item())
