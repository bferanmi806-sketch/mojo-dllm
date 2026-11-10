"""Checks on a token table read from a GGUF file, shared by the byte-level BPE
tokenizer (`bpe.mojo`) and the gemma4 vocabulary (`gemma4.mojo`).

llama.cpp refuses the same files: a token type array shorter than the token
array throws (src/llama-vocab.cpp:2848 at 5f490d7), and a repeated token text
trips `GGML_ASSERT(id_to_token.size() == token_to_id.size())` (line 2899).
"""

from std.collections import Dict

from mojo_dllm.formats.gguf import GGUFFile


def validate_token_table(
    tokens: List[String], n_types: Int
) raises -> Dict[String, Int]:
    """The id of each token text. The type array must have one entry per token
    and no text may appear twice (an empty text is exempt: llama.cpp renames it,
    and no prompt can hold one)."""
    if n_types != len(tokens):
        raise Error(
            "tokenizer.ggml.token_type has "
            + String(n_types)
            + " entries but tokenizer.ggml.tokens has "
            + String(len(tokens))
        )
    var ids = Dict[String, Int]()
    for i in range(len(tokens)):
        if tokens[i].byte_length() == 0:
            continue
        if tokens[i] in ids:
            raise Error(
                "tokenizer.ggml.tokens has the text "
                + repr(tokens[i])
                + " at ids "
                + String(ids[tokens[i]])
                + " and "
                + String(i)
            )
        ids[tokens[i]] = i
    return ids^


def special_token_id(g: GGUFFile, key: String, n_tokens: Int) raises -> Int:
    """The token id stored under `key`, or -1 when the key is absent. An id
    outside the vocabulary is an error."""
    var id = g.get_int_or(key, -1)
    if id < -1:
        raise Error(
            "metadata key "
            + key
            + " is "
            + String(id)
            + "; a special token id is a token id or -1"
        )
    if id >= n_tokens:
        raise Error(
            "metadata key "
            + key
            + " is "
            + String(id)
            + ", outside the vocabulary of "
            + String(n_tokens)
            + " tokens"
        )
    return id
