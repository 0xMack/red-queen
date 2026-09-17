from tinylm import CharTokenizer


def test_encode_decode_roundtrip():
    tokenizer = CharTokenizer("hello world")

    ids = tokenizer.encode("hello world")
    assert tokenizer.decode(ids) == "hello world"


def test_vocab_size_matches_unique_characters():
    tokenizer = CharTokenizer("aabbbcccc")

    assert tokenizer.vocab_size == 3


def test_same_character_always_encodes_to_the_same_id():
    tokenizer = CharTokenizer("banana")

    ids = tokenizer.encode("banana")
    assert ids[1] == ids[3] == ids[5]  # every 'a'
