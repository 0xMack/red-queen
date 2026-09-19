import numpy as np
from tinylm import CharTokenizer, TinyLM, load, save


def test_save_load_round_trip(tmp_path):
    tokenizer = CharTokenizer("hello, world\n")
    model = TinyLM(vocab_size=tokenizer.vocab_size, max_seq_len=8, d_model=16, n_heads=2, n_layers=2, d_hidden=24, rng=np.random.default_rng(1))
    save(model, tokenizer, tmp_path / "m", steps=3)
    loaded, loaded_tokenizer, meta = load(tmp_path / "m")
    ids = np.array([tokenizer.encode("hello")])
    assert np.array_equal(model(ids).data, loaded(ids).data)
    assert loaded_tokenizer.encode("world") == tokenizer.encode("world")
    assert meta["steps"] == 3
