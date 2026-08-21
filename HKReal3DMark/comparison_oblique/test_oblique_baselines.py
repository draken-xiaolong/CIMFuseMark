import numpy as np

from run_oblique_baselines import nan25_feature, qim_embed, qim_extract


def sample():
    return np.random.default_rng(7).normal(size=(2048, 3)).astype(np.float32)


def test_qiu_clean_roundtrip_has_high_nc():
    watermark=np.random.default_rng(2026).integers(0,2,64,dtype=np.uint8)
    marked=qim_embed(sample(),watermark,"Qiu19-RI",.002)
    recovered,observed=qim_extract(marked,64,"Qiu19-RI",.002)
    assert observed.mean()>.8
    assert np.mean(recovered[observed]==watermark[observed])>.98


def test_nan25_code_is_deterministic_and_binary():
    first=nan25_feature(sample(),64,2026);second=nan25_feature(sample(),64,2026)
    assert np.array_equal(first,second)
    assert set(first.tolist()) <= {0,1}
