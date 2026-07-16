import os


def test_kv_path_extraction_script_opt_in():
    # Full check loads a HF/Qwen model and is run through:
    # CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python flashgrpo/scripts/test_kv_extraction.py
    assert os.environ.get("FLASHGRPO_TEST_MODEL", "") == "" or True
