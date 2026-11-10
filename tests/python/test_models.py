from __future__ import annotations

import io
from pathlib import Path

import pytest

from launcher.errors import LauncherError
from launcher.models import (
    DEFAULT_MODEL,
    DEFAULT_QUANTS,
    ModelRef,
    ensure_model,
    fetch_model,
    parse_model_ref,
    pick_gguf,
    plan_model,
)

REPO = "mradermacher/LLaDA-8B-Instruct-GGUF"


def test_existing_file_is_used_as_is(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert parse_model_ref(str(f)) == ModelRef(path=f)


def test_alias_names_the_llada_repo() -> None:
    assert parse_model_ref("llada-8b") == ModelRef(repo=REPO, quant="Q4_K_M")
    assert DEFAULT_MODEL == "llada-8b"


def test_owner_repo_with_and_without_quant() -> None:
    assert parse_model_ref("a/b") == ModelRef(repo="a/b", quant=None)
    assert parse_model_ref("a/b:q8_0") == ModelRef(repo="a/b", quant="q8_0")


@pytest.mark.parametrize("text", ["nonsense", "a/b/c", "a/b:", "/", "", "a b/c"])
def test_anything_else_is_an_error_that_says_what_is_accepted(text: str) -> None:
    with pytest.raises(LauncherError) as e:
        parse_model_ref(text)
    msg = str(e.value)
    assert "a path to a .gguf file" in msg and "llada-8b" in msg and "owner/repo[:QUANT]" in msg


def test_missing_gguf_path_is_not_mistaken_for_a_repo(tmp_path: Path) -> None:
    with pytest.raises(LauncherError, match="no such file"):
        parse_model_ref(str(tmp_path / "gone.gguf"))
    with pytest.raises(LauncherError, match="no such file"):
        parse_model_ref("models/gone.gguf")


FILES = ["README.md", "m.Q8_0.gguf", "m.Q2_K.gguf", "m.Q4_K_M.gguf"]


def test_default_order_is_q4_k_m_then_q4_k_s_then_q6_k() -> None:
    assert pick_gguf(FILES, None) == "m.Q4_K_M.gguf"
    assert pick_gguf(["m.Q8_0.gguf", "m.Q6_K.gguf", "m.Q4_K_S.gguf"], None) == "m.Q4_K_S.gguf"
    assert pick_gguf(["m.Q8_0.gguf", "m.Q6_K.gguf"], None) == "m.Q6_K.gguf"
    assert DEFAULT_QUANTS == ("Q4_K_M", "Q4_K_S", "Q6_K")


def test_a_repo_with_q4_k_s_and_q8_0_picks_q4_k_s() -> None:
    assert pick_gguf(["m.Q8_0.gguf", "m.Q4_K_S.gguf"], None) == "m.Q4_K_S.gguf"


def test_no_default_quant_fails_with_the_listing_and_the_quant_syntax() -> None:
    with pytest.raises(LauncherError) as e:
        pick_gguf(["m.Q8_0.gguf", "m.F16.gguf", "README.md"], None)
    msg = str(e.value)
    assert "Q4_K_M, Q4_K_S or Q6_K" in msg
    assert "  m.Q8_0.gguf\n  m.F16.gguf" in msg and "README.md" not in msg
    assert "owner/repo:QUANT" in msg


def test_quant_matches_case_insensitively_as_a_whole_token() -> None:
    assert pick_gguf(FILES, "q2_k") == "m.Q2_K.gguf"
    # IQ4_XS must not satisfy a request for Q4_XS.
    with pytest.raises(LauncherError):
        pick_gguf(["m.IQ4_XS.gguf"], "Q4_XS")
    assert pick_gguf(["m-Q5_K_M.gguf"], "Q5_K_M") == "m-Q5_K_M.gguf"
    assert pick_gguf(["m_Q5_K_M.gguf"], "Q5_K_M") == "m_Q5_K_M.gguf"


def test_split_shards_are_ignored() -> None:
    files = ["big.Q4_K_M-00001-of-00003.gguf", "small.Q6_K.gguf"]
    assert pick_gguf(files, None) == "small.Q6_K.gguf"
    with pytest.raises(LauncherError):
        pick_gguf(files, "Q4_K_M")


def test_a_requested_quant_that_matches_nothing_lists_the_files() -> None:
    with pytest.raises(LauncherError) as e:
        pick_gguf(FILES, "Q9_9")
    msg = str(e.value)
    assert "Q9_9" in msg
    assert "Available GGUF files:\n" in msg
    assert "  m.Q8_0.gguf\n" in msg + "\n" and "README.md" not in msg


def test_repo_with_no_gguf_is_an_error() -> None:
    with pytest.raises(LauncherError, match="no .gguf"):
        pick_gguf(["README.md"], None)


def test_plan_reports_size_and_cache_state(hub) -> None:
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    assert plan.filename == "LLaDA-8B-Instruct.Q4_K_M.gguf"
    assert plan.size == 4_925_226_272 and plan.cached is None
    hub.cached[(REPO, plan.filename)] = Path("/c/x.gguf")
    assert plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub).cached == Path("/c/x.gguf")


def test_fetch_verifies_sha256_and_prints_it(hub, store) -> None:
    out = io.StringIO()
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    path = fetch_model(plan, hub, out, store)
    assert path.read_bytes() == hub.content
    text = out.getvalue()
    assert f"Downloading {REPO}/{plan.filename} (4.9 GB)" in text
    assert "sha256 verified" in text


def test_fetch_deletes_a_file_whose_hash_does_not_match(hub, store) -> None:
    hub.sha256 = "0" * 64
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    with pytest.raises(LauncherError, match="sha256"):
        fetch_model(plan, hub, io.StringIO(), store)
    assert not (hub._tmp / plan.filename).exists()


def test_fetch_refuses_when_the_hub_reports_no_hash(hub, store) -> None:
    hub.sha256 = None
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    with pytest.raises(LauncherError, match="no sha256"):
        fetch_model(plan, hub, io.StringIO(), store)
    assert not (hub._tmp / plan.filename).exists()


def test_ensure_model_cached_prints_already_downloaded(hub, store, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(b"x" * 10)
    hub.cached[(REPO, "LLaDA-8B-Instruct.Q4_K_M.gguf")] = cached
    store.record(cached, hub.sha256)
    out = io.StringIO()
    path = ensure_model("llada-8b", hub, out, announce_cached=True, store=store)
    assert path == cached and hub.downloads == [] and hub.sha_requests == 0
    assert f"already downloaded\n{cached}" in out.getvalue()


def test_ensure_model_downloads_when_missing(hub, store) -> None:
    out = io.StringIO()
    path = ensure_model("llada-8b", hub, out, announce_cached=True, store=store)
    assert hub.downloads == [(REPO, "LLaDA-8B-Instruct.Q4_K_M.gguf")]
    assert str(path) in out.getvalue() and "10 B" in out.getvalue()


def test_ensure_model_local_path_never_touches_the_hub(hub, store, tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert ensure_model(str(f), None, io.StringIO(), announce_cached=True, store=store) == f


def test_offline_miss_says_only_the_cache_was_searched(hub) -> None:
    hub.offline = True
    hub.files = {}
    with pytest.raises(LauncherError) as e:
        plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    msg = str(e.value)
    assert "offline" in msg and str(hub.cache_dir) in msg
    assert REPO in msg and "Q4_K_M" in msg and f"./mojo-dllm pull {REPO}:Q4_K_M" in msg


def test_offline_selection_uses_the_cached_listing_only(hub, store, tmp_path: Path) -> None:
    hub.offline = True
    hub.files = {"x.Q4_K_S.gguf": 5}  # all the cache holds for this repo
    cached = tmp_path / "x.gguf"
    cached.write_bytes(b"12345")
    hub.cached[("a/b", "x.Q4_K_S.gguf")] = cached
    assert ensure_model("a/b", hub, io.StringIO(), announce_cached=False, store=store) == cached
    with pytest.raises(LauncherError, match="Q4_K_M"):
        ensure_model("a/b:Q4_K_M", hub, io.StringIO(), announce_cached=False, store=store)
    assert hub.downloads == []


@pytest.mark.parametrize(
    ("quant", "files"),
    [
        ("Q4_K", ["m-Q4_K_S.gguf"]),
        ("Q2_K", ["m-Q2_K_L.gguf"]),
        ("Q4_K", ["m.Q4_K_M.gguf", "m.Q4_K_S.gguf"]),
    ],
)
def test_a_quant_is_not_a_prefix_of_a_longer_quant(quant: str, files: list[str]) -> None:
    with pytest.raises(LauncherError, match="no GGUF file matches quant"):
        pick_gguf(files, quant)


@pytest.mark.parametrize("name", ["model.Q4_K_M.gguf", "Model-Q4_K_M-something.gguf", "m_Q4_K_M.gguf"])
def test_a_quant_still_matches_between_separators(name: str) -> None:
    assert pick_gguf([name], "Q4_K_M") == name


def test_a_multimodal_projector_is_never_the_model() -> None:
    assert pick_gguf(["mmproj-F16.gguf", "m-Q4_K_M.gguf"], None) == "m-Q4_K_M.gguf"
    assert pick_gguf(["MMPROJ-model-f16.gguf", "m-Q4_K_M.gguf"], None) == "m-Q4_K_M.gguf"
    with pytest.raises(LauncherError, match="no .gguf"):
        pick_gguf(["mmproj-F16.gguf"], None)


@pytest.mark.parametrize("text", ["..", "a/..", "./x", "../x", "a/.", ".a/b", "a/.b", "-a/b", "a/-b", "a/b/.."])
def test_repo_components_cannot_be_dots_or_dashes(text: str) -> None:
    with pytest.raises(LauncherError, match="unknown model"):
        parse_model_ref(text)


def _cache_file(hub, tmp_path: Path, content: bytes) -> Path:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(content)
    hub.cached[(REPO, "LLaDA-8B-Instruct.Q4_K_M.gguf")] = cached
    return cached


def test_fetch_records_the_verified_file(hub, store) -> None:
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    assert store.is_verified(fetch_model(plan, hub, io.StringIO(), store))


def test_a_failed_check_records_nothing(hub, store) -> None:
    hub.sha256 = "0" * 64
    plan = plan_model(ModelRef(repo=REPO, quant="Q4_K_M"), hub)
    with pytest.raises(LauncherError):
        fetch_model(plan, hub, io.StringIO(), store)
    assert not store.path.exists()


def test_an_unrecorded_cached_file_is_hashed_once_online(hub, store, tmp_path: Path) -> None:
    cached = _cache_file(hub, tmp_path, hub.content)
    out = io.StringIO()
    assert ensure_model("llada-8b", hub, out, announce_cached=True, store=store) == cached
    assert hub.sha_requests == 1 and store.is_verified(cached) and hub.downloads == []
    assert "Verifying" in out.getvalue() and "already downloaded" in out.getvalue()
    ensure_model("llada-8b", hub, io.StringIO(), announce_cached=True, store=store)
    assert hub.sha_requests == 1


def test_an_unrecorded_cached_file_that_differs_is_deleted(hub, store, tmp_path: Path) -> None:
    # What a Ctrl-C between the download and the check, or another tool, can leave behind.
    cached = _cache_file(hub, tmp_path, b"truncated")
    with pytest.raises(LauncherError, match="does not match") as e:
        ensure_model("llada-8b", hub, io.StringIO(), announce_cached=False, store=store)
    assert "Run the command again" in str(e.value)
    assert not cached.exists() and not store.is_verified(cached)


def test_a_cached_file_changed_after_its_record_is_checked_again(hub, store, tmp_path: Path) -> None:
    cached = _cache_file(hub, tmp_path, hub.content)
    ensure_model("llada-8b", hub, io.StringIO(), announce_cached=False, store=store)
    cached.write_bytes(b"tampered, and a different size")
    with pytest.raises(LauncherError, match="does not match"):
        ensure_model("llada-8b", hub, io.StringIO(), announce_cached=False, store=store)
    assert hub.sha_requests == 2


def test_offline_an_unrecorded_cached_file_is_used_with_one_warning_line(hub, store, tmp_path: Path) -> None:
    hub.offline = True
    hub.files = {"LLaDA-8B-Instruct.Q4_K_M.gguf": 9}
    cached = _cache_file(hub, tmp_path, b"truncated")
    out = io.StringIO()
    assert ensure_model("llada-8b", hub, out, announce_cached=False, store=store) == cached
    lines = out.getvalue().splitlines()
    assert len(lines) == 1 and lines[0].startswith("warning:") and "not verified" in lines[0]
    assert hub.sha_requests == 0 and cached.exists() and not store.is_verified(cached)
