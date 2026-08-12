"""
Append-only run log for extraction results.

Manages `{run_root}/run_log.json`, a JSON array where each entry records
one extraction attempt with provenance metadata. The run root is always
supplied by the caller; there is no CWD-relative default.
"""

import fcntl
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Anchor git introspection to the meltiro package directory, not the process
# working directory. `git rev-parse` and `git status` walk up from their `cwd`
# to find the enclosing repository, so anchoring here ties the recorded commit
# and dirty flag to the repository that holds the running code, whatever
# directory the operator invoked from (and whatever unrelated repository that
# directory happens to sit in). A site-packages install sits outside any repo,
# so git returns non-zero and both fields degrade to None.
_CODE_ANCHOR = Path(__file__).resolve().parent


def _get_git_commit():
    """Get the short git commit of the code repo, or None if not in one.

    Runs `git rev-parse` anchored to the meltiro package directory
    (`_CODE_ANCHOR`), so the commit belongs to the repository that holds the
    running code, not the operator's working directory. None when git is
    unavailable, times out, or the package sits outside any repo (a
    site-packages install).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=_CODE_ANCHOR,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _git_tree_dirty():
    """Whether the code repo's working tree has uncommitted changes.

    Runs `git status --porcelain` anchored to the meltiro package directory
    (`_CODE_ANCHOR`), so it reports the state of the repository that holds the
    running code. Returns True on ANY porcelain output in that repo (staged,
    unstaged, or untracked files), False when the tree is clean, and None when
    git is unavailable, times out, or the package sits outside any repo,
    matching `_get_git_commit`'s None so a consumer can tell an unknown tree
    apart from a known-clean one.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=_CODE_ANCHOR,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def git_state():
    """Return `(short_commit, dirty)` for code-version anchoring.

    Both values describe the git repository that holds the meltiro package: the
    subprocess calls are anchored to `_CODE_ANCHOR` (the package directory), so
    they never read whatever repository the operator ran the command from.
    `short_commit` is the abbreviated HEAD hash or None; `dirty` is True when
    that code repo's working tree has any uncommitted changes, meaning any
    `git status --porcelain` output at all (staged, unstaged, or untracked
    files in the code repo), False when clean, and None when git is unavailable
    or the package sits outside any repo. Recorded in both `run.json` (at
    session start, session.py) and the run-log entry (at append time), so a
    reader can find the checkout a run came from and see whether it carried
    uncommitted work. WHICH CODE ran is a separate question, answered by
    `source_hash()` beside this and by the engine fingerprint built on it; this
    pair points at the repository.

    The recordings are independent `git_state()` calls taken at different
    moments, so they can legitimately differ: a code change mid-run (for
    example a resume under a new commit) moves the run-log anchor without
    rewriting the session-start run.json anchor. Consumers must not assume they
    are equal. The run-log entry reflects the code at completion, run.json the
    code at session start.

    There is a third recording between those two, and it is what makes the
    difference readable rather than merely possible: each resumed segment
    writes its own commit, dirtiness and engine fingerprint onto the `resumed`
    event in the session's event log. So a run that changed code mid-way has a
    commit for its start, one per segment, and one for its completion, and a
    consumer that needs the whole history reads the events rather than
    inferring it from two endpoints that happen to disagree.
    """
    return _get_git_commit(), _git_tree_dirty()


def _hash_tree(directory):
    """One sha256 over every `*.py` under `directory`, or None if there is none.

    Files are visited in sorted relative-POSIX-path order, and each contributes
    `relpath\\x00bytes`, so the digest depends on the file names and their
    contents and on nothing else — not on filesystem walk order, not on where
    the directory happens to sit. `__pycache__` directories and compiled `.pyc`
    files are skipped: they are derived from the source, they differ between
    interpreters, and hashing them would make one checkout report two digests.

    None means there was nothing to hash or it could not be read — a path that
    is not a directory, a directory holding no source, a file that vanished or
    refused to open mid-walk. A digest over zero files would otherwise be a
    fixed constant claiming to identify whatever produced it.
    """
    root = Path(directory)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    try:
        paths = sorted(
            (path for path in root.rglob("*.py")
             if "__pycache__" not in path.parts),
            key=lambda path: path.relative_to(root).as_posix())
        if not paths:
            return None
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def source_hash():
    """A sha256 hex digest of the imported meltiro package's own source files.

    The other half of what `git_state` reads, and the half that identifies the
    CODE rather than the checkout. The version names a release and the commit
    names a tree in one repository; this names the bytes that ran, so a patched
    install, a working edit and a wheel built from a tag are each distinct and
    each identifiable wherever they sit. It is the *meltiro* component of
    `fingerprint.engine_fingerprint`.

    Anchored to the imported package (`Path(meltiro.__file__).parent`), not to
    the repository around it, so an installed copy and the checkout it was
    built from hash the same way. Returns the token `"nosource"` when the
    package's source cannot be read, as for a frozen or zipimported copy.

    direktoro's own `source_hash()` is the same function over that package
    (see `direktoro.provenance`), so the two halves of the engine are named by
    the same recipe.
    """
    import meltiro

    here = getattr(meltiro, "__file__", None)
    if here is None:
        return "nosource"
    digest = _hash_tree(Path(here).parent)
    return "nosource" if digest is None else digest


def direktoro_version():
    """The installed direktoro's version string, or None when it is absent.

    direktoro is half the engine: it builds the provider-call identity block
    that is the first component of every stage fingerprint, it decides which
    decoding params are actually sent, and it declares each model's image
    capability. A change to the SHAPE of any of those moves every fingerprint
    meltiro computes, so the version has to be recorded beside the meltiro
    version and folded into the engine axis; see
    `fingerprint.engine_fingerprint`.

    Read lazily, inside the function, and None rather than an exception when
    the import fails: `import meltiro` must keep working with direktoro absent
    (a consumer that installs the wheel `--no-deps` has none, and reads and
    validates bundles without ever placing a call). None is the absent marker
    a run record carries, on the same terms as an absent `git_commit`, and it
    folds into the engine fingerprint as its own fixed token so an engine with
    no direktoro is never mistaken for one with a direktoro whose version went
    unrecorded.
    """
    try:
        import direktoro
    except ImportError:
        return None
    return getattr(direktoro, "__version__", None)


def direktoro_source_hash():
    """direktoro's own source digest, or None when the package is absent.

    The content half of direktoro's engine identity, on the same terms as
    `source_hash()` is meltiro's: the version names the release, this names
    the bytes. It is direktoro's function over direktoro's package directory
    (see `direktoro.provenance`), so the value belongs to the package that
    owns the code, exactly as the version string does.

    Imported lazily and None on ImportError, matching `direktoro_version()`,
    so `import meltiro` keeps working with direktoro absent.
    """
    try:
        import direktoro
    except ImportError:
        return None
    return direktoro.source_hash()


def engine_identity():
    """The engine's identity as a `(meltiro_version, meltiro_src,
    direktoro_version, direktoro_src)` quadruple.

    The inputs to `fingerprint.engine_fingerprint`, in its argument order and
    gathered in one place so the session, the resumed segment and the dry-run
    report cannot disagree about what the engine axis is made of. The meltiro
    version is always available; its source digest comes from `source_hash()`;
    the direktoro pair comes from `direktoro_version()` and
    `direktoro_source_hash()` and is `(None, None)` when that package is not
    installed.

    The git commit and dirty flag are NOT here. They are recorded with every
    run by `git_state()` beside this, because they say which checkout the code
    came from, which a reader wants; the axis itself is content, so what
    identifies the code is the digests above.

    Like `git_state()`, this takes a reading at the moment it is called. A
    session, each of its resumed segments and its run-log entry read it
    separately, so they can legitimately differ when the code changes mid-run.
    """
    from meltiro import __version__
    return (__version__, source_hash(),
            direktoro_version(), direktoro_source_hash())


def current_engine_fp(identity=None):
    """The `engine_fp` the code running right now would record.

    One expression of "which engine is this", so every consumer of the answer
    — the fingerprint a new session records, the one a resumed segment
    records, the comparison a refused resume makes to name the axis that moved
    — computes it the same way and can be compared against the others.

    `identity` is an `engine_identity()` quadruple for a caller that already
    has one and wants the fingerprint OF that reading; omitted, a fresh
    reading is taken. Passing the one in hand is what keeps a segment's
    recorded versions and its recorded fingerprint two views of a single
    reading rather than two readings taken moments apart.

    Imported lazily so this module keeps importing with nothing behind it.
    """
    from meltiro.fingerprint import engine_fingerprint
    return engine_fingerprint(*(identity if identity is not None
                                else engine_identity()))


def _log_path(log_dir):
    """Resolve the run log file path."""
    return Path(log_dir) / "run_log.json"


def load_log(log_dir):
    """Load the full run log. Returns list of entry dicts."""
    path = _log_path(log_dir)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_run(entry, log_dir):
    """Append a run entry to the log.

    entry should be a dict with keys: study_id, result_file,
    prompt_hash, model, status, input_tokens, output_tokens, cost_usd,
    cost_rates, usage_by_role, validation_passed, validation_errors,
    template_hash.

    Automatically adds: timestamp, git_commit, git_dirty, from `git_state()`
    (anchored to the code repo, not the operator's cwd). `git_dirty` records
    whether the code repo's tree had uncommitted changes at append time, so a
    run against an uncommitted tree is not mistaken for one its recorded
    commit fully describes. A separate `git_state()` reading from run.json's
    session-start one; the two can legitimately differ (see `git_state`).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    git_commit, git_dirty = git_state()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    entry["git_commit"] = git_commit
    entry["git_dirty"] = git_dirty

    path = _log_path(log_dir)
    # Serialise the whole read-modify-write under an exclusive lock so two
    # sessions finishing concurrently (e.g. --all batch) can't clobber each
    # other's append. The lock lives on a sidecar so it survives the atomic
    # replace of run_log.json itself.
    lock_path = log_dir / "run_log.json.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            log = load_log(log_dir)
            log.append(entry)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
            tmp_path.replace(path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return entry
