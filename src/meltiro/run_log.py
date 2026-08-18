"""
Append-only run log for extraction results.

Manages `{run_root}/run_log.json`, a JSON array where each entry records
one extraction attempt with provenance metadata. The run root is always
supplied by the caller; there is no CWD-relative default.
"""

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Anchor git introspection to the meltiro package directory, not the process
# working directory. `git rev-parse` and `git status` walk up from their `cwd`
# to find the enclosing repository, so anchoring here ties the recorded commit
# and dirty flag to the code that is running, whatever directory the operator
# invoked from (and whatever unrelated repository that directory happens to
# sit in).
#
# Sitting inside a repository is not the same as belonging to it. The walk
# stops at the first `.git` above the package, and an installed copy commonly
# sits inside a consumer's own tree: a virtualenv at a project root puts
# site-packages several levels under it, and an install from a git URL leaves
# no `.git` of its own to stop the walk earlier. The repository found that way
# is the consumer's, and its HEAD describes their work rather than this code.
# So the enclosing repository is attributed only when it TRACKS the package's
# own files (`_anchor_tracked_in_repo`), which is exactly the condition under
# which its HEAD is a description of the bytes that ran.
_CODE_ANCHOR = Path(__file__).resolve().parent


# Git's own environment overrides, dropped from the subprocess environment
# rather than inherited. `GIT_DIR` is the dangerous one: set without
# `GIT_WORK_TREE`, it makes git treat the process CWD as that repository's
# work tree, so `git ls-files` anchored at this package would list an
# unrelated repository's index and every check below would answer for it. Git
# exports these into anything it spawns — hooks, `git bisect run`, `git rebase
# --exec`, `git submodule foreach` — so a run started from inside one of those
# would otherwise attribute this code to whatever repository invoked it. The
# anchor is the whole of what decides which repository answers; an environment
# variable must not be able to move it.
_GIT_ENV_OVERRIDES = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def _git(*args):
    """Stdout of one git command run against `_CODE_ANCHOR`, or None when it
    could not answer.

    None covers every way the question goes unanswered — git absent from PATH,
    an invocation that hangs, a non-zero exit (no enclosing repository, most
    often) — because they leave the same gap: nothing can be said about this
    copy's origin, and a guess would be written down as provenance.

    Decoded with `errors="replace"`, which is not belt-and-braces: `git
    ls-files` streams tracked path NAMES, and a repository may hold a name
    that is not valid in the process encoding — bytes from another platform,
    or any non-ASCII name under an ASCII locale. Strict decoding would raise
    `UnicodeDecodeError` out of a helper whose whole contract is that it
    answers or returns None, and it would do so from `append_run`, after a
    completed extraction, taking the run-log entry with it. A path this cannot
    spell is not a reason to lose a run: every caller reads these bytes to ask
    whether there is output at all, or to take a hex commit, and a replacement
    character changes neither answer.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, errors="replace",
            timeout=5, cwd=_CODE_ANCHOR,
            env={k: v for k, v in os.environ.items()
                 if k not in _GIT_ENV_OVERRIDES},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _anchor_tracked_in_repo():
    """Whether the enclosing repository tracks the package's own files.

    The test that turns "there is a repository above this code" into "this
    code came from that repository". `git ls-files` lists the tracked files
    under its `cwd`, so output here means the repository has these very files
    under version control and its HEAD therefore says something true about
    them; no output means the package is merely parked inside someone else's
    tree — under an ignored virtualenv, in a build directory — where the
    commit and the tree state belong to work that is not this code's.

    False, not None: this answers only whether to ask further, and every way
    of failing to establish the link is a reason not to.
    """
    listing = _git("ls-files", "--", ".")
    return listing is not None and bool(listing.strip())


def _installed_commit():
    """The commit this copy was installed from, per its own install metadata,
    or None when that metadata does not answer.

    An install from a git URL records the resolved commit in the
    distribution's `direct_url.json` (PEP 610), and for a copy that was
    installed rather than checked out that record is the whole answer to
    "which meltiro is this": there is no repository holding the package to
    ask, and the installer already wrote down what it fetched. It is the
    common case for a consumer, who pins the engine and installs it rather
    than working in it.

    Two things have to hold before the record is believed, because
    `Distribution.from_name` returns the FIRST distribution matching the name
    and expresses no preference between several.

    The metadata must belong to the package that is actually imported:
    `locate_file` has to point back at `_CODE_ANCHOR`. A distribution
    installed in the environment and a source tree ahead of it on `sys.path`
    can both answer to the name `meltiro`, and only one of them is running.

    And its version must be the version that is running. That is not
    redundant: `locate_file` resolves through the dist-info's PARENT
    directory, so every distribution in one site-packages passes the first
    check identically and it cannot discriminate between them at all. Two
    `meltiro-*.dist-info` directories side by side is not exotic — `pip
    install --target` twice leaves both, and an interrupted uninstall leaves
    one — and without this check the older one's commit can be recorded
    beside the newer one's version, which is a run record that contradicts
    itself.

    None when either check fails, when the install carries no VCS record (a
    wheel from an index, an editable install, a plain directory install), or
    when the file is unreadable or malformed, leaving `git_state()` to ask the
    repository instead. The commit is abbreviated to seven characters; a
    commit read from a repository instead honours that repository's
    `core.abbrev` and may be longer, so the two sources agree on the prefix
    rather than on the width.
    """
    from meltiro import __version__

    try:
        from importlib.metadata import Distribution
        dist = Distribution.from_name("meltiro")
        if Path(dist.locate_file("meltiro")).resolve() != _CODE_ANCHOR:
            return None
        if dist.version != __version__:
            return None
        raw = dist.read_text("direct_url.json")
        record = json.loads(raw) if raw else None
    except (ImportError, OSError, ValueError, TypeError):
        return None
    vcs_info = record.get("vcs_info") if isinstance(record, dict) else None
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        return None
    commit = vcs_info.get("commit_id")
    if not isinstance(commit, str) or not commit.strip():
        return None
    return commit.strip()[:7]


def _get_git_commit():
    """The enclosing repository's short HEAD, or None if there is none.

    A raw reading, taken with `cwd` at the package directory. Whether that
    repository is the one this code came from is `git_state`'s question, not
    this one's.
    """
    commit = _git("rev-parse", "--short", "HEAD")
    return commit.strip() if commit is not None else None


def _git_tree_dirty():
    """Whether the package's own files have uncommitted changes, or None if
    no repository tracks them.

    Scoped to `_CODE_ANCHOR` by the pathspec, not asked of the whole
    repository, because the flag is read as a statement about the CODE THAT
    RAN: a reader who sees it set concludes the recorded commit does not fully
    describe the engine. Asked of the whole repository it would be set by an
    edit to a file the engine never loads, which for a copy vendored into a
    consumer's tree means their unrelated work marks this package modified —
    the exact false claim this module exists to stop making, arriving through
    a different door. Scoped, it aligns with what `source_hash()` digests: the
    package directory, which is where an edit to the engine has to land to
    change what ran.

    True on ANY porcelain output under the anchor (staged, unstaged, or
    untracked), so a package directory that is not exactly its commit never
    reports as one that is.
    """
    status = _git("status", "--porcelain", "--", ".")
    return bool(status.strip()) if status is not None else None


def git_state():
    """Return `(short_commit, dirty)` for code-version anchoring.

    Both values describe the copy of meltiro that is running, and both are
    withheld rather than guessed. A checkout answers them together: when the
    enclosing repository tracks the package's files it is the repository this
    code came from, and its HEAD and tree state are the pair. An installed
    copy answers only the first, from its own `direct_url.json` (see
    `_installed_commit`), which names the commit the installer fetched. And a
    copy that can be placed neither way answers neither.

    So `short_commit` is an abbreviated commit or None, and `dirty` is True
    when the tree carrying this code has any uncommitted changes — any `git
    status --porcelain` output at all, staged, unstaged or untracked — False
    when it is clean, and None when there is no such tree to read. `(commit,
    None)` is therefore an ordinary pair, not a degraded one: it is what an
    installed copy looks like, a commit known from the install with no working
    tree in existence to be clean or dirty. `(None, None)` is a copy whose
    origin nothing on disk records.

    Recorded in both `run.json` (at session start, session.py) and the run-log
    entry (at append time), so a reader can find where a run's code came from
    and see whether it carried uncommitted work. WHICH CODE ran is a separate
    question, answered by `source_hash()` beside this and by the engine
    fingerprint built on it; this pair points at the origin. That division is
    why the pair may be withheld without loss: the fingerprint identifies the
    bytes wherever they sit, and a commit belonging to some other repository
    would identify nothing while reading as though it did.

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
    # The two questions are answered separately, because the sources that can
    # answer them are not the same.
    #
    # The COMMIT: the install's own record first, where it exists. It names a
    # meltiro commit, which is the answer to "which meltiro is this"; a
    # repository that merely tracks these files answers with ITS commit, which
    # for a copy vendored into a consumer's tree is a commit in their project.
    # Both describe the bytes, and the one that describes them AS A MELTIRO
    # VERSION is the better provenance.
    #
    # The DIRTY FLAG: only a repository tracking these files can measure it,
    # and it can do so whichever source named the commit. An installed copy
    # inside a tree that tracks it — a vendored dependency, committed — has a
    # working tree to read, so reporting None there would deny a fact git will
    # state on request. None is reserved for a copy nothing tracks, where
    # there genuinely is no tree to be clean or dirty.
    installed = _installed_commit()
    tracked = _anchor_tracked_in_repo()
    if installed is not None:
        return installed, (_git_tree_dirty() if tracked else None)
    if not tracked:
        return None, None
    return _get_git_commit(), _git_tree_dirty()


# What counts as the package's own source. The modules, and the engine prompt
# files beside them: `engine_prompts/*.md` is prose the engine sends to a model
# on its own authority, exactly like the framing written in the modules, and it
# reaches no config fingerprint by design (see `meltiro.prompt_partials`). If
# the digest below did not cover it, an edit to the engine's own contract would
# move nothing at all and every run before and after it would claim the same
# engine.
_SOURCE_GLOBS = ("*.py", "engine_prompts/*.md")


def _hash_tree(directory, globs=_SOURCE_GLOBS):
    """One sha256 over the package's source files under `directory`, or None if
    there are none.

    Files are visited in sorted relative-POSIX-path order, and each contributes
    `relpath\\x00bytes`, so the digest depends on the file names and their
    contents and on nothing else — not on filesystem walk order, not on where
    the directory happens to sit, not on which glob matched a file. Each glob
    is matched at any depth. `__pycache__` directories and compiled `.pyc`
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
        matched = set()
        for pattern in globs:
            matched.update(
                path for path in root.rglob(pattern)
                if "__pycache__" not in path.parts and path.is_file())
        paths = sorted(
            matched,
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

    Source here is the modules AND `engine_prompts/*.md` (see `_SOURCE_GLOBS`):
    the engine's own prompts are engine prose that reaches no config
    fingerprint, so this digest is the only thing that names them.

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


def alteksto_version():
    """The installed alteksto's version string, or None when it is absent.

    alteksto owns the paper bundle format: it decides whether a directory is
    a bundle at all, and which files under `figures/` are the crops a run
    attaches. So it is what a run's INPUT was admitted by, and a record that
    named the engine and the paper but not the version that read the paper
    would leave a reader unable to say which contract the input met.

    It reaches no fingerprint, and that is deliberate rather than an
    oversight. An engine fingerprint answers "was this the same question",
    and a validator that accepts a bundle changes nothing about the question
    asked of it; what the enumeration actually produced is already hashed on
    the paper axis, as sorted (label, content-digest) pairs in `figures_fp`.
    So the version is recorded and compared by a human, beside the versions
    that do move fingerprints.

    Read lazily and None rather than an exception on ImportError, matching
    `direktoro_version()`, so a record can still be written by a tree where
    the import has gone missing.
    """
    try:
        import alteksto
    except ImportError:
        return None
    return getattr(alteksto, "__version__", None)


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
    (the origin of the running code, not the operator's cwd). `git_dirty`
    records whether the tree carrying that code had uncommitted changes at
    append time, so a run against an uncommitted tree is not mistaken for one
    its recorded commit fully describes; it is null when there is no such tree
    to read, which an installed copy has not. A separate `git_state()` reading
    from run.json's session-start one; the two can legitimately differ (see
    `git_state`).
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
