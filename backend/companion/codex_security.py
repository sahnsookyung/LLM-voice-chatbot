"""Local transport checks, not a sandbox against other apps running as this user."""

import ctypes
import errno
import os
import re
import socket
import stat
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

BUNDLED_CLI = Path(
    "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"
)


class CodexSecurityError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _installed_code_hash(file_identity):
    # Cache only for this exact installed file. Strict signature verification
    # prevents comparing against a modified executable's stale CodeDirectory.
    del file_identity
    try:
        subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", str(BUNDLED_CLI)],
            check=True,
            capture_output=True,
            timeout=5,
        )
        result = subprocess.run(
            ["/usr/bin/codesign", "-d", "--verbose=4", str(BUNDLED_CLI)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"^CDHash=([0-9a-f]{40})$", result.stderr, re.MULTILINE)
    return bytes.fromhex(match[1]) if match else None


def relocated_host_matches(pid):
    """Verify code identity when Sparkle removed a still-running host's old path.

    This is not a path/name/argv exception: the kernel's running CodeDirectory
    hash must match the strictly verified installed desktop host byte-for-byte.
    A changed binary or an unavailable hash continues to fail closed.
    """
    try:
        info = BUNDLED_CLI.stat()
        expected = _installed_code_hash((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
        if expected is None or BUNDLED_CLI.stat() != info:
            return False
        libc = ctypes.CDLL(None, use_errno=True)
        libc.csops.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t]
        libc.csops.restype = ctypes.c_int
        digest = ctypes.create_string_buffer(20)
        return libc.csops(pid, 5, digest, 20) == 0 and digest.raw == expected  # CS_OPS_CDHASH
    except (OSError, AttributeError):
        return False


def private_directory_acl(path):
    if sys.platform != "darwin":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    libc.acl_get_fd_np.restype = ctypes.c_void_p
    libc.acl_get_entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    libc.acl_valid.argtypes = [ctypes.c_void_p]
    libc.acl_get_tag_type.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    libc.acl_free.argtypes = [ctypes.c_void_p]
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        acl = libc.acl_get_fd_np(descriptor, 0x100)  # ACL_TYPE_EXTENDED
        failure = ctypes.get_errno()
    finally:
        os.close(descriptor)
    if not acl:
        if failure == errno.ENOENT:
            return  # The valid open directory has no extended ACL.
        raise CodexSecurityError("Cannot verify the Codex socket directory ACL.")
    try:
        if libc.acl_valid(acl) != 0:
            raise CodexSecurityError("Invalid Codex socket directory ACL.")
        entry, tag = ctypes.c_void_p(), ctypes.c_int()
        cursor = 0  # ACL_FIRST_ENTRY, then ACL_NEXT_ENTRY
        while True:
            result = libc.acl_get_entry(acl, cursor, ctypes.byref(entry))
            if result == -1 and ctypes.get_errno() == errno.EINVAL:
                break  # Darwin reports end-of-list as EINVAL, unlike Linux.
            if result != 0 or libc.acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                raise CodexSecurityError("Cannot read the Codex socket directory ACL.")
            if tag.value == 1:  # ACL_EXTENDED_ALLOW; deny-only ACLs cannot widen access.
                raise CodexSecurityError(
                    "Codex socket directory has an extended access grant. Review it before connecting."
                )
            cursor = -1  # ACL_NEXT_ENTRY
    finally:
        libc.acl_free(acl)


def validate_socket_directory(path: Path):
    if not path.is_absolute() or ".." in path.parts:
        raise CodexSecurityError("Codex socket must have an absolute, normalized path.")
    # Check before resolving: resolving first would hide redirected directories.
    for parent in path.parents:
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, os.getuid()}:
            raise CodexSecurityError("Codex socket directory has an unexpected owner or link.")
        if info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX):
            raise CodexSecurityError("Codex socket directory is writable by other users.")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise CodexSecurityError(
            "Codex socket needs a private directory owned by this user (0700)."
        )
    private_directory_acl(path.parent)


def daemon_socket_root():
    return Path("/private/tmp") / f"codex-daemon-{os.getuid()}"


def validate_socket_path(path: Path):
    validate_socket_directory(path)
    info = path.lstat()
    original = (info.st_dev, info.st_ino)
    target = path
    if stat.S_ISLNK(info.st_mode):
        # The inspected bundled host shortens Unix paths with this managed link.
        target = Path(os.readlink(path))
        if (
            info.st_uid != os.getuid()
            or path.parent.name != "app-server-control"
            or path.name != "app-server-control.sock"
            or target.parent != daemon_socket_root()
            or not re.fullmatch(r"[0-9a-f]{64}", target.name)
        ):
            raise CodexSecurityError("Unrecognized Codex socket link.")
        validate_socket_directory(target)
        info = target.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise CodexSecurityError("Codex endpoint must be a socket owned by this user, not a link.")
    # macOS relies on access to the containing directory for Unix socket isolation.
    return target, (original, info.st_dev, info.st_ino)


def peer_identity(connection: socket.socket):
    if sys.platform != "darwin":
        raise CodexSecurityError("Verified Codex desktop attachment currently requires macOS.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.getpeereid.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    libc.getpeereid.restype = ctypes.c_int
    uid, gid = ctypes.c_uint(), ctypes.c_uint()
    if libc.getpeereid(connection.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
        raise CodexSecurityError("Cannot verify the Codex socket peer credentials.")
    # SOL_LOCAL=0, LOCAL_PEERPID=2, from the macOS SDK's sys/un.h.
    pid = connection.getsockopt(0, 2)
    proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    proc.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    proc.proc_pidpath.restype = ctypes.c_int
    executable = ctypes.create_string_buffer(4096)
    if pid <= 0:
        raise CodexSecurityError("Cannot verify the Codex server executable.")
    if proc.proc_pidpath(pid, executable, len(executable)) <= 0:
        if ctypes.get_errno() == errno.ENOENT and relocated_host_matches(pid):
            return uid.value, pid, BUNDLED_CLI
        raise CodexSecurityError("Cannot verify the Codex server executable.")
    return uid.value, pid, Path(os.fsdecode(executable.value))


def verify_peer(connection, expected_binary=BUNDLED_CLI):
    uid, pid, executable = peer_identity(connection)
    if uid != os.getuid() or executable.resolve() != expected_binary.resolve():
        raise CodexSecurityError(
            "The socket peer is not this user's desktop-compatible Codex host."
        )
    return pid
