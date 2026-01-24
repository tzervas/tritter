"""Checkpoint offloading to remote servers.

Why: Training generates large checkpoints that can quickly fill local storage.
Offloading to a remote server keeps local storage manageable while preserving
all checkpoints for analysis and model selection.

Usage:
    # Offload checkpoint to homelab server
    offload_checkpoint(
        "checkpoints/7b-step-10000/",
        "homelab",
        remote_dir="/data/tritter/checkpoints/"
    )

    # List remote checkpoints
    checkpoints = list_remote_checkpoints("homelab", "/data/tritter/checkpoints/")

    # Retrieve checkpoint
    retrieve_checkpoint(
        "homelab",
        "/data/tritter/checkpoints/7b-step-10000/",
        "checkpoints/7b-step-10000/"
    )
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteCheckpoint:
    """Information about a remote checkpoint."""

    host: str
    path: str
    size_bytes: int
    modified_time: str


def offload_checkpoint(
    local_path: str | Path,
    remote_host: str,
    remote_dir: str = "/data/tritter/checkpoints/",
    delete_local: bool = False,
    compress: bool = True,
) -> str:
    """Offload checkpoint to remote server via SSH/rsync.

    Args:
        local_path: Local checkpoint path (file or directory)
        remote_host: SSH host alias (e.g., "homelab")
        remote_dir: Remote directory to store checkpoints
        delete_local: If True, delete local copy after successful upload
        compress: If True, compress during transfer (rsync -z)

    Returns:
        Remote path where checkpoint was stored

    Why: Prevents local storage exhaustion during long training runs.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {local_path}")

    # Construct remote path
    remote_path = f"{remote_host}:{remote_dir}/{local_path.name}"

    # Build rsync command
    cmd = ["rsync", "-av", "--progress"]
    if compress:
        cmd.append("-z")

    # Add source (with trailing slash for directory contents)
    if local_path.is_dir():
        cmd.append(f"{local_path}/")
        remote_path = f"{remote_host}:{remote_dir}/{local_path.name}/"
    else:
        cmd.append(str(local_path))

    cmd.append(remote_path)

    # Ensure remote directory exists
    mkdir_cmd = ["ssh", remote_host, f"mkdir -p {remote_dir}"]
    subprocess.run(mkdir_cmd, check=True)

    # Run rsync
    print(f"Offloading {local_path.name} to {remote_host}...")
    subprocess.run(cmd, check=True)

    # Verify upload
    verify_cmd = ["ssh", remote_host, f"ls -la {remote_dir}/{local_path.name}"]
    result = subprocess.run(verify_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Upload verification failed: {result.stderr}")

    print(f"Successfully offloaded to {remote_path}")

    # Optionally delete local copy
    if delete_local:
        if local_path.is_dir():
            import shutil

            shutil.rmtree(local_path)
        else:
            local_path.unlink()
        print(f"Deleted local copy: {local_path}")

    return remote_path


def retrieve_checkpoint(
    remote_host: str,
    remote_path: str,
    local_path: str | Path,
    compress: bool = True,
) -> Path:
    """Retrieve checkpoint from remote server.

    Args:
        remote_host: SSH host alias (e.g., "homelab")
        remote_path: Remote checkpoint path
        local_path: Local path to save checkpoint
        compress: If True, compress during transfer

    Returns:
        Path to downloaded checkpoint

    Why: Retrieve specific checkpoints for evaluation or continued training.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Build rsync command
    cmd = ["rsync", "-av", "--progress"]
    if compress:
        cmd.append("-z")

    # Add source and destination
    if not remote_path.startswith("/"):
        remote_path = f"/{remote_path}"

    cmd.append(f"{remote_host}:{remote_path}")
    cmd.append(str(local_path))

    print(f"Retrieving {remote_path} from {remote_host}...")
    subprocess.run(cmd, check=True)

    print(f"Successfully retrieved to {local_path}")
    return local_path


def list_remote_checkpoints(
    remote_host: str,
    remote_dir: str = "/data/tritter/checkpoints/",
) -> list[RemoteCheckpoint]:
    """List checkpoints on remote server.

    Args:
        remote_host: SSH host alias
        remote_dir: Remote directory to list

    Returns:
        List of RemoteCheckpoint objects

    Why: Discover available checkpoints for evaluation or retrieval.
    """
    # Get listing with sizes and times
    cmd = [
        "ssh",
        remote_host,
        f"ls -la --time-style=long-iso {remote_dir} 2>/dev/null || echo 'EMPTY'",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    if "EMPTY" in result.stdout or "No such file" in result.stderr:
        return []

    checkpoints = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) < 8:
            continue
        if parts[0].startswith("d") or parts[0].startswith("-"):
            # Parse ls output
            # drwxr-xr-x 2 user group 4096 2026-01-23 12:00 checkpoint-name
            size = int(parts[4])
            modified = f"{parts[5]} {parts[6]}"
            name = parts[-1]

            if name in (".", ".."):
                continue

            checkpoints.append(
                RemoteCheckpoint(
                    host=remote_host,
                    path=f"{remote_dir}/{name}",
                    size_bytes=size,
                    modified_time=modified,
                )
            )

    return checkpoints


def get_remote_disk_usage(
    remote_host: str,
    remote_dir: str = "/data/tritter/checkpoints/",
) -> dict[str, int | str]:
    """Get disk usage statistics for remote checkpoint directory.

    Args:
        remote_host: SSH host alias
        remote_dir: Remote directory to check

    Returns:
        Dictionary with usage stats

    Why: Monitor storage consumption on remote server.
    """
    # Get directory size
    du_cmd = ["ssh", remote_host, f"du -sb {remote_dir} 2>/dev/null || echo '0'"]
    du_result = subprocess.run(du_cmd, capture_output=True, text=True)
    total_bytes = int(du_result.stdout.strip().split()[0])

    # Get disk free space
    df_cmd = [
        "ssh",
        remote_host,
        f"df -B1 {remote_dir} | tail -1",
    ]
    df_result = subprocess.run(df_cmd, capture_output=True, text=True)
    df_parts = df_result.stdout.strip().split()

    free_bytes = int(df_parts[3]) if len(df_parts) > 3 else 0

    # Count checkpoints
    ls_cmd = ["ssh", remote_host, f"ls -1 {remote_dir} 2>/dev/null | wc -l"]
    ls_result = subprocess.run(ls_cmd, capture_output=True, text=True)
    checkpoint_count = int(ls_result.stdout.strip())

    return {
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024**3),
        "free_bytes": free_bytes,
        "free_gb": free_bytes / (1024**3),
        "checkpoint_count": checkpoint_count,
    }


def cleanup_old_checkpoints(
    remote_host: str,
    remote_dir: str = "/data/tritter/checkpoints/",
    keep_last_n: int = 5,
    keep_pattern: str | None = None,
    dry_run: bool = True,
) -> list[str]:
    """Remove old checkpoints from remote server.

    Args:
        remote_host: SSH host alias
        remote_dir: Remote directory
        keep_last_n: Number of most recent checkpoints to keep
        keep_pattern: Glob pattern for checkpoints to always keep (e.g., "*-final")
        dry_run: If True, only show what would be deleted

    Returns:
        List of deleted (or would-be-deleted) paths

    Why: Prevent unbounded storage growth during long training.
    """
    checkpoints = list_remote_checkpoints(remote_host, remote_dir)

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda c: c.modified_time, reverse=True)

    # Determine which to delete
    to_delete = []
    kept = 0

    for checkpoint in checkpoints:
        name = Path(checkpoint.path).name

        # Check if should be kept by pattern
        if keep_pattern:
            import fnmatch

            if fnmatch.fnmatch(name, keep_pattern):
                continue

        # Keep the most recent N
        if kept < keep_last_n:
            kept += 1
            continue

        to_delete.append(checkpoint.path)

    if dry_run:
        print(f"Would delete {len(to_delete)} checkpoints:")
        for path in to_delete:
            print(f"  {path}")
    else:
        for path in to_delete:
            cmd = ["ssh", remote_host, f"rm -rf {path}"]
            subprocess.run(cmd, check=True)
            print(f"Deleted: {path}")

    return to_delete


__all__ = [
    "RemoteCheckpoint",
    "cleanup_old_checkpoints",
    "get_remote_disk_usage",
    "list_remote_checkpoints",
    "offload_checkpoint",
    "retrieve_checkpoint",
]
