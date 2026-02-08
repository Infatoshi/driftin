"""FID computation wrapper. Uses pytorch-fid."""

import subprocess
import sys


def compute_fid(path_real, path_fake, device="cuda:0"):
    """Compute FID between two directories of images using pytorch-fid.

    Returns FID score as float, or None if computation fails.
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytorch_fid",
                path_real, path_fake,
                "--device", device,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        # Parse FID from output: "FID:  XX.XX"
        for line in result.stdout.strip().split("\n"):
            if "FID" in line:
                fid = float(line.split(":")[-1].strip())
                return fid
        print(f"pytorch-fid stdout: {result.stdout}")
        print(f"pytorch-fid stderr: {result.stderr}")
        return None
    except Exception as e:
        print(f"FID computation failed: {e}")
        return None
