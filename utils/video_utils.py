from __future__ import annotations

import os

import imageio.v2 as imageio


def save_mp4(frames, output_path, fps=50):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=16)
    return output_path
