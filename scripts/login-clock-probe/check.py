#!/usr/bin/env python3
"""Check the clock band in host captures of the monochrome login fixture.

Usage: python3 scripts/login-clock-probe/check.py FRAME.png [FRAME.png ...]
Accepts the undecorated 16:9 login display at native or scaled resolution.
Only use this on the grey login screen with the large top-centred clock.
Sample across minute changes; a pass measures only the supplied frames.
Returns 0 for achromatic clock bands, 1 for colour, 2 for unscorable input.
Requires ffmpeg. No guest reference image or recognition of the time needed.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check(path):
    # Normalize to the probe's measurement grid; letterboxing is not part
    # of this fixture. Keep decoded RGB, rather than subsampling the chroma.
    band = subprocess.check_output([
        'ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-vf',
        "select='eq(iw*9,ih*16)',format=rgb24,scale=960:540,crop=240:65:360:68",
        '-frames:v', '1', '-threads', '1', '-pix_fmt', 'rgb24',
        '-f', 'rawvideo', '-',
    ])
    if len(band) != 240 * 65 * 3:
        raise ValueError('not a 16:9 display capture')
    pixels = list(zip(band[0::3], band[1::3], band[2::3]))
    chromatic = sum(max(p) - min(p) > 32 for p in pixels)
    if chromatic:
        return 1, f'{chromatic} coloured pixels in clock band'
    # A flat grey or black capture is not evidence of a clean clock. Check
    # for a population of light strokes against the median background.
    light = sorted(min(p) for p in pixels)
    if light[-100] - light[len(light) // 2] < 32:
        raise ValueError('no contrasting strokes in clock band')
    return 0, 'achromatic clock band'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame', type=Path, nargs='+')
    status = 0
    for frame in parser.parse_args().frame:
        try:
            code, message = check(frame)
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            code, message = 2, str(error)
        print(f'{frame}: {message}')
        status = max(status, code)
    return status


if __name__ == '__main__':
    sys.exit(main())
