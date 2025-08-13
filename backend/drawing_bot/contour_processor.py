#!/usr/bin/env python3
import re
import json
from pathlib import Path

# EV3 축 기준 최대값
AXIS_MAX_X = 400
AXIS_MAX_Y = 1100

def parse_contours(raw_text):
    blocks = re.split(r'#\s*contour\s*\d+', raw_text)[1:]
    contours = []
    for blk in blocks:
        pts = []
        for line in blk.strip().splitlines():
            m = re.match(r'\s*(\d+)\s*,\s*(\d+)\s*$', line)
            if m:
                pts.append((int(m.group(1)), int(m.group(2))))
        if pts:
            contours.append(pts)
    return contours

def scale_contours(contours):
    xs = [x for c in contours for x,_ in c]
    ys = [y for c in contours for _,y in c]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    def mx(px): return (px - minx) / (maxx - minx) * AXIS_MAX_X
    def my(py): return (py - miny) / (maxy - miny) * AXIS_MAX_Y

    return [
        [(mx(x), my(y)) for x,y in contour]
        for contour in contours
    ]

def main():
    infile  = Path("/root/instruct-pix2pix/contours.txt")
    outfile = Path("/root/backend/drawing_bot/drawing_paths_stream.json")

    raw      = infile.read_text(encoding="utf-8")
    contours = parse_contours(raw)
    scaled   = scale_contours(contours)

    # contour 하나당 json.dumps → 한 줄씩 기록
    with outfile.open("w", encoding="utf-8") as f:
        for contour in scaled:
            f.write(json.dumps(contour, ensure_ascii=False))
            f.write("\n")

    print(f"✔ {outfile.name} 생성 완료: {len(scaled)} contours (한 줄에 1 contour씩)")

if __name__ == "__main__":
    main()
