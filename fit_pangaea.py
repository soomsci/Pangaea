#!/usr/bin/env python3
"""판게아 목표 위치 계산 -> index.html 의 TARGET 블록을 교체.

각 대륙에 대해 "이 지점(경위도)이 아프리카 기준으로 여기(경위도)에 와야 한다"는
대응점을 적으면, 강체 정합(회전+이동)으로 {x,y,rot} 을 뽑는다.
아프리카를 고정 기준으로 쓰는 것은 실제 고지리 복원에서 쓰는 관례와 같다.

실행: python3 fit_pangaea.py
"""
import json, math, re, subprocess, sys

SCALE = 3.2
AF_POS = (566.0, 286.0)      # 아프리카를 화면 어디에 고정할지

# ---- shapes.js 에서 cx, cy, proj 읽기 -------------------------------------
src = open("shapes.js").read()
GEO = {}
for blk in src.split('{ id:"')[1:]:
    pid = blk[:blk.index('"')]
    GEO[pid] = {k: float(re.search(rf"{k}:(-?[\d.]+)", blk).group(1))
                for k in ("cx", "cy")}
    GEO[pid]["proj"] = re.search(r'proj:"(\w+)"', blk).group(1)


def local(pid, lon, lat):
    """경위도 -> 조각 로컬 좌표 (build_shapes.py 의 project 와 동일해야 한다)."""
    g = GEO[pid]
    if g["proj"] == "spole":
        r, a = (90 + lat) * SCALE, math.radians(lon)
        return r * math.sin(a) - g["cx"], r * math.cos(a) - g["cy"]
    return lon * SCALE - g["cx"], -lat * SCALE - g["cy"]


def world_af(lon, lat):
    """아프리카 위의 경위도 -> 화면 좌표 (아프리카는 회전 0으로 고정)."""
    lx, ly = local("af", lon, lat)
    return AF_POS[0] + lx, AF_POS[1] + ly


def fit(pid, pairs, force_rot=None, nudge=(0, 0)):
    """pairs = [((조각 경도,위도), (아프리카 기준 경도,위도)), ...]
    강체 정합(스케일 고정) -> {x, y, rot}. 대응점 1개면 회전을 직접 준다.
    nudge 는 계산 결과를 눈으로 보고 겹침을 없애는 px 보정값이다.
    정거원통도법에서는 위도에 따라 가로 축척이 달라 대응점만으로는 겹침이 남는다."""
    src_pts = [local(pid, *a) for a, _ in pairs]
    dst_pts = [world_af(*b) for _, b in pairs]
    n = len(pairs)
    sx = sum(p[0] for p in src_pts) / n; sy = sum(p[1] for p in src_pts) / n
    dx = sum(p[0] for p in dst_pts) / n; dy = sum(p[1] for p in dst_pts) / n

    if force_rot is not None or n == 1:
        rot = force_rot or 0.0
    else:
        num = sum((p[0]-sx)*(q[1]-dy) - (p[1]-sy)*(q[0]-dx)
                  for p, q in zip(src_pts, dst_pts))
        den = sum((p[0]-sx)*(q[0]-dx) + (p[1]-sy)*(q[1]-dy)
                  for p, q in zip(src_pts, dst_pts))
        rot = math.degrees(math.atan2(num, den))
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return {"x": round(dx - (sx*c - sy*s)) + nudge[0],
            "y": round(dy - (sx*s + sy*c)) + nudge[1], "rot": round(rot)}


# ---- 판게아 조립 명세 ------------------------------------------------------
# (조각의 어느 지점) 이 (아프리카의 어느 지점) 위치에 오도록.
# 정거원통도법은 고위도에서 동서로 늘어나므로, 적도에서 먼 조각은 대응점 대신
# 회전을 직접 지정하고 한 점만 맞춘다.
SPEC = {
    # 남아메리카: 브라질 동해안이 기니만에 들어간다. 적도 부근이라 정합이 잘 맞는다.
    "sa": dict(pairs=[((-35, -7), (8.5, 3.5)),
                      ((-42, -2), (-2, 5)),
                      ((-43, -23), (12, -9))], nudge=(-22, 6)),
    # 북아메리카: 동해안이 아프리카 북서안(모리타니~모로코)에 붙는다.
    "na": dict(pairs=[((-79, 26), (-15, 20))], force_rot=14, nudge=(-36, -34)),
    # 유라시아: 이베리아반도가 모로코 위에 얹힌다(테티스해가 닫힌 상태).
    "eu": dict(pairs=[((-9, 38), (-6, 40))], force_rot=-12),
    # 인도: 아프리카 동해안(소말리아~케냐) 바깥에 붙는다.
    "in": dict(pairs=[((72, 20), (46, -3))], force_rot=-38, nudge=(8, 10)),
    # 오스트레일리아: 인도 남동쪽, 남극과 붙어 있다.
    "oc": dict(pairs=[((114, -22), (57, -30))], force_rot=-40, nudge=(-20, 45)),
    # 남극: 아프리카 남쪽. 남아메리카 남단~오스트레일리아와 이어진다.
    "an": dict(pairs=[((20, -70), (24, -48))], force_rot=10, nudge=(30, 74)),
}

targets = {"af": {"x": int(AF_POS[0]), "y": int(AF_POS[1]), "rot": 0}}
for pid, spec in SPEC.items():
    targets[pid] = fit(pid, spec["pairs"], spec.get("force_rot"),
                       spec.get("nudge", (0, 0)))

order = ["af", "sa", "na", "eu", "in", "oc", "an"]
block = "const TARGET = {\n" + "".join(
    f'  {p}: {{x:{targets[p]["x"]}, y:{targets[p]["y"]}, rot:{targets[p]["rot"]}}},\n'
    for p in order) + "};"

html = open("index.html").read()
new = re.sub(r"const TARGET = \{.*?\n\};", block, html, count=1, flags=re.S)
if new == html:
    sys.exit("TARGET 블록을 찾지 못했습니다")
open("index.html", "w").write(new)
print(block)
