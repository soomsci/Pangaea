#!/usr/bin/env python3
"""판게아 목표 위치 계산 -> index.html 의 TARGET 블록을 교체.

두 단계로 맞춘다.

1) 출발 위치(SPEC 의 init): 판게아 복원도를 보고 잡아 화면에서 확인한 배치.
   아프리카를 고정 기준으로 쓰는 것은 실제 고지리 복원의 관례와 같다.

2) 해안선 밀착(hug): 이미 배치된 대륙들의 외곽선으로 부호 있는 거리장(SDF)을
   만들고, 움직이는 대륙이 "겹치지 않으면서 가장 촘촘히 붙는" 위치를
   출발 위치 주변 좁은 범위에서만 찾는다.

밀착 탐색을 넓게 열면 안 된다. 촘촘함만 최적화하면 대륙이 아무 해안 홈에나
끼워져 지질학적으로 엉뚱한 판게아가 나온다(북아메리카가 아프리카 위로 올라가고
오스트레일리아가 적도까지 올라오는 식). 그래서 범위를 좁게 두고, 출발 위치에서
멀어지는 데 벌점을 준다.

실행: python3 fit_pangaea.py
"""
import math, re, sys
import numpy as np

AF_POS = (566.0, 286.0)      # 아프리카를 화면 어디에 고정할지

# 맞물림을 재는 외곽선 점의 개수. 비율(예: 25%)로 잡으면 안 된다 —
# 북아메리카처럼 큰 대륙은 알래스카부터 그린란드까지가 외곽선이라
# 그 25%가 아프리카에 닿는 일이 불가능하고, 최적화가 대륙을 아프리카 속으로
# 밀어 넣어 버린다. 개수로 잡으면 대륙 크기와 무관하게 "맞닿는 해안 한 구간"이 된다.
CONTACT_N = 45
SLACK     = 3.0    # px. 이 정도 겹침은 눈에 띄지 않고 해안선 정합에도 원래 오차가 있다
OVER_W    = 0.6    # 그보다 깊이 파고들 때 1px 당 벌점
MAX_OVER  = 8.0    # px. 이보다 깊이 파고드는 배치는 후보에서 제외한다.
                   # 고위도 대륙은 정거원통도법에서 그린란드도 유럽도 동서로
                   # 부풀어 이 정도로는 맞출 수 없다. SPEC 에서 대륙별로 완화한다.
DRIFT_XY  = 0.055  # 출발 위치에서 1px 벗어날 때의 벌점
DRIFT_XY  = 0.055  # 출발 위치에서 1px 벗어날 때의 벌점
DRIFT_R   = 0.13   # 출발 각도에서 1도 벗어날 때의 벌점

# ---- 판게아 조립 명세 ------------------------------------------------------
# init: 판게아 복원도를 보고 잡아 화면에서 확인한 배치 (x, y, 회전).
# against: 여기에 밀착시킨다. 순서가 중요하다 — 앞에서 배치된 대륙에 붙는다.
SPEC = [
    # 브라질 동해안이 기니만에 들어간다. 적도 부근이라 도법 왜곡이 가장 작고
    # 그래서 이 정합이 제일 잘 맞는다. 수업의 핵심 장면.
    ("sa", (492, 362, -44), ["af"]),
    # 동해안(플로리다~뉴펀들랜드)이 아프리카 북서안에 붙는다.
    ("na", (409, 123, 14), ["af"]),
    # 이베리아반도가 모로코 위에 얹히고 북아메리카와도 이어진다.
    # 유럽은 고위도라 도법 왜곡이 커서 겹침을 더 허용해야 후보가 생긴다.
    # 판게아에서 유럽과 북아메리카는 그린란드를 사이에 두고 붙어 있었다(로라시아).
    # 그 고위도 구간은 이 도법에서 양쪽이 다 부풀어 겹침을 피할 수 없으므로
    # 허용치를 크게 두고, 대신 파고드는 깊이를 최소화하도록 맡긴다.
    ("eu", (782, 107, -12), ["af", "na"], 34.0),
    # 아프리카 동해안(소말리아~케냐) 바깥. 마다가스카르 자리다.
    ("in", (693, 287, -38), ["af"]),
    # 아프리카 남쪽. 인도와도 맞닿는다.
    ("an", (610, 456, 10), ["af", "in"]),
    # 남극 북쪽, 인도 남동쪽. 남극과의 정합이 특히 잘 맞는다.
    ("oc", (758, 372, -40), ["an", "in"]),
]

# ---- shapes.js 읽기 -------------------------------------------------------
GEO = {}
for blk in open("shapes.js").read().split('{ id:"')[1:]:
    pid = blk[:blk.index('"')]
    d = re.search(r'd:"([^"]+)"', blk).group(1)
    rings = [np.array([[float(v) for v in pt.split(",")] for pt in sub.split("L")])
             for sub in d.lstrip("M").rstrip("Z").split("ZM")]
    GEO[pid] = {"rings": rings, "pts": np.vstack(rings)}


def xform(pts, x, y, rot):
    """SVG 의 translate(x,y) rotate(rot) 과 같은 변환."""
    t = math.radians(rot)
    c, s = math.cos(t), math.sin(t)
    return np.column_stack([x + pts[:, 0]*c - pts[:, 1]*s,
                            y + pts[:, 0]*s + pts[:, 1]*c])


def point_in(pts, rings):
    """광선 교차법. 여러 링을 XOR 해 다중 폴리곤을 함께 처리한다."""
    res = np.zeros(len(pts), bool)
    x, y = pts[:, 0:1], pts[:, 1:2]
    for r in rings:
        if len(r) < 3:
            continue
        r = np.vstack([r, r[:1]])
        y1, y2 = r[:-1, 1], r[1:, 1]
        x1, x2 = r[:-1, 0], r[1:, 0]
        straddle = (y1 > y) != (y2 > y)
        xi = (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1
        res ^= (straddle & (x < xi)).sum(1) % 2 == 1
    return res


class SDF:
    """이미 배치된 대륙들에 대한 부호 있는 거리장. 대륙 안쪽이 음수."""
    def __init__(self, outline, rings, pad=90.0, res=2.0):
        lo, hi = outline.min(0) - pad, outline.max(0) + pad
        self.res, self.lo = res, lo
        self.nx = int((hi[0] - lo[0]) / res) + 1
        self.ny = int((hi[1] - lo[1]) / res) + 1
        gx = lo[0] + np.arange(self.nx) * res
        gy = lo[1] + np.arange(self.ny) * res
        grid = np.stack(np.meshgrid(gx, gy, indexing="xy"), -1).reshape(-1, 2)
        d = np.full(len(grid), np.inf)
        for i in range(0, len(outline), 200):          # 메모리 아끼려고 잘라서
            chunk = outline[i:i+200]
            d = np.minimum(d, np.min(np.linalg.norm(
                grid[:, None, :] - chunk[None, :, :], axis=2), axis=1))
        d[point_in(grid, rings)] *= -1
        self.g = d.reshape(self.ny, self.nx)

    def sample(self, pts):
        ix = np.clip(((pts[:, 0] - self.lo[0]) / self.res).astype(int), 0, self.nx-1)
        iy = np.clip(((pts[:, 1] - self.lo[1]) / self.res).astype(int), 0, self.ny-1)
        return self.g[iy, ix]


def gap_cost(sdf, pts, k, max_over):
    """맞닿는 해안 구간의 평균 틈. 너무 깊이 파고들면 무한대(후보 제외)."""
    sd = sdf.sample(pts)
    deepest = -sd.min()
    if deepest > max_over:
        return float("inf")
    gap = np.sort(np.clip(sd, 0, None))[:k].mean()   # 맞닿는 해안 구간의 평균 틈
    return gap + OVER_W * max(0.0, deepest - SLACK)


def hug(pid, init, against, max_over=MAX_OVER, span=(17, 17, 8)):
    """출발 위치 근처에서만, 이미 배치된 대륙들에 촘촘히 붙인다."""
    outline = np.vstack([xform(GEO[a]["pts"], *TARGETS[a]) for a in against])
    rings = [xform(r, *TARGETS[a]) for a in against for r in GEO[a]["rings"]]
    sdf = SDF(outline, rings)
    pts = GEO[pid]["pts"]
    k = min(CONTACT_N, max(12, len(pts) // 4))

    best, bx, by, br = float("inf"), *init
    for step, rstep in ((3.0, 1.5), (1.0, 0.5), (0.4, 0.2)):   # 성긴 -> 촘촘한
        cx, cy, cr = bx, by, br
        rx, ry, rr = span if step > 2 else (step*4, step*4, rstep*4)
        for rot in np.arange(cr - rr, cr + rr + 1e-9, rstep):
            base = xform(pts, 0, 0, rot)
            for dx in np.arange(cx - rx, cx + rx + 1e-9, step):
                for dy in np.arange(cy - ry, cy + ry + 1e-9, step):
                    c = (gap_cost(sdf, base + (dx, dy), k, max_over)
                         + DRIFT_XY * math.hypot(dx - init[0], dy - init[1])
                         + DRIFT_R * abs(rot - init[2]))
                    if c < best:
                        best, bx, by, br = c, dx, dy, rot
    # 보고용으로는 벌점을 뺀 실제 틈을 다시 잰다
    raw = gap_cost(sdf, xform(pts, bx, by, br), k, max_over)
    return (bx, by, br), raw


TARGETS = {"af": (AF_POS[0], AF_POS[1], 0.0)}
for pid, init, against, *rest in SPEC:
    init = tuple(float(v) for v in init)
    fitted, raw = hug(pid, init, against, rest[0] if rest else MAX_OVER)
    TARGETS[pid] = fitted
    print("%s: 출발 (%.0f,%.0f,%.0f) -> 밀착 (%.0f,%.0f,%.1f)   맞닿는 면 평균 틈 %.2fpx"
          % (pid, *init, *fitted, raw), file=sys.stderr)

order = ["af", "sa", "na", "eu", "in", "oc", "an"]
block = "const TARGET = {\n" + "".join(
    "  %s: {x:%d, y:%d, rot:%.1f},\n" % (p, round(TARGETS[p][0]), round(TARGETS[p][1]),
                                        TARGETS[p][2]) for p in order) + "};"

html = open("index.html").read()
new, n = re.subn(r"const TARGET = \{.*?\n\};", lambda m: block, html, count=1, flags=re.S)
if n == 0:
    sys.exit("index.html 에서 TARGET 블록을 찾지 못했습니다")
open("index.html", "w").write(new)
print(block)
