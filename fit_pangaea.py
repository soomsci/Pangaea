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

SCALE  = 3.2                 # build_shapes.py 와 같아야 한다
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
    # 수업의 대표 장면이라 다른 대륙보다 공들여 맞춘다. 탐색을 넓히고 출발
    # 위치에 매어 두는 힘을 줄이며, 맞물림을 재는 점을 늘려 더 긴 구간이
    # 맞닿도록 한다.
    ("sa", (492, 362, -44), ["af"], MAX_OVER, (26, 26, 14), (0.02, 0.05), 70),
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
    GEO[pid] = {"rings": rings, "pts": np.vstack(rings),
                "cx": float(re.search(r"cx:(-?[\d.]+)", blk).group(1)),
                "cy": float(re.search(r"cy:(-?[\d.]+)", blk).group(1)),
                "proj": re.search(r'proj:"(\w+)"', blk).group(1)}


def xform(pts, x, y, rot):
    """SVG 의 translate(x,y) rotate(rot) 과 같은 변환."""
    t = math.radians(rot)
    c, s = math.cos(t), math.sin(t)
    return np.column_stack([x + pts[:, 0]*c - pts[:, 1]*s,
                            y + pts[:, 0]*s + pts[:, 1]*c])


def local(pid, lon, lat):
    """경위도 -> 조각 로컬 좌표. build_shapes.py 의 project 와 같은 식이어야 한다."""
    g = GEO[pid]
    if g["proj"] == "spole":
        r, a = (90 + lat) * SCALE, math.radians(lon)
        return r * math.sin(a) - g["cx"], r * math.cos(a) - g["cy"]
    return lon * SCALE - g["cx"], -lat * SCALE - g["cy"]


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


def hug(pid, init, against, max_over=MAX_OVER, span=(17, 17, 8),
        drift=(DRIFT_XY, DRIFT_R), contact_n=CONTACT_N):
    """출발 위치 근처에서만, 이미 배치된 대륙들에 촘촘히 붙인다."""
    outline = np.vstack([xform(GEO[a]["pts"], *TARGETS[a]) for a in against])
    rings = [xform(r, *TARGETS[a]) for a in against for r in GEO[a]["rings"]]
    sdf = SDF(outline, rings)
    pts = GEO[pid]["pts"]
    k = min(contact_n, max(12, len(pts) // 4))

    best, bx, by, br = float("inf"), *init
    for step, rstep in ((3.0, 1.5), (1.0, 0.5), (0.4, 0.2)):   # 성긴 -> 촘촘한
        cx, cy, cr = bx, by, br
        rx, ry, rr = span if step > 2 else (step*4, step*4, rstep*4)
        for rot in np.arange(cr - rr, cr + rr + 1e-9, rstep):
            base = xform(pts, 0, 0, rot)
            for dx in np.arange(cx - rx, cx + rx + 1e-9, step):
                for dy in np.arange(cy - ry, cy + ry + 1e-9, step):
                    c = (gap_cost(sdf, base + (dx, dy), k, max_over)
                         + drift[0] * math.hypot(dx - init[0], dy - init[1])
                         + drift[1] * abs(rot - init[2]))
                    if c < best:
                        best, bx, by, br = c, dx, dy, rot
    # 보고용으로는 벌점을 뺀 실제 틈을 다시 잰다
    raw = gap_cost(sdf, xform(pts, bx, by, br), k, max_over)
    return (bx, by, br), raw



# ---- 화석 띠 --------------------------------------------------------------
# 같은 화석이 여러 대륙에서 나온다는 증거는 점 몇 개로는 "이어진다"가 읽히지 않는다.
# 그래서 띠를 판게아 위에 하나로 그린 뒤 대륙별로 잘라 각 조각의 로컬 좌표로 저장한다.
# 학생이 판게아를 맞추면 잘렸던 띠가 제자리에서 다시 이어진다.
#
# 좌표는 (대륙, 경도, 위도). 실제 화석 산지를 하나하나 옮긴 것이 아니라
# 그 화석이 나오는 지역을 지나가도록 잡은 개략 경로다.
BANDS = [
    ("메소사우루스", "#8a3fa8", 17, [
        # 민물 파충류. 남대서양 양쪽 좁은 띠 하나로만 나온다.
        [("sa", -58, -33), ("sa", -48, -25), ("sa", -42, -20),
         ("af", 12, -13), ("af", 18, -22), ("af", 24, -31)],
    ]),
    ("키노그나투스", "#1f7a5a", 17, [
        # 육상 파충류. 남아메리카 중부와 아프리카 남부.
        [("sa", -67, -36), ("sa", -57, -31), ("sa", -47, -27),
         ("af", 17, -28), ("af", 27, -21), ("af", 36, -13)],
    ]),
    ("리스트로사우루스", "#c2621d", 17, [
        # 육상 파충류. 아프리카 남부에서 인도로, 그리고 남극으로.
        [("af", 30, -27), ("af", 38, -14), ("af", 46, -3), ("in", 72, 16), ("in", 82, 23)],
        [("af", 26, -32), ("an", -10, -72), ("an", 15, -71), ("an", 40, -70), ("an", 60, -69)],
    ]),
    ("글로소프테리스", "#2f6d8c", 21, [
        # 양치식물. 남반구 다섯 대륙 전부에서 나온다. 가장 넓은 증거.
        [("sa", -60, -39), ("sa", -50, -31), ("af", 15, -33), ("af", 30, -26),
         ("af", 44, -6), ("in", 74, 15), ("in", 84, 21)],
        [("af", 20, -34), ("an", -20, -74), ("an", 10, -75), ("an", 45, -73),
         ("an", 80, -70), ("an", 105, -68), ("oc", 118, -31), ("oc", 134, -33),
         ("oc", 148, -29)],
    ]),
]


HOPS = []


def contact(pa, pb, near, radius=170.0):
    """두 대륙 외곽선에서 near 근처의 가장 가까운 점 쌍(= 맞닿는 지점).

    띠가 대륙을 건너갈 때 이 지점으로 지나가게 하면, 손으로 좌표를 맞추지
    않아도 양쪽 조각이 이음매에서 정확히 만난다."""
    A = xform(GEO[pa]["pts"], *TARGETS[pa])
    B = xform(GEO[pb]["pts"], *TARGETS[pb])
    A = A[np.linalg.norm(A - near, axis=1) < radius]
    B = B[np.linalg.norm(B - near, axis=1) < radius]
    if not len(A) or not len(B):
        return None
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return A[i], B[j]


# ---- 고생대 조산대 --------------------------------------------------------
# 갈라진 대륙 양쪽에서 같은 시기·같은 구조의 산맥이 이어진다. 화석 띠와 같은
# 방식으로 판게아 위에 그린 뒤 대륙별로 자른다.
BELTS = [
    ("고생대 조산대", "#6b4423", 13, [
        # 애팔래치아 - 칼레도니아. 대륙이동설의 대표 증거.
        # 북서아프리카(모리타니데스) - 애팔래치아 - 아일랜드/스코틀랜드 - 스칸디나비아
        [("af", -10, 21), ("af", -8, 29),
         ("na", -85, 33), ("na", -79, 38), ("na", -71, 43), ("na", -59, 47),
         ("eu", -8, 54), ("eu", 1, 58), ("eu", 11, 64), ("eu", 18, 69)],
        # 동그린란드 칼레도니아
        [("na", -44, 61), ("na", -33, 69), ("na", -26, 75)],
        # 케이프 습곡대(남아프리카) - 시에라 데 라 벤타나(아르헨티나)
        [("sa", -63, -38), ("sa", -58, -37), ("af", 19, -34), ("af", 27, -33)],
    ]),
]


def chain_world(chain, step=1.5):
    """(대륙, 경도, 위도) 목록 -> 판게아 화면 좌표의 촘촘한 점열.

    같은 대륙 안의 구간은 경위도에서 보간한 뒤 투영한다. 화면 좌표에서 곧장
    직선을 그으면 남극처럼 극 중심 도법을 쓰는 대륙에서 띠가 헤어핀으로 꺾인다.
    대륙이 바뀌는 구간만 화면 좌표에서 직선으로 잇는다."""
    out = []
    for (p1, lo1, la1), (p2, lo2, la2) in zip(chain[:-1], chain[1:]):
        if p1 == p2:
            n = max(2, int(math.dist((lo1, la1), (lo2, la2)) * SCALE / step))
            pts = [local(p1, lo1 + (lo2-lo1)*i/n, la1 + (la2-la1)*i/n) for i in range(n)]
            out.extend(xform(np.array(pts), *TARGETS[p1]))
        else:
            a = xform(np.array([local(p1, lo1, la1)]), *TARGETS[p1])[0]
            b = xform(np.array([local(p2, lo2, la2)]), *TARGETS[p2])[0]
            # 두 대륙이 실제로 맞닿는 지점을 거쳐 가도록 경로를 꺾는다
            via = contact(p1, p2, (a + b) / 2)
            legs = [(a, via[0]), (via[0], via[1]), (via[1], b)] if via else [(a, b)]
            HOPS.append((p1, p2, math.dist(*legs[1]) if via else math.dist(a, b)))
            for u, v in legs:
                n = max(2, int(math.dist(u, v) / step))
                out.extend(u + (v - u) * i / n for i in range(n))
    pid, lo, la = chain[-1]
    out.append(xform(np.array([local(pid, lo, la)]), *TARGETS[pid])[0])
    return np.array(out)


def to_local(pts, x, y, rot):
    """xform 의 역변환. 판게아 화면 좌표 -> 조각 로컬 좌표."""
    t = math.radians(-rot)
    c, s = math.cos(t), math.sin(t)
    q = pts - np.array([x, y])
    return np.column_stack([q[:, 0]*c - q[:, 1]*s, q[:, 0]*s + q[:, 1]*c])


# ---- 갈라지는 틈에 생기는 해령(발산 경계) ------------------------------
# 대륙이 갈라지면 그 사이에서 새 해양 지각이 만들어진다. 판게아에서 맞닿아
# 있던 지점 쌍을 뽑아 두면 두 점의 중간을 이어 해령을 그릴 수 있다. 해령은
# 양쪽으로 대칭으로 벌어지므로 벌어진 틈의 한가운데가 맞다.
#
# 오늘날의 판 경계를 판게아에 그리면 틀린다. 판 경계는 대륙에 붙어 있는 것이
# 아니라 생겼다 사라진다. 여기서 만드는 것은 판게아가 갈라지며 새로 생긴
# 발산 경계뿐이고, 오늘날의 판 경계는 애니메이션 끝에서 실제 데이터로 겹친다.
RIFT_PAIRS = [
    ("sa", "af"),   # 남대서양
    ("na", "af"),   # 중앙대서양
    ("na", "eu"),   # 북대서양
    ("af", "an"),   # 인도양 서쪽
    ("af", "in"),   # 아프리카 동해안 - 인도
    ("an", "oc"),   # 남극 - 오스트레일리아
]


def ridge_pairs(pa, pb, thresh=45.0, n=9):
    """맞닿은 구간을 따라 고르게 퍼진 지점 쌍을 뽑는다.

    맞닿은 점들을 그냥 이으면 선이 톱니처럼 튄다. 접촉면이 가장 길게 뻗은
    방향(가장 먼 두 중점을 잇는 축)으로 투영해 구간을 나누고, 구간마다 가장
    잘 맞닿은 쌍 하나씩만 골라 순서대로 잇는다."""
    A = xform(GEO[pa]["pts"], *TARGETS[pa])
    B = xform(GEO[pb]["pts"], *TARGETS[pb])
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    j = D.argmin(1)
    gap = D[np.arange(len(A)), j]
    idx = np.where(gap < thresh)[0]
    if len(idx) < 4:
        return None

    # 중점이 어느 한쪽 대륙 안이면 선이 육지를 가로지른다. 버린다.
    mid = (A[idx] + B[j[idx]]) / 2
    inside = (point_in(mid, [xform(r, *TARGETS[pa]) for r in GEO[pa]["rings"]]) |
              point_in(mid, [xform(r, *TARGETS[pb]) for r in GEO[pb]["rings"]]))
    idx, mid = idx[~inside], mid[~inside]
    if len(idx) < 4:
        return None

    # 가장 멀리 떨어진 두 중점을 잇는 축 = 접촉면이 뻗은 방향
    M = np.linalg.norm(mid[:, None, :] - mid[None, :, :], axis=2)
    p, q = np.unravel_index(M.argmax(), M.shape)
    axis = mid[q] - mid[p]
    span = np.linalg.norm(axis)
    if span < 60:                      # 접촉면이 점에 가까우면 해령을 그릴 게 없다
        return None
    t = (mid - mid[p]) @ (axis / span) / span      # 0~1

    pick = []
    for k in range(n):                 # 구간마다 가장 잘 맞닿은 쌍 하나씩
        sel = np.where((t >= k / n) & (t < (k + 1) / n + (1e-9 if k == n-1 else 0)))[0]
        if len(sel):
            pick.append(idx[sel[np.argmin(gap[idx[sel]])]])
    if len(pick) < 3:
        return None
    return ([GEO[pa]["pts"][i] for i in pick],
            [GEO[pb]["pts"][j[i]] for i in pick])


def split_bands(table):
    """판게아 위의 띠를 대륙별 조각으로 자른다."""
    out = []
    for name, color, width, chains in table:
        parts = []
        HOPS.clear()
        for chain in chains:
            world = chain_world(chain)
            for pid in GEO:
                rings = [xform(r, *TARGETS[pid]) for r in GEO[pid]["rings"]]
                inside = point_in(world, rings)
                run = []
                for i, ins in enumerate(list(inside) + [False]):
                    if ins:
                        run.append(i)
                    elif run:
                        seg_w = world[run]
                        arc = float(np.linalg.norm(np.diff(seg_w, axis=0), axis=1).sum()) \
                              if len(run) > 1 else 0.0
                        if arc >= 15:           # 모서리를 스친 자투리는 버린다
                            seg = to_local(world[run], *TARGETS[pid])
                            seg = seg[::max(1, len(seg)//14)]   # 점 수를 줄인다
                            parts.append((pid, np.round(seg, 1)))
                        run = []
        far = [h for h in HOPS if h[2] > 14]
        if far:
            print("  경고 %s: 대륙 사이를 %s 건너뜀 — 띠가 끊겨 보인다"
                  % (name, ", ".join("%s->%s %.0fpx" % h for h in far)), file=sys.stderr)
        out.append((name, color, width, parts))
    return out


TARGETS = {"af": (AF_POS[0], AF_POS[1], 0.0)}
for pid, init, against, *rest in SPEC:
    init = tuple(float(v) for v in init)
    fitted, raw = hug(pid, init, against, *rest)
    TARGETS[pid] = fitted
    print("%s: 출발 (%.0f,%.0f,%.0f) -> 밀착 (%.0f,%.0f,%.1f)   맞닿는 면 평균 틈 %.2fpx"
          % (pid, *init, *fitted, raw), file=sys.stderr)

ridges = []
for pa, pb in RIFT_PAIRS:
    r = ridge_pairs(pa, pb)
    if r is None:
        print("  경고: %s-%s 접촉면이 해령을 그릴 만큼 뻗어 있지 않다"
              % (pa, pb), file=sys.stderr)
        continue
    ridges.append((pa, pb, r[0], r[1]))
for pa, pb, la, lb in ridges:
    a = xform(np.array(la), *TARGETS[pa])
    b = xform(np.array(lb), *TARGETS[pb])
    gap = np.linalg.norm(a - b, axis=1)
    mid = (a + b) / 2
    span = float(np.linalg.norm(np.diff(mid, axis=0), axis=1).sum())
    print("해령 %s-%s: 점 %d개, 판게아에서의 틈 %.0f~%.0fpx, 길이 %.0fpx"
          % (pa, pb, len(la), gap.min(), gap.max(), span), file=sys.stderr)

ridge_block = "const RIDGES = [\n" + "".join(
    '  {a:"%s", b:"%s", pa:[%s], pb:[%s]},\n'
    % (pa, pb, ",".join("[%g,%g]" % (p[0], p[1]) for p in la),
       ",".join("[%g,%g]" % (p[0], p[1]) for p in lb))
    for pa, pb, la, lb in ridges) + "];"

bands = split_bands(BANDS)
belts = split_bands(BELTS)
fossil_block = "const FOSSILS = [\n" + "".join(
    '  {name:"%s", color:"%s", w:%d, parts:[\n%s  ]},\n'
    % (name, color, width, "".join(
        '    ["%s",[%s]],\n' % (pid, ",".join("[%g,%g]" % tuple(p) for p in seg))
        for pid, seg in parts))
    for name, color, width, parts in bands) + "];"
def emit(table, var):
    return "const %s = [\n" % var + "".join(
        '  {name:"%s", color:"%s", w:%d, parts:[\n%s  ]},\n'
        % (name, color, width, "".join(
            '    ["%s",[%s]],\n' % (pid, ",".join("[%g,%g]" % tuple(p) for p in seg))
            for pid, seg in parts))
        for name, color, width, parts in table) + "];"

belt_block = emit(belts, "BELTS")
for name, _, _, parts in bands + belts:
    print("%s: %d개 대륙에 걸쳐 %d조각"
          % (name, len({p for p, _ in parts}), len(parts)), file=sys.stderr)

order = ["af", "sa", "na", "eu", "in", "oc", "an"]
block = "const TARGET = {\n" + "".join(
    "  %s: {x:%d, y:%d, rot:%.1f},\n" % (p, round(TARGETS[p][0]), round(TARGETS[p][1]),
                                        TARGETS[p][2]) for p in order) + "};"

html = open("index.html").read()
for pat, new_block, what in ((r"const TARGET = \{.*?\n\};", block, "TARGET"),
                             (r"const FOSSILS = \[.*?\n\];", fossil_block, "FOSSILS"),
                             (r"const RIDGES = \[.*?\n\];", ridge_block, "RIDGES"),
                             (r"const BELTS = \[.*?\n\];", belt_block, "BELTS")):
    html, n = re.subn(pat, lambda m: new_block, html, count=1, flags=re.S)
    if n == 0:
        sys.exit("index.html 에서 %s 블록을 찾지 못했습니다" % what)
open("index.html", "w").write(html)
print(block)
