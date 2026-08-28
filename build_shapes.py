#!/usr/bin/env python3
"""1회용: 외부 지리 데이터 -> 시뮬레이션이 쓰는 JS 좌표.

1) Natural Earth 110m 국가 경계 -> 판게아 퍼즐용 대륙 조각 SVG path (표준출력)
2) PB2002(Bird 2003) -> 오늘날의 판 경계. index.html 의 PLATES 블록에 직접 써 넣는다.

실행: python3 build_shapes.py > shapes.js
"""
import json, math, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_110m_admin_0_countries.geojson")
CACHE = os.path.join(HERE, ".ne110.geojson")

# 오늘날의 판 경계. Bird(2003) PB2002. 각 구간에 경계 유형이 들어 있다.
PLATES_URL = ("https://raw.githubusercontent.com/fraxen/tectonicplates/"
              "master/GeoJSON/PB2002_steps.json")
PLATES_CACHE = os.path.join(HERE, ".pb2002_steps.json")

# Bird 의 구간 분류를 중학교에서 쓰는 세 가지로 묶는다.
#   OSR 해령, CRB 대륙 열곡 / SUB 섭입대, OCB·CCB 충돌 / OTF·CTF 변환 단층
STEP_KIND = {"OSR":"발산", "CRB":"발산", "SUB":"수렴", "OCB":"수렴", "CCB":"수렴",
             "OTF":"보존", "CTF":"보존"}
PLATE_TOL = 0.7        # 판 경계 단순화 허용 오차(도)

SCALE = 3.2          # 경도 1도 = 3.2px  -> 세계 지도 1152 x 576
TOLERANCE = 0.30     # Douglas-Peucker 허용 오차 (도)
MIN_AREA = 2.0       # 이보다 작은 폴리곤(섬)은 버림 (제곱도)

# 인도 판은 판게아에서 아프리카/남극 쪽에 붙어 있었으므로 유라시아에서 떼어낸다.
INDIA = {"India", "Sri Lanka", "Bangladesh", "Nepal", "Bhutan"}

# 해외 영토 걸러내기. 예: 프랑스(CONTINENT=Europe) 폴리곤에 프랑스령 기아나가
# 들어 있어 유라시아 조각이 남아메리카 위에 파란 점을 찍는다.
# 조각별로 허용할 경도 범위(되감기 후 기준)를 둔다.
LON_CLIP = {"eu": (-32, 210), "af": (-30, 65), "na": (-180, -8)}

# 조각 정의: id -> (한글 이름, CONTINENT 값, 경도 되감기 중심, 도법)
# 경도 되감기 중심: 날짜변경선을 넘는 조각(러시아 축치, 알류샨, 피지)이
# 지도 반대편으로 튀는 것을 막는다.
# 도법: "cyl" = 정거원통(교과서 세계지도와 동일).
#       "spole" = 남극점 중심 방위도법. 남극은 원통도법에서 가로 1146px로
#       뭉개져 퍼즐 조각이 안 되므로 이 조각만 예외.
PIECES = [
    ("na",   "북아메리카",   {"North America"}, -100, "cyl"),
    ("sa",   "남아메리카",   {"South America"},  -60, "cyl"),
    ("af",   "아프리카",     {"Africa"},          20, "cyl"),
    ("eu",   "유라시아",     {"Asia", "Europe"},  80, "cyl"),
    ("in",   "인도",         set(),               80, "cyl"),
    ("oc",   "오스트레일리아", {"Oceania"},       140, "cyl"),
    ("an",   "남극",         {"Antarctica"},       0, "spole"),
]


def fetch(url=URL, cache=CACHE):
    if not os.path.exists(cache):
        print("다운로드 중: %s" % os.path.basename(cache), file=sys.stderr)
        urllib.request.urlretrieve(url, cache)
    with open(cache) as f:
        return json.load(f)


def rings(geom):
    """Polygon / MultiPolygon -> 바깥 링 목록 (구멍은 버린다: 카스피해 정도)."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    if geom["type"] == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    return []


def area(ring):
    """신발끈 공식. 제곱도 단위 절댓값."""
    s = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def _dp(pts, tol):
    """열린 선분에 대한 Douglas-Peucker. 재귀 대신 스택."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        ax, ay = pts[lo]
        bx, by = pts[hi]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy)
        best, bi = -1.0, lo
        for i in range(lo + 1, hi):
            px, py = pts[i]
            if norm < 1e-12:          # 시작=끝이면 점까지의 거리로
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs(dy * (px - ax) - dx * (py - ay)) / norm
            if d > best:
                best, bi = d, i
        if best > tol:
            keep[bi] = True
            stack.append((lo, bi))
            stack.append((bi, hi))
    return [p for p, k in zip(pts, keep) if k]


def simplify(ring, tol):
    """닫힌 링 전용. 링을 그냥 DP에 넣으면 시작=끝이라 전체가 뭉개지므로,
    시작점에서 가장 먼 점을 기준으로 두 조각으로 갈라 각각 단순화한다."""
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 4:
        return ring
    ax, ay = ring[0]
    far = max(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - ax,
                                                         ring[i][1] - ay))
    first = _dp(ring[:far + 1], tol)
    second = _dp(ring[far:] + [ring[0]], tol)
    return first[:-1] + second[:-1]


def unwrap(ring, center):
    """경도를 center 기준 ±180 안으로 되감아 조각이 지도 양끝으로 찢기지 않게."""
    return [(center + ((lon - center + 180) % 360 - 180), lat) for lon, lat in ring]


def project(lon, lat, proj):
    """경위도 -> px. 원점은 (lon=0, lat=0) 또는 남극점."""
    if proj == "spole":
        r = (90 + lat) * SCALE          # 남극점에서의 각거리
        a = math.radians(lon)
        return r * math.sin(a), r * math.cos(a)   # 본초자오선이 아래쪽
    return lon * SCALE, -lat * SCALE


def build():
    data = fetch()
    out = []
    for pid, name, continents, lon_center, proj in PIECES:
        kept = []
        for feat in data["features"]:
            props = feat["properties"]
            in_india = props["NAME"] in INDIA
            if pid == "in":
                if not in_india:
                    continue
            else:
                if in_india or props["CONTINENT"] not in continents:
                    continue
            for ring in rings(feat["geometry"]):
                ring = unwrap(ring, lon_center)
                if area(ring) < MIN_AREA:
                    continue
                lo, hi = LON_CLIP.get(pid, (-360, 360))
                mid = sum(lon for lon, _ in ring) / len(ring)
                if not (lo <= mid <= hi):
                    continue
                ring = simplify(ring, TOLERANCE)
                kept.append([project(lon, lat, proj) for lon, lat in ring])

        if not kept:
            print(f"경고: {pid} 조각이 비었습니다", file=sys.stderr)
            continue

        xs = [x for ring in kept for x, _ in ring]
        ys = [y for ring in kept for _, y in ring]
        # 회전 중심 = 바운딩 박스 중심 (조각을 제자리에서 돌리기 위해)
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        subpaths = []
        for ring in kept:
            pts = [f"{x - cx:.1f},{y - cy:.1f}" for x, y in ring]
            subpaths.append("M" + "L".join(pts) + "Z")

        # 세계 지도(현재) 위치. 지도 원점(lon -180, lat 90)이 화면 (0,0).
        px = 180 * SCALE + cx
        py = 90 * SCALE + cy
        if proj == "spole":
            px, py = 180 * SCALE, 90 * SCALE + 66 * SCALE  # 남극은 아래 중앙에

        out.append({
            "id": pid, "name": name, "proj": proj,
            "cx": round(cx, 1), "cy": round(cy, 1),
            "present": {"x": round(px, 1), "y": round(py, 1), "rot": 0},
            "d": "".join(subpaths),
        })
    return out


def build_plates():
    """PB2002 구간 -> 유형별 폴리라인. 화면 좌표(정거원통도법)로 투영해 둔다."""
    data = fetch(PLATES_URL, PLATES_CACHE)
    # (경계 이름, 유형)별로 순번대로 모은다
    groups = {}
    for f in data["features"]:
        p = f["properties"]
        kind = STEP_KIND.get(p["STEPCLASS"])
        if kind is None:
            continue
        groups.setdefault((p["PLATEBOUND"], kind), []).append(
            (p["SEQNUM"], f["geometry"]["coordinates"]))

    out = {k: [] for k in ("발산", "수렴", "보존")}
    for (_, kind), steps in groups.items():
        steps.sort(key=lambda s: s[0])
        chain = []
        for _, coords in steps:
            pts = [(c[0], c[1]) for c in coords]
            # 이어지지 않으면 끊는다. 날짜변경선을 넘는 구간도 여기서 끊긴다.
            if chain and math.dist(chain[-1], pts[0]) > 3.0:
                if len(chain) > 2:
                    out[kind].append(simplify_open(chain))
                chain = []
            chain.extend(pts if not chain else pts[1:])
        if len(chain) > 2:
            out[kind].append(simplify_open(chain))

    lines = {k: [] for k in out}
    for kind, chains in out.items():
        for ch in chains:
            seg = []
            for lon, lat in ch:
                if seg and abs(lon - seg[-1][0]) > 180:   # 날짜변경선
                    if len(seg) > 1:
                        lines[kind].append(seg)
                    seg = []
                seg.append((lon, lat))
            if len(seg) > 1:
                lines[kind].append(seg)
    return lines


def simplify_open(pts):
    """열린 선분 단순화. 링이 아니므로 _dp 를 그대로 쓴다."""
    return _dp(pts, PLATE_TOL)


def write_plates(lines):
    """index.html 의 PLATES 블록을 갈아 끼운다."""
    body = []
    for kind in ("발산", "수렴", "보존"):
        segs = ",".join(
            "[" + ",".join("%.0f,%.0f" % ((lon + 180) * SCALE, (90 - lat) * SCALE)
                           for lon, lat in seg) + "]"
            for seg in lines[kind])
        body.append('  {kind:"%s", lines:[%s]},\n' % (kind, segs))
    block = "const PLATES = [\n" + "".join(body) + "];"
    path = os.path.join(HERE, "index.html")
    html = open(path).read()
    html, n = re.subn(r"const PLATES = \[.*?\n\];", lambda m: block, html,
                      count=1, flags=re.S)
    if n == 0:
        sys.exit("index.html 에서 PLATES 블록을 찾지 못했습니다")
    open(path, "w").write(html)
    for kind in lines:
        print("판 경계 %s: 선 %d개, 점 %d개"
              % (kind, len(lines[kind]), sum(len(s) for s in lines[kind])),
              file=sys.stderr)


if __name__ == "__main__":
    pieces = build()
    print(f"// build_shapes.py 생성. Natural Earth 110m, scale={SCALE}, "
          f"tol={TOLERANCE}, minArea={MIN_AREA}")
    print(f"const SCALE = {SCALE};")
    print("const PIECES = [")
    for p in pieces:
        print(f'  {{ id:"{p["id"]}", name:"{p["name"]}", proj:"{p["proj"]}", '
              f'cx:{p["cx"]}, cy:{p["cy"]},')
        print(f'    present:{{x:{p["present"]["x"]},y:{p["present"]["y"]},rot:0}},')
        print(f'    d:"{p["d"]}" }},')
    print("];")
    total = sum(len(p["d"]) for p in pieces)
    print(f"// path 총 {total:,}자", file=sys.stderr)
    write_plates(build_plates())
