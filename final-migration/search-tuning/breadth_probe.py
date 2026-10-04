#!/usr/bin/env python3
"""KataGo 根节点搜索广度探针（走实战 genmove 路径）。

原理
----
PlayUtils::printGenmoveLog（cpp/program/playutils.cpp:1015-1046）在每次
genmove / kata-search 结束后，把根节点的前 10 个子节点写进 GTP 日志：

    Root visits: 20000
    NN avg batch size: 11.98
    PV: Q3 ...
    Tree:
    ---B(^)---
    Q3  : T ... LCB -30.06c P 24.48% WF 53517.5 PSV 53442 N   19343  --  Q3 D16 ...
    Q4  : T ...                                             N     272  --  ...

本脚本用固定 maxVisits 跑若干局面，解析该段，算根访问分布的广度指标。

⚠️ 三个必须知道的坑（都已在源码里核对过）
  1. printTree 的子节点排序依据是 **playSelectionValue（PSV）**，不是访问量
     （cpp/search/searchdata.cpp 的 `operator<`：先按 PSV 降序，再按 numVisits）。
     所以「日志里前 10 个」= PSV 前 10，不保证是访问量前 10。
     本脚本在解析后**重新按 visits 降序排**，并用 shown_share 衡量
     「前 10 覆盖了根访问量的多少」；shown_share≈1 时 top1 才可信。
     useLcbForSelection=true 会改变 PSV 的含义 ⇒ 换臂后尤其要盯 shown_share。
  2. 只列前 10 个点（maxChildrenToShow(10)）⇒ 指标是「前 10 的分布」。
     要绝对熵需改源码（Tier 3）。
  3. 必须用 genmove / kata-search，不要用 kata-analyze —— 后者走 analysisParams，
     是另一套参数（含 analysisWideRootNoise），与实战无关。

⚠️ -config 的绝对路径限制（本项目踩过的坑）
  KataGo 的**第一个** -config 走 ConfigParser::initialize()，绝对路径可以；
  但**第二个及以后**走 ConfigParser::overrideKeys() → processIncludedFile()，
  后者对 baseDir 以 '/' 开头的一律抛
      ConfigParsingError: Absolute paths in the included files are not supported yet
  （cpp/core/config_parser.cpp:102-105）
  因此本脚本把**所有** -config 都换算成相对 cwd 的路径（relpath，可含 ../）。
  -model / katago 可执行文件不受此限制，保持绝对路径。

用法
----
  python3 breadth_probe.py --katago ./bin/katago-xxx --model models/x.bin.gz \\
      --base-config config/base.cfg [--override-config config/search-tier0.cfg] \\
      --visits 20000 --repeats 3 [--tag armB] [--json out.json]
  # 只跑部分局面 / 加临时参数
  ... --only midgame8 --extra-override "cpuctExploration=1.4"

纪律（沿用本项目既有约定）
  * 连续测量有 1.7-1.9% 单调热漂移 ⇒ 同轮配对、每轮轮换臂顺序。
    本脚本只负责单臂单轮的采集；配对与轮换由外层驱动负责。
  * 丢弃每个 session 的首轮（GPU 冷启动偏高约 1.6%）—— 用 --repeats 多跑，
    分析时丢第一个。
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

MOVE_LINE_RE = re.compile(r'^\s*([A-Za-z]\d+)\s*:')
VISITS_RE = re.compile(r'\sN\s+(\d+)\s')
ROOT_VISITS_RE = re.compile(r'^Root visits:\s*(\d+)', re.M)
AVG_BATCH_RE = re.compile(r'^NN avg batch size:\s*([\d.]+)', re.M)
NN_ROWS_RE = re.compile(r'^NN rows:\s*(\d+)', re.M)
TIME_TAKEN_RE = re.compile(r'^Time taken:\s*([\d.]+)', re.M)

DEFAULT_POSITIONS = [
    ("empty", []),
    ("opening4", [("B", "Q16"), ("W", "D4"), ("B", "Q4"), ("W", "D16")]),
    ("midgame8", [("B", "Q16"), ("W", "D4"), ("B", "Q4"), ("W", "D16"),
                  ("B", "R14"), ("W", "C3"), ("B", "F17"), ("W", "D18")]),
]


def parse_tree_block(text):
    """从日志文本中取最后一个 Tree: 段，返回 [(move, visits), ...]。"""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "Tree:"]
    if not starts:
        return []
    out = []
    for ln in lines[starts[-1] + 1:]:
        if ln.strip() == "" or ln.startswith("-----"):
            break
        m = MOVE_LINE_RE.match(ln)
        if not m:
            continue
        v = VISITS_RE.search(ln)
        if not v:
            continue
        out.append((m.group(1), int(v.group(1))))
    return out


def metrics(pairs, root_visits):
    """由根子节点访问量算广度指标。pairs 已按 visits 降序。"""
    total = sum(v for _, v in pairs)
    if total <= 0:
        return None
    ps = [v / total for _, v in pairs]
    top1 = ps[0]
    H = -sum(p * math.log(p) for p in ps if p > 0)
    Hnorm = H / math.log(len(ps)) if len(ps) > 1 else 0.0
    cum, w90 = 0.0, 0
    for p in ps:
        cum += p
        w90 += 1
        if cum >= 0.90:
            break
    # top-k 覆盖
    def topk(k):
        return round(sum(ps[:k]), 5)
    return {
        "root_visits": root_visits,
        "visits": total,
        "shown": len(pairs),
        "shown_share": round(total / root_visits, 5) if root_visits else None,
        "top1": round(top1, 5),
        "top2": round(ps[1], 5) if len(ps) > 1 else 0.0,
        "top3_share": topk(3),
        "top5_share": topk(5),
        "entropy": round(H, 5),
        "entropy_norm": round(Hnorm, 5),
        "w90": w90,
        "n_ge_1pct": sum(1 for p in ps if p >= 0.01),
        "n_ge_0p1pct": sum(1 for p in ps if p >= 0.001),
        "top_moves": [[m, v, round(v / total, 5)] for m, v in pairs[:8]],
    }


def run_position(args, moves, logdir):
    """跑一个局面，返回 (metrics, None) 或 (None, err)。"""
    cmds = ["boardsize 19", "komi 7.5"]
    if args.rules:
        cmds.append("kata-set-rules " + args.rules)
    cmds.append("kata-set-param maxVisits %d" % args.visits)
    for color, mv in moves:
        cmds.append("play %s %s" % (color, mv))
    next_pla = "B" if len(moves) % 2 == 0 else "W"
    cmds.append("kata-search %s" % next_pla)   # 与 genmove 同一套参数，但不落子
    cmds.append("quit")

    shutil.rmtree(logdir, ignore_errors=True)

    # -config 必须用相对路径（见文件头说明）。第一个也不能绝对，统一处理最省心。
    cfg_args = []
    for c in args.base_config:
        cfg_args += ["-config", os.path.relpath(c, args.cwd)]
    for c in args.override_config:
        cfg_args += ["-config", os.path.relpath(c, args.cwd)]

    # 强制日志落点，保证不依赖 base cfg 里的 logDir 相对路径。
    ov = "logDir=%s,logAllGTPCommunication=true" % logdir
    if args.extra_override:
        ov += "," + args.extra_override

    cmd = ([args.katago, "gtp"] + cfg_args +
           ["-override-config", ov, "-model", args.model])

    t0 = time.time()
    try:
        p = subprocess.run(cmd, input="\n".join(cmds) + "\n",
                           cwd=args.cwd, capture_output=True, text=True,
                           timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout(%ds)" % args.timeout
    elapsed = time.time() - t0

    if not os.path.isdir(logdir):
        return None, "no logdir; stderr tail: " + (p.stderr or "")[-500:]
    logs = sorted(os.path.join(logdir, f) for f in os.listdir(logdir))
    if not logs:
        return None, "empty logdir; stderr tail: " + (p.stderr or "")[-500:]
    with open(logs[-1], "r", errors="replace") as fh:
        text = fh.read()

    pairs = parse_tree_block(text)
    if not pairs:
        return None, "no Tree block; stderr tail: " + (p.stderr or "")[-500:]

    rm = ROOT_VISITS_RE.search(text)
    root_visits = int(rm.group(1)) if rm else sum(v for _, v in pairs)

    pairs.sort(key=lambda t: -t[1])          # 按 visits 重排（日志原序是 PSV）
    m = metrics(pairs, root_visits)
    if m is None:
        return None, "zero visits"
    bm = AVG_BATCH_RE.search(text)
    m["avg_batch"] = float(bm.group(1)) if bm else None
    nm = NN_ROWS_RE.search(text)
    m["nn_rows"] = int(nm.group(1)) if nm else None
    tm = TIME_TAKEN_RE.search(text)
    m["secs"] = float(tm.group(1)) if tm else round(elapsed, 2)
    if m["nn_rows"] and m["secs"]:
        m["nnEvals_per_s"] = round(m["nn_rows"] / m["secs"], 1)
    return m, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-config", action="append", required=True)
    ap.add_argument("--override-config", action="append", default=[])
    ap.add_argument("--extra-override", default="", help="附加 key=value,key=value")
    ap.add_argument("--visits", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--rules", default="japanese")
    ap.add_argument("--only", default="", help="逗号分隔的局面名，默认全部")
    ap.add_argument("--json", default="")
    ap.add_argument("--cwd", default="", help="子进程工作目录，默认当前目录")
    ap.add_argument("--logdir", default="", help="日志目录，默认 <cwd>/gtp_logs")
    ap.add_argument("--keep-logdir", action="store_true")
    args = ap.parse_args()

    # 只有 katago / model 用绝对路径；config 必须相对（见文件头）
    args.katago = os.path.abspath(args.katago)
    args.model = os.path.abspath(args.model)
    args.base_config = [os.path.abspath(c) for c in args.base_config]
    args.override_config = [os.path.abspath(c) for c in args.override_config]
    args.cwd = os.path.abspath(args.cwd) if args.cwd else os.getcwd()
    if not os.path.isdir(args.cwd):
        print("cwd 不存在: " + args.cwd, file=sys.stderr)
        return 2

    logdir = args.logdir or os.path.join(args.cwd, "gtp_logs")
    logdir = os.path.abspath(logdir)

    if args.json:
        args.json = os.path.abspath(args.json)

    # 校验：所有 config 必须能相对 cwd 表达（POSIX 下 relpath 恒为相对，稳妥起见查一下）
    for c in args.base_config + args.override_config:
        rel = os.path.relpath(c, args.cwd)
        if rel.startswith("/"):
            print("无法把 %s 表达为相对 %s 的路径" % (c, args.cwd), file=sys.stderr)
            return 2

    positions = DEFAULT_POSITIONS
    if args.only:
        want = set(x.strip() for x in args.only.split(",") if x.strip())
        positions = [p for p in DEFAULT_POSITIONS if p[0] in want]
        if not positions:
            print("--only 没匹配到局面", file=sys.stderr)
            return 2

    results = {}
    for rep in range(args.repeats):
        for name, moves in positions:
            m, err = run_position(args, moves, logdir)
            key = "%s#r%d" % (name, rep + 1)
            if m is None:
                results[key] = {"error": err}
                print("  %-14s ERROR %s" % (key, err), file=sys.stderr)
            else:
                results[key] = m
                print("  %-14s root=%-7d top1=%.4f H=%.4f w90=%d n>=1%%=%d "
                      "share=%.3f batch=%.1f %.0f nn/s"
                      % (key, m["root_visits"], m["top1"], m["entropy"],
                         m["w90"], m["n_ge_1pct"], m["shown_share"] or -1,
                         m["avg_batch"] or -1, m["nnEvals_per_s"] or -1))

    if not args.keep_logdir:
        shutil.rmtree(logdir, ignore_errors=True)

    payload = {
        "tag": args.tag, "visits": args.visits, "repeats": args.repeats,
        "cwd": args.cwd, "katago": args.katago, "model": args.model,
        "base_config": args.base_config, "override_config": args.override_config,
        "extra_override": args.extra_override,
        "positions": [n for n, _ in positions],
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("json -> " + args.json)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
