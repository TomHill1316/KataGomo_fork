#!/usr/bin/env python3
"""广度 A/B 配对驱动：同轮配对 + 臂序轮换 + 漂移标尺 + 丢首轮。

为什么这么写（项目既有纪律）
  * 连续测量有 1.7-1.9% 单调热漂移 ⇒ 只比「轮内差值」，不比「两组中位数相减」。
  * 每轮轮换臂顺序，抵消「后跑的吃亏」。
  * 丢弃每个 session 的首轮（GPU 冷启动偏高约 1.6%）。
  * 报告漂移量作噪声标尺：|Δ| < drift ⇒ 判为噪声内等价。

用法
  python3 ab_breadth.py --katago bin/katago-prune-sched-grace \\
      --model models/b11-ffn-pruned-a8.bin.gz \\
      --base-config config/gtp_sm89_plan_pruned.cfg \\
      --armA "" --armB config/search-tier0.cfg \\
      --visits 20000 --rounds 4 --only midgame8,opening4,empty
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_arm(args, tag, overrides, outjson):
    cmd = [sys.executable, os.path.join(HERE, "breadth_probe.py"),
           "--katago", args.katago, "--model", args.model,
           "--base-config", args.base_config,
           "--visits", str(args.visits), "--tag", tag,
           "--json", outjson, "--cwd", args.cwd,
           "--timeout", str(args.timeout)]
    for o in overrides:
        if o:
            cmd += ["--override-config", o]
    if args.only:
        cmd += ["--only", args.only]
    if args.extra_override:
        cmd += ["--extra-override", args.extra_override]
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=args.cwd)
    if r.returncode != 0:
        print("  !! arm %s 退出码 %d" % (tag, r.returncode), file=sys.stderr)
        return None
    with open(outjson) as fh:
        return json.load(fh)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--armA", default="", help="臂 A 覆盖 cfg（空=纯基线）")
    ap.add_argument("--armB", required=True, help="臂 B 覆盖 cfg")
    ap.add_argument("--armA-extra", default="", help="臂 A 的额外覆盖 cfg")
    ap.add_argument("--armB-extra", default="", help="臂 B 的额外覆盖 cfg")
    ap.add_argument("--extra-override", default="")
    ap.add_argument("--visits", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", default="")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--outdir", default="/tmp/ab-breadth")
    args = ap.parse_args()

    args.cwd = os.path.abspath(args.cwd)
    os.makedirs(args.outdir, exist_ok=True)

    armA_ov = [x for x in (args.armA, args.armA_extra) if x]
    armB_ov = [x for x in (args.armB, args.armB_extra) if x]

    # data[pos] = {"A": [...], "B": [...]}
    data = {}
    for rd in range(1, args.rounds + 1):
        print("===== round %d/%d =====" % (rd, args.rounds), flush=True)
        # 偶数轮 A 先跑，奇数轮 B 先跑 —— 臂序轮换
        order = [("A", armA_ov), ("B", armB_ov)] if rd % 2 == 1 \
            else [("B", armB_ov), ("A", armA_ov)]
        for arm, ov in order:
            out = os.path.join(args.outdir, "%s_r%d.json" % (arm, rd))
            payload = run_arm(args, "%s_r%d" % (arm, rd), ov, out)
            if payload is None:
                continue
            for k, v in payload["results"].items():
                if "error" in v:
                    continue
                pos = k.split("#")[0]
                data.setdefault(pos, {"A": [], "B": []})[arm].append(v)

    print()
    print("=" * 96)
    print("配对结果（每轮同轮配对；漂移 = 臂 A 跨轮极差，作噪声标尺）")
    print("=" * 96)
    summary = {}
    for pos in sorted(data):
        A, B = data[pos]["A"], data[pos]["B"]
        n = min(len(A), len(B))
        if n == 0:
            continue
        print("\n[%s]  n=%d" % (pos, n))
        row = {"n": n}
        for metric, better in (("top1", "lower"), ("entropy", "higher"),
                               ("w90", "higher"), ("n_ge_1pct", "higher"),
                               ("shown_share", "info")):
            a_vals = [x[metric] for x in A[:n] if x.get(metric) is not None]
            b_vals = [x[metric] for x in B[:n] if x.get(metric) is not None]
            if not a_vals or not b_vals:
                continue
            m = min(len(a_vals), len(b_vals))
            deltas = [b_vals[i] - a_vals[i] for i in range(m)]
            drift = max(a_vals[:m]) - min(a_vals[:m])
            d = med(deltas)
            flag = ""
            if better in ("lower", "higher") and abs(d) < drift:
                flag = "  (噪声内)"
            elif better == "lower":
                flag = "  << 更广" if d < 0 else "  >> 更窄"
            elif better == "higher":
                flag = "  << 更广" if d > 0 else "  >> 更窄"
            print("  %-12s A=%.4f B=%.4f  Δmed=%+.4f  driftA=%.4f%s"
                  % (metric, med(a_vals[:m]), med(b_vals[:m]), d, drift, flag))
            row[metric] = {"A": med(a_vals[:m]), "B": med(b_vals[:m]),
                           "dmed": d, "driftA": drift, "noise": abs(d) < drift}
        summary[pos] = row

    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary -> " + os.path.join(args.outdir, "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
