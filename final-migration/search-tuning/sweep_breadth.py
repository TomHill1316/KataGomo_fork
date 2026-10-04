#!/usr/bin/env python3
"""多臂广度扫描：同轮配对 + 轮换臂序 + 漂移标尺。

用途：把「哪些旋钮真正在增广」归因分解出来，而不是把一堆参数一起改。

用法
  python3 sweep_breadth.py \\
      --katago bin/katago-prune-sched-grace --model models/b11-ffn-pruned-a8.bin.gz \\
      --base-config config/gtp_sm89_plan_pruned.cfg \\
      --arms "A=;B=wideRootNoise=0.03;C=wideRootNoise=0.08;D=rootDesiredPerChildVisitsCoeff=2;E=minVisitPropForLCB=0.20" \\
      --visits 20000 --rounds 4 --only midgame8,empty --cwd /root/autodl-tmp/katago-plan

arm 规格： NAME=<逗号分隔的 -override-config k=v>，NAME=A 的那条留空即基线。

⚠️ 臂之间用 **分号** 分隔，不是逗号 —— 逗号是臂内部 override 的分隔符
   （-override-config "k1=v1,k2=v2"）。用逗号分臂会退化成「只有 1 个臂」而直接报错。
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = [("top1", "lower"), ("entropy", "higher"), ("w90", "higher"),
           ("n_ge_1pct", "higher"), ("top3_share", "lower")]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--arms", required=True,
                    help='臂之间用分号分隔，如 "A=;B=wideRootNoise=0.03"')
    ap.add_argument("--visits", type=int, default=20000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--only", default="")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--outdir", default="/tmp/sweep-breadth")
    args = ap.parse_args()

    arms = []
    for spec in args.arms.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        name, _, ov = spec.partition("=")
        arms.append((name.strip(), ov.strip()))
    if len(arms) < 2:
        print("至少需要 2 个臂", file=sys.stderr)
        return 2

    args.cwd = os.path.abspath(args.cwd)
    os.makedirs(args.outdir, exist_ok=True)

    data = {}   # data[arm][pos] = [metricdict, ...]
    for name, _ in arms:
        data[name] = {}

    for rd in range(1, args.rounds + 1):
        print("===== round %d/%d =====" % (rd, args.rounds), flush=True)
        # 轮换：每轮把臂序左移 (rd-1) 位
        order = arms[(rd - 1) % len(arms):] + arms[:(rd - 1) % len(arms)]
        for name, ov in order:
            out = os.path.join(args.outdir, "%s_r%d.json" % (name, rd))
            cmd = [sys.executable, os.path.join(HERE, "breadth_probe.py"),
                   "--katago", args.katago, "--model", args.model,
                   "--base-config", args.base_config,
                   "--visits", str(args.visits), "--tag", "%s_r%d" % (name, rd),
                   "--json", out, "--cwd", args.cwd, "--timeout", str(args.timeout)]
            if ov:
                cmd += ["--extra-override", ov]
            if args.only:
                cmd += ["--only", args.only]
            print("  [%s] %s" % (name, ov or "(baseline)"), flush=True)
            r = subprocess.run(cmd, cwd=args.cwd)
            if r.returncode != 0 or not os.path.exists(out):
                print("  !! 臂 %s 失败" % name, file=sys.stderr)
                continue
            with open(out) as fh:
                payload = json.load(fh)
            for k, v in payload["results"].items():
                if "error" in v:
                    continue
                data[name].setdefault(k.split("#")[0], []).append(v)

    print()
    base = arms[0][0]
    print("=" * 104)
    print("基线臂 = %s；漂移 = 基线臂跨轮极差（噪声标尺）；|Δmed| < drift ⇒ 噪声内" % base)
    print("=" * 104)
    summary = {}
    for pos in sorted(data[base]):
        print("\n[%s]  (rounds=%d)" % (pos, len(data[base][pos])))
        hdr = "  %-14s %-10s" % ("metric", base)
        for name, _ in arms[1:]:
            hdr += " %-18s" % name
        print(hdr)
        summary[pos] = {}
        for metric, better in METRICS:
            bvals = [x[metric] for x in data[base][pos] if x.get(metric) is not None]
            if not bvals:
                continue
            drift = max(bvals) - min(bvals)
            line = "  %-14s %-10.4f" % (metric, med(bvals))
            row = {"base_med": med(bvals), "drift": drift}
            for name, _ in arms[1:]:
                avals = [x[metric] for x in data[name].get(pos, []) if x.get(metric) is not None]
                if not avals:
                    line += " %-18s" % "-"
                    continue
                d = med(avals) - med(bvals)
                tag = "噪声内" if abs(d) < drift else ("更广" if (
                    (better == "lower" and d < 0) or (better == "higher" and d > 0)) else "更窄")
                line += " %-18s" % ("%+.4f %s" % (d, tag))
                row[name] = {"med": med(avals), "dmed": d, "verdict": tag}
            print(line)
            summary[pos][metric] = row

    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary -> " + os.path.join(args.outdir, "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
