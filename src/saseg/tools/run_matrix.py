from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def parse_models(raw: str) -> list[tuple[str, str, int | None]]:
    models = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        label, spec = item.split("=", 1)
        parts = spec.split(":")
        variant = parts[0]
        unknown_queries = int(parts[1]) if len(parts) > 1 and parts[1] else None
        models.append((label.strip(), variant.strip(), unknown_queries))
    return models


def gpu_free_mb() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
    except Exception:
        return {}
    result = {}
    for line in output.strip().splitlines():
        index, free = [item.strip() for item in line.split(",")[:2]]
        result[index] = int(free)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed experiment matrix with GPU scheduling")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocols", default="A1")
    parser.add_argument("--models", default=(
        "perpixel_k=perpixel_k,perpixel_k1=perpixel_k1,saseg=saseg:2,saseg_m0=saseg:0,"
        "saseg_generic=saseg_generic:2,saseg_classagnostic=saseg_classagnostic:2,binary_branch=binary_branch"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--per-gpu", type=int, default=1)
    parser.add_argument("--min-free-mb", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--volume-cache", type=int, default=1)
    parser.add_argument("--volume-cache-mb", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="Do not run the full test evaluation after training (run it separately).")
    parser.add_argument("--stream-eval", action="store_true",
                        help="Use memory-bounded evaluation for A2/final evaluation.")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-start-epoch", type=int, default=10)
    parser.add_argument("--spheres-per-class", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if COMPLETE exists")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--stagger-seconds", type=float, default=180.0,
                        help="Delay between launches so the first process warms the Python/model cache.")
    parser.add_argument("--no-resume", dest="resume_existing", action="store_false", default=True,
                        help="Do not resume from last.pt when rerunning an interrupted run.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    models = parse_models(args.models)
    tasks = []
    for protocol in protocols:
        for label, variant, unknown_queries in models:
            for seed in seeds:
                run_dir = args.run_root / args.dataset / protocol / label / f"seed{seed}"
                task = {"dataset": args.dataset, "protocol": protocol, "label": label,
                        "variant": variant, "unknown_queries": unknown_queries, "seed": seed,
                        "run_dir": run_dir, "manifest": str(args.manifest)}
                command = [sys.executable, "-m", "saseg.e1_train",
                           "--manifest", str(args.manifest), "--variant", variant, "--protocol", protocol,
                           "--output-dir", str(run_dir), "--seed", str(seed),
                           "--size", str(args.size), "--num-workers", str(args.num_workers),
                           "--volume-cache", str(args.volume_cache), "--volume-cache-mb", str(args.volume_cache_mb),
                           "--spheres-per-class", str(args.spheres_per_class),
                           "--decoder-layers", str(args.decoder_layers),
                           "--backbone", args.backbone]
                a1_run_dir = args.run_root / args.dataset / "A1" / label / f"seed{seed}"
                if protocol == "A2":
                    # A1 and A2 share the same training supervision (seen unknowns).
                    # A2 only re-encodes the evaluation splits, so reuse the A1 checkpoint.
                    task["requires"] = str(a1_run_dir / "COMPLETE")
                    task["checkpoint"] = str(a1_run_dir / "best.pt")
                    command += ["--eval-only", "--checkpoint", str(a1_run_dir / "best.pt")]
                    if args.stream_eval:
                        command.append("--stream-eval")
                else:
                    task["resume"] = str(run_dir / "last.pt")
                    command += ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                                "--accumulation-steps", str(args.accumulation_steps),
                                "--lr", str(args.lr), "--backbone-lr", str(args.backbone_lr),
                                "--warmup-steps", str(args.warmup_steps), "--val-every", str(args.val_every),
                                "--early-stop-patience", str(args.early_stop_patience),
                                "--early-stop-min-delta", str(args.early_stop_min_delta),
                                "--early-stop-start-epoch", str(args.early_stop_start_epoch)]
                    if args.max_val_batches is not None:
                        command += ["--max-val-batches", str(args.max_val_batches)]
                    if args.skip_final_eval:
                        command.append("--skip-final-eval")
                    if args.amp:
                        command.append("--amp")
                    if args.pretrained:
                        command.append("--pretrained")
                    if args.augment:
                        command.append("--augment")
                if unknown_queries is not None:
                    command += ["--unknown-queries", str(unknown_queries)]
                if task.get("resume") and args.resume_existing and Path(task["resume"]).is_file():
                    command += ["--resume", task["resume"]]
                task["command"] = command
                tasks.append(task)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / f"matrix_{args.dataset}.json").write_text(json.dumps(
        [{key: value for key, value in task.items() if key != "run_dir"} | {"run_dir": str(task["run_dir"])}
         for task in tasks], indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks), "protocols": protocols, "seeds": seeds, "gpus": gpus}), flush=True)
    if args.dry_run:
        for task in tasks:
            print(" ".join(task["command"]))
        return

    pending = deque(task for task in tasks
                    if args.force or not (task["run_dir"] / "COMPLETE").is_file())
    skipped = len(tasks) - len(pending)
    print(json.dumps({"pending": len(pending), "skipped_complete": skipped}), flush=True)
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(src) + os.pathsep + env_base.get("PYTHONPATH", "")
    env_base.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "SASEG_TORCH_THREADS"):
        env_base[name] = "1"
    concurrency = max(1, len(gpus) * args.per_gpu)
    running: list[tuple[dict, subprocess.Popen]] = []
    launch_index = 0
    last_launch = 0.0
    try:
        while pending or running:
            while pending and len(running) < concurrency:
                if running and time.time() - last_launch < args.stagger_seconds:
                    break
                ready = next((item for item in pending
                              if not item.get("requires") or Path(item["requires"]).is_file()), None)
                if ready is None:
                    blocked = [item for item in pending
                               if item.get("requires") and Path(item["requires"]).with_name("FAILED").is_file()]
                    for item in blocked:
                        pending.remove(item)
                        item["run_dir"].mkdir(parents=True, exist_ok=True)
                        (item["run_dir"] / "FAILED").write_text(json.dumps(
                            {"reason": "required training run failed", "requires": item["requires"]}, indent=2) + "\n")
                        print(json.dumps({"event": "blocked", "label": item["label"], "protocol": item["protocol"],
                                          "seed": item["seed"]}), flush=True)
                    if not running:
                        print(json.dumps({"event": "waiting_for_dependency",
                                          "remaining": len(pending)}), flush=True)
                    break
                free = gpu_free_mb()
                candidates = [gpu for gpu in gpus if free.get(gpu, 10**9) >= args.min_free_mb]
                if not candidates:
                    print(json.dumps({"event": "waiting_for_gpu", "free_mb": free}), flush=True)
                    break
                gpu = candidates[launch_index % len(candidates)]
                launch_index += 1
                task = ready
                pending.remove(task)
                task["run_dir"].mkdir(parents=True, exist_ok=True)
                (task["run_dir"] / "task.json").write_text(json.dumps(
                    {key: value for key, value in task.items() if key not in {"run_dir", "manifest"}} |
                    {"run_dir": str(task["run_dir"]), "manifest": str(task["manifest"]), "gpu": gpu},
                    indent=2) + "\n")
                env = env_base.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                log_path = task["run_dir"] / "launcher.log"
                log = log_path.open("a", encoding="utf-8")
                log.write(f"[launcher] {time.strftime('%Y-%m-%dT%H:%M:%S')} gpu={gpu}\n")
                log.write("[launcher] " + " ".join(task["command"]) + "\n")
                log.flush()
                proc = subprocess.Popen(task["command"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                running.append((task, proc))
                last_launch = time.time()
                print(json.dumps({"event": "start", "label": task["label"], "protocol": task["protocol"],
                                  "seed": task["seed"], "gpu": gpu, "pid": proc.pid}), flush=True)
            time.sleep(args.poll_seconds)
            for task, proc in list(running):
                code = proc.poll()
                if code is None:
                    continue
                running.remove((task, proc))
                marker = task["run_dir"] / ("COMPLETE" if code == 0 else "FAILED")
                marker.write_text(json.dumps({"exit_code": code, "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                             indent=2) + "\n")
                print(json.dumps({"event": "finish", "label": task["label"], "protocol": task["protocol"],
                                  "seed": task["seed"], "exit_code": code}), flush=True)
    except KeyboardInterrupt:
        for task, proc in running:
            proc.terminate()
        raise


if __name__ == "__main__":
    main()
