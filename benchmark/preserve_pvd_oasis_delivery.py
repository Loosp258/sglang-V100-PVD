"""Preserve completed independent Oasis delivery pilots and their actual gates.

No SSH, GPU operations or source deployment. Serving source identity is checked
against the recorded deployment bundle, never against a later working tree.
"""

import argparse
import copy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from statistics import median
import tarfile

from pvd_oasis_delivery_validation import (
    DELIVERY_COMPARISONS, DELIVERY_TIMINGS, validate_delivery_profiles,
)
from verify_pvd_oasis_latency_evidence import verify


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (ROOT / "artifacts").resolve()
RESULTS = (ROOT / "benchmark/results").resolve()
RESULT_PREFIX = {"v-combine": "pvd_oasis_combined_delivery_cloudlab_",
                 "v-slots": "pvd_oasis_receive_slots_cloudlab_"}


def project_path(value, parent, *, exists=True):
    path = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError("path must stay under " + str(parent))
    if exists and not path.exists():
        raise ValueError("missing project evidence: " + str(path))
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def first_formal_time(directory):
    online = read_json(directory / "online.json")
    stamps = [row["started_unix"] for rows in online.values() for row in rows]
    assert stamps and all(type(value) in (int, float) and value > 0 for value in stamps)
    return datetime.fromtimestamp(min(stamps), timezone(timedelta(hours=8)))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def archive_directory(directory, output):
    """Store complete raw bytes and validate CRC without extracting to disk."""
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError("evidence archive must not follow outside links: " + str(path))
            if path.is_file():
                archive.add(path, arcname=directory.name + "/" + path.relative_to(directory).as_posix(), recursive=False)
    raw = gzip.decompress(output.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for item in archive:
            if item.isfile():
                archive.extractfile(item).read()


def assert_gpu_empty(data):
    assert set(data) == {"p", "v", "d"}, "all three nodes must have final GPU proof"
    for role, text in data.items():
        rows = text.splitlines()
        assert len(rows) == 2 and all(int(row.split(",")[1].split()[0]) == 0 for row in rows), role


def deployment_proof(directory, gate):
    expected = read_json(gate / "local_source_hashes.json")
    bundle = gate / "deployed.tar.gz"
    with tarfile.open(bundle, "r:gz") as archive:
        sources = {item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest()
                   for item in archive if item.isfile()}
    assert sources == expected, "recorded deployed bundle differs from its source identity"
    for role in ("v", "d"):
        assert read_json(gate / (role + "_source_hashes.json")) == expected
        for name, digest in read_json(directory / "source_hashes.json")[role].items():
            assert expected.get(name) == digest, (role, name, "serving source differs from tested deployment")
    return dict(source="recorded gate/deployed.tar.gz and runner source_hashes.json",
                bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                deployed_files=len(sources), serving_sources_match_gate=True,
                live_working_tree_sources_used=False)


def gate_counts(gate):
    def unit_count(label):
        unit_status = read_json(gate / (label + "_status.json"))
        assert unit_status == {"exit_code": 0}, "CPU gate did not finish successfully"
        text = (gate / (label + ".txt")).read_text(encoding="utf-8")
        summaries = [line for line in text.splitlines() if re.search(r"\d+ passed", line) and re.search(r"in [\d.]+s", line)]
        assert len(summaries) == 1, "one complete pytest result required"
        line = summaries[0]
        counts = {kind: int(number) for number, kind in re.findall(r"(\d+) (passed|failed|skipped|errors?|warnings?)", line)}
        assert counts.get("passed", 0) > 0 and not any(counts.get(name, 0) for name in ("failed", "error", "errors"))
        return dict(record_file="gate.tar.gz:" + gate.name + "/" + label + ".txt",
                    exit_code=0, counts=counts, exact_summary=line,
                    seconds=float(re.search(r"in ([\d.]+)s", line)[1]))
    result = dict(unit=unit_count("unit"))
    if (gate / "records_unit_status.json").exists():
        result["records_unit"] = unit_count("records_unit")
    if (gate / "native_status.json").exists():
        assert read_json(gate / "native_status.json") == {"exit_code": 0}
        complete_file = gate / "native.json"
        if complete_file.exists():
            native = read_json(complete_file)
            result["native_count_record_file"] = "gate.tar.gz:" + gate.name + "/native.json"
            result["native_full_observations_saved"] = True
        else:
            native_text = (gate / "native.txt").read_text(encoding="utf-8")
            records = [json.loads(line) for line in native_text.splitlines()
                       if line.startswith("{") and line.endswith("}")]
            assert len(records) == 1, "one complete native gate count record required"
            native = records[0]
            result["native_count_record_file"] = "gate.tar.gz:" + gate.name + "/native.txt final JSON"
            result["native_full_observations_saved"] = False
        assert native["status"] == "passed" and native["transport"] == "mooncake_local_session"
        assert type(native["exact_byte_cases"]) is int and native["exact_byte_cases"] == 48
        assert native["after_close"]["closed"] is True
        assert native["after_close"]["physical_bytes"] == native["after_close"]["unknown_slots"] == 0
        if "observations" in native:
            assert len(native["observations"]) == native["exact_byte_cases"]
        result["native"] = {key: value for key, value in native.items() if key != "observations"}
    return result


def delivery_stages(summary):
    output = {}
    for mode in ("baseline", "optimized"):
        rows = [row for arm, requests in summary["requests"].items()
                if arm.startswith("opt") is (mode == "optimized") for row in requests]
        profiles = [profile for row in rows for transport in row["trace"]["transport"][28:]
                    for profile in transport["deliveries"]]
        assert profiles, "no completed steady sparse deliveries"
        output[mode] = dict(rank_deliveries=len(profiles),
            request_steady_wait_sum_ms=[dict(case=row['case'], rid=row['rid'],
                milliseconds=sum(forward['wait_ms'] for forward in row['forward'] if forward['step'] > 0))
                for row in rows],
            median_stage_ms={name.removesuffix("_seconds"): median(item[name] * 1000 for item in profiles)
                             for name in DELIVERY_TIMINGS},
            reserve_start_submission_ms=median((item["reserve_seconds"] + item["start_seconds"] +
                                                item["combined_seconds"]) * 1000 for item in profiles),
            median_payload_bytes=median(item["nbytes"] for item in profiles),
            actual_request_counts={name: sum(row["trace"]["io"][name] for row in rows) for name in (
                "delivery_count", "registration_count", "unregistration_count", "reserve_rpc_count",
                "start_rpc_count", "combined_rpc_count", "poll_rpc_count", "ack_rpc_count")},
            scope="stage medians exclude initial-bank priming; total request counters include priming and final physical pool retirement")
    return output


def report_text(directory, summary, stages, counts, proof, failure_note):
    comparison = read_json(directory / "comparison.json")
    rpc = read_json(directory / "rpc_summary.json")["aggregate"]
    search = read_json(directory / "v_search_summary.json")["aggregate"]
    aggregate = summary["aggregate"]
    title = "合并 reserve＋start" if comparison["comparison"] == "v-combine" else "复用接收区物理注册"
    baseline_label = "独立 reserve/start" if comparison["comparison"] == "v-combine" else "每次交付注册"
    optimized_label = "合并 reserve/start" if comparison["comparison"] == "v-combine" else "请求内复用 slots"
    baseline, optimized = aggregate["baseline"], aggregate["optimized"]
    started = first_formal_time(directory)
    def change(old, new):
        return (new / old - 1) * 100 if old else None
    lines = [f"# Oasis KV 交付：{title}公平对照", "", f"CloudLab {started:%Y-%m-%d}；完整快图＋V/CAGRA＋Oasis 配对逐层 Decode。",
        f"日期按首个正式请求started_unix换算UTC+8：{started.isoformat(timespec='seconds')}。", "",
        "## 结果", "",
        f"客户端完成中位数 **{baseline['wall_seconds']:.3f}→{optimized['wall_seconds']:.3f} s（{change(baseline['wall_seconds'], optimized['wall_seconds']):+.2f}%）**；",
        f"每请求后续Decode累计 KV 等待 **{baseline['steady_request_wait_sum_ms']:.3f}→{optimized['steady_request_wait_sum_ms']:.3f} ms**。",
        "每模式四个正式请求、只覆盖两个Prompt；3136个层/rank profile不视作独立请求样本。两个独立优化的结果不能相加。新选项仍默认关闭。", "", "## 公平性", "",
        "base_a→opt_a→opt_b→base_b，各臂重启全角色，排除相同两次 warmup；两个2159-token Prompt、16输出token、greedy/ignore_eos。",
        "四合一快图、degree16、prefix2048＋tail111、itopk2048、Top4、capacity32、max_new16、workers2保持一致。",
        "V固定 pool/host candidates/GPU finite-Q proof；Triton打包关闭；D显式 reuse_io=false。初始KV仍图门后P→V→D，private Prompt seed计入客户端。",
        ("唯一变量为D combine_reserve_start；所有臂 reuse_receive_slots=false。" if comparison["comparison"] == "v-combine"
         else "唯一变量为D reuse_receive_slots；所有臂 combine_reserve_start=false。"),
        "八次Prompt、实际输出ID和文本一致，cached_tokens=0；每请求420jobs/840搜索RPC，每模式3136稳态查询profile，均2items/14Qrows。",
        f"部署 bundle 中{proof['deployed_files']}个文件的实际哈希与V/D serving source gate一致；源码身份使用记录的部署包，没有读取后来编辑的工作树。",
        "所有接收写入保留完整身份、精确native终态字节证明、安装后ACK及UNKNOWN保留；最终 owned={}、cleanup_errors=[]、六张GPU归零。", "",
        "## 完整路径时间", "",
        f"| 指标（独立中位数） | {baseline_label} | {optimized_label} | 相对变化 |",
        "|---|---:|---:|---:|"]
    for label, old, new, unit in (
        ("V查询wall/rank/层", search["baseline"]["median_stage_ms"]["batch_total"], search["optimized"]["median_stage_ms"]["batch_total"], "ms"),
        ("D search_many", rpc["baseline"]["search_ms"]["total"], rpc["optimized"]["search_ms"]["total"], "ms"),
        ("D整层检索/交付RPC", baseline["layer_rpc_ms"], optimized["layer_rpc_ms"], "ms"),
        ("层worker完整service", baseline["layer_service_ms"], optimized["layer_service_ms"], "ms"),
        ("每请求后续Decode累计KV等待", baseline["steady_request_wait_sum_ms"], optimized["steady_request_wait_sum_ms"], "ms"),
        ("每步逐层等待和中位数", baseline["steady_layer_wait_sum_ms"], optimized["steady_layer_wait_sum_ms"], "ms"),
        ("后续Decode执行", baseline["steady_forward_ms"], optimized["steady_forward_ms"], "ms"),
        ("首个客户端事件", baseline["first_event_seconds"], optimized["first_event_seconds"], "s"),
        ("客户端完成", baseline["wall_seconds"], optimized["wall_seconds"], "s")):
        lines.append(f"| {label} | {old:.3f} {unit} | {new:.3f} {unit} | {change(old,new):+.2f}% |")
    lines += ["", "每请求累计等待先对该请求step>0的各次forward等待求和，再对每模式四个正式请求取中位数；",
              "每步逐层等待和另按forward取中位数。两者范围不同，不能将每步约数百毫秒当作整个请求的累计等待。"]
    lines += ["", "## 实际交付子阶段与调用", "",
        "以下时间为已完成的稳态missing-rank交付，排除初始28层bank priming；注册池首次注册仍计入客户端初始化。",
        "控制提交时间先对每次交付的reserve/start/combined求和，再计算中位数。各阶段独立中位数不能相加重建请求。", "",
        f"| 子阶段 | {baseline_label} | {optimized_label} |", "|---|---:|---:|"]
    for label, field in (("接收区准备", "prepare"), ("物理分配", "allocate"), ("物理注册", "register"),
                         ("poll RPC", "poll"), ("安装后ACK", "ack"), ("接收区close", "close"), ("GPU→CPU缓存copy", "cache_copy")):
        lines.append(f"| {label} | {stages['baseline']['median_stage_ms'][field]:.3f} ms | {stages['optimized']['median_stage_ms'][field]:.3f} ms |")
    lines.append(f"| reserve/start控制提交 | {stages['baseline']['reserve_start_submission_ms']:.3f} ms | {stages['optimized']['reserve_start_submission_ms']:.3f} ms |")
    lines += ["", "实际总调用计数覆盖每模式四个正式请求，含priming及最终物理池退休；poll为实测次数，start直接返回READY时可为零。", "",
              f"| 总计数 | {baseline_label} | {optimized_label} |", "|---|---:|---:|"]
    for label, name in (("missing-rank交付", "delivery_count"), ("物理register", "registration_count"),
                        ("物理unregister", "unregistration_count"), ("reserve RPC", "reserve_rpc_count"),
                        ("start RPC", "start_rpc_count"), ("combined RPC", "combined_rpc_count"),
                        ("poll RPC", "poll_rpc_count"), ("ACK RPC", "ack_rpc_count")):
        lines.append(f"| {label} | {stages['baseline']['actual_request_counts'][name]} | {stages['optimized']['actual_request_counts'][name]} |")
    lines += ["", "## 四臂与流量", "", "| 执行順序 | case " + str(summary["requests"]["base_a"][0]["case"]) + " | case " + str(summary["requests"]["base_a"][1]["case"]) + " |", "|---|---:|---:|"]
    for arm, rows in summary["requests"].items():
        lines.append(f"| {arm} | {rows[0]['wall_seconds']:.3f}s | {rows[1]['wall_seconds']:.3f}s |")
    drift = {}
    for mode, first, last in (("baseline", "base_a", "base_b"), ("optimized", "opt_a", "opt_b")):
        old = median(row["wall_seconds"] for row in summary["requests"][first])
        new = median(row["wall_seconds"] for row in summary["requests"][last])
        drift[mode] = change(old, new)
    lines += ["", f"前后同配置arm的客户端中位数变化：baseline {drift['baseline']:+.2f}%，optimized {drift['optimized']:+.2f}%；保留顺序漂移，不据八个请求宣称统计显著性。",
        f"初始逻辑全KV为{baseline['startup_full_kv_bytes_per_request']:.0f}/{optimized['startup_full_kv_bytes_per_request']:.0f} B/request；稀疏payload中位数{baseline['network_sparse_bytes_per_request']:.0f}/{optimized['network_sparse_bytes_per_request']:.0f} B/request。",
        "CAGRA候选允许原生抖动；没有冻结missing集合。查询、D RPC和交付时间包含不同范围，不把差值当作纯网络或纯native传输时间。", "", "## 验证与范围", "",
        f"CPU gate实际结果：`{counts['unit']['exact_summary']}`；完整输出见gate.tar.gz与gate_count_record.json。"]
    if "native" in counts:
        lines.append(f"原生本地Mooncake gate通过{counts['native']['exact_byte_cases']}个精确字节案例，两个执行器复用四个物理MR并安全注销；本地session gate与线上跨节点RDMA对照是独立观测。")
    if "records_unit" in counts:
        lines.append(f"接收record集成重复gate：`{counts['records_unit']['exact_summary']}`；这九项已包含在完整CPU gate中，不相加为新的独立测试。")
    lines += ["单元异常测试采用CPU policy double；没有注入真实RDMA/CUDA故障。长Decode、多请求并发、TP2、更多Prompt质量与服务显存峰值仍开放。",
        "客户端完成与最终安全退休分别验证；本次子阶段profile不能单独证明整个流水线的重叠收益。", "", "## 证据与复现", "",
        "同名目录保存完整raw.tar.gz、gate.tar.gz、紧凑summary/实际IO计数、V/RPC分解、部署来源、GPU归零与LF便携manifest。"]
    if failure_note:
        lines.append("前置失败尝试另存failed_prelaunch.tar.gz：" + failure_note)
    tag = "fresh_combine" if comparison["comparison"] == "v-combine" else "fresh_slots"
    lines += ["", "```text",
        f"wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/run_pvd_oasis_serving_cloudlab.py --tag {tag} --comparison {comparison['comparison']} --arms base_a,opt_a,opt_b,base_b --cases 99401,99402 --tokens 16",
        f"wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_serving.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/{tag} --comparison {comparison['comparison']}",
        f"wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_v_search.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/{tag}",
        f"wsl -d Ubuntu -- python3 /mnt/d/code/sglang-V100-PVD-oasiskv/benchmark/analyze_pvd_oasis_rpc.py /mnt/d/code/sglang-V100-PVD-oasiskv/artifacts/{tag}",
        "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-prelaunch", type=Path)
    parser.add_argument("--failure-note", default="前置尝试未产生正式请求，未纳入计时。")
    parser.add_argument("--report", action="store_true", help="write sibling Chinese Markdown report from completed evidence")
    args = parser.parse_args()
    directory, gate = project_path(args.directory, ARTIFACTS), project_path(args.gate, ARTIFACTS)
    output = project_path(args.output, RESULTS, exists=False)
    assert not output.exists(), "fresh result destination required"
    comparison = read_json(directory / "comparison.json")
    assert comparison["comparison"] in DELIVERY_COMPARISONS
    assert re.fullmatch(RESULT_PREFIX[comparison["comparison"]] + r"\d{8}", output.name)
    assert output.name.endswith(first_formal_time(directory).strftime("%Y%m%d")), "result filename must match the first formal request date in UTC+8"
    summary = read_json(directory / "summary.json")
    assert comparison["arms"] == list(summary["requests"]) == ["base_a", "opt_a", "opt_b", "base_b"]
    assert read_json(directory / "owned.json") == {} and read_json(directory / "cleanup_errors.json") == []
    assert_gpu_empty(read_json(directory / "final_gpu_memory.json"))
    proof, counts = deployment_proof(directory, gate), gate_counts(gate)
    for arm, rows in summary["requests"].items():
        assert len(rows) == 2
        for row in rows:
            row["delivery_validation"] = validate_delivery_profiles(row["trace"], comparison=comparison["comparison"], arm=arm)
            row["steady_wait_sum_ms"] = sum(forward["wait_ms"] for forward in row["forward"] if forward["step"] > 0)
    for mode in ("baseline", "optimized"):
        summary["aggregate"][mode]["steady_request_wait_sum_ms"] = median(
            row["steady_wait_sum_ms"] for arm, rows in summary["requests"].items()
            if arm.startswith("opt") is (mode == "optimized") for row in rows)
    stages = delivery_stages(summary)
    failed = project_path(args.failed_prelaunch, ARTIFACTS) if args.failed_prelaunch else None
    if failed is not None:
        assert not (failed / "online.json").exists() or read_json(failed / "online.json") == {}, "prelaunch archive contains measured requests"
    report_path = output.with_suffix(".md")
    assert report_path.is_relative_to(RESULTS) and (not args.report or not report_path.exists())
    output.mkdir(parents=True, exist_ok=False)
    # Do not rewrite complete raw inputs; preserve their existing observed bytes.
    archive_directory(directory, output / "raw.tar.gz")
    archive_directory(gate, output / "gate.tar.gz")
    if failed is not None:
        archive_directory(failed, output / "failed_prelaunch.tar.gz")
        write_json(output / "failed_prelaunch.json", dict(directory=failed.relative_to(ROOT).as_posix(),
            measurements_included=False, note=args.failure_note))
    for name in ("rpc_summary.json", "v_search_summary.json", "source_hashes.json", "comparison.json",
                 "checkout_heads.json", "final_gpu_memory.json", "cleanup_errors.json", "owned.json"):
        write_json(output / name, read_json(directory / name))
    write_json(output / "implementation.json", proof)
    write_json(output / "gate_count_record.json", counts)
    write_json(output / "delivery_stage_summary.json", stages)
    compact = copy.deepcopy(summary)
    for rows in compact["requests"].values():
        for row in rows:
            trace = row.pop("trace")
            row["trace_counts"] = {name: len(trace[name]) for name in ("layers", "transport")}
            row["io"] = trace["io"]
    compact["full_trace_location"] = "raw.tar.gz: " + directory.name + "/summary.json and *_d.log"
    write_json(output / "summary.json", compact)
    manifest = {}
    for path in sorted(output.iterdir()):
        if not path.is_file():
            raise ValueError("unexpected nested result output")
        data = path.read_bytes()
        normalization = "raw" if path.name.endswith(".tar.gz") else "lf"
        if normalization == "lf":
            data = data.replace(b"\r\n", b"\n")
        manifest[path.name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), normalization=normalization)
    write_json(output / "manifest.json", manifest)
    checked = verify(output)
    if args.report:
        report_path.write_text(report_text(directory, summary, stages, counts, proof,
                                          args.failure_note if failed is not None else None),
                               encoding="utf-8", newline="\n")
    print(json.dumps(dict(output=output.relative_to(ROOT).as_posix(), verification=checked,
                         report=report_path.relative_to(ROOT).as_posix() if args.report else None), ensure_ascii=False))


if __name__ == "__main__":
    main()
