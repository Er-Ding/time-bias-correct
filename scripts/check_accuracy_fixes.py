"""修复验收：完整回归、独立噪声重复、CPU/GPU 一致性和保存观测重放。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import numpy as np


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def detector_check(output, backend):
    from time_bias_localization.path_detection import detect_csi_paths
    from time_bias_localization.signal import synthesize_ula_csi
    arguments = dict(subcarrier_frequencies_hz=np.arange(64)*3.125e6,
        carrier_frequency_hz=3.5e9,antenna_spacing_m=None,
        aoa_grid_rad=np.linspace(-1.2,1.2,61),delay_grid_s=np.linspace(0,280e-9,141),
        settings=dict(enabled=True,max_paths=6,false_alarm_probability=.01,
                      calibration_trials=1023,max_refine_evaluations=60))
    rows = []
    for kind,repeats in [("noise_only",100),("single_path",30),("strong_and_weak",30)]:
        for index in range(repeats):
            rng = np.random.default_rng(2026091200+index)
            clean = np.zeros((8,64),complex)
            if kind != "noise_only":
                angles,delays,gains = [.173],[123.456e-9],[1.]
                if kind == "strong_and_weak":
                    angles += [-.612]; delays += [217.321e-9]; gains += [.035]
                clean = synthesize_ula_csi(path_aoa_rad=angles,path_delay_s=delays,path_coefficients=gains,
                    num_bs_antennas=8,subcarrier_frequencies_hz=arguments["subcarrier_frequencies_hz"],
                    carrier_frequency_hz=3.5e9)
            csi = clean+.01*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))/np.sqrt(2)
            began = time.monotonic()
            result = detect_csi_paths(csi,backend=backend,**arguments)
            row = dict(kind=kind,repeat=index,count=len(result.peaks),elapsed_s=time.monotonic()-began,
                       accepted=[asdict(p) for p in result.peaks],diagnostics=result.diagnostics)
            rows.append(row)
            if index%10 == 0:
                print(f"[独立噪声验证] {kind} {index+1}/{repeats}，检出 {len(result.peaks)} 条",flush=True)
            if backend == "cuda" and index == 0:
                cpu = detect_csi_paths(csi,backend="numpy",**arguments)
                assert len(cpu.peaks) == len(result.peaks)
                np.testing.assert_allclose([[p.aoa_rad,p.delay_s] for p in cpu.peaks],
                                           [[p.aoa_rad,p.delay_s] for p in result.peaks],rtol=1e-7,atol=1e-11)
                row["cpu_gpu_agree"] = True
    summaries = {}
    for kind in ("noise_only","single_path","strong_and_weak"):
        selected=[r for r in rows if r["kind"]==kind]
        expected={"noise_only":0,"single_path":1,"strong_and_weak":2}[kind]
        summaries[kind]=dict(repeats=len(selected),expected_count=expected,
            correct_count=sum(r["count"]==expected for r in selected),
            extra_component_count=sum(r["count"]>expected for r in selected),
            missed_component_count=sum(r["count"]<expected for r in selected))
    write(output/"detector_trials.json",rows)
    write(output/"detector_summary.json",dict(summaries=summaries,
        scope="implementation_validation_on_synthetic_csi; not_full_30UE_accuracy",
        threshold_scope="whole_2d_search; independent_noise_repeats_not_used_to_tune_threshold"))
    # 有限重复不是误检率的严格上界；保留全部样本，不只保存通过的样本。
    assert summaries["single_path"]["correct_count"]>=27, summaries
    assert summaries["strong_and_weak"]["correct_count"]>=27, summaries
    return summaries


def replay(output, experiment, scene_path):
    from audit_boundary_accuracy import read,sha,closest
    from time_bias_localization.bias_interval_candidates import generate_bias_interval_points
    from time_bias_localization.initial_candidates import generate_initial_candidate_points
    from time_bias_localization.candidates import PathObservationSample
    from time_bias_localization.scene import Scene2D
    frozen = read(experiment/"experiment.json")
    points = {p["ue_id"]:p for p in read(experiment/"pilot/plan.json")["points"]}
    trials = [json.loads(line) for line in (experiment/"pilot/trials.jsonl").read_text().splitlines()]
    expected_sha = next(iter(points.values()))["observations"][0]["scene_sha256"]
    if sha(scene_path)!=expected_sha:
        raise ValueError("重放公共地图与原批次摘要不同")
    scene = Scene2D.from_dict(read(scene_path))
    result_rows=[]
    for ue in ("PILOT_0006","PILOT_0011","PILOT_0026","PILOT_0030","PILOT_0008","PILOT_0027"):
        trial = next(t for t in trials if t["ue_id"]==ue and t["repeat_index"]==0 and t["strategy"]=="coverage")
        root = Path(trial["result_dir"])/"localization"
        manifest = read(root/"localization_manifest.json")
        peaks = read(root/"music_peaks.json",manifest["artifacts"]["music_peaks"]["sha256"])
        saved = read(root/"initial_candidates.json",manifest["artifacts"]["initial_candidates"]["sha256"])
        config = read(root/"localization_config.json",manifest["config_snapshot"]["file_sha256"])["resolved_config"]
        samples = [PathObservationSample(**s) for s in peaks["observation_samples"]]
        args = dict(reference_bias_s=config["localization"]["initial_reference_bias_s"],
            max_reflections=config["scene"]["max_reflections"],max_diffractions=config["scene"]["max_diffractions"],
            diffraction_directions_per_sample=config["localization"]["diffraction_directions_per_sample"],
            diffraction_angle_tolerance_deg=config["localization"]["diffraction_angle_tolerance_deg"])
        old = generate_initial_candidate_points(scene,config["radio"]["bs_position_m"],samples,**args)
        assert len(old.points)==len(saved["points"])
        for p,q in zip(old.points,saved["points"]):
            assert p.sample_id==q["sample_id"] and p.topology_id==q["topology_id"]
            np.testing.assert_allclose(p.position_m,q["position_m"],rtol=0,atol=1e-10)
        full = generate_bias_interval_points(scene,config["radio"]["bs_position_m"],samples,**args,
            bias_interval_s=(config["localization"]["bias_min_s"],config["localization"]["bias_max_s"]))
        # 真值只在下列距离评估使用；候选生成既不接收真值位置，也不接收真值 b。
        truth_bias = frozen["generation_config"]["simulation"]["clock_bias_s"]
        old_distances = closest(saved["points"],points[ue]["position_m"],truth_bias)
        new_distances = closest([asdict(p) for p in full.points],points[ue]["position_m"],truth_bias)
        for observation,distance in old_distances.items():
            assert new_distances.get(observation,np.inf)<=distance+1e-7
        row=dict(ue_id=ue,old_count=len(old.points),new_count=len(full.points),
                 old_nearest_m=old_distances,new_nearest_m=new_distances,
                 baseline_reproduces_saved=True,reference_group_count=len(full.diagnostics["reference_groups"]))
        result_rows.append(row)
        write(output/f"replay_{ue}.json",row)
        print(f"[保存观测重放] {ue}: {old_distances} -> {new_distances}",flush=True)
    write(output/"replay_summary.json",dict(rows=result_rows,scene_sha256=expected_sha,
         scope="saved_music_samples_only; raw_CSI_unavailable; not_a_new_localization_accuracy_result"))
    return result_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--backend",choices=("numpy","cuda"),default="numpy")
    parser.add_argument("--saved-experiment",type=Path,required=True)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--skip-tests",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from time_bias_localization.boundary_experiment import source_fingerprint
    provenance=source_fingerprint()
    write(args.output/"source_before.json",provenance)
    if not args.skip_tests:
        print("[代码回归] 开始完整测试",flush=True)
        subprocess.run([sys.executable,"-u","-m","pytest","-q",
                        "--junitxml="+str(args.output/"pytest.xml")],check=True)
    detector=detector_check(args.output,args.backend)
    geometry=replay(args.output,args.saved_experiment,args.scene)
    assert source_fingerprint()==provenance,"验证过程中源码发生变化，结果不能代表最终代码"
    write(args.output/"validation_summary.json",dict(status="passed",detector=detector,
        replay_count=len(geometry),full_tests_run=not args.skip_tests,backend=args.backend))
    print(f"[验收完成] {args.output}/validation_summary.json",flush=True)


if __name__=="__main__":
    main()
