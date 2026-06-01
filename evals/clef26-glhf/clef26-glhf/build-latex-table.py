#!/usr/bin/env python3
from pathlib import Path
from glob import glob
import json


MEASURES_TO_REPORT = {
    "rougeL_f1": {"file": "gold-answer-comparison", "display_name": "Rouge"},
    "bertscore_f1": {"file": "gold-answer-comparison", "display_name": "BertScore"},
    "avg_prec_at_q": {"file": "retrieval", "display_name": "precision"},
    "NUGGET_COVERAGE": {"file": "trec-auto-judge-prefnugget-grounded", "display_name": "Nugget Cov."},
    "AVG_GRADE": {"file": "trec-auto-judge-prefnugget-grounded", "display_name": "Avg. Grade"},
    "GEN-TFC1": {"file": "trec-auto-judge-ir-axioms", "display_name": "TFC1"},
}


TABLE_PREFIX = """\\begin{table}[t]
\\caption{Evaluation for Task 4 of LongEval. We report the F$_{1}$ score of Rouge-L, BertScore as primary measures. Additionally, we report the precision of retrieval, nugget coverage, average grade, and TFC1.}
\\centering
\\begin{tabular}{@{}lcccccc@{}}
\\toprule
\\bf Approach  & \\bf Rouge & \\bf BertScore & \\bf Precision & \\bf Nugget Cov. & \\bf Avg. Grade & \\bf TFC1 \\\\
\\midrule

"""

TABLE_SUFFIX = """

\\bottomrule
\\end{tabular}
\\label{table-evaluation}
\\end{table}"""

def read_from_jsonl(f, m):
    f = glob(f"{f}/*.json")
    assert len(f) == 1

    return json.loads(Path(f[0]).read_text(encoding="utf-8"))["summary"][f"avg_{m}"]


def read_retrieval_score(f, m):
    f = glob(f"{f}/*.txt")
    assert len(f) == 1

    ret = Path(f[0]).read_text(encoding="utf-8").split("\n")

    ret = [i for i in ret if m in i]
    assert len(ret) == 1
    return float(ret[0].split(": ")[-1])


def read_trec_measure(f, m):
    ret = (f/"auto-judge.eval.txt").read_text(encoding="utf-8").split("\n")
    ret = [i for i in ret if f"{m}\t" in i]
    ret = [i for i in ret if "all\t" in i.lower()]
    assert len(ret) == 1
    return float(ret[0].split("\t")[-1])


def collect_evaluation(run, measure):
    measure_dir = run.parent.parent / "evals" / MEASURES_TO_REPORT[measure]["file"]
    assert measure_dir.is_dir()

    if measure_dir.name == "gold-answer-comparison":
        return read_from_jsonl(measure_dir / run.name, measure)

    if measure_dir.name == "retrieval":
        return read_retrieval_score(measure_dir / run.name, measure)

    if "trec-auto-judge" in measure_dir.name:
        return read_trec_measure(measure_dir / run.name, measure)
                
    raise ValueError("Unsupported format")

def collect_evaluations(run):
    ret = {}
    for m in MEASURES_TO_REPORT.keys():
        ret[m] = collect_evaluation(run, m)
    return ret

def main(base_dir):
    ret = []
    for run in sorted(glob(f"{base_dir}/*")):
        run = Path(run)
        ev = collect_evaluations(run)
        ret += [run.name + " & " + (" & ".join(f"{ev[i]:.3f}" for i in MEASURES_TO_REPORT.keys()) + "\\\\\n")]
    Path(base_dir.parent / "table-evaluation.tex").write_text(
        TABLE_PREFIX + ("\n\n".join(ret)) + TABLE_SUFFIX,
        encoding="utf-8",
    )

if __name__ == '__main__':
    main(Path(__file__).parent / "runs")
