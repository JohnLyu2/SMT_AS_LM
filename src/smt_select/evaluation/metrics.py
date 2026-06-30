"""Metrics and presentation for algorithm-selection evaluation."""

import logging


def compute_metrics(result_dataset, multi_perf_data) -> dict:
    total_count = len(result_dataset)
    solved_count = result_dataset.get_solved_count()
    avg_par2 = (
        sum(result_dataset.get_par2(path) for path in result_dataset.keys())
        / total_count
        if total_count
        else 0.0
    )
    sbs = multi_perf_data.get_best_solver_dataset()
    vbs = multi_perf_data.get_virtual_best_solver_dataset()
    sbs_solved, vbs_solved = sbs.get_solved_count(), vbs.get_solved_count()
    sbs_par2 = sum(sbs.get_par2(path) for path in sbs.keys()) / total_count if total_count else 0.0
    vbs_par2 = sum(vbs.get_par2(path) for path in vbs.keys()) / total_count if total_count else 0.0
    solved_denominator = vbs_solved - sbs_solved
    par2_denominator = vbs_par2 - sbs_par2
    return {
        "total_count": total_count,
        "solved": solved_count,
        "avg_par2": avg_par2,
        "sbs_solved": sbs_solved,
        "sbs_avg_par2": sbs_par2,
        "vbs_solved": vbs_solved,
        "vbs_avg_par2": vbs_par2,
        "gap_cls_solved": (
            (solved_count - sbs_solved) / solved_denominator
            if solved_denominator
            else float(solved_count == vbs_solved)
        ),
        "gap_cls_par2": (
            (avg_par2 - sbs_par2) / par2_denominator
            if par2_denominator
            else float(avg_par2 == vbs_par2)
        ),
    }


def format_evaluation_short(metrics: dict) -> str:
    return "\n".join(
        [
            f"Instances: {metrics['total_count']}, Solved: {metrics['solved']}",
            f"Avg PAR2: {metrics['avg_par2']:.2f}, gap closed (PAR-2): "
            f"{metrics['gap_cls_par2'] * 100:.2f}%",
        ]
    )


def log_evaluation_summary(metrics: dict, multi_perf_data) -> None:
    sbs = multi_perf_data.get_best_solver_dataset()
    vbs = multi_perf_data.get_virtual_best_solver_dataset()
    logging.info("=" * 60)
    logging.info("Evaluation Results:")
    logging.info("  Total instances: %d", metrics["total_count"])
    logging.info("  Solved: %d", metrics["solved"])
    logging.info("  Average PAR-2: %.2f", metrics["avg_par2"])
    logging.info("  Gap closed (solved): %.2f%%", metrics["gap_cls_solved"] * 100)
    logging.info("  Gap closed (PAR-2): %.2f%%", metrics["gap_cls_par2"] * 100)
    logging.info("SBS: %s; solved %d; PAR-2 %.2f", sbs.get_solver_name(), metrics["sbs_solved"], metrics["sbs_avg_par2"])
    logging.info("VBS: %s; solved %d; PAR-2 %.2f", vbs.get_solver_name(), metrics["vbs_solved"], metrics["vbs_avg_par2"])
    logging.info("=" * 60)
