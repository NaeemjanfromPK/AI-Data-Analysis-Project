from ..export import export_results_to_excel


def excel_export_node(state):
    """Write RESULTS to a structured .xlsx workbook alongside the report."""
    run_dir = state.get("run_dir")
    if run_dir:
        export_results_to_excel(run_dir)
    return {}