"""Add explicit terminators to the two approved manuscript flowcharts.

Read the existing code/vector sources; preserve historical renders and all
existing nodes/branches. No training, data or measurement work is performed.
"""
import ast
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "docs/capstone_papers/review_assets"))
import build_review_assets as bra
from matplotlib.patches import Ellipse


def application_navigation():
    source = (ROOT / "artifacts/reports/paper_condensation_20261014/build_diagrams.py").read_text()
    tree = ast.parse(source)
    helper_nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {"__file__": str(OUT / "build_flowcharts.py")}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(OUT / "helpers"), "exec"), namespace)
    namespace.update(P=OUT, BLUE="#245b85", TEAL="#14776f")
    original_node = namespace["node"]

    def node(ax, x, y, text, w=2.5, h=.7, kind="box"):
        if kind == "terminator":
            ax.add_patch(Ellipse((x, y), w, h, facecolor="#eaf4f1", edgecolor=namespace["TEAL"]))
            ax.text(x, y, text, ha="center", va="center", fontsize=10)
        else:
            original_node(ax, x, y, text, w=w, h=h, kind=kind)

    namespace["node"] = node
    section = source.split("# Compact navigation; functional loops are explicit.", 1)[1].split("# Streaming decisions.", 1)[0]
    section = section.replace("f,a=canvas(15,9)", "f,a=canvas(16.4,10.3)")
    section = section.replace("save(f,'app_flow')", "")
    exec(compile(section, str(OUT / "navigation"), "exec"), namespace)
    ax = namespace["a"]
    arrow = namespace["arrow"]
    node(ax, 7.5, 9.65, "Start", w=1.6, h=.65, kind="terminator")
    arrow(ax, [(7.5, 9.325), (7.5, 8.89)])
    node(ax, 15.6, .9, "End", w=1.2, h=.65, kind="terminator")
    arrow(ax, [(9.54, 8.5), (15.6, 8.5), (15.6, 1.225)], "Close application", (15.6, 6.7))
    namespace["save"](namespace["f"], "app_flow")


def processing_pipeline():
    source = ROOT / "artifacts/reports/post_defense_metrics_v17_20261027/plot_pipeline_flowchart.py"
    tree = ast.parse(source.read_text())
    call = next(n.value for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "flow")
    name, title, nodes, edges, figsize, limits = [ast.literal_eval(a) for a in call.args]
    nodes = {"start": (6, 30.7, "Start", "terminator"), **nodes,
             "end": (6, .3, "End", "terminator")}
    edges = [("start", "camera", ""), *edges, ("output", "end", "")]
    # Reuse the paper renderer with an explicit terminator shape and boundary.
    helper_source = (ROOT / "docs/capstone_papers/review_assets/build_review_assets.py").read_text()
    helper_tree = ast.parse(helper_source)
    flow_node = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == "flow")
    flow_source = ast.get_source_segment(helper_source, flow_node)
    flow_source = flow_source.replace("('{' if kind == 'decision' else '[')", "('([' if kind == 'terminator' else '{' if kind == 'decision' else '[')")
    flow_source = flow_source.replace("('}' if kind == 'decision' else ']')", "('])' if kind == 'terminator' else '}' if kind == 'decision' else ']')")
    flow_source = flow_source.replace("w, h = (3.5, 1.45) if kind == 'decision' else (3.35, 1.0)", "w, h = (1.8, .7) if kind == 'terminator' else (3.5, 1.45) if kind == 'decision' else (3.35, 1.0)")
    flow_source = flow_source.replace("if kind == 'decision':\n            patch", "if kind == 'terminator':\n            patch = Ellipse((x,y),w,h,facecolor='#e7f3ef',edgecolor=TEAL,linewidth=1.2)\n        elif kind == 'decision':\n            patch", 1)
    flow_source = flow_source.replace("w, h = (1.75, .725) if kind == 'decision' else (1.675, .5)", "w, h = (.9, .35) if kind == 'terminator' else (1.75, .725) if kind == 'decision' else (1.675, .5)")
    bra.HERE = OUT
    namespace = {**vars(bra), "Ellipse": Ellipse}
    exec(compile(flow_source, str(OUT / "pipeline_renderer"), "exec"), namespace)
    namespace["flow"](name, title, nodes, edges, (9.5, 22), (12.2, 31.5))
    (OUT / "pipeline_structure.json").write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2) + "\n")


if __name__ == "__main__":
    application_navigation()
    processing_pipeline()
    print("Rendered navigation and pipeline with Start/End terminators")
