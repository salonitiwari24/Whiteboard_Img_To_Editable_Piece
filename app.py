"""
Flowchart to Editable Diagram — Streamlit UI (Workflow Diagram Analyzer engine V5.1)

- Upload a diagram
- Real-time progress bar
- Preview + annotated overlay
- Statistics + summary + warnings
- Download: Excel, JSON, Annotated image
- No hardcoded outputs, everything is derived from the uploaded pixels.
"""
import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src.pipeline import DiagramPipeline
from src.config import PipelineConfig
from src.excel_export import export_to_bytes as excel_export_bytes
from src.llm_cleanup import ollama_available
from src.editable_diagram import (
    build_digraph_from_pipeline,
    render_digraph_to_graphviz,
    export_graphviz_to_bytes,
)

# ── Cache the pipeline so EasyOCR loads only once, not on every button click ──
@st.cache_resource
def get_pipeline(
    ollama_host: str,
    vision_model: str,
    text_model: str,
    ocr_engine: str,
    min_ocr_confidence: float,
    use_vlm: bool,
    vlm_snap: bool,
    use_text_llm: bool,
    strict_edges: bool,
    max_deg: int,
    enable_debug: bool,
    min_arrow_confidence: float,
) -> DiagramPipeline:
    """Returns a cached DiagramPipeline. Re-creates it only when settings change."""
    config = PipelineConfig(
        ollama_host=ollama_host,
        vision_model=vision_model,
        text_model=text_model,
        min_ocr_confidence=min_ocr_confidence,
        min_arrow_confidence=min_arrow_confidence,
        use_text_cleanup=use_text_llm,
        use_vlm_detection=use_vlm,
        vlm_snap_boxes=vlm_snap,
        keep_debug=enable_debug,
        require_arrowhead_or_shared_component=strict_edges,
        max_edges_per_node=max_deg,
    )
    return DiagramPipeline(config)


def render_table(rows: list) -> None:
    """Render a list of dicts as a plain HTML table."""
    if not rows:
        st.caption("No data.")
        return
    cols = list(rows[0].keys())
    thead = "".join(f"<th>{c}</th>" for c in cols)
    trs = []
    for r in rows:
        tds = "".join(f"<td>{'' if r.get(c) is None else r.get(c)}</td>" for c in cols)
        trs.append(f"<tr>{tds}</tr>")
    html = f"""
    <div style="max-height:420px; overflow:auto; border:1px solid rgba(148,163,184,0.25); border-radius:10px;">
    <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
      <thead style="position:sticky; top:0; background:#1f2937; color:#fff;">
        <tr>{thead}</tr>
      </thead>
      <tbody>
        {''.join(trs)}
      </tbody>
    </table>
    </div>
    <style>
      table td, table th {{ padding:6px 10px; border-bottom:1px solid rgba(148,163,184,0.18); text-align:left; }}
      table tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.03); }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flowchart to Editable Diagram",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #111827 40%, #0f172a 100%);
        color: #e5e7eb;
    }
    .main-card {
        background: rgba(17, 24, 39, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 20px 45px rgba(0,0,0,0.25);
    }
    .hero {
        padding: 1.3rem 1.4rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(37,99,235,0.32), rgba(16,185,129,0.18));
        border: 1px solid rgba(96,165,250,0.24);
        margin-bottom: 1rem;
    }
    .small-note { color: #cbd5e1; font-size: 0.93rem; }
    .pill {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background: rgba(59,130,246,0.18); border:1px solid rgba(96,165,250,0.35);
        color:#dbeafe; font-size:0.82rem; margin-right:6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("result", "excel_bytes", "annotated_bytes", "editable_graph"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 style="margin:0; font-size:2rem;">🧭 Flowchart to Editable Diagram <span class="pill">V5.1</span></h1>
    <p style="margin:0.45rem 0 0 0; color:#dbeafe; font-size:1.03rem;">
        Local-first pipeline — <b>OpenCV + OCR + local Ollama</b>.
        Accurate edge detection. Excel + JSON + annotated PNG downloads.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("100% local. Ollama is optional but improves labels & summary.")

    _defaults = PipelineConfig()

    ollama_host  = st.text_input("Ollama host", value=_defaults.ollama_host)

    st.divider()
    st.subheader("🧠 Vision detection")
    use_vlm = st.checkbox(
        "Use vision LLM as primary detector (recommended)",
        value=_defaults.use_vlm_detection,
        help="Reads the whole diagram with a local vision model. Falls back to classical CV "
             "automatically if the model isn't pulled or Ollama isn't reachable.",
    )
    vision_model = st.text_input(
        "Vision LLM (Ollama)", value=_defaults.vision_model,
        help="Pull first: `ollama pull qwen2.5vl:7b` (best, ~6 GB) or `ollama pull minicpm-v`.",
    )
    vlm_snap = st.checkbox(
        "Snap vision-model boxes to pixel-accurate contours",
        value=_defaults.vlm_snap_boxes,
        help="Tightens approximate VLM boxes onto real detected borders.",
    )
    text_model = st.text_input(
        "Text LLM (Ollama)", value=_defaults.text_model,
        help="~2 GB model for label cleanup + summary.",
    )

    st.divider()
    st.subheader("Classical CV fallback")
    st.caption("Only used when the vision model is unavailable.")
    min_conf = st.slider(
        "Minimum OCR confidence", 0.0, 1.0, _defaults.min_ocr_confidence, 0.05,
        help="Lower = more text detected but may include noise.",
    )
    arrow_threshold = st.slider(
        "Arrow confidence threshold", 0.0, 1.0, _defaults.min_arrow_confidence, 0.02,
        help="Higher = fewer false arrows. Default 0.51.",
    )
    strict_edges = st.checkbox(
        "Strict edge gate (recommended)",
        value=_defaults.require_arrowhead_or_shared_component,
        help="Only keep arrows with a traced stroke OR clear arrowhead.",
    )
    max_deg = st.slider(
        "Max edges per node", 1, 20, _defaults.max_edges_per_node,
        help="Cap on combined in+out degree per node.",
    )
    use_text_llm = st.checkbox("Use local LLM for label cleanup + summary", value=True)
    enable_debug = st.checkbox("Show debug artifacts", value=True)

    st.divider()
    if st.button("🔌 Check Ollama connection"):
        ok = ollama_available(ollama_host)
        (st.success if ok else st.warning)(
            f"Ollama {'reachable ✅' if ok else 'NOT reachable ❌'} at {ollama_host}"
        )

    st.divider()
    st.subheader("💡 Tips")
    st.markdown("""
- Use **screenshots**, not photos.
- Higher resolution = better detection.
- Green-highlighted nodes are detected as start states.
- Lower **Arrow threshold** if edges are missing.
- Enable **Strict edge gate** to remove phantom arrows.
""")

# ── Main columns ──────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 1.15], gap="large")

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload diagram")

    uploaded = st.file_uploader(
        "Supported: PNG, JPG, JPEG, WEBP, BMP",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded diagram", use_container_width=True)

        if st.button("🔍 Analyze diagram", type="primary", use_container_width=True):

            # Save upload to a temp file (pipeline needs a file path)
            suffix = Path(uploaded.name).suffix or ".png"
            tmp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)
            image.save(temp_path)

            # Get (or re-create if settings changed) the cached pipeline
            # EasyOCR model loads here on first call; cached for subsequent clicks
            with st.spinner("Loading models (first run may take ~30 s)..."):
                try:
                    pipeline = get_pipeline(
                        ollama_host=ollama_host.strip(),
                        vision_model=vision_model.strip(),
                        text_model=text_model.strip(),
                        ocr_engine=_defaults.ocr_engine,
                        min_ocr_confidence=float(min_conf),
                        use_vlm=bool(use_vlm),
                        vlm_snap=bool(vlm_snap),
                        use_text_llm=bool(use_text_llm),
                        strict_edges=bool(strict_edges),
                        max_deg=int(max_deg),
                        enable_debug=bool(enable_debug),
                        min_arrow_confidence=float(arrow_threshold),
                    )
                except Exception as load_exc:
                    st.error(f"Failed to load pipeline: {load_exc}")
                    st.stop()

            progress = st.progress(0, text="Starting...")

            def cb(pct, msg):
                progress.progress(min(100, int(pct)), text=msg)

            t0 = time.time()
            try:
                result = pipeline.run(temp_path, progress_cb=cb)
                elapsed = time.time() - t0

                st.session_state.result        = result
                st.session_state.excel_bytes   = excel_export_bytes(result)
                st.session_state.annotated_bytes = result.get("annotated_image_png", b"")
                st.session_state.editable_graph  = build_digraph_from_pipeline(
                    result.get("nodes", []), result.get("edges", [])
                )

                method       = result.get("detection_method", "classical_cv")
                method_label = "vision LLM" if method == "vlm" else "classical CV (fallback)"
                progress.progress(100, text="Done!")
                st.success(
                    f"✅ Analysis complete in {elapsed:.1f}s  •  detector: **{method_label}**  •  "
                    f"confidence **{int(result['confidence'] * 100)}%**"
                )
                if method == "classical_cv" and use_vlm:
                    st.info(
                        f"Vision model **{vision_model}** wasn't reachable — classical CV ran instead. "
                        f"Run `ollama pull {vision_model}` to enable the VLM path."
                    )

            except Exception as exc:
                st.exception(exc)
            finally:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    st.markdown("</div>", unsafe_allow_html=True)

# ── Right column: results ─────────────────────────────────────────────────────
with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🧾 Structured output")

    result = st.session_state.result

    if result is None:
        st.info("Upload a workflow image and click **Analyze diagram**.")
    else:
        stats = result.get("statistics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes",      stats.get("node_count", 0))
        c2.metric("Edges",      stats.get("edge_count", 0))
        c3.metric("Starts",     stats.get("start_count", 0))
        c4.metric("Ends",       stats.get("end_count", 0))
        c5.metric("Confidence", f"{int(result.get('confidence', 0.0) * 100)}%")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🖼️ Annotated", "📋 Summary", "📊 Data", "⚠️ Warnings", "⬇️ Download"]
        )

        # ── Annotated image ───────────────────────────────────────────────────
        with tab1:
            if st.session_state.annotated_bytes:
                st.image(
                    st.session_state.annotated_bytes,
                    caption="Detected nodes & connectors",
                    use_container_width=True,
                )
            else:
                st.write("Annotated image unavailable.")

        # ── Summary ───────────────────────────────────────────────────────────
        with tab2:
            st.markdown("### Extracted summary")
            st.write(result.get("summary") or "_No summary available._")
            obs = result.get("observations", [])
            if obs:
                st.markdown("### Observations")
                for o in obs:
                    st.write(f"- {o}")
            st.markdown("### Start / End states")
            id2lbl = {n["id"]: n["label"] for n in result.get("nodes", [])}
            starts_display = ", ".join(
                f"{sid} · {id2lbl.get(sid, '')}" for sid in result.get("start_states", [])
            ) or "—"
            ends_display = ", ".join(
                f"{eid} · {id2lbl.get(eid, '')}" for eid in result.get("end_states", [])
            ) or "—"
            st.write(f"**Start:** {starts_display}")
            st.write(f"**End:** {ends_display}")

        # ── Data tables ───────────────────────────────────────────────────────
        with tab3:
            st.markdown("#### Nodes")
            render_table([
                {
                    "ID":     n["id"],
                    "Label":  n["label"],
                    "Shape":  n["shape"],
                    "Style":  n.get("style", ""),
                    "Conf.":  n["confidence"],
                    "Source": n.get("source", ""),
                    "Center": f"({n['center'][0]}, {n['center'][1]})",
                }
                for n in result.get("nodes", [])
            ])
            st.markdown("#### Connections")
            id2lbl = {n["id"]: n["label"] for n in result.get("nodes", [])}
            render_table([
                {
                    "ID":        e["id"],
                    "From":      f"{e['source']} · {id2lbl.get(e['source'], '')}",
                    "To":        f"{e['target']} · {id2lbl.get(e['target'], '')}",
                    "Direction": e.get("direction", ""),
                    "Routing":   e.get("routing", ""),
                    "Conf.":     e.get("confidence", 0.0),
                    "Arrowhead": "Yes" if e.get("has_arrowhead") else "No",
                }
                for e in result.get("edges", [])
            ])
            st.markdown("#### Statistics")
            st.json(stats)

        # ── Warnings ──────────────────────────────────────────────────────────
        with tab4:
            uncs = result.get("uncertainties", [])
            if not uncs:
                st.success("No warnings — every element passed the confidence checks.")
            else:
                for u in uncs:
                    st.warning(u)
            low_nodes = [n for n in result.get("nodes", []) if n.get("confidence", 1.0) < 0.45]
            low_edges = [e for e in result.get("edges", []) if e.get("confidence", 1.0) < 0.40]
            if low_nodes:
                st.markdown("**Low-confidence nodes:**")
                for n in low_nodes:
                    st.write(f"- {n['id']} — '{n['label']}' (conf {n['confidence']})")
            if low_edges:
                st.markdown("**Low-confidence edges:**")
                for e in low_edges:
                    st.write(f"- {e['id']} {e['source']} → {e['target']} (conf {e['confidence']})")

        # ── Downloads ─────────────────────────────────────────────────────────
        with tab5:
            st.markdown("Download the analysis results:")
            colA, colB, colC = st.columns(3)
            if st.session_state.excel_bytes:
                colA.download_button(
                    "⬇️ Excel (.xlsx)",
                    data=st.session_state.excel_bytes,
                    file_name="workflow_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            json_payload  = {k: v for k, v in result.items() if k != "annotated_image_png"}
            json_bytes    = json.dumps(json_payload, indent=2, ensure_ascii=False).encode("utf-8")
            colB.download_button(
                "⬇️ JSON",
                data=json_bytes,
                file_name="workflow_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
            if st.session_state.annotated_bytes:
                colC.download_button(
                    "⬇️ Annotated PNG",
                    data=st.session_state.annotated_bytes,
                    file_name="workflow_annotated.png",
                    mime="image/png",
                    use_container_width=True,
                )

        # ── Editable diagram ──────────────────────────────────────────────────
        # Rebuild editable graph if session state was cleared
        if st.session_state.editable_graph is None and result is not None:
            st.session_state.editable_graph = build_digraph_from_pipeline(
                result.get("nodes", []), result.get("edges", [])
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ── Edit & Export panel (below both columns) ──────────────────────────────────
if st.session_state.get("editable_graph") is not None:
    st.divider()
    with st.expander("✏️ Edit & Export Diagram", expanded=True):
        g = st.session_state.editable_graph

        # Render the DOT once per render cycle
        dot_obj = render_digraph_to_graphviz(g)

        st.subheader("Interactive Digital Diagram")
        st.graphviz_chart(dot_obj.source, use_container_width=True)

        st.markdown("---")
        st.subheader("Diagram Editing & Export Controls")

        edit_tab1, edit_tab2, edit_tab3, edit_tab4, edit_tab5 = st.tabs([
            "✏️ Rename Node", "➕ Add Node", "🗑️ Delete Node",
            "🔗 Edit Edges",  "📥 Export PNG / PDF",
        ])

        node_ids  = list(g.nodes())
        edge_list = list(g.edges())

        # ── Rename node ───────────────────────────────────────────────────────
        with edit_tab1:
            if not node_ids:
                st.info("No nodes available to rename.")
            else:
                sel = st.selectbox(
                    "Select node to rename", node_ids,
                    format_func=lambda nid: f"{nid}: {g.nodes[nid].get('label', nid)}",
                    key="sb_rename_node",
                )
                new_lbl_rename = st.text_input(
                    "New label",
                    value=g.nodes[sel].get("label", sel),
                    key=f"txt_rename_{sel}",
                )
                if st.button("Update Label", type="primary", key="btn_rename_node"):
                    g.nodes[sel]["label"] = new_lbl_rename.strip()
                    st.session_state.editable_graph = g
                    st.success(f"Renamed {sel} → '{new_lbl_rename.strip()}'")
                    st.rerun()

        # ── Add node ──────────────────────────────────────────────────────────
        with edit_tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                new_node_lbl   = st.text_input("New node label", placeholder="e.g. Review Step", key="txt_add_node_lbl")
                new_node_shape = st.selectbox(
                    "Shape",
                    ["box", "rounded_rectangle", "diamond", "ellipse", "parallelogram", "cylinder"],
                    key="sb_add_node_shape",
                )
            with col_b:
                conn_from = st.selectbox(
                    "Connect from (optional)",
                    ["(None)"] + node_ids,
                    format_func=lambda nid: nid if nid == "(None)" else f"{nid}: {g.nodes[nid].get('label', nid)}",
                    key="sb_add_node_conn",
                )
            if st.button("Add Node", type="primary", key="btn_add_node"):
                if not new_node_lbl.strip():
                    st.warning("Please enter a node label.")
                else:
                    idx = 1
                    while f"N{idx}" in g.nodes():
                        idx += 1
                    new_nid = f"N{idx}"
                    g.add_node(new_nid, label=new_node_lbl.strip(),
                               type=new_node_shape, shape=new_node_shape,
                               bbox=[0, 0, 0, 0], confidence=1.0)
                    if conn_from != "(None)":
                        g.add_edge(conn_from, new_nid, confidence=1.0)
                    st.session_state.editable_graph = g
                    st.success(f"Added node {new_nid} ('{new_node_lbl.strip()}')")
                    st.rerun()

        # ── Delete node ───────────────────────────────────────────────────────
        with edit_tab3:
            if not node_ids:
                st.info("No nodes to delete.")
            else:
                del_id = st.selectbox(
                    "Select node to delete", node_ids,
                    format_func=lambda nid: f"{nid}: {g.nodes[nid].get('label', nid)}",
                    key="sb_del_node",
                )
                st.caption("Deleting a node also removes all its edges.")
                if st.button("Delete Node", type="primary", key="btn_del_node"):
                    g.remove_node(del_id)
                    st.session_state.editable_graph = g
                    st.success(f"Deleted {del_id} and its edges.")
                    st.rerun()

        # ── Edit edges ────────────────────────────────────────────────────────
        with edit_tab4:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown("#### Add Edge")
                if len(node_ids) < 2:
                    st.info("Need at least 2 nodes.")
                else:
                    with st.form("add_edge_form"):
                        src_e = st.selectbox(
                            "Source", node_ids,
                            format_func=lambda nid: f"{nid}: {g.nodes[nid].get('label', nid)}",
                            key="sb_edge_src",
                        )
                        tgt_e = st.selectbox(
                            "Target", node_ids,
                            format_func=lambda nid: f"{nid}: {g.nodes[nid].get('label', nid)}",
                            key="sb_edge_tgt",
                        )
                        if st.form_submit_button("Add Edge", type="primary"):
                            if src_e == tgt_e:
                                st.warning("Source and target must be different nodes.")
                            else:
                                g.add_edge(src_e, tgt_e, confidence=1.0)
                                st.session_state.editable_graph = g
                                st.success(f"Added {src_e} ➔ {tgt_e}")
                                st.rerun()

            with col_e2:
                st.markdown("#### Delete Edge")
                if not edge_list:
                    st.info("No edges to delete.")
                else:
                    sel_edge = st.selectbox(
                        "Select edge",
                        edge_list,
                        format_func=lambda e: (
                            f"{e[0]} ({g.nodes[e[0]].get('label', e[0])}) ➔ "
                            f"{e[1]} ({g.nodes[e[1]].get('label', e[1])})"
                        ),
                        key="sb_del_edge",
                    )
                    if st.button("Delete Edge", type="primary", key="btn_del_edge"):
                        g.remove_edge(sel_edge[0], sel_edge[1])
                        st.session_state.editable_graph = g
                        st.success(f"Deleted {sel_edge[0]} ➔ {sel_edge[1]}")
                        st.rerun()

        # ── Export ────────────────────────────────────────────────────────────
        with edit_tab5:
            st.markdown("Export the current (edited) diagram:")
            if st.button("Generate PNG & PDF", type="primary", key="btn_gen_exports"):
                with st.spinner("Rendering with Graphviz..."):
                    try:
                        exp_png = export_graphviz_to_bytes(dot_obj, "png")
                        exp_pdf = export_graphviz_to_bytes(dot_obj, "pdf")
                        st.session_state["_exp_png"] = exp_png
                        st.session_state["_exp_pdf"] = exp_pdf
                        st.success("Render complete — click below to download.")
                    except Exception as exc:
                        st.error(f"Graphviz render failed: {exc}")
                        st.info(
                            "Make sure the Graphviz system binary is installed and on PATH. "
                            "Check with: `dot -V`"
                        )

            exp_col1, exp_col2 = st.columns(2)
            if st.session_state.get("_exp_png"):
                exp_col1.download_button(
                    "⬇️ Download PNG",
                    data=st.session_state["_exp_png"],
                    file_name="edited_diagram.png",
                    mime="image/png",
                    use_container_width=True,
                    key="btn_dl_png",
                )
            if st.session_state.get("_exp_pdf"):
                exp_col2.download_button(
                    "⬇️ Download PDF",
                    data=st.session_state["_exp_pdf"],
                    file_name="edited_diagram.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_dl_pdf",
                )

st.markdown(
    '<p class="small-note">V5.1 — preprocess → OCR → nodes → arrows → graph → LLM cleanup. '
    'Position-aware editable diagram (neato layout). No hardcoded outputs.</p>',
    unsafe_allow_html=True,
)
