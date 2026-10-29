"""Research Demonstration Dashboard for Early-Stage Ransomware Detection.

Technology: Streamlit Research Demonstration Interface.
Security & Defensive Safety Guarantee:
- This dashboard is strictly a telemetry viewer and ML results visualizer.
- It NEVER executes uploaded EXE/DLL binaries or arbitrary commands.
- File inspection is restricted exclusively to defensive CSV/JSON research telemetry.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Ensure working directory is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_streamlit_app():
    st.set_page_config(
        page_title="Ransomware ML Research Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -------------------------------------------------------------
    # Sidebar Navigation & Safety Notice
    # -------------------------------------------------------------
    st.sidebar.title("🛡️ Research Navigation")
    section = st.sidebar.radio(
        "Navigate Pipeline",
        [
            "1. Project Overview",
            "2. Dataset Statistics",
            "3. Model Comparison",
            "4. Confusion Matrix",
            "5. Feature Importance",
            "6. Temporal Observation Viewer",
            "7. PPO Decision Trace",
            "8. Research Limitations",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Defensive Cybersecurity ML Research")
    st.sidebar.info(
        "🔒 **Safety Protocol Enforced**:\n"
        "No malware binary execution permitted. Research telemetry viewer only."
    )

    # -------------------------------------------------------------
    # Section 1: Project Overview
    # -------------------------------------------------------------
    if section == "1. Project Overview":
        st.title("🛡️ Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning")
        st.markdown(
            """
            ### Research Objective
            Build an end-to-end, scientifically grounded, reproducible defensive ML pipeline comparing:
            - **Static Classifiers**: Random Forest & XGBoost on sandbox dynamic features.
            - **Early-Stage Feature-Restricted Models**: 16 non-redundant early indicators.
            - **Temporal Multi-Window Baselines**: Causal delta, rate, and rolling statistics.
            - **Sequential Reinforcement Learning**: Proximal Policy Optimization (PPO) for early stopping alerts.
            """
        )

        st.info(
            "⚠️ **RESEARCH STATUS & CAVEAT**: Temporal and PPO results are evaluated on synthetic demonstration "
            "telemetry designed for schema/pipeline validation. No live malware was executed."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pipeline Phases", "18 Phases", "Completed")
        col2.metric("Models Compared", "7 Architectures", "Unified")
        col3.metric("Static Samples", "1,750 Cleaned", "Sandbox Ground Truth")
        col4.metric("Temporal Features", "60 Engineered", "Strictly Causal")

    # -------------------------------------------------------------
    # Section 2: Dataset Statistics
    # -------------------------------------------------------------
    elif section == "2. Dataset Statistics":
        st.title("📊 Dataset Statistics & Preprocessing Analysis")
        st.markdown(
            """
            Comparison between the original raw dynamic dataset and the cleaned, validated research dataset.
            """
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Raw Dataset Rows", "2,000", "50 Features")
        col2.metric("Cleaned Dataset Rows", "1,750", "46 Features (-250 duplicates/constants)")
        col3.metric("Early-Stage Subset", "16 Features", "Non-redundant")

        st.subheader("Class Distribution (Ground Truth Sandbox Dataset)")
        dist_df = pd.DataFrame({
            "Ransomware Family": ["Family G", "Family E", "Family L (Minority)"],
            "Sample Count": [820, 767, 163],
            "Percentage": ["46.86%", "43.83%", "9.31%"]
        })
        st.dataframe(dist_df, use_container_width=True)

        st.bar_chart(dist_df.set_index("Ransomware Family")["Sample Count"])

    # -------------------------------------------------------------
    # Section 3: Model Comparison
    # -------------------------------------------------------------
    elif section == "3. Model Comparison":
        st.title("📈 Unified Model Comparison Across All 7 Architectures")
        st.markdown(
            """
            Standardized evaluation table contrasting supervised classifiers with the sequential PPO decision agent.
            """
        )

        comp_csv = PROJECT_ROOT / "results" / "reports" / "final_model_comparison.csv"
        if comp_csv.exists():
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.warning("final_model_comparison.csv not found.")

        comp_fig = PROJECT_ROOT / "results" / "figures" / "final_model_comparison.png"
        if comp_fig.exists():
            st.image(str(comp_fig), caption="Comparative Performance Progression", use_column_width=True)

    # -------------------------------------------------------------
    # Section 4: Confusion Matrix
    # -------------------------------------------------------------
    elif section == "4. Confusion Matrix":
        st.title("🎯 Cross-Paradigm Confusion Matrices")
        st.markdown("Visualizing classification precision across static, temporal, and PPO models.")

        fig_cm = PROJECT_ROOT / "results" / "figures" / "final_confusion_matrices.png"
        if fig_cm.exists():
            st.image(str(fig_cm), caption="Confusion Matrices: Static RF, Temporal RF, and Temporal PPO", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Temporal Random Forest (Window 4)")
            rf_cm = PROJECT_ROOT / "results" / "figures" / "temporal_rf_confusion_matrix.png"
            if rf_cm.exists():
                st.image(str(rf_cm), use_column_width=True)

        with col2:
            st.subheader("Temporal PPO Policy (Stopping Decision)")
            ppo_cm = PROJECT_ROOT / "results" / "figures" / "temporal_ppo_confusion_matrix.png"
            if ppo_cm.exists():
                st.image(str(ppo_cm), use_column_width=True)

    # -------------------------------------------------------------
    # Section 5: Feature Importance
    # -------------------------------------------------------------
    elif section == "5. Feature Importance":
        st.title("🔍 Predictive Feature Importance Across Regimes")
        st.markdown("Contrasting feature rankings across static, early-stage, and temporal feature sets.")

        imp_csv = PROJECT_ROOT / "results" / "reports" / "final_feature_importance.csv"
        if imp_csv.exists():
            df_imp = pd.read_csv(imp_csv)
            st.dataframe(df_imp, use_container_width=True)

        imp_fig = PROJECT_ROOT / "results" / "figures" / "final_feature_importance.png"
        if imp_fig.exists():
            st.image(str(imp_fig), caption="Top Predictive Features Across Experimental Regimes", use_column_width=True)

    # -------------------------------------------------------------
    # Section 6: Temporal Observation Viewer
    # -------------------------------------------------------------
    elif section == "6. Temporal Observation Viewer":
        st.title("⏱️ Sequential Temporal Observation Viewer")
        st.markdown("Interactive inspection of multi-window telemetry ($W_0 \\to W_4$).")

        demo_csv = PROJECT_ROOT / "data" / "processed" / "synthetic_temporal_demo.csv"
        if demo_csv.exists():
            df_demo = pd.read_csv(demo_csv)
            sample_ids = df_demo["sample_id"].unique().tolist()
            selected_sample = st.selectbox("Select Process Sample", sample_ids)

            sub_sample = df_demo[df_demo["sample_id"] == selected_sample].sort_values("observation_window")
            st.dataframe(sub_sample, use_container_width=True)

            st.line_chart(sub_sample.set_index("observation_window")[["entropy", "file_write_count", "api_rate"]])
        else:
            st.warning("synthetic_temporal_demo.csv not found.")

    # -------------------------------------------------------------
    # Section 7: PPO Decision Trace
    # -------------------------------------------------------------
    elif section == "7. PPO Decision Trace":
        st.title("🤖 PPO Sequential Decision Traces")
        st.markdown("Step-by-step sequential actions and reward feedback for test samples.")

        trace_csv = PROJECT_ROOT / "results" / "reports" / "ppo_decision_traces.csv"
        if trace_csv.exists():
            df_traces = pd.read_csv(trace_csv)
            st.dataframe(df_traces, use_container_width=True)

        act_fig = PROJECT_ROOT / "results" / "figures" / "temporal_ppo_action_distribution.png"
        if act_fig.exists():
            st.image(str(act_fig), caption="PPO Action Distribution Across Observation Windows", use_column_width=True)

    # -------------------------------------------------------------
    # Section 8: Research Limitations
    # -------------------------------------------------------------
    elif section == "8. Research Limitations":
        st.title("⚖️ Research Limitations & Scientific Integrity")

        st.warning(
            "**MANDATORY RESEARCH CAVEAT**:\n\n"
            "Current temporal/PPO results are proof-of-concept results using synthetic temporal observations "
            "unless real temporal telemetry is available."
        )

        st.markdown(
            """
            ### Scientific Limitations Documented:
            1. **Static Origin of the Benchmark**:
               The primary 2,000-sample dataset aggregates dynamic activity into single cumulative numbers without microsecond timestamps.
            2. **Synthetic Temporal Demonstration**:
               The multi-window sequence ($W_0 \\to W_4$) demonstrates schema validation, mathematical deltas, and causal rolling moments. It is not live ransomware telemetry.
            3. **Sample Size Constraints**:
               The temporal demo dataset contains 8 samples ($N=8$). While structurally sound, statistical power requires physical sandbox execution.
            4. **Absence of Ground-Truth Latency**:
               Detection latency cannot be claimed in absolute seconds until verified against canary file destruction timestamps in an authorized sandbox.
            """
        )


def run_terminal_fallback():
    """Terminal / text-based fallback when Streamlit is not installed."""
    print("=" * 75)
    print("EARLY-STAGE RANSOMWARE DETECTION RESEARCH DASHBOARD")
    print("=" * 75)
    print("Technology: Python Streamlit Research Interface")
    print("File: dashboard/app.py")
    print("\nSections Available:")
    print("  1. Project Overview")
    print("  2. Dataset Statistics (1,750 cleaned / 2,000 raw samples)")
    print("  3. Model Comparison (7 models: Static, Early, Temporal, PPO)")
    print("  4. Confusion Matrix (Multi-panel comparative matrices)")
    print("  5. Feature Importance (Cross-paradigm rankings)")
    print("  6. Temporal Observation Viewer (Sequential W0-W4 telemetry)")
    print("  7. PPO Decision Trace (Step-by-step actions and rewards)")
    print("  8. Research Limitations (Mandatory synthetic data caveats)")
    print("=" * 75)
    print("To launch the interactive GUI, run:\n    streamlit run dashboard/app.py")
    print("=" * 75)


if __name__ == "__main__":
    if HAS_STREAMLIT:
        run_streamlit_app()
    else:
        run_terminal_fallback()
