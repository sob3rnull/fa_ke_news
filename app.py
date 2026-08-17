"""
Streamlit UI for the fake news detector.

Five tabs:
  Classify  - paste text / a URL / a screenshot, get a prediction *and* the
              evidence behind it, then flag whether it was right
  Batch     - upload a CSV, score every row, download the result
  Data      - the EDA findings, including the leakage audit
  Model     - metrics, baselines, model card, evaluation figures
  History   - everything the app has classified, with analytics and export

Run with:
    python -m streamlit run app.py

Prerequisites:
    python src/setup_data.py    (extracts data/Fake.csv + True.csv)
    python src/train.py         (writes models/)
    python src/eda.py           (optional - fills the Data tab)
    python src/evaluate.py      (optional - fills the Model tab)

Layout note: compute is kept separate from render throughout. Streamlit reruns
the whole script on every interaction, and `st.button(...)` is True only on the
run right after its click - so anything drawn inside `if st.button(...)` is
destroyed by the next interaction. Results go into session state; the render
functions read from there and run unconditionally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import subprocess

import joblib
import pandas as pd
import streamlit as st

from config import (DATA_DIR, DB_PATH, FIGURES_DIR, LABEL_NAMES, METRICS_JSON,
                    MODEL_CARD, MODEL_KEYS, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR,
                    RUN_META, VECTORIZER_FILE, model_name, model_path)
from db import (fetch_predictions, log_prediction, record_feedback,
                summary_stats)
from explain import explain_prediction, global_top_features
from preprocessing import clean_text, text_stats

# extract.py pulls in optional scraping/OCR stacks. Import it lazily so a
# missing bs4 or Tesseract degrades the URL/Screenshot inputs instead of
# taking down the whole app.
try:
    from extract import extract_text_from_image, extract_text_from_url
    EXTRACT_ERROR = None
except ImportError as _exc:  # pragma: no cover - depends on local install
    extract_text_from_image = extract_text_from_url = None
    EXTRACT_ERROR = str(_exc)


def _bootstrap_pipeline() -> None:
    """
    First-run setup for a fresh deploy container.

    Streamlit Community Cloud gives every deploy a brand-new filesystem, and
    data/*.csv + models/*.joblib are gitignored (they're derived artifacts,
    not source). This extracts the dataset from the committed archive.zip
    and trains the models if they aren't already on disk. Safe to call on
    every app startup - each step no-ops once its output already exists, so
    subsequent reruns/restarts are instant.
    """
    need_data = not (DATA_DIR / "Fake.csv").exists() or not (DATA_DIR / "True.csv").exists()
    if need_data:
        with st.spinner("First run: extracting dataset from archive.zip..."):
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "setup_data.py")],
                capture_output=True, text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                st.error("Failed to extract the dataset from archive.zip. Check the app logs.")
                print(result.stderr)
                st.stop()

    need_models = not (MODELS_DIR / VECTORIZER_FILE).exists()
    if need_models:
        with st.spinner("First run: training models (this can take a minute or two)..."):
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "train.py")],
                capture_output=True, text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                st.error("Model training failed. Check the app logs.")
                print(result.stderr)
                st.stop()




RESULT_KEY = "classification_result"
BATCH_KEY = "batch_result"

FIGURE_LABELS = {
    "class_balance": "Class balance",
    "subject_by_class": "Subjects",
    "length_distribution": "Article length",
    "articles_over_time": "Timeline",
    "style_features": "Writing style",
    "top_terms": "Top terms",
    "leakage_audit": "Leakage audit",
    "roc_pr_curves": "ROC / PR",
    "calibration_curves": "Calibration",
    "confusion_matrices": "Confusion matrices",
    "threshold_sweep": "Threshold sweep",
    "learning_curve": "Learning curve",
    "cross_dataset": "Cross-dataset",
}
EDA_FIGURES = ["leakage_audit", "class_balance", "subject_by_class",
               "length_distribution", "articles_over_time", "style_features",
               "top_terms", "data_quality"]
EVAL_FIGURES = ["roc_pr_curves", "calibration_curves", "confusion_matrices",
                "threshold_sweep", "learning_curve", "cross_dataset"]

# The five independent lines of evidence, each produced by its own module.
# label -> (figure stems, report file, one-line claim)
EVIDENCE = {
    "Significance": (
        ["significance"], "significance_report.md",
        "Is the best model actually better than grepping for one word?"),
    "Feature ablation": (
        ["feature_ablation", "tuning_results"], "tuning_report.md",
        "How many features does the task really need?"),
    "Temporal split": (
        ["temporal_validation"], "temporal_report.md",
        "What happens when the test articles are published after the training ones?"),
    "Representations": (
        ["alt_models"], "alt_models_report.md",
        "Do models that cannot see the fingerprint still work?"),
    "Error taxonomy": (
        ["error_taxonomy"], "error_taxonomy.md",
        "Which mistakes are the dataset's fault rather than the model's?"),
}


# --- cached loaders --------------------------------------------------------

@st.cache_resource
def load_vectorizer():
    return joblib.load(MODELS_DIR / VECTORIZER_FILE)


@st.cache_resource
def load_model(key: str):
    return joblib.load(model_path(key))


@st.cache_data
def load_metrics():
    path = MODELS_DIR / METRICS_JSON
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def load_run_meta():
    path = MODELS_DIR / RUN_META
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def load_eda_table(name: str):
    path = REPORTS_DIR / "eda_tables" / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_data
def load_report_csv(name: str):
    """A table written by one of the analysis modules, or None if not run yet."""
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_data_quality():
    path = REPORTS_DIR / "data_quality.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_report(name: str):
    path = REPORTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def available_models() -> list[str]:
    return [k for k in MODEL_KEYS if model_path(k).exists()]


def figure_gallery(stems: list[str], key: str) -> bool:
    """
    Show one figure at a time behind a pill selector.

    Stacking a dozen full-width PNGs turns these tabs into a scrolling wall;
    one at a time keeps each readable and the page short.
    """
    available = [s for s in stems if (FIGURES_DIR / f"{s}.png").exists()]
    if not available:
        return False

    labels = [FIGURE_LABELS.get(s, s.replace("_", " ")) for s in available]
    choice = st.pills("Figure", labels, default=labels[0], key=key,
                      label_visibility="collapsed")
    if choice:
        stem = available[labels.index(choice)]
        st.image(str(FIGURES_DIR / f"{stem}.png"), width="stretch")
    return True


# --- classify --------------------------------------------------------------

def classify_and_store(text: str, model_key: str, source: str, origin: str = None):
    """Run one classification, log it once, and stash it in session state."""
    vectorizer, model = load_vectorizer(), load_model(model_key)
    result = explain_prediction(text, model, vectorizer, top_n=12)

    if "error" in result:
        st.session_state[RESULT_KEY] = {"error": result["error"]}
        return

    contrib = result.get("contributions")
    top_terms = [[r.term, round(float(r.contribution), 5)] for r in contrib.itertuples()] \
        if contrib is not None and not contrib.empty else []

    row_id = log_prediction(text, result, model_key, source=source,
                            origin=origin, top_terms=top_terms)

    st.session_state[RESULT_KEY] = {
        "result": result, "text": text, "model_key": model_key,
        "row_id": row_id, "feedback": None,
    }
    st.session_state["last_prediction_id"] = row_id


def render_result():
    """Render whatever is in session state. Safe to call on every rerun."""
    state = st.session_state.get(RESULT_KEY)
    if not state:
        return
    if "error" in state:
        st.warning(f"{state['error']} Try a longer or different input.",
                   icon=":material/text_fields:")
        return

    result, text, row_id = state["result"], state["text"], state["row_id"]
    label, confidence = result["label"], result["confidence"]
    is_real = label == "REAL"

    # --- verdict ---
    with st.container(border=True):
        verdict, conf_col = st.columns([3, 2], vertical_alignment="center")
        with verdict:
            if is_real:
                st.success("Reads as **REAL**", icon=":material/check_circle:")
            else:
                st.error("Reads as **FAKE**", icon=":material/gpp_maybe:")
            if confidence:
                st.progress(min(float(confidence), 1.0))
        with conf_col:
            st.metric("Confidence", f"{confidence * 100:.1f}%" if confidence else "n/a")

        if confidence and confidence < 0.70:
            st.warning("Close to a coin flip - read this as 'unclear', not a verdict.",
                       icon=":material/help:")
        if result["n_terms_matched"] < 5:
            st.warning(
                f"Only {result['n_terms_matched']} features matched the model's "
                "vocabulary. This input is too short or too far from the training "
                "data for a meaningful call.",
                icon=":material/warning:",
            )

    # --- evidence ---
    contrib = result.get("contributions")
    if contrib is not None and not contrib.empty:
        with st.container(border=True):
            st.subheader("Why", anchor=False)
            caption = ("Each term's effect = its TF-IDF value x the model's learned "
                       "weight. These sum to the decision score, so this is the "
                       "actual arithmetic, not an approximation.")
            if state["model_key"] == "svm":
                caption += (" For the calibrated SVM the weights are averaged over "
                            "3 calibration folds - the ranking is exact, magnitudes "
                            "are very close.")
            st.caption(caption)

            st.bar_chart(contrib.set_index("term")["contribution"],
                         color="#3d5a80", horizontal=True, height=280)

            push = st.columns(2)
            push[0].metric("Evidence for REAL", f"{result['total_real_push']:.3f}")
            push[1].metric("Evidence for FAKE", f"{result['total_fake_push']:.3f}")

            with st.expander("Contribution table", icon=":material/table_chart:"):
                st.dataframe(
                    contrib.style.format({"tfidf": "{:.3f}", "weight": "{:.3f}",
                                          "contribution": "{:.4f}"}),
                    width="stretch", hide_index=True,
                )

    # --- secondary detail ---
    detail_l, detail_r = st.columns(2)
    with detail_l, st.expander("Writing-style signals", icon=":material/edit_note:"):
        stats = text_stats(text)
        with st.container(horizontal=True):
            st.metric("Words", f"{stats['n_words']:,}", border=True)
            st.metric("Exclamations", stats["exclamation_count"], border=True)
        with st.container(horizontal=True):
            st.metric("ALL-CAPS words", stats["allcaps_words"], border=True)
            st.metric("Avg word length", f"{stats['avg_word_len']:.1f}", border=True)
        st.caption("Not used by the classifier - shown because heavy exclamation "
                   "and ALL-CAPS use are classic sensationalism markers.")

    with detail_r, st.expander("Input and probabilities", icon=":material/article:"):
        if result["proba"]:
            st.write({"fake": f"{result['proba'][0] * 100:.2f}%",
                      "real": f"{result['proba'][1] * 100:.2f}%"})
        st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

    # --- feedback ---
    # Rendered outside every `if st.button(...)` block, so the rerun triggered
    # by the vote still draws this and the handler below actually runs.
    with st.container(border=True):
        if state["feedback"]:
            st.success(f"Recorded as **{state['feedback']}**. Exportable as "
                       "training data from the History tab.",
                       icon=":material/task_alt:")
        else:
            st.caption(f"Prediction #{row_id} logged. Was it right?")
            vote = st.feedback("thumbs", key=f"fb_{row_id}")
            if vote is not None:
                record_feedback(row_id, correct=bool(vote))
                state["feedback"] = "correct" if vote else "incorrect"
                st.rerun()


def tab_classify(model_key: str):
    source = st.segmented_control(
        "Input source", ["Text", "URL", "Screenshot"], default="Text",
        key="input_source", label_visibility="collapsed",
    ) or "Text"

    with st.container(border=True):
        if source == "Text":
            text = st.text_area("Paste a news headline or article", height=200,
                                key="text_input",
                                placeholder="Paste the article text here...")
            if st.button("Classify", type="primary", icon=":material/search:"):
                if not text.strip():
                    st.warning("Paste some text first.", icon=":material/edit:")
                else:
                    classify_and_store(text, model_key, source="text")

        elif source == "URL":
            if EXTRACT_ERROR:
                st.warning(f"URL input unavailable: {EXTRACT_ERROR}",
                           icon=":material/link_off:")
                st.code("pip install -r requirements.txt", language="bash")
            url = st.text_input("Article URL", placeholder="https://example.com/story",
                                disabled=bool(EXTRACT_ERROR))
            if st.button("Fetch and classify", type="primary",
                         icon=":material/download:", disabled=bool(EXTRACT_ERROR)):
                if not url.strip():
                    st.warning("Enter a URL first.", icon=":material/link:")
                else:
                    with st.spinner("Fetching article..."):
                        try:
                            extracted = extract_text_from_url(url)
                        except Exception as exc:
                            st.error(f"Couldn't fetch or parse that URL: {exc}",
                                     icon=":material/error:")
                            extracted = ""
                    if extracted:
                        if len(extracted) < 100:
                            st.warning(
                                "Very little text extracted - this site probably "
                                "blocks scraping or renders with JavaScript. Paste "
                                "the text manually instead.",
                                icon=":material/warning:",
                            )
                        classify_and_store(extracted, model_key, source="url", origin=url)

        else:
            st.caption("Needs the Tesseract OCR engine installed on this machine.")
            if EXTRACT_ERROR:
                st.warning(f"Screenshot input unavailable: {EXTRACT_ERROR}",
                           icon=":material/image_not_supported:")
                st.code("pip install -r requirements.txt", language="bash")
            uploaded = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"],
                                        disabled=bool(EXTRACT_ERROR))
            if uploaded and st.button("Extract and classify", type="primary",
                                      icon=":material/document_scanner:"):
                with st.spinner("Running OCR..."):
                    try:
                        extracted = extract_text_from_image(uploaded)
                    except Exception as exc:
                        st.error(f"OCR failed: {exc}", icon=":material/error:")
                        extracted = ""
                if extracted:
                    if len(extracted) < 30:
                        st.warning("Very little text extracted - try a clearer, "
                                   "higher-resolution screenshot.",
                                   icon=":material/warning:")
                    classify_and_store(extracted, model_key, source="image",
                                       origin=uploaded.name)

    render_result()


# --- batch -----------------------------------------------------------------

def tab_batch(model_key: str):
    st.caption("Upload a CSV with a text column. Every row goes through the same "
               "pipeline as a single prediction.")

    uploaded = st.file_uploader("CSV file", type=["csv"], key="batch_csv")
    if not uploaded:
        st.info("A `text` or `content` column is detected automatically.",
                icon=":material/upload_file:")
        return

    df = pd.read_csv(uploaded)

    with st.container(border=True):
        st.caption(f"{len(df):,} rows, {len(df.columns)} columns")
        str_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        default = next((c for c in ("content", "text", "article", "body")
                        if c in df.columns), str_cols[0])
        picker = st.columns([2, 2, 1], vertical_alignment="bottom")
        text_col = picker[0].selectbox("Text column", str_cols,
                                       index=str_cols.index(default))
        label_col = picker[1].selectbox("True-label column (optional, 0=fake 1=real)",
                                        ["(none)"] + list(df.columns))
        score = picker[2].button("Score rows", type="primary",
                                 icon=":material/play_arrow:")
        st.dataframe(df.head(3), width="stretch")

    if score:
        with st.spinner(f"Scoring {len(df):,} rows..."):
            vectorizer, model = load_vectorizer(), load_model(model_key)
            texts = df[text_col].fillna("").astype(str)
            if text_col != "title" and "title" in df.columns:
                texts = (df["title"].fillna("").astype(str) + " " + texts).str.strip()

            cleaned = texts.map(clean_text)
            usable = cleaned.str.len() > 0
            X = vectorizer.transform(cleaned)
            proba_real = model.predict_proba(X)[:, list(model.classes_).index(1)]

            out = df.copy()
            out["prediction"] = [LABEL_NAMES[int(p >= 0.5)] if u else "UNUSABLE"
                                 for p, u in zip(proba_real, usable)]
            out["proba_real"] = proba_real.round(4)
            out["confidence"] = [max(p, 1 - p).round(4) for p in proba_real]
            out.loc[~usable, ["proba_real", "confidence"]] = None

        # Stash it: scoring a large CSV is expensive, and clicking download
        # triggers a rerun that would otherwise discard everything.
        st.session_state[BATCH_KEY] = {"out": out, "usable": usable,
                                       "label_col": label_col}

    state = st.session_state.get(BATCH_KEY)
    if not state:
        return
    out, usable, label_col = state["out"], state["usable"], state["label_col"]

    if len(out) != len(df):
        st.info("Showing results from the previously scored file. Click "
                "**Score rows** to score the file now uploaded.",
                icon=":material/history:")

    flagged = int((out["prediction"] == "FAKE").sum())
    mean_conf = out["confidence"].mean()
    with st.container(horizontal=True):
        st.metric("Rows scored", f"{len(out):,}", border=True)
        st.metric("Flagged FAKE", f"{flagged:,}",
                  f"{flagged / max(len(out), 1):.1%} of rows", border=True)
        st.metric("Mean confidence",
                  f"{mean_conf:.1%}" if pd.notna(mean_conf) else "n/a", border=True)
        st.metric("Low confidence", int((out["confidence"] < 0.7).sum()),
                  help="Predictions below 70% confidence", border=True)

        if label_col != "(none)" and label_col in out.columns:
            y = pd.to_numeric(out[label_col], errors="coerce")
            mask = y.notna() & usable
            if mask.any():
                pred_int = (out.loc[mask, "prediction"] == "REAL").astype(int)
                acc = (pred_int.to_numpy() == y[mask].astype(int).to_numpy()).mean()
                st.metric("Accuracy vs labels", f"{acc:.2%}", border=True)

    st.dataframe(out.head(50), width="stretch", hide_index=True)
    st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name="scored.csv", mime="text/csv",
                       icon=":material/download:")


# --- data ------------------------------------------------------------------

def tab_data():
    if not (REPORTS_DIR / "eda_report.md").exists():
        st.info("No EDA report yet. Run `python src/eda.py`, then reload.",
                icon=":material/analytics:")
        return

    leak = load_eda_table("leakage_audit.csv")
    overview = load_eda_table("overview.csv")
    subjects = load_eda_table("subject_by_class.csv")

    if leak is not None and not leak.empty:
        worst = leak.iloc[0]
        st.error(
            f"**Label leakage.** The marker `{worst.name}` appears in "
            f"{worst['in_real']:.1%} of real articles and {worst['in_fake']:.1%} of "
            f"fake ones. A single-rule classifier using it alone scores "
            f"**{worst['one_rule_accuracy']:.1%}** - matching the trained models. "
            "Most of the headline accuracy is publisher fingerprinting, not "
            "fake-vs-real signal.",
            icon=":material/report:",
        )

    with st.container(horizontal=True):
        if overview is not None:
            st.metric("Articles analysed", f"{int(overview['articles'].sum()):,}",
                      border=True)
            st.metric("Fake / real split",
                      f"{overview.loc['fake', 'share']:.0%} / "
                      f"{overview.loc['real', 'share']:.0%}", border=True)
        if subjects is not None:
            exclusive = int((subjects["exclusive_to"] != "mixed").sum())
            st.metric("Single-class subjects", f"{exclusive} of {len(subjects)}",
                      help="Subject categories that appear in only one class - "
                           "where true, `subject` alone predicts the label",
                      border=True)
        if leak is not None and not leak.empty:
            st.metric("Best one-rule accuracy",
                      f"{leak['one_rule_accuracy'].max():.1%}", border=True)
        dq = load_data_quality()
        if dq:
            counts = dq.get("status_counts", {})
            st.metric("Data quality score", f"{dq['score']:.0f}/100",
                      f"{counts.get('FAIL', 0)} fail / {counts.get('WARN', 0)} warn",
                      delta_color="off", border=True,
                      help="From src/data_quality.py. Two failures are permanent "
                           "properties of this corpus, not bugs to fix: disjoint "
                           "subjects, and the two source files not being samples "
                           "of one population.")

    with st.container(border=True):
        figure_gallery(EDA_FIGURES, key="eda_figs")

    dq = load_data_quality()
    if dq:
        with st.expander(f"Data quality checks ({len(dq['checks'])})",
                         icon=":material/checklist:"):
            checks = pd.DataFrame(dq["checks"])[
                ["status", "category", "name", "summary"]]
            st.dataframe(checks, width="stretch", hide_index=True)
            st.caption("Run `python src/data_quality.py --strict` to exit "
                       "non-zero on any failure - use it as a gate before training.")

    if leak is not None:
        with st.expander("Leakage audit table", icon=":material/rule:"):
            st.dataframe(
                leak.style.format({"in_fake": "{:.2%}", "in_real": "{:.2%}",
                                   "gap": "{:.2%}", "one_rule_accuracy": "{:.2%}"}),
                width="stretch",
            )

    with st.expander("Full EDA report", icon=":material/description:"):
        st.markdown((REPORTS_DIR / "eda_report.md").read_text(encoding="utf-8"))


# --- evidence --------------------------------------------------------------

def evidence_headline():
    """
    The four numbers that make the project's case, pulled from the modules'
    own output rather than hardcoded, so they cannot drift from the reports.
    """
    pair = load_report_csv("significance_pairwise.csv")
    ablation = load_report_csv("feature_ablation.csv")
    alt = load_report_csv("alt_models_table.csv")

    shown = False
    with st.container(horizontal=True):
        if pair is not None and not pair.empty:
            # the comparison between the keyword rule and the best trained model
            row = pair[pair["comparison"].str.contains("Reuters", na=False)
                       & pair["comparison"].str.contains("SVM", na=False)]
            if not row.empty:
                r = row.iloc[0]
                st.metric("Best model vs keyword rule", f"p = {r['p_holm']:.2f}",
                          str(r["verdict"]), delta_color="off", border=True,
                          help="McNemar, Holm-adjusted. A large p means the "
                               "difference between the trained model and a "
                               "one-word grep is indistinguishable from chance.")
                shown = True

        if ablation is not None and not ablation.empty:
            raw = ablation[(ablation["variant"] == "raw")
                           & ablation["model"].eq("svm")]
            if not raw.empty:
                small = raw.loc[raw["max_features"].idxmin()]
                big = raw.loc[raw["max_features"].idxmax()]
                st.metric(f"SVM at {int(small['max_features'])} features",
                          f"{small['accuracy']:.2%}",
                          f"{small['accuracy'] - big['accuracy']:+.2%} "
                          f"vs {int(big['max_features']):,} features",
                          delta_color="off", border=True,
                          help="A flat ablation curve means there is nothing to "
                               "learn - the shortcut is available immediately.")
                shown = True

        if alt is not None and not alt.empty:
            style = alt[alt["model"].str.contains("Stylometry", na=False)]
            leak = alt[alt["model"].str.contains("stump", na=False)]
            if not style.empty:
                raw_acc = style[style["condition"] == "raw"]["accuracy"]
                strip = style[style["condition"] == "stripped"]["accuracy"]
                delta = (f"{strip.iloc[0] - raw_acc.iloc[0]:+.2%} when stripped"
                         if not strip.empty else None)
                st.metric("Stylometry only (blind to the tag)",
                          f"{raw_acc.iloc[0]:.2%}", delta, delta_color="off",
                          border=True,
                          help="This model reads only punctuation and casing "
                               "counts, so it cannot see '(Reuters)' at all. "
                               "Closest thing here to the real task difficulty.")
                shown = True
            if not leak.empty:
                raw_acc = leak[leak["condition"] == "raw"]["accuracy"]
                strip = leak[leak["condition"] == "stripped"]["accuracy"]
                if not raw_acc.empty and not strip.empty:
                    st.metric("Keyword stump", f"{raw_acc.iloc[0]:.2%}",
                              f"{strip.iloc[0] - raw_acc.iloc[0]:+.2%} when stripped",
                              delta_color="off", border=True,
                              help="One binary feature: does the text contain "
                                   "'reuters'. Collapses to the majority class "
                                   "once the fingerprint is removed.")
                    shown = True
    return shown


def tab_evidence():
    any_report = any((REPORTS_DIR / rpt).exists()
                     for _, rpt, _ in EVIDENCE.values())
    if not any_report:
        st.info("No analysis reports yet. Run the deeper analyses first:",
                icon=":material/science:")
        st.code("python src/significance.py\npython src/temporal_eval.py\n"
                "python src/tune.py\npython src/alt_models.py\n"
                "python src/error_taxonomy.py", language="bash")
        return

    st.error(
        "**The headline accuracy does not measure fake-news detection.** Five "
        "independent analyses agree: a one-word keyword rule matches the best "
        "model, 50 features score as well as 5,000, and the only model that "
        "cannot see the publisher fingerprint scores far lower. The label leaks "
        "through three separate columns - text, subject and date.",
        icon=":material/priority_high:",
    )

    evidence_headline()

    labels = list(EVIDENCE)
    choice = st.segmented_control("Line of evidence", labels, default=labels[0],
                                  key="evidence_pick",
                                  label_visibility="collapsed") or labels[0]
    stems, report_name, claim = EVIDENCE[choice]

    with st.container(border=True):
        st.subheader(choice, anchor=False)
        st.caption(claim)

        if choice == "Significance":
            scores = load_report_csv("significance_scores.csv")
            if scores is not None and not scores.empty:
                view = scores[["system", "accuracy", "acc_ci_low",
                               "acc_ci_high", "errors"]].copy()
                st.dataframe(
                    view.style.format({"accuracy": "{:.4f}", "acc_ci_low": "{:.4f}",
                                       "acc_ci_high": "{:.4f}"}),
                    width="stretch", hide_index=True)
                st.caption("95% percentile intervals from a paired bootstrap - "
                           "every system scored on the same resampled articles.")

        if not figure_gallery(stems, key=f"ev_{choice}"):
            st.caption("Figure not generated yet - run the matching module.")

    body = read_report(report_name)
    if body:
        with st.expander(f"Full report - {report_name}",
                         icon=":material/description:"):
            st.markdown(body)


# --- model -----------------------------------------------------------------

def tab_model(model_key: str):
    metrics = load_metrics()
    if not metrics:
        st.info("No metrics.json yet. Run `python src/train.py`.",
                icon=":material/model_training:")
        return

    run, models, baselines = metrics["run"], metrics["models"], metrics.get("baselines", {})

    st.caption(f"Trained {run['trained_at']} - {run['n_train']:,} train / "
               f"{run['n_test']:,} test articles, {run['n_features']:,} features. "
               f"Boilerplate stripped: **{run['strip_boilerplate']}**")

    best_acc = max(v["accuracy"] for v in models.values()) if models else 0
    with st.container(horizontal=True):
        for key, v in models.items():
            st.metric(model_name(key), f"{v['accuracy']:.2%}",
                      f"F1 {v['f1']:.4f}", delta_color="off", border=True)

    if baselines:
        with st.container(border=True):
            st.subheader("Compared against doing almost nothing", anchor=False)
            one_rule = baselines.get("one_rule_mentions_reuters")
            with st.container(horizontal=True):
                st.metric("Majority class", f"{baselines['majority_class']:.2%}",
                          border=True)
                if one_rule is not None:
                    st.metric('One rule: "mentions Reuters"', f"{one_rule:.2%}",
                              f"{one_rule - best_acc:+.2%} vs best model", border=True)
            if one_rule is not None and one_rule >= best_acc - 0.005:
                st.warning(
                    "A one-line keyword rule matches the trained models. That is the "
                    "clearest possible sign the dataset, not the model, is doing the work.",
                    icon=":material/priority_high:",
                )

    with st.container(border=True):
        st.subheader("What the model leans on overall", anchor=False)
        try:
            top = global_top_features(load_model(model_key), load_vectorizer(), 15)
            cols = st.columns(2)
            cols[0].caption("Strongest REAL indicators")
            cols[0].dataframe(
                top[["real_term", "real_weight"]].style.format({"real_weight": "{:.3f}"}),
                hide_index=True, width="stretch")
            cols[1].caption("Strongest FAKE indicators")
            cols[1].dataframe(
                top[["fake_term", "fake_weight"]].style.format({"fake_weight": "{:.3f}"}),
                hide_index=True, width="stretch")
        except Exception as exc:
            st.info(f"Feature weights unavailable for this model: {exc}",
                    icon=":material/info:")

    with st.container(border=True):
        if not figure_gallery(EVAL_FIGURES, key="eval_figs"):
            st.caption("Run `python src/evaluate.py` for ROC, calibration and "
                       "threshold plots.")

    card = MODELS_DIR / MODEL_CARD
    if card.exists():
        with st.expander("Model card", icon=":material/badge:"):
            st.markdown(card.read_text(encoding="utf-8"))


# --- history ---------------------------------------------------------------

def tab_history():
    stats = summary_stats()
    if not stats["total"]:
        st.info("Nothing classified yet. Predictions made in this app are logged here.",
                icon=":material/inbox:")
        return

    with st.container(horizontal=True):
        st.metric("Predictions", f"{stats['total']:,}", border=True)
        st.metric("Mean confidence",
                  f"{stats['avg_confidence']:.1%}" if stats["avg_confidence"] else "n/a",
                  border=True)
        st.metric("Low confidence", stats["low_confidence_count"],
                  f"{stats['low_confidence_share']:.0%} of all",
                  delta_color="off", border=True)
        st.metric("Human-reviewed", stats["reviewed"], border=True)
        if stats["observed_accuracy"] is not None:
            st.metric("Observed accuracy", f"{stats['observed_accuracy']:.1%}",
                      help="Measured on real inputs users submitted and reviewed, "
                           "not on the held-out test split. A gap below the reported "
                           "test accuracy is distribution shift.",
                      border=True)

    charts = st.columns(2)
    with charts[0], st.container(border=True):
        st.caption("By predicted label")
        st.bar_chart(pd.Series(stats["by_label"]), height=200)
    with charts[1], st.container(border=True):
        st.caption("By input source")
        st.bar_chart(pd.Series(stats["by_source"]), height=200)

    with st.container(border=True):
        st.subheader("Recent predictions", anchor=False)
        limit = st.slider("How many to show", 10, 500, 50, step=10,
                          label_visibility="collapsed")
        df = fetch_predictions(limit=limit)
        view = df[["id", "created_at", "source", "model", "label", "confidence",
                   "user_feedback", "text_preview"]].copy()
        view["confidence"] = view["confidence"].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else "")
        st.dataframe(view, width="stretch", hide_index=True)

    # Both exports query the whole store, not the slider-limited view above -
    # otherwise the corrections count silently under-reports.
    full_log = fetch_predictions(limit=0)
    corrections = full_log[full_log["true_label"].notna()]
    with st.container(horizontal=True):
        st.download_button(f"Full log ({len(full_log)} rows)",
                           full_log.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_log.csv", mime="text/csv",
                           icon=":material/download:")
        st.download_button(f"Corrections ({len(corrections)} rows)",
                           corrections.to_csv(index=False).encode("utf-8"),
                           file_name="corrections.csv", mime="text/csv",
                           icon=":material/rate_review:", disabled=corrections.empty)
    st.caption(f"Store: `{DB_PATH}`")


# --- main ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Fake news detector",
                       page_icon=":material/fact_check:", layout="wide")
  
    _bootstrap_pipeline()
  
    st.title("Fake news detector", anchor=False)
    st.caption("TF-IDF + classical ML (Naive Bayes / SVM / Logistic Regression) "
               "trained on the ISOT dataset - with the evidence behind every call")

    if not (MODELS_DIR / VECTORIZER_FILE).exists():
        st.error("No trained models found.", icon=":material/error:")
        st.code("python src/setup_data.py\npython src/train.py", language="bash")
        st.stop()

    keys = available_models()
    if not keys:
        st.error("The vectorizer exists but no model files do. Re-run "
                 "`python src/train.py`.", icon=":material/error:")
        st.stop()

    with st.sidebar:
        st.subheader("Model", anchor=False)
        choice = st.radio("Model", [model_name(k) for k in keys],
                          label_visibility="collapsed")
        model_key = next(k for k in keys if model_name(k) == choice)

        meta = load_run_meta()
        if meta:
            st.caption(f"Trained {meta['trained_at']}  \n"
                       f"{meta['n_train']:,} train / {meta['n_test']:,} test")
            if meta.get("strip_boilerplate"):
                st.badge("Boilerplate stripped", icon=":material/check:", color="green")
            else:
                st.badge("Leakage present", icon=":material/warning:", color="red")
                st.caption("Trained **with** publisher boilerplate, so its accuracy "
                           "is inflated. See the Data tab.")

        st.caption("This model recognises writing style and vocabulary patterns. "
                   "**It does not fact-check.** A well-written lie reads REAL to it; "
                   "a badly-written truth reads FAKE.")

    tabs = st.tabs([":material/search: Classify", ":material/table_rows: Batch",
                    ":material/analytics: Data", ":material/science: Evidence",
                    ":material/model_training: Model",
                    ":material/history: History"])
    with tabs[0]:
        tab_classify(model_key)
    with tabs[1]:
        tab_batch(model_key)
    with tabs[2]:
        tab_data()
    with tabs[3]:
        tab_evidence()
    with tabs[4]:
        tab_model(model_key)
    with tabs[5]:
        tab_history()


if __name__ == "__main__":
    main()
