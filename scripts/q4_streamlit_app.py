"""Question 4: Streamlit Web Application — Integrated Background Editor.

This entry point provides a rich, interactive web UI hosting Question 4's background
editor. It supports both Simulated Fast-Typing (with animated streaming and live alert feed)
and Genuine Incremental Live Typing (with st.session_state tracking).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.q1_word_segmentation import build_system
from src.q3_spelling_corrector import build_corrector_from_brown
from src.q4_live_editor import EditorState, IntegratedEditor
from src.q4_passage_analysis import PassageAnalyzer
from src.q4_pcfg_parser import PCFGParser
from src.q4_shared_lm import SharedLanguageModel


@st.cache_resource(show_spinner="Loading Question 1, 3, and 4 models (cached for fast start)...")
def get_integrated_editor(p_merge: float = 0.08, trigger_interval: int = 5, data_dir: str = "data") -> tuple[IntegratedEditor, PassageAnalyzer]:
    data_path = Path(data_dir)
    cache_dir = data_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "q4_editor_models.pkl"

    if cache_file.exists():
        try:
            with cache_file.open("rb") as f:
                q1_sys, q3_corr, pcfg_pars, shared_lm = pickle.load(f)
        except Exception:
            q1_sys = build_system("english", data_dir=data_path)
            q3_corr = build_corrector_from_brown(seed=7, data_dir=data_dir)[0]
            pcfg_pars = PCFGParser.train_from_treebank(data_dir=data_path)
            shared_lm = SharedLanguageModel.train_and_tune(data_dir=data_path)
            with cache_file.open("wb") as f:
                pickle.dump((q1_sys, q3_corr, pcfg_pars, shared_lm), f)
    else:
        q1_sys = build_system("english", data_dir=data_path)
        q3_corr = build_corrector_from_brown(seed=7, data_dir=data_dir)[0]
        pcfg_pars = PCFGParser.train_from_treebank(data_dir=data_path)
        shared_lm = SharedLanguageModel.train_and_tune(data_dir=data_path)
        try:
            with cache_file.open("wb") as f:
                pickle.dump((q1_sys, q3_corr, pcfg_pars, shared_lm), f)
        except Exception:
            pass

    editor = IntegratedEditor(
        p_merge=p_merge,
        trigger_interval=trigger_interval,
        data_dir=data_path,
        q1_system=q1_sys,
        q3_corrector=q3_corr,
        shared_lm=shared_lm,
    )
    analyzer = PassageAnalyzer(pcfg_parser=pcfg_pars, shared_lm=shared_lm)
    return editor, analyzer


def render_q4_app(standalone: bool = True) -> None:
    if standalone:
        st.set_page_config(page_title="Integrated Background Editor — Q4", page_icon="✍️", layout="wide")

    st.title("✍️ Question 4: Integrated Background Editor")
    st.caption("Live Word Segmentation, Spelling Correction, & Constituency-Based Grammar Checking")

    # Sidebar Controls
    st.sidebar.header("⚙️ Editor Settings")
    mode = st.sidebar.radio("Execution Mode", ["Simulated Fast-Typing Demo", "Genuine Live User Typing"])
    p_merge = st.sidebar.slider("Space Omission Probability (p)", min_value=0.0, max_value=0.20, value=0.08, step=0.01)
    trigger_interval = st.sidebar.slider("Grammar Trigger Interval (N words)", min_value=2, max_value=10, value=5, step=1)
    data_dir = st.sidebar.text_input("Data Directory", value="data")

    try:
        editor, analyzer = get_integrated_editor(p_merge=p_merge, trigger_interval=trigger_interval, data_dir=data_dir)
        editor.p_merge = p_merge
        editor.trigger_interval = trigger_interval
    except Exception as exc:
        st.error("Failed to load models. Ensure corpus setup has been completed.")
        st.exception(exc)
        return

    if mode == "Simulated Fast-Typing Demo":
        st.subheader("🎬 Simulated Typing Passage")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            corpus_choice = st.selectbox("Sample Passage Source Corpus", ["gutenberg", "brown", "reuters"])
        with col_c2:
            sleep_delay = st.slider("Typing Delay (seconds per word)", min_value=0.0, max_value=0.20, value=0.04, step=0.01)

        if st.button("🚀 Start Fast-Typing Simulation", type="primary"):
            passage_words = editor.sample_passage(corpus_name=corpus_choice, seed=None)
            st.info(f"Sampled {len(passage_words)} words from {corpus_choice} corpus.")

            text_container = st.empty()
            metrics_container = st.empty()

            tab_table, tab_alerts = st.tabs(["📋 Part 4: Final Passage Structural Analysis", "🔔 Live Alert Feed"])

            with tab_alerts:
                alert_feed = st.container()

            final_state = None
            for state, alert in editor.simulate_typing_stream(passage_words, sleep_delay_sec=sleep_delay):
                final_state = state

                # Update live text display
                text_container.markdown(f"**Live Token Stream:**\n> {' '.join(state.tokens)}")

                # Update live metrics
                with metrics_container.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Words Processed", len(state.tokens))
                    col2.metric("Seg Merges Resolved", state.segment_merges_resolved)
                    col3.metric("Spelling Corrections", state.spelling_corrections_applied)
                    col4.metric("Per-Token Latency", f"{state.avg_per_token_latency_ms:.2f} ms")
                    col5.metric("Per-Trigger Latency", f"{state.avg_per_trigger_latency_ms:.2f} ms")

                # Emit live alert card if triggered
                if alert:
                    with alert_feed:
                        if alert.alert_type == "SEGMENT-ALERT":
                            st.warning(f"🧩 **[SEGMENT-ALERT]** Token `{alert.original}` split into `{alert.replacement}` (POS: `{alert.pos_tags}`)")
                        elif alert.alert_type == "SPELL-ALERT":
                            st.error(f"✏️ **[SPELL-ALERT]** Non-word `{alert.original}` corrected to `{alert.replacement}`")
                        elif alert.alert_type == "GRAMMAR-ALERT":
                            st.info(f"📊 **[GRAMMAR-ALERT]** {alert.details}")

            # Final Passage Structural Analysis Table in Tab 1
            if final_state:
                with tab_table:
                    report = analyzer.analyze_passage(
                        tokens=final_state.tokens,
                        pos_tags=final_state.pos_tags,
                        total_seg_merges=final_state.segment_merges_resolved,
                        total_spell_corrections=final_state.spelling_corrections_applied,
                    )
                    df = pd.DataFrame(report.to_dict_list())
                    st.dataframe(df, use_container_width=True)

    else:
        st.subheader("⌨️ Genuine Incremental Live Typing")
        st.write(
            "Type below. Alerts update automatically ~300ms after you stop typing — "
            "no need to press Ctrl+Enter or click away."
        )

        # --- Bug 1 fix -------------------------------------------------------
        # Once a widget is created with a given `key`, Streamlit (and, per its
        # own docstring, the st_keyup component too) treats st.session_state
        # as the source of truth for that key on every later rerun and will
        # NOT re-adopt a new `value=` — for native widgets, re-assigning
        # st.session_state[key] in a callback does work; but st_keyup's
        # docstring is explicit that assigning to st.session_state[key]
        # does *not* push a new value into its input, and that the only
        # supported way to reset it is to give it a *different* key. So
        # instead of touching the text value directly, the preset dropdown's
        # on_change bumps a version counter that is folded into the text
        # widget's key — a version bump gives the widget a brand-new key,
        # which is treated as a first render and *does* honor `value=`.
        # This approach resets correctly for both st_keyup and the
        # st.text_area fallback below.
        PRESET_KEY = "q4_live_typing_preset"
        VERSION_KEY = "q4_live_typing_text_version"

        PRESETS = {
            "-- Type Custom Text --": "",
            "Thequick brown fox jumps over the lazy dog.": "Thequick brown fox jumps over the lazy dog.",
            "This is a test sentnce.": "This is a test sentnce.",
            "I would like to sea the world.": "I would like to sea the world.",
        }

        def _bump_text_version() -> None:
            st.session_state[VERSION_KEY] = st.session_state.get(VERSION_KEY, 0) + 1

        if PRESET_KEY not in st.session_state:
            st.session_state[PRESET_KEY] = "-- Type Custom Text --"
        if VERSION_KEY not in st.session_state:
            st.session_state[VERSION_KEY] = 0

        st.selectbox(
            "Or choose a sample preset:",
            list(PRESETS.keys()),
            key=PRESET_KEY,
            on_change=_bump_text_version,
        )

        preset_value = PRESETS[st.session_state[PRESET_KEY]]
        text_key = f"q4_live_typing_text_{st.session_state[VERSION_KEY]}"

        # --- Bugs 2 & 3 fix ----------------------------------------------------
        # Plain st.text_area only reruns the script on blur / Ctrl+Enter, so
        # nothing fires per keystroke. streamlit-keyup's st_keyup reruns on
        # every keystroke with a debounce, giving near-real-time alerts.
        # NOTE: st_keyup is single-line, so this mode is a single running
        # sentence rather than a multi-line passage — an accepted trade-off
        # documented in the report. `value=preset_value` is only actually
        # used on the first render of a given `text_key` (see Bug 1 fix
        # above), which is exactly what we want: it seeds new text when the
        # key changes, and is otherwise ignored in favor of whatever the user
        # has typed under st.session_state[text_key].
        try:
            from st_keyup import st_keyup

            user_input = st_keyup(
                "Enter / Edit Text",
                value=preset_value,
                key=text_key,
                debounce=300,
            )
        except ImportError:
            st.warning(
                "`streamlit-keyup` is not installed (see requirements.txt) — "
                "falling back to st.text_area, which only updates on "
                "Ctrl+Enter / losing focus. Run `pip install streamlit-keyup` "
                "for real-time, per-keystroke alerts."
            )
            user_input = st.text_area(
                "Enter / Edit Text", value=preset_value, height=120, key=text_key
            )

        # st_keyup can return None before the component has mounted.
        if user_input is None:
            user_input = st.session_state.get(text_key, "") or ""

        # --- Bug 3 (continued) --------------------------------------------
        # `editor.process_full_text` always reprocesses the *entire* current
        # widget value from scratch (there is no persistent "last processed
        # index" carried across reruns in IntegratedEditor), so it is
        # inherently safe against the text shrinking or being reset by the
        # preset switch above — there's no stale partial-diff state to get
        # out of sync. We still guard against an empty string explicitly so a
        # reset to "-- Type Custom Text --" clears the results below instead
        # of erroring on an empty token stream.
        if user_input.strip():
            state = editor.process_full_text(user_input)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Words Processed", len(state.tokens))
            col2.metric("Seg Merges Resolved", state.segment_merges_resolved)
            col3.metric("Spelling Corrections", state.spelling_corrections_applied)
            col4.metric("Per-Token Latency", f"{state.avg_per_token_latency_ms:.2f} ms")
            col5.metric("Per-Trigger Latency", f"{state.avg_per_trigger_latency_ms:.2f} ms")

            st.markdown(f"**Processed Token Stream:**\n> {' '.join(state.tokens)}")

            st.subheader("🔔 Real-Time Live Alerts")
            if not state.alerts:
                st.success("No errors detected in current text.")
            else:
                for alert in state.alerts:
                    if alert.alert_type == "SEGMENT-ALERT":
                        st.warning(f"**[SEGMENT-ALERT]** Merged token `{alert.original}` split into `{alert.replacement}` (POS: `{alert.pos_tags}`)")
                    elif alert.alert_type == "SPELL-ALERT":
                        st.error(f"**[SPELL-ALERT]** Non-word `{alert.original}` corrected to `{alert.replacement}`")
                    elif alert.alert_type == "GRAMMAR-ALERT":
                        st.info(f"**[GRAMMAR-ALERT]** {alert.details}")

            if state.tokens:
                st.markdown("---")
                st.subheader(" Part 4: Final Passage Structural Analysis")
                report = analyzer.analyze_passage(
                    tokens=state.tokens,
                    pos_tags=state.pos_tags,
                    total_seg_merges=state.segment_merges_resolved,
                    total_spell_corrections=state.spelling_corrections_applied,
                )
                df = pd.DataFrame(report.to_dict_list())
                st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    render_q4_app(standalone=True)