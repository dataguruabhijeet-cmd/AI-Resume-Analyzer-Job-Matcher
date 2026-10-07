"""AI Resume Analyzer & Job Matcher - Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import database as db
from utils.job_matcher import JobMatcher, MatchResult, SemanticMatcher
from utils.job_recommender import JobRecommender, load_jobs
from utils.resume_analyzer import ResumeAnalyzer
from utils.resume_parser import ResumeParsingError, extract_resume_text
from utils.resume_scorer import job_specific_suggestions, score_resume
from utils.skill_extractor import SkillExtractor

st.set_page_config(page_title="AI Resume Analyzer & Job Matcher", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
    <style>
      .chip {display:inline-block;padding:3px 11px;margin:3px 4px 3px 0;border-radius:14px;font-size:0.85rem;font-weight:500;}
      .chip-ok  {background:#e6f4ea;color:#137333;border:1px solid #b7dfc3;}
      .chip-bad {background:#fce8e6;color:#c5221f;border:1px solid #f4b9b5;}
      .chip-neu {background:#e8f0fe;color:#1a56c4;border:1px solid #bcd0f7;}
      .block-container {padding-top: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def get_extractor() -> SkillExtractor:
    return SkillExtractor()


@st.cache_resource(show_spinner=False)
def get_semantic() -> SemanticMatcher:
    return SemanticMatcher()


@st.cache_resource(show_spinner="Loading NLP models...")
def get_analyzer() -> ResumeAnalyzer:
    return ResumeAnalyzer(get_extractor())  # uses spaCy automatically if a model is installed


@st.cache_resource(show_spinner=False)
def get_default_recommender() -> JobRecommender:
    return JobRecommender(skill_extractor=get_extractor(), matcher=JobMatcher(get_extractor(), get_semantic()))


@st.cache_data(show_spinner="Matching against all jobs...")
def run_recommendations(resume_text: str, jobs_key: str, use_semantic: bool, _recommender: JobRecommender):
    return _recommender.match_all(resume_text, use_semantic)


db.init_db()


# ----------------------------------------------------------------------------- helpers
def chips(items: list[str], kind: str = "neu") -> None:
    if not items:
        st.caption("None")
        return
    body = "".join(f'<span class="chip chip-{kind}">{html.escape(i)}</span>' for i in items)
    st.markdown(body, unsafe_allow_html=True)


def gauge(value: float, title: str, key: str) -> None:
    color = "#d93025" if value < 45 else "#f9ab00" if value < 65 else "#34a853" if value < 80 else "#188038"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%", "font": {"size": 44}},
        title={"text": title, "font": {"size": 16}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 45], "color": "#fce8e6"}, {"range": [45, 65], "color": "#fef7e0"},
                         {"range": [65, 100], "color": "#e6f4ea"}]},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)


def priority_icon(p: str) -> str:
    return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(p, "⚪")


def show_suggestions(items: list[dict]) -> None:
    if not items:
        st.success("Nothing to improve here - great job!")
    for s in items:
        st.markdown(f"{priority_icon(s['priority'])} **{s['area']}** - {s['suggestion']}")


def read_jd_file(upload) -> str | None:
    try:
        return extract_resume_text(upload, upload.name)
    except ResumeParsingError as exc:
        st.error(str(exc))
        return None


def show_match_result(res: MatchResult, key: str, job_title: str) -> None:
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        gauge(res.overall_score, f"Overall match - {res.label}", key=f"{key}_gauge")
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Skill match", f"{res.skill_score:.0f}%")
        m2.metric("Text similarity (TF-IDF)", f"{res.tfidf_score:.0f}%")
        m3.metric("Semantic match", f"{res.semantic_score:.0f}%" if res.semantic_score is not None else "off")
        st.caption("Score = 60% weighted skill coverage + 40% TF-IDF similarity "
                   "(50% / 25% / 25% when semantic matching is enabled).")
        if not res.jd_has_skills:
            st.warning("No known skills were detected in this job description, so the score relies on text similarity only.")

    a, b = st.columns(2)
    with a:
        st.markdown(f"#### ✅ Matched skills ({len(res.matched_skills)})")
        chips(res.matched_skills, "ok")
    with b:
        st.markdown(f"#### ❌ Missing skills ({len(res.missing_skills)})")
        chips(res.missing_skills, "bad")

    st.markdown("#### 📉 Skill gap analysis")
    if res.skill_gap.empty:
        st.success("You cover every skill detected in this job description.")
    else:
        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.dataframe(res.skill_gap.rename(columns={"jd_mentions": "mentions in JD"}), use_container_width=True, hide_index=True)
        with g2:
            if not res.category_coverage.empty:
                fig = px.bar(res.category_coverage, x="coverage_pct", y="category", orientation="h",
                             range_x=[0, 100], text="coverage_pct", title="Skill coverage by category (%)")
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None, xaxis_title=None)
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_cat")

    st.markdown("#### 🔑 Keyword analysis")
    if res.keywords.empty:
        st.caption("No keywords available.")
    else:
        kw = res.keywords.copy()
        kw["status"] = kw["in_resume"].map({True: "In resume", False: "Missing"})
        fig = px.bar(kw.sort_values("jd_weight"), x="jd_weight", y="keyword", orientation="h", color="status",
                     color_discrete_map={"In resume": "#34a853", "Missing": "#d93025"},
                     title="Most important job-description keywords")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None, xaxis_title="TF-IDF weight")
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_kw")

    st.markdown("#### 💡 Tailoring suggestions for this job")
    show_suggestions(job_specific_suggestions(res, job_title))


# ----------------------------------------------------------------------------- sidebar
extractor = get_extractor()
analyzer = get_analyzer()
semantic = get_semantic()

with st.sidebar:
    st.title("📄 Resume Analyzer")
    st.caption("Upload a resume, compare it with job descriptions and find your best-fit roles.")
    upload = st.file_uploader("Upload Resume", type=["pdf", "docx"], help="Supported formats: PDF, DOCX")

    st.divider()
    st.subheader("Settings")
    want_semantic = st.toggle("Semantic AI matching (advanced)", value=False,
                              help="Uses Sentence-Transformers embeddings. Requires `pip install sentence-transformers`.")
    use_semantic = False
    if want_semantic:
        with st.spinner("Loading semantic model (first run downloads ~90 MB)..."):
            ok = semantic.is_available()
        if ok:
            use_semantic = True
        else:
            st.warning("Semantic model unavailable - falling back to TF-IDF. "
                       "Install it with `pip install sentence-transformers` and make sure you are online for the first download.")
    top_n = st.slider("Jobs to recommend", 3, 15, 5)
    st.divider()
    st.caption("Built with Python · spaCy · scikit-learn · Streamlit · SQLite")

# ----------------------------------------------------------------------------- parse resume
if upload is not None:
    file_key = f"{upload.name}:{upload.size}"
    if st.session_state.get("file_key") != file_key:
        st.session_state.pop("resume", None)
        st.session_state["file_key"] = file_key
        try:
            with st.spinner("Reading and analysing your resume..."):
                text = extract_resume_text(upload, upload.name)
                analysis = analyzer.analyze(text)
                score = score_resume(analysis, text)
                rid = db.save_resume(upload.name, analysis.name, analysis.email, score["total"], analysis.skills)
            st.session_state["resume"] = {"text": text, "analysis": analysis, "score": score, "id": rid, "filename": upload.name}
        except ResumeParsingError as exc:
            st.error(f"⚠️ {exc}")
        except Exception:  # never crash the UI
            st.error("⚠️ Something went wrong while analysing this file. Please try a different file.")
else:
    st.session_state.pop("resume", None)
    st.session_state.pop("file_key", None)

R = st.session_state.get("resume")

# ----------------------------------------------------------------------------- landing
st.title("AI Resume Analyzer & Job Matcher")
if not R:
    st.info("👈 Upload your resume (PDF or DOCX) in the sidebar to get started.")
    st.markdown(
        """
        **What you will get**
        - 🧠 Automatic extraction of skills, education, experience, projects and certifications
        - 🎯 A resume ↔ job-description match score with matched / missing skills
        - 📉 Skill-gap and keyword analysis
        - 🏆 Ranked job recommendations from a built-in job catalogue
        - ✅ A resume quality score with actionable improvement tips
        """
    )
    with st.expander("Don't have a resume handy? Try the sample"):
        try:
            with open("assets/sample_resume.pdf", "rb") as f:
                st.download_button("Download sample_resume.pdf", f.read(), "sample_resume.pdf", "application/pdf")
        except FileNotFoundError:
            st.caption("Sample file not found in assets/.")
    st.divider()
    st.subheader("🕘 Previous sessions")
    hist = db.get_resume_history()
    st.dataframe(hist, use_container_width=True, hide_index=True) if not hist.empty else st.caption("No history yet.")
    st.stop()

A, text, score = R["analysis"], R["text"], R["score"]

tabs = st.tabs(["📄 Overview", "🛠 Skills", "🎯 Job Match", "🏆 Recommendations", "✅ Resume Score & Tips", "🕘 History"])

# ----------------------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader(A.name or "Candidate")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resume quality score", f"{score['total']}/100")
    c2.metric("Skills detected", len(A.skills))
    c3.metric("Experience", f"{A.experience_years:g} yrs" if A.experience_years else ("Internship" if A.has_internship else "-"))
    c4.metric("Projects / Certs", f"{len(A.projects)} / {len(A.certifications)}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Contact")
        st.write(f"📧 {A.email or '—'}")
        st.write(f"📱 {A.phone or '—'}")
        st.write(f"🔗 {A.linkedin or '—'}")
        st.write(f"💻 {A.github or '—'}")
        st.markdown("#### Education")
        if A.education:
            for e in A.education:
                st.markdown(f"- **{e['degree']}** {('· ' + e['score']) if e['score'] else ''} {('· ' + e['year']) if e['year'] else ''}  \n  <small>{html.escape(e['details'])}</small>",
                            unsafe_allow_html=True)
        else:
            st.caption("No education details detected.")
    with right:
        st.markdown("#### Experience")
        if A.experience_entries:
            for e in A.experience_entries:
                st.markdown(f"- {e}")
        else:
            st.caption("No dated experience entries detected.")
        st.markdown("#### Projects")
        for p in A.projects:
            st.markdown(f"- {p}")
        if not A.projects:
            st.caption("No projects detected.")
        st.markdown("#### Certifications")
        for c in A.certifications:
            st.markdown(f"- {c}")
        if not A.certifications:
            st.caption("No certifications detected.")
        if A.achievements:
            st.markdown("#### Achievements")
            for c in A.achievements:
                st.markdown(f"- {c}")

    with st.expander("Show extracted resume text"):
        st.text(text)

# ----------------------------------------------------------------------------- Skills
with tabs[1]:
    df_sk = extractor.extract_detailed(text)
    if df_sk.empty:
        st.warning("No known skills were detected in this resume.")
    else:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            by_cat = df_sk.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
            fig = px.pie(by_cat, names="category", values="count", hole=0.45, title="Skills by category")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, key="skills_pie")
        with c2:
            fig = px.bar(df_sk.head(15).sort_values("mentions"), x="mentions", y="skill", orientation="h",
                         color="category", title="Most mentioned skills")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, key="skills_bar")
        for cat, items in A.skills_by_category.items():
            st.markdown(f"**{cat}**")
            chips(items, "neu")

# ----------------------------------------------------------------------------- Job Match
with tabs[2]:
    st.subheader("Compare your resume with a job description")
    source = st.radio("Job description source", ["Paste text", "Upload file", "Pick from job catalogue"], horizontal=True)
    jd_text, job_title, company = "", "Custom job", ""

    if source == "Paste text":
        job_title = st.text_input("Job title (optional)", value="Custom job")
        jd_text = st.text_area("Paste the job description", height=220, placeholder="Paste the full job description here...")
    elif source == "Upload file":
        jd_up = st.file_uploader("Upload job description (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], key="jd_upload")
        if jd_up is not None:
            jd_text = read_jd_file(jd_up) or ""
            job_title = jd_up.name
    else:
        rec_default = get_default_recommender()
        options = {f"{r.title} - {r.company}": i for i, r in rec_default.jobs.iterrows()}
        choice = st.selectbox("Choose a job", list(options))
        row = rec_default.jobs.iloc[options[choice]]
        jd_text, job_title, company = row["description"], row["title"], row["company"]
        with st.expander("View job description"):
            st.write(jd_text)

    if st.button("Analyze match", type="primary", disabled=not jd_text.strip()):
        if len(jd_text.split()) < 8:
            st.warning("That job description is very short - results may not be meaningful.")
        matcher = JobMatcher(extractor, semantic, background_corpus=get_default_recommender().jobs["description"].tolist())
        with st.spinner("Comparing..."):
            res = matcher.match(text, jd_text, use_semantic)
        st.session_state["last_match"] = (res, job_title)
        db.save_match(R["id"], job_title, company, res.overall_score, res.matched_skills, res.missing_skills)

    if "last_match" in st.session_state:
        res, jt = st.session_state["last_match"]
        st.divider()
        st.markdown(f"### Result for: {jt}")
        show_match_result(res, "single", jt)

# ----------------------------------------------------------------------------- Recommendations
with tabs[3]:
    st.subheader("Best-matching jobs for your resume")
    custom_csv = st.file_uploader("Optional: use your own jobs CSV (columns: title, description, + optional company, location, experience_level)",
                                  type=["csv"], key="jobs_csv")
    recommender = get_default_recommender()
    jobs_key = "default"
    if custom_csv is not None:
        try:
            custom_jobs = load_jobs(df=pd.read_csv(custom_csv))
            recommender = JobRecommender(custom_jobs, extractor, JobMatcher(extractor, semantic))
            jobs_key = f"custom:{custom_csv.name}:{len(custom_jobs)}"
        except Exception as exc:
            st.error(f"Could not use this jobs file: {exc}")

    ranked, results = run_recommendations(text, jobs_key, use_semantic, recommender)

    top = ranked.head(top_n)
    fig = px.bar(top.sort_values("match_score"), x="match_score", y="title", orientation="h", color="match_score",
                 color_continuous_scale=["#d93025", "#f9ab00", "#34a853"], range_color=[0, 100],
                 text="match_score", title=f"Top {len(top)} recommended jobs")
    fig.update_layout(height=120 + 55 * len(top), margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None,
                      xaxis_title="Match score (%)", coloraxis_showscale=False, xaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, key="rec_bar")

    show_cols = ["title", "company", "location", "level", "match_score", "skill_score", "tfidf_score", "n_matched", "n_missing"]
    st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

    st.markdown("#### 🔍 Compare multiple jobs")
    all_titles = [f"{r.title} - {r.company}" for r in ranked.itertuples()]
    picked = st.multiselect("Select jobs to compare side by side", all_titles, default=all_titles[:3])
    if picked:
        sel = ranked[[f"{r.title} - {r.company}" in picked for r in ranked.itertuples()]]
        comp = sel.melt(id_vars=["title"], value_vars=["match_score", "skill_score", "tfidf_score"],
                        var_name="metric", value_name="score")
        fig = px.bar(comp, x="title", y="score", color="metric", barmode="group", title="Score breakdown")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), xaxis_title=None, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, key="rec_compare")

    st.markdown("#### 📌 Job details")
    detail_title = st.selectbox("Inspect a recommended job", all_titles[:max(top_n, 1)], key="detail_job")
    drow = ranked[[f"{r.title} - {r.company}" == detail_title for r in ranked.itertuples()]].iloc[0]
    show_match_result(results[int(drow["_idx"])], "rec_detail", drow["title"])

# ----------------------------------------------------------------------------- Score & tips
with tabs[4]:
    st.subheader("Resume quality")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        gauge(score["total"], "Overall resume score", key="resume_gauge")
    with c2:
        bd = pd.DataFrame([{"section": k, "points": v[0], "max": v[1], "pct": round(100 * v[0] / v[1])} for k, v in score["breakdown"].items()])
        fig = go.Figure(go.Scatterpolar(r=bd["pct"].tolist() + [bd["pct"].iloc[0]], theta=bd["section"].tolist() + [bd["section"].iloc[0]],
                                        fill="toself", name="Score"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=300, margin=dict(l=40, r=40, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="resume_radar")
    st.dataframe(bd.rename(columns={"pct": "% of max"}), use_container_width=True, hide_index=True)
    st.markdown("#### Improvement suggestions")
    show_suggestions(score["suggestions"])

# ----------------------------------------------------------------------------- History
with tabs[5]:
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("#### Resumes analysed")
        st.dataframe(db.get_resume_history(), use_container_width=True, hide_index=True)
    with h2:
        st.markdown("#### Job matches")
        st.dataframe(db.get_match_history(), use_container_width=True, hide_index=True)
    if st.button("Clear history"):
        db.clear_history()
        for k in ("last_match", "resume", "file_key"):  # force re-save of the current resume on rerun
            st.session_state.pop(k, None)
        st.rerun()
