# 📄 AI Resume Analyzer & Job Matcher

An end-to-end, locally-running web app that parses a resume (PDF/DOCX), extracts skills and profile
information, compares it with any job description, scores the match, shows skill gaps and keyword analysis,
recommends the best-fitting jobs, and gives resume-improvement suggestions - all with **traditional NLP + ML**
(no paid APIs, no API keys). An optional Sentence-Transformers layer adds semantic matching.

> Built as a Data Science Engineering portfolio project: Python · NLP · TF-IDF/cosine similarity ·
> recommendation systems · SQLite · Streamlit · Plotly.

---

## ✨ Features

| Area | What it does |
|------|--------------|
| Resume upload | PDF & DOCX; handles multi-page PDFs, pages without text, empty / corrupted / wrong-format files with clear error messages |
| Extraction | Skills (204 across 13 categories, alias-aware), education, experience (date-range → years), projects, certifications, achievements, contact info |
| Job matching | Resume-vs-JD score from weighted skill coverage + TF-IDF cosine similarity (+ optional semantic similarity) |
| Gap analysis | Matched skills, missing skills ranked by priority (High/Medium/Low), coverage by category |
| Keyword analysis | Top TF-IDF keywords of the JD and whether your resume contains them |
| Recommendations | Ranks a catalogue of 20 jobs (or your own CSV) and lets you compare several side by side |
| Resume quality | 0-100 score across 8 sections + prioritised, actionable suggestions |
| History | Every analysis and match is stored in SQLite |

## 🧱 Architecture

```
Upload (PDF/DOCX)
   │
   ▼
resume_parser ──► text_processor ──► skill_extractor ──► resume_analyzer ──► resume_scorer
 (pdfplumber,       (clean, tokenise,   (data/skills.csv,    (sections, contact,    (quality score +
  python-docx)       protect C++/.NET)   regex + aliases)      education, exp...)     suggestions)
                                              │
 Job description ──► skill_extractor ─────────┤
                                              ▼
                                        job_matcher ──► job_recommender ──► Streamlit dashboard
                          (TF-IDF cosine + skill overlap     (rank data/jobs.csv)       (Plotly charts)
                           + optional embeddings)                                              │
                                                                                          SQLite history
```

## 🚀 Setup

```bash
# 1. Python 3.11+ recommended
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Recommended) NLP resources - the app still works without them using built-in fallbacks
python -m spacy download en_core_web_sm        # better name / organisation detection
python -m nltk.downloader stopwords            # fuller stopword list

# 4. Run
streamlit run app.py
```

**Optional semantic matching:** `pip install sentence-transformers`, then switch on
*"Semantic AI matching"* in the sidebar (first use downloads the ~90 MB `all-MiniLM-L6-v2` model).

Try it immediately with `assets/sample_resume.pdf` (also available as `.docx`).

## 🧮 How the match score works

```
skill_score  = weighted share of the JD's skills that appear in the resume
               (skills mentioned repeatedly in the JD weigh up to 2x; soft skills weigh 0.5x)
tfidf_score  = cosine(TF-IDF(resume), TF-IDF(JD)) / 0.30, capped at 1
               (raw cosine between a resume and a JD is naturally low, so it is rescaled)

overall = 0.60 * skill_score + 0.40 * tfidf_score                      # default
overall = 0.50 * skill_score + 0.25 * tfidf_score + 0.25 * semantic    # with Sentence-Transformers
```

The TF-IDF vectoriser is fitted on the whole job catalogue plus the two documents, so IDF weights stay
meaningful even when only one resume and one JD are compared. `notebooks/model_testing.ipynb` includes a
sensitivity check showing how stable the ranking is when the weights change.

## 📁 Project structure

```
AI-Resume-Analyzer/
├── app.py                    # Streamlit dashboard (UI only - no business logic)
├── requirements.txt
├── data/
│   ├── skills.csv            # skill, category, aliases (pipe-separated), case_sensitive
│   └── jobs.csv              # job catalogue used for recommendations
├── database/database.py      # SQLite persistence (resumes + match history)
├── utils/
│   ├── resume_parser.py      # extract_pdf_text / extract_docx_text / extract_resume_text
│   ├── text_processor.py     # cleaning, tokenisation, stopwords - tech-term safe
│   ├── skill_extractor.py    # CSV-driven, boundary-aware skill matching
│   ├── resume_analyzer.py    # sections, contact, education, experience, projects, certs
│   ├── job_matcher.py        # TF-IDF + skill overlap + optional semantic matcher
│   ├── job_recommender.py    # rank many jobs for one resume
│   └── resume_scorer.py      # resume quality score + suggestions
├── notebooks/model_testing.ipynb
├── models/  assets/
```

### Extending it
* **Add skills:** append a row to `data/skills.csv` (`skill,category,aliases,case_sensitive`) - no code changes.
* **Add jobs:** append rows to `data/jobs.csv`, or upload your own CSV in the *Recommendations* tab
  (needs `title` and `description` columns).

## 🧪 Testing performed

* Parsing: PDF, DOCX, multi-page PDF, PDF with no text, empty file, corrupted file, wrong extension → each returns a clear message, never a crash.
* Preprocessing: `Python, C++, C#, .NET, SQL, AWS, Power BI, Node.js, scikit-learn` all survive.
* Skill extraction on a small hand-labelled set: precision 0.96 / recall 1.00 (see notebook §2).
* Ranking sanity check: a Data Science resume ranks Data Science Intern, Data Analyst, Data Scientist at the top and DevOps / Java / Cloud at the bottom.
* Edge cases: JD with no recognisable skills falls back to text similarity (with a warning in the UI).

## ⚠️ Known limitations (good to be upfront about in an interview)

* Skill detection is dictionary-based: it only finds skills listed in `skills.csv`, and it can't tell
  *"used Docker daily"* from *"familiar with Docker"*. The single-letter skill **C** can still false-positive
  (e.g. *"grade C"*) - case-sensitive matching reduces but doesn't eliminate this.
* Section / education / experience parsing is rule-based and assumes conventional headings; very creative
  layouts (multi-column PDFs, graphics) may parse poorly.
* Scanned (image-only) PDFs are not supported - OCR is not included.
* The job catalogue is a small synthetic sample, not live job postings.
* Score weights and the TF-IDF rescaling constant were tuned by hand on a few examples, not learned from labelled hiring data.

## 🔭 Future work

* Learn the score weights from labelled resume/job pairs (e.g. logistic regression on the three sub-scores).
* Skill discovery with spaCy `EntityRuler` / n-gram mining to go beyond the CSV.
* OCR for scanned PDFs, FastAPI backend, PDF export of the analysis report, live job-board integration.

## 📝 Portfolio talking points

* Why TF-IDF + cosine first, embeddings second (interpretability, no GPU, no API cost).
* Why raw cosine is rescaled, and how you checked the weights' sensitivity.
* How you stopped `C++`, `C#` and `.NET` from being destroyed by tokenisation and stopword removal.
* Modular design: UI (`app.py`) is separate from parsing, NLP, matching and persistence.
