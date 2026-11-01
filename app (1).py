
import streamlit as st
import pandas as pd
import re
import io
import html
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume Analyzer and Job Recommendation System")
st.write("Upload your resume to discover suitable job roles and skill gaps.")

# Job role dataset
job_roles = {
    "Job Role": [
        "Data Analyst",
        "Machine Learning Engineer",
        "AI Engineer",
        "NLP Engineer",
        "Computer Vision Engineer"
    ],
    "Required Skills": [
        "Python, SQL, Excel, Pandas, NumPy, Power BI, Statistics",
        "Python, Machine Learning, Scikit-learn, Pandas, NumPy, FastAPI, Docker, MLflow",
        "Python, Deep Learning, TensorFlow, PyTorch, LLM, RAG, APIs",
        "Python, NLP, Transformers, Hugging Face, NLTK, spaCy, Deep Learning",
        "Python, OpenCV, CNN, Deep Learning, TensorFlow, PyTorch, YOLO"
    ]
}

job_df = pd.DataFrame(job_roles)

skill_dictionary = {
    "Programming": ["Python", "Java", "C++", "C#", "JavaScript", "R"],
    "Data Analysis": ["SQL", "Excel", "Power BI", "Tableau", "Pandas", "NumPy", "Statistics"],
    "Machine Learning": ["Machine Learning", "Scikit-learn", "TensorFlow", "PyTorch", "Deep Learning", "MLflow"],
    "Artificial Intelligence": ["Artificial Intelligence", "LLM", "RAG", "Generative AI", "FastAPI", "APIs"],
    "Natural Language Processing": ["NLP", "NLTK", "spaCy", "Transformers", "Hugging Face"],
    "Computer Vision": ["OpenCV", "CNN", "YOLO", "Image Processing"],
    "Cloud and Tools": ["Docker", "AWS", "Azure", "Git", "GitHub", "Linux"],
    "Databases": ["MySQL", "PostgreSQL", "MongoDB", "SQLite"]
}

def extract_resume_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        return " ".join(page.extract_text() or "" for page in reader.pages)

    elif filename.endswith(".docx"):
        document = Document(io.BytesIO(uploaded_file.getvalue()))
        return " ".join(paragraph.text for paragraph in document.paragraphs)

    return ""

def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_skills(text):
    detected = {}

    for category, skills in skill_dictionary.items():
        found = []

        for skill in skills:
            pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

            if re.search(pattern, text):
                found.append(skill)

        if found:
            detected[category] = found

    return detected

def calculate_match_scores(resume_text):
    descriptions = job_df["Required Skills"].tolist()
    documents = [resume_text] + descriptions

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)

    scores = cosine_similarity(matrix[0:1], matrix[1:])[0]

    results = job_df.copy()
    results["Match Score (%)"] = (scores * 100).round(2)

    return results.sort_values(
        "Match Score (%)",
        ascending=False
    ).reset_index(drop=True)

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"]
)

if uploaded_file:

    resume_text = extract_resume_text(uploaded_file)
    cleaned_text = clean_resume_text(resume_text)

    if not cleaned_text:
        st.error("No text could be extracted. Please try another resume.")
        st.stop()

    extracted_skills = extract_skills(cleaned_text)
    results = calculate_match_scores(cleaned_text)

    st.success("Resume analyzed successfully!")

    st.subheader("Extracted Skills")

    if extracted_skills:
        for category, skills in extracted_skills.items():
            st.write(f"**{category}:** {', '.join(skills)}")
    else:
        st.write("No skills detected.")

    st.subheader("Recommended Job Roles")

    st.dataframe(
        results[["Job Role", "Match Score (%)"]],
        use_container_width=True
    )

    st.subheader("Top 3 Job Recommendations")

    for _, row in results.head(3).iterrows():
        st.write(
            f"**{row['Job Role']}** — "
            f"{row['Match Score (%)']}% match"
        )

    st.subheader("Target Role Skill Gap Analysis")

    target_role = st.selectbox(
        "Choose your target job role",
        job_df["Job Role"].tolist()
    )

    selected_job = job_df[
        job_df["Job Role"] == target_role
    ].iloc[0]

    required_skills = [
        skill.strip()
        for skill in selected_job["Required Skills"].split(",")
    ]

    found_skills = [
        skill
        for skills in extracted_skills.values()
        for skill in skills
    ]

    matched = [
        skill for skill in required_skills
        if skill.lower() in [s.lower() for s in found_skills]
    ]

    missing = [
        skill for skill in required_skills
        if skill.lower() not in [s.lower() for s in found_skills]
    ]

    st.write("**Skills found:**", ", ".join(matched) or "None")
    st.write("**Missing skills:**", ", ".join(missing) or "None")

    st.subheader("Suggested Learning Roadmap")

    for week, skill in enumerate(missing, start=1):
        st.write(
            f"**Week {week}:** Learn {skill} and complete a small practical project."
        )


    # Downloadable PDF Analysis Report
    st.subheader("Download Your Resume Analysis")

    report_buffer = io.BytesIO()
    pdf = SimpleDocTemplate(report_buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph(
        "AI Resume Analyzer - Analysis Report",
        styles["Title"]
    ))
    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "Target Role: " + html.escape(target_role),
        styles["Heading2"]
    ))

    target_score = results.loc[
        results["Job Role"] == target_role,
        "Match Score (%)"
    ].iloc[0]

    story.append(Paragraph(
        f"Target Role Match Score: {target_score}%",
        styles["Normal"]
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("Extracted Skills", styles["Heading2"]))

    for category, skills in extracted_skills.items():
        story.append(Paragraph(
            f"<b>{html.escape(category)}:</b> "
            + html.escape(", ".join(skills)),
            styles["Normal"]
        ))

    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Top 3 Job Recommendations",
        styles["Heading2"]
    ))

    recommendation_data = [["Job Role", "Match Score"]]

    for _, row in results.head(3).iterrows():
        recommendation_data.append([
            row["Job Role"],
            f'{row["Match Score (%)"]}%'
        ])

    recommendation_table = Table(recommendation_data)
    recommendation_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(recommendation_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Skills Found", styles["Heading2"]))
    story.append(Paragraph(
        html.escape(", ".join(matched) or "None"),
        styles["Normal"]
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Missing Skills", styles["Heading2"]))
    story.append(Paragraph(
        html.escape(", ".join(missing) or "None"),
        styles["Normal"]
    ))

    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Suggested Learning Roadmap",
        styles["Heading2"]
    ))

    for week, skill in enumerate(missing, start=1):
        story.append(Paragraph(
            f"Week {week}: Learn {html.escape(skill)} "
            "and complete a practical exercise.",
            styles["Normal"]
        ))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Disclaimer: Match scores are estimates based on text similarity "
        "and are not hiring probabilities or recruiter decisions.",
        styles["Italic"]
    ))

    pdf.build(story)
    report_buffer.seek(0)

    st.download_button(
        label="Download Analysis Report (PDF)",
        data=report_buffer.getvalue(),
        file_name="AI_Resume_Analysis_Report.pdf",
        mime="application/pdf"
    )

    st.caption(
        "Match scores are estimates based on text similarity, "
        "not hiring probabilities or recruiter decisions."
    )
