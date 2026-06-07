from flask import Flask, request, render_template
import fitz  # PyMuPDF
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")  # switched from 2.0-flash

# === Flask Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Helper: Extract Text from PDF ===
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# === Helper: Generate with Retry ===
def generate_with_retry(prompt, retries=3, wait=35):
    for i in range(retries):
        try:
            return model.generate_content([prompt]).text
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                time.sleep(wait)
            else:
                raise e

# === Route: Home Page ===
@app.route("/", methods=["GET", "POST"])
def index():
    parsed_data = None
    match_score = None
    feedback = None
    error = None

    if request.method == "POST":
        file = request.files["resume"]
        job_description = request.form["job_description"]

        if file:
            try:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                resume_text = extract_text_from_pdf(filename)

                # Step 1: Parse resume
                extract_prompt = f"""
You are an AI resume parser. Extract the following details in JSON format:
- Name
- Email
- Phone
- Skills
- Education
- Work Experience
- Certifications

Resume:
{resume_text}
"""
                parsed_data = generate_with_retry(extract_prompt)

                # Step 2: Match with Job Description
                match_prompt = f"""
Given the resume and job description, provide:
1. Match Score (0-100)
2. Short Explanation of the match

Resume:
{resume_text}

Job Description:
{job_description}
"""
                match_score = generate_with_retry(match_prompt)

                # Step 3: Feedback
                feedback_prompt = f"""
Provide feedback on how the following resume can be improved:
{resume_text}
"""
                feedback = generate_with_retry(feedback_prompt)

            except Exception as e:
                error = f"Something went wrong: {str(e)}"

    return render_template("index.html", parsed_data=parsed_data, match_score=match_score, feedback=feedback, error=error)

if __name__ == "__main__":
    app.run(debug=True)