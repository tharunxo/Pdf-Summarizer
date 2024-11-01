from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import PyPDF2

# Initialize Flask app
app = Flask(__name__)

# Initialize Hugging Face summarizer model
summarizer = pipeline("summarization", framework="pt")  # `pt` specifies PyTorch as the backend


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_file = request.files["pdf"]
        if pdf_file:
            text = extract_text_from_pdf(pdf_file)
            # Summarize the text (limit input size for summarization)
            summary = summarizer(text[:3000])[0]['summary_text']
            return jsonify(summary=summary)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
