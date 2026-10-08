from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Securely load API key


def generate_text(prompt):
    response = openai.responses.create(
        model="gpt-4.1",
        input=[{"role": "developer", "content": "you are nick land."},
               {"role": "user", "content": prompt}],
        temperature=1.2,
        max_output_tokens=50
    )
    return response.output_text


def generate_image(prompt):
    response = openai.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="medium"
    )
    return response.data[0].b64_json  # base64-encoded PNG


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    result = None
    image = None
    image_error = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        # Run both API calls at the same time so the page waits for the slower one, not both
        with ThreadPoolExecutor(max_workers=2) as pool:
            text_job = pool.submit(generate_text, prompt)
            image_job = pool.submit(generate_image, prompt)
            try:
                result = text_job.result()
            except Exception as e:
                result = f"Error: {str(e)}"
            try:
                image = image_job.result()
            except Exception as e:
                image_error = f"Error: {str(e)}"
    return render_template("index.html", prompt=prompt, result=result,
                           image=image, image_error=image_error)


if __name__ == "__main__":
    app.run(debug=True)  # Run locally for testing
