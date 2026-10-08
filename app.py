from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Securely load API key

# Choices shown in the settings dropdowns (first item is the default)
TEXT_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]
IMAGE_QUALITIES = ["low", "medium", "high"]
IMAGE_SIZES = {"1024x1024": "Square", "1024x1536": "Portrait", "1536x1024": "Landscape"}

DEFAULTS = {
    "persona": "you are nick land.",
    "model": "gpt-4.1",
    "max_tokens": 50,
    "temperature": 1.2,
    "image_on": True,
    "image_from_reply": False,
    "image_quality": "medium",
    "image_size": "1024x1024",
    "image_style": "",
}


def clamp(value, low, high, default, cast):
    try:
        return max(low, min(high, cast(value)))
    except (TypeError, ValueError):
        return default


def read_settings(form):
    """Read the settings from the submitted form, falling back to defaults for bad values."""
    return {
        "persona": form.get("persona", DEFAULTS["persona"]),
        "model": form.get("model") if form.get("model") in TEXT_MODELS else DEFAULTS["model"],
        "max_tokens": clamp(form.get("max_tokens"), 16, 4000, DEFAULTS["max_tokens"], int),
        "temperature": clamp(form.get("temperature"), 0, 2, DEFAULTS["temperature"], float),
        "image_on": "image_on" in form,
        "image_from_reply": "image_from_reply" in form,
        "image_quality": form.get("image_quality") if form.get("image_quality") in IMAGE_QUALITIES else DEFAULTS["image_quality"],
        "image_size": form.get("image_size") if form.get("image_size") in IMAGE_SIZES else DEFAULTS["image_size"],
        "image_style": form.get("image_style", "").strip(),
    }


def generate_text(prompt, s):
    response = openai.responses.create(
        model=s["model"],
        input=[{"role": "developer", "content": s["persona"]},
               {"role": "user", "content": prompt}],
        temperature=s["temperature"],
        max_output_tokens=s["max_tokens"]
    )
    return response.output_text


def generate_image(prompt, s):
    if s["image_style"]:
        prompt = f"{s['image_style']}. {prompt}"
    response = openai.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size=s["image_size"],
        quality=s["image_quality"]
    )
    return response.data[0].b64_json  # base64-encoded PNG


@app.route("/", methods=["GET", "POST"])
def index():
    settings = dict(DEFAULTS)
    prompt = ""
    result = None
    image = None
    image_error = None
    if request.method == "POST":
        settings = read_settings(request.form)
        prompt = request.form["prompt"]
        if settings["image_on"] and settings["image_from_reply"]:
            # The image needs the reply, so the two calls run one after the other
            try:
                result = generate_text(prompt, settings)
            except Exception as e:
                result = f"Error: {str(e)}"
                image_error = "Skipped: no reply to illustrate."
            if image_error is None:
                try:
                    image = generate_image(result, settings)
                except Exception as e:
                    image_error = f"Error: {str(e)}"
        elif settings["image_on"]:
            # Run both API calls at the same time so the page waits for the slower one, not both
            with ThreadPoolExecutor(max_workers=2) as pool:
                text_job = pool.submit(generate_text, prompt, settings)
                image_job = pool.submit(generate_image, prompt, settings)
                try:
                    result = text_job.result()
                except Exception as e:
                    result = f"Error: {str(e)}"
                try:
                    image = image_job.result()
                except Exception as e:
                    image_error = f"Error: {str(e)}"
        else:
            try:
                result = generate_text(prompt, settings)
            except Exception as e:
                result = f"Error: {str(e)}"
    return render_template("index.html", prompt=prompt, result=result,
                           image=image, image_error=image_error, s=settings,
                           text_models=TEXT_MODELS, image_qualities=IMAGE_QUALITIES,
                           image_sizes=IMAGE_SIZES)


if __name__ == "__main__":
    app.run(debug=True)  # Run locally for testing
