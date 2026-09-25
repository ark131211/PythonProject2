import base64
import os

import anthropic
from flask import Flask, jsonify, render_template, request

MODEL = "claude-opus-5"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

SYSTEM_PROMPT = """Ты — терпеливый репетитор по математике. Помогаешь решать задачи \
любого уровня: арифметика, алгебра, геометрия, тригонометрия, анализ, вероятность, \
линейная алгебра.

Как отвечать:
- Отвечай на языке пользователя (по умолчанию — по-русски).
- Решай пошагово, коротко объясняя каждый шаг, чтобы ученик понял логику.
- Формулы пиши в LaTeX: $...$ внутри строки, $$...$$ для отдельных строк.
- Проверь ответ (подстановкой, оценкой или другим способом), если это возможно.
- В конце выдели итог строкой «**Ответ:** ...».
- Если условие неполное или неоднозначное — скажи, какое допущение делаешь."""

app = Flask(__name__)
client = anthropic.Anthropic()


def build_content(text: str, image) -> list:
    content = []
    if image is not None and image.filename:
        if image.mimetype not in ALLOWED_IMAGE_TYPES:
            raise ValueError("Поддерживаются только PNG, JPEG, GIF и WEBP.")
        data = image.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Картинка больше 5 МБ.")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mimetype,
                "data": base64.standard_b64encode(data).decode("utf-8"),
            },
        })
    if text:
        content.append({"type": "text", "text": text})
    elif content:
        content.append({"type": "text", "text": "Реши задачу на картинке."})
    return content


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/api/solve")
def solve():
    text = (request.form.get("problem") or "").strip()
    try:
        content = build_content(text, request.files.get("image"))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not content:
        return jsonify(error="Введите задачу или прикрепите фото."), 400

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError:
        return jsonify(error="Неверный ANTHROPIC_API_KEY на сервере."), 500
    except anthropic.RateLimitError:
        return jsonify(error="Слишком много запросов, попробуйте через минуту."), 429
    except anthropic.APIStatusError as e:
        return jsonify(error=f"Ошибка API ({e.status_code}): {e.message}"), 502
    except anthropic.APIConnectionError:
        return jsonify(error="Нет связи с API."), 502

    if response.stop_reason == "refusal":
        return jsonify(error="Модель отказалась отвечать на этот запрос."), 422

    answer = "".join(b.text for b in response.content if b.type == "text")
    return jsonify(answer=answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
