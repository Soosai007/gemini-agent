import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

def get_weather(city: str) -> str:
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 1}, timeout=5).json()
        if not geo.get("results"):
            return f"Couldn't find '{city}'."
        r = geo["results"][0]
        wx = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": r["latitude"], "longitude": r["longitude"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto"
        }, timeout=5).json().get("current", {})
        conditions = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                      61:"Light rain",63:"Rain",65:"Heavy rain",80:"Showers",95:"Thunderstorm"}
        return (f"Weather in {r['name']}, {r.get('country','')}:\n"
                f"🌡️ Temperature : {wx.get('temperature_2m')}°C\n"
                f"💧 Humidity    : {wx.get('relative_humidity_2m')}%\n"
                f"💨 Wind speed  : {wx.get('wind_speed_10m')} km/h\n"
                f"☁️ Condition   : {conditions.get(wx.get('weather_code',0), 'Unknown')}")
    except Exception as e:
        return f"Weather lookup failed: {e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for any city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city e.g. Chennai, Mumbai, London"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

chat_histories = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if session_id not in chat_histories:
        chat_histories[session_id] = [
            {"role": "system", "content": "You are a helpful AI assistant called 'Aria'. Never reveal what AI model, LLM, or technology you are built on. If asked, just say you are Aria, a custom AI assistant. When users ask about weather or temperature in any city, you MUST use the get_weather function tool. Always pass the city name exactly as mentioned by the user."}
        ]

    chat_histories[session_id].append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chat_histories[session_id],
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # Handle tool call if needed
        if msg.tool_calls:
            import json
            tool_call = msg.tool_calls[0]
            city = json.loads(tool_call.function.arguments).get("city", "")
            weather_result = get_weather(city)

            # Add assistant + tool result to history
            chat_histories[session_id].append(msg)
            chat_histories[session_id].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": weather_result
            })

            # Get final response from model
            final = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=chat_histories[session_id]
            )
            reply = final.choices[0].message.content
        else:
            reply = msg.content

        chat_histories[session_id].append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    session_id = request.get_json(force=True).get("session_id", "default")
    chat_histories.pop(session_id, None)
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
