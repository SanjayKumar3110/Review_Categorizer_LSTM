from flask import Flask, render_template, request, redirect, url_for, flash
from model.test import get_youtube_comments, predict_sentiment, visualize_sentiments

app = Flask(__name__)
app.secret_key = "FLASK_SECRET_KEY"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_counts = None
    chart = None

    if request.method == "POST":
        video_link = request.form.get("video_link").strip()
        
        try:
            comments = get_youtube_comments(video_link)
            sentiments = predict_sentiment(comments)
            sentiment_counts = {
                "Positive": sentiments.count("Positive"),
                "Negative": sentiments.count("Negative"),
            }
            chart = visualize_sentiments(sentiment_counts)
        except Exception as e:
            flash(f"Error: {e}")
            return redirect(url_for("index"))
    return render_template("index.html",sentiment_counts=sentiment_counts, chart=chart)

if __name__ == "__main__":
     app.run(debug=True)


