import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request

from core.platform import platform
from memory.memory import Memory
from search.search import Search
from research.researcher import Researcher


app = Flask(__name__)

memory = Memory()
search = Search()
researcher = Researcher()


def setup_platform():
    platform.register_component(
        "memory",
        memory
    )

    platform.register_component(
        "search",
        search
    )

    platform.register_component(
        "research",
        researcher
    )


def get_platform_data():
    return {
        "platform": platform.status(),
        "memory_count": memory.count(),
        "search_count": search.count(),
        "research_count": len(
            researcher.tasks
        )
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        **get_platform_data()
    )


@app.route("/remember", methods=["POST"])
def remember():
    text = request.form.get(
        "text",
        ""
    ).strip()

    message = ""

    if text:
        memory.remember(text)

        search.add(
            content=text,
            title="Memory",
            category="memory",
            tags=["memory"]
        )

        message = (
            "Information saved to memory."
        )
    else:
        message = (
            "Please enter something."
        )

    return render_template(
        "index.html",
        message=message,
        **get_platform_data()
    )


@app.route("/search", methods=["POST"])
def search_platform():
    query = request.form.get(
        "query",
        ""
    ).strip()

    results = []

    if query:
        results = search.find(
            query
        )

    return render_template(
        "index.html",
        query=query,
        results=results,
        **get_platform_data()
    )


@app.route("/research", methods=["POST"])
def research():
    question = request.form.get(
        "question",
        ""
    ).strip()

    message = ""

    if question:
        task = researcher.create_task(
            question
        )

        memory.remember(
            f"Research question: {question}"
        )

        search.add(
            content=question,
            title="Research Question",
            category="research",
            tags=["research"]
        )

        message = (
            f"Research task #{task['id']} "
            "created."
        )
    else:
        message = (
            "Please enter a research question."
        )

    return render_template(
        "index.html",
        message=message,
        **get_platform_data()
    )


@app.route("/status")
def status():
    return platform.health_check()


def start_platform():
    setup_platform()

    if not platform.running:
        platform.start()


if __name__ == "__main__":
    start_platform()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )