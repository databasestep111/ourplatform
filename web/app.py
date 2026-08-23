from flask import Flask, render_template, request

from core.platform import platform
from memory.memory import Memory
from search.search import Search
from research.researcher import Researcher


# ==========================================
# APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# PLATFORM COMPONENTS
# ==========================================

memory = Memory()
search = Search()
researcher = Researcher()


# ==========================================
# PLATFORM SETUP
# ==========================================

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


setup_platform()


# ==========================================
# START PLATFORM
# ==========================================

platform.start()


# ==========================================
# SHARED TEMPLATE DATA
# ==========================================

def template_data():

    return {
        "platform": platform.status(),
        "memory_count": memory.count(),
        "search_count": search.count(),
        "research_count": len(researcher.tasks),
    }


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        **template_data()
    )


# ==========================================
# SEARCH
# ==========================================

@app.route("/search", methods=["GET", "POST"])
def search_page():

    query = ""
    results = []

    if request.method == "POST":

        query = request.form.get(
            "query",
            ""
        ).strip()

        if query:

            results = search.find(query)

    return render_template(
        "search.html",
        query=query,
        results=results,
        **template_data()
    )


# ==========================================
# MEMORY
# ==========================================

@app.route("/memory", methods=["GET", "POST"])
def memory_page():

    message = None

    if request.method == "POST":

        text = request.form.get(
            "text",
            ""
        ).strip()

        if text:

            memory.remember(text)

            search.add(
                content=text,
                title="Memory",
                category="memory",
                tags=["memory"],
            )

            message = (
                "Information saved to memory."
            )

        else:

            message = (
                "Please enter something "
                "to remember."
            )

    return render_template(
        "memory.html",
        message=message,
        **template_data()
    )


# ==========================================
# RESEARCH
# ==========================================

@app.route("/research", methods=["GET", "POST"])
def research_page():

    message = None

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

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
                tags=["research"],
            )

            message = (
                f"Research task "
                f"#{task['id']} created."
            )

        else:

            message = (
                "Please enter a research "
                "question."
            )

    return render_template(
        "research.html",
        message=message,
        **template_data()
    )


# ==========================================
# AI ASSISTANT
# ==========================================

@app.route("/assistant")
def assistant_page():

    return render_template(
        "assistant.html",
        **template_data()
    )


# ==========================================
# SYSTEM
# ==========================================

@app.route("/system")
def system_page():

    health = platform.health_check()
    logs = platform.get_logs()
    errors = platform.get_errors()

    return render_template(
        "system.html",
        health=health,
        logs=logs,
        errors=errors,
        **template_data()
    )


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "platform": platform.status(),
        "health": platform.health_check(),
    }


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )