from datetime import datetime


class Researcher:
    def __init__(self):
        self.name = "OurPlatform Researcher"
        self.tasks = []
        self.sources = []
        self.findings = []
        self.next_task_id = 1
        self.next_source_id = 1
        self.next_finding_id = 1

    # -------------------------
    # RESEARCH TASKS
    # -------------------------

    def create_task(self, question, priority="normal"):
        task = {
            "id": self.next_task_id,
            "question": question,
            "priority": priority,
            "status": "planned",
            "created_at": datetime.now().isoformat(),
        }

        self.tasks.append(task)
        self.next_task_id += 1

        return task

    def get_task(self, task_id):
        for task in self.tasks:
            if task["id"] == task_id:
                return task

        return None

    def update_task(self, task_id, status=None, priority=None):
        task = self.get_task(task_id)

        if task is None:
            return None

        if status is not None:
            task["status"] = status

        if priority is not None:
            task["priority"] = priority

        return task

    def list_tasks(self, status=None):
        if status is None:
            return self.tasks.copy()

        return [
            task
            for task in self.tasks
            if task["status"] == status
        ]

    # -------------------------
    # SOURCE MANAGEMENT
    # -------------------------

    def add_source(
        self,
        title,
        url=None,
        source_type="unknown",
        credibility=0,
    ):
        source = {
            "id": self.next_source_id,
            "title": title,
            "url": url,
            "type": source_type,
            "credibility": credibility,
            "added_at": datetime.now().isoformat(),
        }

        self.sources.append(source)
        self.next_source_id += 1

        return source

    def get_source(self, source_id):
        for source in self.sources:
            if source["id"] == source_id:
                return source

        return None

    def list_sources(self):
        return self.sources.copy()

    # -------------------------
    # FINDINGS
    # -------------------------

    def add_finding(
        self,
        task_id,
        statement,
        source_ids=None,
        confidence=0,
    ):
        if source_ids is None:
            source_ids = []

        finding = {
            "id": self.next_finding_id,
            "task_id": task_id,
            "statement": statement,
            "source_ids": source_ids,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
        }

        self.findings.append(finding)
        self.next_finding_id += 1

        return finding

    def get_findings(self, task_id):
        return [
            finding
            for finding in self.findings
            if finding["task_id"] == task_id
        ]

    # -------------------------
    # COMPARISON
    # -------------------------

    def compare_findings(self, task_id):
        findings = self.get_findings(task_id)

        if not findings:
            return {
                "task_id": task_id,
                "findings": [],
                "agreement": "no_data",
            }

        statements = [
            finding["statement"]
            for finding in findings
        ]

        confidence = sum(
            finding["confidence"]
            for finding in findings
        ) / len(findings)

        return {
            "task_id": task_id,
            "findings": statements,
            "average_confidence": confidence,
            "agreement": self._estimate_agreement(
                findings
            ),
        }

    def _estimate_agreement(self, findings):
        if len(findings) <= 1:
            return "insufficient_data"

        confidence_values = [
            finding["confidence"]
            for finding in findings
        ]

        difference = max(
            confidence_values
        ) - min(
            confidence_values
        )

        if difference <= 1:
            return "high"

        if difference <= 3:
            return "moderate"

        return "mixed"

    # -------------------------
    # RESEARCH PLAN
    # -------------------------

    def create_plan(self, question):
        return {
            "question": question,
            "steps": [
                "Define the research question",
                "Identify useful sources",
                "Collect relevant information",
                "Compare findings",
                "Check source quality",
                "Identify uncertainty",
                "Create a final report",
            ],
        }

    # -------------------------
    # REPORTING
    # -------------------------

    def create_report(self, task_id):
        task = self.get_task(task_id)

        if task is None:
            return None

        findings = self.get_findings(task_id)

        return {
            "title": f"Research Report: {task['question']}",
            "question": task["question"],
            "status": task["status"],
            "findings": findings,
            "source_count": len(self.sources),
            "created_at": datetime.now().isoformat(),
        }

    # -------------------------
    # STATISTICS
    # -------------------------

    def statistics(self):
        return {
            "tasks": len(self.tasks),
            "sources": len(self.sources),
            "findings": len(self.findings),
            "completed_tasks": len(
                self.list_tasks("completed")
            ),
        }


researcher = Researcher()