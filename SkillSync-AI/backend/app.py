from flask import Flask, request, jsonify

app = Flask(__name__)

# Temporary Database
projects = [
    {
        "project id": 1,
        "project name": "Skillsync AI",
        "description": "AI powered project management system",
        "owner": "Honey",
        "status": "In Progress",
        "deadline": "2026-07-20"
    }
]

# Home Route
@app.route("/")
def home():
    return {"message": "Skillsync AI Backend Running 🚀"}

# GET All Projects
@app.route("/api/projects", methods=["GET"])
def get_projects():
    return jsonify(projects)

# POST Create Project
@app.route("/api/projects", methods=["POST"])
def create_project():

    data = request.get_json()

    new_project = {
        "project id": len(projects) + 1,
        "project name ": data["projectName"],
        "description": data["description"],
        "owner": data["owner"],
        "status": data["status"],
        "deadline": data["deadline"]
    }

    projects.append(new_project)

    return jsonify({
        "message": "Project Created Successfully",
        "project": new_project
    }), 201


# PUT Update Project
@app.route("/api/projects/<int:project_id>", methods=["PUT"])
def update_project(project_id):

    data = request.get_json()

    for project in projects:
        if project["project id"] == project_id:

            project["project name "] = data.get("projectName", project["project name "])
            project["description"] = data.get("description", project["description"])
            project["owner"] = data.get("owner", project["owner"])
            project["status"] = data.get("status", project["status"])
            project["deadline"] = data.get("deadline", project["deadline"])

            return jsonify({
                "message": "Project Updated Successfully",
                "project": project
            })

    return jsonify({
        "message": "Project Not Found"
    }), 404


# DELETE Project
@app.route("/api/projects/<int:project_id>", methods=["DELETE"])
def delete_project(project_id):

    for project in projects:
        if project["project id"] == project_id:
            projects.remove(project)

            return jsonify({
                "message": "Project Deleted Successfully"
            }), 200

    return jsonify({
        "message": "Project Not Found"
    }), 404


if __name__ == "__main__":
    app.run(debug=True)