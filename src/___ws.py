""""
curl -X POST "http://127.0.0.1:5000/api/groups/batch/all" -H "Authorization: Bearer test" -H "Content-Type: application/json" -d '{"semester": "WS23/24", "time_until": 42, "format": "json"}'
 
"""
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_jwt_extended import JWTManager
from datetime import datetime
from flask import Response
import json
import sqlite3
import time

from load import Load
from extract_easy_sync import Extract_Easy_Sync

DB_PATH = "jobs.db"

# Initialize Flask app
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "test"  # Change this to a secure key
jwt = JWTManager(app)

# Initialize Flask-RESTX API
api = Api(app, version="1.0", title="Secure API", description="API with Bearer Token Authentication")

# Define API Namespace
ns = api.namespace("api", description="User Actions")

# Swagger Model for API Documentation
action_model = api.model(
    "Action",
    {
        "semester": fields.String(required=True, description="Semester"),
        #"group_id": fields.Integer(required=True, description="Group ID"),
        #"user_id": fields.Integer(required=True, description="User ID"),
        "time_until": fields.String(required=False, description="Time range (ISO 8601 format, optional)"),
        "format": fields.String(required=False, description="Response format (json, xml, etc.), default: json"),
    },
)
jobs_model = api.model(
    "Jobs",
    {
        #"group_id": fields.Integer(required=True, description="Group ID"),
        #"user_id": fields.Integer(required=True, description="User ID"),
        #"action": fields.String(required=True, description="Action name"),
        #"time_range": fields.String(required=False, description="Time range (ISO 8601 format, optional)"),
        #"format": fields.String(required=False, description="Response format (json, xml, etc.), default: json"),
    },
)

# Dummy Bearer Token for Testing
VALID_BEARER_TOKEN = "test"

# Helper function to validate bearer token
def verify_bearer_token():
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return False, {"message": "Missing or invalid Bearer token", "status": 401}

    token = auth_header.split(" ")[1]
    if token != VALID_BEARER_TOKEN:
        return False, {"message": "Unauthorized: Invalid Bearer token", "status": 403}

    return True, None

@ns.route("/groups/batch/all")
class UserAction(Resource):
    @api.expect(action_model)
    @api.response(200, "Success")
    @api.response(400, "Validation Error")
    @api.response(401, "Missing or invalid Bearer Token")
    @api.response(403, "Unauthorized")
    def post(self):
        """Process user actions with authentication"""
        is_valid, error_response = verify_bearer_token()
        if not is_valid:
            return error_response, error_response["status"]

        data = request.json
        semester = data.get("semester")
        time_range = data.get("time_until")
        format_type = data.get("format", "json")
        params = {"semester": semester, "time_range": time_range}
        jt = JobTracker()
        job_id = jt.add_job("some example job")
        #process_job(job_id, "example_job", params)
        process_job(job_id, "batch", params)
        
        return {
            "semester": semester,
            "time_until": time_range or "Not provided",
            "format": format_type,
            "status": "Processed successfully",
        }, 200

@ns.route("/jobs/get_all")
class JobHandling(Resource):
    #@api.expect(jobs_model)
    @api.response(200, "Success")
    @api.response(400, "Validation Error")
    #@api.response(401, "Missing or invalid Bearer Token")
    #@api.response(403, "Unauthorized")
    def get(self):
        jt = JobTracker()
        jobs = jt.get_all_jobs()
        print('ALL JOBS',jobs)
        return Response(json.dumps(jobs), status=200, mimetype="application/json")


class JobTracker:
    def __init__(self):
        self._initialize_db()

    def _initialize_db(self):
        """Initialize SQLite database for job tracking."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id INTEGER PRIMARY KEY,
                status TEXT,
                description TEXT    
            )
        """)
        conn.commit()
        conn.close()

    def add_job(self, job_description):
        """Add a new job to the tracking database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO jobs (status, description) VALUES (?, ?)", ("PENDING", job_description))
            conn.commit()
            job_id = cursor.lastrowid  # Store last inserted ID before closing the connection
            print("new job_id", job_id)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            conn.rollback()  # Rollback in case of error
            job_id = None
        finally:
            cursor.close()  # Close cursor
            conn.close()  # Close connection
        
        return job_id  # Return the job ID or None if failed

    def get_job_status(self, job_id):
        """Retrieve the job status and progress."""
        res = {"status": 'no information'}
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT job_id, status, description FROM jobs WHERE job_id=?", (job_id))
        for row in cursor.fetchall():
            res = {"status": row[1], "description": row[2]}
        conn.close()
        return res

    def update_job_status(self, job_id, job_status, job_description):
        """Update the job status and description."""
        res = {"status": 'no information'}
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE jobs SET description=?, status=? WHERE job_id=?", (job_description, job_status, job_id))
            conn.commit()
            if cursor.rowcount > 0:
                res["status"] = "updated"
            else:
                res["status"] = "job not found"
        except sqlite3.Error as e:
            res["status"] = f"error: {str(e)}"
            print(res["status"])
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
        return res
        

    def get_all_jobs(self):
        """Retrieve all jobs and their statuses."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        jobs = []
        try:
            cursor.execute("SELECT * FROM jobs")
            jobs = [{"job_id": row[0], "status": row[1], "description": row[2]} for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            jobs = [{"error": "db error"}]
        finally:
            cursor.close()
            conn.close()
        return jobs


def process_job(job_id, job_name, params):
    jt = JobTracker()
    match job_name:
        case "example_job":
            for i in range(10):
                print(f"example_job {i + 1}")
                jt.update_job_status(job_id, "PENDING", f"iteration {i +1}")
                time.sleep(2)
            jt.update_job_status(job_id, "SUCCESS", f"finished {i +1} iterations")
        case "batch":
            jt.update_job_status(job_id, "PENDING", f"start loading csv")
            l = Load()
            df = l.load_from_csv()
            jt.update_job_status(job_id, "PENDING", f"finished loading csv of {df.size} rows")
            jt.update_job_status(job_id, "PENDING", f"start easy sync")
            es = Extract_Easy_Sync()
            df_textchanges = es.extract_easy_sync(df)
            jt.update_job_status(job_id, "PENDING", f"finished easy sync of about {df_textchanges.size} rows")
            # df_textchanges
            jt.update_job_status(job_id, "SUCCESS", f"finished all")
            

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
    