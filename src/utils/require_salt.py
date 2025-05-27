import os
from functools import wraps
from flask import Flask, request, jsonify

REQUIRE_API_TOKEN = os.getenv('REQUIRE_API_TOKEN','')

def require_salt(expected):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token != expected and REQUIRE_API_TOKEN.lower() == 'true':
                return jsonify({"error": "Unauthorized"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator
