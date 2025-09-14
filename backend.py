"""
FastAPI backend for Food Calorie Calculator & Tracker
- Raw MySQL via PyMySQL (no ORM)
- JWT auth (Bearer in Swagger), bcrypt password hashing
- Optional LLM (Groq) parsing for food entries

How to run:
1) Make sure MySQL is running and the schema below exists.
2) Create .env (see example at the end).
3) Install deps: pip install -r requirements.txt
4) Run: uvicorn backend:app --reload
5) Open Swagger: http://127.0.0.1:8000/docs  (click "Authorize")

MySQL schema:

CREATE DATABASE IF NOT EXISTS food_tracker;
USE food_tracker;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  gender ENUM('male','female','other') DEFAULT 'other',
  username VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profiles (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  age INT,
  height_cm FLOAT,
  weight_kg FLOAT,
  activity_level ENUM('sedentary','light','moderate','active','very_active') DEFAULT 'sedentary',
  bmr FLOAT,
  tdee FLOAT,
  bmi FLOAT,
  weight_status VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS food_entries (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  entry_date DATE NOT NULL,
  raw_input TEXT,
  parsed_json JSON,
  total_calories FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

import os
import json
import datetime
from typing import Optional, List
from enum import Enum

import pymysql
from pymysql.err import OperationalError
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import bcrypt
import jwt
import requests
from dotenv import load_dotenv

# ---------- Env ----------
load_dotenv()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "food_tracker")
JWT_SECRET = os.getenv("JWT_SECRET", "replace_with_secret")
GROQ_API_URL = os.getenv("GROQ_API_URL")  # optional
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # optional

# ---------- App ----------
app = FastAPI(
    title="Food Calorie Tracker API",
    description="Signup/Login/Profile/Food tracking with JWT Bearer auth and optional Groq parsing.",
    version="1.0.0",
)

# CORS (allow local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB Helpers ----------
def get_db_conn():
    try:
        return pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(os.getenv("DB_PORT", 3306)),  
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except OperationalError as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")


def db_execute(
    query: str,
    params: tuple = (),
    *,
    fetchone: bool = False,
    fetchall: bool = False,
    commit: bool = False,
):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if commit:
                conn.commit()
            if fetchone:
                return cur.fetchone()
            if fetchall:
                return cur.fetchall()
            return None
    finally:
        conn.close()

# ---------- Enums / Models ----------
class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"

class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    active = "active"
    very_active = "very_active"

class SignupRequest(BaseModel):
    name: str
    gender: Optional[Gender] = Gender.other
    username: str
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileRequest(BaseModel):
    age: int
    height_cm: float
    weight_kg: float
    activity_level: ActivityLevel

class FoodTrackRequest(BaseModel):
    text: str

# ---------- JWT / Swagger Security ----------
bearer_scheme = HTTPBearer(auto_error=False)  # shows "Authorize" in /docs

def create_jwt(payload: dict, expires_days: int = 7) -> str:
    payload = payload.copy()
    payload["exp"] = datetime.datetime.utcnow() + datetime.timedelta(days=expires_days)
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token if isinstance(token, str) else token.decode("utf-8")

def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    # credentials will be None if not provided via Swagger Authorize
    if not credentials or not credentials.scheme or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    data = decode_jwt(token)
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db_execute(
        "SELECT id, name, username, gender FROM users WHERE id=%s",
        (user_id,),
        fetchone=True,
    )
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---------- Password Helpers ----------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

# ---------- Calculations ----------
def calc_bmr_mifflin(weight_kg: float, height_cm: float, age: int, gender: Optional[str]) -> float:
    g = (gender or "other").lower()
    if g == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif g == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr_m = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        bmr_f = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        bmr = (bmr_m + bmr_f) / 2
    return round(bmr, 2)

ACTIVITY_MULTIPLIER = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

def calc_tdee(bmr: float, activity_level: str) -> float:
    mult = ACTIVITY_MULTIPLIER.get(activity_level, 1.2)
    return round(bmr * mult, 2)

def calc_bmi(weight_kg: float, height_cm: float) -> float:
    if height_cm <= 0:
        return 0.0
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m * h_m), 2)

def weight_status_from_bmi(bmi: float) -> str:
    if bmi == 0:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"

# ---------- LLM (Groq) ----------
def build_groq_prompt(user_input: str) -> str:
    return f"""
When given a user's natural-language description of foods eaten, you MUST return ONLY a valid JSON object (no extra commentary).
The JSON must follow this structure:
{{
  "food_items": [
    {{"name": "<food name>", "quantity": "<text quantity>", "calories": <number>, "protein_g": <number>, "fat_g": <number>, "carbs_g": <number>}}
  ],
  "total_calories": <number>,
  "healthiness": "Healthy|Moderate|Unhealthy"
}}
Use reasonable approximate nutrition values for typical portion sizes.
Now process this input exactly and respond only with JSON:
{user_input}
""".strip()



def send_to_llm(prompt: str) -> str:
    if not GROQ_API_URL or not GROQ_API_KEY:
        # You can choose to 502 here; returning a dummy JSON can help local testing
        # Raise error to be explicit:
        raise HTTPException(status_code=500, detail="GROQ_API_URL and GROQ_API_KEY must be set")
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    # Format request for Groq Chat Completions API
    body = {
        "model": "llama-3.1-8b-instant",  # Using Llama 3.1 model available on Groq
        "messages": [
            {"role": "system", "content": "You are a nutrition assistant that responds only with JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800
    }
    
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=20)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
        
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LLM returned {resp.status_code}: {resp.text}")

    data = resp.json()
    # Extract content from Groq Chat Completions API response
    if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
        message = data["choices"][0].get("message", {})
        if isinstance(message, dict) and "content" in message:
            return message["content"]
    
    # Fallback to older response formats
    if isinstance(data, dict):
        if isinstance(data.get("output"), str):
            return data["output"]
        if isinstance(data.get("text"), str):
            return data["text"]
        if isinstance(data.get("choices"), list) and data["choices"]:
            first = data["choices"][0]
            if isinstance(first, dict):
                if isinstance(first.get("text"), str):
                    return first["text"]
                if isinstance(first.get("output"), str):
                    return first["output"]
    
    # If we couldn't parse the API response
    return resp.text

# ---------- Endpoints ----------
@app.post("/signup")
def signup(req: SignupRequest):
    if db_execute("SELECT id FROM users WHERE username=%s", (req.username,), fetchone=True):
        raise HTTPException(status_code=400, detail="Username already exists")
    pwd_hash = hash_password(req.password)
    db_execute(
        "INSERT INTO users (name, gender, username, password_hash) VALUES (%s,%s,%s,%s)",
        (req.name, (req.gender or "other"), req.username, pwd_hash),
        commit=True,
    )
    return {"status": "ok", "message": "User created"}

@app.post("/login")
def login(req: LoginRequest):
    user = db_execute(
        "SELECT id, username, password_hash, name, gender FROM users WHERE username=%s",
        (req.username,),
        fetchone=True,
    )
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_jwt({"user_id": user["id"], "username": user["username"]})
    return {"access_token": token}

@app.post("/profile", dependencies=[Depends(bearer_scheme)])
def create_or_update_profile(req: ProfileRequest, user=Depends(get_current_user)):
    user_id = user["id"]
    bmr = calc_bmr_mifflin(req.weight_kg, req.height_cm, req.age, user.get("gender"))
    tdee = calc_tdee(bmr, req.activity_level.value if isinstance(req.activity_level, ActivityLevel) else req.activity_level)
    bmi = calc_bmi(req.weight_kg, req.height_cm)
    status_txt = weight_status_from_bmi(bmi)

    exists = db_execute("SELECT id FROM profiles WHERE user_id=%s", (user_id,), fetchone=True)
    if exists:
        db_execute(
            """UPDATE profiles
               SET age=%s, height_cm=%s, weight_kg=%s, activity_level=%s, bmr=%s, tdee=%s, bmi=%s, weight_status=%s
               WHERE user_id=%s""",
            (req.age, req.height_cm, req.weight_kg, req.activity_level, bmr, tdee, bmi, status_txt, user_id),
            commit=True,
        )
    else:
        db_execute(
            """INSERT INTO profiles (user_id, age, height_cm, weight_kg, activity_level, bmr, tdee, bmi, weight_status)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (user_id, req.age, req.height_cm, req.weight_kg, req.activity_level, bmr, tdee, bmi, status_txt),
            commit=True,
        )
    return {"bmr": bmr, "tdee": tdee, "bmi": bmi, "weight_status": status_txt}

@app.post("/food/track", dependencies=[Depends(bearer_scheme)])
def track_food(req: FoodTrackRequest, user=Depends(get_current_user)):
    profile = db_execute("SELECT tdee FROM profiles WHERE user_id=%s", (user["id"],), fetchone=True)
    if not profile:
        raise HTTPException(status_code=400, detail="Set up profile first to calculate calorie limit")

    prompt = build_groq_prompt(req.text)
    llm_text = send_to_llm(prompt)

    # Parse JSON strictly (with fallback extraction)
    try:
        parsed = json.loads(llm_text)
    except Exception:
        s, e = llm_text.find("{"), llm_text.rfind("}")
        if s == -1 or e == -1:
            raise HTTPException(status_code=502, detail="LLM did not return valid JSON")
        try:
            parsed = json.loads(llm_text[s : e + 1])
        except Exception:
            raise HTTPException(status_code=502, detail="LLM returned invalid JSON payload")

    if "total_calories" not in parsed:
        raise HTTPException(status_code=502, detail="LLM JSON missing total_calories")

    total_calories = float(parsed["total_calories"])
    today = datetime.date.today()

    db_execute(
        "INSERT INTO food_entries (user_id, entry_date, raw_input, parsed_json, total_calories) VALUES (%s,%s,%s,%s,%s)",
        (user["id"], today, req.text, json.dumps(parsed), total_calories),
        commit=True,
    )

    row = db_execute(
        "SELECT SUM(total_calories) AS sum_cal FROM food_entries WHERE user_id=%s AND entry_date=%s",
        (user["id"], today),
        fetchone=True,
    )
    today_total = float(row["sum_cal"] or 0.0)
    tdee = float(profile["tdee"])
    remaining = tdee - today_total

    status_flag = "ok"
    if today_total > tdee:
        status_flag = "exceeded"
    elif today_total < 0.5 * tdee:
        status_flag = "very_low"

    return {
        "parsed": parsed,
        "total_calories_entry": total_calories,
        "today_total_calories": today_total,
        "tdee": tdee,
        "remaining": remaining,
        "status": status_flag,
    }

@app.get("/dashboard/daily", dependencies=[Depends(bearer_scheme)])
def dashboard_daily(date: Optional[str] = None, user=Depends(get_current_user)):
    if not date:
        d = datetime.date.today()
    else:
        try:
            d = datetime.date.fromisoformat(date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format (use YYYY-MM-DD)")

    profile = db_execute("SELECT tdee FROM profiles WHERE user_id=%s", (user["id"],), fetchone=True)
    tdee = float(profile["tdee"]) if profile else None

    entries = db_execute(
        """SELECT id, raw_input, parsed_json, total_calories, created_at
           FROM food_entries
           WHERE user_id=%s AND entry_date=%s
           ORDER BY created_at""",
        (user["id"], d),
        fetchall=True,
    ) or []

    total_row = db_execute(
        "SELECT SUM(total_calories) AS sum_cal FROM food_entries WHERE user_id=%s AND entry_date=%s",
        (user["id"], d),
        fetchone=True,
    )
    today_total = float(total_row["sum_cal"] or 0.0)

    return {"date": d.isoformat(), "tdee": tdee, "today_total": today_total, "entries": entries}

# Health check
@app.get("/")
def root():
    return {"status": "ok", "now": datetime.datetime.utcnow().isoformat()}
