from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import cv2
import numpy as np
import json
import os
from datetime import datetime, timedelta
from utils.face_engine import (
    register_employee,
    recognize_face,
    log_attendance,
    log_attendance_with_event,
    get_today_summary,
    load_employees,
    load_attendance,
    get_employee_visit_count,
    check_face_exists,
    delete_employee,
    find_employee_by_id,
    calculate_work_hours,
    register_unknown_face
)

app = FastAPI(title="FaceTrack API")

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Key Protection ────────────────────────────────────────────────
API_KEY = "facetrack_secret_2024"

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/":
            return await call_next(request)
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            return JSONResponse(
                status_code=403,
                content={"error": "Unauthorized. Invalid API key."}
            )
        return await call_next(request)

app.add_middleware(APIKeyMiddleware)

ADMIN_PIN = "1234"
WATCHMAN_PIN = "0000"

# ── Auth ──────────────────────────────────────────────────────────────
@app.post("/login")
async def login(pin: str = Form(...)):
    if pin == ADMIN_PIN:
        return {"role": "admin", "success": True}
    elif pin == WATCHMAN_PIN:
        return {"role": "watchman", "success": True}
    raise HTTPException(status_code=401, detail="Wrong PIN")

# ── Check face duplicate ──────────────────────────────────────────────
@app.post("/check-duplicate")
async def check_duplicate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    existing_employee = check_face_exists(frame)
    if existing_employee:
        return {
            "success": True,
            "is_duplicate": True,
            "employee": existing_employee,
            "message": f"Face matches existing employee: {existing_employee['name']}"
        }
    return {
        "success": True,
        "is_duplicate": False,
        "message": "Face is new - safe to register"
    }

# ── Register employee (single image) ─────────────────────────────────
@app.post("/register")
async def register(
    name: str = Form(...),
    department: str = Form("General"),
    skip_duplicate_check: bool = Form(False),
    file: UploadFile = File(...)
):
    if not name or not name.strip():
        return {"success": False, "error": "Employee name required"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "error": "Invalid image"}

    if not skip_duplicate_check:
        existing = check_face_exists(frame)
        if existing:
            return {
                "success": False,
                "is_duplicate": True,
                "employee": existing,
                "error": f"Face matches existing employee: {existing['name']}"
            }

    success, result = register_employee(name.strip(), department, [frame])
    if success:
        return {
            "success": True,
            "employee_id": result,
            "name": name.strip(),
            "images_processed": 1
        }
    return {"success": False, "error": result}

# ── Register employee (multiple images) ──────────────────────────────
@app.post("/register-multi")
async def register_multi(
    name: str = Form(...),
    department: str = Form("General"),
    skip_duplicate_check: bool = Form(False),
    files: list[UploadFile] = File(...)
):
    if not name or not name.strip():
        return {"success": False, "error": "Employee name required"}

    if len(files) == 0:
        return {"success": False, "error": "At least 1 image required"}

    if len(files) > 5:
        return {"success": False, "error": "Maximum 5 images allowed"}

    frames = []
    for f in files:
        try:
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        except:
            pass

    if len(frames) == 0:
        return {"success": False, "error": "No valid images detected"}

    if not skip_duplicate_check:
        existing = check_face_exists(frames[0])
        if existing:
            return {
                "success": False,
                "is_duplicate": True,
                "employee": existing,
                "error": f"Face matches existing employee: {existing['name']}"
            }

    success, result = register_employee(name.strip(), department, frames)
    if success:
        return {
            "success": True,
            "employee_id": result,
            "name": name,
            "images_processed": len(frames)
        }
    return {"success": False, "error": result}

# ── Recognize face ────────────────────────────────────────────────────
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    employee, score, bbox = recognize_face(frame)

    if score == 0.0:
        return {
            "success": True,
            "recognized": False,
            "confidence": 0.0,
            "message": "No human detected"
        }

    if employee:
        ins, outs = get_employee_visit_count(employee['id'])
        return {
            "success": True,
            "recognized": True,
            "employee": employee,
            "confidence": round(float(score) * 100, 1),
            "today_in": ins,
            "today_out": outs
        }
    return {
        "success": True,
        "recognized": False,
        "confidence": round(float(score) * 100, 1),
        "message": "Unknown person"
    }

# ── Register unknown face ─────────────────────────────────────────────
@app.post("/register-unknown")
async def register_unknown(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "error": "Invalid image"}

    emp_id, result = register_unknown_face(frame)

    if emp_id:
        return {
            "success": True,
            "employee_id": emp_id,
            "name": result,
            "message": f"Registered as {result}"
        }
    return {"success": False, "error": result}

# ── Log attendance IN or OUT ──────────────────────────────────────────
@app.post("/log-attendance")
async def log_event(
    employee_id: str = Form(...),
    event: str = Form(...)
):
    if event not in ["IN", "OUT"]:
        raise HTTPException(status_code=400, detail="Event must be IN or OUT")

    employee = find_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    log_attendance_with_event(employee, event)
    ins, outs = get_employee_visit_count(employee_id)

    return {
        "success": True,
        "event": event,
        "employee": employee,
        "today_in": ins,
        "today_out": outs
    }

# ── Dashboard (admin only) ────────────────────────────────────────────
@app.get("/dashboard")
async def dashboard(pin: str):
    if pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Admins only")
    today_logs = get_today_summary()
    employees = load_employees()
    return {
        "total_employees": len(employees),
        "today_logs": today_logs,
        "present_today": len(set(l['employee_id'] for l in today_logs))
    }

# ── All attendance (admin only) ───────────────────────────────────────
@app.get("/attendance")
async def attendance(pin: str, date: str = None):
    if pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Admins only")
    all_logs = load_attendance()
    if date:
        all_logs = [l for l in all_logs if l['date'] == date]
    return {"logs": all_logs}

# ── Employee detailed report (admin only) ─────────────────────────────
@app.get("/employee-report")
async def employee_report(pin: str, employee_id: str, date: str = None):
    if pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Admins only")

    employee = find_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    all_logs = load_attendance()
    emp_logs = [l for l in all_logs if l['employee_id'] == employee_id]

    if date:
        emp_logs = [l for l in emp_logs if l['date'] == date]

    work_sessions = []
    in_time = None
    current_date = None

    for log in sorted(emp_logs, key=lambda x: x['timestamp']):
        if log['event'] == 'IN':
            in_time = log['timestamp']
            current_date = log['date']
        elif log['event'] == 'OUT' and in_time:
            hours, minutes, seconds = calculate_work_hours(in_time, log['timestamp'])
            work_sessions.append({
                "date": current_date,
                "in_time": in_time,
                "out_time": log['timestamp'],
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds,
                "total_minutes": hours * 60 + minutes + seconds / 60
            })
            in_time = None

    total_seconds_all = sum(
        s['hours'] * 3600 + s['minutes'] * 60 + s['seconds']
        for s in work_sessions
    )

    return {
        "success": True,
        "employee": employee,
        "work_sessions": work_sessions,
        "total_logs": len(emp_logs),
        "summary": {
            "total_hours": total_seconds_all // 3600,
            "total_minutes": (total_seconds_all % 3600) // 60,
            "total_seconds": total_seconds_all % 60,
            "days_present": len(set(s['date'] for s in work_sessions))
        }
    }

# ── All employees (admin only) ────────────────────────────────────────
@app.get("/employees")
async def employees(pin: str):
    if pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Admins only")
    return {"employees": load_employees()}

# ── Delete employee (admin only) ──────────────────────────────────────
@app.delete("/employees/{employee_id}")
async def delete_emp(employee_id: str, pin: str):
    if pin != ADMIN_PIN:
        raise HTTPException(status_code=403, detail="Admins only")
    success, message = delete_employee(employee_id)
    if success:
        return {"success": True, "message": message, "deleted_id": employee_id}
    raise HTTPException(status_code=404, detail=message)

# ── IP Camera config ──────────────────────────────────────────────────
@app.post("/start-ip-camera")
async def start_ip_camera(rtsp_url: str = Form(...), pin: str = Form(...)):
    if pin != ADMIN_PIN and pin != WATCHMAN_PIN:
        raise HTTPException(status_code=403, detail="Not authorized")
    with open('./data/camera_config.json', 'w') as f:
        json.dump({"rtsp_url": rtsp_url}, f)
    return {"success": True, "message": "Camera URL saved"}

@app.get("/camera-config")
async def get_camera_config():
    try:
        with open('./data/camera_config.json') as f:
            return json.load(f)
    except:
        return {"rtsp_url": None}

# ── Health check ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "FaceTrack API running", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)