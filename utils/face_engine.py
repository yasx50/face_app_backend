import cv2
import numpy as np
import threading
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from utils.database import (
    init_db,
    save_employee,
    get_all_employees,
    get_employee_by_id,
    delete_employee_db,
    find_similar_face,
    rename_employee_db,
    save_attendance_log,
    get_attendance_logs,
    get_today_logs,
    get_visit_count_today,
    count_employees
)

# ── Load ArcFace model once at startup ───────────────────────────────
face_app = FaceAnalysis(
    name='buffalo_l',
    root='./models',
    allowed_modules=['detection', 'recognition']  # skip unnecessary modules
)
face_app.prepare(ctx_id=0, det_size=(320, 320))  # 320 faster than 640 for speed

# Init DB tables
init_db()

# ── In-memory employee cache ──────────────────────────────────────────
_employee_cache = {}
_cache_lock = threading.Lock()
_cache_time = 0
CACHE_TTL = 60  # refresh every 60 seconds

def _refresh_cache():
    global _employee_cache, _cache_time
    _employee_cache = get_all_employees()
    _cache_time = time.time()

def get_cached_employees():
    global _cache_time
    with _cache_lock:
        if time.time() - _cache_time > CACHE_TTL:
            _refresh_cache()
    return _employee_cache

def invalidate_cache():
    global _cache_time
    with _cache_lock:
        _cache_time = 0

# Preload cache on startup
_refresh_cache()

# ── Get face embedding ────────────────────────────────────────────────
def get_embedding(frame):
    # Resize frame to 320x320 before processing — 3x faster
    small = cv2.resize(frame, (320, 320))
    faces = face_app.get(small)
    if not faces:
        return None, None
    # Get best face (largest area)
    best = max(faces, key=lambda f: (
        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    ))
    return best.embedding, best.bbox.astype(int)

# ── Register employee ─────────────────────────────────────────────────
def register_employee(name, department, frames):
    embeddings = []
    for frame in frames:
        emb, _ = get_embedding(frame)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return False, "No face detected"

    avg_embedding = np.mean(embeddings, axis=0)
    emp_id = f"EMP{str(count_employees() + 1).zfill(3)}"
    save_employee(emp_id, name, department, avg_embedding.tolist())
    invalidate_cache()
    return True, emp_id

# ── Register unknown face ─────────────────────────────────────────────
def register_unknown_face(frame):
    employees = get_cached_employees()
    unknown_count = sum(
        1 for e in employees.values()
        if e['name'].startswith('Unknown')
    )
    unknown_name = f"Unknown {unknown_count + 1}"
    success, result = register_employee(unknown_name, "Unknown", [frame])
    if success:
        return result, unknown_name
    return None, result

# ── Check duplicate face ──────────────────────────────────────────────
def check_face_exists(frame):
    emb, _ = get_embedding(frame)
    if emb is None:
        return None
    employee, score = find_similar_face(emb.tolist(), threshold=0.6)
    return employee

# ── Recognize face (uses pgvector — fast) ────────────────────────────
def recognize_face(frame):
    emb, bbox = get_embedding(frame)
    if emb is None:
        return None, 0.0, None
    employee, score = find_similar_face(emb.tolist(), threshold=0.4)
    return employee, score, bbox

# ── Log attendance ────────────────────────────────────────────────────
def log_attendance(employee):
    ins, outs = get_visit_count_today(employee['id'])
    event = 'IN' if ins == outs else 'OUT'
    save_attendance_log(
        employee['id'],
        employee['name'],
        employee['department'],
        event
    )
    return event

def log_attendance_with_event(employee, event):
    save_attendance_log(
        employee['id'],
        employee['name'],
        employee['department'],
        event
    )
    return event

# ── Employee helpers ──────────────────────────────────────────────────
def load_employees():
    return get_cached_employees()

def load_attendance():
    return get_attendance_logs()

def get_today_summary():
    return get_today_logs()

def get_employee_visit_count(emp_id):
    return get_visit_count_today(emp_id)

def find_employee_by_id(emp_id):
    # Check cache first
    cached = get_cached_employees()
    if emp_id in cached:
        return cached[emp_id]
    return get_employee_by_id(emp_id)

def delete_employee(emp_id):
    success = delete_employee_db(emp_id)
    if success:
        invalidate_cache()
        return True, "Employee deleted"
    return False, "Employee not found"

def rename_employee(emp_id, new_name):
    success = rename_employee_db(emp_id, new_name)
    if success:
        invalidate_cache()
        return True, "Employee renamed"
    return False, "Employee not found"

def calculate_work_hours(in_timestamp, out_timestamp):
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t1 = datetime.strptime(in_timestamp[:19], fmt)
        t2 = datetime.strptime(out_timestamp[:19], fmt)
    except Exception:
        t1 = datetime.fromisoformat(in_timestamp[:19])
        t2 = datetime.fromisoformat(out_timestamp[:19])
    delta = t2 - t1
    total = int(delta.total_seconds())
    return total // 3600, (total % 3600) // 60, total % 60