import psycopg2
from psycopg2 import pool
import numpy as np
from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", 5432)
}

_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    sslmode="require",
    **DB_CONFIG
)

def get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=10, **DB_CONFIG)
    return _pool

def get_conn():
    return get_pool().getconn()

def release_conn(conn):
    get_pool().putconn(conn)

# ── Init tables ───────────────────────────────────────────────────────
def init_db():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id VARCHAR(20) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                department VARCHAR(100) DEFAULT 'General',
                embedding vector(512),
                registered_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                employee_id VARCHAR(20) REFERENCES employees(id)
                    ON DELETE CASCADE,
                employee_name VARCHAR(100),
                department VARCHAR(100),
                event VARCHAR(10) CHECK (event IN ('IN', 'OUT')),
                date DATE NOT NULL,
                time TIME NOT NULL,
                timestamp TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS camera_config (
                id SERIAL PRIMARY KEY,
                rtsp_url TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS employees_embedding_idx
            ON employees
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10);
        """)
        conn.commit()
        cur.close()
        print("✅ Database initialized")
    except Exception as e:
        conn.rollback()
        print(f"DB init error: {e}")
    finally:
        release_conn(conn)

# ── Employee functions ────────────────────────────────────────────────
def save_employee(emp_id, name, department, embedding):
    conn = get_conn()
    try:
        cur = conn.cursor()
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        cur.execute("""
            INSERT INTO employees (id, name, department, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE
            SET name = EXCLUDED.name,
                department = EXCLUDED.department,
                embedding = EXCLUDED.embedding
        """, (emp_id, name, department, embedding_str))
        conn.commit()
        cur.close()
    finally:
        release_conn(conn)

def get_all_employees():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, department, registered_at
            FROM employees ORDER BY registered_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        employees = {}
        for row in rows:
            emp_id, name, dept, reg_at = row
            employees[emp_id] = {
                "id": emp_id,
                "name": name,
                "department": dept,
                "registered_at": str(reg_at)
            }
        return employees
    finally:
        release_conn(conn)

def get_employee_by_id(emp_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, department, registered_at
            FROM employees WHERE id = %s
        """, (emp_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        emp_id, name, dept, reg_at = row
        return {
            "id": emp_id,
            "name": name,
            "department": dept,
            "registered_at": str(reg_at)
        }
    finally:
        release_conn(conn)

def delete_employee_db(emp_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM employees WHERE id = %s", (emp_id,))
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        return deleted > 0
    finally:
        release_conn(conn)

def count_employees():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM employees")
        count = cur.fetchone()[0]
        cur.close()
        return count
    finally:
        release_conn(conn)

def find_similar_face(embedding, threshold=0.4):
    conn = get_conn()
    try:
        cur = conn.cursor()
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        cur.execute("SET ivfflat.probes = 10;")
        cur.execute("""
            SELECT id, name, department,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM employees
            ORDER BY embedding <=> %s::vector
            LIMIT 1
        """, (embedding_str, embedding_str))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None, 0.0
        emp_id, name, dept, similarity = row
        if float(similarity) < threshold:
            return None, float(similarity)
        return {"id": emp_id, "name": name, "department": dept}, float(similarity)
    finally:
        release_conn(conn)

def rename_employee_db(emp_id, new_name):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE employees SET name = %s WHERE id = %s",
                    (new_name, emp_id))
        updated = cur.rowcount
        conn.commit()
        cur.close()
        return updated > 0
    finally:
        release_conn(conn)

# ── Attendance functions ──────────────────────────────────────────────
def save_attendance_log(employee_id, employee_name, department, event):
    conn = get_conn()
    try:
        cur = conn.cursor()
        now = datetime.now()
        cur.execute("""
            INSERT INTO attendance
            (employee_id, employee_name, department, event, date, time, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (employee_id, employee_name, department, event,
              now.date(), now.time(), now))
        conn.commit()
        cur.close()
    finally:
        release_conn(conn)

def get_attendance_logs(date=None, employee_id=None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        query = """
            SELECT id, employee_id, employee_name, department,
                   event, date, time, timestamp
            FROM attendance WHERE 1=1
        """
        params = []
        if date:
            query += " AND date = %s"
            params.append(date)
        if employee_id:
            query += " AND employee_id = %s"
            params.append(employee_id)
        query += " ORDER BY timestamp DESC"
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        logs = []
        for row in rows:
            log_id, emp_id, emp_name, dept, event, date, time, ts = row
            logs.append({
                "id": log_id,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "department": dept,
                "event": event,
                "date": str(date),
                "time": str(time),
                "timestamp": str(ts)
            })
        return logs
    finally:
        release_conn(conn)

def get_today_logs():
    today = datetime.now().date()
    return get_attendance_logs(date=today)

def get_visit_count_today(employee_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        today = datetime.now().date()
        cur.execute("""
            SELECT event, COUNT(*) FROM attendance
            WHERE employee_id = %s AND date = %s
            GROUP BY event
        """, (employee_id, today))
        rows = cur.fetchall()
        cur.close()
        ins = outs = 0
        for event, count in rows:
            if event == 'IN':
                ins = count
            elif event == 'OUT':
                outs = count
        return ins, outs
    finally:
        release_conn(conn)

# ── Camera config ─────────────────────────────────────────────────────
def save_camera_config(rtsp_url):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM camera_config")
        cur.execute("INSERT INTO camera_config (rtsp_url) VALUES (%s)",
                    (rtsp_url,))
        conn.commit()
        cur.close()
    finally:
        release_conn(conn)

def get_camera_config_db():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT rtsp_url FROM camera_config LIMIT 1")
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None
    finally:
        release_conn(conn)