import os
import json
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
}
DB_NAME = os.getenv("DB_NAME", "neuroscan_ai")


def get_server_connection():
    return mysql.connector.connect(**DB_CONFIG)


def _get_db_connection():
    return mysql.connector.connect(database=DB_NAME, **DB_CONFIG)


def initialize_database():
    """Create tables if needed and safely add prediction columns to old scan_records."""
    conn = get_server_connection()
    cur = conn.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cur.close()
    conn.close()

    conn = _get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(120) NOT NULL,
            age INT NULL,
            sex VARCHAR(30) NULL,
            contact VARCHAR(40) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_records (
            scan_id INT AUTO_INCREMENT PRIMARY KEY,
            patient_id INT NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            scan_type VARCHAR(30) DEFAULT 'MRI',
            ai_status VARCHAR(80) DEFAULT 'Pending model validation',
            notes TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_scan_patient
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                ON DELETE CASCADE
        )
    """)


    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(120) NOT NULL,
            email VARCHAR(190) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role ENUM(
                'admin',
                'researcher',
                'radiologist',
                'guest'
            ) NOT NULL DEFAULT 'researcher',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


    # Non-destructive migration: add columns only if they are absent.
    cur.execute("SHOW COLUMNS FROM scan_records")
    existing = {row[0] for row in cur.fetchall()}
    migrations = [
        ("predicted_class", "VARCHAR(80) NULL"),
        ("top_score", "DECIMAL(6,2) NULL"),
        ("class_scores_json", "JSON NULL"),
    ]
    for column, definition in migrations:
        if column not in existing:
            cur.execute(f"ALTER TABLE scan_records ADD COLUMN {column} {definition}")

    conn.commit()
    cur.close()
    conn.close()


def add_patient(full_name, age, sex, contact):
    conn = _get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO patients (full_name, age, sex, contact) VALUES (%s, %s, %s, %s)",
            (full_name, age, sex, contact)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        cur.close()
        conn.close()


def get_patients():
    conn = _get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT patient_id, full_name, age, sex, contact, created_at
            FROM patients ORDER BY patient_id DESC
        """)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def add_scan_record(patient_id, file_name, notes="", prediction=None):
    """
    Save the MRI upload and optional model output.
    Existing calls without prediction remain compatible.
    """
    prediction = prediction or {}
    predicted_class = prediction.get("predicted_class")
    top_score = prediction.get("top_score")
    scores = prediction.get("class_scores")
    ai_status = (
        "Educational model output saved"
        if predicted_class is not None
        else "Image uploaded - prediction not run"
    )

    conn = _get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO scan_records
                (patient_id, file_name, scan_type, ai_status, notes,
                 predicted_class, top_score, class_scores_json)
            VALUES (%s, %s, 'MRI', %s, %s, %s, %s, %s)
        """, (
            patient_id,
            file_name,
            ai_status,
            notes,
            predicted_class,
            top_score,
            json.dumps(scores) if scores is not None else None
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        cur.close()
        conn.close()


def get_scan_records():
    conn = _get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT s.scan_id, p.full_name, s.file_name, s.scan_type,
                   s.ai_status, s.predicted_class, s.top_score,
                   s.class_scores_json, s.notes, s.created_at
            FROM scan_records s
            JOIN patients p ON p.patient_id = s.patient_id
            ORDER BY s.scan_id DESC
        """)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def create_user(full_name, email, password_hash):
    conn = _get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO users
                (full_name, email, password_hash, role)
            VALUES (%s, %s, %s, 'researcher')
        """, (full_name, email.lower().strip(), password_hash))

        conn.commit()
        return cur.lastrowid

    finally:
        cur.close()
        conn.close()


def get_user_by_email(email):
    conn = _get_db_connection()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute("""
            SELECT user_id, full_name, email,
                   password_hash, role, is_active
            FROM users
            WHERE email = %s
            LIMIT 1
        """, (email.lower().strip(),))

        return cur.fetchone()

    finally:
        cur.close()
        conn.close()
