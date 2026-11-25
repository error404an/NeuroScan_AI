NEUROSCAN AI - BEGINNER STARTER PROJECT
========================================

REQUIREMENTS
------------
1. Python 3.10 or 3.11 recommended
2. Visual Studio Code
3. MySQL Server 8.x (MySQL Workbench is optional)
4. Internet connection for installing Python packages

SETUP
-----
1. Extract this ZIP file.
2. Open the NeuroScan_AI folder in VS Code.
3. Open VS Code Terminal: Terminal > New Terminal.
4. Create a virtual environment:

   python -m venv .venv

5. Activate it on Windows PowerShell:

   .venv\Scripts\Activate.ps1

   If PowerShell blocks activation, use Command Prompt:

   .venv\Scripts\activate.bat

6. Install packages:

   python -m pip install --upgrade pip
   pip install -r requirements.txt

MYSQL SETUP
-----------
1. Install and start MySQL Server.
2. In this project folder, copy .env.example and rename the copy to .env.
3. Edit .env and enter your MySQL password.
4. Example:
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_actual_password
   DB_NAME=neuroscan_ai

The app creates the database and tables automatically on first run.
Do not share your .env file or password.

RUN
---
In the VS Code terminal:

   streamlit run app.py

A browser tab should open. If it does not, open the Local URL shown in terminal,
usually http://localhost:8501.

IMPORTANT PROJECT LIMITATIONS
-----------------------------
This starter application does NOT contain a trained AI model and does NOT diagnose
MRI scans. The MRI page currently accepts PNG/JPG image slices and stores a record.
A later phase will add a properly sourced dataset, model training, evaluation,
and a clearly limited research/demo inference workflow.

Do not upload real patient-identifiable data without appropriate consent,
security, and privacy controls. Never rely on this prototype for medical decisions.
