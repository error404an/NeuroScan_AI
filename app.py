from database import (
    initialize_database,
    add_patient,
    get_patients,
    add_scan_record,
    get_scan_records
)
import os
import json
import numpy as np
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io

st.set_page_config(
    page_title="NeuroScan AI | Brain Health Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

 

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Model artifacts created by scripts/train_model.py
MODEL_PATH = APP_DIR / "models" / "brain_mri_model.keras"
CLASS_NAMES_PATH = APP_DIR / "models" / "class_names.json"


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the saved model once per Streamlit session cache."""
    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        return None, [], "missing"

    try:
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
            class_names = json.load(file)

        if not isinstance(class_names, list) or not class_names:
            return None, [], "invalid_class_names"

        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, class_names, "loaded"
    except Exception as exc:
        return None, [], f"error: {exc}"


trained_model, model_class_names, model_status = load_trained_model()

def predict_mri_image(image):
    if trained_model is None:
        raise RuntimeError("CNN model is not loaded.")

    image = image.convert("RGB")
    image = image.resize((160, 160))

    image_array = np.asarray(
        image, dtype=np.float32
    )

    image_array = np.expand_dims(image_array, axis=0)

    predictions = trained_model.predict(
        image_array, verbose=0
    )[0]

    if len(predictions) != len(model_class_names):
        raise ValueError("Model output and class labels do not match.")

    results = {
        name: float(score)
        for name, score in zip(model_class_names, predictions)
    }

    predicted_class = model_class_names[
        int(np.argmax(predictions))
    ]

    return {
        "predicted_class": predicted_class,
        "top_score": float(np.max(predictions)) * 100,
        "class_scores": results
    }


# ==================================
# PDF MRI REPORT GENERATION
# ==================================

def generate_pdf_report(patient_name, prediction, image):

    buffer = io.BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle("NeuroScan AI MRI Report")

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 55, "NeuroScan AI")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, height - 80, "Brain MRI Analysis Report")

    pdf.line(50, height - 95, width - 50, height - 95)

    y = height - 130

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Patient Information")

    y -= 25
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Patient: {patient_name}")

    y -= 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "AI Analysis Result")

    y -= 25
    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        50, y,
        f"Predicted Class: {prediction['predicted_class']}"
    )

    y -= 20
    pdf.drawString(
        50, y,
        f"Model Confidence: {prediction['top_score']:.2f}%"
    )

    y -= 40
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Class-wise Scores")

    y -= 25
    pdf.setFont("Helvetica", 10)

    for name, score in prediction["class_scores"].items():
        pdf.drawString(60, y, f"{name}: {score * 100:.2f}%")
        y -= 20

    y -= 20
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(50, y, "Uploaded MRI Image")

    pdf.drawImage(
        ImageReader(image.convert("RGB")),
        50, 100,
        width=220,
        height=180,
        preserveAspectRatio=True,
        anchor="c"
    )

    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        50, 55,
        "Educational AI output only medical diagnosis."
    )

    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# ==================================
# EXISTING APP DESIGN STARTS HERE
# ==================================

st.markdown("""

<style>
    .stApp { background: #081824; color: #EAF4F7; }
    [data-testid="stSidebar"] { background: #102A38; }
    .hero {
        padding: 25px 28px; border-radius: 18px;
        background: linear-gradient(120deg,#123D50,#087E83);
        border: 1px solid #286979; margin-bottom: 18px;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.25rem; }
    .hero p { color: #D8F1F1; margin: 8px 0 0 0; }
    .info-card {
        background: #102A38; border: 1px solid #254858;
        border-radius: 14px; padding: 17px; min-height: 120px;
    }
    .small-muted { color: #A9C2CB; font-size: .9rem; }
    h1,h2,h3 { color: #EAF4F7; }
    div[data-testid="stMetric"] {
        background: #102A38; padding: 14px; border-radius: 12px;
        border: 1px solid #254858;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🧠 NeuroScan AI")
st.sidebar.caption("Brain Anatomy • MRI Workflow • Health Education")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Brain Anatomy", "Disease Library", "Symptoms Guide",
     "MRI Upload", "Patient Records", "About & Safety"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Educational prototype • Not a medical device")

# Database connection check
db_ready = False
db_error = ""
try:
    initialize_database()
    db_ready = True
except Exception as exc:
    db_error = str(exc)

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

if page == "Dashboard":
    hero("NeuroScan AI", "Intelligent brain-health education and MRI workflow prototype")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(get_patients()) if db_ready else 0)

    c2.metric("Total MRI Scans", len(get_scan_records()) if db_ready else 0)

    c3.metric(
    "CNN Model",
    "Loaded" if model_status == "loaded" else "Not Ready"
)

    c4.metric(
    "Database",
    "Connected" if db_ready else "Not Connected"
)
    st.subheader("Welcome")
    st.write(
        "Explore brain structures, learn about neurological conditions, "
        "review general symptom information, and store MRI upload records."
    )
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="info-card"><h3>🧠 Brain Anatomy</h3>'
                    '<p class="small-muted">Learn the major brain regions and their functions.</p></div>',
                    unsafe_allow_html=True)
    with right:
        mri_card_text = ("AI prediction is connected. Upload an MRI image to run the CNN model."
            
            if model_status == "loaded" else
            "Upload an image slice and save its record. The trained model files are not currently ready to load."
        )
        st.markdown(
            '<div class="info-card"><h3>🩻 MRI Workflow</h3>'
            f'<p class="small-muted">{mri_card_text}</p></div>',
            unsafe_allow_html=True
        )

    if model_status == "loaded":
        st.success(
            "Saved CNN model loaded successfully. "
            "AI prediction is connected and ready for MRI testing."
        )
        st.caption("Model classes: " + ", ".join(str(name) for name in model_class_names))
    elif model_status == "missing":
        st.warning(
            "CNN model files were not found. Expected: models/brain_mri_model.keras "
            "and models/class_names.json."
        )
    else:
        st.error("CNN model files were found but could not be loaded. Check the model files and TensorFlow error below.")
        st.code(model_status)

    st.warning(
        "programmed by YASHRAJ SAHANI."
    )
    if not db_ready:
        st.error("MySQL connection failed. Check your .env settings and make sure MySQL Server is running.")
        st.code(db_error)

elif page == "Brain Anatomy":
    hero("Brain Anatomy", "Major structures and their general functions")
    anatomy = [
        ("Frontal lobe", "Planning, decision-making, voluntary movement, behaviour and speech production."),
        ("Parietal lobe", "Processes touch, body position and other sensory information."),
        ("Temporal lobe", "Important for hearing, language comprehension and memory."),
        ("Occipital lobe", "Mainly processes visual information."),
        ("Cerebellum", "Helps coordinate movement, balance and motor learning."),
        ("Brainstem", "Connects the brain and spinal cord and supports vital functions such as breathing and alertness."),
        ("Hippocampus", "Plays an important role in forming new memories."),
        ("Thalamus", "Relays many sensory and motor signals to other brain regions.")
    ]
    for name, desc in anatomy:
        with st.expander(name):
            st.write(desc)
    st.caption("Educational overview. Brain functions are distributed across connected networks; this is a simplified guide.")

elif page == "Disease Library":
    hero("Neurological Disease Library", "General educational information — not an individual diagnosis")
    diseases = {
        "Brain tumour": {
            "overview": "A brain tumour is an abnormal growth in or near the brain. Tumours differ in type, location and behaviour.",
            "symptoms": "Possible symptoms include new or worsening headaches, seizures, vomiting, weakness, or vision/speech changes. These symptoms can have many other causes.",
            "care": "Care depends on tumour type and may include monitoring, surgery, radiation therapy, medicines or other specialist-directed treatment.",
            "precautions": "Seek medical assessment for persistent or progressive neurological symptoms. There is no single precaution that prevents all brain tumours."
        },
        "Stroke": {
            "overview": "A stroke occurs when blood flow to part of the brain is interrupted or when bleeding occurs in the brain.",
            "symptoms": "Sudden face drooping, arm weakness, speech difficulty, vision loss, severe imbalance or sudden severe headache can be warning signs.",
            "care": "Stroke is an emergency. Hospital teams determine whether clot-busting medicines, procedures, bleeding management and rehabilitation are appropriate.",
            "precautions": "Control blood pressure, diabetes and cholesterol with a clinician; avoid tobacco; follow prescribed medicines."
        },
        "Alzheimer's disease": {
            "overview": "A progressive brain disease that affects memory and other thinking abilities.",
            "symptoms": "Increasing memory difficulty, confusion, problems with familiar tasks and changes in reasoning may occur.",
            "care": "A clinician evaluates causes of memory changes. Management may include selected medicines, support, safety planning and caregiver resources.",
            "precautions": "Manage cardiovascular risks, stay physically and socially active, and seek assessment for persistent memory changes."
        },
        "Epilepsy": {
            "overview": "A neurological condition involving a tendency to have recurrent unprovoked seizures.",
            "symptoms": "Seizures can involve shaking, staring, altered awareness, unusual sensations or temporary confusion.",
            "care": "Assessment may include history, EEG and imaging. Anti-seizure medicines and other specialist treatments may be considered.",
            "precautions": "Take prescribed medicine as directed, maintain sleep routines and discuss personal seizure-safety advice with a clinician."
        },
        "Parkinson's disease": {
            "overview": "A progressive neurological condition affecting movement and sometimes sleep, mood and other functions.",
            "symptoms": "Slowness of movement, stiffness, tremor at rest and balance changes can occur.",
            "care": "Treatment may include medication, physiotherapy, exercise and specialist follow-up.",
            "precautions": "Regular safe exercise, fall-risk reduction and medication review can support wellbeing."
        },
        "Multiple sclerosis": {
            "overview": "An immune-mediated condition affecting the central nervous system, including the brain and spinal cord.",
            "symptoms": "Symptoms vary and may include vision problems, numbness, weakness, balance difficulty or fatigue.",
            "care": "Neurologists may use disease-modifying medicines, treatment for relapses and rehabilitation.",
            "precautions": "Follow a neurologist's care plan and seek advice about new or worsening neurological symptoms."
        }
    }
    choice = st.selectbox("Select a condition", list(diseases.keys()))
    item = diseases[choice]
    st.subheader(choice)
    st.write("**Overview:**", item["overview"])
    st.write("**Possible symptoms:**", item["symptoms"])
    st.write("**Treatment / care information:**", item["care"])
    st.write("**Precautions / risk reduction:**", item["precautions"])
    st.info("Treatment depends on the individual and must be decided by a qualified healthcare professional.")

elif page == "Symptoms Guide":
    hero("Symptoms Guide", "General information to help people understand symptoms")
    symptom = st.selectbox("Choose a symptom",
        ["Headache", "Memory problems", "Seizure", "Weakness or numbness",
         "Vision changes", "Dizziness or balance problems", "Speech difficulty"])
    guidance = {
        "Headache": "Headaches have many causes. A sudden, extremely severe headache or headache with neurological symptoms needs urgent assessment.",
        "Memory problems": "Memory changes can have many causes, including sleep, medication, mood and neurological conditions. Arrange clinical assessment if persistent or worsening.",
        "Seizure": "A first seizure requires urgent medical assessment. If a seizure lasts 5 minutes or more, repeats without recovery, or breathing is difficult, seek emergency help.",
        "Weakness or numbness": "Sudden one-sided weakness or numbness may be a stroke warning sign. Call emergency services immediately.",
        "Vision changes": "Sudden loss of vision or vision changes with weakness, speech difficulty or severe headache needs emergency assessment.",
        "Dizziness or balance problems": "Causes range from inner-ear problems to neurological conditions. Sudden severe imbalance with other stroke symptoms is an emergency.",
        "Speech difficulty": "Sudden trouble speaking or understanding speech can be a stroke warning sign. Seek emergency help immediately."
    }
    st.info(guidance[symptom])
    st.warning("A symptom alone cannot identify a brain disease. Do not use this page to self-diagnose.")
    
elif page == "MRI Upload":
    hero(
        "MRI Image Upload",
        "Upload a brain MRI image for educational AI research"
    )

    if not db_ready:
        st.error("MySQL database is not connected.")

    else:
        patients = get_patients()

        if not patients:
            st.warning(
                "Please add a patient in Patient Records first."
            )

        else:
            patient_options = {
                f'{p["patient_id"]} — {p["full_name"]}':
                p["patient_id"]
                for p in patients
            }

            selected_patient = st.selectbox(
                "Select Patient",
                list(patient_options.keys())
            )

            uploaded = st.file_uploader(
                "Upload Brain MRI Image",
                type=["png", "jpg", "jpeg"]
            )

            notes = st.text_area(
                "Additional Notes (Optional)"
            )

            if uploaded:
                try:
                    image = Image.open(uploaded).convert("RGB")

                    st.image(
                        image,
                        caption="Uploaded MRI Image",
                        use_container_width=True
                    )

                    st.info(
                        "Image uploaded. Click below to run "
                        "the educational CNN model."
                    )

                    if st.button(
                        "Run AI Analysis",
                        type="primary"
                    ):

                        if model_status != "loaded":
                            st.error(
                                "CNN model is not loaded. "
                                "Please check model files."
                            )

                        else:
                            with st.spinner(
                                "Running CNN model..."
                            ):
                                prediction = predict_mri_image(
                                    image
                                )

                            safe_name = Path(uploaded.name).name

                            save_path = UPLOAD_DIR / safe_name

                            if save_path.exists():
                                save_path = UPLOAD_DIR / (
                                    f"{save_path.stem}_"
                                    f"{os.urandom(3).hex()}"
                                    f"{save_path.suffix}"
                                )

                            image.save(save_path)

                            scan_id = add_scan_record(
                                patient_options[selected_patient],
                                save_path.name,
                                notes.strip(),
                                prediction=prediction
                            )

                            st.success(
                                "AI analysis completed. "
                                f"Record ID: {scan_id}"
                            )

                            st.subheader("CNN Model Output")

                            st.write(
                                "Predicted dataset class:",
                                prediction["predicted_class"]
                            )

                            st.metric(
                                "Model confidence score",
                                f'{prediction["top_score"]:.2f}%'
                            )

                            st.subheader("Class-wise Scores")

                            for name, score in prediction[
                                "class_scores"
                            ].items():

                                st.write(
                                    f"{name}: {score * 100:.2f}%"
                                )

                                st.progress(
                                    max(0.0, min(1.0, score))
                                )

                            pdf_file = generate_pdf_report(
                            selected_patient,
                           prediction,
                           image
                   )

                        st.download_button(
                            label="📄 Download MRI Report (PDF)",
                            data=pdf_file,
                            file_name="NeuroScan_MRI_Report.pdf",
                            mime="application/pdf"
                        )
                            
        
    
                except (UnidentifiedImageError, OSError):
                    st.error(
                        "Invalid image. Please upload a valid MRI image."
                    )

    st.caption(
        "This prototype supports 2D PNG and JPG images only. "
        "DICOM and 3D MRI support are not implemented."
    )

elif page == "Patient Records":
    hero("Patient Records", "Create and view local project records")
    if not db_ready:
        st.error("MySQL is not connected. Check your .env settings.")
        st.code(db_error)
    else:
        with st.form("patient_form", clear_on_submit=True):
            st.subheader("Add patient")
            name = st.text_input("Full name *")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=0)
                sex = st.selectbox("Sex", ["Prefer not to say", "Female", "Male", "Other"])
            with col2:
                contact = st.text_input("Contact (optional)")
            submitted = st.form_submit_button("Save patient")
            if submitted:
                if not name.strip():
                    st.error("Please enter the patient's name.")
                else:
                    try:
                        patient_id = add_patient(
                            name.strip(),
                            int(age) if age else None,
                            sex,
                            contact.strip()
                        )
                        st.success(f"Patient saved. Patient ID: {patient_id}")
                    except Exception as exc:
                        st.error(f"Could not save patient: {exc}")

        st.subheader("Registered patients")
        rows = get_patients()
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No patient records yet.")

        st.subheader("MRI upload history")
        scans = get_scan_records()
        if scans:
            st.dataframe(pd.DataFrame(scans), use_container_width=True, hide_index=True)
        else:
            st.info("No MRI upload records yet.")

elif page == "About & Safety":
    hero("About NeuroScan AI", "A learning project for brain anatomy and MRI workflow")
    st.markdown("""
    **Current capabilities**
    - Brain anatomy learning pages
    - General neurological disease and symptom information
    - MySQL patient and MRI upload record management
    - Local upload of PNG/JPG MRI image slices
    - Saved CNN model file detection and load check
    - MRI prediction display and PDF report generation: connected
    -Saved CNN model: """ + ("loaded successfully" if model_status == "loaded" else "not currently loadable") + """
    
    **Development status**
    - DICOM / NIfTI medical imaging pipeline: not implemented
    - Clinical validation, calibration or radiologist workflow: not implemented
    - Production-grade authentication, encryption, access control and audit logging: not implemented
    """)

    st.markdown("**EMERGENCY:** Sudden facial droop, arm weakness, speech difficulty, "
                "new seizure, loss of consciousness, or sudden severe headache requires "
                "urgent medical attention. In India, call 112 for emergency assistance.")
