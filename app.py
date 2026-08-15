import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import re
from datetime import datetime
from fpdf import FPDF
import tempfile
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CardioCare AI | Clinical Decision System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KUSTOM TEMA KESEHATAN PREMIUM ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #F8FAFC; }
    
    /* Hero Banner Premium */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #0B5A5C 100%);
        padding: 35px 40px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(11, 90, 92, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: ''; position: absolute; top: -50%; right: -10%; width: 50%; height: 200%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%);
        transform: rotate(30deg);
    }
    .hero-title { font-size: 38px; font-weight: 800; margin-bottom: 5px; letter-spacing: -0.5px; }
    .hero-title span { color: #2DD4BF; }
    .hero-subtitle { font-size: 16px; font-weight: 400; color: #CBD5E1; margin: 0; }
    
    /* Card Styling */
    .card { background-color: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03); margin-bottom: 20px; border-top: 4px solid #0B5A5C; }
    
    /* Custom Action Button */
    .btn-predict>button { background: linear-gradient(135deg, #0B5A5C 0%, #0d9488 100%); color: white; border-radius: 10px; border: none; width: 100%; font-size: 18px; font-weight: 700; padding: 14px; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3); }
    .btn-predict>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(13, 148, 136, 0.4); color: white;}
    
    /* Sidebar Simulasi Buttons */
    .sim-btn-low>button, .sim-btn-med>button, .sim-btn-high>button {
        white-space: nowrap !important; 
        font-size: 14px !important; 
        padding: 8px 2px !important; 
        font-weight: 700 !important; 
        width: 100%;
        border-radius: 8px;
    }
    .sim-btn-low>button { background-color: #F0FDF4; color: #15803D; border: 1px solid #86EFAC; }
    .sim-btn-med>button { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE047; }
    .sim-btn-high>button { background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FCA5A5; }
    
    /* Risk Labels */
    .risk-badge { padding: 10px 20px; border-radius: 50px; font-weight: 800; font-size: 22px; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .high-risk { background-color: #FEF2F2; color: #DC2626; border: 2px solid #FECACA; }
    .med-risk { background-color: #FFFBEB; color: #D97706; border: 2px solid #FDE047; }
    .low-risk { background-color: #F0FDF4; color: #16A34A; border: 2px solid #BBF7D0; }
    
    /* Clinical Note Box (REVISI: Judul Diperbarui) */
    .clinical-note-box { 
        background-color: #F8FAFC; 
        border-left: 6px solid #0d9488; 
        padding: 20px 25px; 
        border-radius: 8px; 
        font-size: 16px; 
        line-height: 1.6;
        color: #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .clinical-note-title {
        font-weight: 700; color: #0f172a; font-size: 18px; margin-bottom: 12px;
        display: flex; align-items: center; gap: 8px;
    }
    
    /* System Status Footer */
    .sys-status { font-size: 12px; color: #64748B; text-align: center; margin-top: 30px; border-top: 1px solid #E2E8F0; padding-top: 15px;}
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('lgbm_model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, encoders, feature_names

try:
    model, encoders, feature_names = load_assets()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file model (.pkl) telah di-generate dan berada di folder yang sama.")
    st.stop()

# --- KAMUS TRANSLASI (NLG) ---
translate = {
    'Length_of_Stay': 'Lama Rawat Inap', 'Troponin': 'Kadar Troponin', 
    'Diastolic_BP': 'Tekanan Darah Diastolik', 'Systolic_BP': 'Tekanan Darah Sistolik', 
    'Ejection_Fraction': 'Kekuatan Pompa Jantung (EF)', 'Heart_Rate': 'Detak Jantung', 
    'Oxygen_Saturation': 'Saturasi Oksigen', 'Medication_Adherence': 'Disiplin Minum Obat', 
    'Physical_Activity': 'Aktivitas Fisik', 'ICU_Admission': 'Riwayat Masuk ICU',
    'Creatinine': 'Fungsi Ginjal (Kreatinin)', 'Cholesterol': 'Kadar Kolesterol',
    'Followup_Attendance': 'Kehadiran Kontrol Medis', 'Diet_Adherence': 'Kepatuhan Diet',
    'Number_of_Admissions': 'Total Kunjungan RS', 'Emergency_Admission': 'Riwayat Masuk IGD',
    'Age': 'Usia Pasien', 'Medication_Count': 'Jumlah Resep Obat'
}

# 📌 REVISI BARU: Format Value Cerdas (Memberikan makna pada skala)
def format_value(feat, val):
    try:
        val_float = float(val)
    except:
        return str(val)

    if feat in ['Diet_Adherence', 'Physical_Activity', 'Medication_Adherence', 'Followup_Attendance']:
        if val_float <= 3: return f"{int(val_float)}/10 (Sangat Buruk)"
        elif val_float <= 6: return f"{int(val_float)}/10 (Sedang)"
        else: return f"{int(val_float)}/10 (Sangat Baik)"
    elif feat == 'Smoking_Behavior':
        if val_float == 0: return f"0/10 (Bebas Asap)"
        elif val_float <= 4: return f"{int(val_float)}/10 (Perokok Ringan)"
        else: return f"{int(val_float)}/10 (Perokok Berat)"
    elif feat in ['Systolic_BP', 'Diastolic_BP']: return f"{val_float} mmHg"
    elif feat == 'Heart_Rate': return f"{int(val_float)} bpm"
    elif feat == 'Oxygen_Saturation': return f"{val_float}%"
    elif feat == 'Ejection_Fraction': return f"{val_float}%"
    elif feat == 'Length_of_Stay': return f"{int(val_float)} hari"
    elif feat == 'Creatinine': return f"{val_float} mg/dL"
    elif feat == 'Troponin': return f"{val_float} ng/mL"
    elif feat == 'Cholesterol': return f"{val_float} mg/dL"
    elif feat == 'Age': return f"{int(val_float)} Tahun"
    elif feat in ['Medication_Count', 'Number_of_Admissions', 'Previous_Hospitalization']: return f"{int(val_float)} kali/jenis"
    else: return str(val)

# --- GENERATOR PARAGRAF KLINIS ---
# 📌 REVISI BARU: Menggunakan <b> untuk BOLD di HTML, dan formatter cerdas
def generate_clinical_paragraph(predicted_label, increasers, decreasers):
    if predicted_label == 'High':
        status_text = "kondisi kardiovaskular pasien saat ini berada dalam zona <b>Risiko Tinggi</b> untuk mengalami komplikasi lanjutan dan berpotensi sangat besar membutuhkan rawat inap berulang."
    elif predicted_label == 'Medium':
        status_text = "pasien saat ini berada pada ambang batas <b>Risiko Sedang</b> dan membutuhkan observasi pencegahan berkelanjutan."
    else:
        status_text = "kondisi klinis kardiovaskular pasien tergolong <b>Sangat Stabil (Risiko Rendah)</b>."

    good_features_text = ""
    if decreasers:
        good_items = [f"<b>{translate.get(f[0], f[0])}</b> ({format_value(f[0], f[2])})" for f in decreasers[:3]]
        if len(good_items) == 1:
            good_features_text = f"Stabilitas ini sangat didukung oleh indikator {good_items[0]} yang merespon amat baik terhadap manajemen medis."
        elif len(good_items) > 1:
            good_features_text = f"Ketahanan tubuh pasien saat ini dipengaruhi secara signifikan oleh stabilitas pada {', '.join(good_items[:-1])}, serta {good_items[-1]}. Parameter positif ini secara aktif melindungi fungsi organ vital pasien."

    warning_features_text = ""
    if increasers:
        bad_items = [f"<b>{translate.get(f[0], f[0])}</b> ({format_value(f[0], f[2])})" for f in increasers[:3]]
        if predicted_label == 'High':
            warning_features_text = f" Namun, faktor esensial yang mendesak untuk segera diintervensi dokter adalah {', '.join(bad_items[:-1]) + ' dan ' if len(bad_items)>1 else ''}{bad_items[-1]}. Beban pada parameter medis ini memicu peringatan bahaya klinis tertinggi dari sistem."
        else:
            warning_features_text = f" Meskipun demikian, sistem peringatan dini mencatat bahwa riwayat {', '.join(bad_items[:-1]) + ' dan ' if len(bad_items)>1 else ''}{bad_items[-1]} harus dijadikan titik pantauan ketat (Follow-up) agar tidak memicu rehospitalisasi."

    paragraph = f"Berdasarkan hasil kalkulasi AI, {status_text} {good_features_text}{warning_features_text}"
    return paragraph

# --- DATA PRESET SIMULASI ---
presets = {
    'low': {'Age': 45, 'Gender': 'Male', 'Marital_Status': 'Married', 'Education': 'Bachelor', 'Occupation': 'Civil Servant', 'Smoking_Status': 'Non-Smoker', 'Smoking_Behavior': 0, 'Physical_Activity': 8, 'Diet_Adherence': 9, 'Systolic_BP': 115.0, 'Diastolic_BP': 75.0, 'Heart_Rate': 72.0, 'Respiratory_Rate': 16.0, 'Oxygen_Saturation': 99.0, 'BMI': 22.5, 'Ejection_Fraction': 65.0, 'Creatinine': 0.8, 'Troponin': 0.01, 'Cholesterol': 160.0, 'Diabetes': 'No', 'Hypertension': 'No', 'CKD': 'No', 'Length_of_Stay': 2, 'ICU_Admission': 0, 'Previous_Hospitalization': 0, 'Number_of_Admissions': 1, 'Emergency_Admission': 0, 'Medication_Count': 1, 'Medication_Adherence': 10, 'Followup_Attendance': 10},
    'medium': {'Age': 62, 'Gender': 'Female', 'Marital_Status': 'Married', 'Education': 'Secondary', 'Occupation': 'Private Sector', 'Smoking_Status': 'Smoker', 'Smoking_Behavior': 4, 'Physical_Activity': 4, 'Diet_Adherence': 5, 'Systolic_BP': 145.0, 'Diastolic_BP': 90.0, 'Heart_Rate': 85.0, 'Respiratory_Rate': 20.0, 'Oxygen_Saturation': 94.0, 'BMI': 27.5, 'Ejection_Fraction': 45.0, 'Creatinine': 1.4, 'Troponin': 0.12, 'Cholesterol': 220.0, 'Diabetes': 'Yes', 'Hypertension': 'Yes', 'CKD': 'No', 'Length_of_Stay': 6, 'ICU_Admission': 0, 'Previous_Hospitalization': 2, 'Number_of_Admissions': 3, 'Emergency_Admission': 1, 'Medication_Count': 5, 'Medication_Adherence': 6, 'Followup_Attendance': 6},
    'high': {'Age': 78, 'Gender': 'Male', 'Marital_Status': 'Widowed', 'Education': 'Primary', 'Occupation': 'Retired', 'Smoking_Status': 'Smoker', 'Smoking_Behavior': 9, 'Physical_Activity': 1, 'Diet_Adherence': 3, 'Systolic_BP': 175.0, 'Diastolic_BP': 105.0, 'Heart_Rate': 110.0, 'Respiratory_Rate': 26.0, 'Oxygen_Saturation': 88.0, 'BMI': 32.0, 'Ejection_Fraction': 30.0, 'Creatinine': 2.8, 'Troponin': 0.65, 'Cholesterol': 285.0, 'Diabetes': 'Yes', 'Hypertension': 'Yes', 'CKD': 'Yes', 'Length_of_Stay': 14, 'ICU_Admission': 1, 'Previous_Hospitalization': 5, 'Number_of_Admissions': 7, 'Emergency_Admission': 1, 'Medication_Count': 9, 'Medication_Adherence': 3, 'Followup_Attendance': 2}
}

if 'Age' not in st.session_state:
    for k, v in presets['low'].items():
        st.session_state[k] = v

def apply_preset(level):
    for k, v in presets[level].items():
        st.session_state[k] = v

# --- GENERATOR REPORT (PDF) ---
def generate_pdf(patient_name, predicted_label, confidence, clinical_paragraph, vitals):
    pdf = FPDF()
    pdf.add_page()
    
    # Design Premium Header PDF
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(11, 90, 92)
    pdf.cell(0, 12, txt="CardioCare AI", ln=True, align='L')
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, txt="Clinical Decision Support System - HES Framework", ln=True, align='L')
    pdf.line(10, 32, 200, 32)
    
    pdf.set_xy(10, 37)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, txt=f"Identitas Pasien   : {patient_name}", ln=True)
    pdf.cell(0, 6, txt=f"Tanggal Evaluasi : {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", ln=True)
    
    pdf.ln(6)
    pdf.set_font("Arial", 'B', 14)
    if predicted_label == 'High':
        pdf.set_text_color(220, 38, 38)
        risk_text = "RISIKO TINGGI (HIGH)"
    elif predicted_label == 'Medium':
        pdf.set_text_color(217, 119, 6)
        risk_text = "RISIKO SEDANG (MEDIUM)"
    else:
        pdf.set_text_color(22, 163, 74)
        risk_text = "RISIKO RENDAH (LOW)"
        
    pdf.cell(0, 8, txt=f"KESIMPULAN AI : {risk_text}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, txt=f"Tingkat Kepercayaan Prediksi (Confidence Level): {confidence:.1f}%", ln=True)
    
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(5)
    pdf.cell(0, 8, txt="Rekaman Tanda Vital Utama:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 6, txt=f"- Tekanan Darah  : {int(vitals['Systolic_BP'])}/{int(vitals['Diastolic_BP'])} mmHg", ln=True)
    pdf.cell(0, 6, txt=f"- Detak Jantung  : {int(vitals['Heart_Rate'])} bpm    |  Saturasi Oksigen: {int(vitals['Oxygen_Saturation'])}%", ln=True)
    pdf.cell(0, 6, txt=f"- Pompa (EF)       : {vitals['Ejection_Fraction']}%           |  Kadar Troponin  : {vitals['Troponin']} ng/mL", ln=True)
    
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, txt="Catatan Evaluasi Klinis (Narrative Summary):", ln=True)
    
    # Memasukkan Paragraf Klinis ke PDF
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(30, 41, 59)
    
    # 📌 REVISI BARU: Menghapus tag HTML <b> dan </b> agar bersih saat diprint ke PDF
    clean_paragraph = re.sub(r'<[^>]+>', '', clinical_paragraph)
    pdf.multi_cell(0, 6, txt=clean_paragraph)

    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 4, txt="*DISCLAIMER: Dokumen klinis elektronik ini dihasilkan secara otomatis oleh kecerdasan buatan (Explainable AI). Hasil ini ditujukan sebagai Clinical Decision Support System dan tidak mensubstitusi opini profesional Dokter Spesialis Kardiologi.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.unlink(tmp.name)
    return pdf_bytes

# --- SIDEBAR MENU ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=70) 
    st.markdown("### Control Panel")
    st.divider()
    
    st.markdown("### 🧪 Simulasi Skenario")
    st.caption("Auto-fill data pasien (Demo Mode)")
    
    col_low, col_med, col_high = st.columns(3)
    with col_low:
        st.markdown('<div class="sim-btn-low">', unsafe_allow_html=True)
        st.button("🟢 Sehat", on_click=apply_preset, args=('low',))
        st.markdown('</div>', unsafe_allow_html=True)
    with col_med:
        st.markdown('<div class="sim-btn-med">', unsafe_allow_html=True)
        st.button("🟡 Sedang", on_click=apply_preset, args=('medium',))
        st.markdown('</div>', unsafe_allow_html=True)
    with col_high:
        st.markdown('<div class="sim-btn-high">', unsafe_allow_html=True)
        st.button("🔴 Kritis", on_click=apply_preset, args=('high',))
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="sys-status">🟢 System Online | Model: HES-LightGBM v2.1<br>Accuracy Validation: 98.00%</div>', unsafe_allow_html=True)

# --- HEADER APLIKASI UTAMA (HERO BANNER) ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">CardioCare <span>AI</span></div>
    <div class="hero-subtitle">Platform pendukung keputusan medis tingkat lanjut (CDSS). Memanfaatkan arsitektur <i>Hybrid Entropy Stacking</i> untuk memprediksi risiko rehospitalisasi penyakit kardiovaskular secara presisi.</div>
</div>
""", unsafe_allow_html=True)

patient_name = st.text_input("Identitas Pasien / No. Rekam Medis (Digunakan untuk header cetak PDF)", placeholder="Masukkan nama atau ID rekam medis pasien di sini...", max_chars=50)

st.markdown('<div class="card">', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["👤 Demografi & Gaya Hidup", "🩺 Tanda Vital & Laboratorium", "🏥 Riwayat Perawatan & Farmasi"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Umur (Tahun)", 1, 120, key='Age')
        st.selectbox("Jenis Kelamin", ["Male", "Female"], key='Gender')
        st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"], key='Marital_Status')
        st.selectbox("Pendidikan", ["Primary", "Secondary", "Bachelor", "Master", "PhD"], key='Education')
    with c2:
        st.selectbox("Pekerjaan", ["Civil Servant", "Private Sector", "Entrepreneur", "Farmer", "Teacher", "Retired", "Other"], key='Occupation')
        st.selectbox("Status Merokok", ["Non-Smoker", "Smoker"], key='Smoking_Status')
        st.number_input("Tingkat Kecanduan Rokok (0-10)", 0, 10, key='Smoking_Behavior')
        st.slider("Aktivitas Fisik (0=Malas, 10=Aktif)", 0, 10, key='Physical_Activity')
        st.slider("Kepatuhan Diet (0=Buruk, 10=Disiplin)", 0, 10, key='Diet_Adherence')

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        st.number_input("Tekanan Sistolik (mmHg)", 50.0, 250.0, key='Systolic_BP')
        st.number_input("Tekanan Diastolik (mmHg)", 30.0, 150.0, key='Diastolic_BP')
        st.number_input("Detak Jantung (bpm)", 30.0, 200.0, key='Heart_Rate')
        st.number_input("Laju Pernapasan (x/menit)", 10.0, 40.0, key='Respiratory_Rate')
        st.number_input("Saturasi Oksigen (SpO2 %)", 50.0, 100.0, key='Oxygen_Saturation')
        st.number_input("Indeks Massa Tubuh (BMI)", 10.0, 60.0, key='BMI')
        st.number_input("Ejection Fraction (%)", 10.0, 80.0, key='Ejection_Fraction')
    with c4:
        st.number_input("Kreatinin Darah (mg/dL)", 0.1, 15.0, key='Creatinine')
        st.number_input("Troponin (ng/mL)", 0.0, 10.0, key='Troponin')
        st.number_input("Kolesterol Total (mg/dL)", 50.0, 400.0, key='Cholesterol')
        st.selectbox("Komorbiditas: Diabetes", ["No", "Yes"], key='Diabetes')
        st.selectbox("Komorbiditas: Hipertensi", ["No", "Yes"], key='Hypertension')
        st.selectbox("Komorbiditas: Penyakit Ginjal (CKD)", ["No", "Yes"], key='CKD')

with tab3:
    c5, c6 = st.columns(2)
    with c5:
        st.number_input("Lama Rawat Inap Terakhir (Hari)", 1, 100, key='Length_of_Stay')
        st.selectbox("Sempat Masuk Ruang ICU?", [0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", key='ICU_Admission')
        st.number_input("Jumlah Riwayat Opname Sebelumnya", 0, 20, key='Previous_Hospitalization')
        st.number_input("Total Kunjungan RS", 1, 50, key='Number_of_Admissions')
    with c6:
        st.selectbox("Masuk Lewat Jalur IGD?", [0, 1], format_func=lambda x: "Tidak" if x==0 else "Ya", key='Emergency_Admission')
        st.number_input("Jumlah Resep Obat Berjalan", 1, 20, key='Medication_Count')
        st.slider("Kepatuhan Minum Obat (0-10)", 0, 10, key='Medication_Adherence')
        st.slider("Kehadiran Kontrol Dokter (0-10)", 0, 10, key='Followup_Attendance')
st.markdown('</div>', unsafe_allow_html=True)

# --- ACTION BUTTON ---
st.markdown('<div class="btn-predict">', unsafe_allow_html=True)
predict_btn = st.button("🔬 JALANKAN DIAGNOSIS AI SEKARANG")
st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    with st.spinner('Menghubungkan ke Engine AI. Melakukan komputasi Shapley Additive Explanations...'):
        input_data = {col: st.session_state[col] for col in feature_names}
        df_input = pd.DataFrame([input_data])
        df_encoded = df_input.copy()
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col in encoders:
                le = encoders[col]
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df_encoded[col] = le.transform(df_encoded[col])
                
        prediction_encoded = model.predict(df_encoded)[0]
        prediction_proba = model.predict_proba(df_encoded)[0]
        
        target_le = encoders['Risk_Label']
        predicted_label = target_le.inverse_transform([prediction_encoded])[0]
        confidence = prediction_proba[prediction_encoded] * 100

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df_encoded)
        class_idx = prediction_encoded
        instance_shap_values = shap_values.values[0, :, class_idx]
        
        # --- TAMPILAN HASIL UTAMA ---
        st.markdown("### 📋 Executive Summary: Keputusan Sistem AI")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        r1, r2, r3 = st.columns([1.5, 1, 1])
        with r1:
            st.write("Tingkat Prediksi Risiko Rehospitalisasi:")
            if predicted_label == 'High':
                st.markdown('<div class="risk-badge high-risk">🚨 RISIKO TINGGI (HIGH)</div>', unsafe_allow_html=True)
            elif predicted_label == 'Medium':
                st.markdown('<div class="risk-badge med-risk">⚠️ RISIKO SEDANG (MEDIUM)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-badge low-risk">✅ RISIKO RENDAH (LOW)</div>', unsafe_allow_html=True)
        
        with r2:
            st.metric(label="Keyakinan Mesin AI", value=f"{confidence:.1f}%")
            st.progress(confidence / 100)
            
        with r3:
            st.metric(label="Saturasi O2 / Detak Jantung", value=f"{st.session_state['Oxygen_Saturation']}% / {st.session_state['Heart_Rate']}")
            st.metric(label="Tensi Darah", value=f"{int(st.session_state['Systolic_BP'])}/{int(st.session_state['Diastolic_BP'])}")
        st.markdown('</div>', unsafe_allow_html=True)

        feature_importance = list(zip(feature_names, instance_shap_values, df_input.iloc[0].values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        risk_increasers = [f for f in feature_importance if f[1] > 0]
        risk_decreasers = [f for f in feature_importance if f[1] < 0]

        # MENGGUNAKAN PENDEKATAN PARAGRAF NARATIF (NLG)
        clinical_paragraph = generate_clinical_paragraph(predicted_label, risk_increasers, risk_decreasers)

        st.markdown("### 🗣️ Catatan Evaluasi Klinis (Narrative Summary)")
        st.markdown(f"""
        <div class="clinical-note-box">
            <div class="clinical-note-title">📑 Ringkasan Evaluasi Medis (AI-Assisted)</div>
            {clinical_paragraph}
        </div><br>
        """, unsafe_allow_html=True)

        # --- GENERATE PDF BUTTON ---
        pdf_data = generate_pdf(patient_name if patient_name else "NN (Tanpa Nama)", predicted_label, confidence, clinical_paragraph, input_data)
        
        st.download_button(
            label="📄 CETAK BERKAS LAPORAN DIAGNOSIS (.PDF)",
            data=pdf_data,
            file_name=f"Report_CardioCare_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        # --- GRAFIK SHAP WATERFALL ---
        st.markdown("---")
        st.markdown("### 📊 Detail Statistik Transparansi AI (Grafik SHAP)")
        st.caption("Khusus untuk verifikasi tenaga medis: Grafik Waterfall Shapley di bawah menunjukkan besaran distribusi matematis yang melatarbelakangi narasi klinis di atas.")
        fig = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0, :, class_idx], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)