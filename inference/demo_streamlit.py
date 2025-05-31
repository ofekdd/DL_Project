
import streamlit as st, tempfile, subprocess, yaml, torch
st.set_page_config(page_title="Instrument Classifier Demo")
st.title("🎹 Instrument Classifier")

cfg = yaml.safe_load(open("configs/default.yaml"))
ckpt_path = st.text_input("Checkpoint path", "best.ckpt")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded and st.button("Predict"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        res = subprocess.check_output(
            ["python", "inference/predict.py", ckpt_path, tmp.name]
        )
        st.json(eval(res))
