import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Dog Behavior Interpreter", page_icon="🐶")

st.title("🐶 Dog Behavior Interpreter")
st.caption("AI prototype to interpret dog behavior using context inputs")

# -----------------------------
# Synthetic Data
# -----------------------------
def generate_data(n=500):
    np.random.seed(42)

    behaviors = ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"]
    activity = ["none", "low", "high"]
    environment = ["indoor", "outdoor", "warm", "noisy"]
    duration = ["short", "medium", "long"]
    assumption = ["anxiety", "boredom", "overstimulation", "recovery", "unsure"]

    rows = []

    for _ in range(n):
        b = np.random.choice(behaviors)
        a = np.random.choice(activity)
        e = np.random.choice(environment)
        d = np.random.choice(duration)
        s = np.random.choice(assumption)

        if b == "panting" and a == "high":
            label = "recovery"
        elif b == "panting" and e == "warm":
            label = "recovery"
        elif b in ["pacing", "whining", "hiding"] and e == "noisy":
            label = "anxiety"
        elif b == "pacing" and d == "long" and a == "none":
            label = "anxiety"
        elif b == "toy-seeking":
            label = "boredom"
        elif b in ["panting", "pacing"] and a == "high":
            label = "overstimulation"
        elif b == "resting":
            label = "neutral"
        else:
            label = np.random.choice(["anxiety", "boredom", "overstimulation", "recovery", "neutral"])

        rows.append({
            "behavior": b,
            "activity": a,
            "environment": e,
            "duration": d,
            "assumption": s,
            "label": label
        })

    return pd.DataFrame(rows)

df = generate_data()

# -----------------------------
# Model
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)
])

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

# -----------------------------
# UI Inputs
# -----------------------------
st.subheader("1. Upload Image (optional)")
image = st.file_uploader("Upload dog image", type=["jpg", "png", "jpeg"])

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)

st.subheader("2. Provide Context")

behavior = st.selectbox("Observed behavior", ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"])
activity = st.selectbox("Recent activity", ["none", "low", "high"])
environment = st.selectbox("Environment", ["indoor", "outdoor", "warm", "noisy"])
duration = st.selectbox("Duration", ["short", "medium", "long"])
assumption = st.selectbox("Your assumption (optional)", ["unsure", "anxiety", "boredom", "overstimulation", "recovery"])

# -----------------------------
# Prediction
# -----------------------------
st.subheader("3. Analyze Behavior")

if st.button("Analyze"):

    input_df = pd.DataFrame([{
        "behavior": behavior,
        "activity": activity,
        "environment": environment,
        "duration": duration,
        "assumption": assumption
    }])

    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    classes = model.named_steps["rf"].classes_

    prob_df = pd.DataFrame({
        "state": classes,
        "prob": probs
    }).sort_values("prob", ascending=False)

    top_prob = prob_df.iloc[0]["prob"]
    second = prob_df.iloc[1]

    st.success("AI Interpretation Complete")

    st.markdown("### Behavioral Interpretation")
    st.write(f"**Likely state:** {pred}")
    st.write(f"**Confidence:** {top_prob:.0%}")
    st.write(f"**Secondary:** {second['state']} ({second['prob']:.0%})")

    st.markdown("### Explanation")
    st.write("Prediction is based on behavior + context patterns learned from data.")

    st.markdown("### Recommendation")

    if pred == "recovery":
        st.info("Provide water and allow rest.")
    elif pred == "anxiety":
        st.info("Reduce stimuli and observe.")
    elif pred == "boredom":
        st.info("Provide enrichment or play.")
    elif pred == "overstimulation":
        st.info("Pause activity and calm environment.")
    else:
        st.info("No immediate action needed.")

    st.subheader("4. Feedback")

    accurate = st.radio("Was this accurate?", ["Yes", "No"])
    improved = st.radio("Did behavior improve?", ["Yes", "No", "Not sure"])

    if st.button("Submit Feedback"):
        st.success("Feedback recorded (prototype simulation).")

# -----------------------------
# Metrics Sidebar
# -----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")

st.sidebar.header("Model Metrics")
st.sidebar.write(f"Accuracy: {acc:.2f}")
st.sidebar.write(f"Precision: {prec:.2f}")
st.sidebar.write(f"Recall: {rec:.2f}")
