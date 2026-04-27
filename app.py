import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Dog Behavior Interpreter", page_icon="🐶", layout="centered")

st.title("🐶 Dog Behavior Interpreter")
st.caption("AI prototype for interpreting short-duration dog behavior using structured context inputs.")

# -----------------------------
# Synthetic Data
# -----------------------------
def generate_data(n=900):
    np.random.seed(42)

    behaviors = ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"]
    activity = ["none", "low", "high"]
    environment = ["indoor", "outdoor", "warm", "noisy"]
    duration = ["short", "medium", "long"]
    assumption = ["anxiety", "boredom", "overstimulation", "recovery", "unsure"]
    age_group = ["puppy", "adult", "senior"]
    size = ["small", "medium", "large"]
    energy = ["low", "moderate", "high"]
    sensitivity = ["low", "moderate", "high"]

    rows = []

    for _ in range(n):
        b = np.random.choice(behaviors)
        a = np.random.choice(activity)
        e = np.random.choice(environment)
        d = np.random.choice(duration)
        s = np.random.choice(assumption)
        ag = np.random.choice(age_group)
        sz = np.random.choice(size)
        en = np.random.choice(energy)
        sens = np.random.choice(sensitivity)

        # Prototype labeling logic
        if b == "panting" and (a == "high" or e == "warm") and d in ["short", "medium"]:
            label = "recovery"
        elif b in ["pacing", "whining", "hiding"] and e == "noisy":
            label = "anxiety"
        elif b == "pacing" and d == "long" and a == "none":
            label = "anxiety"
        elif b in ["whining", "hiding"] and sens == "high":
            label = "anxiety"
        elif b == "toy-seeking" and a in ["none", "low"]:
            label = "boredom"
        elif b in ["pacing", "panting"] and a == "high" and en == "high":
            label = "overstimulation"
        elif b == "resting":
            label = "neutral"
        elif b == "panting" and ag == "senior" and d == "long":
            label = "needs observation"
        else:
            label = np.random.choice(
                ["anxiety", "boredom", "overstimulation", "recovery", "neutral", "needs observation"],
                p=[0.23, 0.20, 0.18, 0.18, 0.13, 0.08]
            )

        rows.append({
            "behavior": b,
            "activity": a,
            "environment": e,
            "duration": d,
            "assumption": s,
            "age_group": ag,
            "size": sz,
            "energy": en,
            "sensitivity": sens,
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
    ("rf", RandomForestClassifier(n_estimators=175, random_state=42, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# -----------------------------
# Helper Functions
# -----------------------------
def confidence_label(prob):
    if prob >= 0.75:
        return "High confidence"
    elif prob >= 0.55:
        return "Moderate confidence"
    return "Low confidence"

def build_reasoning(pred, behavior, activity, environment, duration, age_group, energy, sensitivity):
    reasons = []

    if behavior == "panting":
        reasons.append("Panting is ambiguous and can reflect recovery, heat, anxiety, or overstimulation.")
    if behavior == "pacing":
        reasons.append("Pacing can indicate anxiety, boredom, anticipation, or excess arousal.")
    if behavior == "whining":
        reasons.append("Whining can signal stress, attention-seeking, unmet needs, or discomfort.")
    if behavior == "hiding":
        reasons.append("Hiding is often associated with avoidance, fear, stress, or a need for space.")
    if behavior == "toy-seeking":
        reasons.append("Toy-seeking often suggests a desire for interaction, enrichment, or play.")
    if behavior == "resting":
        reasons.append("Resting generally lowers concern unless paired with unusual context or prolonged change.")

    if activity == "high":
        reasons.append("Recent high activity increases the likelihood of recovery or overstimulation.")
    elif activity == "none":
        reasons.append("No recent activity makes recovery less likely and increases the relevance of boredom or stress signals.")

    if environment == "warm":
        reasons.append("A warm environment increases the likelihood that panting is related to thermoregulation.")
    if environment == "noisy":
        reasons.append("A noisy environment can increase stress sensitivity and anxiety-related behavior.")

    if duration == "long":
        reasons.append("Longer duration raises concern because repeated behavior is less likely to be a brief normal response.")
    elif duration == "short":
        reasons.append("Short duration lowers concern and may indicate a temporary state.")

    if age_group == "senior":
        reasons.append("Senior dogs may require more caution when behavior persists or changes suddenly.")
    if energy == "high":
        reasons.append("High-energy dogs may show pacing or toy-seeking when stimulation needs are unmet.")
    if sensitivity == "high":
        reasons.append("High sensitivity increases the relevance of anxiety or environmental triggers.")

    if not reasons:
        reasons.append("The prediction is based on the combined behavior and context pattern.")

    return reasons

def risk_and_protective_factors(behavior, activity, environment, duration, age_group, energy, sensitivity):
    risk = []
    protective = []

    if behavior in ["pacing", "whining", "hiding"]:
        risk.append(f"Observed behavior: {behavior}")
    if environment == "noisy":
        risk.append("Noisy environment")
    if duration == "long":
        risk.append("Long duration")
    if sensitivity == "high":
        risk.append("High stress sensitivity")
    if age_group == "senior":
        risk.append("Senior age group")
    if activity == "none" and behavior in ["pacing", "toy-seeking"]:
        risk.append("No recent activity with active behavior")

    if activity == "high" and behavior == "panting":
        protective.append("Panting after high activity can indicate normal recovery")
    if duration == "short":
        protective.append("Short duration")
    if environment in ["indoor", "outdoor"] and environment != "noisy":
        protective.append("No obvious environmental stressor selected")
    if behavior == "resting":
        protective.append("Resting is generally neutral")
    if sensitivity in ["low", "moderate"]:
        protective.append(f"{sensitivity.title()} stress sensitivity")

    if not risk:
        risk.append("No major risk factors identified from selected inputs")
    if not protective:
        protective.append("Limited protective context provided")

    return risk, protective

def recommendation_for(pred):
    recs = {
        "recovery": {
            "summary": "Provide water, allow rest, and reassess in 10–15 minutes.",
            "steps": [
                "Offer water.",
                "Move to a calm, comfortable area.",
                "Avoid more intense activity until breathing and posture normalize."
            ]
        },
        "anxiety": {
            "summary": "Reduce stimuli and observe for additional stress signals.",
            "steps": [
                "Lower noise and environmental stimulation.",
                "Give space rather than forcing interaction.",
                "Watch for repeated pacing, hiding, trembling, or refusal to settle."
            ]
        },
        "boredom": {
            "summary": "Offer enrichment, light play, or a short training activity.",
            "steps": [
                "Try a puzzle toy, sniff game, or short training session.",
                "Keep the activity low-pressure.",
                "Reassess whether the behavior decreases afterward."
            ]
        },
        "overstimulation": {
            "summary": "Pause activity and create a calmer environment.",
            "steps": [
                "Stop exciting play or activity.",
                "Move to a quiet space.",
                "Use calm routines and let the dog decompress."
            ]
        },
        "neutral": {
            "summary": "Continue observing. No immediate intervention appears necessary.",
            "steps": [
                "Monitor for changes.",
                "Avoid over-intervening if the dog appears relaxed.",
                "Reassess if behavior changes or persists."
            ]
        },
        "needs observation": {
            "summary": "Observe closely and consider professional guidance if behavior persists.",
            "steps": [
                "Monitor duration and intensity.",
                "Look for changes in breathing, posture, appetite, or responsiveness.",
                "Contact a veterinarian or trainer if the behavior is unusual or persistent."
            ]
        }
    }
    return recs[pred]

def escalation_guidance():
    return [
        "Seek veterinary guidance if panting is intense, unexplained, prolonged, or paired with collapse, vomiting, pale gums, labored breathing, or severe distress.",
        "Seek trainer or behaviorist support if anxiety-like behaviors are frequent, escalating, or interfering with daily life.",
        "This prototype is decision support only and does not diagnose medical or behavioral conditions."
    ]

# -----------------------------
# Sidebar Metrics
# -----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

st.sidebar.header("Prototype Model Metrics")
st.sidebar.metric("Accuracy", f"{acc:.0%}")
st.sidebar.metric("Weighted Precision", f"{prec:.0%}")
st.sidebar.metric("Weighted Recall", f"{rec:.0%}")

with st.sidebar.expander("Data transparency"):
    st.write(
        """
        This prototype uses synthetic labeled examples to demonstrate the AI workflow.
        The uploaded image supports the intended user journey, but the current prediction is based on structured behavior and context inputs.
        A production version would use trainer-labeled video/image data and real outcome feedback.
        """
    )

# -----------------------------
# UI
# -----------------------------
st.subheader("1. Upload Dog Image (optional)")
image = st.file_uploader("Upload dog image", type=["jpg", "png", "jpeg"])

if image:
    st.image(image, caption="Uploaded image", use_container_width=True)

st.subheader("2. Dog Profile")
dog_name = st.text_input("Dog name", value="Sadie")
age_group = st.selectbox("Age group", ["puppy", "adult", "senior"], index=1)
size = st.selectbox("Size category", ["small", "medium", "large"], index=1)
energy = st.selectbox("Typical energy level", ["low", "moderate", "high"], index=2)
sensitivity = st.selectbox("Known stress sensitivity", ["low", "moderate", "high"], index=1)

st.subheader("3. Current Behavior Context")
behavior = st.selectbox("Observed behavior", ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"])
activity = st.selectbox("Recent activity", ["none", "low", "high"])
environment = st.selectbox("Environment", ["indoor", "outdoor", "warm", "noisy"])
duration = st.selectbox("Duration", ["short", "medium", "long"])
assumption = st.selectbox("Your assumption", ["unsure", "anxiety", "boredom", "overstimulation", "recovery"])

st.subheader("4. Analyze Behavior")

if st.button("Analyze Behavior", type="primary"):
    input_df = pd.DataFrame([{
        "behavior": behavior,
        "activity": activity,
        "environment": environment,
        "duration": duration,
        "assumption": assumption,
        "age_group": age_group,
        "size": size,
        "energy": energy,
        "sensitivity": sensitivity
    }])

    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    classes = model.named_steps["rf"].classes_

    prob_df = pd.DataFrame({
        "Behavior State": classes,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    top_prob = float(prob_df.iloc[0]["Probability"])
    second_state = prob_df.iloc[1]["Behavior State"]
    second_prob = float(prob_df.iloc[1]["Probability"])

    st.success("AI interpretation complete")

    st.markdown("## 🧠 AI Interpretation")
    st.write(f"**Dog:** {dog_name}")
    st.write(f"**Likely state:** {pred.title()}")
    st.write(f"**Confidence:** {top_prob:.0%} ({confidence_label(top_prob)})")
    st.write(f"**Secondary possibility:** {second_state.title()} ({second_prob:.0%})")

    if top_prob < 0.55:
        st.warning(
            "Confidence is low. The system recommends observing additional signals before taking strong action."
        )

    st.markdown("## Why this prediction was generated")
    for reason in build_reasoning(pred, behavior, activity, environment, duration, age_group, energy, sensitivity):
        st.write(f"- {reason}")

    risk, protective = risk_and_protective_factors(
        behavior, activity, environment, duration, age_group, energy, sensitivity
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Risk factors")
        for item in risk:
            st.write(f"- {item}")

    with col2:
        st.markdown("### Protective factors")
        for item in protective:
            st.write(f"- {item}")

    st.markdown("## Owner assumption comparison")
    if assumption == "unsure":
        st.write("No owner assumption was provided, so the system cannot compare against a baseline interpretation.")
    elif assumption == pred:
        st.write(
            f"The AI interpretation aligns with the owner’s assumption of **{assumption}**."
        )
    else:
        st.write(
            f"The owner assumed **{assumption}**, while the model predicted **{pred}**. "
            "This may represent a potential interpretation correction and is useful for measuring misinterpretation reduction."
        )

    st.markdown("## Recommended action plan")
    rec = recommendation_for(pred)
    st.info(rec["summary"])
    for step in rec["steps"]:
        st.write(f"- {step}")

    with st.expander("When to escalate"):
        for item in escalation_guidance():
            st.write(f"- {item}")

    st.markdown("## Probability distribution")
    chart_df = prob_df.set_index("Behavior State")
    st.bar_chart(chart_df)

    st.subheader("5. Feedback Loop")
    accurate = st.radio("Was this interpretation accurate?", ["Yes", "No", "Unsure"], horizontal=True)
    action_taken = st.radio("Did you take the recommended action?", ["Yes", "No", "Partially"], horizontal=True)
    improved = st.radio("Did the behavior improve afterward?", ["Yes", "No", "Not sure"], horizontal=True)

    if st.button("Submit Feedback"):
        feedback_row = {
            "timestamp": datetime.now().isoformat(),
            "dog_name": dog_name,
            "behavior": behavior,
            "activity": activity,
            "environment": environment,
            "duration": duration,
            "assumption": assumption,
            "age_group": age_group,
            "size": size,
            "energy": energy,
            "sensitivity": sensitivity,
            "prediction": pred,
            "confidence": round(top_prob, 3),
            "secondary_state": second_state,
            "secondary_probability": round(second_prob, 3),
            "accurate_feedback": accurate,
            "action_taken": action_taken,
            "behavior_improved": improved
        }

        feedback_df = pd.DataFrame([feedback_row])

        try:
            existing = pd.read_csv("feedback_log.csv")
            updated = pd.concat([existing, feedback_df], ignore_index=True)
        except FileNotFoundError:
            updated = feedback_df

        updated.to_csv("feedback_log.csv", index=False)

        st.success("Feedback saved for future dataset improvement and retraining.")
        st.dataframe(feedback_df)

else:
    st.info("Enter the dog profile and behavior context, then click Analyze Behavior.")
