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

st.set_page_config(
    page_title="Dog Behavior Interpreter",
    page_icon="🐶",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
    max-width: 1120px;
}

.sticky-disclaimer {
    position: sticky;
    top: 0;
    z-index: 1000;
    background-color: #FFF7ED;
    border: 1px solid #FDBA74;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
    color: #7C2D12;
}

.sticky-progress {
    position: sticky;
    top: 72px;
    z-index: 999;
    background-color: white;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.progress-track {
    height: 10px;
    background-color: #E5E7EB;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
    margin-bottom: 8px;
}

.progress-fill {
    height: 10px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    border-radius: 999px;
}

.progress-label {
    font-size: 13px;
    color: #6B7280;
}

.card {
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    background-color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}

.result-card {
    padding: 1.4rem;
    border-radius: 20px;
    border: 1px solid #D1FAE5;
    background-color: #F0FDF4;
    margin-bottom: 1rem;
}

.sticky-result {
    position: sticky;
    top: 152px;
    z-index: 998;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #BBF7D0;
    background-color: #F0FDF4;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.warning-card {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #FDE68A;
    background-color: #FFFBEB;
}

.small-muted {
    color: #6B7280;
    font-size: 0.9rem;
}

.step-pill {
    display: inline-block;
    padding: 0.4rem 0.75rem;
    border-radius: 999px;
    background-color: #EEF2FF;
    color: #3730A3;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

.kpi-box {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    background-color: #F9FAFB;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = False


def progress_state():
    if st.session_state.feedback_done:
        return 1.0, "Complete: feedback captured for learning loop"
    if st.session_state.analysis_done:
        return 0.75, "Step 3 of 4: review AI interpretation and decide action"
    return 0.35, "Step 1–2 of 4: enter context and visual cues"


progress_value, progress_text = progress_state()

# -----------------------------
# Persistent safety disclaimer + progress
# -----------------------------
st.markdown("""
<div class="sticky-disclaimer">
⚠️ <b>Disclaimer:</b> This prototype provides behavioral guidance only. It is <b>not a medical or diagnostic tool</b>.
Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress, or symptoms that are unusual, persistent, or worsening.
For recurring anxiety-related behavior, consult a licensed veterinarian, certified trainer, or veterinary behavior professional.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="sticky-progress">
    <b>Prototype progress</b>
    <div class="progress-track">
        <div class="progress-fill" style="width:{int(progress_value * 100)}%;"></div>
    </div>
    <div class="progress-label">{progress_text}</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Data generation
# -----------------------------
@st.cache_data
def generate_data(n=1200):
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

    mouth = ["unknown", "closed mouth", "open mouth / panting"]
    posture = ["unknown", "relaxed", "alert", "tense", "crouched"]
    ears = ["unknown", "neutral", "back/pinned"]
    tail = ["unknown", "relaxed", "tucked", "high"]
    eyes = ["unknown", "relaxed", "wide-eyed / whale eye"]
    hiding_visible = ["unknown", "yes", "no"]

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

        m = np.random.choice(mouth)
        p = np.random.choice(posture)
        er = np.random.choice(ears)
        t = np.random.choice(tail)
        ey = np.random.choice(eyes)
        hv = np.random.choice(hiding_visible)

        anxiety_visuals = (
            p in ["tense", "crouched"]
            or er == "back/pinned"
            or t == "tucked"
            or ey == "wide-eyed / whale eye"
            or hv == "yes"
        )

        relaxed_visuals = (
            p == "relaxed"
            and er in ["neutral", "unknown"]
            and t in ["relaxed", "unknown"]
            and ey in ["relaxed", "unknown"]
        )

        if (b == "panting" or m == "open mouth / panting") and (a == "high" or e == "warm") and d in ["short", "medium"]:
            label = "recovery"
        elif anxiety_visuals and (e == "noisy" or sens == "high" or d == "long"):
            label = "anxiety"
        elif b in ["pacing", "whining", "hiding"] and e == "noisy":
            label = "anxiety"
        elif b == "pacing" and d == "long" and a == "none":
            label = "anxiety"
        elif b == "toy-seeking" and a in ["none", "low"] and not anxiety_visuals:
            label = "boredom"
        elif b in ["pacing", "panting"] and a == "high" and en == "high":
            label = "overstimulation"
        elif b == "resting" and relaxed_visuals:
            label = "neutral"
        elif (b == "panting" or m == "open mouth / panting") and ag == "senior" and d == "long":
            label = "needs observation"
        else:
            label = np.random.choice(
                ["anxiety", "boredom", "overstimulation", "recovery", "neutral", "needs observation"],
                p=[0.23, 0.19, 0.18, 0.18, 0.14, 0.08]
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
            "visual_mouth": m,
            "visual_posture": p,
            "visual_ears": er,
            "visual_tail": t,
            "visual_eyes": ey,
            "visual_hiding": hv,
            "label": label
        })

    return pd.DataFrame(rows)


@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    return model, metrics


df = generate_data()
model, metrics = train_model(df)

# -----------------------------
# Helper functions
# -----------------------------
def confidence_label(prob):
    if prob >= 0.75:
        return "High confidence"
    if prob >= 0.55:
        return "Moderate confidence"
    return "Low confidence"


def confidence_color(prob):
    if prob >= 0.75:
        return "🟢"
    if prob >= 0.55:
        return "🟡"
    return "🔴"


def build_reasoning(
    behavior,
    activity,
    environment,
    duration,
    age_group,
    energy,
    sensitivity,
    visual_mouth,
    visual_posture,
    visual_ears,
    visual_tail,
    visual_eyes,
    visual_hiding
):
    reasons = []

    if behavior == "panting":
        reasons.append("Panting is ambiguous and may reflect recovery, heat, anxiety, or overstimulation.")
    if behavior == "pacing":
        reasons.append("Pacing may indicate anxiety, boredom, anticipation, or excess arousal.")
    if behavior == "whining":
        reasons.append("Whining may signal stress, attention-seeking, unmet needs, or discomfort.")
    if behavior == "hiding":
        reasons.append("Hiding is often associated with avoidance, fear, stress, or a need for space.")
    if behavior == "toy-seeking":
        reasons.append("Toy-seeking often suggests a desire for interaction, enrichment, or play.")
    if behavior == "resting":
        reasons.append("Resting generally lowers concern unless paired with unusual context or sudden change.")

    if visual_mouth == "open mouth / panting":
        reasons.append("The image-assisted cue indicates an open mouth or visible panting.")
    if visual_posture in ["tense", "crouched"]:
        reasons.append(f"The image-assisted posture cue is {visual_posture}, which can increase concern.")
    if visual_ears == "back/pinned":
        reasons.append("Pinned-back ears can be associated with stress, uncertainty, or avoidance.")
    if visual_tail == "tucked":
        reasons.append("A tucked tail can indicate fear, anxiety, or discomfort.")
    if visual_eyes == "wide-eyed / whale eye":
        reasons.append("Wide eyes or whale eye can be a stress-related visual cue.")
    if visual_hiding == "yes":
        reasons.append("Visible hiding or avoidance increases the likelihood of stress-related interpretation.")

    if activity == "high":
        reasons.append("Recent high activity increases the likelihood of recovery or overstimulation.")
    elif activity == "none":
        reasons.append("No recent activity makes recovery less likely and increases the relevance of boredom or stress.")

    if environment == "warm":
        reasons.append("Warm conditions increase the likelihood that panting is related to thermoregulation.")
    if environment == "noisy":
        reasons.append("Noise can increase stress sensitivity and anxiety-related behavior.")

    if duration == "long":
        reasons.append("Long duration raises concern because repeated behavior is less likely to be a brief normal response.")
    elif duration == "short":
        reasons.append("Short duration lowers concern and may indicate a temporary state.")

    if age_group == "senior":
        reasons.append("Senior dogs require more caution when behavior persists or changes suddenly.")
    if energy == "high":
        reasons.append("High-energy dogs may pace or seek toys when stimulation needs are unmet.")
    if sensitivity == "high":
        reasons.append("High stress sensitivity increases the relevance of anxiety or environmental triggers.")

    return reasons


def risk_and_protective_factors(
    behavior,
    activity,
    environment,
    duration,
    age_group,
    energy,
    sensitivity,
    visual_mouth,
    visual_posture,
    visual_ears,
    visual_tail,
    visual_eyes,
    visual_hiding
):
    risk = []
    protective = []

    if behavior in ["pacing", "whining", "hiding"]:
        risk.append(f"Observed behavior: {behavior}")
    if visual_posture in ["tense", "crouched"]:
        risk.append(f"Visual posture cue: {visual_posture}")
    if visual_ears == "back/pinned":
        risk.append("Visual cue: ears back/pinned")
    if visual_tail == "tucked":
        risk.append("Visual cue: tucked tail")
    if visual_eyes == "wide-eyed / whale eye":
        risk.append("Visual cue: wide-eyed / whale eye")
    if visual_hiding == "yes":
        risk.append("Visual cue: hiding or avoidance")
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

    if activity == "high" and (behavior == "panting" or visual_mouth == "open mouth / panting"):
        protective.append("Panting after high activity can indicate normal recovery")
    if duration == "short":
        protective.append("Short duration")
    if environment in ["indoor", "outdoor"]:
        protective.append("No obvious heat/noise trigger selected")
    if behavior == "resting":
        protective.append("Resting is generally neutral")
    if visual_posture == "relaxed":
        protective.append("Visual posture cue: relaxed")
    if visual_ears == "neutral":
        protective.append("Visual cue: neutral ears")
    if visual_tail == "relaxed":
        protective.append("Visual cue: relaxed tail")
    if visual_eyes == "relaxed":
        protective.append("Visual cue: relaxed eyes")
    if sensitivity in ["low", "moderate"]:
        protective.append(f"{sensitivity.title()} stress sensitivity")

    if not risk:
        risk.append("No major risk factors identified")
    if not protective:
        protective.append("Limited protective context provided")

    return risk, protective


def recommendation_for(pred):
    recs = {
        "recovery": {
            "summary": "Provide water, allow rest, and reassess in 10–15 minutes.",
            "steps": [
                "Offer water.",
                "Move to a calm area.",
                "Avoid more intense activity until breathing and posture normalize."
            ]
        },
        "anxiety": {
            "summary": "Reduce stimuli and observe for additional stress signals.",
            "steps": [
                "Lower noise and stimulation.",
                "Give space rather than forcing interaction.",
                "Watch for repeated pacing, hiding, trembling, or inability to settle."
            ]
        },
        "boredom": {
            "summary": "Offer enrichment, light play, or a short training activity.",
            "steps": [
                "Try a puzzle toy, sniff game, or short training session.",
                "Keep the activity low-pressure.",
                "Reassess whether behavior decreases afterward."
            ]
        },
        "overstimulation": {
            "summary": "Pause activity and create a calmer environment.",
            "steps": [
                "Stop exciting play.",
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
                "Contact a veterinarian or trainer if unusual or persistent."
            ]
        }
    }
    return recs[pred]


def escalation_guidance():
    return [
        "Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress, or other emergency signs.",
        "Consult a veterinarian if symptoms persist, worsen, or are unusual for your dog.",
        "Consult a certified trainer or veterinary behavior professional if anxiety-related behaviors are frequent, escalating, or interfering with daily life.",
        "This tool is for behavioral decision support only and does not replace professional medical or behavioral evaluation."
    ]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Prototype Model Metrics")
    st.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    st.metric("Weighted Precision", f"{metrics['precision']:.0%}")
    st.metric("Weighted Recall", f"{metrics['recall']:.0%}")

    with st.expander("Data transparency"):
        st.write(
            """
            This prototype uses synthetic labeled examples to demonstrate the AI workflow.
            The uploaded image supports image-assisted manual visual cue extraction.
            The user reviews the image and selects visible cues such as posture, ears, tail, eyes, and panting visibility.
            These cues are included as model features.
            A production version could automate this step with computer vision or short video analysis.
            """
        )

# -----------------------------
# Header
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.title("🐶 Dog Behavior Interpreter")
    st.write(
        "A mobile-first AI prototype that uses dog profile, behavior context, and image-assisted visual cues "
        "to interpret ambiguous behavior and recommend a next action."
    )

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**How it works**")
    st.markdown(
        "<span class='step-pill'>Input</span>"
        "<span class='step-pill'>Visual cues</span>"
        "<span class='step-pill'>AI interpretation</span>"
        "<span class='step-pill'>Feedback loop</span>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Input form
# -----------------------------
st.markdown("## Step 1 — Enter dog profile and behavior context")
st.progress(0.25)

form_col, preview_col = st.columns([1.2, 0.8])

with form_col:
    with st.form("behavior_form"):
        st.markdown("### Dog profile")
        profile_cols = st.columns(2)

        with profile_cols[0]:
            dog_name = st.text_input("Dog name", value="Sadie")
            age_group = st.selectbox("Age group", ["puppy", "adult", "senior"], index=1)
            size = st.selectbox("Size category", ["small", "medium", "large"], index=1)

        with profile_cols[1]:
            energy = st.selectbox("Typical energy level", ["low", "moderate", "high"], index=2)
            sensitivity = st.selectbox("Known stress sensitivity", ["low", "moderate", "high"], index=1)
            assumption = st.selectbox("Your assumption", ["unsure", "anxiety", "boredom", "overstimulation", "recovery"])

        st.markdown("### Current behavior")
        behavior_cols = st.columns(2)

        with behavior_cols[0]:
            behavior = st.selectbox("Observed behavior", ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"])
            activity = st.selectbox("Recent activity", ["none", "low", "high"])

        with behavior_cols[1]:
            environment = st.selectbox("Environment", ["indoor", "outdoor", "warm", "noisy"])
            duration = st.selectbox("Duration", ["short", "medium", "long"])

        st.markdown("### Upload image and extract visual cues")
        image = st.file_uploader("Optional dog image", type=["jpg", "png", "jpeg"])

        st.caption(
            "Use the image or your real-time observation to select visible cues. "
            "These cues are used by the model."
        )

        visual_cols = st.columns(2)

        with visual_cols[0]:
            visual_mouth = st.selectbox("Mouth / panting visible", ["unknown", "closed mouth", "open mouth / panting"])
            visual_posture = st.selectbox("Body posture", ["unknown", "relaxed", "alert", "tense", "crouched"])
            visual_ears = st.selectbox("Ear position", ["unknown", "neutral", "back/pinned"])

        with visual_cols[1]:
            visual_tail = st.selectbox("Tail position", ["unknown", "relaxed", "tucked", "high"])
            visual_eyes = st.selectbox("Eye expression", ["unknown", "relaxed", "wide-eyed / whale eye"])
            visual_hiding = st.selectbox("Hiding / avoidance visible", ["unknown", "yes", "no"])

        submitted = st.form_submit_button("Analyze Behavior", type="primary", use_container_width=True)

with preview_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Input summary")
    st.write("Only inputs that contribute to interpretation are collected.")
    st.write(f"**Dog:** {dog_name}")
    st.write(f"**Behavior:** {behavior}")
    st.write(f"**Activity:** {activity}")
    st.write(f"**Environment:** {environment}")
    st.write(f"**Duration:** {duration}")
    st.write(f"**Assumption:** {assumption}")

    st.markdown("**Visual cues**")
    st.write(f"- Mouth: {visual_mouth}")
    st.write(f"- Posture: {visual_posture}")
    st.write(f"- Ears: {visual_ears}")
    st.write(f"- Tail: {visual_tail}")
    st.write(f"- Eyes: {visual_eyes}")
    st.write(f"- Hiding/avoidance: {visual_hiding}")

    if image:
        st.image(image, caption="Uploaded dog image for visual cue review", use_container_width=True)
    else:
        st.markdown(
            "<span class='small-muted'>Image optional: visual cues can still be entered manually if observed directly.</span>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Results
# -----------------------------
if submitted:
    st.session_state.analysis_done = True

    st.divider()
    st.markdown("## Step 2 — AI interpretation")
    st.progress(0.70)

    input_df = pd.DataFrame([{
        "behavior": behavior,
        "activity": activity,
        "environment": environment,
        "duration": duration,
        "assumption": assumption,
        "age_group": age_group,
        "size": size,
        "energy": energy,
        "sensitivity": sensitivity,
        "visual_mouth": visual_mouth,
        "visual_posture": visual_posture,
        "visual_ears": visual_ears,
        "visual_tail": visual_tail,
        "visual_eyes": visual_eyes,
        "visual_hiding": visual_hiding
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

    rec = recommendation_for(pred)
    risk, protective = risk_and_protective_factors(
        behavior,
        activity,
        environment,
        duration,
        age_group,
        energy,
        sensitivity,
        visual_mouth,
        visual_posture,
        visual_ears,
        visual_tail,
        visual_eyes,
        visual_hiding
    )

    st.markdown(f"""
    <div class="sticky-result">
        <b>{confidence_color(top_prob)} Current interpretation:</b> {pred.title()} &nbsp; | &nbsp;
        <b>Confidence:</b> {top_prob:.0%} ({confidence_label(top_prob)})
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"### {confidence_color(top_prob)} Likely state: **{pred.title()}**")
    st.write(f"**Dog:** {dog_name}")
    st.write(f"**Confidence:** {top_prob:.0%} — {confidence_label(top_prob)}")
    st.progress(top_prob)
    st.write(f"**Secondary possibility:** {second_state.title()} ({second_prob:.0%})")
    st.markdown("</div>", unsafe_allow_html=True)

    if top_prob < 0.55:
        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
        st.warning("Confidence is low. Observe additional signals before taking strong action.")
        st.markdown("</div>", unsafe_allow_html=True)

    output_cols = st.columns([1, 1])

    with output_cols[0]:
        st.markdown("### Why this prediction was generated")
        reasons = build_reasoning(
            behavior,
            activity,
            environment,
            duration,
            age_group,
            energy,
            sensitivity,
            visual_mouth,
            visual_posture,
            visual_ears,
            visual_tail,
            visual_eyes,
            visual_hiding
        )
        for reason in reasons:
            st.write(f"- {reason}")

    with output_cols[1]:
        st.markdown("### Recommended action")
        st.info(rec["summary"])
        for step in rec["steps"]:
            st.write(f"- {step}")

    factor_cols = st.columns(2)

    with factor_cols[0]:
        st.markdown("### Risk factors")
        for item in risk:
            st.write(f"- {item}")

    with factor_cols[1]:
        st.markdown("### Protective factors")
        for item in protective:
            st.write(f"- {item}")

    st.markdown("### Owner assumption comparison")
    if assumption == "unsure":
        st.write("No owner assumption was provided, so the system cannot compare against a baseline interpretation.")
    elif assumption == pred:
        st.success(f"The AI interpretation aligns with the owner’s assumption of **{assumption}**.")
    else:
        st.warning(
            f"The owner assumed **{assumption}**, while the model predicted **{pred}**. "
            "This may represent a potential interpretation correction."
        )

    with st.expander("Probability distribution"):
        chart_df = prob_df.set_index("Behavior State")
        st.bar_chart(chart_df)

    with st.expander("When to seek professional help"):
        for item in escalation_guidance():
            st.write(f"- {item}")

    st.divider()
    st.markdown("## Step 3 — Feedback loop")
    st.progress(1.0)

    with st.form("feedback_form"):
        fb_cols = st.columns(3)

        with fb_cols[0]:
            accurate = st.radio("Was this accurate?", ["Yes", "No", "Unsure"], horizontal=True)

        with fb_cols[1]:
            action_taken = st.radio("Did you take the action?", ["Yes", "No", "Partially"], horizontal=True)

        with fb_cols[2]:
            improved = st.radio("Did behavior improve?", ["Yes", "No", "Not sure"], horizontal=True)

        feedback_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

    if feedback_submitted:
        st.session_state.feedback_done = True

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
            "visual_mouth": visual_mouth,
            "visual_posture": visual_posture,
            "visual_ears": visual_ears,
            "visual_tail": visual_tail,
            "visual_eyes": visual_eyes,
            "visual_hiding": visual_hiding,
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
        with st.expander("Feedback record"):
            st.dataframe(feedback_df)

else:
    st.info("Complete the profile, behavior context, and visual cue review. Then click Analyze Behavior.")
