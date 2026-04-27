import os
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


st.set_page_config(page_title="Dog Behavior Interpreter", page_icon="🐶", layout="wide")


# ==================================================
# CSS
# ==================================================

st.markdown("""
<style>
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1180px;
}
.disclaimer-card {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #FDBA74;
    background-color: #FFF7ED;
    color: #7C2D12;
    margin-bottom: 1rem;
    font-size: 0.92rem;
}
.stepper {
    display: flex;
    gap: 8px;
    margin: 12px 0 20px 0;
    flex-wrap: wrap;
}
.step-active {
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: #4F46E5;
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
}
.step-complete {
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: #DCFCE7;
    color: #166534;
    font-weight: 700;
    font-size: 0.9rem;
}
.step-inactive {
    padding: 0.55rem 0.85rem;
    border-radius: 999px;
    background: #F3F4F6;
    color: #6B7280;
    font-weight: 600;
    font-size: 0.9rem;
}
.result-card {
    padding: 1.4rem;
    border-radius: 20px;
    border: 1px solid #BBF7D0;
    background-color: #F0FDF4;
    margin-bottom: 1rem;
}
.warning-card {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #FDE68A;
    background-color: #FFFBEB;
    margin-bottom: 1rem;
}
.kpi {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    background-color: #F9FAFB;
}
.action-box {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #DBEAFE;
    background-color: #EFF6FF;
    margin-bottom: 0.75rem;
}
.education-box {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #DDD6FE;
    background-color: #F5F3FF;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

def set_phase(n):
    st.session_state.phase = n


if "phase" not in st.session_state:
    st.session_state.phase = 1

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = False

for key, default in {
    "visual_mouth": "unknown",
    "visual_posture": "unknown",
    "visual_ears": "unknown",
    "visual_tail": "unknown",
    "visual_eyes": "unknown",
    "visual_hiding": "unknown",
    "image_available": "no",
    "image_ai_confidence": None,
    "image_ai_reason": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==================================================
# DATA + MODEL
# ==================================================

@st.cache_data
def generate_data(n=1400):
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
    image_available = ["yes", "no"]

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

        img = np.random.choice(image_available, p=[0.65, 0.35])
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

        panting_signal = b == "panting" or m == "open mouth / panting"

        if panting_signal and (a == "high" or e == "warm") and d in ["short", "medium"] and not anxiety_visuals:
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
        elif panting_signal and ag == "senior" and d == "long":
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
            "image_available": img,
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
        ("rf", RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced"))
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


# ==================================================
# HELPERS
# ==================================================

def get_openai_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


def analyze_image_with_ai(uploaded_image):
    api_key = get_openai_key()

    if not api_key:
        return {"error": "No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets."}

    client = OpenAI(api_key=api_key)

    image_bytes = uploaded_image.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_image.type or "image/jpeg"

    prompt = """
You are analyzing an uploaded image for a dog behavior interpretation prototype.

Your job:
1. Determine whether the image clearly shows a dog.
2. If it does not clearly show a dog, return image_valid=false.
3. If it shows a dog, infer visible body-language cues using ONLY the allowed values below.

Return valid JSON only. No markdown. No explanation outside JSON.

Allowed schema:
{
  "image_valid": true or false,
  "reason": "short explanation",
  "visual_mouth": "unknown | closed mouth | open mouth / panting",
  "visual_posture": "unknown | relaxed | alert | tense | crouched",
  "visual_ears": "unknown | neutral | back/pinned",
  "visual_tail": "unknown | relaxed | tucked | high",
  "visual_eyes": "unknown | relaxed | wide-eyed / whale eye",
  "visual_hiding": "unknown | yes | no",
  "confidence": 0.0 to 1.0
}

Be conservative. If a cue is not clearly visible, use "unknown".
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_base64}"
                        }
                    ]
                }
            ]
        )

        return json.loads(response.output_text)

    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}


def visual_cue_completeness(image_available, *visuals):
    known = sum(v != "unknown" for v in visuals)
    base = known / len(visuals)
    if image_available == "yes":
        return min(1.0, base + 0.15)
    return base * 0.75


def confidence_adjustment(base_prob, completeness):
    if completeness >= 0.75:
        return min(0.98, base_prob + 0.07)
    if completeness >= 0.4:
        return min(0.95, base_prob + 0.02)
    return max(0.35, base_prob - 0.08)


def confidence_label(prob):
    if prob >= 0.75:
        return "High confidence"
    if prob >= 0.55:
        return "Moderate confidence"
    return "Low confidence"


def quality_label(score):
    if score >= 0.75:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def confidence_color(prob):
    if prob >= 0.75:
        return "🟢"
    if prob >= 0.55:
        return "🟡"
    return "🔴"


def get_feedback_count():
    try:
        feedback = pd.read_csv("feedback_log.csv")
        return len(feedback)
    except FileNotFoundError:
        return 0


def recommendation_for(pred):
    return {
        "recovery": {
            "summary": "Likely recovery. Support rest, hydration, and reassessment.",
            "do_now": ["Offer water.", "Move your dog to a calm, comfortable area.", "Pause additional play, running, or training for now."],
            "observe": ["Breathing and posture should gradually normalize.", "Panting should decrease as the dog rests and cools down.", "Energy should return to baseline after recovery."],
            "avoid": ["Do not continue intense activity immediately.", "Do not assume panting always means anxiety without considering recent activity or heat."]
        },
        "anxiety": {
            "summary": "Possible anxiety or stress. Reduce triggers and support calm observation.",
            "do_now": ["Lower noise and stimulation.", "Create distance from the trigger if possible.", "Give your dog space instead of forcing interaction."],
            "observe": ["Watch for repeated pacing, hiding, trembling, lip licking, or inability to settle.", "Track whether the behavior improves after the environment becomes calmer."],
            "avoid": ["Avoid punishment or forcing exposure.", "Avoid overwhelming the dog with attention if they are trying to retreat."]
        },
        "boredom": {
            "summary": "Possible unmet stimulation need. Use low-pressure enrichment.",
            "do_now": ["Offer a puzzle toy, sniff game, or short training session.", "Try calm engagement before escalating intensity.", "Use a structured activity rather than random excitement."],
            "observe": ["Behavior should decrease after mental stimulation.", "Toy-seeking or pacing may reduce when the dog has an appropriate outlet."],
            "avoid": ["Avoid only using high-arousal play if the dog is already restless.", "Avoid reinforcing demand behavior every time without structure."]
        },
        "overstimulation": {
            "summary": "Possible overstimulation. Pause activity and help your dog decompress.",
            "do_now": ["Stop exciting play or training.", "Move to a quieter area.", "Use calm routines and allow decompression."],
            "observe": ["Look for reduction in pacing, panting, jumping, or inability to settle.", "Reassess after 10–15 minutes of calm."],
            "avoid": ["Avoid more chase, fetch, or intense play immediately.", "Avoid adding more stimulation to an already aroused state."]
        },
        "neutral": {
            "summary": "No strong concern detected. Continue observing without over-intervening.",
            "do_now": ["Let your dog continue resting or behaving normally.", "Monitor for changes."],
            "observe": ["Look for sudden shifts in posture, breathing, responsiveness, or appetite.", "Reassess if the behavior becomes prolonged or unusual."],
            "avoid": ["Avoid unnecessary intervention if the dog appears relaxed.", "Avoid assuming every behavior requires correction."]
        },
        "needs observation": {
            "summary": "Unclear or potentially elevated concern. Observe closely and consider professional guidance if it persists.",
            "do_now": ["Reduce activity and keep the dog comfortable.", "Monitor breathing, posture, appetite, responsiveness, and duration.", "Document what you are seeing."],
            "observe": ["Watch whether the behavior resolves or escalates.", "Track whether it repeats under similar conditions."],
            "avoid": ["Avoid ignoring persistent or unusual behavior.", "Avoid relying on this prototype for medical decisions."]
        }
    }[pred]


def escalation_guidance():
    return [
        "Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress, or other emergency signs.",
        "Consult a veterinarian if symptoms persist, worsen, or are unusual for your dog.",
        "Consult a certified trainer or veterinary behavior professional if anxiety-related behaviors are frequent, escalating, or interfering with daily life.",
        "This tool is for behavioral decision support only and does not replace professional medical or behavioral evaluation."
    ]


def build_reasoning(inputs, image_available, completeness):
    reasons = []

    if image_available == "yes":
        reasons.append("AI-assisted image cues were included, so the prediction uses both owner context and visible body-language signals.")
    else:
        reasons.append("No image was used in this run. The interpretation is based on profile and behavior context only.")

    if inputs["behavior"] == "panting":
        reasons.append("Panting can mean recovery, heat, anxiety, or overstimulation, so context is important.")
    if inputs["behavior"] == "pacing":
        reasons.append("Pacing can reflect anxiety, boredom, anticipation, or excess arousal.")
    if inputs["behavior"] == "whining":
        reasons.append("Whining can signal stress, attention-seeking, unmet needs, or discomfort.")
    if inputs["behavior"] == "hiding":
        reasons.append("Hiding can be associated with fear, stress, avoidance, or a need for space.")
    if inputs["behavior"] == "toy-seeking":
        reasons.append("Toy-seeking often suggests desire for interaction, enrichment, or play.")

    if inputs["visual_mouth"] == "open mouth / panting":
        reasons.append("The visual cue indicates an open mouth or visible panting.")
    if inputs["visual_posture"] in ["tense", "crouched"]:
        reasons.append(f"Body posture was marked as {inputs['visual_posture']}, which can increase concern.")
    if inputs["visual_ears"] == "back/pinned":
        reasons.append("Pinned-back ears can be a stress or uncertainty signal.")
    if inputs["visual_tail"] == "tucked":
        reasons.append("A tucked tail can indicate fear, anxiety, or discomfort.")
    if inputs["visual_eyes"] == "wide-eyed / whale eye":
        reasons.append("Wide-eyed expression or whale eye can be associated with stress.")
    if inputs["visual_hiding"] == "yes":
        reasons.append("Visible hiding or avoidance increases concern for stress-related behavior.")

    if inputs["activity"] == "high":
        reasons.append("Recent high activity increases likelihood of recovery or overstimulation.")
    if inputs["activity"] == "none":
        reasons.append("No recent activity makes recovery less likely and increases relevance of boredom or stress.")
    if inputs["environment"] == "warm":
        reasons.append("Warm conditions increase likelihood that panting is related to cooling or thermoregulation.")
    if inputs["environment"] == "noisy":
        reasons.append("Noise can increase stress sensitivity and anxiety-related behavior.")
    if inputs["duration"] == "long":
        reasons.append("Long duration raises concern because the behavior is less likely to be a brief normal response.")
    if inputs["duration"] == "short":
        reasons.append("Short duration lowers concern and may indicate a temporary state.")

    reasons.append(f"Interpretation quality is {quality_label(completeness)} based on available visual and contextual cues.")
    return reasons


def factors(inputs):
    risk = []
    protective = []
    missing = []

    for key, label in [
        ("visual_mouth", "mouth / panting visibility"),
        ("visual_posture", "body posture"),
        ("visual_ears", "ear position"),
        ("visual_tail", "tail position"),
        ("visual_eyes", "eye expression"),
        ("visual_hiding", "hiding / avoidance visibility")
    ]:
        if inputs[key] == "unknown":
            missing.append(label)

    if inputs["behavior"] in ["pacing", "whining", "hiding"]:
        risk.append(f"Observed behavior: {inputs['behavior']}")
    if inputs["visual_posture"] in ["tense", "crouched"]:
        risk.append(f"Visual posture cue: {inputs['visual_posture']}")
    if inputs["visual_ears"] == "back/pinned":
        risk.append("Ears back/pinned")
    if inputs["visual_tail"] == "tucked":
        risk.append("Tail tucked")
    if inputs["visual_eyes"] == "wide-eyed / whale eye":
        risk.append("Wide-eyed / whale eye")
    if inputs["visual_hiding"] == "yes":
        risk.append("Hiding or avoidance visible")
    if inputs["environment"] == "noisy":
        risk.append("Noisy environment")
    if inputs["duration"] == "long":
        risk.append("Long duration")
    if inputs["sensitivity"] == "high":
        risk.append("High stress sensitivity")
    if inputs["age_group"] == "senior":
        risk.append("Senior age group")

    if inputs["activity"] == "high" and (
        inputs["behavior"] == "panting" or inputs["visual_mouth"] == "open mouth / panting"
    ):
        protective.append("Panting after high activity can indicate recovery")
    if inputs["duration"] == "short":
        protective.append("Short duration")
    if inputs["visual_posture"] == "relaxed":
        protective.append("Relaxed posture")
    if inputs["visual_ears"] == "neutral":
        protective.append("Neutral ears")
    if inputs["visual_tail"] == "relaxed":
        protective.append("Relaxed tail")
    if inputs["visual_eyes"] == "relaxed":
        protective.append("Relaxed eyes")
    if inputs["sensitivity"] in ["low", "moderate"]:
        protective.append(f"{inputs['sensitivity'].title()} stress sensitivity")

    if not risk:
        risk.append("No major risk factors identified")
    if not protective:
        protective.append("Limited protective signals provided")

    return risk, protective, missing


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("Prototype Model Metrics")
    st.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    st.metric("Weighted Precision", f"{metrics['precision']:.0%}")
    st.metric("Weighted Recall", f"{metrics['recall']:.0%}")

    with st.expander("How to read this result", expanded=True):
        current_image_status = None

        if st.session_state.get("prediction_result") is not None:
            current_image_status = st.session_state.prediction_result["inputs"].get("image_available")

        if current_image_status == "yes":
            st.success(
                "Image cues were used in this interpretation. The model considered AI-extracted body-language cues "
                "such as mouth position, posture, ears, tail, eyes, and hiding/avoidance."
            )
        elif current_image_status == "no":
            st.warning(
                "No image was used in this run. This result is based on your dog’s profile and behavior context only. "
                "Because no dog image was uploaded, the system could not use visible body-language cues like posture, "
                "ears, tail, or eye expression. Adding a dog image may improve interpretation quality."
            )
        else:
            st.info(
                "Complete the interpreter to see whether the final result uses AI-extracted image cues or context only."
            )

    with st.expander("Data transparency"):
        st.write(
            """
            This prototype uses synthetic labeled examples to demonstrate the AI workflow.
            When a dog image is uploaded, a vision model validates whether the image contains a dog and pre-fills visible cues.
            The user can review and override those cues before generating the final interpretation.
            Feedback is logged for future retraining, but the model does not automatically retrain after each submission.
            """
        )


# ==================================================
# HEADER
# ==================================================

st.title("🐶 Dog Behavior Interpreter")
st.write(
    "A mobile-first AI prototype that helps owners interpret ambiguous dog behavior using profile, context, and AI-extracted image cues."
)

st.markdown("""
<div class="disclaimer-card">
⚠️ <b>Important:</b> This prototype provides behavioral guidance only. It is <b>not a medical or diagnostic tool</b>.
Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress,
or symptoms that are unusual, persistent, or worsening.
</div>
""", unsafe_allow_html=True)

steps = ["Profile", "Context", "Visual Cues", "Interpretation", "Feedback"]

step_html = "<div class='stepper'>"
for i, step in enumerate(steps, start=1):
    if i < st.session_state.phase:
        cls = "step-complete"
    elif i == st.session_state.phase:
        cls = "step-active"
    else:
        cls = "step-inactive"
    step_html += f"<span class='{cls}'>{i}. {step}</span>"
step_html += "</div>"
st.markdown(step_html, unsafe_allow_html=True)


# ==================================================
# PHASE 1
# ==================================================

if st.session_state.phase == 1:
    st.markdown("## Phase 1 — Dog Profile")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.session_state.dog_name = st.text_input("Dog name", value=st.session_state.get("dog_name", "Sadie"), help="Used to personalize the output only.")
        st.session_state.age_group = st.selectbox("Age group", ["puppy", "adult", "senior"], index=["puppy", "adult", "senior"].index(st.session_state.get("age_group", "adult")), help="Age can change how cautious the system should be about persistent behavior.")

    with c2:
        st.session_state.size = st.selectbox("Size category", ["small", "medium", "large"], index=["small", "medium", "large"].index(st.session_state.get("size", "medium")), help="Size is included as a general dog profile signal.")
        st.session_state.energy = st.selectbox("Typical energy level", ["low", "moderate", "high"], index=["low", "moderate", "high"].index(st.session_state.get("energy", "high")), help="High-energy dogs may show pacing or toy-seeking when stimulation needs are unmet.")

    with c3:
        st.session_state.sensitivity = st.selectbox("Known stress sensitivity", ["low", "moderate", "high"], index=["low", "moderate", "high"].index(st.session_state.get("sensitivity", "moderate")), help="Use this if your dog is reactive to noise, strangers, separation, or new environments.")
        st.session_state.assumption = st.selectbox("Your initial assumption", ["unsure", "anxiety", "boredom", "overstimulation", "recovery"], index=["unsure", "anxiety", "boredom", "overstimulation", "recovery"].index(st.session_state.get("assumption", "unsure")), help="This becomes a baseline so the prototype can compare owner interpretation to AI interpretation.")

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Next: Context", type="primary", on_click=set_phase, args=(2,), use_container_width=True)


# ==================================================
# PHASE 2
# ==================================================

elif st.session_state.phase == 2:
    st.markdown("## Phase 2 — Current Behavior Context")

    c1, c2 = st.columns(2)

    with c1:
        st.session_state.behavior = st.selectbox("Observed behavior", ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"], index=["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"].index(st.session_state.get("behavior", "panting")), help="Choose the main behavior you are trying to interpret.")
        st.session_state.activity = st.selectbox("Recent activity", ["none", "low", "high"], index=["none", "low", "high"].index(st.session_state.get("activity", "none")), help="Recent activity helps distinguish recovery from anxiety or overstimulation.")

    with c2:
        st.session_state.environment = st.selectbox("Environment", ["indoor", "outdoor", "warm", "noisy"], index=["indoor", "outdoor", "warm", "noisy"].index(st.session_state.get("environment", "indoor")), help="Context matters: heat, noise, or unfamiliar settings can change interpretation.")
        st.session_state.duration = st.selectbox("Duration", ["short", "medium", "long"], index=["short", "medium", "long"].index(st.session_state.get("duration", "short")), help="Longer duration raises concern because the behavior is less likely to be a brief normal response.")

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Back", on_click=set_phase, args=(1,), use_container_width=True)
    with nav2:
        st.button("Next: Visual Cues", type="primary", on_click=set_phase, args=(3,), use_container_width=True)


# ==================================================
# PHASE 3
# ==================================================

elif st.session_state.phase == 3:
    st.markdown("## Phase 3 — AI Image Cue Review")

    st.write(
        "Upload a dog image if available. The app will use AI to validate the image and pre-fill visible body-language cues. You can review and override the selections before analysis."
    )

    image = st.file_uploader(
        "Upload dog image",
        type=["jpg", "jpeg", "png"],
        help="Only dog images are accepted. The image is analyzed to pre-fill visual cues such as posture, ears, tail, eyes, and panting visibility."
    )

    invalid_image = False

    if image:
        st.image(image, caption="Uploaded image for AI cue extraction", width=350)

        if st.button("Analyze Image Cues with AI", type="secondary"):
            with st.spinner("Analyzing image cues..."):
                image_result = analyze_image_with_ai(image)

            if "error" in image_result:
                st.error(image_result["error"])
                st.session_state.image_available = "no"

            elif not image_result.get("image_valid", False):
                st.error("Invalid image for this prototype. Please upload an image that clearly shows a dog.")
                st.write(image_result.get("reason", "The image could not be validated as a dog image."))
                st.session_state.image_available = "no"
                invalid_image = True

            else:
                st.success("Dog image validated. Visual cues were auto-filled.")
                st.write(image_result.get("reason", ""))

                st.session_state.image_available = "yes"
                st.session_state.visual_mouth = image_result.get("visual_mouth", "unknown")
                st.session_state.visual_posture = image_result.get("visual_posture", "unknown")
                st.session_state.visual_ears = image_result.get("visual_ears", "unknown")
                st.session_state.visual_tail = image_result.get("visual_tail", "unknown")
                st.session_state.visual_eyes = image_result.get("visual_eyes", "unknown")
                st.session_state.visual_hiding = image_result.get("visual_hiding", "unknown")
                st.session_state.image_ai_confidence = image_result.get("confidence", None)
                st.session_state.image_ai_reason = image_result.get("reason", None)

                st.rerun()

    else:
        st.session_state.image_available = "no"

    if st.session_state.image_available == "yes":
        st.success("Image cues are included in this run.")
        if st.session_state.image_ai_confidence is not None:
            st.write(f"**Image analysis confidence:** {st.session_state.image_ai_confidence:.0%}")
        if st.session_state.image_ai_reason:
            st.write(f"**Image analysis note:** {st.session_state.image_ai_reason}")
    else:
        st.info("No valid dog image is currently included. You can continue using profile and behavior context only.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.session_state.visual_mouth = st.selectbox("Mouth / panting visible", ["unknown", "closed mouth", "open mouth / panting"], index=["unknown", "closed mouth", "open mouth / panting"].index(st.session_state.visual_mouth), help="Auto-filled by image AI when available; you can override it.")
        st.session_state.visual_posture = st.selectbox("Body posture", ["unknown", "relaxed", "alert", "tense", "crouched"], index=["unknown", "relaxed", "alert", "tense", "crouched"].index(st.session_state.visual_posture), help="Auto-filled by image AI when available; you can override it.")

    with c2:
        st.session_state.visual_ears = st.selectbox("Ear position", ["unknown", "neutral", "back/pinned"], index=["unknown", "neutral", "back/pinned"].index(st.session_state.visual_ears), help="Auto-filled by image AI when available; you can override it.")
        st.session_state.visual_tail = st.selectbox("Tail position", ["unknown", "relaxed", "tucked", "high"], index=["unknown", "relaxed", "tucked", "high"].index(st.session_state.visual_tail), help="Auto-filled by image AI when available; you can override it.")

    with c3:
        st.session_state.visual_eyes = st.selectbox("Eye expression", ["unknown", "relaxed", "wide-eyed / whale eye"], index=["unknown", "relaxed", "wide-eyed / whale eye"].index(st.session_state.visual_eyes), help="Auto-filled by image AI when available; you can override it.")
        st.session_state.visual_hiding = st.selectbox("Hiding / avoidance visible", ["unknown", "yes", "no"], index=["unknown", "yes", "no"].index(st.session_state.visual_hiding), help="Auto-filled by image AI when available; you can override it.")

    visuals = [
        st.session_state.visual_mouth,
        st.session_state.visual_posture,
        st.session_state.visual_ears,
        st.session_state.visual_tail,
        st.session_state.visual_eyes,
        st.session_state.visual_hiding
    ]

    completeness = visual_cue_completeness(st.session_state.image_available, *visuals)

    st.markdown("### Input Quality Preview")
    st.write(f"**Dog image used:** {'Yes' if st.session_state.image_available == 'yes' else 'No'}")
    st.write(f"**Visual cue completeness:** {quality_label(completeness)}")
    st.progress(completeness)

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Back", on_click=set_phase, args=(2,), use_container_width=True)

    with nav2:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True, disabled=invalid_image)

    if analyze_clicked:
        inputs = {
            "behavior": st.session_state.behavior,
            "activity": st.session_state.activity,
            "environment": st.session_state.environment,
            "duration": st.session_state.duration,
            "assumption": st.session_state.assumption,
            "age_group": st.session_state.age_group,
            "size": st.session_state.size,
            "energy": st.session_state.energy,
            "sensitivity": st.session_state.sensitivity,
            "image_available": st.session_state.image_available,
            "visual_mouth": st.session_state.visual_mouth,
            "visual_posture": st.session_state.visual_posture,
            "visual_ears": st.session_state.visual_ears,
            "visual_tail": st.session_state.visual_tail,
            "visual_eyes": st.session_state.visual_eyes,
            "visual_hiding": st.session_state.visual_hiding
        }

        input_df = pd.DataFrame([inputs])
        pred = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        classes = model.named_steps["rf"].classes_

        prob_df = pd.DataFrame({"Behavior State": classes, "Probability": probs}).sort_values("Probability", ascending=False)

        base_prob = float(prob_df.iloc[0]["Probability"])
        adjusted_prob = confidence_adjustment(base_prob, completeness)
        second_state = prob_df.iloc[1]["Behavior State"]
        second_prob = float(prob_df.iloc[1]["Probability"])

        st.session_state.prediction_result = {
            "inputs": inputs,
            "pred": pred,
            "prob_df": prob_df,
            "base_prob": base_prob,
            "adjusted_prob": adjusted_prob,
            "second_state": second_state,
            "second_prob": second_prob,
            "completeness": completeness
        }

        set_phase(4)
        st.rerun()


# ==================================================
# PHASE 4
# ==================================================

elif st.session_state.phase == 4:
    st.markdown("## Phase 4 — AI Interpretation")

    result = st.session_state.prediction_result

    if result is None:
        st.warning("No analysis has been generated yet.")
        st.button("Go to Visual Cues", on_click=set_phase, args=(3,))
    else:
        inputs = result["inputs"]
        pred = result["pred"]
        prob_df = result["prob_df"]
        adjusted_prob = result["adjusted_prob"]
        second_state = result["second_state"]
        second_prob = result["second_prob"]
        completeness = result["completeness"]

        risk, protective, missing = factors(inputs)
        rec = recommendation_for(pred)

        context_only_inputs = inputs.copy()
        context_only_inputs["image_available"] = "no"
        context_only_inputs["visual_mouth"] = "unknown"
        context_only_inputs["visual_posture"] = "unknown"
        context_only_inputs["visual_ears"] = "unknown"
        context_only_inputs["visual_tail"] = "unknown"
        context_only_inputs["visual_eyes"] = "unknown"
        context_only_inputs["visual_hiding"] = "unknown"

        context_only_df = pd.DataFrame([context_only_inputs])
        context_only_pred = model.predict(context_only_df)[0]
        context_only_probs = model.predict_proba(context_only_df)[0]
        context_only_classes = model.named_steps["rf"].classes_

        context_only_prob_df = pd.DataFrame({"Behavior State": context_only_classes, "Probability": context_only_probs}).sort_values("Probability", ascending=False)
        context_only_confidence = float(context_only_prob_df.iloc[0]["Probability"])

        r1, r2, r3 = st.columns([1.1, 0.9, 0.9])

        with r1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"### {confidence_color(adjusted_prob)} Likely state: **{pred.title()}**")
            st.write(f"**Dog:** {st.session_state.dog_name}")
            st.write(f"**Confidence:** {adjusted_prob:.0%} — {confidence_label(adjusted_prob)}")
            st.progress(adjusted_prob)
            st.write(f"**Secondary possibility:** {second_state.title()} ({second_prob:.0%})")
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='kpi'>", unsafe_allow_html=True)
            st.markdown("### Interpretation Quality")
            st.write(f"**Input completeness:** {quality_label(completeness)}")
            st.write(f"**Image cues used:** {'Yes' if inputs['image_available'] == 'yes' else 'No'}")
            st.write(f"**Missing visual cues:** {len(missing)}")
            st.progress(completeness)
            st.markdown("</div>", unsafe_allow_html=True)

        with r3:
            st.markdown("<div class='education-box'>", unsafe_allow_html=True)
            st.markdown("### Learning Takeaway")
            if pred == "recovery":
                st.write("Panting after activity or warmth may reflect recovery rather than anxiety.")
            elif pred == "anxiety":
                st.write("Stress interpretation becomes stronger when behavior is paired with tense body-language cues or environmental triggers.")
            elif pred == "boredom":
                st.write("Boredom is more likely when active behavior appears without recent stimulation or stress signals.")
            elif pred == "overstimulation":
                st.write("High arousal after activity can look like distress, but may respond best to decompression.")
            elif pred == "neutral":
                st.write("Not every behavior needs intervention. Relaxed cues and short duration can lower concern.")
            else:
                st.write("Unclear signals should be observed over time rather than treated as a definitive conclusion.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Why This Outcome Changed")
        if inputs["image_available"] == "yes":
            st.success("A dog image was used in this run. The model considered AI-extracted visual cues along with behavior context.")
        else:
            st.warning(
                "No image was used in this run. The interpretation is based on your dog’s profile and behavior context only. "
                "Because no dog image was uploaded, the system could not use visible body-language cues such as posture, ears, tail, or eye expression."
            )

        for reason in build_reasoning(inputs, inputs["image_available"], completeness):
            st.write(f"- {reason}")

        st.markdown("### Image Cue Impact Comparison")

        compare_col1, compare_col2 = st.columns(2)

        with compare_col1:
            st.markdown("<div class='kpi'>", unsafe_allow_html=True)
            st.markdown("#### Context Only")
            st.write("Profile + behavior context")
            st.write(f"**Prediction:** {context_only_pred.title()}")
            st.write(f"**Confidence:** {context_only_confidence:.0%}")
            st.markdown("</div>", unsafe_allow_html=True)

        with compare_col2:
            st.markdown("<div class='kpi'>", unsafe_allow_html=True)
            st.markdown("#### Context + Dog Image Cues")
            st.write("Profile + context + AI-extracted visual cues")
            st.write(f"**Prediction:** {pred.title()}")
            st.write(f"**Confidence:** {adjusted_prob:.0%}")
            st.markdown("</div>", unsafe_allow_html=True)

        if inputs["image_available"] == "yes":
            if context_only_pred != pred:
                st.success(
                    "The dog image cues changed the predicted state. This shows that visible body-language inputs are contributing to the outcome."
                )
            else:
                st.info(
                    "The dog image cues did not change the top predicted state, but they may still affect confidence and explanation quality."
                )
        else:
            st.warning(
                "Because no dog image was used, both outputs rely mostly on structured context. Uploading a dog image and extracting visible cues can make the interpretation more specific."
            )

        st.markdown("### Clear Action Plan")
        a1, a2, a3 = st.columns(3)

        with a1:
            st.markdown("<div class='action-box'>", unsafe_allow_html=True)
            st.markdown("#### Do Now")
            st.write(rec["summary"])
            for step in rec["do_now"]:
                st.write(f"- {step}")
            st.markdown("</div>", unsafe_allow_html=True)

        with a2:
            st.markdown("<div class='action-box'>", unsafe_allow_html=True)
            st.markdown("#### Observe Next")
            for step in rec["observe"]:
                st.write(f"- {step}")
            st.markdown("</div>", unsafe_allow_html=True)

        with a3:
            st.markdown("<div class='action-box'>", unsafe_allow_html=True)
            st.markdown("#### Avoid")
            for step in rec["avoid"]:
                st.write(f"- {step}")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Risk, protective, and missing signals"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("#### Risk Factors")
                for item in risk:
                    st.write(f"- {item}")
            with c2:
                st.markdown("#### Protective Factors")
                for item in protective:
                    st.write(f"- {item}")
            with c3:
                st.markdown("#### Missing Signals")
                if missing:
                    for item in missing:
                        st.write(f"- {item}")
                else:
                    st.write("- No major visual cues missing")

        st.markdown("### Owner Assumption Comparison")
        if inputs["assumption"] == "unsure":
            st.info("No owner assumption was provided, so the system cannot compare against a baseline interpretation.")
        elif inputs["assumption"] == pred:
            st.success(f"The AI interpretation aligns with your initial assumption of **{inputs['assumption']}**.")
        else:
            st.warning(
                f"You assumed **{inputs['assumption']}**, while the model predicted **{pred}**. "
                "This may represent a potential interpretation correction."
            )

        with st.expander("Probability distribution"):
            st.bar_chart(prob_df.set_index("Behavior State"))

        with st.expander("When to seek professional help"):
            for item in escalation_guidance():
                st.write(f"- {item}")

        nav1, nav2, spacer = st.columns([1, 1, 3])
        with nav1:
            st.button("Back", on_click=set_phase, args=(3,), use_container_width=True)
        with nav2:
            st.button("Continue to Feedback", type="primary", on_click=set_phase, args=(5,), use_container_width=True)


# ==================================================
# PHASE 5
# ==================================================

elif st.session_state.phase == 5:
    st.markdown("## Phase 5 — Feedback")
    st.write("Feedback helps demonstrate how the prototype could learn from real-world outcomes. You can also start a new interpretation without submitting feedback.")

    result = st.session_state.prediction_result

    if result is None:
        st.warning("No prediction available.")
        st.button("Start Over", on_click=set_phase, args=(1,))
    else:
        inputs = result["inputs"]
        pred = result["pred"]
        adjusted_prob = result["adjusted_prob"]
        second_state = result["second_state"]
        second_prob = result["second_prob"]
        completeness = result["completeness"]

        feedback_count = get_feedback_count()

        st.markdown("### Learning Loop Status")
        ll1, ll2, ll3 = st.columns(3)

        with ll1:
            st.metric("Feedback records collected", feedback_count)

        with ll2:
            st.metric("Current training source", "Synthetic data")

        with ll3:
            st.metric("Retraining mode", "Manual / future batch")

        st.info(
            "This prototype logs feedback for future dataset improvement. "
            "It does not automatically retrain the model after each feedback submission, "
            "because production AI systems typically retrain after enough validated feedback has been collected."
        )

        with st.form("feedback_form"):
            fb1, fb2, fb3 = st.columns(3)

            with fb1:
                accurate = st.radio("Was this accurate?", ["Yes", "No", "Unsure"], horizontal=True)

            with fb2:
                action_taken = st.radio("Did you take the action?", ["Yes", "No", "Partially"], horizontal=True)

            with fb3:
                improved = st.radio("Did behavior improve?", ["Yes", "No", "Not sure"], horizontal=True)

            feedback_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

        if feedback_submitted:
            feedback_row = {
                "timestamp": datetime.now().isoformat(),
                "dog_name": st.session_state.dog_name,
                **inputs,
                "prediction": pred,
                "confidence": round(adjusted_prob, 3),
                "secondary_state": second_state,
                "secondary_probability": round(second_prob, 3),
                "input_completeness": round(completeness, 3),
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

            st.session_state.feedback_done = True
            st.success("Feedback saved for future dataset improvement and retraining.")

            with st.expander("Feedback record"):
                st.dataframe(feedback_df)

        nav1, nav2, spacer = st.columns([1, 1, 3])
        with nav1:
            st.button("Back", on_click=set_phase, args=(4,), use_container_width=True)
        with nav2:
            if st.button("Start New", use_container_width=True):
                for key in [
                    "prediction_result",
                    "feedback_done",
                    "phase",
                    "image_available",
                    "visual_mouth",
                    "visual_posture",
                    "visual_ears",
                    "visual_tail",
                    "visual_eyes",
                    "visual_hiding",
                    "image_ai_confidence",
                    "image_ai_reason"
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
