import os, json, base64, tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="Dog Behavior Interpreter", page_icon="🐶", layout="wide")

# ==================================================
# STYLE
# ==================================================

st.markdown("""
<style>
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1180px;
    background-color: #FAF7F2;
}

html, body, [class*="css"] {
    color: #2F2A25;
}

.disclaimer-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #D9A441;
    background-color: #FFF4D8;
    color: #5C3B00;
    margin-bottom: 1rem;
    font-size: 0.92rem;
}

.stepper {
    display: flex;
    gap: 10px;
    margin: 12px 0 20px 0;
    flex-wrap: wrap;
}

.step-active,
.step-complete,
.step-inactive {
    width: 130px;
    min-height: 42px;
    padding: 0.55rem 0.7rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    white-space: nowrap;
    font-weight: 700;
    font-size: 0.88rem;
}

.step-active {
    background: #7A9E7E;
    color: white;
}

.step-complete {
    background: #DCEBD8;
    color: #355E3B;
}

.step-inactive {
    background: #EFE8DD;
    color: #7A6A58;
}

.result-card {
    padding: 1.4rem;
    border-radius: 22px;
    border: 1px solid #BFD8BD;
    background-color: #EEF6EA;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(76, 97, 70, 0.08);
}

.action-hero {
    padding: 1.4rem;
    border-radius: 22px;
    border: 1px solid #B7D4D1;
    background-color: #EAF5F3;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(65, 99, 96, 0.08);
}

.quality-card, .kpi, .section-card {
    padding: 1.2rem;
    border-radius: 20px;
    border: 1px solid #D8C8B2;
    background-color: #FFFCF7;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(82, 67, 50, 0.05);
}

.warning-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #D9A441;
    background-color: #FFF4D8;
    margin-bottom: 1rem;
}

.escalation-card {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #D9A19A;
    background-color: #FBEDEA;
    margin-bottom: 0.75rem;
}

.action-box {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #B7D4D1;
    background-color: #EFF8F6;
    margin-bottom: 0.75rem;
}

.education-box {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #CFC3A8;
    background-color: #F8F1E5;
    margin-bottom: 0.75rem;
}

.ai-box {
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #AFC9C3;
    background-color: #EAF5F3;
    margin-bottom: 0.75rem;
}

button[kind="primary"] {
    background-color: #7A9E7E !important;
    border-color: #7A9E7E !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    background-color: #EFE8DD;
    padding: 8px 14px;
}

.stTabs [aria-selected="true"] {
    background-color: #DCEBD8;
    color: #355E3B;
    font-weight: 700;
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

defaults = {
    "visual_mouth": "unknown",
    "visual_posture": "unknown",
    "visual_ears": "unknown",
    "visual_tail": "unknown",
    "visual_eyes": "unknown",
    "visual_hiding": "unknown",
    "image_available": "no",
    "video_available": "no",
    "media_type": "none",
    "image_ai_confidence": None,
    "image_ai_reason": None,
    "movement_pattern": "unknown",
    "movement_level": "unknown",
    "behavior_change": "unknown",
    "behavior_continuity": "unknown",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

CLASSES = ["anxiety", "boredom", "overstimulation", "recovery", "neutral", "needs observation"]

# ==================================================
# HYBRID PREDICTION LOGIC
# ==================================================

def evidence_scores(inputs):
    scores = {c: 0.2 for c in CLASSES}

    behavior = inputs["behavior"]
    activity = inputs["activity"]
    environment = inputs["environment"]
    duration = inputs["duration"]
    age = inputs["age_group"]
    energy = inputs["energy"]
    sensitivity = inputs["sensitivity"]

    mouth = inputs["visual_mouth"]
    posture = inputs["visual_posture"]
    ears = inputs["visual_ears"]
    tail = inputs["visual_tail"]
    eyes = inputs["visual_eyes"]
    hiding = inputs["visual_hiding"]

    movement_pattern = inputs.get("movement_pattern", "unknown")
    movement_level = inputs.get("movement_level", "unknown")
    behavior_change = inputs.get("behavior_change", "unknown")
    behavior_continuity = inputs.get("behavior_continuity", "unknown")

    stress_cues = 0
    relaxed_cues = 0

    if posture in ["tense", "crouched"]:
        scores["anxiety"] += 2.2
        scores["needs observation"] += 0.8
        stress_cues += 1

    if ears == "back/pinned":
        scores["anxiety"] += 1.8
        stress_cues += 1

    if tail == "tucked":
        scores["anxiety"] += 2.0
        scores["needs observation"] += 0.6
        stress_cues += 1

    if eyes == "wide-eyed / whale eye":
        scores["anxiety"] += 2.4
        stress_cues += 1

    if hiding == "yes":
        scores["anxiety"] += 2.3
        scores["needs observation"] += 0.7
        stress_cues += 1

    if posture == "relaxed":
        scores["neutral"] += 1.7
        scores["recovery"] += 0.8
        relaxed_cues += 1

    if ears == "neutral":
        scores["neutral"] += 0.8
        relaxed_cues += 1

    if tail == "relaxed":
        scores["neutral"] += 1.0
        relaxed_cues += 1

    if eyes == "relaxed":
        scores["neutral"] += 1.0
        relaxed_cues += 1

    if mouth == "closed mouth":
        scores["neutral"] += 0.9
        scores["anxiety"] -= 0.4

    if mouth == "open mouth / panting":
        scores["recovery"] += 0.9
        scores["overstimulation"] += 0.6

    if behavior == "panting":
        scores["recovery"] += 1.0
        scores["overstimulation"] += 0.7
        if activity == "none" and duration == "long":
            scores["needs observation"] += 1.2

    if behavior == "pacing":
        scores["anxiety"] += 1.1
        scores["overstimulation"] += 0.8
        scores["boredom"] += 0.7

    if behavior == "whining":
        scores["anxiety"] += 1.4
        scores["needs observation"] += 0.5

    if behavior == "hiding":
        scores["anxiety"] += 2.0
        scores["needs observation"] += 0.7

    if behavior == "toy-seeking":
        scores["boredom"] += 2.2
        scores["neutral"] += 0.5

    if behavior == "resting":
        scores["neutral"] += 2.2
        scores["anxiety"] -= 0.7
        scores["overstimulation"] -= 0.5

    if activity == "high":
        scores["recovery"] += 1.3
        scores["overstimulation"] += 1.3
        if stress_cues >= 2:
            scores["anxiety"] += 0.8

    if activity == "low":
        scores["neutral"] += 0.4
        scores["boredom"] += 0.4

    if activity == "none":
        scores["boredom"] += 0.8
        scores["recovery"] -= 0.8
        if behavior in ["pacing", "toy-seeking"]:
            scores["boredom"] += 1.2

    if environment == "warm":
        scores["recovery"] += 1.2
        if mouth == "open mouth / panting" or behavior == "panting":
            scores["recovery"] += 1.0

    if environment == "noisy":
        scores["anxiety"] += 1.4
        scores["overstimulation"] += 0.5

    if environment in ["indoor", "outdoor"] and stress_cues == 0:
        scores["neutral"] += 0.5

    if duration == "short":
        scores["neutral"] += 0.5
        scores["recovery"] += 0.4
        scores["needs observation"] -= 0.6

    if duration == "medium":
        scores["overstimulation"] += 0.3
        scores["recovery"] += 0.3

    if duration == "long":
        scores["anxiety"] += 0.8
        scores["needs observation"] += 1.4
        scores["neutral"] -= 0.7

    if sensitivity == "high":
        scores["anxiety"] += 0.9
    elif sensitivity == "low":
        scores["anxiety"] -= 0.5
        scores["neutral"] += 0.3

    if energy == "high":
        scores["boredom"] += 0.5
        scores["overstimulation"] += 0.5
    elif energy == "low":
        scores["neutral"] += 0.4
        scores["overstimulation"] -= 0.5

    if age == "senior" and duration == "long":
        scores["needs observation"] += 1.0

    # Video / temporal evidence
    if movement_pattern == "repetitive pacing":
        scores["anxiety"] += 1.4
        scores["overstimulation"] += 0.8
        scores["boredom"] += 0.5

    if movement_pattern == "brief movement":
        scores["neutral"] += 0.5

    if movement_pattern == "settling down":
        scores["recovery"] += 1.0
        scores["neutral"] += 0.8
        scores["anxiety"] -= 0.5

    if movement_level == "still / resting":
        scores["neutral"] += 1.0
        scores["overstimulation"] -= 0.6

    if movement_level == "moderate movement":
        scores["boredom"] += 0.4
        scores["neutral"] += 0.2

    if movement_level == "high movement":
        scores["overstimulation"] += 1.0
        scores["anxiety"] += 0.4

    if behavior_continuity == "brief / one-time":
        scores["neutral"] += 0.6
        scores["needs observation"] -= 0.5

    if behavior_continuity == "repeated":
        scores["anxiety"] += 0.7
        scores["boredom"] += 0.5

    if behavior_continuity == "continuous":
        scores["anxiety"] += 1.0
        scores["needs observation"] += 0.7

    if behavior_change == "improving / settling":
        scores["recovery"] += 1.0
        scores["neutral"] += 0.8
        scores["anxiety"] -= 0.6

    if behavior_change == "same / unchanged":
        scores["needs observation"] += 0.4

    if behavior_change == "worsening / escalating":
        scores["anxiety"] += 1.3
        scores["overstimulation"] += 0.9
        scores["needs observation"] += 1.0
        scores["neutral"] -= 0.8

    if relaxed_cues >= 3 and stress_cues == 0:
        scores["neutral"] += 2.0
        scores["anxiety"] -= 1.3
        scores["overstimulation"] -= 0.6

    if stress_cues >= 3:
        scores["anxiety"] += 2.0
        scores["neutral"] -= 1.2

    if stress_cues == 0 and behavior == "resting" and mouth == "closed mouth":
        scores["neutral"] += 1.8

    for key in scores:
        scores[key] = max(0.01, scores[key])

    total = sum(scores.values())
    probs = {k: v / total for k, v in scores.items()}
    return probs, scores


def hybrid_predict(inputs, ml_model):
    input_df = pd.DataFrame([inputs])

    ml_classes = list(ml_model.named_steps["rf"].classes_)
    ml_raw_probs = ml_model.predict_proba(input_df)[0]
    ml_probs = {c: 0.0 for c in CLASSES}

    for c, p in zip(ml_classes, ml_raw_probs):
        ml_probs[c] = float(p)

    evidence_probs, raw_scores = evidence_scores(inputs)

    final_probs = {}
    for c in CLASSES:
        final_probs[c] = (0.78 * evidence_probs[c]) + (0.22 * ml_probs.get(c, 0))

    total = sum(final_probs.values())
    final_probs = {k: v / total for k, v in final_probs.items()}

    prob_df = pd.DataFrame({
        "Behavior State": list(final_probs.keys()),
        "Probability": list(final_probs.values())
    }).sort_values("Probability", ascending=False)

    pred = prob_df.iloc[0]["Behavior State"]
    return pred, prob_df, evidence_probs, ml_probs, raw_scores


# ==================================================
# DATA + MODEL
# ==================================================

@st.cache_data
def generate_data(n=2600):
    np.random.seed(42)

    rows = []
    values = {
        "behavior": ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"],
        "activity": ["none", "low", "high"],
        "environment": ["indoor", "outdoor", "warm", "noisy"],
        "duration": ["short", "medium", "long"],
        "assumption": ["anxiety", "boredom", "overstimulation", "recovery", "unsure"],
        "age_group": ["puppy", "adult", "senior"],
        "size": ["small", "medium", "large"],
        "energy": ["low", "moderate", "high"],
        "sensitivity": ["low", "moderate", "high"],
        "image_available": ["yes", "no"],
        "video_available": ["yes", "no"],
        "media_type": ["none", "image", "video"],
        "visual_mouth": ["unknown", "closed mouth", "open mouth / panting"],
        "visual_posture": ["unknown", "relaxed", "alert", "tense", "crouched"],
        "visual_ears": ["unknown", "neutral", "back/pinned"],
        "visual_tail": ["unknown", "relaxed", "tucked", "high"],
        "visual_eyes": ["unknown", "relaxed", "wide-eyed / whale eye"],
        "visual_hiding": ["unknown", "yes", "no"],
        "movement_pattern": ["unknown", "brief movement", "repetitive pacing", "settling down"],
        "movement_level": ["unknown", "still / resting", "moderate movement", "high movement"],
        "behavior_change": ["unknown", "improving / settling", "same / unchanged", "worsening / escalating"],
        "behavior_continuity": ["unknown", "brief / one-time", "repeated", "continuous"],
    }

    for _ in range(n):
        row = {k: np.random.choice(v) for k, v in values.items()}
        probs, _ = evidence_scores(row)
        label = max(probs, key=probs.get)
        rows.append({**row, "label": label})

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
        ("rf", RandomForestClassifier(n_estimators=260, random_state=42, class_weight="balanced"))
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

Determine whether the image clearly shows a dog. If not, return image_valid=false.
If it shows a dog, infer visible body-language cues using only the allowed values.

Return valid JSON only:
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
                        {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_base64}"}
                    ]
                }
            ]
        )
        return json.loads(response.output_text)
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}


def extract_video_frames(video_file, max_frames=4):
    video_bytes = video_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name

    frames = []
    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return frames

    frame_indices = np.linspace(0, total_frames - 1, max_frames).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()

        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    try:
        os.remove(temp_path)
    except Exception:
        pass

    return frames


def visual_cue_completeness(image_available, video_available, *visuals):
    known = sum(v != "unknown" for v in visuals)
    base = known / len(visuals)

    if video_available == "yes":
        return min(1.0, base + 0.22)

    if image_available == "yes":
        return min(1.0, base + 0.15)

    return base * 0.75


def temporal_completeness(*temporal_fields):
    known = sum(v != "unknown" for v in temporal_fields)
    return known / len(temporal_fields)


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


def education_for(pred):
    return {
        "recovery": {
            "what_it_means": "Recovery means the behavior may be a normal response after activity, excitement, or warmth.",
            "common_cues": ["Panting after activity", "Short duration", "Relaxed or neutral body posture", "Settling over time"],
            "watch_for": ["Panting that does not reduce with rest", "Labored breathing", "Lethargy or collapse"]
        },
        "anxiety": {
            "what_it_means": "Anxiety means the dog may be responding to stress, uncertainty, fear, or environmental triggers.",
            "common_cues": ["Pacing", "Hiding", "Pinned ears", "Tucked tail", "Whale eye", "Repeated or continuous behavior"],
            "watch_for": ["Escalating avoidance", "Repeated inability to settle", "Frequent fear responses"]
        },
        "boredom": {
            "what_it_means": "Boredom means the dog may need mental stimulation, engagement, or structured activity.",
            "common_cues": ["Toy-seeking", "Restlessness", "Low recent activity", "Attention-seeking", "Repeated mild movement"],
            "watch_for": ["Demand behavior increasing", "Destructive behavior", "Difficulty settling"]
        },
        "overstimulation": {
            "what_it_means": "Overstimulation means the dog may be over-aroused and needs decompression rather than more activity.",
            "common_cues": ["Pacing after activity", "Panting after excitement", "High movement", "Inability to settle"],
            "watch_for": ["Jumping, mouthing, or frantic behavior", "Escalation after more play", "Difficulty calming down"]
        },
        "neutral": {
            "what_it_means": "Neutral means the available signals do not strongly suggest stress, boredom, overstimulation, or concern.",
            "common_cues": ["Relaxed posture", "Short duration", "Resting", "No obvious stressors", "Stillness or settling"],
            "watch_for": ["Sudden behavior changes", "Persistent unusual behavior", "Changes in appetite or responsiveness"]
        },
        "needs observation": {
            "what_it_means": "Needs observation means the signals are unclear or potentially elevated, so the safest response is monitoring and cautious escalation.",
            "common_cues": ["Long duration", "Senior dog", "Unclear cause", "Incomplete visual cues", "Worsening behavior"],
            "watch_for": ["Symptoms worsening", "Repeated episodes", "Any medical warning signs"]
        }
    }[pred]


def escalation_guidance():
    return [
        "Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress, or other emergency signs.",
        "Consult a veterinarian if symptoms persist, worsen, or are unusual for your dog.",
        "Consult a certified trainer or veterinary behavior professional if anxiety-related behaviors are frequent, escalating, or interfering with daily life.",
        "This tool is for behavioral decision support only and does not replace professional medical or behavioral evaluation."
    ]


def factors(inputs):
    risk, protective, missing = [], [], []

    for key, label in [
        ("visual_mouth", "mouth / panting visibility"),
        ("visual_posture", "body posture"),
        ("visual_ears", "ear position"),
        ("visual_tail", "tail position"),
        ("visual_eyes", "eye expression"),
        ("visual_hiding", "hiding / avoidance visibility"),
        ("movement_pattern", "movement pattern"),
        ("movement_level", "movement level"),
        ("behavior_change", "change over time"),
        ("behavior_continuity", "behavior continuity"),
    ]:
        if inputs.get(key, "unknown") == "unknown":
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
    if inputs.get("movement_pattern") == "repetitive pacing":
        risk.append("Repeated pacing pattern")
    if inputs.get("movement_level") == "high movement":
        risk.append("High movement level")
    if inputs.get("behavior_change") == "worsening / escalating":
        risk.append("Behavior worsening over time")
    if inputs.get("behavior_continuity") == "continuous":
        risk.append("Continuous behavior")

    if inputs["activity"] == "high" and (
        inputs["behavior"] == "panting" or inputs["visual_mouth"] == "open mouth / panting"
    ):
        protective.append("Panting after high activity can indicate recovery")
    if inputs["duration"] == "short":
        protective.append("Short duration")
    if inputs["visual_mouth"] == "closed mouth":
        protective.append("Closed mouth")
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
    if inputs.get("movement_pattern") == "settling down":
        protective.append("Dog appears to be settling down")
    if inputs.get("movement_level") == "still / resting":
        protective.append("Still or resting movement level")
    if inputs.get("behavior_change") == "improving / settling":
        protective.append("Behavior improving over time")

    if not risk:
        risk.append("No major risk factors identified")
    if not protective:
        protective.append("Limited protective signals provided")

    return risk, protective, missing


def build_reasoning(inputs, completeness, temporal_quality):
    reasons = []

    if inputs["media_type"] == "video":
        reasons.append("Video-assisted cues were included, so the interpretation considers both visible body-language signals and behavior over time.")
    elif inputs["media_type"] == "image":
        reasons.append("Image-assisted cues were included, so the interpretation considers visible body-language signals.")
    else:
        reasons.append("No media was used in this run. The interpretation is based on profile and behavior context only.")

    if inputs["visual_posture"] == "relaxed" and inputs["visual_eyes"] == "relaxed":
        reasons.append("Relaxed posture and relaxed eyes reduce concern for anxiety.")
    if inputs["visual_eyes"] == "wide-eyed / whale eye":
        reasons.append("Whale eye is a strong stress cue and increases concern for anxiety.")
    if inputs["visual_tail"] == "tucked" or inputs["visual_ears"] == "back/pinned":
        reasons.append("Tucked tail or pinned ears increase stress likelihood.")
    if inputs["visual_mouth"] == "closed mouth":
        reasons.append("Closed mouth can lower concern for panting-related recovery or overstimulation.")
    if inputs["activity"] == "high":
        reasons.append("Recent high activity increases likelihood of recovery or overstimulation.")
    if inputs["activity"] == "none":
        reasons.append("No recent activity makes recovery less likely and increases relevance of boredom or stress.")
    if inputs["duration"] == "long":
        reasons.append("Long duration raises concern because the behavior is less likely to be brief or incidental.")
    if inputs["duration"] == "short":
        reasons.append("Short duration lowers concern and may indicate a temporary state.")

    if inputs.get("movement_pattern") == "repetitive pacing":
        reasons.append("Repeated pacing over time increases concern for anxiety, boredom, or overstimulation.")
    if inputs.get("movement_pattern") == "settling down":
        reasons.append("Settling over time reduces concern and supports recovery or neutral interpretation.")
    if inputs.get("behavior_change") == "worsening / escalating":
        reasons.append("Worsening behavior increases concern and lowers confidence in a neutral explanation.")
    if inputs.get("behavior_change") == "improving / settling":
        reasons.append("Improvement over time supports recovery or neutral interpretation.")

    reasons.append(f"Visual interpretation quality is {quality_label(completeness)}.")
    reasons.append(f"Temporal interpretation quality is {quality_label(temporal_quality)}.")

    return reasons


def confidence_drivers(inputs, visual_quality, temporal_quality, adjusted_prob):
    drivers = []

    if inputs["media_type"] == "video":
        drivers.append("Video cues included")
    elif inputs["media_type"] == "image":
        drivers.append("Image cues included")
    else:
        drivers.append("No media cues included")

    drivers.append(f"Visual cue completeness: {quality_label(visual_quality)}")
    drivers.append(f"Temporal cue completeness: {quality_label(temporal_quality)}")

    if inputs["assumption"] != "unsure":
        drivers.append("Owner baseline assumption available")
    else:
        drivers.append("No owner baseline assumption")

    if adjusted_prob < 0.55:
        drivers.append("Low confidence: signals are mixed or incomplete")
    elif adjusted_prob >= 0.75:
        drivers.append("High confidence: signals are internally consistent")
    else:
        drivers.append("Moderate confidence: some ambiguity remains")

    return drivers


def what_would_change_prediction(inputs):
    suggestions = []

    if inputs["media_type"] == "none":
        suggestions.append("Uploading a clear dog image or short video could add body-language and movement cues.")
    if inputs["media_type"] != "video":
        suggestions.append("A short video could help determine whether the behavior is brief, repeated, continuous, improving, or worsening.")
    if inputs["visual_posture"] == "unknown":
        suggestions.append("Knowing whether posture is relaxed, tense, or crouched would help distinguish anxiety from recovery or neutrality.")
    if inputs["visual_eyes"] == "unknown":
        suggestions.append("Eye expression could strongly affect anxiety scoring, especially whale eye versus relaxed eyes.")
    if inputs["visual_tail"] == "unknown" or inputs["visual_ears"] == "unknown":
        suggestions.append("Ear and tail position could shift the interpretation toward or away from stress.")
    if inputs.get("behavior_change") == "unknown":
        suggestions.append("Knowing whether the behavior is improving or worsening would improve confidence.")
    if inputs["behavior"] == "panting":
        suggestions.append("If panting decreases after rest, recovery becomes more likely; if it persists without heat/activity, concern increases.")
    if inputs["behavior"] == "pacing":
        suggestions.append("If pacing decreases after enrichment, boredom becomes more likely; if it persists with stress cues, anxiety becomes more likely.")

    if not suggestions:
        suggestions.append("More follow-up feedback would improve future interpretation.")

    return suggestions


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("Prototype Model Metrics")
    st.metric("Accuracy", f"{metrics['accuracy']:.0%}")
    st.metric("Weighted Precision", f"{metrics['precision']:.0%}")
    st.metric("Weighted Recall", f"{metrics['recall']:.0%}")

    with st.expander("How AI is used", expanded=True):
        st.write(
            """
            This prototype uses a hybrid AI approach:
            
            1. A supervised ML classifier learns behavior patterns from synthetic labeled examples.
            2. A behavioral evidence scoring layer calibrates strong cues like whale eye, tucked tail, relaxed posture, activity, and movement over time.
            3. Optional image AI can validate dog images and pre-fill visual cues when API credits are available.
            4. Video-assisted review samples frames locally and lets the user review temporal cues like repeated pacing, settling, or escalation.
            
            The app is preventive behavioral decision support, not diagnosis.
            """
        )

    with st.expander("Data transparency"):
        st.write(
            """
            The current model uses synthetic training data and structured behavioral logic.
            Video processing samples representative frames locally and does not use API credits.
            Feedback is logged for future retraining but does not automatically retrain the model after each submission.
            """
        )


# ==================================================
# HEADER
# ==================================================

st.title("🐶 Dog Behavior Interpreter")
st.write(
    "A preventive AI decision-support prototype that helps owners understand ambiguous dog behavior using profile, context, and optional image/video cues."
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
        st.session_state.dog_name = st.text_input(
            "Dog name",
            value=st.session_state.get("dog_name", "Sadie"),
            help="Used to personalize the output only."
        )
        st.session_state.age_group = st.selectbox(
            "Age group",
            ["puppy", "adult", "senior"],
            index=["puppy", "adult", "senior"].index(st.session_state.get("age_group", "adult"))
        )

    with c2:
        st.session_state.size = st.selectbox(
            "Size category",
            ["small", "medium", "large"],
            index=["small", "medium", "large"].index(st.session_state.get("size", "medium"))
        )
        st.session_state.energy = st.selectbox(
            "Typical energy level",
            ["low", "moderate", "high"],
            index=["low", "moderate", "high"].index(st.session_state.get("energy", "high"))
        )

    with c3:
        st.session_state.sensitivity = st.selectbox(
            "Known stress sensitivity",
            ["low", "moderate", "high"],
            index=["low", "moderate", "high"].index(st.session_state.get("sensitivity", "moderate"))
        )
        st.session_state.assumption = st.selectbox(
            "Your initial assumption",
            ["unsure", "anxiety", "boredom", "overstimulation", "recovery"],
            index=["unsure", "anxiety", "boredom", "overstimulation", "recovery"].index(
                st.session_state.get("assumption", "unsure")
            )
        )

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
        st.session_state.behavior = st.selectbox(
            "Observed behavior",
            ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"],
            index=["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"].index(
                st.session_state.get("behavior", "panting")
            )
        )
        st.session_state.activity = st.selectbox(
            "Recent activity",
            ["none", "low", "high"],
            index=["none", "low", "high"].index(st.session_state.get("activity", "none"))
        )

    with c2:
        st.session_state.environment = st.selectbox(
            "Environment",
            ["indoor", "outdoor", "warm", "noisy"],
            index=["indoor", "outdoor", "warm", "noisy"].index(st.session_state.get("environment", "indoor"))
        )
        st.session_state.duration = st.selectbox(
            "Duration",
            ["short", "medium", "long"],
            index=["short", "medium", "long"].index(st.session_state.get("duration", "short"))
        )

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Back", on_click=set_phase, args=(1,), use_container_width=True)
    with nav2:
        st.button("Next: Visual Cues", type="primary", on_click=set_phase, args=(3,), use_container_width=True)


# ==================================================
# PHASE 3
# ==================================================

elif st.session_state.phase == 3:
    st.markdown("## Phase 3 — Image / Video Cue Review")

    st.write(
        "Upload a dog image or short video if available. Video adds temporal signals like repeated pacing, settling, or escalation over time."
    )

    media_choice = st.radio(
        "Input type",
        ["No media", "Image", "Video"],
        horizontal=True
    )

    image = None
    video = None
    invalid_media = False

    if media_choice == "Image":
        image = st.file_uploader("Upload dog image", type=["jpg", "jpeg", "png"])

        if image:
            st.image(image, caption="Uploaded image for AI cue extraction", width=350)
            st.session_state.media_type = "image"
            st.session_state.image_available = "yes"
            st.session_state.video_available = "no"

            if st.button("Analyze Image Cues with AI", type="secondary"):
                with st.spinner("Analyzing image cues..."):
                    image_result = analyze_image_with_ai(image)

                if "error" in image_result:
                    st.error(image_result["error"])
                    st.info("You can still continue by manually selecting cues.")
                    st.session_state.image_available = "no"
                    st.session_state.media_type = "none"

                elif not image_result.get("image_valid", False):
                    st.error("Invalid image. Please upload an image that clearly shows a dog.")
                    st.session_state.image_available = "no"
                    st.session_state.media_type = "none"
                    invalid_media = True

                else:
                    st.success("Dog image validated. Visual cues were auto-filled.")
                    st.write(image_result.get("reason", ""))
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
            st.session_state.media_type = "none"
            st.session_state.image_available = "no"
            st.session_state.video_available = "no"

    elif media_choice == "Video":
        video = st.file_uploader("Upload short dog video", type=["mp4", "mov", "avi", "m4v"])

        if video:
            st.video(video)
            st.session_state.media_type = "video"
            st.session_state.image_available = "no"
            st.session_state.video_available = "yes"

            frames = extract_video_frames(video, max_frames=4)

            if frames:
                st.markdown("### Sampled video frames")
                frame_cols = st.columns(len(frames))
                for i, frame in enumerate(frames):
                    with frame_cols[i]:
                        st.image(frame, caption=f"Frame {i + 1}", use_container_width=True)

                st.info(
                    "Video frames were sampled locally for review. This does not use API credits. "
                    "In this prototype, the video helps you review movement over time, but the temporal cues below are user-reviewed rather than automatically inferred by a vision model."
                )
            else:
                st.warning("Could not extract frames from this video. You can still manually select temporal cues.")
        else:
            st.session_state.media_type = "none"
            st.session_state.image_available = "no"
            st.session_state.video_available = "no"

    else:
        st.session_state.media_type = "none"
        st.session_state.image_available = "no"
        st.session_state.video_available = "no"

    st.markdown("### Visual cues")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.session_state.visual_mouth = st.selectbox(
            "Mouth / panting visible",
            ["unknown", "closed mouth", "open mouth / panting"],
            index=["unknown", "closed mouth", "open mouth / panting"].index(st.session_state.visual_mouth)
        )
        st.session_state.visual_posture = st.selectbox(
            "Body posture",
            ["unknown", "relaxed", "alert", "tense", "crouched"],
            index=["unknown", "relaxed", "alert", "tense", "crouched"].index(st.session_state.visual_posture)
        )

    with c2:
        st.session_state.visual_ears = st.selectbox(
            "Ear position",
            ["unknown", "neutral", "back/pinned"],
            index=["unknown", "neutral", "back/pinned"].index(st.session_state.visual_ears)
        )
        st.session_state.visual_tail = st.selectbox(
            "Tail position",
            ["unknown", "relaxed", "tucked", "high"],
            index=["unknown", "relaxed", "tucked", "high"].index(st.session_state.visual_tail)
        )

    with c3:
        st.session_state.visual_eyes = st.selectbox(
            "Eye expression",
            ["unknown", "relaxed", "wide-eyed / whale eye"],
            index=["unknown", "relaxed", "wide-eyed / whale eye"].index(st.session_state.visual_eyes)
        )
        st.session_state.visual_hiding = st.selectbox(
            "Hiding / avoidance visible",
            ["unknown", "yes", "no"],
            index=["unknown", "yes", "no"].index(st.session_state.visual_hiding)
        )

    st.markdown("### Temporal cues")
    t1, t2 = st.columns(2)

    with t1:
        st.session_state.movement_pattern = st.selectbox(
            "Movement pattern",
            ["unknown", "brief movement", "repetitive pacing", "settling down"],
            index=["unknown", "brief movement", "repetitive pacing", "settling down"].index(st.session_state.movement_pattern),
            help="Video is strongest when it shows whether behavior repeats or changes over time."
        )
        st.session_state.movement_level = st.selectbox(
            "Movement level",
            ["unknown", "still / resting", "moderate movement", "high movement"],
            index=["unknown", "still / resting", "moderate movement", "high movement"].index(st.session_state.movement_level)
        )

    with t2:
        st.session_state.behavior_change = st.selectbox(
            "Change over time",
            ["unknown", "improving / settling", "same / unchanged", "worsening / escalating"],
            index=["unknown", "improving / settling", "same / unchanged", "worsening / escalating"].index(st.session_state.behavior_change)
        )
        st.session_state.behavior_continuity = st.selectbox(
            "Behavior continuity",
            ["unknown", "brief / one-time", "repeated", "continuous"],
            index=["unknown", "brief / one-time", "repeated", "continuous"].index(st.session_state.behavior_continuity)
        )

    visuals = [
        st.session_state.visual_mouth,
        st.session_state.visual_posture,
        st.session_state.visual_ears,
        st.session_state.visual_tail,
        st.session_state.visual_eyes,
        st.session_state.visual_hiding
    ]

    temporal_fields = [
        st.session_state.movement_pattern,
        st.session_state.movement_level,
        st.session_state.behavior_change,
        st.session_state.behavior_continuity
    ]

    visual_quality = visual_cue_completeness(
        st.session_state.image_available,
        st.session_state.video_available,
        *visuals
    )

    temporal_quality = temporal_completeness(*temporal_fields)

    st.markdown("### Input Quality Preview")
    q1, q2, q3 = st.columns(3)

    with q1:
        st.write(f"**Media used:** {st.session_state.media_type.title()}")
    with q2:
        st.write(f"**Visual cue quality:** {quality_label(visual_quality)}")
        st.progress(visual_quality)
    with q3:
        st.write(f"**Temporal cue quality:** {quality_label(temporal_quality)}")
        st.progress(temporal_quality)

    nav1, nav2, spacer = st.columns([1, 1, 3])
    with nav1:
        st.button("Back", on_click=set_phase, args=(2,), use_container_width=True)

    with nav2:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True, disabled=invalid_media)

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
            "video_available": st.session_state.video_available,
            "media_type": st.session_state.media_type,
            "visual_mouth": st.session_state.visual_mouth,
            "visual_posture": st.session_state.visual_posture,
            "visual_ears": st.session_state.visual_ears,
            "visual_tail": st.session_state.visual_tail,
            "visual_eyes": st.session_state.visual_eyes,
            "visual_hiding": st.session_state.visual_hiding,
            "
