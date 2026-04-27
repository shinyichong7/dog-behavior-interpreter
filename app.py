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

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Dog Behavior Interpreter",
    page_icon="🐶",
    layout="wide"
)

# ==================================================
# CSS
# ==================================================

st.markdown("""
<style>
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1180px;
}

.phase-card {
    padding: 1.25rem;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    background-color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
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

.small-muted {
    color: #6B7280;
    font-size: 0.9rem;
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

.danger-box {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #FECACA;
    background-color: #FEF2F2;
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

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = False

# ==================================================
# DATA GENERATION
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
        ("rf", RandomForestClassifier(
            n_estimators=220,
            random_state=42,
            class_weight="balanced"
        ))
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

def recommendation_for(pred):
    return {
        "recovery": {
            "summary": "Likely recovery. Support rest, hydration, and reassessment.",
            "do_now": [
                "Offer water.",
                "Move your dog to a calm, comfortable area.",
                "Pause additional play, running, or training for now."
            ],
            "observe": [
                "Breathing and posture should gradually normalize.",
                "Panting should decrease as the dog rests and cools down.",
                "Energy should return to baseline after recovery."
            ],
            "avoid": [
                "Do not continue intense activity immediately.",
                "Do not assume panting always means anxiety without considering recent activity or heat."
            ]
        },
        "anxiety": {
            "summary": "Possible anxiety or stress. Reduce triggers and support calm observation.",
            "do_now": [
                "Lower noise and stimulation.",
                "Create distance from the trigger if possible.",
                "Give your dog space instead of forcing interaction."
            ],
            "observe": [
                "Watch for repeated pacing, hiding, trembling, lip licking, or inability to settle.",
                "Track whether the behavior improves after the environment becomes calmer."
            ],
            "avoid": [
                "Avoid punishment or forcing exposure.",
                "Avoid overwhelming the dog with attention if they are trying to retreat."
            ]
        },
        "boredom": {
            "summary": "Possible unmet stimulation need. Use low-pressure enrichment.",
            "do_now": [
                "Offer a puzzle toy, sniff game, or short training session.",
                "Try calm engagement before escalating intensity.",
                "Use a structured activity rather than random excitement."
            ],
            "observe": [
                "Behavior should decrease after mental stimulation.",
                "Toy-seeking or pacing may reduce when the dog has an appropriate outlet."
            ],
            "avoid": [
                "Avoid only using high-arousal play if the dog is already restless.",
                "Avoid reinforcing demand behavior every time without structure."
            ]
        },
        "overstimulation": {
            "summary": "Possible overstimulation. Pause activity and help your dog decompress.",
            "do_now": [
                "Stop exciting play or training.",
                "Move to a quieter area.",
                "Use calm routines and allow decompression."
            ],
            "observe": [
                "Look for reduction in pacing, panting, jumping, or inability to settle.",
                "Reassess after 10–15 minutes of calm."
            ],
            "avoid": [
                "Avoid more chase, fetch, or intense play immediately.",
                "Avoid adding more stimulation to an already aroused state."
            ]
        },
        "neutral": {
            "summary": "No strong concern detected. Continue observing without over-intervening.",
            "do_now": [
                "Let your dog continue resting or behaving normally.",
                "Monitor for changes."
            ],
            "observe": [
                "Look for sudden shifts in posture, breathing, responsiveness, or appetite.",
                "Reassess if the behavior becomes prolonged or unusual."
            ],
            "avoid": [
                "Avoid unnecessary intervention if the dog appears relaxed.",
                "Avoid assuming every behavior requires correction."
            ]
        },
        "needs observation": {
            "summary": "Unclear or potentially elevated concern. Observe closely and consider professional guidance if it persists.",
            "do_now": [
                "Reduce activity and keep the dog comfortable.",
                "Monitor breathing, posture, appetite, responsiveness, and duration.",
                "Document what you are seeing."
            ],
            "observe": [
                "Watch whether the behavior resolves or escalates.",
                "Track whether it repeats under similar conditions."
            ],
            "avoid": [
                "Avoid ignoring persistent or unusual behavior.",
                "Avoid relying on this prototype for medical decisions."
            ]
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
    b = inputs["behavior"]
    a = inputs["activity"]
    e = inputs["environment"]
    d = inputs["duration"]
    age = inputs["age_group"]
    energy = inputs["energy"]
    sensitivity = inputs["sensitivity"]

    if image_available == "yes":
        reasons.append("Image-assisted cues were included, so the prediction uses both owner context and visible body-language signals.")
    else:
        reasons.append("No image was uploaded, so the prediction relies more heavily on owner-entered context and has less visual support.")

    if b == "panting":
        reasons.append("Panting can mean recovery, heat, anxiety, or overstimulation, so context is important.")
    if b == "pacing":
        reasons.append("Pacing can reflect anxiety, boredom, anticipation, or excess arousal.")
    if b == "whining":
        reasons.append("Whining can signal stress, attention-seeking, unmet needs, or discomfort.")
    if b == "hiding":
        reasons.append("Hiding can be associated with fear, stress, avoidance, or a need for space.")
    if b == "toy-seeking":
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

    if a == "high":
        reasons.append("Recent high activity increases likelihood of recovery or overstimulation.")
    if a == "none":
        reasons.append("No recent activity makes recovery less likely and increases relevance of boredom or stress.")
    if e == "warm":
        reasons.append("Warm conditions increase likelihood that panting is related to cooling or thermoregulation.")
    if e == "noisy":
        reasons.append("Noise can increase stress sensitivity and anxiety-related behavior.")
    if d == "long":
        reasons.append("Long duration raises concern because the behavior is less likely to be a brief normal response.")
    if d == "short":
        reasons.append("Short duration lowers concern and may indicate a temporary state.")
    if age == "senior":
        reasons.append("Senior dogs require more caution when behavior persists or changes suddenly.")
    if energy == "high":
        reasons.append("High-energy dogs may show pacing or toy-seeking when stimulation needs are unmet.")
    if sensitivity == "high":
        reasons.append("High stress sensitivity increases the relevance of anxiety or environmental triggers.")

    reasons.append(f"Interpretation quality is {quality_label(completeness)} based on available visual and contextual cues.")

    return reasons

def factors(inputs):
    risk = []
    protective = []
    missing = []

    if inputs["visual_mouth"] == "unknown":
        missing.append("mouth / panting visibility")
    if inputs["visual_posture"] == "unknown":
        missing.append("body posture")
    if inputs["visual_ears"] == "unknown":
        missing.append("ear position")
    if inputs["visual_tail"] == "unknown":
        missing.append("tail position")
    if inputs["visual_eyes"] == "unknown":
        missing.append("eye expression")
    if inputs["visual_hiding"] == "unknown":
        missing.append("hiding / avoidance visibility")

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

    with st.expander("Data transparency"):
        st.write(
            """
            This prototype uses synthetic labeled examples to demonstrate the AI workflow.
            The image is not automatically analyzed by computer vision. Instead, it supports
            image-assisted visual cue extraction: the user reviews the image and selects visible
            cues such as posture, ears, tail, eyes, and panting visibility. Those visual cues are
            included as model features and affect the prediction and confidence.
            """
        )

# ==================================================
# HEADER
# ==================================================

st.title("🐶 Dog Behavior Interpreter")
st.write(
    "A mobile-first AI prototype that helps owners interpret ambiguous dog behavior using profile, context, and image-assisted visual cues."
)

st.markdown("""
<div class="disclaimer-card">
⚠️ <b>Important:</b> This prototype provides behavioral guidance only. It is <b>not a medical or diagnostic tool</b>.
Seek veterinary care immediately if your dog shows labored breathing, collapse, pale gums, vomiting, severe distress,
or symptoms that are unusual, persistent, or worsening.
</div>
""", unsafe_allow_html=True)

# Stepper
if st.session_state.feedback_done:
    current_step = 5
elif st.session_state.analysis_done:
    current_step = 4
else:
    current_step = 1

steps = ["Profile", "Context", "Visual Cues", "Interpretation", "Optional Feedback"]

step_html = "<div class='stepper'>"
for i, step in enumerate(steps, start=1):
    if i < current_step:
        cls = "step-complete"
    elif i == current_step:
        cls = "step-active"
    else:
        cls = "step-inactive"
    step_html += f"<span class='{cls}'>{i}. {step}</span>"
step_html += "</div>"

st.markdown(step_html, unsafe_allow_html=True)

# ==================================================
# INPUTS
# ==================================================

left_col, right_col = st.columns([1.15, 0.85])

with left_col:
    with st.form("interpreter_form"):
        st.markdown("### Phase 1 — Dog profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            dog_name = st.text_input("Dog name", value="Sadie", help="Used to personalize the output only.")
            age_group = st.selectbox(
                "Age group",
                ["puppy", "adult", "senior"],
                index=1,
                help="Age can change how cautious the system should be about persistent behavior."
            )

        with c2:
            size = st.selectbox(
                "Size category",
                ["small", "medium", "large"],
                index=1,
                help="Size is included as a general dog profile signal."
            )
            energy = st.selectbox(
                "Typical energy level",
                ["low", "moderate", "high"],
                index=2,
                help="High-energy dogs may show pacing or toy-seeking when stimulation needs are unmet."
            )

        with c3:
            sensitivity = st.selectbox(
                "Known stress sensitivity",
                ["low", "moderate", "high"],
                index=1,
                help="Use this if your dog is generally reactive to noise, strangers, separation, or new environments."
            )
            assumption = st.selectbox(
                "Your initial assumption",
                ["unsure", "anxiety", "boredom", "overstimulation", "recovery"],
                help="This becomes a baseline so the prototype can compare owner interpretation to AI interpretation."
            )

        st.markdown("### Phase 2 — Current behavior context")
        b1, b2 = st.columns(2)

        with b1:
            behavior = st.selectbox(
                "Observed behavior",
                ["panting", "pacing", "whining", "hiding", "toy-seeking", "resting"],
                help="Choose the main behavior you are trying to interpret."
            )
            activity = st.selectbox(
                "Recent activity",
                ["none", "low", "high"],
                help="Recent activity helps distinguish recovery from anxiety or overstimulation."
            )

        with b2:
            environment = st.selectbox(
                "Environment",
                ["indoor", "outdoor", "warm", "noisy"],
                help="Context matters: heat, noise, or unfamiliar settings can change interpretation."
            )
            duration = st.selectbox(
                "Duration",
                ["short", "medium", "long"],
                help="Longer duration raises concern because the behavior is less likely to be a brief normal response."
            )

        st.markdown("### Phase 3 — Image-assisted visual cue review")
        image = st.file_uploader(
            "Upload dog image",
            type=["jpg", "jpeg", "png"],
            help="The image is used to support manual visual cue extraction. It is not automatically analyzed by computer vision in this prototype."
        )

        image_available = "yes" if image else "no"

        vc1, vc2, vc3 = st.columns(3)

        with vc1:
            visual_mouth = st.selectbox(
                "Mouth / panting visible",
                ["unknown", "closed mouth", "open mouth / panting"],
                help="Open mouth or visible panting can support recovery, heat, anxiety, or overstimulation depending on context."
            )
            visual_posture = st.selectbox(
                "Body posture",
                ["unknown", "relaxed", "alert", "tense", "crouched"],
                help="Tense or crouched posture may increase concern; relaxed posture can be protective."
            )

        with vc2:
            visual_ears = st.selectbox(
                "Ear position",
                ["unknown", "neutral", "back/pinned"],
                help="Pinned-back ears can be a stress or uncertainty signal."
            )
            visual_tail = st.selectbox(
                "Tail position",
                ["unknown", "relaxed", "tucked", "high"],
                help="A tucked tail may indicate fear, anxiety, or discomfort."
            )

        with vc3:
            visual_eyes = st.selectbox(
                "Eye expression",
                ["unknown", "relaxed", "wide-eyed / whale eye"],
                help="Wide eyes or whale eye can be a stress-related cue."
            )
            visual_hiding = st.selectbox(
                "Hiding / avoidance visible",
                ["unknown", "yes", "no"],
                help="Hiding or avoidance can increase concern for stress-related behavior."
            )

        submitted = st.form_submit_button("Analyze Behavior", type="primary", use_container_width=True)

with right_col:
    st.markdown("<div class='phase-card'>", unsafe_allow_html=True)
    st.markdown("### Input quality preview")

    visuals = [
        visual_mouth,
        visual_posture,
        visual_ears,
        visual_tail,
        visual_eyes,
        visual_hiding
    ]

    completeness = visual_cue_completeness(image_available, *visuals)

    st.write(f"**Image uploaded:** {'Yes' if image_available == 'yes' else 'No'}")
    st.write(f"**Visual cue completeness:** {quality_label(completeness)}")
    st.progress(completeness)

    if image:
        st.image(image, caption="Uploaded image for visual cue review", use_container_width=True)
    else:
        st.info("No image uploaded. The model will rely on owner-entered profile and context only.")

    st.markdown("### What affects the outcome")
    st.write("- Dog profile")
    st.write("- Behavior context")
    st.write("- Image-assisted visual cues")
    st.write("- Missing or unknown cues lower interpretation quality")

    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# OUTPUTS
# ==================================================

if submitted:
    st.session_state.analysis_done = True

    inputs = {
        "behavior": behavior,
        "activity": activity,
        "environment": environment,
        "duration": duration,
        "assumption": assumption,
        "age_group": age_group,
        "size": size,
        "energy": energy,
        "sensitivity": sensitivity,
        "image_available": image_available,
        "visual_mouth": visual_mouth,
        "visual_posture": visual_posture,
        "visual_ears": visual_ears,
        "visual_tail": visual_tail,
        "visual_eyes": visual_eyes,
        "visual_hiding": visual_hiding
    }

    input_df = pd.DataFrame([inputs])

    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    classes = model.named_steps["rf"].classes_

    prob_df = pd.DataFrame({
        "Behavior State": classes,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    base_prob = float(prob_df.iloc[0]["Probability"])
    adjusted_prob = confidence_adjustment(base_prob, completeness)
    second_state = prob_df.iloc[1]["Behavior State"]
    second_prob = float(prob_df.iloc[1]["Probability"])

    risk, protective, missing = factors(inputs)
    rec = recommendation_for(pred)

    st.divider()
    st.markdown("## Phase 4 — AI interpretation")

    r1, r2, r3 = st.columns([1.1, 0.9, 0.9])

    with r1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"### {confidence_color(adjusted_prob)} Likely state: **{pred.title()}**")
        st.write(f"**Dog:** {dog_name}")
        st.write(f"**Confidence:** {adjusted_prob:.0%} — {confidence_label(adjusted_prob)}")
        st.progress(adjusted_prob)
        st.write(f"**Secondary possibility:** {second_state.title()} ({second_prob:.0%})")
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='kpi'>", unsafe_allow_html=True)
        st.markdown("### Interpretation quality")
        st.write(f"**Input completeness:** {quality_label(completeness)}")
        st.write(f"**Image cues used:** {'Yes' if image_available == 'yes' else 'No'}")
        st.write(f"**Missing visual cues:** {len(missing)}")
        st.progress(completeness)
        st.markdown("</div>", unsafe_allow_html=True)

    with r3:
        st.markdown("<div class='education-box'>", unsafe_allow_html=True)
        st.markdown("### Learning takeaway")
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

    if adjusted_prob < 0.55:
        st.markdown("<div class='warning-card'>", unsafe_allow_html=True)
        st.warning("Confidence is low. The system recommends observing additional signals before taking strong action.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Why this outcome changed")
    if image_available == "yes":
        st.success("Image-assisted visual cues were included, so the model used both context and visible body-language signals.")
    else:
        st.warning("No image was uploaded. The model relied on profile and behavior context only, which reduces visual certainty.")

    for reason in build_reasoning(inputs, image_available, completeness):
        st.write(f"- {reason}")

    st.markdown("### Clear action plan")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("<div class='action-box'>", unsafe_allow_html=True)
        st.markdown("#### Do now")
        st.write(rec["summary"])
        for step in rec["do_now"]:
            st.write(f"- {step}")
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown("<div class='action-box'>", unsafe_allow_html=True)
        st.markdown("#### Observe next")
        for step in rec["observe"]:
            st.write(f"- {step}")
        st.markdown("</div>", unsafe_allow_html=True)

    with a3:
        st.markdown("<div class='action-box'>", unsafe_allow_html=True)
        st.markdown("#### Avoid")
        for step in rec["avoid"]:
            st.write(f"- {step}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Risk and protective signals")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("#### Risk factors")
        for item in risk:
            st.write(f"- {item}")

    with f2:
        st.markdown("#### Protective factors")
        for item in protective:
            st.write(f"- {item}")

    with f3:
        st.markdown("#### Missing signals")
        if missing:
            for item in missing:
                st.write(f"- {item}")
        else:
            st.write("- No major visual cues missing")

    st.markdown("### Owner assumption comparison")
    if assumption == "unsure":
        st.info("No owner assumption was provided, so the system cannot compare against a baseline interpretation.")
    elif assumption == pred:
        st.success(f"The AI interpretation aligns with your initial assumption of **{assumption}**.")
    else:
        st.warning(
            f"You assumed **{assumption}**, while the model predicted **{pred}**. "
            "This may represent a potential interpretation correction."
        )

    with st.expander("Probability distribution"):
        chart_df = prob_df.set_index("Behavior State")
        st.bar_chart(chart_df)

    with st.expander("When to seek professional help"):
        for item in escalation_guidance():
            st.write(f"- {item}")

    st.divider()
    st.markdown("## Phase 5 — Optional feedback")
    st.write("Optional: help improve future predictions by providing outcome feedback.")

    with st.form("feedback_form"):
        fb1, fb2, fb3 = st.columns(3)

        with fb1:
            accurate = st.radio("Was this accurate?", ["Yes", "No", "Unsure"], horizontal=True)

        with fb2:
            action_taken = st.radio("Did you take the action?", ["Yes", "No", "Partially"], horizontal=True)

        with fb3:
            improved = st.radio("Did behavior improve?", ["Yes", "No", "Not sure"], horizontal=True)

        feedback_submitted = st.form_submit_button("Submit Optional Feedback", use_container_width=True)

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
            "image_available": image_available,
            "visual_mouth": visual_mouth,
            "visual_posture": visual_posture,
            "visual_ears": visual_ears,
            "visual_tail": visual_tail,
            "visual_eyes": visual_eyes,
            "visual_hiding": visual_hiding,
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

        st.success("Feedback saved for future dataset improvement and retraining.")
        with st.expander("Feedback record"):
            st.dataframe(feedback_df)

else:
    st.info("Complete Phases 1–3, then click Analyze Behavior to generate the AI interpretation.")
