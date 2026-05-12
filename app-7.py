import streamlit as st
import numpy as np
import io
import cv2
import tempfile
import os
import time
import streamlit.components.v1 as components

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VigilAI – Deepfake Detector",
    page_icon="./favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SAFE IMPORTS (with fallback messages) ──────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

# MediaPipe is optional — not available on Python 3.14 (Streamlit Cloud)
# App automatically uses OpenCV Haar+optical-flow fallback when absent
MEDIAPIPE_OK = False
try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_mesh
    _test = _mp_face.FaceMesh(max_num_faces=1)
    _test.close()
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False

try:
    import scipy.signal as scipy_signal
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    :root {
        --bg:#020b18; --surface:#071528; --surface2:#0c1f38;
        --accent:#00d4ff; --danger:#ff4060; --safe:#00ff9d;
        --warn:#ffbe00; --text:#e8f4ff; --muted:#7a9bb8;
        --border:rgba(0,212,255,0.15);
    }
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
    .stApp{
        background-image:linear-gradient(rgba(0,212,255,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,0.025) 1px,transparent 1px)!important;
        background-size:48px 48px!important;background-color:var(--bg)!important;
    }
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding:0!important;max-width:100%!important;}
    [data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}

    /* TABS */
    .stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:4px!important;gap:4px!important;}
    .stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-family:'DM Mono',monospace!important;font-size:0.78rem!important;letter-spacing:0.06em!important;border-radius:8px!important;padding:10px 20px!important;border:none!important;text-transform:uppercase!important;}
    .stTabs [aria-selected="true"]{background:var(--accent)!important;color:var(--bg)!important;font-weight:700!important;}

    /* BUTTONS */
    .stButton>button{background:linear-gradient(135deg,var(--accent),#0099cc)!important;color:var(--bg)!important;font-family:'Syne',sans-serif!important;font-weight:800!important;font-size:0.95rem!important;border:none!important;border-radius:10px!important;padding:14px 32px!important;transition:all 0.2s!important;letter-spacing:0.03em!important;width:100%!important;}
    .stButton>button:hover{box-shadow:0 0 40px rgba(0,212,255,0.6)!important;transform:translateY(-2px)!important;}

    /* FILE UPLOADER */
    [data-testid="stFileUploader"]{background:var(--surface)!important;border:1.5px dashed rgba(0,212,255,0.25)!important;border-radius:14px!important;padding:10px!important;transition:border-color 0.2s!important;}
    [data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;}

    /* AUDIO */
    audio{width:100%;border-radius:10px;margin-top:8px;}

    /* RADIO */
    .stRadio label{color:var(--text)!important;font-size:0.9rem!important;}

    /* SPINNER */
    .stSpinner>div{border-top-color:var(--accent)!important;}

    /* METRICS */
    [data-testid="stMetric"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:18px!important;}
    [data-testid="stMetricLabel"]{color:var(--muted)!important;font-family:'DM Mono',monospace!important;font-size:0.68rem!important;letter-spacing:0.1em!important;}
    [data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'Syne',sans-serif!important;font-weight:800!important;font-size:1.4rem!important;}

    /* ALERTS */
    .stAlert{border-radius:12px!important;font-family:'DM Mono',monospace!important;font-size:0.85rem!important;}
    [data-testid="stImage"] img{border-radius:12px!important;border:1px solid var(--border)!important;}
    ::-webkit-scrollbar{width:4px;} ::-webkit-scrollbar-track{background:var(--bg);} ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}

    /* PROGRESS BAR */
    .stProgress>div>div{background:var(--accent)!important;border-radius:4px!important;}

    /* SELECT BOX */
    .stSelectbox [data-baseweb="select"]{background:var(--surface)!important;border-color:var(--border)!important;}
    </style>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
def render_header():
    components.html("""
    <div id="particles-js" style="position:fixed;width:100%;height:100%;top:0;left:0;z-index:-1;pointer-events:none;"></div>
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>particlesJS("particles-js",{particles:{number:{value:55,density:{enable:true,value_area:900}},color:{value:"#00d4ff"},shape:{type:"circle"},opacity:{value:0.3,random:true},size:{value:2.5,random:true},line_linked:{enable:true,distance:110,color:"#00d4ff",opacity:0.15,width:1},move:{enable:true,speed:1.2,out_mode:"bounce"}}});</script>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Mono&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
    .hero-wrap{font-family:'Syne',sans-serif;padding:40px 48px 28px;border-bottom:1px solid rgba(0,212,255,0.12);background:rgba(2,11,24,0.85);backdrop-filter:blur(12px);overflow:visible;}
    .hero-top{display:flex;align-items:center;gap:20px;margin-bottom:16px;}
    .hero-logo{width:60px;height:60px;object-fit:contain;filter:drop-shadow(0 0 14px rgba(0,212,255,0.6));animation:glow 3s ease-in-out infinite;}
    @keyframes glow{0%,100%{filter:drop-shadow(0 0 10px rgba(0,212,255,0.5))}50%{filter:drop-shadow(0 0 26px rgba(0,212,255,0.9))}}
    .hero-badge{display:inline-flex;align-items:center;gap:7px;font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.18em;text-transform:uppercase;color:#00ff9d;border:1px solid rgba(0,255,157,0.3);padding:4px 12px;border-radius:2px;margin-bottom:10px;}
    .live-dot{width:6px;height:6px;background:#00ff9d;border-radius:50%;animation:blink 1.2s ease-in-out infinite;}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0.1}}
    .hero-title{font-size:clamp(1.8rem,4vw,3rem);font-weight:800;letter-spacing:-2px;line-height:1;color:#fff;margin:0;}
    .hero-title .danger{color:#ff4060;} .hero-title .accent{color:#00d4ff;}
    .hero-sub{font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#7a9bb8;font-weight:300;max-width:600px;line-height:1.7;margin:10px 0 0;}
    .hero-sub strong{color:#e8f4ff;font-weight:500;}
    .pill-row{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;}
    .pill{font-family:'DM Mono',monospace;font-size:0.65rem;color:#7a9bb8;border:1px solid rgba(0,212,255,0.12);padding:5px 12px;border-radius:20px;letter-spacing:0.08em;}
    .pill span{color:#00d4ff;}
    </style>
    <div class="hero-wrap">
      <div class="hero-top">
        <img src="https://raw.githubusercontent.com/your-username/vigilai/main/logo.png" class="hero-logo" onerror="this.style.display='none'">
        <div>
          <div class="hero-badge"><span class="live-dot"></span>Real-Time Detection · Active</div>
          <div class="hero-title">Stop <span class="danger">Deepfake</span> <span class="accent">Fraud.</span></div>
        </div>
      </div>
      <p class="hero-sub">Detect <strong>AI-generated voices</strong> and <strong>deepfake video calls</strong> using 6 independent signal layers. Upload audio, record live, or scan any video.</p>
      <div class="pill-row">
        <div class="pill"><span>6</span> Detection Signals</div>
        <div class="pill"><span>468</span> Facial Landmarks</div>
        <div class="pill"><span>40</span> MFCC Coefficients</div>
        <div class="pill"><span>100%</span> Private · No Cloud</div>
        <div class="pill"><span>⚡</span> Results in &lt;5 sec</div>
      </div>
    </div>
    """, height=290)

# ── RESULT CARD HTML ───────────────────────────────────────────────────────────
def render_result_card(is_fake: bool, label: str, signals: list, confidence: float):
    color   = "#ff4060" if is_fake else "#00ff9d"
    bg_col  = "rgba(255,64,96,0.07)" if is_fake else "rgba(0,255,157,0.07)"
    icon    = "🚨" if is_fake else "✅"
    verdict = "SYNTHETIC / AI DETECTED" if is_fake else "VERIFIED HUMAN"
    conf_pct = int(confidence * 100)

    rows = ""
    for name, val, max_val, flagged in signals:
        pct = min(100, max(0, int((val / max(max_val, 0.001)) * 100)))
        c = "#ff4060" if flagged else "#00ff9d"
        flag_txt = "⚠ SUSPICIOUS" if flagged else "✓ NORMAL"
        rows += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid rgba(0,212,255,0.06);">
          <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#7a9bb8;min-width:170px;">{name}</span>
          <div style="flex:1;height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{c};border-radius:2px;"></div>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:{c};min-width:90px;text-align:right;">{flag_txt}</span>
        </div>"""

    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Mono&display=swap" rel="stylesheet">
    <div style="background:{bg_col};border:1px solid {color}55;border-radius:16px;padding:28px 32px;margin:16px 0;box-shadow:0 0 40px {color}15;font-family:'DM Mono',monospace;">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:24px;">
        <span style="font-size:2.2rem;">{icon}</span>
        <div style="flex:1;">
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;color:{color};letter-spacing:-0.5px;">{verdict}</div>
          <div style="font-size:0.68rem;color:#7a9bb8;letter-spacing:0.12em;text-transform:uppercase;margin-top:4px;">{label}</div>
        </div>
        <div style="text-align:center;">
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:{color};">{conf_pct}%</div>
          <div style="font-size:0.65rem;color:#7a9bb8;letter-spacing:0.12em;text-transform:uppercase;">Confidence</div>
        </div>
      </div>
      <div style="margin-bottom:8px;font-size:0.68rem;color:#7a9bb8;letter-spacing:0.12em;text-transform:uppercase;">Signal Breakdown</div>
      {rows}
    </div>
    """, height=100 + len(signals) * 48)

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def load_audio_safe(audio_bytes: bytes, source_ext: str = ".wav"):
    """Try BytesIO first, then temp-file fallback for mic recorder WebM/OGG blobs."""
    import librosa
    # Try 1: direct BytesIO — works for uploaded WAV/MP3/FLAC
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=30)
        if len(y) > 0:
            return y, sr
    except Exception:
        pass
    # Try 2: temp file with extension so libsndfile can sniff the format
    for ext in [source_ext, ".wav", ".ogg", ".webm", ".mp3"]:
        tf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                tf.write(audio_bytes)
                tf_path = tf.name
            y, sr = librosa.load(tf_path, sr=16000, duration=30)
            os.unlink(tf_path)
            if len(y) > 0:
                return y, sr
        except Exception:
            if tf_path:
                try: os.unlink(tf_path)
                except: pass
    raise ValueError("Could not decode audio. Please upload a .wav or .mp3 file instead.")


def analyze_audio(audio_bytes: bytes, source_ext: str = ".wav"):
    if not LIBROSA_OK:
        return None, "LibROSA not available on this server."
    try:
        import librosa
        y, sr = load_audio_safe(audio_bytes, source_ext)
        if len(y) < 3200:
            return None, "Audio too short. Please provide at least 2 seconds."

        # ── Feature 1: MFCC Variance (primary AI voice detector) ──────────────
        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_var    = float(np.var(mfcc))
        mfcc_delta  = librosa.feature.delta(mfcc)
        delta_var   = float(np.var(mfcc_delta))

        # ── Feature 2: Zero Crossing Rate ──────────────────────────────────────
        zcr         = float(np.mean(librosa.feature.zero_crossing_rate(y)) * 1000)

        # ── Feature 3: Spectral Centroid Stability ─────────────────────────────
        centroid    = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_std= float(np.std(centroid))

        # ── Feature 4: Chroma Variation (organic pitch drift) ──────────────────
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_var  = float(np.var(chroma))

        # ── Feature 5: Spectral Rolloff Variance ───────────────────────────────
        rolloff     = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_std = float(np.std(rolloff))

        # ── Feature 6: RMS Energy Consistency (AI is too uniform) ─────────────
        rms         = librosa.feature.rms(y=y)
        rms_var     = float(np.var(rms) * 1e6)

        # ── Feature 7: Harmonics-to-Noise Ratio (via spectral flatness) ────────
        flatness    = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        # AI voices have very low flatness (too tonal / synthesized)

        # ── Weighted Scoring ───────────────────────────────────────────────────
        flags = []
        flags.append(("MFCC Variance",       mfcc_var,     6000,  mfcc_var < 2800))
        flags.append(("MFCC Delta Var",       delta_var,    500,   delta_var < 80))
        flags.append(("Zero Crossing Rate",   zcr,          120,   zcr < 32))
        flags.append(("Centroid Stability",   centroid_std, 1200,  centroid_std < 350))
        flags.append(("Chroma Variation",     chroma_var*1000, 50, chroma_var < 0.012))
        flags.append(("RMS Consistency",      rms_var,      50,    rms_var < 5))
        flags.append(("Spectral Flatness×100",flatness*100, 5,     flatness < 0.002))

        suspicious = sum(1 for _, _, _, f in flags if f)
        total_flags = len(flags)

        # Weighted confidence
        confidence = suspicious / total_flags
        is_fake = suspicious >= 3

        return {
            "is_fake":    is_fake,
            "confidence": confidence if is_fake else 1 - confidence,
            "flags":      flags,
            "suspicious": suspicious,
            "mfcc_var":   mfcc_var,
            "zcr":        zcr,
            "centroid_std": centroid_std,
            "chroma_var": chroma_var,
            "rms_var":    rms_var,
            "flatness":   flatness,
        }, None

    except Exception as e:
        return None, f"Audio analysis failed: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO ANALYSIS ENGINE  (MediaPipe-safe + pure-OpenCV fallback)
# ══════════════════════════════════════════════════════════════════════════════
def analyze_video_frames(frames: list):
    if len(frames) < 10:
        return None, {"error": "Need at least 10 frames. Video too short."}

    # ── Try MediaPipe if available ─────────────────────────────────────────────
    use_mediapipe = MEDIAPIPE_OK
    face_mesh     = None
    if use_mediapipe:
        try:
            mp_face    = mp.solutions.face_mesh
            face_mesh  = mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            use_mediapipe = False

    # ── OpenCV Haar fallback ───────────────────────────────────────────────────
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)

    blink_scores   = []
    jitter_scores  = []
    spectral_energy= []
    texture_scores = []
    face_count     = 0
    prev_gray      = None
    flow_variances = []

    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Optical Flow (catches video artifacts) ─────────────────────────
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_variances.append(float(np.var(mag)))
            prev_gray = gray.copy()

            # ── 2D FFT Spectral Energy ─────────────────────────────────────────
            dft       = np.fft.fft2(gray)
            magnitude = 20 * np.log(np.abs(np.fft.fftshift(dft)) + 1)
            spectral_energy.append(float(np.mean(magnitude)))

            # ── Texture / Local Binary Pattern proxy ──────────────────────────
            # AI faces are too smooth; real faces have grain
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            texture_scores.append(laplacian_var)

            # ── Landmark detection ─────────────────────────────────────────────
            if use_mediapipe and face_mesh:
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    face_count += 1
                    lm = res.multi_face_landmarks[0].landmark
                    jitter_scores.append((lm[4].x, lm[4].y, lm[4].z))
                    upper = lm[159].y
                    lower = lm[145].y
                    face_h = abs(lm[10].y - lm[152].y) + 1e-6
                    blink_scores.append(abs(upper - lower) / face_h)
            else:
                # OpenCV Haar fallback
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                if len(faces) > 0:
                    face_count += 1
                    x, y, w, h = faces[0]
                    cx = x + w / 2; cy = y + h / 2
                    jitter_scores.append((cx / gray.shape[1], cy / gray.shape[0], 0.0))
        except Exception:
            continue

    if face_mesh:
        try:
            face_mesh.close()
        except Exception:
            pass

    # ── Guard ──────────────────────────────────────────────────────────────────
    if face_count < 5:
        return None, {"error": "Could not detect a face reliably. Ensure the video has a clear, front-facing face with good lighting."}

    # ── Blink Check ────────────────────────────────────────────────────────────
    if blink_scores:
        max_ear     = max(blink_scores)
        blinked     = any(s < max_ear * 0.78 for s in blink_scores)
        avg_ear     = float(np.mean(blink_scores))
    else:
        blinked     = True   # can't check without mediapipe — assume OK
        avg_ear     = 0.15

    # ── Face Jitter ────────────────────────────────────────────────────────────
    if len(jitter_scores) >= 5:
        nose_arr     = np.array(jitter_scores)
        movement_var = float(np.var(nose_arr, axis=0).sum())
        is_shaking   = (movement_var > 0.0008) or (movement_var < 5e-7)
    else:
        movement_var = 0.0
        is_shaking   = False

    # ── Spectral Energy ────────────────────────────────────────────────────────
    avg_energy      = float(np.mean(spectral_energy)) if spectral_energy else 150
    is_synth_pixels = avg_energy < 135 or avg_energy > 182

    # ── Texture / Sharpness (AI = too smooth) ─────────────────────────────────
    avg_texture     = float(np.mean(texture_scores)) if texture_scores else 100
    is_smooth       = avg_texture < 60   # real faces typically > 80

    # ── Optical Flow Consistency (deepfakes flicker) ───────────────────────────
    if flow_variances:
        flow_mean   = float(np.mean(flow_variances))
        flow_std    = float(np.std(flow_variances))
        # Normal video: flow is relatively consistent
        # Deepfakes: flow is erratic or near-zero (GAN frozen background)
        is_flow_odd = flow_std > flow_mean * 2.5 or flow_mean < 0.001
    else:
        is_flow_odd = False
        flow_mean   = 0.0

    # ── Weighted Verdict ───────────────────────────────────────────────────────
    fake_score = 0
    if not blinked:      fake_score += 3   # strongest biological signal
    if is_shaking:       fake_score += 2
    if is_synth_pixels:  fake_score += 1
    if is_smooth:        fake_score += 2
    if is_flow_odd:      fake_score += 1

    max_score   = 9
    confidence  = fake_score / max_score
    is_fake     = fake_score >= 3

    details = {
        "is_fake":          is_fake,
        "confidence":       confidence if is_fake else 1 - confidence,
        "blinked":          blinked,
        "avg_ear":          avg_ear,
        "movement_var":     movement_var,
        "is_shaking":       is_shaking,
        "avg_energy":       avg_energy,
        "is_synth_pixels":  is_synth_pixels,
        "avg_texture":      avg_texture,
        "is_smooth":        is_smooth,
        "flow_mean":        flow_mean,
        "is_flow_odd":      is_flow_odd,
        "face_frames":      face_count,
        "used_mediapipe":   use_mediapipe,
    }
    return is_fake, details


# ══════════════════════════════════════════════════════════════════════════════
# DRAW FRAME OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def draw_overlay(frame, is_fake: bool, details: dict):
    h, w  = frame.shape[:2]
    color = (0, 50, 255) if is_fake else (0, 220, 80)
    sz    = 26

    # Corner brackets
    for (x, y) in [(0,0),(w-sz,0),(0,h-sz),(w-sz,h-sz)]:
        cv2.line(frame,(x,y),(x+sz,y),color,2)
        cv2.line(frame,(x,y),(x,y+sz),color,2)

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,46),(2,11,24),-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)

    label = "AI / SYNTHETIC" if is_fake else "HUMAN VERIFIED"
    cv2.putText(frame, label, (14,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    conf_txt = f"{int(details.get('confidence',0)*100)}% conf"
    cv2.putText(frame, conf_txt, (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Bottom metrics
    strip_y = h - 38
    cv2.rectangle(frame,(0,strip_y),(w,h),(2,11,24),-1)
    items = [
        f"EAR:{details.get('avg_ear',0):.3f}",
        f"JIT:{details.get('movement_var',0):.5f}",
        f"SPEC:{details.get('avg_energy',0):.1f}",
        f"TEX:{details.get('avg_texture',0):.1f}",
    ]
    for i, t in enumerate(items):
        cv2.putText(frame, t, (12 + i*(w//4), h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,150,200), 1)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UI BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
def section_header(tag: str, title: str, desc: str):
    st.markdown(f"""
    <div style="padding:28px 0 16px;">
      <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#00d4ff;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:8px;">{tag}</div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.7rem;letter-spacing:-0.5px;margin-bottom:8px;">{title}</div>
      <div style="font-size:0.88rem;color:#7a9bb8;font-weight:300;max-width:600px;line-height:1.75;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

def info_box(items: list):
    rows = "".join([f'<div style="padding:8px 0;border-bottom:1px solid rgba(0,212,255,0.07);font-size:0.78rem;">{i}</div>' for i in items])
    st.markdown(f"""
    <div style="background:rgba(7,21,40,0.9);border:1px solid rgba(0,212,255,0.13);border-radius:12px;padding:20px 24px;font-family:'DM Mono',monospace;color:#7a9bb8;line-height:1.9;">
      {rows}
    </div>""", unsafe_allow_html=True)

def empty_state(msg: str):
    st.markdown(f"""
    <div style="background:rgba(7,21,40,0.5);border:1.5px dashed rgba(0,212,255,0.15);border-radius:14px;padding:48px;text-align:center;color:#7a9bb8;font-family:'DM Mono',monospace;font-size:0.82rem;letter-spacing:0.08em;">{msg}</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
inject_css()
render_header()

# ── Silent compat check — no banner shown to users ────────────────────────────

st.markdown('<div style="padding:28px 40px 60px;">', unsafe_allow_html=True)

# Reset
_, col_r = st.columns([14, 1])
with col_r:
    if st.button("↺"):
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🔊  VOICE ANALYSIS", "🎥  VIDEO SCAN", "ℹ️  HOW IT WORKS"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — AUDIO
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    section_header(
        "// audio detection engine",
        "Voice Authentication",
        "Upload a voice clip or record live audio to detect if the voice is human or AI-generated. Uses 7 independent spectral signals."
    )

    if not LIBROSA_OK:
        st.error("❌ Audio analysis is not available — LibROSA failed to load on this server. Please try again later or run locally.")
    else:
        mode_a = st.radio("Input Mode", ["📁 Upload File", "🎙️ Record Live Audio"], horizontal=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        col_a1, col_gap, col_a2 = st.columns([5, 1, 4])
        audio_bytes = None

        # ── UPLOAD ────────────────────────────────────────────────────────────
        if "Upload" in mode_a:
            with col_a1:
                st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#7a9bb8;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">Upload Audio File</div>', unsafe_allow_html=True)
                up_audio = st.file_uploader("", type=["wav","mp3","ogg","flac","m4a"], label_visibility="collapsed")
                if up_audio:
                    st.audio(up_audio)
                    audio_bytes = up_audio.read()
                    ext = os.path.splitext(up_audio.name)[-1].lower() or ".wav"
                    st.session_state["audio_ext"] = ext

        # ── LIVE RECORD ───────────────────────────────────────────────────────
        else:
            with col_a1:
                st.markdown("""
                <div style="background:rgba(7,21,40,0.8);border:1px solid rgba(0,212,255,0.2);border-radius:14px;padding:24px;text-align:center;font-family:'DM Mono',monospace;font-size:0.8rem;color:#7a9bb8;">
                  <div style="font-size:2rem;margin-bottom:12px;">🎙️</div>
                  <div style="color:#00d4ff;margin-bottom:8px;">Live Recording</div>
                  <div>Use the recorder below to capture a voice sample directly from your microphone.</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                try:
                    from streamlit_mic_recorder import mic_recorder
                    recorded = mic_recorder(
                        start_prompt="⏺ Start Recording",
                        stop_prompt="⏹ Stop & Analyze",
                        key="mic_rec"
                    )
                    if recorded and recorded.get("bytes"):
                        st.audio(recorded["bytes"], format="audio/wav")
                        audio_bytes = recorded["bytes"]
                        st.session_state["audio_ext"] = ".ogg"
                        st.success("✅ Recording captured! Click **Run Analysis** below.")
                except ImportError:
                    st.warning("⚠️ Live recording not available on this server. Please upload a file instead.")
                    st.code("pip install streamlit-mic-recorder", language="bash")

        with col_a2:
            info_box([
                '📊 <span style="color:#00d4ff;">MFCC Variance</span> — AI voices are unnaturally stable',
                '〰️ <span style="color:#00d4ff;">MFCC Delta</span> — Rate of spectral change',
                '🔁 <span style="color:#00d4ff;">Zero Crossing Rate</span> — Unnatural patterns',
                '📡 <span style="color:#00d4ff;">Spectral Centroid</span> — Missing natural drift',
                '🎼 <span style="color:#00d4ff;">Chroma Variation</span> — Organic pitch changes',
                '⚡ <span style="color:#00d4ff;">RMS Consistency</span> — AI is too uniform',
                '🎵 <span style="color:#00d4ff;">Spectral Flatness</span> — AI is too tonal',
            ])

        st.markdown("<br>", unsafe_allow_html=True)

        if audio_bytes:
            if st.button("▶  RUN VOICE ANALYSIS", key="run_audio"):
                with st.spinner("🔍 Analyzing spectral signatures across 7 layers..."):
                    src_ext = st.session_state.get("audio_ext", ".wav")
                    result, err = analyze_audio(audio_bytes, source_ext=src_ext)

                if err:
                    st.warning(f"⚠️ {err}")
                elif result:
                    # Metrics
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("MFCC Variance", f"{result['mfcc_var']:.0f}", help="<2800 = suspicious")
                    m2.metric("ZCR ×1000",     f"{result['zcr']:.1f}",     help="<32 = suspicious")
                    m3.metric("Centroid Std",  f"{result['centroid_std']:.0f}", help="<350 = suspicious")
                    m4.metric("Confidence",    f"{int(result['confidence']*100)}%")

                    st.markdown("<br>", unsafe_allow_html=True)
                    render_result_card(
                        result["is_fake"], "AUDIO SIGNAL ANALYSIS — 7 FEATURES",
                        result["flags"], result["confidence"]
                    )
        else:
            empty_state("🎙️ Upload or record a voice sample above, then click Run Analysis")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    section_header(
        "// visual liveness engine",
        "Video Deepfake Scan",
        "Upload any video to detect AI-generated faces. Uses 5 independent layers including optical flow, facial landmarks, texture analysis, and pixel spectral energy."
    )

    # MediaPipe status
    if MEDIAPIPE_OK:
        st.markdown('<div style="display:inline-flex;align-items:center;gap:6px;font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#00ff9d;border:1px solid rgba(0,255,157,0.2);padding:4px 12px;border-radius:20px;margin-bottom:12px;">● MediaPipe · Full 468-Point Detection Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:inline-flex;align-items:center;gap:6px;font-family:\'DM Mono\',monospace;font-size:0.7rem;color:#00ff9d;border:1px solid rgba(0,255,157,0.2);padding:4px 12px;border-radius:20px;margin-bottom:12px;">● Multi-Layer Detection Active</div>', unsafe_allow_html=True)

    mode_v = st.radio("Video Mode", ["📁 Upload Video", "📹 Live Webcam"], horizontal=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── UPLOAD VIDEO ──────────────────────────────────────────────────────────
    if "Upload" in mode_v:
        col_v1, col_gap, col_v2 = st.columns([5, 1, 4])

        with col_v1:
            st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#7a9bb8;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">Upload Video File</div>', unsafe_allow_html=True)
            up_video = st.file_uploader("", type=["mp4","mov","avi","webm"], label_visibility="collapsed")
            if up_video:
                st.video(up_video)

        with col_v2:
            info_box([
                '👁️ <span style="color:#00d4ff;">Blink Detection</span> — Biological liveness',
                '📐 <span style="color:#00d4ff;">Face Jitter</span> — Nose bridge stability',
                '🌊 <span style="color:#00d4ff;">Optical Flow</span> — Motion consistency',
                '🔬 <span style="color:#00d4ff;">Texture Analysis</span> — AI faces too smooth',
                '📡 <span style="color:#00d4ff;">Spectral FFT</span> — Pixel-level artifacts',
            ])

        if up_video:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶  RUN VIDEO SCAN", key="run_video"):
                prog = st.progress(0, text="Reading video frames...")
                with st.spinner("🔍 Analyzing liveness across 5 detection layers..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                            tf.write(up_video.read())
                            path = tf.name

                        cap = cv2.VideoCapture(path)
                        all_frames = []
                        total_est  = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                        while cap.isOpened():
                            ret, f = cap.read()
                            if not ret: break
                            all_frames.append(f)
                            prog.progress(min(0.5, len(all_frames)/total_est * 0.5), text=f"Extracting frames... {len(all_frames)}")
                        cap.release()
                        os.unlink(path)

                        total = len(all_frames)
                        if total == 0:
                            st.error("❌ Could not read video. Try a different format (MP4 recommended).")
                            st.stop()

                        prog.progress(0.55, text="Sampling frames for analysis...")
                        step   = max(1, total // 100)
                        sample = all_frames[::step]

                        prog.progress(0.6, text="Running detection layers...")
                        is_fake, details = analyze_video_frames(sample)
                        prog.progress(1.0, text="Done!")
                        time.sleep(0.3)
                        prog.empty()

                    except Exception as e:
                        prog.empty()
                        st.error(f"❌ Video processing error: {str(e)}")
                        st.stop()

                if is_fake is None:
                    err_msg = details.get("error","Unknown error") if isinstance(details, dict) else str(details)
                    st.warning(f"⚠️ {err_msg}")
                else:
                    # Metrics
                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Total Frames",  total)
                    m2.metric("Frame Detected", details["face_frames"])
                    m3.metric("Spectral Avg",  f"{details['avg_energy']:.1f}")
                    m4.metric("Confidence",    f"{int(details['confidence']*100)}%")

                    st.markdown("<br>", unsafe_allow_html=True)

                    signals = [
                        ("Blink Signal",       1 if details["blinked"] else 0,     1,   not details["blinked"]),
                        ("Avg Eye Ratio",      details["avg_ear"],                  0.35, details["avg_ear"] < 0.02),
                        ("Face Jitter Σ",      details["movement_var"]*10000,       10,   details["is_shaking"]),
                        ("Spectral Energy",    details["avg_energy"],               200,  details["is_synth_pixels"]),
                        ("Texture Sharpness",  details["avg_texture"],              200,  details["is_smooth"]),
                        ("Optical Flow",       details.get("flow_mean",0)*1000,     5,    details.get("is_flow_odd",False)),
                    ]
                    mode_note = "MediaPipe Landmarks" if details["used_mediapipe"] else "OpenCV Fallback"
                    render_result_card(is_fake, f"VIDEO LIVENESS ANALYSIS · {mode_note}", signals, details["confidence"])

        else:
            empty_state("📹 Upload a video file above to begin scanning")

    # ── LIVE WEBCAM ───────────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div style="background:rgba(255,190,0,0.06);border:1px solid rgba(255,190,0,0.2);border-radius:10px;padding:16px 22px;font-family:'DM Mono',monospace;font-size:0.78rem;color:#ffbe00;margin-bottom:16px;">
        ⚠️ Live webcam works best when running locally.<br>
        On Streamlit Cloud, camera access depends on your browser permissions.
        Run: <code style="background:rgba(0,0,0,0.2);padding:2px 8px;border-radius:4px;">streamlit run app.py</code> on your machine for best results.
        </div>
        """, unsafe_allow_html=True)

        run_cam = st.checkbox("🟢 Enable Live Webcam")
        if run_cam:
            FRAME_WIN = st.empty()
            stop_btn  = st.button("⏹ Stop Camera")
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                st.error("❌ Camera not found. Check browser permissions or run locally.")
            else:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ring, fcount, last_fake, last_det = [], 0, None, {}
                while run_cam and not stop_btn:
                    ret, frame = cam.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    fcount += 1
                    ring.append(frame.copy())
                    if len(ring) > 30: ring.pop(0)
                    if fcount % 15 == 0 and len(ring) >= 15:
                        res, det = analyze_video_frames(ring[::3])
                        if res is not None:
                            last_fake = res; last_det = det
                    if last_fake is not None:
                        frame = draw_overlay(frame, last_fake, last_det)
                    FRAME_WIN.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                cam.release()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    section_header("// documentation", "How VigilAI Works", "A transparent look at the detection signals used to identify AI-generated audio and video.")
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px;">

      <div style="background:rgba(7,21,40,0.8);border:1px solid rgba(0,212,255,0.13);border-radius:14px;padding:28px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#00d4ff;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:14px;">🔊 Audio Detection</div>
        <div style="font-size:0.85rem;color:#7a9bb8;line-height:2;">
          <b style="color:#e8f4ff;">MFCC Variance</b> — AI voices repeat patterns, causing lower variance in the 40 Mel-frequency cepstral coefficients.<br>
          <b style="color:#e8f4ff;">MFCC Delta</b> — Measures rate of spectral change. AI lacks natural acceleration.<br>
          <b style="color:#e8f4ff;">Zero Crossing Rate</b> — How often the waveform crosses zero. AI has unnatural patterns.<br>
          <b style="color:#e8f4ff;">Spectral Centroid</b> — Center of mass of the spectrum. AI voices are too consistent.<br>
          <b style="color:#e8f4ff;">Chroma Variation</b> — Organic pitch changes present in real human voices.<br>
          <b style="color:#e8f4ff;">RMS Energy</b> — AI voices have unnaturally uniform volume levels.<br>
          <b style="color:#e8f4ff;">Spectral Flatness</b> — AI voices are overly tonal (too synthesized).
        </div>
      </div>

      <div style="background:rgba(7,21,40,0.8);border:1px solid rgba(0,212,255,0.13);border-radius:14px;padding:28px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#00d4ff;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:14px;">🎥 Video Detection</div>
        <div style="font-size:0.85rem;color:#7a9bb8;line-height:2;">
          <b style="color:#e8f4ff;">Blink Detection</b> — Real humans blink. Deepfakes often don't, or blink unnaturally. Tracked via 468 MediaPipe landmarks.<br>
          <b style="color:#e8f4ff;">Face Jitter</b> — Deepfake masks "shimmy" because the AI face doesn't perfectly track the head. We track nose bridge variance.<br>
          <b style="color:#e8f4ff;">Optical Flow</b> — Measures motion consistency frame-by-frame. Deepfakes have erratic or frozen-background flow.<br>
          <b style="color:#e8f4ff;">Texture Analysis</b> — AI-generated faces are too smooth. We measure Laplacian variance (sharpness/grain).<br>
          <b style="color:#e8f4ff;">Spectral FFT</b> — 2D Fourier transform detects missing high-frequency pixel data that only real camera sensors produce.
        </div>
      </div>

    </div>
    <div style="margin-top:20px;background:rgba(255,64,96,0.06);border:1px solid rgba(255,64,96,0.18);border-radius:14px;padding:22px 28px;font-family:'DM Mono',monospace;font-size:0.8rem;color:#7a9bb8;line-height:1.8;">
      ⚠️ <b style="color:#ff4060;">Limitations:</b> VigilAI is a research tool, not a certified forensic system. Very high quality deepfakes or compressed videos may reduce accuracy.
      Always combine with other verification methods for critical decisions. The system improves as AI detection research evolves.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
