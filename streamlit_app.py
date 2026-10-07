from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import gaussian_kde

THRESHOLDS = {"smartphone": (10, 20), "laptop": (15, 30), "television": (20, 40)}
SCREENS = list(THRESHOLDS)
LEVEL_COLORS = {"Safe": "#2e7d32", "Moderate": "#ef6c00", "Excessive": "#c62828"}
SCREEN_COLORS = {"smartphone": "#2563eb", "laptop": "#d97706", "television": "#7c3aed", "none": "#9ca3af", "unknown": "#9ca3af"}
READING_COLUMNS = [
    "ts", "elapsed_s", "r", "g", "b", "w", "lux",
    "detected_device", "detection_confidence", "raw_device", "raw_confidence",
    "detection_age_s", "inference_ms",
]
ONLINE_SECONDS = 30
PAGE = 1000

st.set_page_config(page_title="Blue light exposure monitor", layout="wide")


def settings():
    try:
        url = str(st.secrets["SUPABASE_URL"]).rstrip("/")
        key = str(st.secrets["SUPABASE_KEY"])
    except Exception:
        return None
    try:
        tz = str(st.secrets.get("TIMEZONE", "Asia/Manila"))
    except Exception:
        tz = "Asia/Manila"
    return url, key, tz


CONFIG = settings()
st.title("Blue light exposure monitor")
if CONFIG is None:
    st.error("Add SUPABASE_URL and SUPABASE_KEY to this app's secrets to connect to the database.")
    st.stop()

BASE = f"{CONFIG[0]}/rest/v1"
HEADERS = {"apikey": CONFIG[1]}
if not CONFIG[1].startswith("sb_"):
    HEADERS["Authorization"] = f"Bearer {CONFIG[1]}"
TZ = CONFIG[2]


def get(table, params, limit=None):
    if limit is not None:
        response = requests.get(f"{BASE}/{table}", headers=HEADERS, params=params + [("limit", limit)], timeout=20)
        response.raise_for_status()
        return response.json()
    rows, offset = [], 0
    while True:
        response = requests.get(
            f"{BASE}/{table}", headers=HEADERS,
            params=params + [("limit", PAGE), ("offset", offset)], timeout=20,
        )
        response.raise_for_status()
        chunk = response.json()
        rows.extend(chunk)
        if len(chunk) < PAGE:
            return rows
        offset += PAGE


@st.cache_data(ttl=2, show_spinner=False)
def load_sessions():
    columns = ["id", "participant", "device", "distance_cm", "planned_minutes", "started_at", "ended_at", "samples"]
    df = pd.DataFrame(get("sessions", [("select", ",".join(columns)), ("order", "started_at.desc")]), columns=columns)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, format="ISO8601")
    df["ended_at"] = pd.to_datetime(df["ended_at"], utc=True, format="ISO8601")
    return df


def fetch_readings(session_id):
    rows = get("readings", [("select", "*"), ("session_id", f"eq.{session_id}"), ("order", "elapsed_s.asc")])
    df = pd.DataFrame(rows)
    for c in READING_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[READING_COLUMNS].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    for c in ["elapsed_s", "r", "g", "b", "w", "lux", "detection_confidence", "raw_confidence", "detection_age_s", "inference_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def finished_readings(session_id, version=2):
    return fetch_readings(session_id)


@st.cache_data(ttl=2, show_spinner=False)
def live_readings(session_id, version=2):
    return fetch_readings(session_id)


@st.cache_data(ttl=2, show_spinner=False)
def latest_detection():
    rows = get("detections", [("select", "ts,device,confidence"), ("order", "ts.desc")], limit=1)
    return rows[0] if rows else None


def local(ts):
    return ts.tz_convert(TZ).strftime("%b %d, %I:%M %p").replace(" 0", " ")


def ago(seconds):
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s ago"
    if seconds < 5400:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def describe(row):
    state = "recording" if pd.isna(row.ended_at) else f"{int(row.samples)} samples"
    distance = f", {row.distance_cm:g} cm" if row.distance_cm else ""
    return f"{row.participant}, {row.device}{distance}, {local(row.started_at)} ({state})"


def classify(screen, minutes):
    safe, moderate = THRESHOLDS[screen]
    if minutes <= safe:
        return "Safe"
    if minutes <= moderate:
        return "Moderate"
    return "Excessive"


def screen_name(screen):
    return "no screen" if screen in ("none", "unknown") else screen


def current_session(sessions):
    now = pd.Timestamp.now(tz="UTC")
    planned = pd.to_timedelta(sessions["planned_minutes"].fillna(60).astype(float) + 5, unit="min")
    recording = sessions[sessions["ended_at"].isna() & (sessions["started_at"] + planned >= now)]
    return (recording if not recording.empty else sessions)["id"].iloc[0]


def kde_figure(d, title):
    x = d["elapsed_min"].to_numpy(dtype=float)
    y = d["b"].to_numpy(dtype=float)
    if len(x) < 10 or np.ptp(x) == 0:
        return None
    if np.ptp(y) == 0:
        y = y + np.random.normal(0, max(1.0, abs(y.mean())) * 1e-3, len(y))
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return None
    xi = np.linspace(x.min(), x.max(), 120)
    yi = np.linspace(y.min(), y.max(), 120)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="inferno", extent=[xi[0], xi[-1], yi[0], yi[-1]])
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel("Blue channel (counts)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Density")
    fig.tight_layout()
    return fig


def prepare(df, info, sid):
    df = df.copy()
    for c in READING_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["session"] = sid
    df["participant"] = info.participant
    df["label"] = info.device
    df["distance_cm"] = info.distance_cm
    df["elapsed_min"] = df["elapsed_s"] / 60
    df["dt"] = df["elapsed_s"].diff().fillna(0).clip(lower=0, upper=5)
    df["camera"] = df["detected_device"].notna()
    fallback = info.device if info.device in THRESHOLDS else "unknown"
    df["screen"] = df["detected_device"].where(df["camera"], fallback).fillna("unknown")
    return df


try:
    all_sessions = load_sessions()
except requests.RequestException as e:
    st.error(f"Cannot reach the database. Check SUPABASE_URL and SUPABASE_KEY in the app secrets. ({e})")
    st.stop()

follow = st.sidebar.toggle("Follow current session", value=True,
                           help="Shows the session being recorded now, or the latest one if nothing is recording.")
chosen = []
if not follow:
    options = all_sessions["id"].tolist()
    names = {row.id: describe(row) for row in all_sessions.itertuples()}
    if st.sidebar.checkbox("Select all sessions"):
        chosen = options
        st.sidebar.caption(f"{len(options)} sessions selected")
    else:
        chosen = st.sidebar.multiselect("Sessions", options, default=options[:1], format_func=lambda s: names.get(s, s))
live = st.sidebar.toggle("Live refresh", value=True)
interval = st.sidebar.slider("Refresh every (s)", 1, 10, 2)
if st.sidebar.button("Reload all data"):
    st.cache_data.clear()
    st.rerun()


def dashboard():
    try:
        sessions = load_sessions()
    except requests.RequestException as e:
        st.error(f"Lost connection to the database. Retrying. ({e})")
        return
    if sessions.empty:
        st.info("No sessions yet. Run collect.py on the Raspberry Pi to start recording.")
        return

    ids = [current_session(sessions)] if follow else [s for s in chosen if s in set(sessions["id"])]
    if not ids:
        st.info("Select at least one session in the sidebar.")
        return

    meta = sessions.set_index("id").loc[ids].sort_values("started_at", ascending=False)
    frames = []
    try:
        for sid, info in meta.iterrows():
            df = (live_readings if pd.isna(info.ended_at) else finished_readings)(sid)
            frames.append(prepare(df, info, sid))
    except requests.RequestException as e:
        st.error(f"Lost connection to the database. Retrying. ({e})")
        return

    now = datetime.now(timezone.utc)
    info = meta.iloc[0]
    current = frames[0]
    recording = pd.isna(info.ended_at)

    st.subheader(describe(pd.Series({**info.to_dict(), "id": meta.index[0]})).rsplit(" (", 1)[0])
    if current.empty:
        st.info("Waiting for the first readings from this session.")
        return

    last = current.iloc[-1]
    age = (now - last.ts.to_pydatetime()).total_seconds()
    if recording and age <= ONLINE_SECONDS:
        st.success(f"Recording. Last reading {ago(age)}.")
    elif recording:
        st.warning(f"No new readings for {ago(age)[:-4]}. Check the Raspberry Pi's power and internet connection.")
    else:
        st.caption(f"Finished {local(info.ended_at)}, {int(info.samples)} samples.")

    screen = last.screen
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Screen detected", screen_name(screen))
    if last.camera:
        c1.caption(f"{last.detection_confidence:.0%} confidence")
    else:
        c1.caption("From the session label (no camera data)")
    c2.metric("Blue channel", f"{int(last.b)}")
    c3.metric("Ambient lux", f"{last.lux:.1f}")
    if screen in THRESHOLDS:
        minutes = float(current.loc[current["screen"] == screen, "dt"].sum() / 60)
        level = classify(screen, minutes)
        safe, moderate = THRESHOLDS[screen]
        c4.metric(f"Time on {screen}", f"{minutes:.1f} min")
        c5.markdown(
            f"<div style='font-size:0.875rem'>Exposure level</div>"
            f"<div style='font-size:2.25rem;font-weight:600;line-height:1.3;color:{LEVEL_COLORS[level]}'>{level}</div>"
            f"<div style='font-size:0.8rem;opacity:0.7'>Safe up to {safe} min, moderate up to {moderate} min</div>",
            unsafe_allow_html=True,
        )
    else:
        c4.metric("Session time", f"{current['elapsed_min'].iloc[-1]:.1f} min")
        c5.metric("Exposure level", "No screen detected")

    with_camera = current[current["camera"]]
    if info.device in THRESHOLDS and not with_camera.empty:
        agree = (with_camera["screen"] == info.device).mean()
        st.caption(f"Labelled as {info.device}. The camera detected a {info.device} in {agree:.0%} of this session's readings.")
    elif with_camera.empty:
        try:
            det = latest_detection()
        except requests.RequestException:
            det = None
        if det:
            det_age = (now - pd.to_datetime(det["ts"], utc=True).to_pydatetime()).total_seconds()
            st.caption(f"Latest separate camera detection: {screen_name(det['device'])}, {ago(det_age)}.")

    data = pd.concat(frames, ignore_index=True)
    screens = data[data["screen"].isin(THRESHOLDS)]
    if data["camera"].any() and not data["camera"].all():
        st.caption("Sessions recorded without the camera are grouped by their label.")

    st.subheader("Blue light by detected screen")
    if not data.empty:
        stats = data.groupby("screen").agg(
            readings=("b", "count"),
            minutes=("dt", lambda s: s.sum() / 60),
            blue_mean=("b", "mean"), blue_std=("b", "std"), blue_min=("b", "min"), blue_max=("b", "max"),
            lux_mean=("lux", "mean"), lux_std=("lux", "std"),
        ).round(2)
        order = [s for s in SCREENS + ["none", "unknown"] if s in stats.index]
        stats = stats.loc[order]
        stats.index = [screen_name(s) if s == "none" else s for s in stats.index]
        stats.index.name = "Detected screen"
        stats.columns = ["Readings", "Minutes", "Blue mean", "Blue SD", "Blue min", "Blue max", "Lux mean", "Lux SD"]
        st.dataframe(stats, width="stretch")

    st.subheader("Exposure classification")
    rows = []
    for (sid, scr), group in screens.groupby(["session", "screen"], sort=False):
        mins = float(group["dt"].sum() / 60)
        safe, moderate = THRESHOLDS[scr]
        rows.append({
            "Session": sid, "Detected screen": scr, "Minutes": round(mins, 2),
            "Blue mean": round(group["b"].mean(), 1),
            "Safe up to": safe, "Moderate up to": moderate, "Level": classify(scr, mins),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.write("No smartphone, laptop or television detected in the selected sessions.")

    labelled = data[data["camera"] & data["label"].isin(THRESHOLDS)]
    if not labelled.empty:
        st.subheader("Camera detection against labelled device")
        cols = [c for c in SCREENS + ["none", "unknown"] if c in set(labelled["screen"]) or c in SCREENS]
        cm = pd.crosstab(labelled["label"], labelled["screen"]).reindex(
            index=[s for s in SCREENS if s in set(labelled["label"])], columns=cols, fill_value=0)
        cm.index.name = "Labelled"
        cm.columns.name = "Detected"
        st.dataframe(cm, width="stretch")
        metrics = []
        for s in SCREENS:
            tp = int(((labelled["label"] == s) & (labelled["screen"] == s)).sum())
            fp = int(((labelled["label"] != s) & (labelled["screen"] == s)).sum())
            fn = int(((labelled["label"] == s) & (labelled["screen"] != s)).sum())
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics.append({"Device": s, "Precision": round(precision, 3), "Recall": round(recall, 3),
                            "F1": round(f1, 3), "Readings": tp + fn})
        st.dataframe(pd.DataFrame(metrics), width="stretch", hide_index=True)
        st.caption(f"Accuracy {(labelled['label'] == labelled['screen']).mean():.1%} over {len(labelled)} readings. "
                   "Each reading is one row, using the smoothed detection at that moment.")

    st.subheader("Blue light over time")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for scr in SCREENS + ["none", "unknown"]:
        part = data[data["screen"] == scr]
        if not part.empty:
            ax.scatter(part["elapsed_min"], part["b"], s=6, color=SCREEN_COLORS[scr], label=screen_name(scr))
    handles, labels_ = ax.get_legend_handles_labels()
    unique = dict(zip(labels_, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, markerscale=2, loc="upper right")
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel("Blue channel (counts)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("KDE exposure heatmap")
    options = ["All screens"] + [s for s in SCREENS if s in set(screens["screen"])]
    choice = st.selectbox("Detected screen", options, key="kde_screen")
    subset = screens if choice == "All screens" else screens[screens["screen"] == choice]
    heat = kde_figure(subset, f"Exposure density: {choice}")
    if heat:
        st.pyplot(heat)
        plt.close(heat)
    else:
        st.write("Needs at least 10 readings over time to draw the heatmap.")

    export = data[["session", "participant", "label", "distance_cm", "ts"] + READING_COLUMNS[1:]].copy()
    export["ts"] = export["ts"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    export = export.rename(columns={"ts": "timestamp"})
    st.download_button(
        f"Download {len(meta)} session{'s' if len(meta) != 1 else ''} as one CSV",
        export.to_csv(index=False).encode(),
        file_name="blue_light_sessions.csv",
        mime="text/csv",
    )


st.fragment(dashboard, run_every=interval if live else None)()