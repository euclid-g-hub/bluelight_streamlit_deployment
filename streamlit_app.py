from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import gaussian_kde

THRESHOLDS = {"smartphone": (10, 20), "laptop": (15, 30), "television": (20, 40)}
LEVEL_COLORS = {"Safe": "#2e7d32", "Moderate": "#ef6c00", "Excessive": "#c62828"}
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
    columns = ["ts", "elapsed_s", "r", "g", "b", "w", "lux"]
    rows = get("readings", [
        ("select", ",".join(columns)),
        ("session_id", f"eq.{session_id}"),
        ("order", "elapsed_s.asc"),
    ])
    df = pd.DataFrame(rows, columns=columns)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def finished_readings(session_id):
    return fetch_readings(session_id)


@st.cache_data(ttl=2, show_spinner=False)
def live_readings(session_id):
    return fetch_readings(session_id)


@st.cache_data(ttl=2, show_spinner=False)
def latest_detection():
    rows = get("detections", [("select", "ts,device,confidence"), ("order", "ts.desc")], limit=1)
    return rows[0] if rows else None


@st.cache_data(ttl=5, show_spinner=False)
def detections_between(start_iso, end_iso):
    rows = get("detections", [
        ("select", "ts,device,confidence"),
        ("ts", f"gte.{start_iso}"),
        ("ts", f"lte.{end_iso}"),
        ("order", "ts.asc"),
    ])
    return pd.DataFrame(rows, columns=["ts", "device", "confidence"])


def local(ts):
    return ts.tz_convert(TZ).strftime("%b %d, %I:%M %p").replace(" 0", " ")


def ago(seconds):
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s ago"
    if seconds < 5400:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h ago"


def label(row):
    state = "recording" if pd.isna(row.ended_at) else f"{int(row.samples)} samples"
    return f"{row.participant}, {row.device}, {row.distance_cm:g} cm, {local(row.started_at)} ({state})"


def classify(device, minutes):
    safe, moderate = THRESHOLDS[device]
    if minutes <= safe:
        return "Safe"
    if minutes <= moderate:
        return "Moderate"
    return "Excessive"


def exposure_minutes(df):
    if df.empty:
        return 0.0
    dt = df["elapsed_s"].diff().fillna(0).clip(upper=5)
    return float(dt.sum() / 60)


def current_session(sessions):
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
    recording = sessions[sessions["ended_at"].isna() & (sessions["started_at"] >= cutoff)]
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
    ax.set_xlabel("Exposure time (min)")
    ax.set_ylabel("Blue channel (counts)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Density")
    fig.tight_layout()
    return fig


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
    names = {row.id: label(row) for row in all_sessions.itertuples()}
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
        st.info("No sessions yet. Run sensor_logger.py on the Raspberry Pi to start recording.")
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
            frames.append(df.assign(
                session=sid, participant=info.participant, device=info.device,
                distance_cm=info.distance_cm, elapsed_min=df["elapsed_s"] / 60,
            ))
    except requests.RequestException as e:
        st.error(f"Lost connection to the database. Retrying. ({e})")
        return

    now = datetime.now(timezone.utc)
    current_id = meta.index[0]
    info = meta.iloc[0]
    current = frames[0]
    recording = pd.isna(info.ended_at)

    st.subheader(label(pd.Series({**info.to_dict(), "id": current_id})).rsplit(" (", 1)[0])
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

    minutes = exposure_minutes(current)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Device in front", info.device)
    c2.metric("Blue channel", f"{int(last.b)}")
    c3.metric("Ambient lux", f"{last.lux:.1f}")
    c4.metric("Exposure time", f"{minutes:.1f} min")
    if info.device in THRESHOLDS:
        level = classify(info.device, minutes)
        safe, moderate = THRESHOLDS[info.device]
        c5.markdown(
            f"<div style='font-size:0.875rem'>Exposure level</div>"
            f"<div style='font-size:2.25rem;font-weight:600;line-height:1.3;color:{LEVEL_COLORS[level]}'>{level}</div>"
            f"<div style='font-size:0.8rem;opacity:0.7'>Safe up to {safe} min, moderate up to {moderate} min</div>",
            unsafe_allow_html=True,
        )
    else:
        c5.metric("Exposure level", "Not classified")

    try:
        det = latest_detection()
        start_iso = info.started_at.isoformat()
        end_iso = (info.ended_at if not recording else pd.Timestamp(now)).isoformat()
        window = detections_between(start_iso, end_iso)
    except requests.RequestException:
        det, window = None, pd.DataFrame(columns=["ts", "device", "confidence"])

    d1, d2 = st.columns(2)
    if det:
        det_age = (now - pd.to_datetime(det["ts"], utc=True).to_pydatetime()).total_seconds()
        text = "no device" if det["device"] == "none" else det["device"]
        d1.metric("Camera sees (YOLOv8)", text,
                  help="Latest detection uploaded by detect.py on the Raspberry Pi.")
        d1.caption(f"{det['confidence']:.0%} confidence, {ago(det_age)}")
    else:
        d1.metric("Camera sees (YOLOv8)", "No detections yet")
    if info.device in THRESHOLDS and not window.empty:
        agree = (window["device"] == info.device).mean()
        d2.metric("Camera agreed with the labelled device", f"{agree:.0%}",
                  help="Share of YOLOv8 detections during this session that matched the device you labelled.")
        n = len(window)
        d2.caption(f"{n} detection{'s' if n != 1 else ''} during this session")

    data = pd.concat(frames, ignore_index=True)
    valid = data[data["device"].isin(THRESHOLDS)]

    st.subheader("Exposure classification")
    rows = []
    for (sid, device), group in valid.groupby(["session", "device"], sort=False):
        mins = exposure_minutes(group)
        safe, moderate = THRESHOLDS[device]
        rows.append({
            "Session": sid, "Device": device, "Exposure (min)": round(mins, 2),
            "Safe up to": safe, "Moderate up to": moderate, "Level": classify(device, mins),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.write("No smartphone, laptop or television exposure in the selected sessions.")

    st.subheader("Descriptive statistics per device")
    if not data.empty:
        stats = data.groupby("device")[["b", "lux"]].agg(["count", "mean", "std", "min", "max"]).round(2)
        stats.columns = [f"{'Blue' if a == 'b' else 'Lux'} {s}" for a, s in stats.columns]
        st.dataframe(stats, width="stretch")

    st.subheader("Blue light over time")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for sid, group in data.groupby("session", sort=False):
        ax.plot(group["elapsed_min"], group["b"], linewidth=1, label=sid)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Blue channel (counts)")
    if len(meta) > 1:
        ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("KDE exposure heatmap")
    options = ["All devices"] + sorted(valid["device"].unique().tolist())
    choice = st.selectbox("Device", options, key="kde_device")
    subset = valid if choice == "All devices" else valid[valid["device"] == choice]
    heat = kde_figure(subset, f"Exposure density: {choice}")
    if heat:
        st.pyplot(heat)
        plt.close(heat)
    else:
        st.write("Needs at least 10 readings over time to draw the heatmap.")

    export = data.drop(columns=["elapsed_min"]).copy()
    export["ts"] = export["ts"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    st.download_button(
        "Download selected sessions (CSV)",
        export.to_csv(index=False).encode(),
        file_name="blue_light_sessions.csv",
        mime="text/csv",
    )


st.fragment(dashboard, run_every=interval if live else None)()
