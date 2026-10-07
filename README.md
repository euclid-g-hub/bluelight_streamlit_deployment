# Blue light exposure monitor (Streamlit Cloud)

Dashboard for the smart eyewear. The Raspberry Pi uploads sensor readings and YOLOv8 results
to Supabase; this app reads them and shows live exposure, classification, statistics and the
KDE heatmap.

## 1. Create the database (Supabase)

1. Sign up at supabase.com and create a new project (free plan). Save the database password
   somewhere, though this app does not use it.
2. Open **SQL Editor**, start a new query, paste all of `supabase_setup.sql` and click **Run**.
3. Open **Project Settings → API Keys** and copy a **secret key** (starts with `sb_secret_`).
   The legacy `service_role` key also works.
4. Copy the **Project URL** (`https://xxxx.supabase.co`), shown on the project home page or
   under **Project Settings → Data API**.

Keep the secret key private. Put it only in `supabase.env` on the Pi and in the app's Secrets,
never in a GitHub file.

## 2. Put this folder on GitHub

Create a repository on github.com and upload the files in this folder:
`streamlit_app.py`, `requirements.txt`, `supabase_setup.sql`, `.gitignore`, `README.md`.
The `.streamlit/config.toml` theme file is optional.

## 3. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io and sign in with GitHub.
2. Click **Create app**, choose your repository and branch, and set the main file to
   `streamlit_app.py`.
3. Open **Advanced settings** and paste into **Secrets**:

        SUPABASE_URL = "https://xxxx.supabase.co"
        SUPABASE_KEY = "sb_secret_..."
        TIMEZONE = "Asia/Manila"

4. Click **Deploy**. Secrets can be changed later in the app's **Settings → Secrets**.

## Using the dashboard

- **Follow current session** (on by default) shows the session the Pi is recording now.
- Turn it off to pick and compare several sessions.
- **Live refresh** updates every 2 seconds.
- **Download selected sessions (CSV)** exports the data for analysis.

The free plans sleep when unused. Open the app and the Supabase project a few minutes
before a demo.
