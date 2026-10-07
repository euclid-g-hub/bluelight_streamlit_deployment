# Blue light exposure monitor (Streamlit Cloud)

Dashboard for the smart eyewear. The Raspberry Pi uploads each reading with the screen the
camera detected; this app groups blue light by detected screen type, classifies exposure,
compares detections with the labelled device, and draws the KDE heatmap.

## Updating an existing app

1. In Supabase, open **SQL Editor**, paste `supabase_update.sql` and click **Run**.
2. In your GitHub repository, upload the new `streamlit_app.py` (and the other files) over the
   old ones and commit. Streamlit Cloud redeploys automatically within a minute or two.

## First-time setup

1. Create a Supabase project. In **SQL Editor**, run `supabase_setup.sql`.
2. Copy the **Project URL** and a **secret key** (`sb_secret_...`) from **Project Settings**.
3. Upload this folder's files to a GitHub repository, with `streamlit_app.py` at the top level.
4. On share.streamlit.io, create an app from the repository with main file `streamlit_app.py`,
   and paste into **Advanced settings → Secrets**:

        SUPABASE_URL = "https://xxxx.supabase.co"
        SUPABASE_KEY = "sb_secret_..."
        TIMEZONE = "Asia/Manila"

## Using the dashboard

- **Follow current session** shows the session the Pi is recording now.
- Turn it off and tick **Select all sessions**, then use the download button for one CSV
  with every reading.
- **Blue light by detected screen** is the main result table: readings, minutes, blue and lux
  statistics per detected screen type.
- **Camera detection against labelled device** appears for labelled sessions and gives the
  confusion matrix, precision, recall, F1 and accuracy per reading.

The free plans sleep when unused. Open the app a few minutes before a demo.
