# FarmFit AI — interactive classroom prototype

This repository contains a Streamlit interface for the verified FarmFit AI Phase 2 deployment prototype.

## What the app does

- accepts N, P, K, temperature, humidity, pH and rainfall inputs;
- loads the fitted XGBoost prototype;
- displays the leading crop and top three model candidates;
- applies the frozen Strong / Alternatives / Expert Review policy;
- displays the final held-out evaluation summary from `website_data.json`.

## Evidence boundary

This is an educational decision-support prototype. It was evaluated on one structured dataset and has not been validated on independently collected farm data. It must not be used as autonomous agronomic advice.

## Run locally

Use Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Your browser should open `http://localhost:8501`.

## Deploy from GitHub using Streamlit Community Cloud

1. Create a GitHub repository, for example `farmfit-ai`.
2. Upload the **contents of this folder** to the repository root.
3. Sign in at <https://share.streamlit.io> with the same GitHub account.
4. Select **Create app** and choose the repository.
5. Select branch `main` and entrypoint file `app.py`.
6. Open **Advanced settings** and select Python 3.12.
7. Click **Deploy**.

Do not upload passwords, tokens, `.env` files or `secrets.toml` to GitHub.

## Verified evidence identity

- Evidence class: `real_data`
- Analysis run: `e1c4a416d532fea2`
- Website-data schema: `phase2.1`
