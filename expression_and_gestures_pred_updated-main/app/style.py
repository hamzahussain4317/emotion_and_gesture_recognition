import streamlit as st


def inject():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;500;600&family=Inter:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, sans-serif;
            color: #1A1A1A;
            background-color: #FAFAF7;
        }

        .stApp { background-color: #FAFAF7; }

        h1, h2, h3, h4 {
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 500;
            letter-spacing: -0.01em;
            color: #1A1A1A;
        }

        h1 { font-size: 2.1rem !important; }
        h2 { font-size: 1.5rem !important; margin-top: 1.2rem !important; }
        h3 { font-size: 1.15rem !important; }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        section[data-testid="stSidebar"] {
            background-color: #F2F0E9;
            border-right: 1px solid #E5E3DC;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        .stButton > button {
            background-color: #2F5D50;
            color: #FAFAF7;
            border: none;
            border-radius: 2px;
            padding: 0.5rem 1.2rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        .stButton > button:hover {
            background-color: #244940;
            color: #FAFAF7;
        }

        .stDataFrame, .stTable {
            border: 1px solid #E5E3DC;
            border-radius: 2px;
        }

        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E3DC;
            padding: 0.9rem 1rem;
            border-radius: 2px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #666;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Fraunces', Georgia, serif;
            color: #1A1A1A;
        }

        .card {
            background: #FFFFFF;
            border: 1px solid #E5E3DC;
            padding: 1.2rem 1.4rem;
            border-radius: 2px;
            margin-bottom: 1rem;
        }

        .teamline {
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #777;
            margin-top: -0.5rem;
            margin-bottom: 1.4rem;
        }

        .accent { color: #2F5D50; font-weight: 500; }

        hr { border: none; border-top: 1px solid #E5E3DC; margin: 1.4rem 0; }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background-color: #FFFFFF !important;
            border: 1px dashed #C9C6BC !important;
            color: #1A1A1A !important;
        }
        [data-testid="stFileUploader"] * { color: #1A1A1A !important; }
        [data-testid="stFileUploader"] button {
            background-color: #2F5D50 !important;
            color: #FAFAF7 !important;
            border: none !important;
        }

        div[data-baseweb="notification"] {
            background-color: #F2F0E9 !important;
            color: #1A1A1A !important;
            border: 1px solid #E5E3DC !important;
        }
        div[data-baseweb="notification"] * { color: #1A1A1A !important; }

        [data-testid="stCameraInput"] button {
            background-color: #2F5D50 !important;
            color: #FAFAF7 !important;
            border: none !important;
        }

        .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #E5E3DC; }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #555 !important;
            font-weight: 500;
            padding: 0.4rem 0.2rem;
        }
        .stTabs [aria-selected="true"] {
            color: #2F5D50 !important;
            border-bottom: 2px solid #2F5D50 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
