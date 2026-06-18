import streamlit as st


def apply_custom_style():
    """
    Premium dark design system — Orange accent palette, white/neutral text.
    Does NOT hide any Streamlit navigation elements.
    """
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

    <style>

    /* ── GLOBAL BACKGROUND ──────────────────────────────────────── */
    html, body, [data-testid="stApp"] {
        background: #0a0a0f !important;
        font-family: 'Inter', sans-serif !important;
        color: #f1f5f9 !important;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(234,88,12,0.18), transparent),
            radial-gradient(ellipse 50% 40% at 90% 90%,  rgba(251,146,60,0.07), transparent),
            #0a0a0f !important;
    }

    /* ── MAIN CONTENT PADDING ───────────────────────────────────── */
    .main .block-container {
        padding-top: 5.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1400px !important;
    }

    /* ── HIDE SIDEBAR & DEFAULT HEADER ──────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    #MainMenu  { visibility: hidden !important; }
    footer     { visibility: hidden !important; }
    [data-testid="stDeployButton"]  { display: none !important; }
    [data-testid="stDecoration"]    { display: none !important; }

    /* ── TOP NAV BAR ────────────────────────────────────────────── */
    .top-nav {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: rgba(13, 13, 20, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-bottom: 1px solid rgba(249, 115, 22, 0.18) !important;
        padding: 0.85rem 3rem !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999999 !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4) !important;
    }

    .nav-brand {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        font-size: 1.35rem !important;
        background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        letter-spacing: 0.05em !important;
    }

    .nav-links {
        display: flex !important;
        gap: 1.5rem !important;
        align-items: center !important;
    }

    .nav-link {
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        text-decoration: none !important;
        padding: 0.45rem 1rem !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
    }

    .nav-link:hover {
        color: #fb923c !important;
        background: rgba(249, 115, 22, 0.08) !important;
    }

    .nav-link.active {
        color: #fb923c !important;
        background: linear-gradient(135deg, rgba(234, 88, 12, 0.2), rgba(249, 115, 22, 0.1)) !important;
        border: 1px solid rgba(249, 115, 22, 0.3) !important;
        font-weight: 700 !important;
    }


    /* ── CUSTOM SCROLLBAR ─────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0d0d14; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ea580c, #f97316);
        border-radius: 3px;
    }

    /* ── TYPOGRAPHY ──────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    p, li, div, label, .stMarkdown p {
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0 !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        color: #fb923c !important;
        background: rgba(234,88,12,0.08) !important;
        border-radius: 5px !important;
        padding: 1px 5px !important;
    }

    /* ── GRADIENT TITLE ──────────────────────────────────────────── */
    .gradient-text {
        background: linear-gradient(135deg, #f97316 0%, #fb923c 40%, #fbbf24 80%, #f97316 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 900 !important;
        display: inline-block !important;
        animation: orangeShift 5s ease infinite alternate;
    }

    @keyframes orangeShift {
        0%   { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* ── GLASS CARDS ─────────────────────────────────────────────── */
    .glass-card {
        background: rgba(15,15,25,0.75) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 20px !important;
        padding: 1.75rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        transition: border-color 0.25s, background-color 0.25s, box-shadow 0.25s !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(249,115,22,0.4), transparent);
    }

    .glass-card:hover {
        border-color: rgba(249,115,22,0.25) !important;
        background: rgba(20,20,30,0.85) !important;
        box-shadow: 0 4px 30px rgba(0,0,0,0.6) !important;
    }

    /* ── METRIC CARDS ────────────────────────────────────────────── */
    .metric-card {
        background: rgba(15,15,25,0.85) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 16px !important;
        padding: 1.4rem 1.25rem !important;
        text-align: center !important;
        position: relative !important;
        overflow: hidden !important;
        transition: border-color 0.25s, background-color 0.25s !important;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent, linear-gradient(90deg,#ea580c,#f97316));
    }

    .metric-card:hover {
        border-color: rgba(249,115,22,0.25) !important;
        background: rgba(20,20,30,0.95) !important;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        line-height: 1.1 !important;
        margin: 0.4rem 0 0.2rem !important;
    }

    .metric-label {
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.09em !important;
        color: #475569 !important;
    }

    /* ── BUTTONS ─────────────────────────────────────────────────── */
    div.stButton > button {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 2rem !important;
        box-shadow: 0 4px 20px rgba(234,88,12,0.4) !important;
        transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
        letter-spacing: 0.02em !important;
    }

    div.stButton > button:hover {
        box-shadow: 0 4px 24px rgba(234,88,12,0.55) !important;
        background: linear-gradient(135deg, #f97316 0%, #fb923c 100%) !important;
    }

    div.stButton > button:active {
        background: #ea580c !important;
    }

    /* ── INPUTS ──────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(15,15,25,0.95) !important;
        border: 1px solid rgba(249,115,22,0.3) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #f97316 !important;
        box-shadow: 0 0 0 3px rgba(249,115,22,0.18) !important;
    }

    .stTextInput > div > div > input::placeholder { color: #334155 !important; }

    [data-baseweb="select"] > div {
        background: rgba(15,15,25,0.95) !important;
        border: 1px solid rgba(249,115,22,0.3) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
    }

    [data-baseweb="popover"] [role="listbox"] {
        background: #12121f !important;
        border: 1px solid rgba(249,115,22,0.2) !important;
        border-radius: 12px !important;
    }

    [data-baseweb="popover"] [role="option"]:hover {
        background: rgba(249,115,22,0.1) !important;
    }

    /* ── TABS ────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,15,25,0.7) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        gap: 3px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #475569 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,rgba(234,88,12,0.3),rgba(249,115,22,0.18)) !important;
        color: #fb923c !important;
        box-shadow: 0 2px 8px rgba(234,88,12,0.25) !important;
    }

    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── DATAFRAME ───────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(249,115,22,0.15) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] thead tr th {
        background: rgba(234,88,12,0.1) !important;
        color: #fb923c !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.77rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }

    /* ── STATUS BADGES ───────────────────────────────────────────── */
    .status-badge-passed {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        background: rgba(16,185,129,0.1) !important;
        color: #10b981 !important;
        padding: 0.4rem 1rem !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(16,185,129,0.3) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        box-shadow: 0 0 16px rgba(16,185,129,0.1) !important;
    }

    .status-badge-failed {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        background: rgba(239,68,68,0.1) !important;
        color: #ef4444 !important;
        padding: 0.4rem 1rem !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(239,68,68,0.3) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        box-shadow: 0 0 16px rgba(239,68,68,0.1) !important;
    }

    /* ── HEALTH PILLS ────────────────────────────────────────────── */
    .system-pill {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        padding: 0.7rem 1rem !important;
        margin-bottom: 0.6rem !important;
        transition: background 0.2s, border-color 0.2s !important;
    }

    .system-pill:hover {
        background: rgba(249,115,22,0.05) !important;
        border-color: rgba(249,115,22,0.2) !important;
    }

    .pill-dot {
        width: 9px !important; height: 9px !important;
        border-radius: 50% !important;
        display: inline-block !important;
        animation: pulseDot 2.5s ease-in-out infinite !important;
    }

    @keyframes pulseDot {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.5; transform:scale(0.75); }
    }

    /* ── QUALITY GATE CARDS ──────────────────────────────────────── */
    .gate-passed {
        background: linear-gradient(135deg,rgba(16,185,129,0.12),rgba(16,185,129,0.04)) !important;
        border: 2px solid rgba(16,185,129,0.45) !important;
        border-radius: 18px !important;
        padding: 1.75rem !important;
        text-align: center !important;
        animation: glowGreen 3s ease-in-out infinite alternate !important;
    }

    .gate-failed {
        background: linear-gradient(135deg,rgba(239,68,68,0.12),rgba(239,68,68,0.04)) !important;
        border: 2px solid rgba(239,68,68,0.45) !important;
        border-radius: 18px !important;
        padding: 1.75rem !important;
        text-align: center !important;
        animation: glowRed 3s ease-in-out infinite alternate !important;
    }

    @keyframes glowGreen { from {box-shadow:0 0 20px rgba(16,185,129,0.1);} to {box-shadow:0 0 50px rgba(16,185,129,0.28);} }
    @keyframes glowRed   { from {box-shadow:0 0 20px rgba(239,68,68,0.1);}  to {box-shadow:0 0 50px rgba(239,68,68,0.28);}  }

    /* ── ALERTS ──────────────────────────────────────────────────── */
    [data-testid="stInfo"]    { background:rgba(249,115,22,0.07)!important; border-left:3px solid #f97316!important; border-radius:12px!important; }
    [data-testid="stWarning"] { background:rgba(245,158,11,0.07)!important; border-left:3px solid #f59e0b!important; border-radius:12px!important; }
    [data-testid="stError"]   { background:rgba(239,68,68,0.07)!important;  border-left:3px solid #ef4444!important; border-radius:12px!important; }
    [data-testid="stSuccess"] { background:rgba(16,185,129,0.07)!important; border-left:3px solid #10b981!important; border-radius:12px!important; }

    /* ── FILE UPLOADER ───────────────────────────────────────────── */
    [data-testid="stFileUploader"] > div {
        background: rgba(15,15,25,0.7) !important;
        border: 2px dashed rgba(249,115,22,0.35) !important;
        border-radius: 16px !important;
        transition: border-color 0.3s, background 0.3s !important;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: rgba(249,115,22,0.65) !important;
        background: rgba(234,88,12,0.05) !important;
    }

    /* ── DIVIDER ─────────────────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,transparent,rgba(249,115,22,0.3),transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── SECTION LABEL ───────────────────────────────────────────── */
    .section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #f97316;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label::after {
        content:''; flex:1; height:1px;
        background: linear-gradient(90deg,rgba(249,115,22,0.2),transparent);
    }

    /* ── RESPONSE BUBBLES ────────────────────────────────────────── */
    .response-bubble {
        background: rgba(15,15,25,0.9);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 4px solid #10b981;
        border-radius: 0 18px 18px 18px;
        padding: 1.35rem 1.5rem;
        margin: 1rem 0;
        animation: fadeUp 0.4s cubic-bezier(0.34,1.56,0.64,1);
    }

    .response-bubble-refused {
        background: rgba(15,15,25,0.9);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 4px solid #ef4444;
        border-radius: 0 18px 18px 18px;
        padding: 1.35rem 1.5rem;
        margin: 1rem 0;
        animation: fadeUp 0.4s cubic-bezier(0.34,1.56,0.64,1);
    }

    @keyframes fadeUp {
        from { opacity:0; transform:translateY(10px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* ── CHUNK & CLAIM CARDS ─────────────────────────────────────── */
    .chunk-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.2s, background-color 0.2s;
    }
    .chunk-card:hover { border-color:rgba(249,115,22,0.35); background: rgba(255,255,255,0.04); }

    .claim-verified {
        background: rgba(16,185,129,0.04);
        border-left: 3px solid #10b981;
        border-radius: 4px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }

    .claim-unverified {
        background: rgba(239,68,68,0.04);
        border-left: 3px solid #ef4444;
        border-radius: 4px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }

    /* ── FEATURE & STEP ITEMS ────────────────────────────────────── */
    .feature-item {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1.1rem;
        transition: border-color 0.25s, background-color 0.25s;
    }
    .feature-item:hover { border-color:rgba(249,115,22,0.35); background: rgba(255,255,255,0.04); }

    .step-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 0.75rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 0.5rem;
    }

    </style>
    """, unsafe_allow_html=True)


def render_header(active_page: str):
    """
    Renders a premium horizontal navigation header.
    """
    home_cls = "nav-link active" if active_page == "home" else "nav-link"
    upload_cls = "nav-link active" if active_page == "upload" else "nav-link"
    chat_cls = "nav-link active" if active_page == "chat" else "nav-link"
    analytics_cls = "nav-link active" if active_page == "analytics" else "nav-link"
    evaluation_cls = "nav-link active" if active_page == "evaluation" else "nav-link"

    st.markdown(f"""
    <div class="top-nav">
        <div class="nav-brand">🧬 RIS</div>
        <div class="nav-links">
            <a href="/" target="_self" class="{home_cls}">🏠 Overview</a>
            <a href="/upload_page" target="_self" class="{upload_cls}">📤 Upload</a>
            <a href="/chat_page" target="_self" class="{chat_cls}">💬 Chat</a>
            <a href="/analytics_page" target="_self" class="{analytics_cls}">📊 Analytics</a>
            <a href="/evaluation_page" target="_self" class="{evaluation_cls}">🧬 Evaluation</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_nav():
    """
    Deprecated — sidebar navigation is handled by render_header().
    """
    pass
