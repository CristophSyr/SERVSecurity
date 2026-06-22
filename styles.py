def get_custom_css() -> str:
    """Retorna el bloque CSS completo para inyectar en Streamlit."""
    return """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg-primary:   #060a13;
    --bg-card:      rgba(17,24,39,0.7);
    --bg-glass:     rgba(15,23,42,0.6);
    --accent-blue:  #3b82f6;
    --accent-cyan:  #06b6d4;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple:#8b5cf6;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       rgba(56,68,90,0.5);
    --glow-blue:    rgba(59,130,246,0.25);
    --glow-red:     rgba(239,68,68,0.25);
    --glow-green:   rgba(16,185,129,0.25);
    --glow-purple:  rgba(139,92,246,0.2);
  }

  /* ── Background con gradiente animado ── */
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    background-image:
      radial-gradient(ellipse 80% 50% at 20% 40%, rgba(59,130,246,0.06) 0%, transparent 50%),
      radial-gradient(ellipse 60% 40% at 80% 20%, rgba(139,92,246,0.05) 0%, transparent 50%),
      radial-gradient(ellipse 50% 60% at 50% 90%, rgba(6,182,212,0.04) 0%, transparent 50%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
  }

  /* ── Sidebar Premium ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10,14,26,0.95) 0%, rgba(15,23,42,0.95) 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px) !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
    color: var(--accent-cyan) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 0.5rem !important;
  }

  /* ── Glassmorphism Metric Cards ── */
  [data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
  }
  [data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px var(--glow-blue) !important;
    border-color: rgba(59,130,246,0.3) !important;
  }
  [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
  }
  [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
  }
  [data-testid="stMetricDelta"] > div {
    color: var(--accent-red) !important;
  }

  /* ── Headers ── */
  h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em !important;
  }

  /* ── Premium Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-cyan) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 4px 15px var(--glow-blue), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    position: relative !important;
    overflow: hidden !important;
  }
  .stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px var(--glow-blue), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    filter: brightness(1.1) !important;
  }
  .stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
  }

  /* ── Alert Box (ALERTA ACTIVA) ── */
  .alert-box {
    background: linear-gradient(135deg, rgba(30,5,5,0.9), rgba(50,10,10,0.8));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid var(--accent-red);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    animation: pulse-red 2s ease-in-out infinite, shake-alert 0.5s ease-in-out;
    box-shadow: 0 0 30px var(--glow-red), inset 0 0 60px rgba(239,68,68,0.03);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .alert-box b {
    font-size: 1.05rem;
    color: #ff6b6b;
  }
  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px var(--glow-red); border-color: rgba(239,68,68,0.5); }
    50%      { box-shadow: 0 0 50px var(--glow-red), 0 0 80px rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.8); }
  }
  @keyframes shake-alert {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-3px); }
    75% { transform: translateX(3px); }
  }

  /* ── Normal Box ── */
  .normal-box {
    background: linear-gradient(135deg, rgba(5,30,20,0.8), rgba(10,40,28,0.7));
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 4px solid var(--accent-green);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    margin: 0.3rem 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px var(--glow-green);
    display: flex;
    align-items: center;
    gap: 12px;
    animation: breathe-green 4s ease-in-out infinite;
  }
  .normal-box b { color: #34d399; }
  @keyframes breathe-green {
    0%, 100% { box-shadow: 0 4px 20px var(--glow-green); }
    50%      { box-shadow: 0 4px 30px rgba(16,185,129,0.35); }
  }

  /* ── Header Banner Premium ── */
  .header-banner {
    background: linear-gradient(135deg, rgba(10,14,26,0.9) 0%, rgba(17,24,39,0.85) 50%, rgba(10,14,26,0.9) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
  }
  .header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), var(--accent-cyan), var(--accent-purple), transparent);
    animation: shimmer-top 3s ease-in-out infinite;
  }
  @keyframes shimmer-top {
    0%, 100% { opacity: 0.6; }
    50%      { opacity: 1; }
  }
  .header-title {
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, #60a5fa, #06b6d4, #8b5cf6);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
  }
  @keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
  }

  /* ── Status badges ── */
  .badge-active {
    background: linear-gradient(135deg, #10b981, #34d399);
    color: #000;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    box-shadow: 0 0 12px var(--glow-green);
    animation: pulse-badge 2s ease-in-out infinite;
  }
  @keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 12px var(--glow-green); }
    50%      { box-shadow: 0 0 20px rgba(16,185,129,0.5); }
  }
  .badge-idle {
    background: rgba(100,116,139,0.3);
    color: #94a3b8;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: 1px solid rgba(100,116,139,0.3);
  }

  /* ── Divider ── */
  hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 1rem 0 !important;
  }

  /* ── Video Container ── */
  .video-container {
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 0 40px var(--glow-blue);
    position: relative;
  }

  /* ── Tabs Premium ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--bg-glass) !important;
    border-radius: 14px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--glow-blue) !important;
  }

  /* ── Section Titles ── */
  .section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  /* ── Stat Card (para panel derecho) ── */
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 0.8rem 0;
  }
  .stat-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
  }
  .stat-card:hover {
    border-color: rgba(59,130,246,0.3);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }
  .stat-card .stat-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
  }
  .stat-card .stat-value {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text-primary);
  }
  .stat-card .stat-value.blue { color: var(--accent-blue); }
  .stat-card .stat-value.red { color: var(--accent-red); }
  .stat-card .stat-value.green { color: var(--accent-green); }
  .stat-card .stat-value.amber { color: var(--accent-amber); }

  /* ── Event Log Items ── */
  .event-item {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    transition: all 0.2s ease;
    backdrop-filter: blur(8px);
  }
  .event-item:hover {
    border-color: rgba(59,130,246,0.3);
    transform: translateX(4px);
  }
  .event-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .event-dot.red { background: var(--accent-red); box-shadow: 0 0 8px var(--glow-red); }
  .event-dot.green { background: var(--accent-green); box-shadow: 0 0 8px var(--glow-green); }
  .event-time { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
  .event-text { color: var(--text-primary); flex: 1; }

  /* ── Scrollable log ── */
  .log-container {
    max-height: 300px;
    overflow-y: auto;
    padding-right: 8px;
  }
  .log-container::-webkit-scrollbar { width: 5px; }
  .log-container::-webkit-scrollbar-track { background: transparent; }
  .log-container::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 3px; }
  .log-container::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.5); }

  /* ── Slider styling ── */
  [data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    border-radius: 12px !important;
  }

  /* ── Radio buttons in sidebar ── */
  [data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
  }
</style>
"""
