"""
Streamlit App — Fragrance Note Similarity & Clustering
FYP - Muhammad Afif Akmal bin Nasruddin (2025385601)

Features:
  1. Dupe Finder      — find the most similar fragrances by scent DNA
  2. Scent Journey    — step-by-step path between two fragrances or two scent families
  3. Scent Map        — interactive 2D UMAP visualisation of all 24,063 fragrances
"""

import html
import math

import streamlit as st
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import normalize
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title='Scent Analytics · Fragrance Matcher',
    page_icon='🧴',
    layout='wide',
    initial_sidebar_state='auto',
)

# ─────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────
BG, PANEL, RAISED = '#13100e', '#1b1714', '#231e1a'
LINE, LINE_SOFT = '#332b25', '#272019'
INK, MUTED, FAINT = '#f1e8dc', '#a39483', '#6f6357'
GOLD = '#d9b46c'

# One colour per K-Means scent family (8 clusters from step5_clustering.py)
FAMILY_COLORS = {
    'Citrus A': '#e8c15a',
    'Citrus B': '#6fb9b1',
    'Floral A': '#e28ba8',
    'Floral B': '#b48ae0',
    'Floral C': '#f2b38b',
    'Fruity'  : '#9fd36b',
    'Spicy'   : '#d8694a',
    'Woody'   : '#bf8a57',
}

# Radar axes: each Fragrantica main accord is grouped into one of six broad axes
ACCORD_AXES = {
    'Fresh' : ['citrus', 'fresh', 'green', 'aquatic', 'marine', 'ozonic', 'aromatic', 'herbal',
               'lavender', 'salty', 'mineral', 'camphor', 'conifer', 'anis', 'metallic', 'soapy'],
    'Floral': ['floral', 'white floral', 'rose', 'yellow floral', 'iris', 'violet', 'tuberose',
               'powdery', 'aldehydic'],
    'Fruity': ['fruity', 'tropical', 'cherry', 'champagne', 'wine'],
    'Sweet' : ['sweet', 'vanilla', 'caramel', 'honey', 'almond', 'cacao', 'coffee', 'chocolate',
               'nutty', 'lactonic', 'coconut', 'rum', 'gourmand', 'creamy', 'beeswax', 'coca-cola'],
    'Woody' : ['woody', 'earthy', 'mossy', 'oud', 'patchouli', 'leather', 'smoky', 'tobacco',
               'whiskey', 'cannabis', 'clay', 'sand'],
    'Warm'  : ['warm spicy', 'fresh spicy', 'soft spicy', 'spicy', 'cinnamon', 'amber', 'balsamic',
               'musky', 'animalic', 'oriental'],
}
ACCORD_TO_AXIS = {a: axis for axis, accs in ACCORD_AXES.items() for a in accs}
AXES = list(ACCORD_AXES)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500;1,600&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --bg:{BG}; --panel:{PANEL}; --raised:{RAISED}; --line:{LINE}; --line-soft:{LINE_SOFT};
  --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --gold:{GOLD};
  --gold-soft:rgba(217,180,108,.14); --gold-line:rgba(217,180,108,.38);
  --f-display:"Cormorant Garamond",Georgia,serif;
  --f-body:"Manrope",system-ui,-apple-system,"Segoe UI",sans-serif;
  --f-mono:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
}}

/* App shell */
.stApp {{ background:var(--bg); color:var(--ink); }}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button {{ font-family:var(--f-body); }}
#MainMenu, footer, [data-testid="stDecoration"] {{ display:none; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1240px; }}
h1, h2, h3, h4 {{ font-family:var(--f-display) !important; color:var(--ink) !important; }}
hr {{ border-color:var(--line-soft) !important; }}
code {{ color:var(--gold); background:var(--raised); }}

/* Sidebar */
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#191512,#14110f); border-right:1px solid var(--line-soft); }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap:4px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  width:100%; padding:10px 12px; margin:0; border-radius:8px; border:1px solid transparent;
  color:var(--muted); transition:background .15s, color .15s; cursor:pointer;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:var(--raised); color:var(--ink); }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])),
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child:not([data-testid]) {{ display:none; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background:var(--gold-soft); border-color:var(--gold-line); color:var(--ink);
}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size:14px; font-weight:500; }}

/* Widgets */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input {{
  background:var(--panel) !important; border-color:var(--line) !important; border-radius:10px !important;
}}
[data-baseweb="select"] > div:focus-within {{ border-color:var(--gold-line) !important; box-shadow:0 0 0 3px var(--gold-soft); }}
[data-baseweb="tag"] {{ background:var(--gold-soft) !important; border:1px solid var(--gold-line); }}
[data-baseweb="tag"] span {{ color:var(--gold) !important; }}
.stButton > button, .stDownloadButton > button {{
  background:var(--raised); border:1px solid var(--line); color:var(--ink); border-radius:9px;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--gold-line); color:var(--gold); }}
[data-testid="stExpander"] details {{ background:var(--panel); border:1px solid var(--line-soft); border-radius:12px; }}
[data-testid="stWidgetLabel"] p {{ font-size:11.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--faint); font-weight:600; }}
[data-testid="stPlotlyChart"] {{ background:var(--panel); border:1px solid var(--line-soft); border-radius:14px; padding:8px; }}

/* ── Custom components ── */
.sa-brand {{ display:flex; align-items:center; gap:10px; padding:4px 0 2px; }}
.sa-brand h1 {{ margin:0 !important; padding:0 !important; font-size:25px !important; font-weight:600; color:var(--gold) !important; line-height:1; }}
.sa-brand-sub {{ margin:6px 0 18px; font-size:10.5px; letter-spacing:.18em; text-transform:uppercase; color:var(--faint); }}
.sa-foot {{ font-size:12px; color:var(--faint); line-height:1.6; border-top:1px solid var(--line-soft); padding-top:14px; margin-top:8px; }}
.sa-foot b {{ color:var(--muted); font-weight:600; }}

.sa-head {{ display:flex; flex-direction:column; gap:6px; margin-bottom:22px; }}
.sa-eyebrow {{ font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--gold); font-weight:700; }}
.sa-title {{ margin:0; font-family:var(--f-display); font-weight:600; font-size:clamp(32px,4vw,46px); line-height:1.05; color:var(--ink); }}
.sa-sub {{ margin:0; color:var(--muted); max-width:66ch; font-size:14.5px; }}
.sa-label {{ font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); font-weight:700; }}
.sa-section {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; margin:6px 0 12px; }}
.sa-section h3 {{ margin:0; font-family:var(--f-body) !important; font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted) !important; font-weight:700; }}
.sa-section span {{ font-size:12px; color:var(--faint); }}

.sa-card {{ background:var(--panel); border:1px solid var(--line-soft); border-radius:14px; padding:24px; }}
.sa-chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.sa-chip {{ display:inline-flex; align-items:center; padding:3px 10px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:var(--raised); color:var(--ink); white-space:nowrap; }}
.sa-chip.shared {{ border-color:var(--gold-line); background:var(--gold-soft); color:var(--gold); }}
.sa-chip.dim {{ color:var(--faint); }}
.sa-tag {{ display:inline-flex; align-items:center; gap:5px; padding:2px 8px; border-radius:5px; font-family:var(--f-mono); font-size:11px; border:1px solid var(--line); color:var(--muted); white-space:nowrap; }}
.sa-tag.gold {{ color:var(--gold); border-color:var(--gold-line); }}
.sa-fam {{ display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color:var(--muted); white-space:nowrap; }}
.sa-dot {{ width:8px; height:8px; border-radius:50%; flex:none; display:inline-block; }}

/* Selection card */
.sa-sel {{ display:flex; flex-direction:column; gap:20px; }}
.sa-sel-head {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
.sa-sel-brand {{ font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
.sa-sel-name {{ margin:6px 0 0; font-family:var(--f-display); font-style:italic; font-weight:600; font-size:clamp(32px,3.6vw,42px); line-height:1.02; color:var(--ink); }}
.sa-meta {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:12px; }}
.sa-tier {{ display:grid; grid-template-columns:76px minmax(0,1fr); gap:14px; align-items:start; padding:12px 0; border-top:1px solid var(--line-soft); }}
.sa-tier:first-child {{ border-top:0; padding-top:0; }}
.sa-tier b {{ display:block; font-size:12px; color:var(--ink); }}
.sa-tier small {{ font-size:11px; color:var(--faint); }}

/* Match cards */
.sa-match {{ display:grid; grid-template-columns:64px minmax(0,1fr); gap:18px; padding:18px 20px; background:var(--panel); border:1px solid var(--line-soft); border-radius:14px; margin-bottom:12px; }}
.sa-match.top {{ border-color:var(--gold-line); background:linear-gradient(135deg,rgba(217,180,108,.09),transparent 60%),var(--panel); }}
.sa-ring {{ position:relative; width:64px; height:64px; }}
.sa-ring svg {{ display:block; transform:rotate(-90deg); }}
.sa-ring span {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-family:var(--f-mono); font-size:15px; font-weight:600; color:var(--ink); }}
.sa-ring small {{ font-size:9px; color:var(--faint); margin-left:1px; }}
.sa-m-body {{ min-width:0; display:flex; flex-direction:column; gap:8px; }}
.sa-m-top {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap; }}
.sa-m-name {{ font-family:var(--f-display); font-weight:600; font-size:23px; line-height:1.05; color:var(--ink); }}
.sa-m-brand {{ font-size:12.5px; color:var(--muted); }}
.sa-m-why {{ font-size:12px; color:var(--faint); display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; }}
.sa-m-why b {{ color:var(--muted); font-weight:600; }}
.sa-rank {{ font-family:var(--f-mono); font-size:11px; color:var(--faint); }}

/* Journey */
.sa-bridge {{ display:flex; flex-wrap:wrap; gap:12px 18px; align-items:center; padding:16px 20px; border:1px solid var(--gold-line); background:linear-gradient(90deg,var(--gold-soft),transparent); border-radius:12px; margin:4px 0 22px; }}
.sa-bridge-text {{ display:flex; flex-direction:column; gap:2px; flex:1 1 240px; }}
.sa-bridge-text b {{ color:var(--gold); font-size:13px; }}
.sa-bridge-text span {{ font-size:12.5px; color:var(--muted); }}
.sa-path-scroll {{ overflow-x:auto; padding-bottom:8px; margin-bottom:18px; }}
.sa-path {{ display:grid; grid-auto-flow:column; grid-auto-columns:minmax(196px,1fr); }}
.sa-step {{ display:flex; flex-direction:column; min-width:0; }}
.sa-node {{ position:relative; display:flex; align-items:center; justify-content:center; height:44px; }}
.sa-node::before {{ content:""; position:absolute; left:0; right:0; top:50%; height:2px; background:linear-gradient(90deg,var(--from),var(--to)); }}
.sa-step:first-child .sa-node::before {{ left:50%; }}
.sa-step:last-child .sa-node::before {{ right:50%; }}
.sa-node span {{ position:relative; z-index:1; width:30px; height:30px; border-radius:50%; display:grid; place-items:center; font-family:var(--f-mono); font-size:12px; font-weight:600; background:var(--bg); border:2px solid var(--c); color:var(--ink); }}
.sa-hop {{ position:absolute; right:-20px; top:2px; z-index:2; font-family:var(--f-mono); font-size:10px; color:var(--faint); background:var(--bg); padding:0 4px; }}
.sa-step-card {{ margin:10px 6px 0; padding:16px; display:flex; flex-direction:column; gap:12px; flex:1; background:var(--panel); border:1px solid var(--line-soft); border-radius:12px; }}
.sa-step-card.ends {{ border-color:var(--c); }}
.sa-step-kind {{ font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); font-weight:700; }}
.sa-step-name {{ font-family:var(--f-display); font-weight:600; font-size:21px; line-height:1.05; color:var(--ink); }}
.sa-step-brand {{ font-size:12px; color:var(--muted); margin-top:2px; }}

/* Map / families */
.sa-fam-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:12px; }}
.sa-fam-card {{ background:var(--panel); border:1px solid var(--line-soft); border-radius:12px; padding:16px; display:flex; flex-direction:column; gap:10px; }}
.sa-fam-card .top {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }}
.sa-fam-card b {{ font-size:14px; color:var(--ink); display:flex; align-items:center; gap:8px; }}
.sa-fam-card .n {{ font-family:var(--f-mono); font-size:12px; color:var(--muted); }}
.sa-share {{ height:4px; border-radius:2px; background:var(--line-soft); overflow:hidden; }}
.sa-share i {{ display:block; height:100%; border-radius:2px; }}
.sa-fam-card .eg {{ font-size:12px; color:var(--faint); }}
.sa-fam-card .eg em {{ color:var(--muted); font-style:normal; }}
.sa-stats {{ display:flex; justify-content:space-between; font-family:var(--f-mono); font-size:12px; color:var(--faint); border-top:1px solid var(--line-soft); padding-top:12px; margin-top:4px; }}
</style>
""", unsafe_allow_html=True)


def render(markup):
    """Render HTML through st.markdown. Lines are stripped so Markdown never
    turns indented HTML into a code block."""
    st.markdown(''.join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def show_chart(fig):
    """Full-width Plotly chart; `width='stretch'` replaced use_container_width in Streamlit 1.46."""
    major, minor = (int(x) for x in st.__version__.split('.')[:2])
    if (major, minor) >= (1, 46):
        st.plotly_chart(fig, width='stretch')
    else:
        st.plotly_chart(fig, use_container_width=True)


def esc(text):
    return html.escape(str(text))


# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading fragrance database...')
def load_data():
    # The (24063 × 1649) matrix is 99.4% zeros, so it is stored and kept sparse:
    # 389 KB on disk instead of 155 MB, and small enough for free cloud hosting.
    try:
        X = sp.load_npz('fragrance_feature_matrix_sparse.npz').tocsr()
    except FileNotFoundError:
        X = sp.csr_matrix(np.load('fragrance_feature_matrix.npy'))
    df = pd.read_csv('fragrance_umap.csv')
    vocab = pd.read_csv('note_vocabulary.csv')['note'].tolist()
    X_norm = normalize(X, norm='l2')
    X_bin  = (X > 0).astype(np.float32)                    # pre-binarised for Jaccard
    bin_sz = np.asarray(X_bin.sum(axis=1)).ravel()

    # Several fragrances share a name (e.g. "Sauvage" by Dior and by others),
    # so every lookup is by row index and the label includes the brand.
    df['Label'] = df['Perfume Name'] + ' · ' + df['Brand']
    df['Year']  = df['Year'].astype('Int64')
    df['Accords'] = df[[f'Main Accord {i}' for i in range(1, 6)]].apply(
        lambda r: [a for a in r if isinstance(a, str)], axis=1)
    return X, X_norm, X_bin, bin_sz, df, np.array(vocab)

X, X_norm, X_bin, bin_sz, df, note_arr = load_data()

# Options sorted by popularity so well-known fragrances come first while typing
ORDER = df.sort_values('Rating Count', ascending=False).index.tolist()
FAMILIES = sorted(df['KMeans_Scent_Family'].unique())


def default_index(name, brand):
    hit = df[(df['Perfume Name'] == name) & (df['Brand'] == brand)]
    return int(hit.index[0]) if len(hit) else ORDER[0]


def fam_color(fam):
    return FAMILY_COLORS.get(fam, '#888888')


def notes_of(i):
    """Standardised note set for row i (from the encoded feature matrix)."""
    return set(note_arr[X[i].indices])


def split_notes(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [n.strip() for n in cell.split(',') if n.strip()]


def accord_profile(i):
    """0–10 score per radar axis, from the ranked Main Accords 1–5 (weights 5..1)."""
    score = dict.fromkeys(AXES, 0.0)
    for rank, acc in enumerate(df.at[i, 'Accords']):
        axis = ACCORD_TO_AXIS.get(acc)
        if axis:
            score[axis] += 5 - rank
    return [min(10.0, score[a] * 2) for a in AXES]


@st.cache_data
def family_summary():
    rows = []
    for fam in FAMILIES:
        sub = df[df['KMeans_Scent_Family'] == fam]
        accs = pd.Series([a for lst in sub['Accords'] for a in lst[:3]]).value_counts()
        famous = sub.sort_values('Rating Count', ascending=False)['Perfume Name'].head(3).tolist()
        rows.append(dict(family=fam, count=len(sub), accords=accs.index[:3].tolist(), examples=famous))
    return rows

# ─────────────────────────────────────────────
# SIMILARITY HELPERS
# ─────────────────────────────────────────────
def cosine_scores(idx):
    return (X_norm @ X_norm[idx].T).toarray().ravel()


def jaccard_scores(idx):
    inter = (X_bin @ X_bin[idx].T).toarray().ravel()
    union = bin_sz[idx] + bin_sz - inter
    union[union == 0] = 1
    return inter / union


def find_similar(idx, top_n=10, cw=0.6, exclude_brand=True, same_gender=False):
    cos = cosine_scores(idx)
    jac = jaccard_scores(idx)
    combined = cw * cos + (1 - cw) * jac
    combined[idx] = -1
    if exclude_brand:
        combined[(df['Brand'] == df.at[idx, 'Brand']).values] = -1
    if same_gender:
        combined[(df['Gender'] != df.at[idx, 'Gender']).values] = -1
    top_idx = np.argsort(combined)[::-1][:top_n]

    q_notes = notes_of(idx)
    rows = []
    for rank, i in enumerate(top_idx, 1):
        shared = sorted(q_notes & notes_of(i))
        rows.append({
            'Rank'          : rank,
            'idx'           : int(i),
            'Perfume Name'  : df.at[i, 'Perfume Name'],
            'Brand'         : df.at[i, 'Brand'],
            'Gender'        : df.at[i, 'Gender'],
            'Scent Family'  : df.at[i, 'KMeans_Scent_Family'],
            'Cosine'        : round(float(cos[i]), 3),
            'Jaccard'       : round(float(jac[i]), 3),
            'Combined'      : round(float(combined[i]), 3),
            'Shared Count'  : len(shared),
            'Shared Notes'  : ', '.join(shared) or '—',
        })
    return pd.DataFrame(rows), len(q_notes)

# ─────────────────────────────────────────────
# SCENT JOURNEY HELPERS
# ─────────────────────────────────────────────
UMAP = df[['UMAP_X', 'UMAP_Y']].values


def family_anchor(fam):
    """A well-known fragrance that sits near the centre of a scent family on the map."""
    members = np.where(df['KMeans_Scent_Family'].values == fam)[0]
    centre = UMAP[members].mean(axis=0)
    near = members[np.argsort(np.linalg.norm(UMAP[members] - centre, axis=1))[:300]]
    return int(near[np.argmax(df['Rating Count'].values[near])])


def scent_journey(si, ei, steps=5, prefer_popular=True):
    """
    Find a gradual path between two fragrances by hopping through
    intermediate fragrances that bridge the scent gap.
    Algorithm: at each step aim for the point that splits the remaining
    UMAP distance evenly, then take the nearest unvisited fragrance there.
    With prefer_popular, take the most-rated of the 12 nearest instead,
    so the stops are fragrances people are likely to know.
    """
    end_pos = UMAP[ei]
    visited = {si, ei}
    journey = [si]
    current = si
    for step in range(steps - 2):
        cur_pos = UMAP[current]
        target  = cur_pos + (end_pos - cur_pos) * (1 / (steps - 1 - step))
        dists = np.linalg.norm(UMAP - target, axis=1)
        dists[list(visited)] = np.inf
        if prefer_popular:
            near = np.argpartition(dists, 12)[:12]
            next_idx = int(near[np.argmax(df['Rating Count'].values[near])])
        else:
            next_idx = int(np.argmin(dists))
        visited.add(next_idx)
        journey.append(next_idx)
        current = next_idx
    journey.append(ei)
    return journey

# ─────────────────────────────────────────────
# SVG / HTML COMPONENTS
# ─────────────────────────────────────────────
def radar_svg(vals, color, size=132, ghost=None):
    c, R, n = size / 2, size / 2 - 24, len(vals)

    def pt(v, i, r=R):
        a = -math.pi / 2 + i * 2 * math.pi / n
        return c + math.cos(a) * r * v / 10, c + math.sin(a) * r * v / 10

    def poly(vs):
        return ' '.join(f'{x:.1f},{y:.1f}' for x, y in (pt(v, i) for i, v in enumerate(vs)))

    g = ''.join(f'<polygon points="{poly([lvl] * n)}" fill="none" stroke="{LINE}" stroke-width="1"/>'
                for lvl in (3.33, 6.66, 10))
    for i in range(n):
        x, y = pt(10, i)
        lx, ly = pt(12.8, i)
        g += f'<line x1="{c}" y1="{c}" x2="{x:.1f}" y2="{y:.1f}" stroke="#2a231e"/>'
        g += (f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#8b7d6f" font-size="9" text-anchor="middle" '
              f'dominant-baseline="middle" font-family="Manrope,sans-serif">{AXES[i]}</text>')
    if ghost is not None:
        g += f'<polygon points="{poly(ghost)}" fill="none" stroke="{MUTED}" stroke-dasharray="3 3" stroke-width="1"/>'
    g += (f'<polygon points="{poly(vals)}" fill="{color}" fill-opacity=".28" stroke="{color}" '
          f'stroke-width="1.6" stroke-linejoin="round"/>')
    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="display:block;margin:0 auto;max-width:100%">{g}</svg>'


def ring_svg(p, size=64):
    r = size / 2 - 5
    circ = 2 * math.pi * r
    p = max(0.0, min(1.0, p))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#2c2520" stroke-width="5"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{GOLD}" stroke-width="5" '
            f'stroke-linecap="round" stroke-dasharray="{circ*p:.1f} {circ:.1f}"/></svg>')


def fam_tag(fam):
    return f'<span class="sa-fam"><i class="sa-dot" style="background:{fam_color(fam)}"></i>{esc(fam)}</span>'


def chips(items, cls=''):
    return '<div class="sa-chips">' + ''.join(f'<span class="sa-chip {cls}">{esc(n)}</span>' for n in items) + '</div>'


def page_header(eyebrow, title, sub):
    render(f"""
    <div class="sa-head">
      <span class="sa-eyebrow">{esc(eyebrow)}</span>
      <h2 class="sa-title">{esc(title)}</h2>
      <p class="sa-sub">{esc(sub)}</p>
    </div>""")


def selection_card(i):
    row = df.loc[i]
    fam = row['KMeans_Scent_Family']
    year = '' if pd.isna(row['Year']) else f'<span class="sa-tag">{row["Year"]}</span>'

    def tier(name, sub, cell):
        notes = split_notes(cell)
        body = chips([n.title() for n in notes]) if notes else '<span class="sa-chip dim">Not listed</span>'
        return f'<div class="sa-tier"><div><b>{name}</b><small>{sub}</small></div>{body}</div>'

    render(f"""
    <div class="sa-card sa-sel">
      <div class="sa-sel-head">
        <div>
          <div class="sa-sel-brand">{esc(row['Brand'])}</div>
          <div class="sa-sel-name">{esc(row['Perfume Name'])}</div>
          <div class="sa-meta">
            <span class="sa-tag">{esc(row['Gender'].title())}</span>{year}
            <span class="sa-tag gold">★ {row['Rating']:.2f}</span>
            <span class="sa-m-why">{row['Rating Count']:,} ratings · {esc(row['Country'])}</span>
          </div>
        </div>
        {radar_svg(accord_profile(i), fam_color(fam))}
      </div>
      <div>{fam_tag(fam)}</div>
      <div>
        {tier('Top', 'first 15 min', row['Top Notes'])}
        {tier('Heart', '2–4 hours', row['Middle Notes'])}
        {tier('Base', 'drydown', row['Base Notes'])}
      </div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <span class="sa-label">Main accords, strongest first</span>
        {chips([a.title() for a in row['Accords']])}
      </div>
    </div>""")


def match_card(r, n_query_notes, top=False):
    pct = int(round(r['Combined'] * 100))
    shared = [n.title() for n in r['Shared Notes'].split(', ')] if r['Shared Count'] else []
    shared_html = chips(shared, 'shared') if shared else '<span class="sa-chip dim">Similar profile, no exact notes shared</span>'
    render(f"""
    <div class="sa-match{' top' if top else ''}">
      <div class="sa-ring">{ring_svg(r['Combined'])}<span>{pct}<small>%</small></span></div>
      <div class="sa-m-body">
        <div class="sa-m-top">
          <div>
            <div class="sa-m-name">{esc(r['Perfume Name'])}</div>
            <div class="sa-m-brand">{esc(r['Brand'])}</div>
          </div>
          <div class="sa-meta" style="margin:0"><span class="sa-rank">#{r['Rank']}</span><span class="sa-tag">{esc(r['Gender'].title())}</span></div>
        </div>
        <div class="sa-m-why">
          <span><b>{r['Shared Count']} of {n_query_notes}</b> notes shared</span>
          <span>Cosine <b>{r['Cosine']:.2f}</b> · Jaccard <b>{r['Jaccard']:.2f}</b></span>
          {fam_tag(r['Scent Family'])}
        </div>
        {shared_html}
      </div>
    </div>""")

# ─────────────────────────────────────────────
# SCENT MAP (Plotly)
# ─────────────────────────────────────────────
def base_layout(fig, height=560):
    fig.update_layout(
        plot_bgcolor=PANEL, paper_bgcolor=PANEL,
        font=dict(color=MUTED, family='Manrope, sans-serif', size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11, color=INK), itemsizing='constant'),
        xaxis=dict(showgrid=True, gridcolor='#241e19', zeroline=False, showticklabels=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='#241e19', zeroline=False, showticklabels=False, title=None),
        margin=dict(l=8, r=8, t=36, b=8),
        height=height,
        hovermode='closest',
        hoverlabel=dict(bgcolor=RAISED, bordercolor=LINE, font=dict(color=INK, family='Manrope, sans-serif')),
    )
    return fig


def build_scent_map(highlight_idx=None, sample_n=5000, mask=None, color_by='family',
                    path_idx=None, height=560):
    """Plotly UMAP map. Subsample for speed; points outside `mask` are drawn faint."""
    rng = np.random.RandomState(99)
    idx_sample = rng.choice(len(df), min(sample_n, len(df)), replace=False)
    sub = df.iloc[idx_sample]
    keep = mask[idx_sample] if mask is not None else np.ones(len(sub), bool)

    fig = go.Figure()
    if (~keep).any():
        faint = sub[~keep]
        fig.add_trace(go.Scattergl(
            x=faint['UMAP_X'], y=faint['UMAP_Y'], mode='markers', name='Filtered out',
            marker=dict(color='#3a322b', size=3, opacity=.5), hoverinfo='skip', showlegend=False))

    shown = sub[keep]
    if color_by == 'dbscan':
        groups = [(('Outliers' if c == -1 else f'Cluster {c}'), shown['DBSCAN_Cluster'] == c,
                   '#6f6357' if c == -1 else ['#d9b46c', '#6fb9b1', '#e28ba8', '#b48ae0', '#d8694a'][c % 5])
                  for c in sorted(shown['DBSCAN_Cluster'].unique())]
    else:
        groups = [(fam, shown['KMeans_Scent_Family'] == fam, fam_color(fam)) for fam in FAMILIES]

    for name, m, color in groups:
        fs = shown[m]
        if fs.empty:
            continue
        hover = ('<b>' + fs['Perfume Name'] + '</b><br>' + fs['Brand'].astype(str) +
                 '<br>' + fs['KMeans_Scent_Family'] + ' · ★ ' + fs['Rating'].round(2).astype(str))
        fig.add_trace(go.Scattergl(
            x=fs['UMAP_X'], y=fs['UMAP_Y'], mode='markers', name=f'{name} ({len(fs):,})',
            marker=dict(color=color, size=4, opacity=.7, line=dict(width=0)),
            text=hover, hovertemplate='%{text}<extra></extra>'))

    if path_idx:
        p = df.loc[path_idx]
        fig.add_trace(go.Scatter(x=p['UMAP_X'], y=p['UMAP_Y'], mode='lines', showlegend=False,
                                 line=dict(color=GOLD, width=2, dash='dot'), hoverinfo='skip'))

    if highlight_idx:
        hi = df.loc[highlight_idx]
        labels = [str(k + 1) for k in range(len(hi))] if path_idx else hi['Perfume Name']
        fig.add_trace(go.Scatter(
            x=hi['UMAP_X'], y=hi['UMAP_Y'], mode='markers+text', name='Selected',
            marker=dict(color=BG, size=13, symbol='circle', line=dict(color=GOLD, width=2)),
            text=labels, textposition='top center', textfont=dict(color=GOLD, size=11),
            customdata=hi['Perfume Name'] + ' · ' + hi['Brand'],
            hovertemplate='<b>%{customdata}</b><extra></extra>'))
    return base_layout(fig, height)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
PAGES = ['✦  Dupe Finder', '↝  Scent Journey Navigator', '⌗  Interactive Scent Map']
with st.sidebar:
    render("""
    <div class="sa-brand">
      <svg width="22" height="26" viewBox="0 0 22 26"><rect x="8" y="1" width="6" height="4" rx="1" fill="#d9b46c"/><rect x="6.5" y="5" width="9" height="2.5" rx="1" fill="#8a6f40"/><rect x="2" y="8" width="18" height="17" rx="4" fill="none" stroke="#d9b46c" stroke-width="1.6"/><path d="M5 18c3-2 9-2 12 0v3.5c0 1-1 2-2 2H7c-1 0-2-1-2-2z" fill="#d9b46c" opacity=".55"/></svg>
      <h1>Scent Analytics</h1>
    </div>
    <div class="sa-brand-sub">Fragrance Matcher</div>""")
    page = st.radio('Navigate', PAGES, label_visibility='collapsed')
    render(f"""
    <div class="sa-foot">
      <b>{len(df):,} fragrances</b> · <b>{len(note_arr):,}</b> scent notes<br>
      Cosine + Jaccard similarity<br>K-Means + DBSCAN · UMAP<br><br>
      FYP · Muhammad Afif Akmal
    </div>""")

# ─────────────────────────────────────────────
# PAGE 1: DUPE FINDER
# ─────────────────────────────────────────────
if page == PAGES[0]:
    page_header('Dupe Finder', 'Find the scent you love, from another house',
                'Pick a fragrance. We compare its top, heart and base notes against all '
                f'{len(df):,} fragrances and rank the closest matches.')

    query = st.selectbox(
        'Fragrance', ORDER, index=ORDER.index(default_index('Aventus', 'Creed')),
        format_func=lambda i: df.at[i, 'Label'],
        help='Start typing a fragrance or brand name')

    with st.expander('Search settings'):
        c1, c2 = st.columns(2)
        with c1:
            top_n = st.slider('Number of matches', 5, 20, 8)
            cw = st.slider('Cosine weight', 0.0, 1.0, 0.6, 0.05,
                           help='Higher = emphasise note layering (top / heart / base). '
                                'Lower = emphasise the plain overlap of notes (Jaccard).')
        with c2:
            exclude_brand = st.toggle('Other brands only', value=True,
                                      help="Hide flankers and variants from the same house, which aren't really dupes")
            same_gender = st.toggle('Same gender only', value=False)
            st.caption(f'Combined score = {cw:.2f} × Cosine + {1 - cw:.2f} × Jaccard')

    results, n_q = find_similar(query, top_n=top_n, cw=cw,
                                exclude_brand=exclude_brand, same_gender=same_gender)

    left, right = st.columns([5, 6], gap='large')
    with left:
        render('<div class="sa-section"><h3>Your selection</h3></div>')
        selection_card(query)
    with right:
        render(f'<div class="sa-section"><h3>Top {len(results)} closest matches</h3>'
               f'<span>Ranked by combined score</span></div>')
        for k, row in results.iterrows():
            match_card(row, n_q, top=(k == 0))

    render('<div class="sa-section" style="margin-top:18px"><h3>Where they sit on the scent map</h3>'
           '<span>Your selection and its matches, over a 5,000-fragrance sample</span></div>')
    show_chart(build_scent_map(highlight_idx=[query] + results['idx'].tolist(), height=460))

    st.download_button(
        '⬇  Download results as CSV',
        data=results.drop(columns='idx').to_csv(index=False),
        file_name=f"dupes_{df.at[query, 'Perfume Name'][:30]}.csv",
        mime='text/csv')

# ─────────────────────────────────────────────
# PAGE 2: SCENT JOURNEY
# ─────────────────────────────────────────────
elif page == PAGES[1]:
    page_header('Scent Journey Navigator', 'Move between scents one step at a time',
                'Choose where you are and where you want to go. Each stop is a real fragrance '
                'that sits between the two on the scent map, so the change never feels abrupt.')

    mode = st.radio('Plan a journey between', ['Two scent families', 'Two fragrances'],
                    horizontal=True)
    c1, c2, c3 = st.columns([5, 5, 3])
    if mode == 'Two scent families':
        with c1:
            src_fam = st.selectbox('I wear now', FAMILIES, index=FAMILIES.index('Floral A'))
        with c2:
            dst_opts = [f for f in FAMILIES if f != src_fam]
            dst_fam = st.selectbox('I want to explore', dst_opts,
                                   index=dst_opts.index('Spicy') if 'Spicy' in dst_opts else 0)
        start, end = family_anchor(src_fam), family_anchor(dst_fam)
    else:
        with c1:
            start = st.selectbox('Start fragrance', ORDER,
                                 index=ORDER.index(default_index('Coco Mademoiselle', 'Chanel')),
                                 format_func=lambda i: df.at[i, 'Label'])
        with c2:
            end = st.selectbox('Destination fragrance', ORDER,
                               index=ORDER.index(default_index('Sauvage', 'Dior')),
                               format_func=lambda i: df.at[i, 'Label'])
    with c3:
        steps = st.slider('Stops', 3, 8, 5)
    prefer_popular = st.toggle('Prefer well-known fragrances', value=True,
                               help='Pick the most-rated fragrance near each point on the path')

    if start == end:
        st.warning('Start and destination are the same fragrance. Pick a different destination.')
        st.stop()

    journey = scent_journey(start, end, steps=steps, prefer_popular=prefer_popular)
    note_sets = [notes_of(i) for i in journey]

    # Bridge notes: what the two ends have in common, or failing that, what recurs along the way
    bridge = sorted(note_sets[0] & note_sets[-1])
    if bridge:
        bridge_msg = (f'Both <b style="color:inherit">{esc(df.at[start, "Perfume Name"])}</b> and '
                      f'<b style="color:inherit">{esc(df.at[end, "Perfume Name"])}</b> contain these notes. '
                      'They are what make the change feel familiar.')
    else:
        counts = pd.Series([n for s in note_sets for n in s]).value_counts()
        bridge = counts[counts >= 2].index[:8].tolist()
        bridge_msg = 'The two ends share no notes, so these notes recurring along the path carry you across.'
    render(f"""
    <div class="sa-bridge">
      <div class="sa-bridge-text"><b>{len(bridge)} bridge notes</b><span>{bridge_msg}</span></div>
      {chips([n.title() for n in bridge[:10]], 'shared')}
    </div>""")

    profiles = [accord_profile(i) for i in journey]
    hops = [float(X_norm[a].multiply(X_norm[b]).sum()) for a, b in zip(journey, journey[1:])]
    cards = ''
    for k, i in enumerate(journey):
        fam = df.at[i, 'KMeans_Scent_Family']
        c = fam_color(fam)
        prev_c = fam_color(df.at[journey[k - 1], 'KMeans_Scent_Family']) if k else c
        next_c = fam_color(df.at[journey[k + 1], 'KMeans_Scent_Family']) if k < len(journey) - 1 else c
        kind = 'Start' if k == 0 else ('Destination' if k == len(journey) - 1 else f'Stop {k}')
        if k < len(journey) - 1:
            carry = sorted(note_sets[k] & note_sets[k + 1])
            label = 'Carries forward' if carry else 'Key notes'
            notes = carry[:4] if carry else split_notes(df.at[i, 'Top Notes'])[:4]
            cls = 'shared' if carry else ''
        else:
            label, notes, cls = 'Top notes', split_notes(df.at[i, 'Top Notes'])[:4], ''
        hop = f'<em class="sa-hop" title="Cosine similarity to the next stop">{hops[k]*100:.0f}%</em>' if k < len(hops) else ''
        cards += f"""
        <div class="sa-step" style="--c:{c};--from:{prev_c};--to:{next_c}">
          <div class="sa-node"><span>{k + 1}</span>{hop}</div>
          <div class="sa-step-card{' ends' if kind in ('Start', 'Destination') else ''}">
            <div>
              <div class="sa-step-kind">{kind}</div>
              <div class="sa-step-name">{esc(df.at[i, 'Perfume Name'])}</div>
              <div class="sa-step-brand">{esc(df.at[i, 'Brand'])} · ★ {df.at[i, 'Rating']:.2f}</div>
            </div>
            {fam_tag(fam)}
            {radar_svg(profiles[k], c, 150, profiles[k - 1] if k else None)}
            <div style="display:flex;flex-direction:column;gap:6px">
              <span class="sa-label">{label}</span>
              {chips([n.title() for n in notes] or ['—'], cls)}
            </div>
          </div>
        </div>"""

    render(f'<div class="sa-section"><h3>Your transition pathway</h3>'
           f'<span>Average similarity between stops: {np.mean(hops)*100:.0f}% · '
           f'dashed outline on each chart is the previous stop</span></div>')
    render(f'<div class="sa-path-scroll"><div class="sa-path">{cards}</div></div>')

    render('<div class="sa-section"><h3>Journey on the scent map</h3><span>Numbers match the stops above</span></div>')
    show_chart(build_scent_map(highlight_idx=journey, path_idx=journey, height=480))

# ─────────────────────────────────────────────
# PAGE 3: SCENT MAP
# ─────────────────────────────────────────────
else:
    page_header('Interactive Scent Map', 'The whole library, laid out by how it smells',
                'Every dot is one fragrance, placed by UMAP from its notes. Fragrances close together '
                'share similar notes. Hover a dot for details, scroll to zoom, drag to pan.')

    fcol, mcol = st.columns([1, 3], gap='large')
    with fcol:
        color_by = st.selectbox('Clustering algorithm', ['K-Means scent families', 'DBSCAN clusters'])
        genders = st.multiselect('Gender', ['women', 'men', 'unisex'], format_func=str.title)
        top_brands = df['Brand'].value_counts().index.tolist()
        brands = st.multiselect('Brand', top_brands, placeholder='All brands')
        min_ratings = st.select_slider('Minimum ratings', [0, 50, 100, 500, 1000, 5000], value=0)
        sample_n = st.slider('Points to show', 1000, len(df), 5000, 500,
                             help='More points = richer map but slower to render')
        search = st.selectbox('Highlight a fragrance', [None] + ORDER,
                              format_func=lambda i: '(none)' if i is None else df.at[i, 'Label'])

        mask = np.ones(len(df), bool)
        if genders:
            mask &= df['Gender'].isin(genders).values
        if brands:
            mask &= df['Brand'].isin(brands).values
        if min_ratings:
            mask &= (df['Rating Count'] >= min_ratings).values
        render(f'<div class="sa-stats"><span>Total {len(df):,}</span><span>Matching {int(mask.sum()):,}</span></div>')

    with mcol:
        fig_map = build_scent_map(highlight_idx=[search] if search is not None else None,
                                  sample_n=sample_n, mask=mask,
                                  color_by='dbscan' if color_by.startswith('DBSCAN') else 'family',
                                  height=620)
        show_chart(fig_map)

    render('<div class="sa-section" style="margin-top:12px"><h3>Scent families</h3>'
           '<span>8 K-Means clusters over the note feature matrix</span></div>')
    cards = ''
    biggest = max(f['count'] for f in family_summary())
    for f in family_summary():
        share = f['count'] / len(df) * 100
        cards += f"""
        <div class="sa-fam-card">
          <div class="top"><b><i class="sa-dot" style="background:{fam_color(f['family'])}"></i>{esc(f['family'])}</b>
          <span class="n">{f['count']:,} · {share:.1f}%</span></div>
          <div class="sa-share"><i style="width:{f['count'] / biggest * 100:.0f}%;background:{fam_color(f['family'])}"></i></div>
          {chips([a.title() for a in f['accords']])}
          <div class="eg">e.g. <em>{esc(', '.join(f['examples']))}</em></div>
        </div>"""
    render(f'<div class="sa-fam-grid">{cards}</div>')
