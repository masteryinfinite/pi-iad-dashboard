import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime

# ===================== 頁面設定 =====================
st.set_page_config(
    page_title="PI / IAD Master Clinical Dashboard",
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 語言選擇（預設英文）
language = st.sidebar.selectbox("Language／語言", ["English", "中文"], index=0)

# ===================== 文字字典 =====================
txt = {
    "title": "Pressure Injury & IAD Master Clinical Dashboard" if language == "English" else "壓傷與失禁相關皮膚損傷 臨床總控面板",
    "upload": "Upload your PI/IAD monthly Excel file" if language == "English" else "請上傳你的 PI/IAD 月報 Excel 檔案",
    "waiting": "Upload the Excel file and the dashboard will be generated instantly!" if language == "English" else "上傳 Excel 後即刻自動生成全院 Dashboard！",
    "skin_title": "Skin Status Overview" if language == "English" else "皮膚狀況總覽",
    "doc_title": "Documentation & Q2H Turning Compliance\n(Patients admitted with PI or IAD)" if language == "English" else "文件記錄及每2小時翻身遵從率\n（入院時已有 PI／IAD 病人）",
    "total": "Total Admissions",
    "hapi": "Hospital-Acquired PI",
    "haiad": "Hospital-Acquired IAD",
    "turning": "Q2H Turning Compliance",
    "highrisk": "Patients Requiring Turning",
    "download": "Download Dashboard (High-Res PNG)",
        "footer": "Pressure Injury & IAD Master Clinical Dashboard\nUnited Christian Hospital\n© 2025" if language == "English" else
              "壓傷與失禁相關皮膚損傷 臨床總控面板\n聯合醫院 United Christian Hospital\n© 2025"
}

# ===================== 主標題 =====================
st.title(txt["title"])
st.markdown("---")

uploaded_file = st.file_uploader(txt["upload"], type=["xlsx"])

if not uploaded_file:
    st.info(txt["waiting"])
    st.stop()

# ===================== 讀取資料 =====================
@st.cache_data
def load_data(file):
    return pd.read_excel(file, header=None)

df = load_data(uploaded_file)
data_start_col = 2

# ===================== 核心指標 =====================
total_admissions = int(df.iloc[2, data_start_col:].astype(float).sum())
without_all      = int(df.iloc[3, data_start_col:].astype(float).sum())
pi_before        = int(df.iloc[4,  data_start_col:].astype(float).sum())
iad_before       = int(df.iloc[35, data_start_col:].astype(float).sum())
new_pi           = int(df.iloc[23:33, data_start_col:].astype(float).sum().sum())
new_iad          = int(df.iloc[48:54, data_start_col:].astype(float).sum().sum())
potential_ipad   = int(df.iloc[60, data_start_col:].astype(float).sum())
turned_total     = int(df.iloc[61, data_start_col:].astype(float).sum())

# 文件記錄（只計有 PI 或 IAD 的病人）
doc_raw = df.iloc[14:22, data_start_col:].astype(float).fillna(0)
doc_names = [
    "Braden Scale @ admission", "Patient care plan", "Braden Scale ≤16",
    "Daily PCP if ≤16", "Patient album", "Wound assessment",
    "Nursing care report", "Progress sheet"
] if language == "English" else [
    "入院時 Braden 評估", "護理計劃", "Braden ≤16",
    "Braden ≤16 每日計劃", "病人相冊", "傷口評估",
    "護理記錄", "進度表"
]

n_pi  = int(df.iloc[4, 1])
n_iad = int(df.iloc[35, 1])
total_target = n_pi + n_iad

pi_on_admission  = df.iloc[4, data_start_col:]  > 0
iad_on_admission = df.iloc[35, data_start_col:] > 0

pi_documented  = doc_raw.loc[:, pi_on_admission].sum(axis=1)
iad_documented = doc_raw.loc[:, iad_on_admission].sum(axis=1)

pi_pct  = [round(100 * c / total_target, 1) if total_target > 0 else 0 for c in pi_documented]
iad_pct = [round(100 * c / total_target, 1) if total_target > 0 else 0 for c in iad_documented]

# 翻身遵從率
patients_requiring_turning = int((df.iloc[4, data_start_col:].astype(float) +
                                  df.iloc[35, data_start_col:].astype(float) +
                                  df.iloc[60, data_start_col:].astype(float)).sum())
turning_compliance_rate = round(turned_total / patients_requiring_turning * 100, 1) if patients_requiring_turning > 0 else 0

# ===================== 開始畫圖（已移除 Coming Soon） =====================
fig = plt.figure(figsize=(26, 18))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.4, wspace=0.35)

# ───── 1. 左上：甜甜圈圖 ─────
ax1 = fig.add_subplot(gs[0, 0])
sizes = [without_all, pi_before, iad_before, new_pi, new_iad, potential_ipad]
labels = ["Without Skin Issue","PI on Admission","IAD on Admission",
          "Hospital-Acquired PI","Hospital-Acquired IAD","High Risk"] if language=="English" else \
         ["無皮膚問題","入院時已有壓傷","入院時已有 IAD","院內新發壓傷","院內新發 IAD","高風險需預防"]
colors = ['#4472C4','#E74C3C','#9B59B6','#C00000','#2980B9','#95A5A6']
explode = [0,0,0,0.12,0.12,0]

wedges, _ = ax1.pie(sizes, colors=colors, startangle=90, explode=explode,
                    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3))

for i, wedge in enumerate(wedges):
    ang = (wedge.theta2 - wedge.theta1)/2 + wedge.theta1
    pct = sizes[i] / total_admissions * 100
    label_text = f"{labels[i]}\n{sizes[i]:,}\n({pct:.1f}%)"
    if pct < 3:
        if i==3: x,y = 0,-0.18
        elif i==4: x,y = 0,0.22
        else: x,y = 0,0
        fs, bbox_alpha = 11, 0.95
    else:
        x = 0.75 * np.cos(np.deg2rad(ang))
        y = 0.75 * np.sin(np.deg2rad(ang))
        fs, bbox_alpha = 12, 0.9
    ax1.text(x, y, label_text, ha='center', va='center', fontsize=fs,
             fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
             edgecolor='gray', alpha=bbox_alpha))

centre_circle = plt.Circle((0,0), 0.70, color='white', ec='white', lw=3)
ax1.add_artist(centre_circle)
ax1.set_title(f"{txt['skin_title']}\nTotal Admissions: {total_admissions:,}",
              fontsize=26, fontweight='bold', pad=15, color='#1B4F72')

# ───── 2. 右上：文件＋翻身長條圖（有 total %）─────
ax2 = fig.add_subplot(gs[0, 1])
y_docs = np.arange(len(doc_names))
y_turn = len(doc_names) + 1.5

ax2.barh(y_docs, pi_pct,  color='#E74C3C', height=0.6, edgecolor='black',
         label=f"PI on Admission (n={n_pi})")
ax2.barh(y_docs, iad_pct, left=pi_pct, color='#9B59B6', height=0.6, edgecolor='black',
         label=f"IAD on Admission (n={n_iad})")

turn_color = '#27AE60' if turning_compliance_rate >= 90 else '#E74C3C'
ax2.barh(y_turn, turning_compliance_rate, color=turn_color, height=0.9, linewidth=3)

# 各項標籤 + 總和 %
for i in range(len(doc_names)):
    if pi_pct[i] >= 1:
        ax2.text(pi_pct[i]/2, i, f'{pi_pct[i]}%', color='white', va='center', ha='center', fontweight='bold', fontsize=11)
    if iad_pct[i] >= 1:
        ax2.text(pi_pct[i] + iad_pct[i]/2, i, f'{iad_pct[i]}%', color='white', va='center', ha='center', fontweight='bold', fontsize=11)
    total_item = pi_pct[i] + iad_pct[i]
    if total_item > 0:
        ax2.text(total_item + 3, i, f'{total_item:.1f}%', color='#1B4F72', va='center', ha='left',
                 fontweight='bold', fontsize=13)

ax2.text(turning_compliance_rate/2, y_turn, f'{turning_compliance_rate:.1f}%',
         color='white', va='center', ha='center', fontweight='bold', fontsize=22)

ax2.set_yticks(np.append(y_docs, y_turn))
ax2.set_yticklabels(list(doc_names) + ['Q2H Turning Compliance' if language=="English" else '每2小時翻身遵從率'])
ax2.invert_yaxis()
ax2.set_xlim(0, 118)
ax2.set_xlabel("Percentage (%)", fontsize=14, fontweight='bold')
ax2.set_title(txt['doc_title'], fontsize=20, fontweight='bold', pad=20, color='#1B4F72')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.legend(fontsize=12, loc='lower right')

# ───── 3. 下方整行醫院字樣 ─────
ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')
ax3.text(0.5, 0.5, txt["footer"], ha='center', va='center', fontsize=28,
         fontweight='bold', color='#2E86C1', alpha=0.8)
plt.subplots_adjust(top=0.82)  # 整個圖留多啲頂部空間
plt.suptitle(f"PI/IAD Dashboard — {uploaded_file.name.replace('.xlsx','')}",
             fontsize=38, fontweight='bold', y=0.98, color='#0B3D91')

st.pyplot(fig)

# ===================== KPI 數字列 =====================
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric(txt['total'], f"{total_admissions:,}")
with c2:
    st.metric(txt['hapi'], new_pi,
              delta="ALERT" if new_pi > 0 else None,
              delta_color="inverse" if new_pi > 0 else "off")
with c3: st.metric(txt['haiad'], new_iad)
with c4: st.metric(txt['turning'], f"{turning_compliance_rate}%")
with c5: st.metric(txt['highrisk'], patients_requiring_turning)

# ===================== 下載高清圖 =====================
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
buf.seek(0)
st.download_button(txt['download'], data=buf.getvalue(),
                   file_name=f"PI_IAD_Dashboard_{datetime.now():%Y%m%d_%H%M}.png",
                   mime="image/png")

st.success("Dashboard generated successfully! Share this link with the whole hospital.")


