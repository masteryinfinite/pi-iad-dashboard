# filename: streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime

# ==================== 頁面設定 ====================
st.set_page_config(
    page_title="PI / IAD 臨床總控面板",
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 中英文切換（好體貼不同同事）
language = st.sidebar.selectbox("Language / 語言", ["中文", "English"])
if language == "中文":
    txt = {
        "title": "壓傷與失禁相關皮膚損傷 (PI/IAD) 臨床總控面板",
        "upload": "請上傳你的 PI/IAD 月報 Excel 檔案",
        "waiting": "上傳 Excel 後，即刻自動生成全院級 Dashboard！",
        "skin_title": "皮膚狀況總覽",
        "doc_title": "有 PI/IAD 病人之文件記錄及翻身遵從率",
        "coming": "更多圖表即將推出",
        "total": "總入院人數",
        "hapi": "院內新發壓傷",
        "haiad": "院內新發 IAD",
        "turning": "翻身遵從率",
        "highrisk": "需翻身高危人數",
        "download": "下載高清 Dashboard (PNG)"
    }
else:
    txt = {
        "title": "Pressure Injury & IAD Master Clinical Dashboard",
        "upload": "Upload your PI datas Excel file",
        "waiting": "Upload your Excel file — your perfect dashboard is ready!",
        "skin_title": "Skin Status Overview",
        "doc_title": "Documentation & Turning Compliance\nAmong Patients Admitted with PI or IAD",
        "coming": "More Charts\nComing Soon",
        "total": "Total Admissions",
        "hapi": "Hospital-Acquired PI",
        "haiad": "Hospital-Acquired IAD",
        "turning": "Turning Compliance",
        "highrisk": "High-Risk Patients",
        "download": "Download Dashboard (High-Res PNG)"
    }

st.title(txt["title"])
st.markdown("---")

uploaded_file = st.file_uploader(txt["upload"], type=["xlsx"])

if not uploaded_file:
    st.info(txt["waiting"])
    st.stop()

# ==================== 讀取並處理資料 ====================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_excel(file, header=None)
    return df

df = load_data(uploaded_file)
data_start_col = 2  # 從 C 欄開始

# === 核心數字 ===
total_admissions = int(df.iloc[2, data_start_col:].astype(float).sum())
without_all      = int(df.iloc[3, data_start_col:].astype(float).sum())
pi_before        = int(df.iloc[4,  data_start_col:].astype(float).sum())
iad_before       = int(df.iloc[35, data_start_col:].astype(float).sum())
new_pi           = int(df.iloc[23:33, data_start_col:].astype(float).sum().sum())
new_iad          = int(df.iloc[48:54, data_start_col:].astype(float).sum().sum())
potential_ipad   = int(df.iloc[60, data_start_col:].astype(float).sum())
turned_total     = int(df.iloc[61, data_start_col:].astype(float).sum())

# === 文件記錄率（只計有 PI 或 IAD 的病人）===
doc_raw = df.iloc[14:22, data_start_col:].astype(float).fillna(0)
doc_names_en = [
    "Braden Scale @ admission", "Patient care plan", "Braden Scale ≤16",
    "Daily PCP if ≤16", "Patient album", "Wound assessment",
    "Nursing care report", "Progress sheet"
]
doc_names_zh = [
    "入院時 Braden 評估", "護理計劃", "Braden ≤16",
    "Braden ≤16 每日計劃", "病人相冊", "傷口評估",
    "護理記錄", "進度表"
]
doc_names = doc_names_zh if language == "中文" else doc_names_en

n_pi  = int(df.iloc[4, 1])      # B5
n_iad = int(df.iloc[35, 1])     # B36
total_target = n_pi + n_iad

pi_on_admission  = df.iloc[4, data_start_col:]  > 0
iad_on_admission = df.iloc[35, data_start_col:] > 0

pi_documented  = doc_raw.loc[:, pi_on_admission].sum(axis=1)
iad_documented = doc_raw.loc[:, iad_on_admission].sum(axis=1)

pi_pct  = [round(100 * c / total_target, 1) if total_target > 0 else 0 for c in pi_documented]
iad_pct = [round(100 * c / total_target, 1) if total_target > 0 else 0 for c in iad_documented]

# === 翻身遵從率 ===
pi_series   = df.iloc[4,  data_start_col:].astype(float)
iad_series  = df.iloc[35, data_start_col:].astype(float)
risk_series = df.iloc[60, data_start_col:].astype(float)
patients_requiring_turning = int((pi_series + iad_series + risk_series).sum())
turning_compliance_rate = round(turned_total / patients_requiring_turning * 100, 1) \
                          if patients_requiring_turning > 0 else 0

# ==================== 開始畫大圖 ====================
fig = plt.figure(figsize=(24, 28))
gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.2, 1], hspace=0.5, wspace=0.35)

# 1. 皮膚狀況甜甜圈圖（小數值標籤放中間）
ax1 = fig.add_subplot(gs[0, 0])
sizes = [without_all, pi_before, iad_before, new_pi, new_iad, potential_ipad]
labels = [
    "無皮膚問題" if language=="中文" else "Without Skin Issue",
    "入院時已有壓傷" if language=="中文" else "PI on Admission",
    "入院時已有 IAD" if language=="中文" else "IAD on Admission",
    "院內新發壓傷" if language=="中文" else "Hospital-Acquired PI",
    "院內新發 IAD" if language=="中文" else "Hospital-Acquired IAD",
    "高風險需預防" if language=="中文" else "High Risk"
]
colors = ['#4472C4', '#E74C3C', '#9B59B6', '#C00000', '#2980B9', '#95A5A6']
explode = [0, 0, 0, 0.12, 0.12, 0]

wedges, _ = ax1.pie(sizes, colors=colors, startangle=90, explode=explode,
                    wedgeprops={'linewidth': 3, 'edgecolor': 'white'})

# 手動標籤（小於3%放中間）
for i, wedge in enumerate(wedges):
    ang = (wedge.theta2 - wedge.theta1)/2 + wedge.theta1
    pct = sizes[i] / total_admissions * 100
    label_text = f"{labels[i]}\n{sizes[i]:,}\n({pct:.1f}%)"

    if pct < 3:  # 小塊放甜甜圈中間
        if i == 3:   # 新發壓傷
            x, y = 0, -0.18
        elif i == 4: # 新發IAD
            x, y = 0, 0.22
        else:
            x, y = 0, 0
        fontsize = 11
        bbox = dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.95)
    else:
        x = 0.75 * np.cos(np.deg2rad(ang))
        y = 0.75 * np.sin(np.deg2rad(ang))
        fontsize = 12
        bbox = dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='gray')

    ax1.text(x, y, label_text, ha='center', va='center',
             fontsize=fontsize, fontweight='bold', bbox=bbox)

centre_circle = plt.Circle((0,0), 0.70, color='white', fc='white', linewidth=3)
ax1.add_artist(centre_circle)
ax1.set_title(f"{txt['skin_title']}\n{txt['total']}: {total_admissions:,}",
              fontsize=24, fontweight='bold', pad=30, color='#1B4F72')

# 2. 文件 + 翻身長條圖（已補返最右邊嘅「總和 %」）
ax2 = fig.add_subplot(gs[0, 1])
y_docs = np.arange(len(doc_names))
y_turn = len(doc_names) + 1.5

# 堆疊長條（紅＋紫）
ax2.barh(y_docs, pi_pct,  color='#E74C3C', height=0.6, edgecolor='black',
         label=f"壓傷病人 (n={n_pi})" if language=="中文" else f"PI on Admission (n={n_pi})")
ax2.barh(y_docs, iad_pct, left=pi_pct, color='#9B59B6', height=0.6, edgecolor='black',
         label=f"IAD 病人 (n={n_iad})" if language=="中文" else f"IAD on Admission (n={n_iad})")

# 翻身長條
turn_color = '#27AE60' if turning_compliance_rate >= 90 else '#E74C3C'
ax2.barh(y_turn, turning_compliance_rate, color=turn_color, height=0.9, linewidth=3)

# === 每個文件的 PI % 及 IAD % 標籤 ===
for i in range(len(doc_names)):
    # PI 部分
    if pi_pct[i] >= 8:
        ax2.text(pi_pct[i]/2, i, f'{pi_pct[i]}%', color='white', va='center', ha='center',
                 fontweight='bold', fontsize=11)
    elif pi_pct[i] > 0:
        ax2.text(pi_pct[i] + 2, i, f'{pi_pct[i]}%', color='#E74C3C', va='center', ha='left',
                 fontweight='bold', fontsize=10)

    # IAD 部分
    if iad_pct[i] >= 8:
        ax2.text(pi_pct[i] + iad_pct[i]/2, i, f'{iad_pct[i]}%', color='white', va='center', ha='center',
                 fontweight='bold', fontsize=11)
    elif iad_pct[i] > 0:
        ax2.text(pi_pct[i] + iad_pct[i] + 2, i, f'{iad_pct[i]}%', color='#8E44AD', va='center', ha='left',
                 fontweight='bold', fontsize=10)

    # 重點來了：補返「總和 %」（你原版最右邊嗰個數字！）
    total_this_item = pi_pct[i] + iad_pct[i]
    if total_this_item > 0:
        ax2.text(total_this_item + 3, i, f'{total_this_item:.1f}%', 
                 color='#1B4F72', va='center', ha='left', 
                 fontweight='bold', fontsize=12.5)

# 翻身遵從率標籤
ax2.text(turning_compliance_rate / 2, y_turn, f'{turning_compliance_rate:.1f}%',
         color='white', va='center', ha='center', fontweight='bold', fontsize=21)

# 其餘設定（軸、標題、圖例）
ax2.set_yticks(np.append(y_docs, y_turn))
ax2.set_yticklabels(list(doc_names) + ['每2小時翻身\n遵從率' if language=="中文" else 'Q2H Turning\nCompliance'])
ax2.invert_yaxis()
ax2.set_xlim(0, 115)   # 拉闊啲，俾到總和數字有位
ax2.set_xlabel("百分比 (%)", fontsize=14, fontweight='bold')
ax2.set_title(txt['doc_title'], fontsize=20, fontweight='bold', pad=20, color='#1B4F72')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.legend(fontsize=12, loc='lower right')

# 3 & 4. 預留位置
for i in [0, 1]:
    ax = fig.add_subplot(gs[2, i])
    ax.text(0.5, 0.5, txt['coming'], ha='center', va='center', fontsize=22, color='gray', transform=ax.transAxes)
    ax.axis('off')

plt.suptitle(f"PI/IAD Dashboard — {uploaded_file.name.replace('.xlsx','')}",
             fontsize=34, fontweight='bold', y=0.98, color='#0B3D91')

st.pyplot(fig)

# ==================== 底部 KPI 數字 ====================
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(txt['total'], f"{total_admissions:,}")
with c2:
    delta_color = "inverse" if new_pi > 0 else "off"
    st.metric(txt['hapi'], new_pi, delta="Warning" if new_pi > 0 else None, delta_color=delta_color)
with c3:
    st.metric(txt['haiad'], new_iad)
with c4:
    st.metric(txt['turning'], f"{turning_compliance_rate}%")
with c5:
    st.metric(txt['highrisk'], patients_requiring_turning)

# ==================== 下載高清圖 ====================
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
buf.seek(0)
st.download_button(
    txt['download'],
    data=buf.getvalue(),
    file_name=f"PI_IAD_Dashboard_{datetime.now():%Y%m%d_%H%M}.png",
    mime="image/png"
)


st.success("Dashboard 已生成！可直接截圖或下載高清版本發報告")


