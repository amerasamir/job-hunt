import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import json
import io
from datetime import datetime
from duckduckgo_search import DDGS
import PyPDF2

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Hunt AI — مصر وإنجلترا",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Cairo', sans-serif; direction: rtl; }

.main-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 1rem 0 0.25rem;
}
.subtitle { text-align: center; color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }

.job-card {
    background: #fff; border: 1px solid #e0e0e0;
    border-radius: 12px; padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.job-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.1); }

.badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    margin-left: 4px; margin-bottom: 2px;
}
.badge-eg  { background:#e8f5e9; color:#2e7d32; }
.badge-uk  { background:#e3f2fd; color:#1565c0; }
.badge-visa{ background:#fff3e0; color:#e65100; }
.badge-match{ background:#f3e5f5; color:#6a1b9a; }
.badge-applied { background:#e3f2fd; color:#1565c0; }
.badge-interview { background:#fff3e0; color:#e65100; }
.badge-offer { background:#e8f5e9; color:#2e7d32; }
.badge-rejected { background:#ffebee; color:#c62828; }

.match-bar-bg {
    background:#f0f0f0; border-radius:4px; height:6px; margin:6px 0;
}
.metric-box {
    background:#f8f9fa; border-radius:10px; padding:1rem;
    text-align:center; border:1px solid #eee;
}
.metric-num { font-size:2rem; font-weight:700; color:#1a73e8; }
.metric-lbl { font-size:0.82rem; color:#666; }

.sidebar-section { 
    background:#f8f9fa; border-radius:10px; 
    padding:0.75rem 1rem; margin-bottom:1rem; 
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "profile": {},
        "cv_text": "",
        "analysis": "",
        "jobs": [],
        "tracker": [],
        "gemini_key": "",
        "searched": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def gemini_call(prompt: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return resp.text.strip()

def search_jobs(query: str, max_results: int = 8) -> list[dict]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results, timelimit="m"):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results

def status_badge(status: str) -> str:
    colors = {
        "محفوظة":   "badge-applied",
        "قدّمت":    "badge-applied",
        "انترفيو":  "badge-interview",
        "عرض وصل": "badge-offer",
        "مرفوض":    "badge-rejected",
    }
    css = colors.get(status, "badge-applied")
    return f'<span class="badge {css}">{status}</span>'

def tracker_to_df() -> pd.DataFrame:
    if not st.session_state.tracker:
        return pd.DataFrame(columns=["الوظيفة", "الشركة", "المكان", "الحالة", "تاريخ الإضافة", "ملاحظات", "الرابط"])
    return pd.DataFrame(st.session_state.tracker)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 إعدادات API")
    with st.container():
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="احصل على مفتاح مجاني من aistudio.google.com",
            value=st.session_state.gemini_key,
        )
        if api_key:
            st.session_state.gemini_key = api_key
            st.success("✅ المفتاح محفوظ")

    st.divider()
    st.markdown("### 👤 البروفايل")

    name = st.text_input("الاسم الكامل", placeholder="Ahmed Mohamed",
                         value=st.session_state.profile.get("name", ""))
    field = st.text_input("المجال / التخصص",
                          placeholder="Software Engineer, Data Scientist...",
                          value=st.session_state.profile.get("field", ""))
    exp = st.selectbox("سنين الخبرة",
                       ["اختر...", "0-1 سنة (Entry)", "2-4 سنين (Mid)", "5-8 سنين (Senior)", "9+ سنين (Lead)"],
                       index=["اختر...", "0-1 سنة (Entry)", "2-4 سنين (Mid)", "5-8 سنين (Senior)", "9+ سنين (Lead)"]
                       .index(st.session_state.profile.get("exp", "اختر...")))
    linkedin = st.text_input("لينك LinkedIn", placeholder="linkedin.com/in/yourprofile",
                             value=st.session_state.profile.get("linkedin", ""))

    st.markdown("### 🌍 الأسواق المستهدفة")
    want_eg = st.checkbox("🇪🇬 مصر", value=st.session_state.profile.get("want_eg", True))
    want_uk = st.checkbox("🇬🇧 إنجلترا + Visa Sponsorship",
                          value=st.session_state.profile.get("want_uk", True))

    if st.button("💾 حفظ البروفايل", use_container_width=True):
        st.session_state.profile = {
            "name": name, "field": field, "exp": exp,
            "linkedin": linkedin, "want_eg": want_eg, "want_uk": want_uk,
        }
        st.success("✅ تم حفظ البروفايل!")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 Job Hunt AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ارفع سيفيك — الـ AI هيلاقيلك أحسن وظايف في مصر وإنجلترا</div>',
            unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 السيفي والتحليل",
    "🔍 الوظايف",
    "🏢 شركات الفيزا",
    "📊 تراكر التقديمات",
])

# ═══════════════════════════════════════════════════════
# TAB 1 — CV Upload & AI Analysis
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("#### ارفع السيفي بتاعك")

    col_up, col_txt = st.columns([1, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader("ارفع ملف PDF", type=["pdf"])
        if uploaded:
            with st.spinner("جاري قراءة الـ PDF..."):
                st.session_state.cv_text = extract_pdf_text(uploaded)
            st.success(f"✅ تم استخراج {len(st.session_state.cv_text)} حرف من السيفي")

    with col_txt:
        manual_cv = st.text_area(
            "أو الصق نص السيفي هنا",
            height=200,
            placeholder="ضع هنا الخبرات، المهارات، التعليم...",
            value=st.session_state.cv_text,
        )
        if manual_cv != st.session_state.cv_text:
            st.session_state.cv_text = manual_cv

    st.divider()

    if st.button("🤖 حلّل السيفي بالـ AI", type="primary", use_container_width=True):
        if not st.session_state.gemini_key:
            st.error("⚠️ أدخل Gemini API Key في الـ Sidebar الأول!")
        elif not st.session_state.cv_text:
            st.warning("⚠️ ارفع السيفي أو الصق النص الأول!")
        else:
            p = st.session_state.profile
            markets = ", ".join(filter(None, [
                "مصر" if p.get("want_eg") else "",
                "إنجلترا مع Visa Sponsorship" if p.get("want_uk") else "",
            ])) or "مصر وإنجلترا"

            prompt = f"""أنت خبير توظيف متخصص في سوق العمل في {markets}.

بيانات المرشح:
- الاسم: {p.get('name', 'غير محدد')}
- المجال: {p.get('field', 'غير محدد')}
- الخبرة: {p.get('exp', 'غير محدد')}
- LinkedIn: {p.get('linkedin', 'غير متاح')}

محتوى السيفي:
{st.session_state.cv_text[:2000]}

اكتب تحليل احترافي باللغة العربية يشمل:
## 💪 نقاط القوة الرئيسية
(3-4 نقاط محددة من السيفي)

## 🎯 أنسب المسميات الوظيفية
(5 مسميات مع نسبة تطابق تقريبية %)

## 📈 مهارات يُنصح بتطويرها
(3 مهارات لتقوية الملف)

## 💡 نصايح لسوق {markets}
(3 نصايح عملية ومحددة)

اكتب بأسلوب مباشر ومشجع."""

            with st.spinner("🤖 الـ AI بيحلل سيفيك..."):
                try:
                    result = gemini_call(prompt, st.session_state.gemini_key)
                    st.session_state.analysis = result
                except Exception as e:
                    st.error(f"خطأ في Gemini API: {e}")

    if st.session_state.analysis:
        st.markdown("---")
        st.markdown("#### 📋 نتيجة التحليل")
        st.markdown(st.session_state.analysis)

# ═══════════════════════════════════════════════════════
# TAB 2 — Job Search
# ═══════════════════════════════════════════════════════
with tab2:
    p = st.session_state.profile
    field_val = p.get("field", "")

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_field = st.text_input("المجال / الكلمة المفتاحية",
                                     value=field_val,
                                     placeholder="Software Engineer, Data Analyst...")
    with col_s2:
        search_market = st.selectbox("السوق", ["الاتنين", "مصر فقط", "إنجلترا فقط"])
    with col_s3:
        num_results = st.slider("عدد النتايج", 5, 20, 10)

    search_clicked = st.button("🔍 ابحث عن وظايف", type="primary", use_container_width=True)

    if search_clicked and search_field:
        queries = []
        if search_market in ("الاتنين", "مصر فقط"):
            queries.append((f"{search_field} jobs Cairo Egypt 2024 site:linkedin.com OR site:wuzzuf.net", "egypt"))
        if search_market in ("الاتنين", "إنجلترا فقط"):
            queries.append((f"{search_field} jobs London UK visa sponsorship 2024 site:linkedin.com OR site:indeed.co.uk", "uk"))

        all_raw = []
        with st.spinner("🔍 جاري البحث..."):
            for q, market in queries:
                try:
                    results = search_jobs(q, max_results=num_results // len(queries) + 2)
                    for r in results:
                        r["market"] = market
                    all_raw.extend(results)
                except Exception as e:
                    st.warning(f"تعذّر البحث ({market}): {e}")

        if all_raw and st.session_state.gemini_key:
            snippets = "\n".join([
                f"- {r['title']} | {r['market'].upper()} | {r['url']}\n  {r['snippet'][:120]}"
                for r in all_raw[:15]
            ])
            enrich_prompt = f"""بناءً على نتايج البحث دي، قيّم كل وظيفة وأرجع JSON فقط (بدون أي نص تاني):

نتايج البحث:
{snippets}

معلومات المرشح:
- المجال: {search_field}
- الخبرة: {p.get('exp', 'غير محدد')}
- السيفي (مختصر): {st.session_state.cv_text[:500]}

أرجع array JSON بالشكل ده:
[
  {{
    "title": "اسم الوظيفة",
    "company": "اسم الشركة",
    "location": "المدينة والدولة",
    "market": "egypt أو uk",
    "match": رقم من 50 لـ 98,
    "visa": true أو false,
    "url": "الرابط",
    "why": "جملة واحدة ليه مناسبة للمرشح"
  }}
]
أرجع JSON فقط بدون أي markdown أو شرح."""

            with st.spinner("🤖 الـ AI بيقيّم الوظايف..."):
                try:
                    raw = gemini_call(enrich_prompt, st.session_state.gemini_key)
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    jobs = json.loads(raw)
                    for j in jobs:
                        j["id"] = f"{j.get('title','')}_{j.get('company','')}".replace(" ", "_")[:40]
                    st.session_state.jobs = jobs
                    st.session_state.searched = True
                except Exception as e:
                    st.error(f"خطأ في تحليل النتايج: {e}")
                    st.session_state.jobs = all_raw
        elif all_raw:
            st.session_state.jobs = [
                {"title": r["title"], "company": "", "location": "",
                 "market": r["market"], "match": 70, "visa": False,
                 "url": r["url"], "why": r["snippet"][:100]}
                for r in all_raw
            ]
            st.session_state.searched = True

    # ── Filter bar ──
    if st.session_state.jobs:
        st.markdown(f"**{len(st.session_state.jobs)} وظيفة** — فلتر:")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_market = st.selectbox("السوق", ["الكل", "مصر", "إنجلترا"], key="f_market")
        with fc2:
            f_visa = st.checkbox("Visa Sponsorship فقط", key="f_visa")
        with fc3:
            f_match = st.slider("Match % أدنى", 0, 90, 0, key="f_match")

        filtered = st.session_state.jobs
        if f_market == "مصر":
            filtered = [j for j in filtered if j.get("market") == "egypt"]
        elif f_market == "إنجلترا":
            filtered = [j for j in filtered if j.get("market") == "uk"]
        if f_visa:
            filtered = [j for j in filtered if j.get("visa")]
        filtered = [j for j in filtered if j.get("match", 0) >= f_match]

        st.markdown(f"**{len(filtered)} نتيجة بعد الفلتر**")
        st.markdown("---")

        for j in filtered:
            match = j.get("match", 70)
            bar_color = "#4caf50" if match >= 80 else "#ff9800" if match >= 60 else "#f44336"
            visa_badge = '<span class="badge badge-visa">✈️ Visa</span>' if j.get("visa") else ""
            mkt_badge = ('<span class="badge badge-eg">🇪🇬 مصر</span>' if j.get("market") == "egypt"
                         else '<span class="badge badge-uk">🇬🇧 UK</span>')

            st.markdown(f"""
<div class="job-card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:6px;">
    <div>
      <strong style="font-size:1rem">{j.get('title','')}</strong><br>
      <span style="color:#555; font-size:0.88rem">{j.get('company','')} · {j.get('location','')}</span>
    </div>
    <div>
      <span class="badge badge-match">🎯 {match}%</span>
      {mkt_badge} {visa_badge}
    </div>
  </div>
  <div class="match-bar-bg" style="margin:8px 0 4px;">
    <div style="background:{bar_color}; height:6px; border-radius:4px; width:{match}%;"></div>
  </div>
  <div style="font-size:0.85rem; color:#555; margin-top:4px;">{j.get('why','')}</div>
</div>
""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                if st.button("➕ أضف للتراكر", key=f"add_{j.get('id', j.get('title',''))[:20]}"):
                    entry = {
                        "الوظيفة": j.get("title", ""),
                        "الشركة": j.get("company", ""),
                        "المكان": j.get("location", ""),
                        "الحالة": "محفوظة",
                        "تاريخ الإضافة": datetime.now().strftime("%Y-%m-%d"),
                        "ملاحظات": "",
                        "الرابط": j.get("url", ""),
                    }
                    existing = [t["الوظيفة"] for t in st.session_state.tracker]
                    if entry["الوظيفة"] not in existing:
                        st.session_state.tracker.append(entry)
                        st.success("✅ أُضيفت للتراكر!")
                    else:
                        st.info("موجودة بالفعل في التراكر")
            with c2:
                if j.get("url"):
                    st.link_button("🔗 افتح الوظيفة", j["url"])

    elif st.session_state.searched:
        st.info("مفيش نتايج مع الفلاتر دي، جرب تغيّر الفلتر.")
    else:
        st.info("👆 ابحث عن وظايف بالأعلى")

# ═══════════════════════════════════════════════════════
# TAB 3 — Visa Sponsorship Companies
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🏢 شركات UK بتعمل Visa Sponsorship")
    st.caption("قائمة شركات Tech في إنجلترا معروفة بتعمل Skilled Worker Visa")

    p = st.session_state.profile
    visa_field = st.text_input("المجال للبحث", value=p.get("field", "Software Engineer"),
                               key="visa_field")

    if st.button("🤖 ابحث بالـ AI عن شركات الفيزا", type="primary"):
        if not st.session_state.gemini_key:
            st.error("⚠️ أدخل Gemini API Key في الـ Sidebar!")
        else:
            prompt = f"""أنت متخصص في سوق العمل البريطاني وتأشيرات العمل.

اعمل قائمة بـ 20 شركة حقيقية وموثوقة في إنجلترا بتعمل Skilled Worker Visa Sponsorship 
لمجال: {visa_field}

لكل شركة اكتب بالعربي:
1. اسم الشركة (إنجليزي)
2. المدينة الرئيسية
3. حجم الشركة (Startup / Mid-size / Enterprise)
4. نوع الوظايف المطلوبة في مجال {visa_field}
5. رابط صفحة الوظايف

رتّب الشركات من الأكثر توظيفاً للأجانب.
اكتب بأسلوب واضح ومنظم."""

            with st.spinner("🔍 جاري البحث..."):
                try:
                    result = gemini_call(prompt, st.session_state.gemini_key)
                    st.markdown(result)
                except Exception as e:
                    st.error(f"خطأ: {e}")

    st.divider()
    st.markdown("#### 📌 موارد مهمة للفيزا")

    resources = [
        ("🔗 Register of Licensed Sponsors", "https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers"),
        ("🔗 Indeed UK — Visa Sponsorship Jobs", "https://uk.indeed.com/jobs?q=visa+sponsorship"),
        ("🔗 LinkedIn UK Jobs", "https://www.linkedin.com/jobs/search/?location=United%20Kingdom"),
        ("🔗 Glassdoor UK", "https://www.glassdoor.co.uk/Job/visa-sponsorship-jobs-SRCH_KO0,16.htm"),
        ("🔗 Gov.uk — Skilled Worker Visa", "https://www.gov.uk/skilled-worker-visa"),
    ]
    for label, url in resources:
        st.markdown(f"- [{label}]({url})")

# ═══════════════════════════════════════════════════════
# TAB 4 — Application Tracker
# ═══════════════════════════════════════════════════════
with tab4:
    tracker = st.session_state.tracker

    # Metrics
    total = len(tracker)
    applied = sum(1 for t in tracker if t["الحالة"] == "قدّمت")
    interviews = sum(1 for t in tracker if t["الحالة"] == "انترفيو")
    offers = sum(1 for t in tracker if t["الحالة"] == "عرض وصل")

    m1, m2, m3, m4 = st.columns(4)
    for col, num, label in zip(
        [m1, m2, m3, m4],
        [total, applied, interviews, offers],
        ["إجمالي الوظايف", "قدّمت", "انترفيوهات", "عروض وصلت"],
    ):
        with col:
            st.markdown(f"""
<div class="metric-box">
  <div class="metric-num">{num}</div>
  <div class="metric-lbl">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    if not tracker:
        st.info("👆 أضف وظايف من تاب الوظايف، أو أضف يدوي من هنا")

    # Manual add
    with st.expander("➕ إضافة وظيفة يدوي"):
        with st.form("manual_add"):
            mc1, mc2 = st.columns(2)
            with mc1:
                m_title   = st.text_input("اسم الوظيفة *")
                m_company = st.text_input("الشركة *")
                m_loc     = st.text_input("المكان")
            with mc2:
                m_status = st.selectbox("الحالة", ["محفوظة", "قدّمت", "انترفيو", "عرض وصل", "مرفوض"])
                m_url    = st.text_input("رابط الوظيفة")
                m_notes  = st.text_input("ملاحظات")
            if st.form_submit_button("➕ أضف", use_container_width=True):
                if m_title and m_company:
                    st.session_state.tracker.append({
                        "الوظيفة": m_title, "الشركة": m_company,
                        "المكان": m_loc, "الحالة": m_status,
                        "تاريخ الإضافة": datetime.now().strftime("%Y-%m-%d"),
                        "ملاحظات": m_notes, "الرابط": m_url,
                    })
                    st.success("✅ أُضيفت!")
                    st.rerun()
                else:
                    st.warning("اكتب اسم الوظيفة والشركة على الأقل")

    # Table
    if tracker:
        st.markdown("#### قائمة التقديمات")
        for i, entry in enumerate(st.session_state.tracker):
            with st.container():
                ec1, ec2, ec3, ec4 = st.columns([3, 2, 2, 1])
                with ec1:
                    st.markdown(f"**{entry['الوظيفة']}**  \n{entry['الشركة']} · {entry['المكان']}")
                with ec2:
                    new_status = st.selectbox(
                        "الحالة",
                        ["محفوظة", "قدّمت", "انترفيو", "عرض وصل", "مرفوض"],
                        index=["محفوظة", "قدّمت", "انترفيو", "عرض وصل", "مرفوض"].index(entry["الحالة"]),
                        key=f"status_{i}",
                        label_visibility="collapsed",
                    )
                    if new_status != entry["الحالة"]:
                        st.session_state.tracker[i]["الحالة"] = new_status
                        st.rerun()
                with ec3:
                    note = st.text_input("ملاحظات", value=entry.get("ملاحظات", ""),
                                         key=f"note_{i}", label_visibility="collapsed",
                                         placeholder="ملاحظات...")
                    st.session_state.tracker[i]["ملاحظات"] = note
                with ec4:
                    if st.button("🗑️", key=f"del_{i}", help="احذف"):
                        st.session_state.tracker.pop(i)
                        st.rerun()
                if entry.get("الرابط"):
                    st.caption(f"[🔗 افتح الوظيفة]({entry['الرابط']})")
                st.divider()

        # Download CSV
        df = tracker_to_df()
        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ تحميل كـ CSV",
            data=csv_data.encode("utf-8-sig"),
            file_name=f"job_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
