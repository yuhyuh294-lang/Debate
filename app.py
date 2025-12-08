
import streamlit as st
from PIL import Image
import base64
import os
from dotenv import load_dotenv # Giữ lại để hỗ trợ chạy local
import io
import time
import re
import random 
import json 

# ---------- XỬ LÝ GITHUB TOKEN (BẢO MẬT) ----------
try:
    # 1. Ưu tiên đọc từ Streamlit Secrets khi deploy công khai
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except:
    # 2. Fallback cho môi trường local (chạy bằng file .env)
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    st.error("Lỗi xác thực: GITHUB_TOKEN chưa được thiết lập. Vui lòng kiểm tra file .streamlit/secrets.toml trên Streamlit Cloud hoặc file .env khi chạy local.")
    st.stop()

# ---------- OpenAI client for GitHub AI ----------
from openai import OpenAI

GITHUB_BASE_URL = "https://models.github.ai/inference"
client = OpenAI(base_url=GITHUB_BASE_URL, api_key=GITHUB_TOKEN)

# ----------------------------------------------------------------------------------------------------
# CUSTOM CSS & CONFIG
# ----------------------------------------------------------------------------------------------------
st.set_page_config(page_title="🤖 AI Debate Bot", layout="wide")

# CSS cho bong bóng chat A (xanh) và B (đỏ)
CHAT_STYLE = """
<style>
/* 1. Thay đổi màu sắc tổng thể và Font (Dark Mode) */
.stApp {
    background-color: #0d1117; /* Màu nền Dark Mode */
    color: #c9d1d9; /* Chữ sáng */
}
h1, h2, h3, h4, h5, h6 {
    color: #58a6ff; /* Màu xanh nổi bật cho tiêu đề */
}

/* 2. Kiểu dáng Bong bóng Chat */
.chat-bubble {
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 16px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5); /* Shadow đậm hơn cho Dark Mode */
}

/* Bong bóng Bên A (Left - Màu xanh lá/Bạn) */
.chat-left {
    background-color: #1f362d; /* Xanh lá đậm cho Dark Mode */
    color: #4cd964 !important; /* Chữ xanh lá sáng */
    margin-right: auto;
    border-top-left-radius: 2px;
}
.chat-left b {
    color: #58a6ff !important; /* Tên màu xanh dương nổi bật */
}

/* Bong bóng Bên B (Right - Màu đỏ/AI/Phản đối) */
.chat-right {
    background-color: #3b2225; /* Đỏ đậm cho Dark Mode */
    color: #ff9500 !important; /* Chữ cam/vàng sáng */
    margin-left: auto;
    border-top-right-radius: 2px;
}
.chat-right b {
    color: #58a6ff !important; /* Tên màu xanh dương nổi bật */
}

/* Bong bóng Bên C (User) */
.chat-user {
    background-color: #192f44; /* Xanh dương đậm */
    color: #8bb8e8 !important; /* Chữ xanh dương nhạt sáng */
    margin-left: auto;
    border-top-right-radius: 2px;
}
.chat-user b {
    color: #c9d1d9 !important; /* Tên màu trắng */
}

.chat-container {
    display: flex;
    width: 100%;
    margin-bottom: 10px;
}

/* 3. Cải tiến HP Bar (Visuals) */
.hp-bar-container {
    background-color: #1e2d42; /* Nền tối của bar */
    border-radius: .35rem; /* Bo góc nhẹ */
    height: 1.8rem; /* Cao hơn một chút */
    overflow: hidden;
    margin-bottom: 15px;
    border: 2px solid #58a6ff; /* Border nổi bật */
    box-shadow: 0 0 5px rgba(88, 166, 255, 0.5); /* Hiệu ứng sáng */
}
.hp-bar-fill {
    height: 100%;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 14px;
}

/* 4. Kiểu cho thông báo ưu thế (Advantage Box) */
.advantage-box {
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 20px;
}

.advantage-A {
    background-color: #0e4429; /* Xanh lá đậm */
    color: #4cd964;
    border: 1px solid #1f362d;
}

.advantage-B {
    background-color: #58161b; /* Đỏ/Rượu vang đậm */
    color: #ff9500;
    border: 1px solid #3b2225;
}

.advantage-draw {
    background-color: #423200; /* Vàng/Nâu đậm */
    color: #ffd60a;
    border: 1px solid #332700;
}
</style>
"""
st.markdown(CHAT_STYLE, unsafe_allow_html=True)
st.title("🤖 AI Debate Bot – Thiết lập tranh luận")


# ---------- SESSION INIT ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

# Khởi tạo tất cả session state cần thiết
for key in [
    "topic_used", "final_style", "dialog_A", "dialog_B", "dialog_C",
    "topic", "uploaded_image", "chosen_style", "custom_style", "persona1",
    "persona2", "rounds", "temperature", "model_text", "debate_running",
    "suggested_topics", "current_turn_index", "is_fast_mode", "max_tokens_per_turn",
    "courtroom_analysis", "debate_mode", "A_HP", "B_HP", "rpg_log", "user_input_C",
    "C_persona"
]:
    if key not in st.session_state:
        st.session_state[key] = None

if "dialog_A" not in st.session_state or st.session_state.dialog_A is None:
    st.session_state.dialog_A = []
    st.session_state.dialog_B = []
    st.session_state.dialog_C = [] # Thêm dialog cho bên C

# Đặt giá trị mặc định
if st.session_state.max_tokens_per_turn is None:
    st.session_state.max_tokens_per_turn = 600
if st.session_state.temperature is None:
    st.session_state.temperature = 0.6
if st.session_state.rounds is None:
    st.session_state.rounds = 3
if st.session_state.debate_mode is None:
    st.session_state.debate_mode = "Tranh luận 2 AI (Tiêu chuẩn)"
if st.session_state.C_persona is None:
    st.session_state.C_persona = "Người dùng (Thành viên C)"


# ----------------------------------------------------------------------------------------------------
# API CALLS
# ----------------------------------------------------------------------------------------------------
def call_chat(messages, model, temperature=0.6, max_tokens=600):
    """Gọi GitHub AI GPT-4.1 hoặc các model GitHub"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Lỗi tạo nội dung của Tòa án AI/Bất kỳ nơi nào khác sẽ được trả về với marker này
        st.error(f"Lỗi khi gọi API Text ({model}): {e}. Vui lòng kiểm tra lại GITHUB_TOKEN hoặc chọn Model nhẹ hơn.")
        return f"[[LỖI TẠO NỘI DUNG - API CALL FAILED]]"

# ----------------------------------------------------------------------------------------------------
# RPG DAMAGE ANALYSIS (TÍNH NĂNG MỚI)
# ----------------------------------------------------------------------------------------------------
def rpg_damage_analysis(attacker_name, defender_name, last_reply, final_style, full_transcript_segment):
    """Gọi AI để tính toán damage và crit hit."""
    
    prompt = f"""
    Bạn là hệ thống tính toán sát thương (Damage Calculator) trong Game Tranh luận. 
    Phân tích lời nói gần nhất của {attacker_name} đối với {defender_name} theo phong cách '{final_style}'.

    Đánh giá độ mạnh của lập luận {attacker_name} (chặt chẽ, logic, bất ngờ) trên thang điểm 1-10.
    1-4: Sát thương yếu (Damage 5-10 HP)
    5-7: Sát thương trung bình (Damage 11-19 HP)
    8-9: Sát thương mạnh (Damage 20-25 HP)
    10: Chí mạng (Crit Hit - Damage 30-40 HP)

    Chỉ trả lời bằng JSON sau, không thêm lời giải thích nào khác:
    {{
        "strength_score": [Điểm 1-10],
        "damage": [Số HP sát thương],
        "is_crit": [true/false],
        "log_message": "Tóm tắt ngắn gọn lập luận gây sát thương này."
    }}
    
    Lập luận gần nhất: "{last_reply}"
    """
    
    # Sử dụng model mạnh hơn cho logic phức tạp
    raw_json = call_chat(
        [{"role": "user", "content": prompt}],
        model="openai/gpt-4o-mini", 
        temperature=0.3, 
        max_tokens=250
    )
    
    # Xử lý kết quả JSON
    try:
        # Làm sạch chuỗi để đảm bảo nó là JSON hợp lệ (loại bỏ các text thừa)
        json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            # Chuyển đổi damage sang int an toàn
            if 'damage' in data:
                data['damage'] = int(data['damage'])
            return data
        else:
            raise ValueError("Không tìm thấy JSON hợp lệ.")
    except Exception as e:
        # Fallback logic nếu AI trả về lỗi hoặc không phải JSON
        st.warning(f"Lỗi phân tích JSON RPG: {e}. Sử dụng damage mặc định.")
        damage_base = random.randint(5, 15)
        is_crit = random.random() < 0.15 
        damage = damage_base * 2 if is_crit else damage_base
        return {
            "strength_score": 5,
            "damage": damage,
            "is_crit": is_crit,
            "log_message": f"Hệ thống tính toán thất bại, sát thương mặc định: {damage} HP."
        }


# ----------------------------------------------------------------------------------------------------
# AI COURTROOM ANALYIS FUNCTION (FIXED)
# ----------------------------------------------------------------------------------------------------
def ai_courtroom_analysis(full_transcript, final_style, persona1, persona2, model_text):
    """Gọi AI để phân tích lập luận chi tiết theo vai trò Thẩm phán, Công tố viên và Luật sư."""
    
    prompt = f"""
    Bạn là Thẩm phán AI tối cao, chuyên gia về logic và tranh luận. Nhiệm vụ của bạn là phân tích cuộc tranh luận dưới đây giữa Bên A ({persona1}) và Bên B ({persona2}) dựa trên phong cách '{final_style}'.

    Hãy thực hiện phân tích theo CẤU TRÚC (sử dụng Markdown heading):

    ### 1. Phân tích Lập luận Logic (Judge/Thẩm phán)
    - **Điểm mạnh (Logic):** Đánh giá 3 điểm lập luận logic tốt nhất của cả hai bên.
    - **Lỗi ngụy biện (Fallacies):** Phân tích và chỉ rõ các lỗi ngụy biện (ví dụ: Ad hominem, Strawman, Appeal to authority, Gish gallop, Red herring) được sử dụng bởi A và B. Nếu có, hãy chỉ rõ đoạn đối thoại cụ thể có lỗi.
    - **Phán quyết:** Tổng kết, đưa ra phán quyết cuối cùng dựa trên tính chặt chẽ của lập luận (Ai thắng?).

    ---

    ### 2. Vai trò Công tố viên (AI Prosecutor)
    - **Mục tiêu:** Chỉ rõ 3 lỗ hổng/điểm yếu lớn nhất trong lập luận của BÊN THẮNG CUỘC (theo phán quyết).
    - **Cáo trạng:** Đưa ra cáo trạng về luận điểm yếu nhất mà bên thắng cuộc cần phải trả lời.

    ---

    ### 3. Vai trò Luật sư bào chữa (AI Lawyer)
    - **Mục tiêu:** Đưa ra 3 điểm bào chữa mạnh mẽ nhất cho BÊN THUA CUỘC (theo phán quyết).
    - **Tư vấn cải thiện:** Đưa ra lời khuyên (gồm 3 gạch đầu dòng) để bên thua cuộc cải thiện lập luận của mình trong các cuộc tranh luận tiếp theo.

    Transcript:
    {full_transcript}
    """
    
    # Đảm bảo nội dung không bị rỗng
    if not full_transcript.strip():
        return "[[LỖI TẠO NỘI DUNG]] - Transcript rỗng hoặc lỗi."

    result = call_chat(
        [{"role": "user", "content": prompt}],
        model=model_text, 
        temperature=0.3, 
        max_tokens=2000 
    )
    
    # Kiểm tra lỗi tạo nội dung
    if "[[LỖI TẠO NỘI DUNG]]" in result:
        st.error("Lỗi: AI không thể hoàn thành Phân tích Tòa án. Vui lòng thử lại.")
        return "[[LỖI TẠO NỘI DUNG - PHÂN TÍCH THẤT BẠI]]"
        
    return result

# ----------------------------------------------------------------------------------------------------
# GENERATE AI REPLY (MODULAR & CLEANER) - THAY THẾ CHO HÀM generate_debate_turn CŨ
# ----------------------------------------------------------------------------------------------------
def generate_ai_reply(persona_role, persona_name, last_reply_content, final_style, model_text, temperature, max_tokens_per_turn):
    """Gọi AI tạo phản hồi cho một bên cụ thể."""
    prompt = f"""
    Bạn là Bên {persona_role} ({persona_name}). Hãy phản biện lời nói gần nhất của đối thủ trong 3-5 câu. 
    Sử dụng phong cách '{final_style}'. 
    Lời nói gần nhất của đối thủ: '{last_reply_content}'
    """
    return call_chat(
        [{"role": "user", "content": prompt}],
        model=model_text, temperature=temperature, max_tokens=max_tokens_per_turn
    )


# ------------------- GENERATE SUGGESTED TOPICS / TOPIC FROM IMAGE (GIỮ LẠI) -------------------
def generate_topic_from_image(uploaded_image):
    # ... (Giữ nguyên hàm này)
    img_b64 = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
    msg = [
        {"role": "system", "content": "Hãy tạo 1 câu hỏi làm chủ đề debate thú vị và gây tranh cãi dựa trên ảnh."},
        {"role": "user", "content": [
            {"type": "text", "text": "Tạo chủ đề debate từ ảnh này:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}
    ]
    try:
        topic = call_chat(msg, model="openai/gpt-4o-mini", max_tokens=200, temperature=0.5)
        return topic
    except Exception as e:
        st.warning(f"Không thể tạo chủ đề từ ảnh (Multimodal): {e}")
        return "Không xác định chủ đề."

def generate_suggested_topics(model):
    # ... (Giữ nguyên hàm này)
    prompt = "Gợi ý 5 chủ đề tranh luận gây tranh cãi, thú vị, ngắn gọn, mỗi chủ đề trên 1 dòng. Đảm bảo KHÔNG có số hoặc dấu gạch đầu dòng."
    suggestions = call_chat([{"role": "user", "content": prompt}], model, 0.9, 300)
    cleaned_suggestions = re.sub(r'^\s*[\d\.\-\*]\s*', '', suggestions, flags=re.MULTILINE)
    topics_list = [t.strip() for t in cleaned_suggestions.split('\n') if t.strip()]
    return topics_list

# ----------------------------------------------------------------------------------------------------
# RPG ADVANTAGE CHECK (TÍNH NĂNG MỚI ĐÁP ỨNG YÊU CẦU)
# ----------------------------------------------------------------------------------------------------
def check_rpg_advantage(hp_a, hp_b, persona1, persona2):
    """Kiểm tra và hiển thị bên đang thắng thế dựa trên HP."""
    if hp_a > hp_b:
        diff = hp_a - hp_b
        msg = f"🟢 **ƯU THẾ!** Phe **{persona1} (A)** đang thắng thế với chênh lệch {diff} HP."
        style = "advantage-A"
    elif hp_b > hp_a:
        diff = hp_b - hp_a
        msg = f"🔴 **ƯU THẾ!** Phe **{persona2} (B)** đang thắng thế với chênh lệch {diff} HP."
        style = "advantage-B"
    else:
        msg = "🟡 **NGANG NHAU!** HP của hai phe đang cân bằng."
        style = "advantage-draw"
    
    st.markdown(f"""<div class="advantage-box {style}">{msg}</div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------
# PAGE 1 — HOME 
# ----------------------------------------------------------------------------------------------------
def render_home():
    
    st.subheader("1) Chế độ Tranh luận")
    
    debate_modes = [
        "Tranh luận 2 AI (Tiêu chuẩn)",
        "Tranh luận 1v1 với AI",
        "Chế độ RPG (Game Tranh luận)",
        "Tham gia 3 bên (Thành viên C)"
    ]
    st.session_state.debate_mode = st.selectbox(
        "Chọn chế độ:",
        debate_modes,
        index=debate_modes.index(st.session_state.debate_mode) if st.session_state.debate_mode in debate_modes else 0,
        help="RPG Mode: Lập luận gây sát thương. 1v1/3 bên: Cho phép người dùng nhập câu trả lời."
    )
    
    # --- Cài đặt Nâng cao (Sidebar) ---
    st.sidebar.header("⚙️ Cài đặt Nâng cao")
    st.session_state.model_text = st.sidebar.selectbox(
        "Model:",
        ["openai/gpt-4.1", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"],
        index=0
    )
    st.session_state.temperature = st.sidebar.slider(
        "Độ sáng tạo (Temperature)", 0.0, 1.0, st.session_state.temperature, key="temp_home"
    )
    st.session_state.rounds = st.sidebar.slider(
        "Số lượt Debate ban đầu (A → B)", 1, 10, st.session_state.rounds, key="rounds_home"
    )
    st.session_state.max_tokens_per_turn = st.sidebar.slider(
        "Giới hạn độ dài mỗi lượt nói (Tokens)", 100, 1000, 600, step=50, help="Số token tối đa cho mỗi câu trả lời của A hoặc B."
    )


    st.subheader("2) Chủ đề tranh luận")
    
    col_t1, col_t2 = st.columns([4, 1])
    with col_t1:
        st.session_state.topic = st.text_input("Nhập chủ đề tranh luận:", value=st.session_state.topic if st.session_state.topic else "")
    with col_t2:
        st.write(" ")
        st.write(" ")
        if st.button("💡 Gợi ý chủ đề"):
            with st.spinner("Đang tạo chủ đề thú vị..."):
                st.session_state.suggested_topics = generate_suggested_topics(st.session_state.model_text)
                
    if st.session_state.get('suggested_topics'):
        st.markdown("<p><b>Chọn từ gợi ý:</b></p>", unsafe_allow_html=True)
        selected_topic = st.radio("Danh sách chủ đề gợi ý:", st.session_state.suggested_topics, index=None, key="radio_topics", label_visibility="collapsed")
        col_select, col_copy = st.columns(2)
        if selected_topic and col_select.button("✅ Chọn chủ đề này"):
            st.session_state.topic = selected_topic
            st.session_state.suggested_topics = None
            st.rerun()
        if col_copy.button("📋 Sao chép danh sách"):
            st.code("\n".join(st.session_state.suggested_topics))
            st.success("Đã sao chép danh sách gợi ý!")


    st.session_state.uploaded_image = st.file_uploader("Hoặc upload ảnh gợi ý chủ đề:", type=["jpg", "jpeg", "png"])
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Ảnh đã upload", width=200)

    st.header("3) Phong cách tranh luận")
    preset_styles = ["Trang trọng – Học thuật", "Hài hước", "Hỗn loạn", "Triết gia", "Anime", "Rapper", "Lịch sự – Ngoại giao", "Văn học cổ điển", "Lãng mạn", "Khác"]
    st.session_state.chosen_style = st.selectbox("Chọn phong cách:", preset_styles)
    st.session_state.custom_style = st.text_input("Nhập phong cách riêng:") if st.session_state.chosen_style == "Khác" else ""

    st.header("4) Tính cách các bên (Persona)")
    col_p1, col_p2 = st.columns(2)
    
    # --- LOGIC HIỂN THỊ PERSONA ĐÃ SỬA ---
    
    # Bên A (Ủng hộ) luôn là AI
    with col_p1:
        st.session_state.persona1 = st.text_input(
            "Bên A (Ủng hộ):", 
            st.session_state.get('persona1', "Bình tĩnh, logic"), 
            key="persona1_input",
            help="Tính cách, vai trò, quan điểm sơ bộ của bên A (AI)."
        )
        
    # Bên B (Phản đối) thay đổi tùy theo chế độ
    with col_p2:
        if st.session_state.debate_mode == "Tranh luận 1v1 với AI":
            # Ẩn ô nhập liệu và gán Bên B là Người dùng
            st.info("**Bên B (Phản đối)** là **Bạn** (Người dùng).")
            st.session_state.persona2 = "Người dùng (Phản đối)" # Gán vai trò cho B
            
        else:
            # Hiển thị ô nhập liệu cho Bên B (AI) trong các chế độ khác
            st.session_state.persona2 = st.text_input(
                "Bên B (Phản đối):", 
                st.session_state.get('persona2', "Năng nổ, phản biện"),
                key="persona2_input",
                help="Tính cách, vai trò, quan điểm sơ bộ của bên B (AI)."
            )

    # Bên C (Người dùng) chỉ hiển thị trong chế độ 3 bên
    if st.session_state.debate_mode == "Tham gia 3 bên (Thành viên C)":
        st.session_state.C_persona = st.text_input(
            "Bên C (Người dùng):", 
            st.session_state.get('C_persona', "Bên thứ ba/Đa chiều"), 
            key="C_persona_input",
            help="Bạn sẽ tham gia với tư cách C. (Tính cách, vai trò của bạn)."
        )
    else:
        # Đảm bảo C_persona được reset hoặc không tồn tại khi không ở chế độ 3 bên
        if 'C_persona' in st.session_state:
            del st.session_state['C_persona']

    st.markdown("---")

    if st.button("▶️ Bắt đầu tranh luận", type="primary", use_container_width=True):
        if not st.session_state.topic and not st.session_state.uploaded_image:
            st.error("Vui lòng nhập chủ đề hoặc upload ảnh để bắt đầu!")
            return
            
        # Reset các biến trạng thái
        st.session_state.dialog_A = []
        st.session_state.dialog_B = []
        st.session_state.dialog_C = []
        st.session_state.courtroom_analysis = None
        st.session_state.debate_running = True
        st.session_state.current_turn_index = 0
        st.session_state.is_fast_mode = False
        st.session_state.A_HP = 100
        st.session_state.B_HP = 100
        st.session_state.rpg_log = []
        st.session_state.user_input_C = ""
        st.session_state.page = "debate"
        st.rerun()

# ----------------------------------------------------------------------------------------------------
# PAGE 2 — DEBATE (FIXED LOGIC)
# ----------------------------------------------------------------------------------------------------
def render_debate():

    st.title("🔥 Cuộc tranh luận")

    # Lấy thông tin từ session state
    topic = st.session_state.topic
    uploaded_image = st.session_state.uploaded_image
    final_style = st.session_state.custom_style if st.session_state.custom_style and st.session_state.custom_style.strip() else st.session_state.chosen_style
    st.session_state.final_style = final_style
    persona1 = st.session_state.persona1
    persona2 = st.session_state.persona2
    persona_C = st.session_state.get('C_persona', "") 
    rounds = st.session_state.rounds
    temperature = st.session_state.temperature
    model_text = st.session_state.model_text
    max_tokens_per_turn = st.session_state.max_tokens_per_turn
    debate_mode = st.session_state.debate_mode
    
    # ------------------- SHOW INFO (Sidebar) -------------------
    st.sidebar.header("📌 Thiết lập")
    st.sidebar.info(f"**Chế độ:** {debate_mode}")
    st.sidebar.markdown(f"**Chủ đề:** {st.session_state.topic_used if st.session_state.topic_used else 'Đang tạo...'}")
    st.sidebar.markdown(f"**Phong cách:** *{final_style}*")
    st.sidebar.markdown(f"**A:** {persona1} | **B:** {persona2}")
    if debate_mode == "Tham gia 3 bên (Thành viên C)":
        st.sidebar.markdown(f"**C:** {persona_C} (Bạn)")
    st.sidebar.button("🔙 Về trang thiết lập", on_click=lambda: setattr(st.session_state, 'page', 'home'))

    st.header(f"Chủ đề: {st.session_state.topic_used if st.session_state.topic_used else 'Đang tạo...'}")
    st.markdown("---")

    # ------------------- TẠO TRANSCRIPT (ĐỂ DÙNG CHO RPG & TÒA ÁN) -------------------
    full_transcript_list = []
    max_len_trans = max(len(st.session_state.dialog_A), len(st.session_state.dialog_B), len(st.session_state.dialog_C)) 
    for i in range(max_len_trans):
        # A's turn
        if i < len(st.session_state.dialog_A):
             full_transcript_list.append(f"A{i+1} ({persona1}): {st.session_state.dialog_A[i]}")
        # B's turn
        if i < len(st.session_state.dialog_B):
             full_transcript_list.append(f"B{i+1} ({persona2}): {st.session_state.dialog_B[i]}")
        # C's turn
        if i < len(st.session_state.dialog_C) and debate_mode == "Tham gia 3 bên (Thành viên C)":
             full_transcript_list.append(f"C{i+1} ({persona_C}): {st.session_state.dialog_C[i]}")

    full_transcript = "\n".join(full_transcript_list)
    
    # ------------------- HIỂN THỊ HP (CHẾ ĐỘ RPG) VÀ THÔNG BÁO ƯU THẾ -------------------
    game_over = False 
    if debate_mode == "Chế độ RPG (Game Tranh luận)":
        st.subheader("⚔️ HP (Hit Points)")
        col_hp_a, col_hp_b = st.columns(2)
        
        # Hàm hiển thị HP bar
        def display_hp(col, name, hp):
            # Xanh lá > Cam > Đỏ
            hp_color = "#4cd964" if hp > 70 else ("#ff9500" if hp > 30 else "#ff3b30") 
            hp_percent = max(0, hp)
            with col:
                st.markdown(f"**{name}** ({max(0, hp)} HP)")
                st.markdown(f"""
                <div class="hp-bar-container">
                    <div class="hp-bar-fill" style="width: {hp_percent}%; background-color: {hp_color}; background: linear-gradient(to right, {hp_color}, {hp_color}cc);">
                        {max(0, hp)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        display_hp(col_hp_a, persona1, st.session_state.A_HP)
        display_hp(col_hp_b, persona2, st.session_state.B_HP)
        
        # Thêm tính năng thông báo ưu thế theo thời gian thực
        check_rpg_advantage(st.session_state.A_HP, st.session_state.B_HP, persona1, persona2)
        
        st.markdown("---")

        # Kiểm tra kết thúc game
        if st.session_state.A_HP <= 0 and st.session_state.B_HP <= 0:
             st.error("🏳️ **HÒA!** Cả hai bên đều đã hết máu.")
             st.session_state.debate_running = False
             game_over = True
        elif st.session_state.A_HP <= 0:
            st.error(f"🏆 **CHIẾN THẮNG!** {persona2} đã thắng bằng lập luận sắc bén!")
            st.session_state.debate_running = False
            game_over = True
        elif st.session_state.B_HP <= 0:
            st.error(f"🏆 **CHIẾN THẮNG!** {persona1} đã thắng bằng lập luận sắc bén!")
            st.session_state.debate_running = False
            game_over = True

    # ------------------- LOGIC TẠO NỘI DUNG (INIT & TURN) -------------------
    
    # Hàm áp dụng sát thương (FIXED)
    def apply_rpg_damage(turn_index, attacker_role, receiver_role, attack_content, attacker_persona, receiver_persona, style, current_transcript):
        """Tính toán và áp dụng sát thương, cập nhật log."""
        
        # Tên log để kiểm tra tính duy nhất (chỉ tính 1 lần/lượt)
        log_msg_base = f"Lượt {turn_index+1} ({attacker_role} -> {receiver_role})" 
        if any(log_msg_base in log for log in st.session_state.rpg_log):
            return

        try:
             damage_data = rpg_damage_analysis(attacker_persona, receiver_persona, attack_content, style, current_transcript)
        except NameError:
             damage_data = {'damage': 10, 'is_crit': False, 'log_message': 'Hệ thống tính toán thất bại, sát thương mặc định.'}


        damage_value = damage_data['damage']
        log_icon = "🔴" if receiver_role == "A" else "🟢"
        
        # Cập nhật HP
        if receiver_role == "A":
            st.session_state.A_HP = max(0, st.session_state.A_HP - damage_value)
            st.sidebar.markdown(f"**A {persona1}** nhận **-{damage_value}** HP!")
        elif receiver_role == "B":
            st.session_state.B_HP = max(0, st.session_state.B_HP - damage_value)
            st.sidebar.markdown(f"**B {persona2}** nhận **-{damage_value}** HP!")

        # Cập nhật Log
        st.session_state.rpg_log.append(
            f"{log_icon} {log_msg_base}: {attacker_persona} gây **{damage_value}** sát thương. "
            f"{'🔥 Chí mạng!' if damage_data['is_crit'] else ''} Lời: *{damage_data['log_message']}*"
        )
        
    # Hàm thực hiện một lượt AI (FIXED)
    def execute_ai_turn(persona_role, last_reply_content, current_transcript):
        """Tạo reply và tính RPG (nếu cần) cho một AI (A hoặc B)."""
        
        # Xác định persona
        if persona_role == 'A':
            persona = persona1
            opponent_persona = persona2
            opponent_role = 'B'
        elif persona_role == 'B':
            persona = persona2
            opponent_persona = persona1
            opponent_role = 'A'
        else:
            return "" # Trả về rỗng nếu không phải A hoặc B
        
        # 1. Tạo nội dung reply
        reply = generate_ai_reply(
            persona_role, persona, last_reply_content,
            final_style, model_text, temperature, max_tokens_per_turn
        )
        
        # 2. Cập nhật dialog
        if persona_role == 'A':
            st.session_state.dialog_A.append(reply)
        elif persona_role == 'B':
            st.session_state.dialog_B.append(reply)
        
        # 3. Tính RPG (nếu cần)
        if debate_mode == "Chế độ RPG (Game Tranh luận)":
            current_turn_idx = len(st.session_state.dialog_A) - 1 if persona_role == 'A' else len(st.session_state.dialog_B) - 1
            
            # Attacker là persona_role, Receiver là opponent_role
            apply_rpg_damage(
                current_turn_idx, persona_role, opponent_role, 
                reply, persona, opponent_persona, final_style, current_transcript
            )

        return reply 

    # Hàm thêm lượt mới (Wrapper) - Dùng cho 2 AI / RPG
    def add_next_turn_wrapper_ai_only():
        st.session_state.courtroom_analysis = None
        
        # Lấy lời nói cuối cùng của đối thủ của A (có thể là B)
        last_reply_for_A = st.session_state.dialog_B[-1] if st.session_state.dialog_B else ""

        # Lấy transcript đầy đủ
        # Tái tạo transcript trước khi lượt mới bắt đầu
        full_transcript_current = "\n".join(full_transcript_list)

        # 1. A nói
        with st.spinner(f"Đang tạo lượt A ({persona1})..."):
             reply_A = execute_ai_turn('A', last_reply_for_A, full_transcript_current)

        # 2. B nói (Chỉ nếu không phải 1v1)
        if debate_mode != "Tranh luận 1v1 với AI":
            with st.spinner(f"Đang tạo lượt B ({persona2})..."):
                # Cập nhật transcript sau khi A nói để B có thể phản biện A
                current_transcript_after_A = full_transcript_current + f"\nA{len(st.session_state.dialog_A)} ({persona1}): {reply_A}"
                execute_ai_turn('B', reply_A, current_transcript_after_A)
        
        # Cần tính lại max_messages sau khi đã tạo nội dung
        new_max_messages = len(st.session_state.dialog_A) + len(st.session_state.dialog_B) + len(st.session_state.dialog_C)
        st.session_state.current_turn_index = new_max_messages - 2 # Index quay lại tin nhắn đầu tiên của lượt vừa tạo
        st.session_state.is_fast_mode = False 
        st.rerun()


    # TẠO DEBATE NẾU CHƯA TẠO (INIT)
    if not st.session_state.dialog_A and st.session_state.debate_running:
        with st.spinner("Đang tạo lời mở đầu và các lượt tranh luận..."):
            
            # 1. Xử lý Topic
            if not topic and uploaded_image:
                topic = generate_topic_from_image(uploaded_image) 
            st.session_state.topic_used = topic
            st.header(f"Chủ đề: {st.session_state.topic_used}")

            # 2. Lời mở đầu (Lượt 1)
            is_3_way = debate_mode == "Tham gia 3 bên (Thành viên C)"
            opener_msg = f"""
            Tạo lời mở đầu cho các bên về chủ đề: {topic}.
            Phong cách: {final_style}
            A: Tính cách {persona1} (Ủng hộ chủ đề)
            B: Tính cách {persona2} (Phản đối chủ đề)
            C: Tính cách {persona_C} (Bên thứ ba) (Chỉ tạo nếu là chế độ 3 bên)

            Viết dưới dạng:
            A: [Lời mở đầu của A]
            B: [Lời mở đầu của B]
            {f"C: [Lời mở đầu của C]" if is_3_way else ""}
            """
            raw = call_chat([{"role": "user", "content": opener_msg}],
                             model=model_text, temperature=temperature, max_tokens=max_tokens_per_turn * 3) 

            # Phân tích lời mở đầu (FIXED regex)
            try:
                # Dùng regex để tìm chính xác hơn các đoạn A, B, C
                a_match = re.search(r'A:\s*(.*?)\s*(?:B:|C:|$)', raw, re.DOTALL)
                b_match = re.search(r'B:\s*(.*?)\s*(?:A:|C:|$)', raw, re.DOTALL)
                c_match = re.search(r'C:\s*(.*?)\s*(?:A:|B:|$)', raw, re.DOTALL)
                
                a_open = a_match.group(1).strip() if a_match else "[[LỖI TẠO NỘI DUNG]]"
                b_open = b_match.group(1).strip() if b_match else "[[LỖI TẠO NỘI DUNG]]"
                c_open = c_match.group(1).strip() if c_match and is_3_way else ""
            except Exception:
                # Fallback thô sơ
                parts = re.split(r'(?:A:|B:|C:)', raw)
                a_open = parts[1].strip() if len(parts) > 1 else "[[LỖI TẠO NỘI DUNG]]"
                b_open = parts[2].strip() if len(parts) > 2 else "[[LỖI TẠO NỘI DUNG]]"
                c_open = parts[3].strip() if len(parts) > 3 and is_3_way else ""
                
            st.session_state.dialog_A.append(a_open)
            
            # B1 chỉ được AI tạo ra nếu không phải chế độ 1v1
            if debate_mode != "Tranh luận 1v1 với AI":
                st.session_state.dialog_B.append(b_open)
            else:
                st.session_state.dialog_B.append("[[CHỜ ĐẦU VÀO CỦA NGƯỜI DÙNG]]") # Marker cho người dùng

            if is_3_way:
                st.session_state.dialog_C.append(c_open)
            
            # Tính damage cho lời mở đầu (Lượt 0)
            if debate_mode == "Chế độ RPG (Game Tranh luận)":
                if debate_mode != "Tranh luận 1v1 với AI":
                    # Tạo transcript tạm thời cho tính damage khởi đầu
                    temp_transcript_init = f"A1 ({persona1}): {a_open}\nB1 ({persona2}): {b_open}"
                    apply_rpg_damage(0, "A", "B", a_open, persona1, persona2, final_style, temp_transcript_init)
                    apply_rpg_damage(0, "B", "A", b_open, persona2, persona1, final_style, temp_transcript_init)

            # 3. TURN-BASED DEBATE (Tạo các lượt tiếp theo)
            if debate_mode == "Tranh luận 2 AI (Tiêu chuẩn)" or debate_mode == "Chế độ RPG (Game Tranh luận)":
                for _ in range(rounds - 1): 
                    add_next_turn_wrapper_ai_only()
            
            st.session_state.debate_running = False
            st.rerun() 

    # ------------------- CÁC NÚT ĐIỀU KHIỂN CHAT -------------------
    max_messages = len(st.session_state.dialog_A) + len(st.session_state.dialog_B) + len(st.session_state.dialog_C)
    is_chat_complete = st.session_state.current_turn_index >= max_messages

    col_chat_ctrl = st.columns([1, 1.5, 1.5, 1.5, 1]) 

    if not is_chat_complete:
        # Nếu chưa xong, tiếp tục từng tin nhắn
        if col_chat_ctrl[1].button("▶️ Tiếp tục chat", use_container_width=True, disabled=game_over):
            st.session_state.current_turn_index += 1
            st.session_state.is_fast_mode = False
            st.rerun()

    fast_mode_label = "⏩ Tua nhanh/Hiện toàn bộ" if not st.session_state.is_fast_mode else "⏸️ Dừng tua nhanh"
    if col_chat_ctrl[2].button(fast_mode_label, use_container_width=True, disabled=game_over):
        st.session_state.is_fast_mode = not st.session_state.is_fast_mode
        if st.session_state.is_fast_mode:
            st.session_state.current_turn_index = max_messages
        st.rerun()

    
    # Thêm lượt mới (Logic cho 2 AI / RPG)
    is_finished_initial_rounds = len(st.session_state.dialog_A) >= rounds
    
    if is_finished_initial_rounds and is_chat_complete and not game_over:
        if debate_mode == "Tranh luận 2 AI (Tiêu chuẩn)" or debate_mode == "Chế độ RPG (Game Tranh luận)":
            if col_chat_ctrl[3].button("➕ Thêm 1 lượt", type="secondary", use_container_width=True):
                with st.spinner("Đang tạo thêm 1 lượt tranh luận mới..."):
                    add_next_turn_wrapper_ai_only()
        # Chế độ 1v1 và 3 bên sẽ tự động tạo lượt tiếp theo sau khi người dùng nhập 
    else:
        col_chat_ctrl[3].empty()


    # ------------------- HIỂN THỊ DẠNG CHAT BONG BÓNG -------------------
    
    current_message_count = 0
    max_len_display = max(len(st.session_state.dialog_A), len(st.session_state.dialog_B), len(st.session_state.dialog_C))
    
    for i in range(max_len_display):
        
        # --- Tin nhắn của A ---
        if i < len(st.session_state.dialog_A):
            if st.session_state.is_fast_mode or current_message_count < st.session_state.current_turn_index:
                st.markdown(f"""<div class="chat-container"><div class="chat-bubble chat-left"><b>A{i+1} ({persona1}):</b> {st.session_state.dialog_A[i]}</div></div>""", unsafe_allow_html=True)
                current_message_count += 1
            
            # Thao tác hiển thị từng bước
            elif not st.session_state.is_fast_mode and current_message_count == st.session_state.current_turn_index:
                with st.empty():
                    st.markdown(f"""...""", unsafe_allow_html=True)
                time.sleep(0.5)
                st.markdown(f"""<div class="chat-container"><div class="chat-bubble chat-left"><b>A{i+1} ({persona1}):</b> {st.session_state.dialog_A[i]}</div></div>""", unsafe_allow_html=True)
                st.session_state.current_turn_index += 1 
                st.rerun()
                break 

        # --- Tin nhắn của B ---
        if i < len(st.session_state.dialog_B):
            if st.session_state.is_fast_mode or current_message_count < st.session_state.current_turn_index:
                st.markdown(f"""<div class="chat-container" style="justify-content: flex-end;"><div class="chat-bubble chat-right"><b>B{i+1} ({persona2}):</b> {st.session_state.dialog_B[i]}</div></div>""", unsafe_allow_html=True)
                current_message_count += 1
                
            elif not st.session_state.is_fast_mode and current_message_count == st.session_state.current_turn_index:
                with st.empty():
                    st.markdown(f"""...""", unsafe_allow_html=True)
                time.sleep(0.5)
                st.markdown(f"""<div class="chat-container" style="justify-content: flex-end;"><div class="chat-bubble chat-right"><b>B{i+1} ({persona2}):</b> {st.session_state.dialog_B[i]}</div></div>""", unsafe_allow_html=True)
                st.session_state.current_turn_index += 1 
                st.rerun()
                break 
                
        # --- Tin nhắn của C (Nếu có) ---
        if i < len(st.session_state.dialog_C) and debate_mode == "Tham gia 3 bên (Thành viên C)":
            if st.session_state.is_fast_mode or current_message_count < st.session_state.current_turn_index:
                st.markdown(f"""<div class="chat-container" style="justify-content: center;"><div class="chat-bubble chat-user"><b>C{i+1} ({persona_C}):</b> {st.session_state.dialog_C[i]}</div></div>""", unsafe_allow_html=True)
                current_message_count += 1
            
            elif not st.session_state.is_fast_mode and current_message_count == st.session_state.current_turn_index:
                with st.empty():
                    st.markdown(f"""...""", unsafe_allow_html=True)
                time.sleep(0.5)
                st.markdown(f"""<div class="chat-container" style="justify-content: center;"><div class="chat-bubble chat-user"><b>C{i+1} ({persona_C}):</b> {st.session_state.dialog_C[i]}</div></div>""", unsafe_allow_html=True)
                st.session_state.current_turn_index += 1
                st.rerun()
                break

    # ------------------- INPUT NGƯỜI DÙNG -------------------
    
    is_user_turn = False
    
    if debate_mode == "Tranh luận 1v1 với AI":
        # Người dùng (B) cần nói nếu số lượt A > số lượt B
        if len(st.session_state.dialog_A) > len(st.session_state.dialog_B):
            is_user_turn = True
            user_role = persona2 
            
    elif debate_mode == "Tham gia 3 bên (Thành viên C)":
        # Người dùng (C) cần nói nếu A và B đã nói (A=B) VÀ C chưa nói (C < B)
        if len(st.session_state.dialog_A) == len(st.session_state.dialog_B) and len(st.session_state.dialog_B) > len(st.session_state.dialog_C):
             is_user_turn = True
             user_role = persona_C

    # Chỉ hiển thị ô nhập nếu đang đến lượt người dùng VÀ tất cả tin nhắn đã được hiển thị VÀ không game over
    if is_user_turn and st.session_state.current_turn_index >= current_message_count and not game_over: 
        st.markdown("---")
        st.subheader(f"💬 Lượt của bạn ({user_role})")
        
        input_key = "user_reply_b" if debate_mode == "Tranh luận 1v1 với AI" else "user_reply_c"
        
        if debate_mode == "Tranh luận 1v1 với AI":
            last_ai_reply = st.session_state.dialog_A[-1]
            st.session_state.user_input_C = st.text_area(f"Phản biện lời của {persona1} (A): {last_ai_reply[:100]}...", key=input_key, placeholder="Nhập luận điểm của bạn...")
        else:
            last_ai_reply = st.session_state.dialog_B[-1]
            st.session_state.user_input_C = st.text_area(f"Phản biện lời của {persona2} (B): {last_ai_reply[:100]}...", key=input_key, placeholder="Nhập luận điểm của bạn...")

        if st.button("🚀 Gửi phản biện của bạn", type="primary"):
            if st.session_state.user_input_C and st.session_state.user_input_C.strip():
                user_reply = st.session_state.user_input_C.strip()
                
                # Tạo transcript tạm thời (trước khi người dùng nói)
                temp_transcript = "\n".join(full_transcript_list)
                
                if debate_mode == "Tranh luận 1v1 với AI":
                    st.session_state.dialog_B.append(user_reply)
                    st.session_state.current_turn_index += 1
                    st.session_state.user_input_C = ""
                    
                    # KIỂM TRA CÓ CẦN TẠO TIẾP LƯỢT A KHÔNG (Chỉ tạo nếu chưa đủ rounds)
                    if len(st.session_state.dialog_A) < rounds: 
                        with st.spinner("Đang tạo lượt A tiếp theo..."):
                            # Cần tạo transcript sau khi B nói (user_reply)
                            current_transcript_after_B = temp_transcript + f"\nB{len(st.session_state.dialog_B)} ({persona2}): {user_reply}"
                            # Tạo lượt A (Phản biện B)
                            execute_ai_turn('A', user_reply, current_transcript_after_B)
                            
                    st.rerun() 
                
                elif debate_mode == "Tham gia 3 bên (Thành viên C)":
                    st.session_state.dialog_C.append(user_reply)
                    st.session_state.current_turn_index += 1
                    st.session_state.user_input_C = ""
                    
                    # Cần tạo A và B tiếp theo (Chỉ tạo nếu chưa đủ rounds)
                    if len(st.session_state.dialog_A) < rounds:
                        with st.spinner("Đang tạo lượt A và B tiếp theo..."):
                            # Cần tạo transcript sau khi C nói (user_reply)
                            current_transcript_after_C = temp_transcript + f"\nC{len(st.session_state.dialog_C)} ({persona_C}): {user_reply}"

                            # 1. A nói (Phản biện C)
                            reply_A = execute_ai_turn('A', user_reply, current_transcript_after_C)

                            # 2. B nói (Phản biện A vừa nói)
                            # Cần cập nhật lại transcript sau khi A nói
                            current_transcript_after_A_B = current_transcript_after_C + f"\nA{len(st.session_state.dialog_A)} ({persona1}): {reply_A}"
                            execute_ai_turn('B', reply_A, current_transcript_after_A_B)
                    
                    st.rerun() 
                else:
                    st.warning("Logic lỗi: Không xác định được chế độ debate.")
            else:
                st.warning("Vui lòng nhập nội dung phản biện.")

    # ------------------- LOG & CÔNG CỤ -------------------
    st.markdown("---")
    
    # LOG RPG
    if debate_mode == "Chế độ RPG (Game Tranh luận)" and st.session_state.rpg_log:
        st.subheader("📜 Nhật ký Sát thương (RPG Log)")
        for log in reversed(st.session_state.rpg_log):
            st.markdown(log)
        st.markdown("---")

    # TÒA ÁN AI (FIXED: SỬ DỤNG LẠI courtRoom_analysis)
    st.header("⚖️ Phân tích Tòa án AI (Judge/Prosecutor/Lawyer)")
    if debate_mode != "Tham gia 3 bên (Thành viên C)": 
        # Phiên tòa sẽ hoạt động ở MỌI CHẾ ĐỘ nếu hoàn thành số lượt ban đầu (hoặc tua nhanh)
        is_ready_for_analysis = is_chat_complete and len(st.session_state.dialog_A) >= rounds
        
        if st.button("⚖️ Tổ chức phiên Tòa án AI", type="primary", use_container_width=True, disabled=not is_ready_for_analysis): 
            if is_ready_for_analysis:
                with st.spinner("Đang phân tích lập luận chi tiết, chỉ ra ngụy biện và đưa ra phán quyết..."):
                    analysis_result = ai_courtroom_analysis(full_transcript, final_style, persona1, persona2, model_text)
                    st.session_state.courtroom_analysis = analysis_result
                    st.rerun()
            else:
                st.warning("Vui lòng hoàn thành tất cả các lượt đã chọn hoặc Tua nhanh trước khi phân tích!")

        if "courtroom_analysis" in st.session_state and st.session_state.courtroom_analysis:
            st.subheader("📋 Kết quả Phiên Tòa án AI")
            st.markdown(st.session_state.courtroom_analysis)
            st.markdown("---")
    else:
        st.info("Tính năng Tòa án AI chỉ hỗ trợ chế độ tranh luận 2 bên (A và B).")
        
    if st.button("📥 Tải Transcript", use_container_width=True):
        st.download_button(
            "Tải file .txt",
            data=full_transcript.encode("utf-8"),
            file_name=f"debate_{st.session_state.topic_used[:30].replace(' ', '_')}.txt",
            mime="text/plain"
        )

# ----------------------------------------------------------------------------------------------------
# ROUTING
# ----------------------------------------------------------------------------------------------------
if st.session_state.page == "home":
    render_home()
else:
    render_debate()