import streamlit as st
from PIL import Image
import base64
import os
from dotenv import load_dotenv
import io
import time
import re
import random 
import json 

try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except:
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    st.error("Lỗi xác thực: GITHUB_TOKEN chưa được thiết lập. Vui lòng kiểm tra file .streamlit/secrets.toml trên Streamlit Cloud hoặc file .env khi chạy local.")
    st.stop()

from openai import OpenAI

GITHUB_BASE_URL = "https://models.github.ai/inference"
client = OpenAI(base_url=GITHUB_BASE_URL, api_key=GITHUB_TOKEN)

st.set_page_config(page_title="🤖 AI Debate Bot", layout="wide")

CHAT_STYLE = """
<style>
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}
h1, h2, h3, h4, h5, h6 {
    color: #58a6ff;
}

.chat-bubble {
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 16px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
}

.chat-left {
    background-color: #1f362d;
    color: #4cd964 !important;
    margin-right: auto;
    border-top-left-radius: 2px;
}
.chat-left b {
    color: #58a6ff !important;
}

.chat-right {
    background-color: #3b2225;
    color: #ff9500 !important;
    margin-left: auto;
    border-top-right-radius: 2px;
}
.chat-right b {
    color: #58a6ff !important;
}

.chat-user {
    background-color: #192f44;
    color: #8bb8e8 !important;
    margin-left: auto;
    border-top-right-radius: 2px;
}
.chat-user b {
    color: #c9d1d9 !important;
}

.chat-container {
    display: flex;
    width: 100%;
    margin-bottom: 10px;
}

.hp-bar-container {
    background-color: #1e2d42;
    border-radius: .35rem;
    height: 1.8rem;
    overflow: hidden;
    margin-bottom: 15px;
    border: 2px solid #58a6ff;
    box-shadow: 0 0 5px rgba(88, 166, 255, 0.5);
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

.advantage-box {
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 20px;
}

.advantage-A {
    background-color: #0e4429;
    color: #4cd964;
    border: 1px solid #1f362d;
}

.advantage-B {
    background-color: #58161b;
    color: #ff9500;
    border: 1px solid #3b2225;
}

.advantage-draw {
    background-color: #423200;
    color: #ffd60a;
    border: 1px solid #332700;
}
</style>
"""
st.markdown(CHAT_STYLE, unsafe_allow_html=True)
st.title("🤖 AI Debate Bot – Thiết lập tranh luận")


if "page" not in st.session_state:
    st.session_state.page = "home"

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
    st.session_state.dialog_C = []

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


def call_chat(messages, model, temperature=0.6, max_tokens=600):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Lỗi khi gọi API Text ({model}): {e}. Vui lòng kiểm tra lại GITHUB_TOKEN hoặc chọn Model nhẹ hơn.")
        return f"[[LỖI TẠO NỘI DUNG - API CALL FAILED]]"

def rpg_damage_analysis(attacker_name, defender_name, last_reply, final_style, full_transcript_segment):
    
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
    
    raw_json = call_chat(
        [{"role": "user", "content": prompt}],
        model="openai/gpt-4o-mini", 
        temperature=0.3, 
        max_tokens=250
    )
    
    try:
        json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if 'damage' in data:
                data['damage'] = int(data['damage'])
            return data
        else:
            raise ValueError("Không tìm thấy JSON hợp lệ.")
    except Exception as e:
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


def ai_courtroom_analysis(full_transcript, final_style, persona1, persona2, model_text):
    
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
    
    if not full_transcript.strip():
        return "[[LỖI TẠO NỘI DUNG]] - Transcript rỗng hoặc lỗi."

    result = call_chat(
        [{"role": "user", "content": prompt}],
        model=model_text, 
        temperature=0.3, 
        max_tokens=2000 
    )
    
    if "[[LỖI TẠO NỘI DUNG]]" in result:
        st.error("Lỗi: AI không thể hoàn thành Phân tích Tòa án. Vui lòng thử lại.")
        return "[[LỖI TẠO NỘI DUNG - PHÂN TÍCH THẤT BẠI]]"
        
    return result

def generate_ai_reply(persona_role, persona_name, last_reply_content, final_style, model_text, temperature, max_tokens_per_turn):
    prompt = f"""
    Bạn là Bên {persona_role} ({persona_name}). Hãy phản biện lời nói gần nhất của đối thủ trong 3-5 câu. 
    Sử dụng phong cách '{final_style}'. 
    Lời nói gần nhất của đối thủ: '{last_reply_content}'
    """
    return call_chat(
        [{"role": "user", "content": prompt}],
        model=model_text, temperature=temperature, max_tokens=max_tokens_per_turn
    )


def generate_topic_from_image(uploaded_image):
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
    prompt = "Gợi ý 5 chủ đề tranh luận gây tranh cãi, thú vị, ngắn gọn, mỗi chủ đề trên 1 dòng. Đảm bảo KHÔNG có số hoặc dấu gạch đầu dòng."
    suggestions = call_chat([{"role": "user", "content": prompt}], model, 0.9, 300)
    cleaned_suggestions = re.sub(r'^\s*[\d\.\-\*]\s*', '', suggestions, flags=re.MULTILINE)
    topics_list = [t.strip() for t in cleaned_suggestions.split('\n') if t.strip()]
    return topics_list

def check_rpg_advantage(hp_a, hp_b, persona1, persona2):
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
    
    with col_p1:
        st.session_state.persona1 = st.text_input(
            "Bên A (Ủng hộ):", 
            st.session_state.get('persona1', "Bình tĩnh, logic"), 
            key="persona1_input",
            help="Tính cách, vai trò, quan điểm sơ bộ của bên A (AI)."
        )
        
    with col_p2:
        if st.session_state.debate_mode == "Tranh luận 1v1 với AI":
            st.info("**Bên B (Phản đối)** là **Bạn** (Người dùng).")
            st.session_state.persona2 = "Người dùng (Phản đối)"
            
        else:
            st.session_state.persona2 = st.text_input(
                "Bên B (Phản đối):", 
                st.session_state.get('persona2', "Năng nổ, phản biện"),
                key="persona2_input",
                help="Tính cách, vai trò, quan điểm sơ bộ của bên B (AI)."
            )

    if st.session_state.debate_mode == "Tham gia 3 bên (Thành viên C)":
        st.session_state.C_persona = st.text_input(
            "Bên C (Người dùng):", 
            st.session_state.get('C_persona', "Bên thứ ba/Đa chiều"), 
            key="C_persona_input",
            help="Bạn sẽ tham gia với tư cách C. (Tính cách, vai trò của bạn)."
        )
    else:
        if 'C_persona' in st.session_state:
            del st.session_state['C_persona']

    st.markdown("---")

    if st.button("▶️ Bắt đầu tranh luận", type="primary", use_container_width=True):
        if not st.session_state.topic and not st.session_state.uploaded_image:
            st.error("Vui lòng nhập chủ đề hoặc upload ảnh để bắt đầu!")
            return
            
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

def render_debate():

    st.title("🔥 Cuộc tranh luận")

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

    full_transcript_list = []
    max_len_trans = max(len(st.session_state.dialog_A), len(st.session_state.dialog_B), len(st.session_state.dialog_C)) 
    for i in range(max_len_trans):
        if i < len(st.session_state.dialog_A):
             full_transcript_list.append(f"A{i+1} ({persona1}): {st.session_state.dialog_A[i]}")
        if i < len(st.session_state.dialog_B):
             full_transcript_list.append(f"B{i+1} ({persona2}): {st.session_state.dialog_B[i]}")
        if i < len(st.session_state.dialog_C) and debate_mode == "Tham gia 3 bên (Thành viên C)":
             full_transcript_list.append(f"C{i+1} ({persona_C}): {st.session_state.dialog_C[i]}")

    full_transcript = "\n".join(full_transcript_list)
    
    game_over = False 
    if debate_mode == "Chế độ RPG (Game Tranh luận)":
        st.subheader("⚔️ HP (Hit Points)")
        col_hp_a, col_hp_b = st.columns(2)
        
        def display_hp(col, name, hp):
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
        
        check_rpg_advantage(st.session_state.A_HP, st.session_state.B_HP, persona1, persona2)
        
        st.markdown("---")

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

    
    def apply_rpg_damage(turn_index, attacker_role, receiver_role, attack_content, attacker_persona, receiver_persona, style, current_transcript):
        
        log_msg_base = f"Lượt {turn_index+1} ({attacker_role} -> {receiver_role})"
