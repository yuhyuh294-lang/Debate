import streamlit as st
from PIL import Image
import base64
import os
import time
import re
import random 
import json 
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# --- Cấu hình và Khởi tạo ---
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    from dotenv import load_dotenv
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not GITHUB_TOKEN and not OPENAI_API_KEY:
    st.error("Lỗi xác thực: Chưa thiết lập API key. Vui lòng kiểm tra cấu hình.")
    st.stop()

from openai import OpenAI

# --- Data Classes ---
@dataclass
class DebateConfig:
    mode: str = "Tranh luận 2 AI (Tiêu chuẩn)"
    topic: str = ""
    style: str = "Trang trọng – Học thuật"
    custom_style: str = ""
    persona_a: str = "Bình tĩnh, logic"
    persona_b: str = "Năng nổ, phản biện"
    persona_c: str = "Người dùng (Thành viên C)"
    rounds: int = 3
    temperature: float = 0.6
    max_tokens: int = 600
    model: str = "openai/gpt-4.1"
    api_client: str = "github"

@dataclass
class RPGState:
    hp_a: int = 100
    hp_b: int = 100
    log: List[str] = field(default_factory=list)
    damage_history: List[Dict] = field(default_factory=list)

@dataclass
class TurnState:
    current_turn: str = "A"  # A, B, C, hoặc USER
    turn_count: int = 0
    message_index: int = 0
    is_fast_mode: bool = False

# --- Khởi tạo Session State ---
def init_session_state():
    """Khởi tạo tất cả session state variables"""
    if "config" not in st.session_state:
        st.session_state.config = DebateConfig()
    
    if "dialog_a" not in st.session_state:
        st.session_state.dialog_a = []
    
    if "dialog_b" not in st.session_state:
        st.session_state.dialog_b = []
    
    if "dialog_c" not in st.session_state:
        st.session_state.dialog_c = []
    
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    
    if "suggested_topics" not in st.session_state:
        st.session_state.suggested_topics = None
    
    if "turn_state" not in st.session_state:
        st.session_state.turn_state = TurnState()
    
    if "debate_running" not in st.session_state:
        st.session_state.debate_running = False
    
    if "courtroom_analysis" not in st.session_state:
        st.session_state.courtroom_analysis = None
    
    if "rpg_state" not in st.session_state:
        st.session_state.rpg_state = RPGState()
    
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    if "topic_used" not in st.session_state:
        st.session_state.topic_used = ""
    
    if "final_style" not in st.session_state:
        st.session_state.final_style = ""
    
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = False

# Gọi khởi tạo
init_session_state()

# --- API Helper Functions ---
def get_api_client():
    """Lấy API client dựa trên cấu hình"""
    config = st.session_state.config
    
    if config.api_client == "github" and GITHUB_TOKEN:
        return OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=GITHUB_TOKEN
        )
    elif config.api_client == "openai" and OPENAI_API_KEY:
        return OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY
        )
    else:
        # Fallback: thử Github trước, rồi OpenAI
        if GITHUB_TOKEN:
            return OpenAI(
                base_url="https://models.github.ai/inference",
                api_key=GITHUB_TOKEN
            )
        elif OPENAI_API_KEY:
            return OpenAI(
                base_url="https://api.openai.com/v1",
                api_key=OPENAI_API_KEY
            )
        else:
            raise Exception("Không có API key hợp lệ")

def call_chat(messages: List[Dict], model: str = None, temperature: float = None, 
              max_tokens: int = None) -> str:
    """Gọi API chat với xử lý lỗi"""
    config = st.session_state.config
    
    if model is None:
        model = config.model
    if temperature is None:
        temperature = config.temperature
    if max_tokens is None:
        max_tokens = config.max_tokens
    
    try:
        client = get_api_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Lỗi API: {str(e)[:200]}")
        return f"[[LỖI: {str(e)[:100]}]]"

# --- Debate Logic Functions ---
def generate_opening_statements() -> Tuple[str, str, str]:
    """Tạo lời mở đầu cho tất cả các bên"""
    config = st.session_state.config
    topic = st.session_state.topic_used
    
    prompt = f"""
    Tạo lời mở đầu cho cuộc tranh luận về chủ đề: {topic}
    
    Phong cách: {config.style if not config.custom_style else config.custom_style}
    
    Yêu cầu:
    1. A ({config.persona_a}): Ủng hộ chủ đề, 3-4 câu
    2. B ({config.persona_b}): Phản đối chủ đề, 3-4 câu
    """
    
    if config.mode == "Tham gia 3 bên (Thành viên C)":
        prompt += f"3. C ({config.persona_c}): Quan điểm trung lập/đa chiều, 3-4 câu\n"
    
    response = call_chat([{"role": "user", "content": prompt}])
    
    # Parse response
    a_opening = ""
    b_opening = ""
    c_opening = ""
    
    # Try to extract using patterns
    patterns = [
        r'A[:\-]?\s*(.*?)(?:\n\n|\nB|$)',
        r'B[:\-]?\s*(.*?)(?:\n\n|\nC|$)',
        r'C[:\-]?\s*(.*?)(?:\n\n|$)'
    ]
    
    import re
    
    a_match = re.search(patterns[0], response, re.DOTALL | re.IGNORECASE)
    if a_match:
        a_opening = a_match.group(1).strip()
    
    b_match = re.search(patterns[1], response, re.DOTALL | re.IGNORECASE)
    if b_match:
        b_opening = b_match.group(1).strip()
    
    if config.mode == "Tham gia 3 bên (Thành viên C)":
        c_match = re.search(patterns[2], response, re.DOTALL | re.IGNORECASE)
        if c_match:
            c_opening = c_match.group(1).strip()
    
    # Fallback: split by lines
    if not a_opening or not b_opening:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if len(lines) >= 2:
            a_opening = lines[0].replace('A:', '').replace('A-', '').strip()
            b_opening = lines[1].replace('B:', '').replace('B-', '').strip()
            if len(lines) >= 3 and config.mode == "Tham gia 3 bên (Thành viên C)":
                c_opening = lines[2].replace('C:', '').replace('C-', '').strip()
    
    return a_opening, b_opening, c_opening

def generate_ai_reply(speaker: str, last_message: str) -> str:
    """Tạo câu trả lời cho AI"""
    config = st.session_state.config
    
    if speaker == "A":
        persona = config.persona_a
        role = "Ủng hộ"
        opponent = config.persona_b
    elif speaker == "B":
        persona = config.persona_b
        role = "Phản đối"
        opponent = config.persona_a
    else:
        return "Lỗi: Speaker không hợp lệ"
    
    prompt = f"""
    Bạn đang đóng vai {persona} ({role}) trong cuộc tranh luận.
    Phong cách: {config.style if not config.custom_style else config.custom_style}
    Chủ đề: {st.session_state.topic_used}
    
    Người vừa nói ({opponent}): "{last_message[:300]}"
    
    Hãy trả lời một cách ngắn gọn, sắc bén (3-5 câu) theo đúng tính cách {persona}.
    """
    
    return call_chat([{"role": "user", "content": prompt}])

def process_rpg_damage(attacker: str, defender: str, message: str):
    """Xử lý sát thương RPG"""
    if st.session_state.config.mode != "Chế độ RPG (Game Tranh luận)":
        return
    
    # Tính damage dựa trên độ dài và phức tạp của message
    base_damage = min(25, len(message) // 10)
    
    # Thêm yếu tố ngẫu nhiên
    damage_variation = random.randint(-5, 10)
    final_damage = max(5, base_damage + damage_variation)
    
    # Có 15% cơ hội chí mạng
    is_crit = random.random() < 0.15
    if is_crit:
        final_damage = min(40, final_damage * 2)
    
    # Áp dụng damage
    if defender == "A":
        st.session_state.rpg_state.hp_a = max(0, st.session_state.rpg_state.hp_a - final_damage)
        defender_name = st.session_state.config.persona_a
    else:
        st.session_state.rpg_state.hp_b = max(0, st.session_state.rpg_state.hp_b - final_damage)
        defender_name = st.session_state.config.persona_b
    
    attacker_name = st.session_state.config.persona_a if attacker == "A" else st.session_state.config.persona_b
    
    # Ghi log
    crit_text = "🔥 **CHÍ MẠNG!** " if is_crit else ""
    log_msg = f"⚔️ **{attacker_name}** → **{defender_name}**: {crit_text}-{final_damage} HP"
    st.session_state.rpg_state.log.append(log_msg)
    
    st.session_state.rpg_state.damage_history.append({
        "turn": st.session_state.turn_state.turn_count,
        "attacker": attacker,
        "defender": defender,
        "damage": final_damage,
        "is_crit": is_crit,
        "message": message[:100]
    })

def next_turn():
    """Chuyển sang lượt tiếp theo"""
    config = st.session_state.config
    turn_state = st.session_state.turn_state
    
    if config.mode == "Tranh luận 1v1 với AI":
        # A → USER → A → USER ...
        if turn_state.current_turn == "A":
            turn_state.current_turn = "USER_B"
        else:
            turn_state.current_turn = "A"
    
    elif config.mode == "Tham gia 3 bên (Thành viên C)":
        # A → B → C(USER) → A → B → C(USER) ...
        if turn_state.current_turn == "A":
            turn_state.current_turn = "B"
        elif turn_state.current_turn == "B":
            turn_state.current_turn = "USER_C"
        else:
            turn_state.current_turn = "A"
    
    elif config.mode == "Chế độ RPG (Game Tranh luận)":
        # A → B → A → B ...
        turn_state.current_turn = "B" if turn_state.current_turn == "A" else "A"
    
    else:  # Tranh luận 2 AI (Tiêu chuẩn)
        # A → B → A → B ...
        turn_state.current_turn = "B" if turn_state.current_turn == "A" else "A"
    
    turn_state.turn_count += 1
    turn_state.message_index = len(st.session_state.dialog_a) + len(st.session_state.dialog_b) + len(st.session_state.dialog_c)

def execute_ai_turn(speaker: str):
    """Thực thi lượt của AI"""
    config = st.session_state.config
    
    # Xác định tin nhắn cuối cùng của đối thủ
    if speaker == "A":
        last_message = st.session_state.dialog_b[-1] if st.session_state.dialog_b else ""
        new_message = generate_ai_reply("A", last_message)
        st.session_state.dialog_a.append(new_message)
        
        # Xử lý RPG damage nếu cần
        if config.mode == "Chế độ RPG (Game Tranh luận)" and last_message:
            process_rpg_damage("A", "B", new_message)
    
    elif speaker == "B":
        last_message = st.session_state.dialog_a[-1] if st.session_state.dialog_a else ""
        new_message = generate_ai_reply("B", last_message)
        st.session_state.dialog_b.append(new_message)
        
        # Xử lý RPG damage nếu cần
        if config.mode == "Chế độ RPG (Game Tranh luận)" and last_message:
            process_rpg_damage("B", "A", new_message)
    
    # Chuyển lượt
    next_turn()

def execute_user_turn(user_role: str, message: str):
    """Thực thi lượt của người dùng"""
    if user_role == "USER_B":
        st.session_state.dialog_b.append(message)
    elif user_role == "USER_C":
        st.session_state.dialog_c.append(message)
    
    # Xử lý RPG damage nếu người dùng là B trong chế độ RPG
    config = st.session_state.config
    if config.mode == "Chế độ RPG (Game Tranh luận)" and user_role == "USER_B":
        process_rpg_damage("B", "A", message)
    
    # Chuyển lượt
    next_turn()
    
    # Nếu là chế độ 1v1 và chưa đủ rounds, AI tự động trả lời
    if config.mode == "Tranh luận 1v1 với AI":
        if len(st.session_state.dialog_a) < config.rounds:
            with st.spinner(f"{config.persona_a} đang suy nghĩ..."):
                execute_ai_turn("A")
        else:
            st.session_state.debate_running = False
    
    # Nếu là chế độ 3 bên và chưa đủ rounds, A và B tự động trả lời
    elif config.mode == "Tham gia 3 bên (Thành viên C)":
        if len(st.session_state.dialog_a) < config.rounds:
            with st.spinner(f"{config.persona_a} và {config.persona_b} đang tranh luận..."):
                execute_ai_turn("A")
                if len(st.session_state.dialog_b) < config.rounds:
                    execute_ai_turn("B")
        else:
            st.session_state.debate_running = False

def check_game_over() -> Tuple[bool, str]:
    """Kiểm tra xem trò chơi đã kết thúc chưa"""
    config = st.session_state.config
    rpg_state = st.session_state.rpg_state
    
    if config.mode == "Chế độ RPG (Game Tranh luận)":
        if rpg_state.hp_a <= 0 and rpg_state.hp_b <= 0:
            return True, "🏳️ HÒA! Cả hai đều hết máu."
        elif rpg_state.hp_a <= 0:
            return True, f"🏆 {config.persona_b} CHIẾN THẮNG!"
        elif rpg_state.hp_b <= 0:
            return True, f"🏆 {config.persona_a} CHIẾN THẮNG!"
    
    # Kiểm tra số lượt đã đạt
    if len(st.session_state.dialog_a) >= config.rounds:
        if config.mode == "Tranh luận 2 AI (Tiêu chuẩn)":
            if len(st.session_state.dialog_b) >= config.rounds:
                return True, "✅ Tranh luận đã hoàn thành!"
        elif config.mode == "Tranh luận 1v1 với AI":
            if len(st.session_state.dialog_b) >= config.rounds:
                return True, "✅ Tranh luận đã hoàn thành!"
        elif config.mode == "Tham gia 3 bên (Thành viên C)":
            if len(st.session_state.dialog_b) >= config.rounds and len(st.session_state.dialog_c) >= config.rounds:
                return True, "✅ Tranh luận đã hoàn thành!"
        elif config.mode == "Chế độ RPG (Game Tranh luận)":
            if len(st.session_state.dialog_b) >= config.rounds:
                return True, "✅ Tranh luận đã hoàn thành!"
    
    return False, ""

# --- UI Components ---
def render_hp_bars():
    """Hiển thị thanh HP cho chế độ RPG"""
    config = st.session_state.config
    rpg_state = st.session_state.rpg_state
    
    col1, col2 = st.columns(2)
    
    with col1:
        hp_color = "#4cd964" if rpg_state.hp_a > 50 else ("#ff9500" if rpg_state.hp_a > 25 else "#ff3b30")
        st.markdown(f"**{config.persona_a}** ({rpg_state.hp_a} HP)")
        st.markdown(f"""
        <div style="background-color: #1e2d42; border-radius: .35rem; height: 1.8rem; overflow: hidden; border: 2px solid {hp_color};">
            <div style="height: 100%; width: {rpg_state.hp_a}%; background: linear-gradient(to right, {hp_color}, {hp_color}cc); 
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {rpg_state.hp_a}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hp_color = "#4cd964" if rpg_state.hp_b > 50 else ("#ff9500" if rpg_state.hp_b > 25 else "#ff3b30")
        st.markdown(f"**{config.persona_b}** ({rpg_state.hp_b} HP)")
        st.markdown(f"""
        <div style="background-color: #1e2d42; border-radius: .35rem; height: 1.8rem; overflow: hidden; border: 2px solid {hp_color};">
            <div style="height: 100%; width: {rpg_state.hp_b}%; background: linear-gradient(to right, {hp_color}, {hp_color}cc); 
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {rpg_state.hp_b}%
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_chat_message(speaker: str, message: str, index: int):
    """Hiển thị một tin nhắn trong chat"""
    config = st.session_state.config
    
    if speaker == "A":
        name = config.persona_a
        css_class = "chat-left"
    elif speaker == "B":
        name = config.persona_b
        css_class = "chat-right"
    else:  # C
        name = config.persona_c
        css_class = "chat-user"
    
    st.markdown(f"""
    <div class="chat-container">
        <div class="chat-bubble {css_class}">
            <b>{speaker}{index+1} ({name}):</b> {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Pages ---
def render_home():
    """Trang chủ thiết lập"""
    st.title("🤖 AI Debate Bot – Thiết lập tranh luận")
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Cài đặt Nâng cao")
        
        # API selection
        api_options = []
        if GITHUB_TOKEN:
            api_options.append("GitHub Models (github)")
        if OPENAI_API_KEY:
            api_options.append("OpenAI Official (openai)")
        
        if api_options:
            selected_api = st.selectbox(
                "API Provider:",
                api_options,
                index=0
            )
            st.session_state.config.api_client = "github" if "GitHub" in selected_api else "openai"
        
        # Model selection
        model_options = ["openai/gpt-4.1", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]
        if st.session_state.config.api_client == "openai":
            model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"]
        
        st.session_state.config.model = st.selectbox(
            "Model:",
            model_options,
            index=0
        )
        
        st.session_state.config.temperature = st.slider(
            "Độ sáng tạo", 0.0, 1.0, 0.6, 0.1
        )
        
        st.session_state.config.rounds = st.slider(
            "Số lượt mỗi bên", 1, 10, 3
        )
        
        st.session_state.config.max_tokens = st.slider(
            "Token tối đa/lượt", 100, 1000, 600, 50
        )
        
        if st.button("🔄 Reset Debate", type="secondary"):
            for key in list(st.session_state.keys()):
                if key not in ["config", "page"]:
                    del st.session_state[key]
            init_session_state()
            st.rerun()
    
    # 1. Chế độ tranh luận
    st.subheader("1) Chế độ Tranh luận")
    modes = [
        "Tranh luận 2 AI (Tiêu chuẩn)",
        "Tranh luận 1v1 với AI",
        "Chế độ RPG (Game Tranh luận)",
        "Tham gia 3 bên (Thành viên C)"
    ]
    st.session_state.config.mode = st.selectbox(
        "Chọn chế độ:",
        modes,
        index=modes.index(st.session_state.config.mode) if st.session_state.config.mode in modes else 0
    )
    
    # 2. Chủ đề
    st.subheader("2) Chủ đề tranh luận")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.config.topic = st.text_input(
            "Nhập chủ đề tranh luận:",
            value=st.session_state.config.topic,
            placeholder="Ví dụ: AI có nên được cấp quyền công dân không?"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("💡 Gợi ý chủ đề", use_container_width=True):
            with st.spinner("Đang tạo..."):
                prompt = "Gợi ý 3 chủ đề tranh luận thú vị, gây tranh cãi"
                response = call_chat([{"role": "user", "content": prompt}])
                topics = [t.strip() for t in response.split('\n') if t.strip()]
                st.session_state.suggested_topics = topics[:3]
    
    if st.session_state.suggested_topics:
        st.markdown("**Chọn từ gợi ý:**")
        for topic in st.session_state.suggested_topics:
            if st.button(topic[:80], key=f"topic_{topic[:10]}"):
                st.session_state.config.topic = topic
                st.session_state.suggested_topics = None
                st.rerun()
    
    # 3. Phong cách
    st.subheader("3) Phong cách tranh luận")
    styles = [
        "Trang trọng – Học thuật", "Hài hước", "Hỗn loạn", 
        "Triết gia", "Anime", "Rapper", "Lịch sự – Ngoại giao",
        "Văn học cổ điển", "Lãng mạn", "Khác"
    ]
    
    st.session_state.config.style = st.selectbox(
        "Chọn phong cách:",
        styles,
        index=styles.index(st.session_state.config.style) if st.session_state.config.style in styles else 0
    )
    
    if st.session_state.config.style == "Khác":
        st.session_state.config.custom_style = st.text_input("Mô tả phong cách của bạn:")
    
    # 4. Persona
    st.subheader("4) Tính cách các bên")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state.config.persona_a = st.text_input(
            "Bên A (Ủng hộ):",
            value=st.session_state.config.persona_a
        )
    
    with col_b:
        if st.session_state.config.mode == "Tranh luận 1v1 với AI":
            st.info("**Bạn sẽ là Bên B (Phản đối)**")
            st.session_state.config.persona_b = "Người dùng (Phản đối)"
        else:
            st.session_state.config.persona_b = st.text_input(
                "Bên B (Phản đối):",
                value=st.session_state.config.persona_b
            )
    
    if st.session_state.config.mode == "Tham gia 3 bên (Thành viên C)":
        st.session_state.config.persona_c = st.text_input(
            "Bên C (Bạn - Trung lập/Đa chiều):",
            value=st.session_state.config.persona_c
        )
    
    # Start button
    st.markdown("---")
    col_start, _ = st.columns([1, 3])
    with col_start:
        if st.button("▶️ Bắt đầu tranh luận", type="primary", use_container_width=True):
            if not st.session_state.config.topic.strip():
                st.error("Vui lòng nhập chủ đề tranh luận!")
                return
            
            # Reset state
            st.session_state.dialog_a = []
            st.session_state.dialog_b = []
            st.session_state.dialog_c = []
            st.session_state.rpg_state = RPGState()
            st.session_state.turn_state = TurnState()
            st.session_state.debate_running = True
            st.session_state.debate_started = False
            st.session_state.topic_used = st.session_state.config.topic
            st.session_state.final_style = st.session_state.config.custom_style if st.session_state.config.custom_style else st.session_state.config.style
            st.session_state.page = "debate"
            st.rerun()

def render_debate():
    """Trang tranh luận chính"""
    st.title("🔥 Cuộc tranh luận")
    
    config = st.session_state.config
    turn_state = st.session_state.turn_state
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 Thông tin")
        st.info(f"**Chế độ:** {config.mode}")
        st.info(f"**Chủ đề:** {st.session_state.topic_used}")
        st.info(f"**Phong cách:** {st.session_state.final_style}")
        
        if config.mode == "Chế độ RPG (Game Tranh luận)":
            render_hp_bars()
            
            if st.session_state.rpg_state.log:
                st.subheader("📜 Nhật ký chiến đấu")
                for log in st.session_state.rpg_state.log[-5:]:
                    st.write(log)
        
        if st.button("🔙 Về trang chủ", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    # Hiển thị chủ đề
    st.header(f"Chủ đề: {st.session_state.topic_used}")
    st.markdown("---")
    
    # Khởi tạo debate nếu chưa bắt đầu
    if not st.session_state.debate_started and st.session_state.debate_running:
        with st.spinner("Đang khởi tạo cuộc tranh luận..."):
            a_open, b_open, c_open = generate_opening_statements()
            st.session_state.dialog_a.append(a_open)
            st.session_state.dialog_b.append(b_open)
            
            if config.mode == "Tham gia 3 bên (Thành viên C)":
                st.session_state.dialog_c.append(c_open)
            
            st.session_state.debate_started = True
            
            # Đặt lượt đầu tiên
            if config.mode == "Tranh luận 1v1 với AI":
                turn_state.current_turn = "USER_B"
            elif config.mode == "Tham gia 3 bên (Thành viên C)":
                turn_state.current_turn = "USER_C"
            else:
                turn_state.current_turn = "B"
            
            # Tạo tiếp các lượt nếu cần (cho chế độ 2 AI)
            if config.mode in ["Tranh luận 2 AI (Tiêu chuẩn)", "Chế độ RPG (Game Tranh luận)"]:
                for i in range(config.rounds - 1):
                    execute_ai_turn("A")
                    execute_ai_turn("B")
            
            st.rerun()
    
    # Hiển thị chat history
    max_messages = max(len(st.session_state.dialog_a), 
                       len(st.session_state.dialog_b),
                       len(st.session_state.dialog_c))
    
    for i in range(max_messages):
        if i < len(st.session_state.dialog_a):
            render_chat_message("A", st.session_state.dialog_a[i], i)
        
        if i < len(st.session_state.dialog_b):
            render_chat_message("B", st.session_state.dialog_b[i], i)
        
        if i < len(st.session_state.dialog_c):
            render_chat_message("C", st.session_state.dialog_c[i], i)
    
    # Kiểm tra game over
    game_over, game_over_msg = check_game_over()
    if game_over:
        st.error(game_over_msg)
        st.session_state.debate_running = False
    
    # Hiển thị lượt hiện tại và input cho người dùng
    if st.session_state.debate_running and not game_over:
        st.markdown("---")
        
        # Xác định lượt hiện tại
        current_turn = turn_state.current_turn
        
        if current_turn == "USER_B":
            st.subheader(f"💬 Lượt của bạn (Bên B - {config.persona_b})")
            
            # Tìm tin nhắn cuối cùng của A để hiển thị
            if st.session_state.dialog_a:
                last_a_msg = st.session_state.dialog_a[-1]
                st.info(f"**{config.persona_a} vừa nói:** {last_a_msg[:200]}...")
            
            user_input = st.text_area(
                "Phản biện của bạn:",
                value=st.session_state.user_input,
                key="user_input_b",
                placeholder=f"Nhập phản biện với tư cách {config.persona_b}..."
            )
            
            if st.button("🚀 Gửi phản biện", key="send_b"):
                if user_input.strip():
                    st.session_state.user_input = user_input
                    execute_user_turn("USER_B", user_input.strip())
                    st.session_state.user_input = ""
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập nội dung phản biện!")
        
        elif current_turn == "USER_C":
            st.subheader(f"💬 Lượt của bạn (Bên C - {config.persona_c})")
            
            # Tìm tin nhắn cuối cùng
            if st.session_state.dialog_a and st.session_state.dialog_b:
                last_a_msg = st.session_state.dialog_a[-1]
                last_b_msg = st.session_state.dialog_b[-1]
                st.info(f"**{config.persona_a}:** {last_a_msg[:100]}...")
                st.info(f"**{config.persona_b}:** {last_b_msg[:100]}...")
            
            user_input = st.text_area(
                "Quan điểm của bạn:",
                value=st.session_state.user_input,
                key="user_input_c",
                placeholder=f"Nhập quan điểm với tư cách {config.persona_c}..."
            )
            
            if st.button("🚀 Gửi quan điểm", key="send_c"):
                if user_input.strip():
                    st.session_state.user_input = user_input
                    execute_user_turn("USER_C", user_input.strip())
                    st.session_state.user_input = ""
                    st.rerun()
                else:
                    st.warning("Vui lòng nhập nội dung!")
        
        elif current_turn in ["A", "B"]:
            # Lượt của AI - hiển thị nút để tiếp tục
            st.subheader(f"⏳ Đang chờ lượt của {config.persona_a if current_turn == 'A' else config.persona_b}...")
            
            if st.button("▶️ Tiếp tục tranh luận", key="continue_ai"):
                with st.spinner(f"{config.persona_a if current_turn == 'A' else config.persona_b} đang suy nghĩ..."):
                    execute_ai_turn(current_turn)
                    st.rerun()
    
    # Nếu debate đã kết thúc
    if not st.session_state.debate_running or game_over:
        st.markdown("---")
        st.subheader("🎯 Tranh luận đã kết thúc")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Tranh luận mới", type="primary"):
                st.session_state.page = "home"
                st.rerun()
        
        with col2:
            # Tạo transcript
            transcript_lines = []
            max_len = max(len(st.session_state.dialog_a), 
                         len(st.session_state.dialog_b),
                         len(st.session_state.dialog_c))
            
            for i in range(max_len):
                if i < len(st.session_state.dialog_a):
                    transcript_lines.append(f"A{i+1} ({config.persona_a}): {st.session_state.dialog_a[i]}")
                if i < len(st.session_state.dialog_b):
                    transcript_lines.append(f"B{i+1} ({config.persona_b}): {st.session_state.dialog_b[i]}")
                if i < len(st.session_state.dialog_c):
                    transcript_lines.append(f"C{i+1} ({config.persona_c}): {st.session_state.dialog_c[i]}")
            
            transcript = "\n".join(transcript_lines)
            
            st.download_button(
                "📥 Tải Transcript",
                data=transcript,
                file_name=f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col3:
            if config.mode != "Tham gia 3 bên (Thành viên C)":
                if st.button("⚖️ Phân tích AI"):
                    with st.spinner("Đang phân tích..."):
                        analysis_prompt = f"""
                        Phân tích cuộc tranh luận sau:
                        
                        Chủ đề: {st.session_state.topic_used}
                        
                        Transcript:
                        {transcript[:3000]}
                        
                        Hãy phân tích:
                        1. Điểm mạnh của mỗi bên
                        2. Lỗi logic/ngụy biện nếu có
                        3. Kết luận ai thuyết phục hơn
                        """
                        
                        analysis = call_chat(
                            [{"role": "user", "content": analysis_prompt}],
                            max_tokens=1000
                        )
                        
                        st.session_state.courtroom_analysis = analysis
            
            if st.session_state.courtroom_analysis:
                st.markdown("---")
                st.subheader("📋 Phân tích của AI")
                st.markdown(st.session_state.courtroom_analysis)

# --- CSS Style ---
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

# --- Main App ---
def main():
    """Hàm chính điều hướng ứng dụng"""
    st.set_page_config(
        page_title="🤖 AI Debate Bot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CHAT_STYLE, unsafe_allow_html=True)
    
    if st.session_state.page == "home":
        render_home()
    else:
        render_debate()

if __name__ == "__main__":
    main()
