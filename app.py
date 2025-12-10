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
class DebateState:
    current_turn: str = "A"  # "A", "B", "C", "USER_B", "USER_C"
    turn_count: int = 0
    is_fast_mode: bool = False
    is_auto_playing: bool = False
    current_display_index: int = 0
    waiting_for_user: bool = False

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
    
    if "debate_state" not in st.session_state:
        st.session_state.debate_state = DebateState()
    
    if "debate_running" not in st.session_state:
        st.session_state.debate_running = False
    
    if "courtroom_analysis" not in st.session_state:
        st.session_state.courtroom_analysis = None
    
    if "rpg_state" not in st.session_state:
        st.session_state.rpg_state = RPGState()
    
    if "user_input_b" not in st.session_state:
        st.session_state.user_input_b = ""
    
    if "user_input_c" not in st.session_state:
        st.session_state.user_input_c = ""
    
    if "topic_used" not in st.session_state:
        st.session_state.topic_used = ""
    
    if "final_style" not in st.session_state:
        st.session_state.final_style = ""
    
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = False
    
    if "debate_finished" not in st.session_state:
        st.session_state.debate_finished = False

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
        # Fallback
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
    
    A ({config.persona_a}): Ủng hộ chủ đề, 3-4 câu
    B ({config.persona_b}): Phản đối chủ đề, 3-4 câu
    """
    
    if config.mode == "Tham gia 3 bên (Thành viên C)":
        prompt += f"C ({config.persona_c}): Quan điểm trung lập, 3-4 câu\n"
    
    response = call_chat([{"role": "user", "content": prompt}])
    
    # Parse response
    import re
    
    # Tìm các phần bằng regex
    a_match = re.search(r'A[:\-]?\s*(.*?)(?:\n\n|\nB|$)', response, re.DOTALL | re.IGNORECASE)
    b_match = re.search(r'B[:\-]?\s*(.*?)(?:\n\n|\nC|$)', response, re.DOTALL | re.IGNORECASE)
    c_match = re.search(r'C[:\-]?\s*(.*?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    
    a_opening = a_match.group(1).strip() if a_match else "Xin chào, tôi ủng hộ chủ đề này."
    b_opening = b_match.group(1).strip() if b_match else "Tôi phản đối chủ đề này."
    c_opening = c_match.group(1).strip() if c_match and config.mode == "Tham gia 3 bên (Thành viên C)" else ""
    
    return a_opening, b_opening, c_opening

def generate_ai_reply(speaker: str, last_message: str = "") -> str:
    """Tạo câu trả lời cho AI"""
    config = st.session_state.config
    
    if speaker == "A":
        persona = config.persona_a
        role = "Ủng hộ"
        opponent = config.persona_b
    else:  # speaker == "B"
        persona = config.persona_b
        role = "Phản đối"
        opponent = config.persona_a
    
    prompt = f"""
    Bạn là {persona} ({role}) trong tranh luận.
    Chủ đề: {st.session_state.topic_used}
    Phong cách: {config.style if not config.custom_style else config.custom_style}
    
    Lời vừa rồi của đối phương: "{last_message[:300]}"
    
    Hãy trả lời ngắn gọn, sắc bén (3-5 câu) theo tính cách {persona}.
    """
    
    return call_chat([{"role": "user", "content": prompt}])

def calculate_rpg_damage(message: str, attacker: str, defender: str) -> Dict:
    """Tính toán sát thương RPG"""
    # Độ dài tin nhắn ảnh hưởng đến damage
    length_factor = min(1.0, len(message) / 500)
    
    # Tính điểm "chất lượng" dựa trên từ khóa
    quality_keywords = ["logic", "chứng minh", "bằng chứng", "thực tế", "khoa học", "thuyết phục"]
    quality_score = sum(1 for keyword in quality_keywords if keyword.lower() in message.lower())
    quality_factor = 1 + (quality_score * 0.2)
    
    # Damage cơ bản
    base_damage = random.randint(8, 15)
    final_damage = int(base_damage * length_factor * quality_factor)
    
    # Cơ hội chí mạng 15%
    is_crit = random.random() < 0.15
    if is_crit:
        final_damage = int(final_damage * 1.8)
    
    # Giới hạn damage
    final_damage = max(5, min(35, final_damage))
    
    attacker_name = st.session_state.config.persona_a if attacker == "A" else st.session_state.config.persona_b
    defender_name = st.session_state.config.persona_a if defender == "A" else st.session_state.config.persona_b
    
    reasons = [
        "Lập luận sắc bén",
        "Dẫn chứng thuyết phục",
        "Phản biện logic",
        "Chỉ ra điểm yếu",
        "Đưa ra giải pháp"
    ]
    
    return {
        "damage": final_damage,
        "is_crit": is_crit,
        "reason": random.choice(reasons),
        "attacker": attacker_name,
        "defender": defender_name
    }

def apply_rpg_damage(attacker: str, defender: str, message: str):
    """Áp dụng sát thương RPG"""
    if st.session_state.config.mode != "Chế độ RPG (Game Tranh luận)":
        return
    
    damage_data = calculate_rpg_damage(message, attacker, defender)
    
    # Áp dụng damage
    if defender == "A":
        st.session_state.rpg_state.hp_a = max(0, st.session_state.rpg_state.hp_a - damage_data["damage"])
    else:
        st.session_state.rpg_state.hp_b = max(0, st.session_state.rpg_state.hp_b - damage_data["damage"])
    
    # Ghi log
    crit_text = "🔥 **CHÍ MẠNG!** " if damage_data["is_crit"] else ""
    log_msg = f"{damage_data['attacker']} → {damage_data['defender']}: {crit_text}-{damage_data['damage']} HP ({damage_data['reason']})"
    st.session_state.rpg_state.log.append(log_msg)
    
    # Giới hạn log
    if len(st.session_state.rpg_state.log) > 10:
        st.session_state.rpg_state.log = st.session_state.rpg_state.log[-10:]

def check_victory() -> Tuple[bool, str]:
    """Kiểm tra điều kiện chiến thắng"""
    config = st.session_state.config
    
    if config.mode == "Chế độ RPG (Game Tranh luận)":
        rpg = st.session_state.rpg_state
        
        if rpg.hp_a <= 0 and rpg.hp_b <= 0:
            return True, f"🏳️ **HÒA!** Cả {config.persona_a} và {config.persona_b} đều hết máu."
        elif rpg.hp_a <= 0:
            return True, f"🏆 **{config.persona_b} CHIẾN THẮNG!**"
        elif rpg.hp_b <= 0:
            return True, f"🏆 **{config.persona_a} CHIẾN THẮNG!**"
    
    # Kiểm tra nếu đã đủ số rounds
    if len(st.session_state.dialog_a) >= config.rounds:
        if config.mode == "Tranh luận 2 AI (Tiêu chuẩn)":
            if len(st.session_state.dialog_b) >= config.rounds:
                return True, "✅ **Tranh luận đã hoàn thành!**"
        elif config.mode == "Tranh luận 1v1 với AI":
            if len(st.session_state.dialog_b) >= config.rounds:
                return True, "✅ **Tranh luận đã hoàn thành!**"
        elif config.mode == "Chế độ RPG (Game Tranh luận)":
            if len(st.session_state.dialog_b) >= config.rounds:
                return True, "✅ **Tranh luận đã hoàn thành!**"
        elif config.mode == "Tham gia 3 bên (Thành viên C)":
            if len(st.session_state.dialog_b) >= config.rounds and len(st.session_state.dialog_c) >= config.rounds:
                return True, "✅ **Tranh luận đã hoàn thành!**"
    
    return False, ""

def get_advantage_status() -> str:
    """Trả về trạng thái ưu thế hiện tại"""
    if st.session_state.config.mode != "Chế độ RPG (Game Tranh luận)":
        return ""
    
    rpg = st.session_state.rpg_state
    config = st.session_state.config
    
    if rpg.hp_a > rpg.hp_b:
        diff = rpg.hp_a - rpg.hp_b
        return f"🟢 **{config.persona_a} đang thắng thế** (+{diff} HP)"
    elif rpg.hp_b > rpg.hp_a:
        diff = rpg.hp_b - rpg.hp_a
        return f"🔴 **{config.persona_b} đang thắng thế** (+{diff} HP)"
    else:
        return "🟡 **Hai bên ngang nhau**"

def initialize_debate():
    """Khởi tạo cuộc tranh luận"""
    config = st.session_state.config
    
    with st.spinner("Đang khởi tạo cuộc tranh luận..."):
        a_open, b_open, c_open = generate_opening_statements()
        st.session_state.dialog_a.append(a_open)
        
        # Xác định chế độ để khởi tạo phù hợp
        if config.mode == "Tranh luận 1v1 với AI":
            # Chế độ 1v1: A mở đầu, chờ user nhập
            st.session_state.debate_state.waiting_for_user = True
            st.session_state.debate_state.current_turn = "USER_B"
            
        elif config.mode == "Tham gia 3 bên (Thành viên C)":
            # Chế độ 3 bên: A và B mở đầu, chờ user nhập
            st.session_state.dialog_b.append(b_open)
            st.session_state.debate_state.waiting_for_user = True
            st.session_state.debate_state.current_turn = "USER_C"
            
            if config.mode == "Chế độ RPG (Game Tranh luận)":
                apply_rpg_damage("A", "B", a_open)
                apply_rpg_damage("B", "A", b_open)
                
        else:
            # Chế độ 2 AI hoặc RPG: cả A và B đều AI
            st.session_state.dialog_b.append(b_open)
            st.session_state.debate_state.current_turn = "B"
            st.session_state.debate_state.waiting_for_user = False
            
            if config.mode == "Chế độ RPG (Game Tranh luận)":
                apply_rpg_damage("A", "B", a_open)
                apply_rpg_damage("B", "A", b_open)
        
        st.session_state.debate_started = True
        st.rerun()

def add_ai_turn_auto():
    """Thêm lượt AI tự động (cho chế độ 2 AI)"""
    config = st.session_state.config
    
    if not st.session_state.dialog_a:
        return
    
    # Thêm lượt cho A (nếu cần)
    if st.session_state.debate_state.current_turn == "A":
        last_b = st.session_state.dialog_b[-1] if st.session_state.dialog_b else ""
        reply_a = generate_ai_reply("A", last_b)
        st.session_state.dialog_a.append(reply_a)
        
        if config.mode == "Chế độ RPG (Game Tranh luận)" and last_b:
            apply_rpg_damage("A", "B", reply_a)
        
        st.session_state.debate_state.current_turn = "B"
    
    # Thêm lượt cho B (nếu cần)
    elif st.session_state.debate_state.current_turn == "B":
        last_a = st.session_state.dialog_a[-1] if st.session_state.dialog_a else ""
        reply_b = generate_ai_reply("B", last_a)
        st.session_state.dialog_b.append(reply_b)
        
        if config.mode == "Chế độ RPG (Game Tranh luận)" and last_a:
            apply_rpg_damage("B", "A", reply_b)
        
        st.session_state.debate_state.current_turn = "A"
    
    st.session_state.debate_state.turn_count += 1

def process_user_reply(user_role: str, message: str):
    """Xử lý phản hồi của người dùng"""
    config = st.session_state.config
    
    if user_role == "USER_B":
        # Chế độ 1v1: User là B
        st.session_state.dialog_b.append(message)
        st.session_state.user_input_b = ""
        st.session_state.debate_state.waiting_for_user = False
        st.session_state.debate_state.current_turn = "A"
        
        # Áp dụng RPG damage nếu cần
        if config.mode == "Chế độ RPG (Game Tranh luận)":
            apply_rpg_damage("B", "A", message)
        
        # AI tự động trả lời nếu chưa đủ rounds
        if len(st.session_state.dialog_a) < config.rounds:
            with st.spinner(f"{config.persona_a} đang trả lời..."):
                last_b = message
                reply_a = generate_ai_reply("A", last_b)
                st.session_state.dialog_a.append(reply_a)
                
                if config.mode == "Chế độ RPG (Game Tranh luận)":
                    apply_rpg_damage("A", "B", reply_a)
                
                # Chuyển sang chờ user tiếp
                st.session_state.debate_state.waiting_for_user = True
                st.session_state.debate_state.current_turn = "USER_B"
    
    elif user_role == "USER_C":
        # Chế độ 3 bên: User là C
        st.session_state.dialog_c.append(message)
        st.session_state.user_input_c = ""
        st.session_state.debate_state.waiting_for_user = False
        
        # A và B tự động trả lời nếu chưa đủ rounds
        if len(st.session_state.dialog_a) < config.rounds:
            with st.spinner(f"{config.persona_a} và {config.persona_b} đang tranh luận..."):
                # A trả lời C
                reply_a = generate_ai_reply("A", message)
                st.session_state.dialog_a.append(reply_a)
                
                # B trả lời A
                reply_b = generate_ai_reply("B", reply_a)
                st.session_state.dialog_b.append(reply_b)
                
                if config.mode == "Chế độ RPG (Game Tranh luận)":
                    apply_rpg_damage("A", "B", reply_a)
                    apply_rpg_damage("B", "A", reply_b)
                
                # Chuyển sang chờ user tiếp
                st.session_state.debate_state.waiting_for_user = True
                st.session_state.debate_state.current_turn = "USER_C"

# --- UI Components ---
def render_hp_display():
    """Hiển thị thanh HP và nhật ký"""
    config = st.session_state.config
    rpg = st.session_state.rpg_state
    
    if config.mode != "Chế độ RPG (Game Tranh luận)":
        return
    
    st.markdown("---")
    
    # Container cho thông tin RPG
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {config.persona_a}")
            hp_percent_a = max(0, rpg.hp_a)
            hp_color_a = "#4cd964" if hp_percent_a > 70 else ("#ff9500" if hp_percent_a > 30 else "#ff3b30")
            
            st.markdown(f"""
            <div style="background-color: #1e2d42; border-radius: 10px; height: 30px; overflow: hidden; margin: 10px 0; border: 2px solid {hp_color_a};">
                <div style="height: 100%; width: {hp_percent_a}%; background: linear-gradient(to right, {hp_color_a}, {hp_color_a}cc); 
                            display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                    {hp_percent_a}% ({rpg.hp_a} HP)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### {config.persona_b}")
            hp_percent_b = max(0, rpg.hp_b)
            hp_color_b = "#4cd964" if hp_percent_b > 70 else ("#ff9500" if hp_percent_b > 30 else "#ff3b30")
            
            st.markdown(f"""
            <div style="background-color: #1e2d42; border-radius: 10px; height: 30px; overflow: hidden; margin: 10px 0; border: 2px solid {hp_color_b};">
                <div style="height: 100%; width: {hp_percent_b}%; background: linear-gradient(to right, {hp_color_b}, {hp_color_b}cc); 
                            display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px;">
                    {hp_percent_b}% ({rpg.hp_b} HP)
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Hiển thị trạng thái ưu thế
    advantage = get_advantage_status()
    if advantage and not st.session_state.debate_finished:
        st.info(advantage)
    
    # Nhật ký chiến đấu
    if rpg.log:
        with st.expander("📜 Nhật ký chiến đấu", expanded=True):
            for log in reversed(rpg.log[-8:]):
                st.write(f"• {log}")
    
    st.markdown("---")

def render_control_buttons():
    """Hiển thị các nút điều khiển"""
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())
    
    # Đảm bảo waiting_for_user tồn tại
    if not hasattr(debate_state, 'waiting_for_user'):
        debate_state.waiting_for_user = False
    
    # Chỉ hiển thị nút điều khiển nếu không phải đang chờ user nhập
    if not debate_state.waiting_for_user:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Tiếp tục", use_container_width=True, 
                        disabled=st.session_state.get('debate_finished', False)):
                if not st.session_state.get('debate_started', False):
                    initialize_debate()
                else:
                    with st.spinner("Đang thêm lượt tranh luận..."):
                        add_ai_turn_auto()
                        
                        # Kiểm tra chiến thắng
                        is_victory, victory_msg = check_victory()
                        if is_victory:
                            st.session_state.debate_finished = True
                            st.session_state.debate_running = False
                        
                        st.rerun()
        
        with col2:
            # Tính năng tua nhanh (chỉ cho chế độ AI vs AI)
            if config.mode in ["Tranh luận 2 AI (Tiêu chuẩn)", "Chế độ RPG (Game Tranh luận)"]:
                if debate_state.is_fast_mode:
                    if st.button("⏸️ Dừng tua", use_container_width=True):
                        debate_state.is_fast_mode = False
                        st.rerun()
                else:
                    if st.button("⏩ Tua nhanh", use_container_width=True, 
                                disabled=st.session_state.get('debate_finished', False)):
                        debate_state.is_fast_mode = True
                        
                        # Tua nhanh đến khi đủ rounds
                        target_rounds = config.rounds
                        
                        with st.spinner(f"Đang tua nhanh đến {target_rounds} lượt..."):
                            while len(st.session_state.dialog_a) < target_rounds:
                                add_ai_turn_auto()
                                time.sleep(0.1)
                        
                        debate_state.is_fast_mode = False
                        st.session_state.debate_finished = True
                        st.session_state.debate_running = False
                        st.rerun()
            else:
                st.button("⏩ Tua nhanh", disabled=True, use_container_width=True,
                         help="Tính năng chỉ khả dụng cho chế độ AI vs AI")
        
        with col3:
            # Thêm 1 lượt (chỉ cho chế độ AI vs AI)
            if config.mode in ["Tranh luận 2 AI (Tiêu chuẩn)", "Chế độ RPG (Game Tranh luận)"]:
                if st.button("➕ Thêm 1 lượt", use_container_width=True,
                           disabled=st.session_state.get('debate_finished', False)):
                    with st.spinner("Đang thêm lượt..."):
                        add_ai_turn_auto()
                        
                        # Kiểm tra chiến thắng
                        is_victory, victory_msg = check_victory()
                        if is_victory:
                            st.session_state.debate_finished = True
                            st.session_state.debate_running = False
                        
                        st.rerun()
            else:
                st.button("➕ Thêm 1 lượt", disabled=True, use_container_width=True,
                         help="Tính năng chỉ khả dụng cho chế độ AI vs AI")
        
        with col4:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.session_state.debate_state.current_display_index = 0
                st.rerun()

def render_user_input():
    """Hiển thị ô input cho người dùng"""
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())
    
    # Đảm bảo waiting_for_user tồn tại
    if not hasattr(debate_state, 'waiting_for_user'):
        debate_state.waiting_for_user = False
    
    if not debate_state.waiting_for_user:
        return
    
    st.markdown("---")
    
    if debate_state.current_turn == "USER_B":
        # Chế độ 1v1
        st.subheader(f"💬 Lượt của bạn ({config.persona_b})")
        
        # Hiển thị tin nhắn cuối cùng của A
        if st.session_state.dialog_a:
            last_a_msg = st.session_state.dialog_a[-1]
            with st.container():
                st.markdown(f"""
                <div style="background-color: #1e2d42; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 4px solid #58a6ff;">
                    <strong>{config.persona_a} vừa nói:</strong><br>
                    {last_a_msg[:300]}...
                </div>
                """, unsafe_allow_html=True)
        
        # Ô input cho user
        user_input = st.text_area(
            "Phản biện của bạn:",
            value=st.session_state.get('user_input_b', ''),
            key="user_input_b_area",
            placeholder=f"Nhập phản biện với tư cách {config.persona_b}...",
            height=120
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Gửi", key="send_b", use_container_width=True):
                if user_input.strip():
                    st.session_state.user_input_b = user_input
                    with st.spinner("Đang xử lý..."):
                        process_user_reply("USER_B", user_input.strip())
                        st.rerun()
                else:
                    st.warning("Vui lòng nhập nội dung phản biện!")
        
        with col2:
            if st.button("🗑️ Xóa", key="clear_b", type="secondary", use_container_width=True):
                st.session_state.user_input_b = ""
                st.rerun()
    
    elif debate_state.current_turn == "USER_C":
        # Chế độ 3 bên
        st.subheader(f"💬 Lượt của bạn ({config.persona_c})")
        
        # Hiển thị tin nhắn cuối cùng của A và B
        if st.session_state.dialog_a and st.session_state.dialog_b:
            last_a_msg = st.session_state.dialog_a[-1]
            last_b_msg = st.session_state.dialog_b[-1]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="background-color: #1f362d; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #2a4a3d;">
                    <strong>{config.persona_a}:</strong><br>
                    {last_a_msg[:150]}...
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div style="background-color: #3b2225; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #4d2c30;">
                    <strong>{config.persona_b}:</strong><br>
                    {last_b_msg[:150]}...
                </div>
                """, unsafe_allow_html=True)
        
        # Ô input cho user
        user_input = st.text_area(
            "Quan điểm của bạn:",
            value=st.session_state.get('user_input_c', ''),
            key="user_input_c_area",
            placeholder=f"Nhập quan điểm với tư cách {config.persona_c}...",
            height=120
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Gửi", key="send_c", use_container_width=True):
                if user_input.strip():
                    st.session_state.user_input_c = user_input
                    with st.spinner("Đang xử lý..."):
                        process_user_reply("USER_C", user_input.strip())
                        st.rerun()
                else:
                    st.warning("Vui lòng nhập nội dung!")
        
        with col2:
            if st.button("🗑️ Xóa", key="clear_c", type="secondary", use_container_width=True):
                st.session_state.user_input_c = ""
                st.rerun()

def render_chat_messages():
    """Hiển thị các tin nhắn trong chat"""
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())
    
    # Xác định số tin nhắn cần hiển thị
    if debate_state.is_fast_mode:
        display_count = max(len(st.session_state.dialog_a), 
                           len(st.session_state.dialog_b),
                           len(st.session_state.dialog_c))
    else:
        display_count = debate_state.current_display_index + 1
        display_count = min(display_count, 
                          max(len(st.session_state.dialog_a), 
                              len(st.session_state.dialog_b),
                              len(st.session_state.dialog_c)))
    
    # Hiển thị từng tin nhắn
    for i in range(display_count):
        if i < len(st.session_state.dialog_a):
            st.markdown(f"""
            <div class="chat-container">
                <div class="chat-bubble chat-left">
                    <div class="speaker-header">
                        <span class="speaker-name">A{i+1} ({config.persona_a})</span>
                    </div>
                    <div class="message-content">
                        {st.session_state.dialog_a[i]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(st.session_state.dialog_b):
            st.markdown(f"""
            <div class="chat-container" style="justify-content: flex-end;">
                <div class="chat-bubble chat-right">
                    <div class="speaker-header">
                        <span class="speaker-name">B{i+1} ({config.persona_b})</span>
                    </div>
                    <div class="message-content">
                        {st.session_state.dialog_b[i]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(st.session_state.dialog_c) and config.mode == "Tham gia 3 bên (Thành viên C)":
            st.markdown(f"""
            <div class="chat-container" style="justify-content: center;">
                <div class="chat-bubble chat-user">
                    <div class="speaker-header">
                        <span class="speaker-name">C{i+1} ({config.persona_c})</span>
                    </div>
                    <div class
