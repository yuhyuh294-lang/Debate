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
            hp_percent_a = max(0, min(100, rpg.hp_a))
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
            hp_percent_b = max(0, min(100, rpg.hp_b))
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
    max_messages = max(len(st.session_state.dialog_a), 
                      len(st.session_state.dialog_b),
                      len(st.session_state.dialog_c))
    
    if debate_state.is_fast_mode:
        display_count = max_messages
    else:
        display_count = min(debate_state.current_display_index + 1, max_messages)
    
    # Hiển thị từng tin nhắn theo thứ tự
    for i in range(display_count):
        # Hiển thị A
        if i < len(st.session_state.dialog_a):
            msg_a = st.session_state.dialog_a[i]
            if msg_a:  # Chỉ hiển thị nếu có nội dung
                st.markdown(f"""
                <div style="display: flex; width: 100%; margin: 5px 0; padding: 0;">
                    <div style="padding: 15px 20px; border-radius: 18px; margin: 8px 0; max-width: 75%; word-wrap: break-word; font-size: 15px; line-height: 1.6; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); transition: transform 0.2s ease; position: relative; background: linear-gradient(135deg, #1f362d 0%, #2a4a3d 100%); color: #e0f7e9 !important; margin-right: auto; border-top-left-radius: 4px; border: 1px solid #2a4a3d;">
                        <div style="margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                            <span style="font-weight: bold; font-size: 14px; letter-spacing: 0.5px; display: block; color: #4cd964 !important;">A{i+1} ({config.persona_a})</span>
                        </div>
                        <div style="font-size: 15px; line-height: 1.7; margin-top: 5px;">
                            {msg_a}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Hiển thị B
        if i < len(st.session_state.dialog_b):
            msg_b = st.session_state.dialog_b[i]
            if msg_b:  # Chỉ hiển thị nếu có nội dung
                st.markdown(f"""
                <div style="display: flex; width: 100%; margin: 5px 0; padding: 0; justify-content: flex-end;">
                    <div style="padding: 15px 20px; border-radius: 18px; margin: 8px 0; max-width: 75%; word-wrap: break-word; font-size: 15px; line-height: 1.6; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); transition: transform 0.2s ease; position: relative; background: linear-gradient(135deg, #3b2225 0%, #4d2c30 100%); color: #ffe5d9 !important; margin-left: auto; border-top-right-radius: 4px; border: 1px solid #4d2c30;">
                        <div style="margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                            <span style="font-weight: bold; font-size: 14px; letter-spacing: 0.5px; display: block; color: #ff9500 !important;">B{i+1} ({config.persona_b})</span>
                        </div>
                        <div style="font-size: 15px; line-height: 1.7; margin-top: 5px;">
                            {msg_b}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Hiển thị C (nếu có)
        if i < len(st.session_state.dialog_c) and config.mode == "Tham gia 3 bên (Thành viên C)":
            msg_c = st.session_state.dialog_c[i]
            if msg_c:  # Chỉ hiển thị nếu có nội dung
                st.markdown(f"""
                <div style="display: flex; width: 100%; margin: 5px 0; padding: 0; justify-content: center;">
                    <div style="padding: 15px 20px; border-radius: 18px; margin: 8px 0; max-width: 85%; word-wrap: break-word; font-size: 15px; line-height: 1.6; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); transition: transform 0.2s ease; position: relative; background: linear-gradient(135deg, #192f44 0%, #2a3f5f 100%); color: #d6e4ff !important; margin: 15px auto; border-radius: 18px; border: 1px solid #2a3f5f;">
                        <div style="margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                            <span style="font-weight: bold; font-size: 14px; letter-spacing: 0.5px; display: block; color: #8bb8e8 !important;">C{i+1} ({config.persona_c})</span>
                        </div>
                        <div style="font-size: 15px; line-height: 1.7; margin-top: 5px;">
                            {msg_c}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tự động tăng display index nếu chưa ở fast mode
    if not debate_state.is_fast_mode and debate_state.current_display_index < max_messages:
        debate_state.current_display_index += 1
        time.sleep(0.3)
        st.rerun()

def run_courtroom_analysis():
    """Chạy phân tích phiên tòa AI"""
    config = st.session_state.config
    
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
    
    prompt = f"""
    Bạn là Thẩm phán AI tối cao. Hãy phân tích cuộc tranh luận sau:
    
    **CHỦ ĐỀ:** {st.session_state.topic_used}
    **PHONG CÁCH:** {st.session_state.final_style}
    
    **TRANSCRIPT:**
    {transcript[:2500]}
    
    Hãy phân tích theo cấu trúc sau:
    
    ### 1. PHÂN TÍCH LẬP LUẬN
    - Điểm mạnh của mỗi bên
    - Lỗi logic/ngụy biện được sử dụng
    - Tính chặt chẽ của lập luận
    
    ### 2. PHÁN QUYẾT
    - Ai có lập luận thuyết phục hơn?
    - Tại sao?
    
    ### 3. KHUYẾN NGHỊ
    - Điểm cần cải thiện cho mỗi bên
    - Cách tranh luận hiệu quả hơn
    
    Phân tích chi tiết, khách quan.
    """
    
    with st.spinner("⏳ Đang phân tích chi tiết..."):
        analysis = call_chat(
            [{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        st.session_state.courtroom_analysis = analysis

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
            api_options.append("GitHub Models")
        if OPENAI_API_KEY:
            api_options.append("OpenAI Official")
        
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
        
        if st.button("🔄 Reset Debate", type="secondary", use_container_width=True):
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
            placeholder="Ví dụ: Giai cấp thống trị và bị trị"
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
            if st.button(topic[:80], key=f"topic_{topic[:10]}", use_container_width=True):
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
            st.session_state.debate_state = DebateState()
            st.session_state.debate_running = True
            st.session_state.debate_started = False
            st.session_state.debate_finished = False
            st.session_state.topic_used = st.session_state.config.topic
            st.session_state.final_style = st.session_state.config.custom_style if st.session_state.config.custom_style else st.session_state.config.style
            st.session_state.courtroom_analysis = None
            st.session_state.user_input_b = ""
            st.session_state.user_input_c = ""
            st.session_state.page = "debate"
            st.rerun()

def render_debate():
    """Trang tranh luận chính"""
    st.title("🔥 Cuộc tranh luận")
    
    config = st.session_state.config
    
    # Sidebar info - sửa lại để không có ô xanh dư thừa
    with st.sidebar:
        st.header("📊 Thông tin")
        
        # Gộp tất cả vào một card duy nhất
        info_html = f"""
        <div style="background-color: #1e2d42; padding: 15px; border-radius: 10px; border-left: 4px solid #58a6ff;">
            <p style="margin: 8px 0;"><strong>Chế độ:</strong> {config.mode}</p>
            <p style="margin: 8px 0;"><strong>Chủ đề:</strong> {st.session_state.topic_used}</p>
            <p style="margin: 8px 0;"><strong>Phong cách:</strong> {st.session_state.final_style}</p>
        """
        
        if config.mode == "Chế độ RPG (Game Tranh luận)":
            rpg = st.session_state.rpg_state
            info_html += f"""
            <p style="margin: 8px 0;"><strong>{config.persona_a}:</strong> {rpg.hp_a} HP</p>
            <p style="margin: 8px 0;"><strong>{config.persona_b}:</strong> {rpg.hp_b} HP</p>
            """
        
        info_html += "</div>"
        
        st.markdown(info_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🔙 Về trang chủ", use_container_width=True, key="back_home"):
            st.session_state.page = "home"
            st.rerun()
    
    # Header với thông tin
    st.header(f"Chủ đề: {st.session_state.topic_used}")
    
    # Container thông tin cuộc tranh luận
    with st.container():
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(f"**Chế độ:** {config.mode}")
            st.markdown(f"**Phong cách:** {st.session_state.final_style}")
        
        with info_col2:
            st.markdown(f"**Bên A:** {config.persona_a}")
            st.markdown(f"**Bên B:** {config.persona_b}")
            
            if config.mode == "Tham gia 3 bên (Thành viên C)":
                st.markdown(f"**Bên C:** {config.persona_c}")
    
    # Hiển thị thanh HP và nhật ký (nếu là chế độ RPG)
    if config.mode == "Chế độ RPG (Game Tranh luận)":
        render_hp_display()
    
    # CHỈ MỘT DÒNG KẺ DUY NHẤT trước các nút điều khiển
    st.markdown("---")
    
    # Hiển thị các nút điều khiển
    render_control_buttons()
    
    # Hiển thị ô input cho người dùng (nếu đang chờ)
    render_user_input()
    
    # Hiển thị tin nhắn chat
    render_chat_messages()
    
    # Kiểm tra và hiển thị kết quả
    is_victory, victory_msg = check_victory()
    if is_victory:
        st.session_state.debate_finished = True
        st.session_state.debate_running = False
        
        st.markdown("---")
        
        # Hiển thị thông báo chiến thắng
        if "CHIẾN THẮNG" in victory_msg or "HÒA" in victory_msg:
            st.success(victory_msg)
        else:
            st.info(victory_msg)
        
        # Hiển thị ưu thế nếu chưa có bên nào hết HP
        if config.mode == "Chế độ RPG (Game Tranh luận)" and "CHIẾN THẮNG" not in victory_msg and "HÒA" not in victory_msg:
            advantage = get_advantage_status()
            if advantage:
                st.info(advantage)
    
    # Phần kết thúc và tùy chọn
    if st.session_state.debate_finished:
        st.markdown("---")
        
        # Nút phân tích AI
        if config.mode != "Tham gia 3 bên (Thành viên C)":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⚖️ Phân tích AI", use_container_width=True, type="secondary", key="ai_analysis"):
                    run_courtroom_analysis()
                    st.rerun()
            
            with col2:
                # Tạo transcript để tải
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
                    mime="text/plain",
                    use_container_width=True,
                    key="download_transcript"
                )
            
            with col3:
                if st.button("🔄 Tranh luận mới", type="primary", use_container_width=True, key="new_debate"):
                    st.session_state.page = "home"
                    st.rerun()
        
        # Hiển thị phân tích AI (full width)
        if st.session_state.courtroom_analysis:
            st.markdown("---")
            st.header("⚖️ Phân tích Phiên Tòa AI")
            
            # Container cho phân tích
            with st.container():
                st.markdown(st.session_state.courtroom_analysis)

# --- CSS Style ---
CHAT_STYLE = """
<style>
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #58a6ff;
    font-weight: 600;
}

/* Chat bubbles - Sửa phần trắng ở đầu */
.chat-bubble {
    padding: 15px 20px;
    border-radius: 18px;
    margin: 8px 0;
    max-width: 75%;
    word-wrap: break-word;
    font-size: 15px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease;
    position: relative;
}

/* Loại bỏ khoảng trắng thừa */
.chat-container {
    display: flex;
    width: 100%;
    margin: 5px 0;
    padding: 0;
}

.speaker-header {
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.speaker-name {
    font-weight: bold;
    font-size: 14px;
    letter-spacing: 0.5px;
    display: block;
}

.message-content {
    font-size: 15px;
    line-height: 1.7;
    margin-top: 5px;
}

/* Sidebar info card - Sửa ô xanh không nội dung */
.sidebar-card {
    background-color: #1e2d42;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    border-left: 4px solid #58a6ff;
}

.sidebar-card p {
    margin: 8px 0;
}

/* HP bars */
.hp-bar {
    background-color: #1e2d42;
    border-radius: 10px;
    height: 30px;
    overflow: hidden;
    margin: 10px 0;
    border: 2px solid;
}

.hp-fill {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 14px;
}

/* Control buttons */
.control-button {
    width: 100%;
    margin: 5px 0;
}

/* Battle log */
.battle-log {
    margin-top: 15px;
    padding-top: 15px;
}

/* Sửa khoảng cách giữa các phần tử */
div[data-testid="stHorizontalBlock"] {
    gap: 8px;
    margin: 5px 0;
}

/* Sửa padding của các container */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Text area */
.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #30363d;
    background-color: #0d1117;
    color: #c9d1d9;
    font-size: 15px;
    padding: 12px;
}

.stTextArea textarea:focus {
    border-color: #58a6ff;
    box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2);
}

/* Button styles */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s;
    border: none;
    padding: 10px 20px;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0d1117;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #58a6ff;
}

/* Success, Warning, Error messages */
.stSuccess {
    background: linear-gradient(135deg, #0e4429 0%, #1f362d 100%);
    border-left: 5px solid #4cd964;
}

.stWarning {
    background: linear-gradient(135deg, #423200 0%, #332700 100%);
    border-left: 5px solid #ffd60a;
}

.stError {
    background: linear-gradient(135deg, #58161b 0%, #3b2225 100%);
    border-left: 5px solid #ff3b30;
}

.stInfo {
    background: linear-gradient(135deg, #1e2d42 0%, #192f44 100%);
    border-left: 5px solid #58a6ff;
}

/* Streamlit divider line style */
hr {
    border: none;
    height: 1px;
    background-color: #30363d;
    margin: 20px 0;
}

/* Fix for extra spacing in containers */
[data-testid="stVerticalBlock"] > div {
    padding: 0 !important;
}

/* Fix for sidebar spacing */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 10px !important;
}

/* Remove extra borders */
[data-testid="stExpander"] {
    border: none !important;
    background-color: transparent !important;
}
</style>
"""

# --- Main App ---
def main():
    """Hàm chính điều hướng ứng dụng"""
    st.set_page_config(
        page_title="🤖 AI Debate Bot",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': "### AI Debate Bot\nTranh luận thông minh với AI"
        }
    )
    
    st.markdown(CHAT_STYLE, unsafe_allow_html=True)
    
    if st.session_state.page == "home":
        render_home()
    else:
        render_debate()

if __name__ == "__main__":
    main()
