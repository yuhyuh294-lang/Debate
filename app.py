import streamlit as st
from PIL import Image
import base64
import os
import time
import re
import random 
import json 
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
import io

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
    current_turn: str = "A"
    turn_count: int = 0
    is_fast_mode: bool = False
    is_auto_playing: bool = False
    current_display_index: int = 0
    waiting_for_user: bool = False

def init_session_state():
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "tab1"
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
    
    if "image_analysis_result" not in st.session_state:
        st.session_state.image_analysis_result = None
    
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
        st.session_state.page = "login"
    
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = False
    
    if "debate_finished" not in st.session_state:
        st.session_state.debate_finished = False
    
    if "_trigger_continue" not in st.session_state:
        st.session_state._trigger_continue = False
    
    if "suggestion_b" not in st.session_state:
        st.session_state.suggestion_b = ""
    
    if "suggestion_c" not in st.session_state:
        st.session_state.suggestion_c = ""
    
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    if "users_db" not in st.session_state:
        st.session_state.users_db = {}
    
    if "saved_debates" not in st.session_state:
        st.session_state.saved_debates = {}

init_session_state()

def strip_persona_prefix(text: str) -> str:
    return re.sub(r"^\s*\([^)]*\)\s*:\s*", "", text) if text else text

def get_api_client():
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

def call_chat(messages: List[Dict], model: str = None, temperature: float = None, max_tokens: int = None) -> str:
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

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image_for_topic(image: Image.Image) -> List[str]:
    try:
        base64_image = encode_image_to_base64(image)
        
        prompt = """
        Hãy phân tích hình ảnh này và đề xuất 3 chủ đề tranh luận thú vị, gây tranh cãi dựa trên nội dung hình ảnh.
        Mỗi chủ đề nên có tính tranh luận cao, có thể phân tích từ nhiều góc độ.
        Trả về dưới dạng:
        1. [Chủ đề 1]
        2. [Chủ đề 2]  
        3. [Chủ đề 3]
        
        Chỉ trả về danh sách chủ đề, không thêm giải thích gì khác.
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ]
        
        model_to_use = "gpt-4o" if OPENAI_API_KEY else st.session_state.config.model
        
        client = get_api_client()
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        topics_text = response.choices[0].message.content
        
        topics = []
        lines = topics_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                clean_line = re.sub(r'^[0-9\.\-\*\)\]]+\s*', '', line)
                if clean_line and len(clean_line) > 10:
                    topics.append(clean_line)
        
        return topics[:3] if topics else ["Không thể phân tích chủ đề từ hình ảnh này."]
            
    except Exception as e:
        st.error(f"Lỗi phân tích ảnh: {str(e)[:200]}")
        return ["Lỗi khi phân tích hình ảnh. Vui lòng thử lại."]

def generate_text_topics() -> List[str]:
    prompt = "Hãy đề xuất 3 chủ đề tranh luận thú vị, gây tranh cãi và đa chiều về xã hội, công nghệ, hoặc đạo đức. Trả về dưới dạng danh sách, mỗi chủ đề một dòng."
    
    response = call_chat([{"role": "user", "content": prompt}])
    
    topics = []
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 10:
            clean_line = re.sub(r'^[0-9\.\-\*\)\]]+\s*', '', line)
            if clean_line:
                topics.append(clean_line)
    
    return topics[:3] if topics else [
        "Trí tuệ nhân tạo sẽ thay thế hay bổ trợ cho con người?",
        "Công nghệ có đang làm con người cô đơn hơn?",
        "Nên ưu tiên phát triển kinh tế hay bảo vệ môi trường?"
    ]

def generate_opening_statements() -> tuple[str, str, str]:
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
    
    a_match = re.search(r'A[:\-]?\s*(.*?)(?:\n\n|\nB|$)', response, re.DOTALL | re.IGNORECASE)
    b_match = re.search(r'B[:\-]?\s*(.*?)(?:\n\n|\nC|$)', response, re.DOTALL | re.IGNORECASE)
    c_match = re.search(r'C[:\-]?\s*(.*?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    
    a_opening = strip_persona_prefix(a_match.group(1).strip()) if a_match else "Xin chào, tôi ủng hộ chủ đề này."
    b_opening = strip_persona_prefix(b_match.group(1).strip()) if b_match else "Tôi phản đối chủ đề này."
    c_opening = c_match.group(1).strip() if c_match and config.mode == "Tham gia 3 bên (Thành viên C)" else ""
    
    return a_opening, b_opening, c_opening

def generate_ai_reply(speaker: str, last_message: str = "") -> str:
    config = st.session_state.config
    
    if speaker == "A":
        persona = config.persona_a
        role = "Ủng hộ"
    else:
        persona = config.persona_b
        role = "Phản đối"
    
    prompt = f"""
Bạn đang tham gia một cuộc tranh luận với vai trò {role}.

CHỦ ĐỀ:
{st.session_state.topic_used}

PHONG CÁCH LẬP LUẬN (chỉ dùng nội bộ, KHÔNG được nhắc tới trong câu trả lời):
- {persona}
- {config.style if not config.custom_style else config.custom_style}

LỜI VỪA RỒI CỦA ĐỐI PHƯƠNG:
"{last_message[:300]}"

LUẬT BẮT BUỘC:
- KHÔNG giới thiệu bản thân
- KHÔNG nhắc tới vai trò, phong cách, persona
- KHÔNG dùng dấu ngoặc () để mở đầu
- Chỉ viết nội dung lập luận trực tiếp (3–5 câu)

Hãy trả lời ngay.
"""
    
    return call_chat([{"role": "user", "content": prompt}])

def calculate_rpg_damage(message: str, attacker: str, defender: str) -> Dict:
    length_factor = min(1.0, len(message) / 500)
    
    quality_keywords = ["logic", "chứng minh", "bằng chứng", "thực tế", "khoa học", "thuyết phục"]
    quality_score = sum(1 for keyword in quality_keywords if keyword.lower() in message.lower())
    quality_factor = 1 + (quality_score * 0.2)
    
    base_damage = random.randint(8, 15)
    final_damage = int(base_damage * length_factor * quality_factor)
    
    is_crit = random.random() < 0.15
    if is_crit:
        final_damage = int(final_damage * 1.8)
    
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
    if st.session_state.config.mode != "Chế độ RPG (Game Tranh luận)":
        return
    
    damage_data = calculate_rpg_damage(message, attacker, defender)
    
    if defender == "A":
        st.session_state.rpg_state.hp_a = max(0, st.session_state.rpg_state.hp_a - damage_data["damage"])
    else:
        st.session_state.rpg_state.hp_b = max(0, st.session_state.rpg_state.hp_b - damage_data["damage"])
    
    crit_text = "🔥 **CHÍ MẠNG!** " if damage_data["is_crit"] else ""
    log_msg = f"{damage_data['attacker']} → {damage_data['defender']}: {crit_text}-{damage_data['damage']} HP ({damage_data['reason']})"
    st.session_state.rpg_state.log.append(log_msg)
    
    if len(st.session_state.rpg_state.log) > 10:
        st.session_state.rpg_state.log = st.session_state.rpg_state.log[-10:]

def check_victory() -> tuple[bool, str]:
    config = st.session_state.config
    
    if config.mode == "Chế độ RPG (Game Tranh luận)":
        rpg = st.session_state.rpg_state
        
        if rpg.hp_a <= 0 and rpg.hp_b <= 0:
            return True, f"🏳️ **HÒA!** Cả {config.persona_a} và {config.persona_b} đều hết máu."
        elif rpg.hp_a <= 0:
            return True, f"🏆 **{config.persona_b} CHIẾN THẮNG!**"
        elif rpg.hp_b <= 0:
            return True, f"🏆 **{config.persona_a} CHIẾN THẮNG!**"
    
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
    config = st.session_state.config
    st.session_state.debate_state.current_display_index = 1
    with st.spinner("Đang khởi tạo cuộc tranh luận..."):
        a_open = generate_ai_reply("A", "")
        
        if config.mode != "Tranh luận 1v1 với AI":
            b_open = generate_ai_reply("B", a_open)
            st.session_state.dialog_b.append(strip_persona_prefix(b_open))
        
        st.session_state.dialog_a.append(strip_persona_prefix(a_open))

        if config.mode == "Tranh luận 1v1 với AI":
            st.session_state.debate_state.waiting_for_user = True
            st.session_state.debate_state.current_turn = "USER_B"
            
        elif config.mode == "Tham gia 3 bên (Thành viên C)":
            st.session_state.dialog_b.append(strip_persona_prefix(b_open))
            st.session_state.debate_state.waiting_for_user = True
            st.session_state.debate_state.current_turn = "USER_C"
            
            if config.mode == "Chế độ RPG (Game Tranh luận)":
                apply_rpg_damage("A", "B", a_open)
                apply_rpg_damage("B", "A", b_open)
                
        else:
            st.session_state.debate_state.current_turn = "B"
            st.session_state.debate_state.waiting_for_user = False
            
            if config.mode == "Chế độ RPG (Game Tranh luận)":
                apply_rpg_damage("A", "B", a_open)
                if config.mode != "Tranh luận 1v1 với AI":
                    apply_rpg_damage("B", "A", b_open)
        
        st.session_state.debate_started = True
        st.rerun()

def add_ai_turn_auto():
    config = st.session_state.config
    debate_state = st.session_state.debate_state

    last_b = st.session_state.dialog_b[-1] if st.session_state.dialog_b else ""

    reply_a = generate_ai_reply("A", last_b)
    st.session_state.dialog_a.append(reply_a)

    if config.mode == "Chế độ RPG (Game Tranh luận)" and last_b:
        apply_rpg_damage("A", "B", reply_a)

    reply_b = generate_ai_reply("B", reply_a)
    st.session_state.dialog_b.append(reply_b)

    if config.mode == "Chế độ RPG (Game Tranh luận)":
        apply_rpg_damage("B", "A", reply_b)

    debate_state.turn_count += 1
    debate_state.current_display_index += 1

def process_user_reply(user_role: str, message: str):
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())
    
    if config.mode == "Tranh luận 1v1 với AI" and user_role == "USER_B":
        if not message or not message.strip():
            st.warning("Vui lòng nhập nội dung!")
            return
            
        st.session_state.dialog_b.append(message.strip())
        st.session_state.user_input_b = ""
        
        reply_a = generate_ai_reply("A", message.strip())
        st.session_state.dialog_a.append(reply_a)
        
        debate_state.waiting_for_user = True
        debate_state.current_turn = "USER_B"
        debate_state.turn_count += 1
        
        if config.mode == "Chế độ RPG (Game Tranh luận)":
            apply_rpg_damage("A", "B", reply_a)
            
        is_victory, victory_msg = check_victory()
        if is_victory:
            st.session_state.debate_finished = True
            st.session_state.debate_running = False
            
        return

def get_writing_suggestion(user_role: str, context: str) -> str:
    config = st.session_state.config
    
    prompt = f"""
    Bạn là trợ lý viết phản biện. Người dùng đang tham gia tranh luận với vai trò:
    - Vai trò: {user_role}
    - Persona: {config.persona_b if user_role == "USER_B" else config.persona_c}
    - Chủ đề: {st.session_state.topic_used}
    
    Ngữ cảnh gần nhất:
    {context[:500]}
    
    Hãy đề xuất 2-3 ý tưởng ngắn gọn để người dùng phản biện tiếp theo.
    Mỗi ý tưởng không quá 1 câu, viết bằng tiếng Việt.
    """
    
    try:
        return call_chat([{"role": "user", "content": prompt}], max_tokens=200)
    except:
        return "1. Hãy tập trung vào điểm mâu thuẫn trong lập luận đối phương.\n2. Đưa ra ví dụ cụ thể để hỗ trợ quan điểm của bạn.\n3. Hỏi những câu hỏi thách thức giả định của đối phương."

def render_progress_bar():
    config = st.session_state.config
    
    if config.mode == "Tranh luận 1v1 với AI":
        current = len(st.session_state.dialog_a)
        total = config.rounds
    elif config.mode == "Tham gia 3 bên (Thành viên C)":
        current = len(st.session_state.dialog_a) + len(st.session_state.dialog_b) + len(st.session_state.dialog_c)
        total = config.rounds * 3
    else:
        current = len(st.session_state.dialog_a) + len(st.session_state.dialog_b)
        total = config.rounds * 2
    
    if total > 0:
        progress = min(1.0, current / total)
        st.progress(progress)
        st.caption(f"📊 **Tiến độ:** {current}/{total} lượt ({int(progress*100)}%)")

def handle_login(username: str) -> bool:
    if not username.strip():
        return False
    
    if username not in st.session_state.users_db:
        st.session_state.users_db[username] = {
            "created_at": datetime.now().isoformat(),
            "debates": []
        }
    
    st.session_state.current_user = username
    return True

def handle_logout():
    st.session_state.current_user = None
    st.session_state.page = "login"
    st.rerun()

def save_current_debate() -> bool:
    if not st.session_state.current_user:
        return False
    
    debate_id = f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    debate_data = {
        "id": debate_id,
        "topic": st.session_state.topic_used,
        "mode": st.session_state.config.mode,
        "created_at": datetime.now().isoformat(),
        "dialog_a": st.session_state.dialog_a,
        "dialog_b": st.session_state.dialog_b,
        "dialog_c": st.session_state.dialog_c,
        "config": {
            "persona_a": st.session_state.config.persona_a,
            "persona_b": st.session_state.config.persona_b,
            "style": st.session_state.final_style
        }
    }
    
    if "debates" not in st.session_state.users_db[st.session_state.current_user]:
        st.session_state.users_db[st.session_state.current_user]["debates"] = []
    
    st.session_state.users_db[st.session_state.current_user]["debates"].append(debate_data)
    
    if st.session_state.current_user not in st.session_state.saved_debates:
        st.session_state.saved_debates[st.session_state.current_user] = []
    
    st.session_state.saved_debates[st.session_state.current_user].append(debate_data)
    
    return True

def load_debate(debate_data: Dict):
    st.session_state.dialog_a = debate_data['dialog_a']
    st.session_state.dialog_b = debate_data['dialog_b']
    st.session_state.dialog_c = debate_data.get('dialog_c', [])
    st.session_state.topic_used = debate_data['topic']
    st.session_state.final_style = debate_data['config']['style']
    st.session_state.config.persona_a = debate_data['config']['persona_a']
    st.session_state.config.persona_b = debate_data['config']['persona_b']
    st.session_state.debate_finished = True
    st.session_state.page = "debate"
    st.rerun()

def render_hp_display():
    config = st.session_state.config
    rpg = st.session_state.rpg_state
    
    if config.mode != "Chế độ RPG (Game Tranh luận)":
        return
    
    st.markdown("---")
    
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
    
    advantage = get_advantage_status()
    if advantage and not st.session_state.debate_finished:
        st.info(advantage)
    
    if rpg.log:
        with st.expander("📜 Nhật ký chiến đấu", expanded=True):
            for log in reversed(rpg.log[-8:]):
                st.write(f"• {log}")

def render_control_buttons():
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())
    if not hasattr(debate_state, 'waiting_for_user'):
        debate_state.waiting_for_user = False
    
    if not debate_state.waiting_for_user:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Tiếp tục", use_container_width=True, 
                        disabled=st.session_state.get('debate_finished', False)):
                if not st.session_state.get('debate_started', False):
                    initialize_debate()
                else:
                    with st.spinner("Đang thêm lượt tranh luận..."):
                        add_ai_turn_auto()
                        
                        is_victory, victory_msg = check_victory()
                        if is_victory:
                            st.session_state.debate_finished = True
                            st.session_state.debate_running = False
                        
                        st.rerun()
        
        with col2:
            if config.mode in ["Tranh luận 2 AI (Tiêu chuẩn)", "Chế độ RPG (Game Tranh luận)"]:
                if debate_state.is_fast_mode:
                    if st.button("⏸️ Dừng tua", use_container_width=True):
                        debate_state.is_fast_mode = False
                        st.rerun()
                else:
                    if st.button("⏩ Tua nhanh", use_container_width=True, 
                                disabled=st.session_state.get('debate_finished', False)):
                        debate_state.is_fast_mode = True
                        
                        target_rounds = config.rounds
                        current_rounds = len(st.session_state.dialog_a)
                        
                        # Tạo một placeholder để hiển thị tiến trình
                        progress_text = st.empty()
                        progress_bar = st.empty()
                        
                        try:
                            # Chạy vòng lặp một lần duy nhất và hiển thị tiến trình
                            for i in range(current_rounds, target_rounds):
                                progress = (i - current_rounds + 1) / (target_rounds - current_rounds)
                                progress_text.text(f"Đang tua nhanh... ({i+1}/{target_rounds})")
                                progress_bar.progress(progress)
                                
                                # Thêm một lượt tranh luận
                                add_ai_turn_auto()
                            
                            # Hoàn thành
                            progress_text.text("✅ Đã tua nhanh xong!")
                            time.sleep(0.5)
                            
                        except Exception as e:
                            st.error(f"Lỗi khi tua nhanh: {str(e)}")
                        
                        finally:
                            # Xóa các placeholder
                            progress_text.empty()
                            progress_bar.empty()
                        
                        debate_state.is_fast_mode = False
                        debate_state.current_display_index = len(st.session_state.dialog_a)
                        st.session_state.debate_finished = True
                        st.session_state.debate_running = False
                        st.rerun()
            else:
                st.button("⏩ Tua nhanh", disabled=True, use_container_width=True,
                         help="Tính năng chỉ khả dụng cho chế độ AI vs AI")
        
        with col3:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.session_state.debate_state.current_display_index = 0
                st.rerun()

def render_user_input():
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())
    
    if not hasattr(debate_state, 'waiting_for_user'):
        debate_state.waiting_for_user = False
    
    if not debate_state.waiting_for_user:
        return
    
    st.markdown("---")
    
    if debate_state.current_turn == "USER_B":
        st.subheader(f"💬 Lượt của bạn ({config.persona_b})")
        
        if st.session_state.dialog_a:
            last_a_msg = st.session_state.dialog_a[-1]
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color: #1e2d42; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin-bottom: 15px; 
                    border-left: 4px solid #58a6ff;
                    max-height: 250px;
                    overflow-y: auto;
                    word-wrap: break-word;
                ">
                    <strong>{config.persona_a} vừa nói:</strong><br>
                    {last_a_msg}
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.suggestion_b:
            with st.expander("💡 Gợi ý từ AI (nhấn để xem)"):
                st.info(st.session_state.suggestion_b)
                col_copy, col_clear = st.columns(2)
                with col_copy:
                    if st.button("📋 Sử dụng gợi ý này", key="use_suggestion_b", use_container_width=True):
                        st.session_state.user_input_b = st.session_state.suggestion_b
                        st.rerun()
                with col_clear:
                    if st.button("🗑️ Xóa gợi ý", key="clear_suggestion_b", use_container_width=True):
                        st.session_state.suggestion_b = ""
                        st.rerun()

        user_input = st.text_area(
            "Phản biện của bạn:",
            value=st.session_state.get('user_input_b', ''),
            key="user_input_b_area",
            placeholder=f"Nhập phản biện với tư cách {config.persona_b}...",
            height=120
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🚀 Gửi", key="send_user_b", use_container_width=True):
                if user_input.strip():
                    st.session_state.user_input_b = user_input
                    process_user_reply("USER_B", user_input.strip())
                else:
                    st.warning("Vui lòng nhập nội dung phản biện!")
        
        with col2:
            if st.button("💡 Gợi ý", key="suggest_b", use_container_width=True):
                if st.session_state.dialog_a:
                    context = st.session_state.dialog_a[-1]
                    with st.spinner("AI đang suy nghĩ..."):
                        suggestion = get_writing_suggestion("USER_B", context)
                        st.session_state.suggestion_b = suggestion
                        st.rerun()
                else:
                    st.warning("Chưa có ngữ cảnh để gợi ý.")
        
        with col3:
            if st.button("🗑️ Xóa", key="clear_user_b", type="secondary", use_container_width=True):
                st.session_state.user_input_b = ""
                st.session_state.suggestion_b = ""
                st.rerun()
    
    elif debate_state.current_turn == "USER_C":
        st.subheader(f"💬 Lượt của bạn ({config.persona_c})")
        
        if st.session_state.dialog_a and st.session_state.dialog_b:
            last_a_msg = st.session_state.dialog_a[-1]
            last_b_msg = st.session_state.dialog_b[-1]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="background-color: #1f362d; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #2a4a3d;">
                    <strong>{config.persona_a}:</strong><br>
                    {last_a_msg}
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div style="background-color: #3b2225; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #4d2c30;">
                    <strong>{config.persona_b}:</strong><br>
                    {last_b_msg}
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.suggestion_c:
            with st.expander("💡 Gợi ý từ AI (nhấn để xem)"):
                st.info(st.session_state.suggestion_c)
                col_copy, col_clear = st.columns(2)
                with col_copy:
                    if st.button("📋 Sử dụng gợi ý này", key="use_suggestion_c", use_container_width=True):
                        st.session_state.user_input_c = st.session_state.suggestion_c
                        st.rerun()
                with col_clear:
                    if st.button("🗑️ Xóa gợi ý", key="clear_suggestion_c", use_container_width=True):
                        st.session_state.suggestion_c = ""
                        st.rerun()

        user_input = st.text_area(
            "Quan điểm của bạn:",
            value=st.session_state.get('user_input_c', ''),
            key="user_input_c_area",
            placeholder=f"Nhập quan điểm với tư cách {config.persona_c}...",
            height=120
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🚀 Gửi", key="send_user_c", use_container_width=True):
                if user_input.strip():
                    st.session_state.user_input_c = user_input
                    process_user_reply("USER_C", user_input.strip())
                else:
                    st.warning("Vui lòng nhập nội dung!")
        
        with col2:
            if st.button("💡 Gợi ý", key="suggest_c", use_container_width=True):
                if st.session_state.dialog_a and st.session_state.dialog_b:
                    context = f"Bên A: {st.session_state.dialog_a[-1]}\nBên B: {st.session_state.dialog_b[-1]}"
                    with st.spinner("AI đang suy nghĩ..."):
                        suggestion = get_writing_suggestion("USER_C", context)
                        st.session_state.suggestion_c = suggestion
                        st.rerun()
                else:
                    st.warning("Chưa có ngữ cảnh để gợi ý.")
        
        with col3:
            if st.button("🗑️ Xóa", key="clear_user_c", type="secondary", use_container_width=True):
                st.session_state.user_input_c = ""
                st.session_state.suggestion_c = ""
                st.rerun()

def render_chat_messages():
    config = st.session_state.config
    debate_state = st.session_state.get('debate_state', DebateState())

    dialog_a = st.session_state.dialog_a
    dialog_b = st.session_state.dialog_b
    dialog_c = st.session_state.dialog_c

    if config.mode == "Tranh luận 1v1 với AI":
        display_rounds = len(dialog_a)
    else:
        max_rounds = min(len(dialog_a), len(dialog_b))
        if debate_state.is_fast_mode:
            display_rounds = max_rounds
        else:
            display_rounds = min(debate_state.current_display_index, max_rounds)

    for i in range(display_rounds):
        if i < len(dialog_a):
            msg_a = strip_persona_prefix(st.session_state.dialog_a[i])
            if msg_a:
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

        if i < len(dialog_b) and i < len(dialog_a):
            msg_b = strip_persona_prefix(st.session_state.dialog_b[i])
            if msg_b:
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

        if (
            config.mode == "Tham gia 3 bên (Thành viên C)"
            and i < len(dialog_c)
            and i < len(dialog_a)
            and i < len(dialog_b)
        ):
            msg_c = dialog_c[i]
            if msg_c:
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
    
    if config.mode not in ["Tranh luận 1v1 với AI", "Tham gia 3 bên (Thành viên C)"]:
        if st.button("▶️ Tiếp tục", key="continue_bottom"):
            st.session_state._trigger_continue = True
            st.rerun()

def run_courtroom_analysis():
    config = st.session_state.config
    
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

def render_login():
    st.title("🔐 Đăng nhập AI Debate Bot")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        ### 🤖 Chào mừng
        
        **Tính năng cho thành viên:**
        - 💾 Lưu lịch sử tranh luận
        - 📊 Xem thống kê cá nhân
        - ⭐ Đánh dấu yêu thích
        - 🔍 Tìm kiếm lịch sử
        
        *Không cần mật khẩu, chỉ cần username!*
        """)
    
    with col2:
        username = st.text_input("Tên người dùng", key="login_username")
        
        if st.button("Đăng nhập", type="primary", use_container_width=True):
            if handle_login(username):
                st.success(f"Chào mừng {username}!")
                time.sleep(1)
                st.session_state.page = "home"
                st.rerun()
            else:
                st.error("Vui lòng nhập tên người dùng!")
        
        st.markdown("---")
        st.write("**Đăng nhập nhanh (demo):**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("👤 User1", use_container_width=True):
                handle_login("User1")
                st.session_state.page = "home"
                st.rerun()
        with col_b:
            if st.button("👤 User2", use_container_width=True):
                handle_login("User2")
                st.session_state.page = "home"
                st.rerun()
        with col_c:
            if st.button("👤 Guest", use_container_width=True):
                handle_login("Guest")
                st.session_state.page = "home"
                st.rerun()

def render_home():
    st.title("🤖 AI Debate Bot – Thiết lập tranh luận")
    
    if st.session_state.current_user:
        col_user, col_logout = st.columns([4, 1])
        with col_user:
            st.markdown(f"👤 **Đang đăng nhập với:** `{st.session_state.current_user}`")
        with col_logout:
            if st.button("🚪 Đăng xuất", key="logout_home"):
                handle_logout()
    
    with st.sidebar:
        st.header("⚙️ Cài đặt Nâng cao")
        
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
        
        model_options = ["openai/gpt-4.1", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"]
        if st.session_state.config.api_client == "openai":
            model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4-vision-preview"]
        
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
        
        if st.button("📚 Kho dữ liệu của tôi", type="secondary", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
        
        if st.button("🔄 Reset Debate", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["config", "page", "current_user", "users_db", "saved_debates"]:
                    del st.session_state[key]
            init_session_state()
            st.session_state.page = "home"
            st.rerun()
    
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
    
    st.subheader("2) Chủ đề tranh luận")
    
    col_info, col_clear = st.columns([4, 1])
    with col_info:
        if st.session_state.config.topic:
            st.markdown(f"""
            <div style="
                background-color: #1e2d42; 
                padding: 12px 15px; 
                border-radius: 8px; 
                border-left: 4px solid #58a6ff;
                margin: 10px 0;
            ">
                <strong>📋 Chủ đề đã chọn:</strong> {st.session_state.config.topic}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Chưa có chủ đề tranh luận. Vui lòng nhập hoặc chọn chủ đề từ các tab bên dưới.")
    
    with col_clear:
        if st.button("🗑️ Xóa", key="clear_topic_btn", help="Xóa chủ đề hiện tại", use_container_width=True):
            st.session_state.config.topic = ""
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["📝 Nhập thủ công", "💡 Gợi ý chủ đề", "🖼️ Phân tích hình ảnh"])
    
    with tab1:
        current_topic = st.text_area(
            "Nhập hoặc chỉnh sửa chủ đề tranh luận:",
            value=st.session_state.config.topic,
            placeholder="Ví dụ: 'Giai cấp thống trị và bị trị trong xã hội hiện đại'",
            height=80,
            key="tab1_topic_input"
        )
        
        if st.button("✅ Áp dụng", key="tab1_apply", use_container_width=True, disabled=not current_topic.strip()):
            if current_topic.strip() != st.session_state.config.topic:
                st.session_state.config.topic = current_topic.strip()
                st.success(f"Đã cập nhật chủ đề!")
                st.rerun()
    
    with tab2:
        st.write("AI sẽ đề xuất chủ đề tranh luận thú vị:")
        
        if st.button("🎲 Tạo chủ đề ngẫu nhiên", use_container_width=True, key="suggest_text_topics"):
            with st.spinner("AI đang tạo chủ đề..."):
                topics = generate_text_topics()
                st.session_state.suggested_topics = topics
                st.session_state.image_analysis_result = None
                st.rerun()
        
        if st.session_state.suggested_topics and not st.session_state.image_analysis_result:
            st.markdown("**Chủ đề đề xuất:**")
            
            for i, topic in enumerate(st.session_state.suggested_topics):
                col_btn, col_txt = st.columns([1, 4])
                with col_btn:
                    if st.button(f"Chọn #{i+1}", key=f"select_text_topic_{i}", use_container_width=True):
                        st.session_state.config.topic = topic
                        st.success(f"Đã chọn: {topic}")
                        st.session_state.suggested_topics = None
                        st.rerun()
                with col_txt:
                    st.markdown(f"`{topic}`")
            
            if st.button("🗑️ Xóa danh sách gợi ý", key="clear_text_suggestions", use_container_width=True):
                st.session_state.suggested_topics = None
                st.rerun()
    
    with tab3:
        st.write("Tải lên hình ảnh để AI phân tích và đề xuất chủ đề:")
        
        uploaded_file = st.file_uploader(
            "Chọn một hình ảnh...", 
            type=["png", "jpg", "jpeg", "webp"],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption="Ảnh đã tải lên", width=200)
                
                with col2:
                    st.write("**Thông tin ảnh:**")
                    st.write(f"- Định dạng: {uploaded_file.type}")
                    st.write(f"- Kích thước: {image.size[0]}x{image.size[1]} pixels")
                
                if st.button("🔍 AI Phân tích ảnh", type="primary", use_container_width=True, key="analyze_image"):
                    with st.spinner("🤖 AI đang phân tích hình ảnh..."):
                        suggested_topics = analyze_image_for_topic(image)
                        
                        if suggested_topics:
                            st.session_state.suggested_topics = suggested_topics
                            st.session_state.image_analysis_result = "Đã phân tích xong"
                            st.rerun()
                
            except Exception as e:
                st.error(f"Lỗi khi mở ảnh: {str(e)}")
        
        if st.session_state.suggested_topics and st.session_state.image_analysis_result:
            st.markdown("**Chủ đề đề xuất từ hình ảnh:**")
            
            for i, topic in enumerate(st.session_state.suggested_topics):
                col_btn, col_txt = st.columns([1, 4])
                with col_btn:
                    # SỬA Ở ĐÂY: "Chọn ảnh #1" thành "Chọn #1"
                    if st.button(f"Chọn #{i+1}", key=f"select_image_topic_{i}", use_container_width=True):
                        st.session_state.config.topic = topic
                        st.success(f"Đã chọn: {topic}")
                        st.session_state.suggested_topics = None
                        st.session_state.image_analysis_result = None
                        st.rerun()
                with col_txt:
                    st.markdown(f"`{topic}`")
            
            if st.button("🗑️ Xóa danh sách", key="clear_image_suggestions", use_container_width=True):
                st.session_state.suggested_topics = None
                st.session_state.image_analysis_result = None
                st.rerun()
    
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
    
    st.markdown("---")
    col_start, _ = st.columns([1, 3])
    with col_start:
        if st.button("▶️ Bắt đầu tranh luận", type="primary", use_container_width=True):
            if not st.session_state.config.topic.strip():
                st.error("Vui lòng nhập chủ đề tranh luận!")
                return
            
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
            st.session_state.suggestion_b = ""
            st.session_state.suggestion_c = ""
            st.session_state.page = "debate"
            st.rerun()

def render_history():
    st.title("📚 Kho tranh luận của bạn")
    
    if not st.session_state.current_user:
        st.warning("Vui lòng đăng nhập để xem lịch sử!")
        if st.button("🔐 Đăng nhập ngay"):
            st.session_state.page = "login"
            st.rerun()
        return
    
    user_debates = st.session_state.users_db.get(st.session_state.current_user, {}).get("debates", [])
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"👤 {st.session_state.current_user}")
    with col2:
        st.metric("Tổng cuộc tranh luận", len(user_debates))
    with col3:
        if st.button("🔙 Về trang chủ", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    
    if not user_debates:
        st.info("Bạn chưa có cuộc tranh luận nào được lưu.")
        if st.button("🎭 Bắt đầu tranh luận mới"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    for i, debate in enumerate(reversed(user_debates[-10:])):
        with st.expander(f"🗣️ {debate['topic'][:60]}...", expanded=i==0):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**Chủ đề:** {debate['topic']}")
                st.markdown(f"**Chế độ:** {debate['mode']}")
                st.markdown(f"**Thời gian:** {datetime.fromisoformat(debate['created_at']).strftime('%d/%m/%Y %H:%M')}")
                st.markdown(f"**Số lượt:** {len(debate['dialog_a'])} lượt A, {len(debate['dialog_b'])} lượt B")
            
            with col2:
                if st.button("📖 Xem lại", key=f"view_{debate['id']}", use_container_width=True):
                    load_debate(debate)
            
            with col3:
                if st.button("🗑️ Xóa", key=f"delete_{debate['id']}", type="secondary", use_container_width=True):
                    st.session_state.users_db[st.session_state.current_user]["debates"] = [
                        d for d in user_debates if d['id'] != debate['id']
                    ]
                    st.success("Đã xóa!")
                    st.rerun()
    
    st.markdown("---")
    if st.button("📥 Xuất tất cả dữ liệu (JSON)", use_container_width=True):
        data = {
            "user": st.session_state.current_user,
            "debates": user_debates,
            "exported_at": datetime.now().isoformat()
        }
        st.download_button(
            "Tải file JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"debate_history_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

def render_debate():
    st.title("🔥 Cuộc tranh luận")
    
    config = st.session_state.config
    
    with st.sidebar:
        st.header("📊 Thông tin")
        
        html_content = f"""
        <div style="
            background-color: #1e2d42; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 4px solid #58a6ff;
            margin-bottom: 15px;
        ">
            <p style="margin: 8px 0;"><strong>Chế độ:</strong> {config.mode}</p>
            <p style="margin: 8px 0;"><strong>Chủ đề:</strong> {st.session_state.topic_used}</p>
            <p style="margin: 8px 0;"><strong>Phong cách:</strong> {st.session_state.final_style}</p>
        """
        
        if st.session_state.current_user:
            html_content += f'<p style="margin: 8px 0;"><strong>Người dùng:</strong> {st.session_state.current_user}</p>'
        
        html_content += "</div>"
        
        st.markdown(html_content, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔙 Trang chủ", use_container_width=True, key="back_home"):
                st.session_state.page = "home"
                st.rerun()
        with col2:
            if st.button("📚 Kho dữ liệu", use_container_width=True, key="to_history"):
                st.session_state.page = "history"
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.current_user:
            if st.button("🚪 Đăng xuất", use_container_width=True):
                handle_logout()
    
    st.header(f"Chủ đề: {st.session_state.topic_used}")
    
    render_progress_bar()
    
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
    
    if config.mode == "Chế độ RPG (Game Tranh luận)":
        render_hp_display()
    
    render_control_buttons()
    
    render_user_input()
    
    render_chat_messages()
    
    is_victory, victory_msg = check_victory()
    if is_victory:
        st.session_state.debate_finished = True
        st.session_state.debate_running = False
        
        st.markdown("---")
        
        if "CHIẾN THẮNG" in victory_msg or "HÒA" in victory_msg:
            st.success(victory_msg)
        else:
            st.info(victory_msg)
        
        if config.mode == "Chế độ RPG (Game Tranh luận)" and "CHIẾN THẮNG" not in victory_msg and "HÒA" not in victory_msg:
            advantage = get_advantage_status()
            if advantage:
                st.info(advantage)
    
    if st.session_state.debate_finished:
        st.markdown("---")
        
        if config.mode != "Tham gia 3 bên (Thành viên C)":
            if st.session_state.current_user:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("💾 Lưu vào kho", use_container_width=True, type="secondary"):
                        if save_current_debate():
                            st.success("Đã lưu cuộc tranh luận!")
                            st.rerun()
                        else:
                            st.error("Lỗi khi lưu!")
                
                with col2:
                    if st.button("⚖️ Phân tích AI", use_container_width=True, type="secondary", key="ai_analysis"):
                        run_courtroom_analysis()
                        st.rerun()
                
                with col3:
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
                
                with col4:
                    if st.button("🔄 Tranh luận mới", type="primary", use_container_width=True, key="new_debate"):
                        st.session_state.page = "home"
                        st.rerun()
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("⚖️ Phân tích AI", use_container_width=True, type="secondary", key="ai_analysis"):
                        run_courtroom_analysis()
                        st.rerun()
                
                with col2:
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
        
        if st.session_state.courtroom_analysis:
            st.markdown("---")
            st.header("⚖️ Phân tích Phiên Tòa AI")
            
            with st.container():
                st.markdown(st.session_state.courtroom_analysis)

CHAT_STYLE = """
<style>
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    color: #c9d1d9;
    font-weight: 600;
}

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

[data-testid="stSidebar"] {
    background-color: #202329!important;
    color: #c9d1d9 !important;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #c9d1d9 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: #c9d1d9 !important;
}

[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stButton {
    color: #c9d1d9 !important;
}

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
    background: #c9d1d9;
}

.stSelectbox, .stSlider, .stTextInput, .stTextArea {
    color: #c9d1d9 !important;
}

.stSuccess {
    background: linear-gradient(135deg, #0e4429 0%, #1f362d 100%);
    border-left: 5px solid #4cd964;
    color: #c9d1d9 !important;
}

.stWarning {
    background: linear-gradient(135deg, #423200 0%, #332700 100%);
    border-left: 5px solid #ffd60a;
    color: #c9d1d9 !important;
}

.stError {
    background: linear-gradient(135deg, #58161b 0%, #3b2225 100%);
    border-left: 5px solid #ff3b30;
    color: #c9d1d9 !important;
}

.stInfo {
    background: linear-gradient(135deg, #1e2d42 0%, #192f44 100%);
    border-left: 5px solid #58a6ff;
    color: #c9d1d9 !important;
}

hr {
    border: none;
    height: 1px;
    background-color: #30363d;
    margin: 20px 0;
}

[data-testid="stVerticalBlock"] > div {
    padding: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 10px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #1e2d42;
    border-radius: 8px 8px 0 0;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}

.stTabs [aria-selected="true"] {
    background-color: #0d1117 !important;
    border-bottom: 3px solid #58a6ff !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: #0d1117;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1e2d42;
    border-radius: 8px 8px 0 0;
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #2a3f5f;
}

.stTabs [aria-selected="true"] {
    background-color: #0d1117 !important;
    border-bottom: 3px solid #c9d1d9 !important;
}

.selected-topic-box {
    background-color: #1f362d;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #4cd964;
    margin: 15px 0;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s;
}

.selected-topic-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

div[data-baseweb="select"] > div {
    background-color: #1e2d42 !important;
    color: #c9d1d9 !important;
    border-color: #30363d !important;
}

div[data-baseweb="slider"] {
    color: #c9d1d9 !important;
}

[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}

[role="listbox"] {
    background-color: #1e2d42 !important;
    color: #c9d1d9 !important;
}

[role="option"] {
    color: #c9d1d9 !important;
}

[role="option"]:hover {
    background-color: #2a3f5f !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(to right, #58a6ff, #8bb8e8) !important;
}

.stProgress > div > div {
    height: 10px !important;
    border-radius: 10px !important;
}
</style>
"""

def main():
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
    
    if "page" not in st.session_state:
        st.session_state.page = "login"
    
    if st.session_state.page == "login":
        render_login()
    elif st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "debate":
        render_debate()
    elif st.session_state.page == "history":
        render_history()
    
    if st.session_state.get("_trigger_continue", False):
        st.session_state._trigger_continue = False
        add_ai_turn_auto()
        st.session_state.debate_state.current_display_index += 1
        is_victory, _ = check_victory()
        if is_victory:
            st.session_state.debate_finished = True
            st.session_state.debate_running = False
        st.rerun()

if __name__ == "__main__":
    main()
