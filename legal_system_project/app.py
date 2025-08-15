import gradio as gr
import os
import json
import pandas as pd
from datetime import datetime
import time
from typing import List
import re
import uuid
import hashlib

from services import (
    ContractAnalyzer,
    CaseAnalysisGenerator,
    ParalegalAssistant,
    LawyerPromptLoader,
)
from utils import (
    classify_contract_type,
    extract_text_from_file,
)
import config
import prompts
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role

# System State
class SystemState:
    def __init__(self):
        self.current_mode = "selection"  # selection, legal_knowledge, case_strategy, contract_review
        self.legal_conversation_history = []
        self.case_conversation_history = []
        self.contract_review_history = []
        self.user_input_round = 0
        self.case_type_selected = None
        self.current_lawyer_prompt = None
        # 新增：用于处理continue_prompt的状态
        self.expecting_continue_response = False
        self.loaded_history_for_prompt = None
        self.loaded_state_for_prompt = None

# Global dictionary to store user states by UUID
user_states = {}

def get_user_id_from_request(request: gr.Request):
    """Generate a stable user ID using request headers (browser + IP)."""
    try:
        # print(request)
        user_agent = request.headers.get('user-agent', '') if hasattr(request, 'headers') else ''
        forwarded_for = request.headers.get('x-forwarded-for', '') if hasattr(request, 'headers') else ''
        real_ip = request.headers.get('x-real-ip', '') if hasattr(request, 'headers') else ''
        # remote_addr = getattr(request, 'client', {}).get('host', '') if hasattr(request, 'client') else ''
        identifier = f"{user_agent}:{forwarded_for}:{real_ip}"
        # print(identifier)
        user_hash = hashlib.md5(identifier.encode()).hexdigest()[:16]
        return f"user_{user_hash}"
    except Exception:
        print("Error in get_user_id_from_request")
        return f"user_{str(uuid.uuid4())[:12]}"

def get_or_create_user_state(request: gr.Request, user_id=None):
    """Get or create user state for a given user ID."""
    if user_id is None:
        user_id = get_user_id_from_request(request)
    if user_id not in user_states:
        user_states[user_id] = SystemState()
    return user_id, user_states[user_id]

# For backward compatibility, create a default system state
system_state = SystemState()

# Instantiate services
contract_analyzer = ContractAnalyzer()
case_analyzer = CaseAnalysisGenerator()
paralegal = ParalegalAssistant()
prompt_loader = LawyerPromptLoader()

# Helper functions
def get_formatted_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_conversation(history, mode):
    if not os.path.exists("conversation_datasets"):
        os.makedirs("conversation_datasets")
    
    filename = f"conversation_datasets/{mode}_conversation_{get_formatted_time()}.json"
    
    formatted_conversation = {
        "conversation_id": f"{mode}_{get_formatted_time()}",
        "model_a": config.LAWYER_MODEL if mode == 'case' else config.LEGAL_MODEL,
        "conversations": []
    }
    
    for user_msg, bot_msg in history:
        formatted_conversation["conversations"].append({"from": "human", "value": user_msg})
        formatted_conversation["conversations"].append({"from": "gpt", "value": bot_msg})
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump([formatted_conversation], f, ensure_ascii=False, indent=4)
    
    return filename

def save_analysis_result(result_text):
    if not os.path.exists("case_analysis_results"):
        os.makedirs("case_analysis_results")
    
    filename = f"case_analysis_results/case_analysis_{get_formatted_time()}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result_text)
    return filename

def get_user_chat_file_path(user_id):
    """Get the file path for storing user's chat history."""
    if not os.path.exists("user_histories"):
        os.makedirs("user_histories")
    return f"user_histories/{user_id}_chat_history.json"

def save_user_chat_history(user_id, history, user_state):
    """Save user's chat history and state to a JSON file."""
    try:
        chat_data = {
            "user_id": user_id,
            "last_updated": datetime.now().isoformat(),
            "chat_history": history,
            "user_state": {
                "current_mode": user_state.current_mode,
                "user_input_round": user_state.user_input_round,
                "case_type_selected": user_state.case_type_selected,
                "current_lawyer_prompt": user_state.current_lawyer_prompt,
                "legal_conversation_history": user_state.legal_conversation_history,
                "case_conversation_history": user_state.case_conversation_history,
                "contract_review_history": user_state.contract_review_history
            }
        }
        
        file_path = get_user_chat_file_path(user_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        return file_path
    except Exception as e:
        print(f"保存用户聊天记录失败: {e}")
        return None

def load_user_chat_history(user_id):
    """Load user's chat history and state from JSON file."""
    try:
        file_path = get_user_chat_file_path(user_id)
        if not os.path.exists(file_path):
            return None, None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Load chat history
        history = chat_data.get("chat_history", [])
        
        # Restore user state
        state_data = chat_data.get("user_state", {})
        user_state = SystemState()
        user_state.current_mode = state_data.get("current_mode", "selection")
        user_state.user_input_round = state_data.get("user_input_round", 0)
        user_state.case_type_selected = state_data.get("case_type_selected", None)
        user_state.current_lawyer_prompt = state_data.get("current_lawyer_prompt", None)
        user_state.legal_conversation_history = state_data.get("legal_conversation_history", [])
        user_state.case_conversation_history = state_data.get("case_conversation_history", [])
        user_state.contract_review_history = state_data.get("contract_review_history", [])
        
        return history, user_state
    except Exception as e:
        print(f"加载用户聊天记录失败: {e}")
        return None, None

def get_initial_prompt():
    return '''
🏛️ **欢迎使用AI法律助手！**

我可以为您提供三种专业的法律服务：

📄 **合同审查** - 上传劳动合同文件，为您分析合同合规性：
   • 自动识别合同类型  • 检测潜在风险和漏洞  • 提供专业的法律建议

📚 **法律知识咨询** - 为您解答各类法律知识问题：
   • 合同法、民法、刑法等法律条文解释  
   • 婚姻家庭法、房地产法、知识产权法等专业领域
   • 消费者权益、公司法等实用法律知识

⚖️ **案件策略咨询** - 针对您的具体案件提供专业分析：
   • 劳动争议案件分析与策略制定  • 具体案情的法律风险评估  • 维权路径和赔偿方案建议

**请直接告诉我您需要什么服务，或描述您的问题：**

💡 您可以说："帮我审查一下劳动合同"（合同审查）
💡 或者说："我想了解劳动法的相关法律规定"（法律知识咨询）
💡 或者说："公司突然辞退了我，我该怎么办？"（案件策略咨询）

**或直接上传合同文件开始合同审查！**
'''

def analyze_contracts(files):
    if not files:
        yield "❌ **错误**：请先上传文件。"
        return

    file_paths = [f.name for f in files]

    if len(file_paths) > 1:
        if all(p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')) for p in file_paths):
            mode = "multi_image"
        else:
            yield "❌ **错误**：上传多个文件时，必须全部是图片。"
            return
    elif file_paths[0].lower().endswith(('.pdf', '.docx', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
        mode = "single_document"
    else:
        yield "❌ **错误**：不支持的文件格式。请上传 PDF, DOCX, 或图片文件。"
        return

    try:
        yield "⏳ **正在提取文本...**"
        if mode == "multi_image":
            text = contract_analyzer.extract_texts_from_multiple_images(file_paths)
        else:
            text = extract_text_from_file(file_paths[0])

        if not text.strip():
            yield "❌ **错误**：未能从文件中提取到任何文本。"
            return

        contract_type = classify_contract_type(text)
        yield f"📌 **初步识别的合同类型**：{contract_type}\n\n---\n\n⏳ **正在进行深度分析，请稍候...**"

        full_response = f"📌 **初步识别的合同类型**：{contract_type}\n\n---\n\n"
        for chunk in contract_analyzer.analyze_contract_stream(text, contract_type, file_paths):
            full_response += chunk
            yield full_response

    except Exception as e:
        yield f"❌ **程序发生严重错误**：\n\n`{str(e)}`"

def reset_system(request: gr.Request=None):
    """Reset the current user's system state and clear their chat history."""
    if request is not None:
        user_id = get_user_id_from_request(request)
        # Reset user state
        user_states[user_id] = SystemState()
        # Clear the saved chat history file
        file_path = get_user_chat_file_path(user_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除用户 {user_id} 的历史记录文件: {file_path}")  # ✅ 添加日志
    else:
        print("警告：reset_system调用时未提供request参数，无法删除特定用户的历史记录")
    
    # Fallback to global state reset if user identification fails
    global system_state
    system_state = SystemState()

def chat_legal_knowledge_stream(message, history, request: gr.Request):
    user_id, system_state = get_or_create_user_state(request)
    
    if not system_state.legal_conversation_history or system_state.current_mode != "legal_knowledge":
        system_state.legal_conversation_history = [{'role': Role.SYSTEM, 'content': prompts.LEGAL_CONSULTANT_PROMPT}]
        system_state.current_mode = "legal_knowledge"
    
    system_state.legal_conversation_history.append({'role': Role.USER, 'content': message})
    
    if not history:
        history = []
    
    history.append((message, ""))
    
    try:
        responses = Generation.call(
            model=config.LEGAL_MODEL,
            messages=system_state.legal_conversation_history,
            result_format='message',
            stream=True,
            incremental_output=True
        )
        
        full_response = ""
        
        for response in responses:
            if response.status_code == 200:
                if hasattr(response.output, 'choices') and response.output.choices:
                    delta_content = response.output.choices[0].message.content
                    if delta_content:
                        full_response += delta_content
                        history[-1] = (message, full_response)
                        yield history
                        time.sleep(0.02)
            else:
                error_msg = f"API错误: {response.code} - {response.message}"
                history[-1] = (message, error_msg)
                yield history
                return
        
        system_state.legal_conversation_history.append({'role': Role.ASSISTANT, 'content': full_response})
        
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        history[-1] = (message, error_msg)
        yield history

def detect_case_type(user_input):
    user_input_lower = user_input.lower().strip()
    
    if user_input_lower in ['1', '2', '3', '4', '5']:
        return user_input_lower
    
    keywords_map = {
        '1': ['劳动关系', '确认关系', '是否存在劳动关系', '认定劳动关系'],
        '2': ['劳动合同', '签合同', '合同纠纷', '合同履行', '合同变更', '合同解除', '合同终止'],
        '3': ['辞退', '辞职', '离职', '除名', '被开除', '被炒', '解雇', '被开了', '不用来', '开除'],
        '4': ['加班', '工作时间', '休假', '年假', '社保', '社会保险', '福利', '培训', '劳动保护'],
        '5': ['工资', '薪水', '报酬', '工伤', '医疗费', '补偿', '赔偿金', '经济补偿']
    }
    
    for case_type, keywords in keywords_map.items():
        for keyword in keywords:
            if keyword in user_input:
                return case_type
    
    return None

def get_case_selection_prompt():
    return """您好！我是您的专业劳动法律师，很高兴为您提供法律咨询服务。

在开始为您提供专业法律建议之前，我需要了解您遇到的具体情况。劳动争议案件涉及面较广，不同类型的争议需要采用不同的法律策略和解决方案。

请您详细描述一下您目前面临的问题，我会根据您的描述帮助您确定争议类型，并为您制定最适合的法律解决方案。您可以放心地向我说明情况，我会严格保护您的隐私，并为您提供专业、客观的法律意见。

请开始描述您的情况吧。"""

def detect_conversation_end(lawyer_response):
    if not lawyer_response:
        return True
    
    has_question_mark = '？' in lawyer_response or '?' in lawyer_response
    
    return not has_question_mark

def chat_case_strategy_stream(message, history, request: gr.Request):
    user_id, system_state = get_or_create_user_state(request)
    
    if not system_state.case_conversation_history or system_state.current_mode != "case_strategy":
        system_state.case_conversation_history = [{'role': Role.SYSTEM, 'content': prompts.DEFAULT_LAWYER_SYSTEM_PROMPT}]
        system_state.current_mode = "case_strategy"
        system_state.user_input_round = 0
        system_state.case_type_selected = None
        system_state.current_lawyer_prompt = prompts.DEFAULT_LAWYER_SYSTEM_PROMPT
    
    system_state.user_input_round += 1
    
    if not history:
        history = []
    
    history.append((message, ""))
    
    try:
        if system_state.user_input_round == 1:
            # 首先尝试检测案件类型
            detected_type = detect_case_type(message)
            
            if detected_type:
                # 如果检测到案件类型，直接确认并设置
                system_state.case_type_selected = detected_type
                system_state.current_lawyer_prompt = prompt_loader.load_lawyer_prompt(detected_type)
                system_state.case_conversation_history = [{'role': Role.SYSTEM, 'content': system_state.current_lawyer_prompt}]
                
                case_name = config.CASE_TYPES[detected_type]["name"]
                response_msg = f"您好！我是您的专业劳动法律师，很高兴为您提供法律咨询服务。我了解您的案件类型是：{case_name}\n\n现在让我们开始详细了解您的具体情况。"
                
                system_state.case_conversation_history.append({'role': Role.USER, 'content': message})
                system_state.case_conversation_history.append({'role': Role.ASSISTANT, 'content': response_msg})
                
                full_response = f"【案件类型确认】\n{response_msg}"
                history[-1] = (message, full_response)
                yield history
                return
            else:
                # 如果检测不到案件类型，显示选择提示
                prompt_response = get_case_selection_prompt()
                system_state.case_conversation_history.append({'role': Role.USER, 'content': message})
                system_state.case_conversation_history.append({'role': Role.ASSISTANT, 'content': prompt_response})
                history[-1] = (message, prompt_response)
                yield history
                return
        
        if system_state.case_type_selected is None:
            detected_type = detect_case_type(message)
            
            if detected_type:
                system_state.case_type_selected = detected_type
                system_state.current_lawyer_prompt = prompt_loader.load_lawyer_prompt(detected_type)
                system_state.case_conversation_history = [{'role': Role.SYSTEM, 'content': system_state.current_lawyer_prompt}]
                
                case_name = config.CASE_TYPES[detected_type]["name"]
                response_msg = f"您好！我是您的专业劳动法律师，很高兴为您提供法律咨询服务。我了解您的案件类型是：{case_name}。\n现在让我们开始详细了解您的具体情况。"
                
                system_state.case_conversation_history.append({'role': Role.USER, 'content': message})
                system_state.case_conversation_history.append({'role': Role.ASSISTANT, 'content': response_msg})
                
                full_response = f"【案件类型确认】\n{response_msg}"
                history[-1] = (message, full_response)
                yield history
                return
            else:
                selection_prompt = "抱歉，我暂时无法对你的描述进行案件类型判断，请您重新描述您的情况。" + get_case_selection_prompt()
                system_state.case_conversation_history.append({'role': Role.USER, 'content': message})
                system_state.case_conversation_history.append({'role': Role.ASSISTANT, 'content': selection_prompt})
                full_response = f"【请重新选择案件类型】\n{selection_prompt}"
                history[-1] = (message, full_response)
                yield history
                return
        
        effective_round = system_state.user_input_round - (2 if system_state.case_type_selected else 1)
        
        if effective_round <= 4:
            polished_message = paralegal.polish_user_input(message)
            system_state.case_conversation_history.append({'role': Role.USER, 'content': polished_message})
            final_message = polished_message
        else:
            system_state.case_conversation_history.append({'role': Role.USER, 'content': message})
            final_message = message
        
        responses = Generation.call(
            model=config.LAWYER_MODEL,
            messages=system_state.case_conversation_history,
            result_format='message',
            stream=True,
            incremental_output=True
        )
        
        full_response = ""
        
        for response in responses:
            if response.status_code == 200:
                if hasattr(response.output, 'choices') and response.output.choices:
                    delta_content = response.output.choices[0].message.content
                    if delta_content:
                        full_response += delta_content
                        display_response = f"【律师回复】\n{full_response}"
                        history[-1] = (message, display_response)
                        yield history
                        time.sleep(0.01)
            else:
                error_msg = f"API错误: {response.code} - {response.message}"
                history[-1] = (message, error_msg)
                yield history
                return
        
        system_state.case_conversation_history.append({'role': Role.ASSISTANT, 'content': full_response})
        
        should_generate_analysis = detect_conversation_end(full_response)

        if should_generate_analysis and system_state.user_input_round > 3:
            try:
                conversation_content = ""
                for msg in system_state.case_conversation_history:
                    if msg['role'] in ['user', 'assistant']:
                        role_name = "用户" if msg['role'] == 'user' else "律师"
                        content = msg['content']
                        content = re.sub(r'^【.*?】\s*', '', content)
                        conversation_content += f"{role_name}: {content}\n\n"
                
                if conversation_content.strip():
                    analysis_start = "\n\n🎯 **正在为您生成专业案例分析报告...**\n\n"
                    history[-1] = (message, full_response + analysis_start)
                    yield history
                    
                    case_analysis = case_analyzer.generate_case_analysis(conversation_content.strip())
                    
                    if case_analysis and case_analysis != "生成分析失败":
                        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        
                        analysis_data = {
                            "timestamp": timestamp,
                            "case_type": config.CASE_TYPES.get(system_state.case_type_selected, {}).get("name", "未知类型"),
                            "conversation": conversation_content,
                            "case_analysis": case_analysis,
                            "analysis_sections": {}
                        }
                        
                        sections = re.split(r'【(案情分析|当前应对方案|维权与赔偿方案)】', case_analysis)
                        for i in range(1, len(sections), 2):
                            if i + 1 < len(sections):
                                section_title = sections[i]
                                section_content = sections[i + 1].strip()
                                analysis_data["analysis_sections"][section_title] = section_content
                        
                        os.makedirs('./case_analysis_results', exist_ok=True)
                        analysis_filename = f"./case_analysis_results/case_analysis_{timestamp}.json"
                        with open(analysis_filename, 'w', encoding='utf-8') as f:
                            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                        
                        final_content = f"{full_response}\n\n🎯 **专业案例分析报告**\n\n"
                        
                        paragraphs = case_analysis.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                final_content += para + "\n\n"
                                history[-1] = (message, final_content)
                                yield history
                                time.sleep(0.1)

                        summary = f"{final_content}\n✅ 案例分析已完成并保存！\n📁 文件路径：{analysis_filename}\n\n💡 您可以：\n1. 继续提问补充信息\n2. 开始新的咨询\n3. 查看保存的分析报告"
                        saved_conversation_file = save_case_conversation_history()
                        if saved_conversation_file:
                            summary += f"\n📁 对话历史已保存：{saved_conversation_file}"
                        history[-1] = (message, summary)
                        yield history
                        
                    else:
                        error_msg = f"{full_response}\n\n❌ 案例分析生成失败，请稍后重试。"
                        history[-1] = (message, error_msg)
                        yield history
                        
            except Exception as e:
                error_msg = f"{full_response}\n\n❌ 案例分析生成出错：{str(e)}"
                history[-1] = (message, error_msg)
                yield history
        
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        history[-1] = (message, error_msg)
        yield history

def save_case_conversation_history(request: gr.Request):
    try:
        user_id, system_state = get_or_create_user_state(request)
        
        if not system_state.case_conversation_history:
            return
        
        sharegpt_data = []
        current_conversation = []
        system_prompt = None
        
        for msg in system_state.case_conversation_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                if current_conversation:
                    sharegpt_data.append({
                        "system_prompt": system_prompt or prompts.DEFAULT_LAWYER_SYSTEM_PROMPT,
                        "conversations": current_conversation
                    })
                    current_conversation = []
                system_prompt = content
            elif role in ['user', 'human']:
                current_conversation.append({"from": "human", "value": content})
            elif role in ['assistant', 'AI']:
                current_conversation.append({"from": "gpt", "value": content})
        
        if current_conversation:
            sharegpt_data.append({
                "system_prompt": system_prompt or prompts.DEFAULT_LAWYER_SYSTEM_PROMPT,
                "conversations": current_conversation
            })
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('./conversation_datasets', exist_ok=True)
        filename = f"./conversation_datasets/case_consultation_{user_id}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
            
        return filename
        
    except Exception as e:
        print(f"保存对话历史失败: {e}")
        return None

def detect_user_intent(user_input):
    user_input_lower = user_input.lower().strip()
    
    contract_keywords = [
        '合同', '劳动合同', '合同审查', '合同分析', '合同检查',
        '合同合规', '合同风险', '合同条款', '合同文件', '三方',
        '审查合同', '分析合同', '检查合同', '合同问题', '协议',
        '竞业限制', '证明'
    ]
    
    legal_knowledge_keywords = [
        '法律知识', '法律咨询', '法律规定', '法律条文', '了解法律',
        '什么是', '如何理解', '法律概念', '法律解释', '普法',
        '法律常识', '法律问题', '法律条款', '法律依据', '法律原理',
        '合同法', '民法', '刑法', '行政法', '婚姻法', '继承法',
        '知识产权法', '消费者权益', '房地产法', '公司法', '法定', '法律法规'
    ]
    
    case_strategy_keywords = [
        '案件咨询', '案件策略', '我的案子', '我遇到', '发生了',
        '公司辞退', '被开除', '被炒', '工资拖欠', '加班费',
        '劳动纠纷', '劳动争议', '工伤', '赔偿', '补偿',
        '仲裁', '起诉', '维权', '我该怎么办', '帮我分析',
        '我的情况', '具体案例', '实际问题', '遇到问题', '我工作', '我的工作' 
    ]
    
    if any(keyword in user_input_lower for keyword in contract_keywords):
        return 'contract_review'
    
    if any(keyword in user_input_lower for keyword in case_strategy_keywords):
        return 'case_strategy'
        
    if any(keyword in user_input_lower for keyword in legal_knowledge_keywords):
        return 'legal_knowledge'
        
    return 'unknown'

def unified_chat_with_clear(message, history, request: gr.Request, files=None):
    """包装函数：执行聊天功能并清空输入框"""
    for response in unified_chat(message, history, request, files):
        yield response, ""

def unified_chat(message, history, request: gr.Request, files=None):
    # Identify user and load existing history if available
    user_id, system_state = get_or_create_user_state(request)
    
    # 处理对continue_prompt的响应
    if system_state.expecting_continue_response:
        system_state.expecting_continue_response = False  # 重置标志
        
        # 定义判断关键词
        continue_keywords = ["是", "yes", "好的", "继续"]
        restart_keywords = ["否", "no", "不", "重新开始"]
        
        message_lower = message.lower().strip()
        
        # 检查是否包含继续关键词
        if any(keyword in message_lower for keyword in continue_keywords):
            # 用户选择继续，加载历史对话和状态
            if system_state.loaded_history_for_prompt:
                history = system_state.loaded_history_for_prompt
                if system_state.loaded_state_for_prompt:
                    user_states[user_id] = system_state.loaded_state_for_prompt
                    system_state = user_states[user_id]
                # 清理临时存储
                system_state.loaded_history_for_prompt = None
                system_state.loaded_state_for_prompt = None
                
                # 添加确认消息
                history.append((message, "✅ **已恢复您的历史对话**\n\n您可以继续之前的对话了。"))
                save_user_chat_history(user_id, history, system_state)
                yield history
                return
        
        # 检查是否包含重新开始关键词
        elif any(keyword in message_lower for keyword in restart_keywords):
            # 用户选择重新开始
            reset_system(request)
            new_history = initialize_user_session(request)
            save_user_chat_history(user_id, new_history, system_state)
            yield new_history
            return
        
        else:
            # 输入不包含指定关键词，重新询问
            system_state.expecting_continue_response = True  # 重新设置标志
            
            continue_prompt = f"""
🔍 **发现您的历史对话记录**

您上次对话的最后一条消息：
> {system_state.loaded_history_for_prompt[-1][0] if system_state.loaded_history_for_prompt and system_state.loaded_history_for_prompt[-1][0] else '系统消息'}

**您想要继续上次的对话吗？**

✅ **继续对话** - 保留之前的对话内容，继续交流
🔄 **重新开始** - 清空历史记录，开始新的对话

**请直接回复"继续"或"重新开始"来选择。**
            """.strip()
            
            # 重新显示提示
            if system_state.loaded_history_for_prompt:
                prompt_history = system_state.loaded_history_for_prompt.copy()
            else:
                prompt_history = []
            prompt_history.append((message, continue_prompt))
            
            yield prompt_history
            return

    loaded = False
    if not history:
        loaded_history, loaded_state = load_user_chat_history(user_id)
        if loaded_history is not None:
            history = loaded_history
            if loaded_state:
                user_states[user_id] = loaded_state
            loaded = True
    
    # Handle file uploads for contract review
    if files and len(files) > 0:
        system_state.current_mode = "contract_review"
        for response in analyze_contracts(files):
            if not history:
                history = []
            if len(history) == 0:
                history.append(("用户上传了合同文件", ""))
            history[-1] = ("用户上传了合同文件", response)
            # Persist after each streaming chunk
            save_user_chat_history(user_id, history, system_state)
            yield history
        system_state.current_mode = "selection"  # Reset after contract review
        save_user_chat_history(user_id, history, system_state)
        return
    
    # If no input message, show initial prompt (or keep loaded history)
    if not message or not message.strip():
        if not history:
            history = []
            history.append(("", get_initial_prompt()))
        # Persist and return current history
        save_user_chat_history(user_id, history, system_state)
        yield history
        return
    
    # Route based on current mode/state
    if system_state.current_mode == "case_strategy":
        for response in chat_case_strategy_stream(message, history, request):
            # Each yielded response is a full history snapshot
            save_user_chat_history(user_id, response, system_state)
            yield response
        if history and "案例分析已完成并保存！" in history[-1][1]:
            system_state.current_mode = "selection"
        save_user_chat_history(user_id, history, system_state)
    elif system_state.current_mode == "legal_knowledge":
        for response in chat_legal_knowledge_stream(message, history):
            save_user_chat_history(user_id, response, system_state)
            yield response
        system_state.current_mode = "selection"
        save_user_chat_history(user_id, history, system_state)
    elif system_state.current_mode == "contract_review":
        if not history:
            history = []
        history.append((message, "📄 **已处于合同审查模式**请上传您的劳动合同文件（支持PDF、DOCX或图片格式），我将为您分析合同合规性。"))
        save_user_chat_history(user_id, history, system_state)
        yield history
        system_state.current_mode = "selection"
    else:
        intent = detect_user_intent(message)
        
        if intent == 'contract_review':
            system_state.current_mode = "contract_review"
            if not history:
                history = []
            history.append((message, "📄 **合同审查模式已激活**\n\n请上传您的劳动合同文件（支持PDF、DOCX或图片格式），我将为您分析合同合规性。\n\n✅ 支持的文件格式：\n• PDF格式合同\n• Word文档(.docx)\n• 图片格式(JPG、PNG等)\n• 支持多张图片同时上传"))
            save_user_chat_history(user_id, history, system_state)
            yield history
        elif intent == 'legal_knowledge':
            system_state.current_mode = "legal_knowledge"
            for response in chat_legal_knowledge_stream(message, history, request):
                save_user_chat_history(user_id, response, system_state)
                yield response
            system_state.current_mode = "selection"
            save_user_chat_history(user_id, history, system_state)
        elif intent == 'case_strategy':
            system_state.current_mode = "case_strategy"
            for response in chat_case_strategy_stream(message, history, request):
                save_user_chat_history(user_id, response, system_state)
                yield response
            if history and "案例分析已完成并保存！" in history[-1][1]:
                system_state.current_mode = "selection"
            save_user_chat_history(user_id, history, system_state)
        else:
            if not history:
                history = []
            
            clarification = f"""很抱歉，我无法从您的描述中准确判断您需要的服务类型。

🤔 **您的输入**："{message}"

为了更好地为您服务，请您明确告诉我您需要：

📄 **合同审查** - 如果您要上传合同文件进行分析
📚 **法律知识咨询** - 如果您想了解法律条文或概念
⚖️ **案件策略咨询** - 如果您遇到具体的法律问题需要解决

**请重新描述您的需求，或直接说"合同审查"、"法律咨询"或"案件咨询"。**"""
            
            history.append((message, clarification))
            save_user_chat_history(user_id, history, system_state)
            yield history

# def initialize_user_session(request: gr.Request):
#     """Initialize user session when page loads."""
#     user_id, system_state = get_or_create_user_state(request)
    
#     # Try to load existing chat history
#     loaded_history, loaded_state = load_user_chat_history(user_id)
    
#     if loaded_history is not None and len(loaded_history) > 0:
#         # Restore previous state
#         if loaded_state:
#             user_states[user_id] = loaded_state
#         return loaded_history
#     else:
#         # Return initial prompt for new users
#         initial_history = [("", get_initial_prompt())]
#         save_user_chat_history(user_id, initial_history, system_state)
#         return initial_history

def initialize_user_session(request: gr.Request=None):
    """Initialize user session when page loads."""
    user_id, system_state = get_or_create_user_state(request)
    
    # Try to load existing chat history
    loaded_history, loaded_state = load_user_chat_history(user_id)
    
    if loaded_history is not None and len(loaded_history) > 0:
        # Restore previous state
        if loaded_state:
            user_states[user_id] = loaded_state
        return loaded_history
    else:
        # Return initial prompt for new users
        initial_history = [("", get_initial_prompt())]
        save_user_chat_history(user_id, initial_history, system_state)
        return initial_history

def initialize_user_session_with_prompt(request: gr.Request):
    """Initialize user session with prompt to continue previous conversation."""
    user_id, system_state = get_or_create_user_state(request)
    
    # Try to load existing chat history
    loaded_history, loaded_state = load_user_chat_history(user_id)
    
    if loaded_history is not None and len(loaded_history) > 0:
        # Create prompt to ask user if they want to continue
        continue_prompt = f"""
🔍 **发现您的历史对话记录**

您上次对话的最后一条消息：
> {loaded_history[-1][0] if loaded_history[-1][0] else '系统消息'}

**您想要继续上次的对话吗？**

✅ **继续对话** - 保留之前的对话内容，继续交流
🔄 **重新开始** - 清空历史记录，开始新的对话

**请直接回复"继续"或"重新开始"来选择，或输入其他内容开始新的对话。**
        """.strip()
        
        # 设置状态，等待用户响应
        system_state.expecting_continue_response = True
        system_state.loaded_history_for_prompt = loaded_history
        system_state.loaded_state_for_prompt = loaded_state
        
        # Add the prompt to history but don't save yet
        prompt_history = loaded_history.copy()
        prompt_history.append(("", continue_prompt))
        
        return prompt_history
    else:
        # No history, proceed with normal initialization
        return initialize_user_session(request)


# Gradio UI
with gr.Blocks(title="AI法律助手 - 专业劳动法咨询平台", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏛️ AI法律助手 - 专业劳动法咨询平台
        
        **欢迎使用AI法律助手！** 我们为您提供专业的劳动法律服务，包括合同审查、法律知识咨询和案件策略分析。
        
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chatbot
            chatbot = gr.Chatbot(
                label="对话记录",
                height=600,
                value=[]
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入您的问题",
                    placeholder="请描述您的问题或上传合同文件...", 
                    lines=1,
                    scale=4
                )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("🔄 重新开始", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📁 合同文件上传")
            file_input = gr.File(
                label="选择合同文件（支持PDF、DOCX、图片等格式）",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]
            )
            upload_btn = gr.Button("开始分析合同", variant="primary")
            
            gr.Markdown(
                """
                ### 💡 使用提示
                
                **📄 合同审查**
                - 上传劳动合同文件
                - 支持PDF、Word、图片格式
                - 获得专业合规分析
                
                **📚 法律知识咨询**
                - 法律条文解释
                - 法律概念说明
                - 实用法律知识
                
                **⚖️ 案件策略咨询**  
                - 描述具体案件情况
                - 获得专业分析建议
                - 制定维权策略
                """
            )
    
    # Load user's previous history when page loads
    demo.load(
        fn=initialize_user_session_with_prompt,
        inputs=None,
        outputs=chatbot
    )
    
    submit_btn.click(
        fn=unified_chat_with_clear,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=unified_chat_with_clear,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    upload_btn.click(
        fn=unified_chat,
        inputs=[msg, chatbot, file_input],
        outputs=chatbot
    )
    
    def _reset_and_init_history(request: gr.Request):
        reset_system(request)
        return initialize_user_session(request)
    
    clear_btn.click(
        fn=_reset_and_init_history,
        inputs=None,
        outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, show_error=True)