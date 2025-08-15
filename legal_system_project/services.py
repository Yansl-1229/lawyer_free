from openai import OpenAI
from typing import Dict, Any, Generator, List, Optional
import utils
import config
import prompts
import dashscope
from dashscope import Generation
import os

class ContractAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    def extract_texts_from_multiple_images(self, image_paths: List[str]) -> str:
        reader = utils.easyocr.Reader(['ch_sim', 'en'])
        all_text = []
        for path in image_paths:
            try:
                text = utils._extract_from_image(path, reader)
                all_text.append(text)
            except Exception as e:
                print(f"\n⚠️ 图片处理失败 [{path}]: {str(e)}")
        return "\n\n".join(all_text)

    def analyze_contract_stream(
        self,
        text: str,
        contract_type: str,
        file_paths: Optional[List[str]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        if not text.strip():
            raise ValueError("合同文本内容为空")

        has_seal_text = utils.detect_seal_in_text(text)
        has_seal_image = False
        if file_paths:
            for p in file_paths:
                if p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')) and utils.detect_seal_in_image(p):
                    has_seal_image = True
                    break

        seal_note = "（检测到公章）" if has_seal_text and has_seal_image else "（未检测到公章）"

        system_prompt = f"""你是一位专业的劳动法律师，负责分析劳动合同的合规性。
该合同初步识别为：{contract_type} {seal_note}

请严格按照以下要求进行分析：
1. 首先判断并确认合同类型（是否准确）
2. 给出总体评价（正规/基本正规/存在明显问题）
3. 根据 {seal_note}说明是否有公章
4. 然后分条列出具体问题或漏洞
5. 对每个问题，简要说明法律依据或建议
6. 重点检查以下关键条款：
   - 合同双方基本信息是否完整
   - 劳动合同期限
   - 工作内容和工作地点
   - 工作时间和休息休假
   - 劳动报酬
   - 社会保险
   - 劳动保护、劳动条件和职业危害防护
   - 劳动合同解除或终止条件
   - 违约责任
   - 竞业限制条款（如有）
   - 保密条款（如有）
"""

        try:
            stream = self.client.chat.completions.create(
                model=config.CONTRACT_ANALYSIS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下劳动合同：\n{text[:15000]}"}
                ],
                temperature=0.3,
                max_tokens=2000,
                stream=True
            )

            collected_content = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    collected_content.append(delta.content)
                    yield delta.content

            return {"metadata": {"complete_response": "".join(collected_content)}}

        except Exception as e:
            raise RuntimeError(f"分析过程中出错: {str(e)}")

class LawyerPromptLoader:
    """律师提示词加载器"""
    
    def __init__(self, base_path="./legal_system_project/"):
        self.base_path = base_path
    
    def load_lawyer_prompt(self, case_type):
        if case_type not in config.CASE_TYPES:
            return prompts.DEFAULT_LAWYER_SYSTEM_PROMPT
        
        file_path = os.path.join(self.base_path, config.CASE_TYPES[case_type]["file"])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return content
        except FileNotFoundError:
            print(f"警告：找不到文件 {file_path}，使用默认提示词")
            return prompts.DEFAULT_LAWYER_SYSTEM_PROMPT
        except Exception as e:
            print(f"加载律师提示词时出错: {e}")
            return prompts.DEFAULT_LAWYER_SYSTEM_PROMPT

class CaseAnalysisGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def extract_conversation_content(self, conversation_data):
        if not conversation_data:
            return ""
        
        if isinstance(conversation_data, list) and len(conversation_data) > 0:
            conversation_data = conversation_data[0]
        
        if not isinstance(conversation_data, dict) or 'conversations' not in conversation_data:
            return ""
        
        conversation_text = ""
        for conv in conversation_data['conversations']:
            role = "用户" if conv['from'] == 'human' else "律师"
            content = conv['value']
            conversation_text += f"{role}: {content}\n\n"
        
        return conversation_text.strip()

    def generate_case_analysis(self, conversation_content: str) -> str:
        prompt = f"""
你是一位资深的劳动法律师。你的任务是根据下方提供的“对话内容”，直接为你的当事人撰写一份专业的法律分析与后续行动建议。

**核心要求 - 语气与称谓（请务必遵守）：**
请全程使用第二人称“您”来称呼当事人，以律师直接对客户提供咨询和建议的口吻进行撰写。在分析内容中，请将“李某”或类似的第三方称谓替换为“您”。

**格式与结构要求（请严格遵守）：**
1.  **纯文本输出**：全文不得包含任何Markdown标记或特殊格式化符号，严禁使用星号（*）、井号（#）、竖线（|）、破折号（-）等。所有内容均以标准段落和文本呈现。
2.  **三段式结构**：报告必须严格按照以下三个部分展开，每个部分作为一个独立的自然段，段落之间用一个空行隔开。
3.  **固定标题**：请在每个部分的开头使用固定的中文全角方括号标题，即【案情分析】、【当前应对方案】和【维权与赔偿方案】。
4.  **字数控制**：总字数请控制在1000字以内。

请按照以下框架填充内容：

**【案情分析】**
（在此处结合法律法条，向您的当事人分析案件的基本事实、争议焦点，并阐述各方的权利义务关系。）

**【当前应对方案】**
（在此处基于现有情况，向您的当事人提出具体的应对策略，分析可能的风险和机会，并提供可操作的建议。）

**【维权与赔偿方案】**
（在此处向您的当事人详细阐述维权路径和步骤，说明可能获得的赔偿项目和金额计算方式，并给出证据收集与保全的建议。）

---
**对话内容：**
{conversation_content}
---
"""
        
        try:
            completion = self.client.chat.completions.create(
                model=config.CASE_ANALYSIS_MODEL,
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"生成案例分析时出错: {e}")
            return "生成分析失败"

class ParalegalAssistant:
    def __init__(self):
        dashscope.api_key = config.DASHSCOPE_API_KEY

    def polish_user_input(self, user_input):
        try:
            messages = [
                {'role': 'system', 'content': prompts.PARALEGAL_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_input}
            ]
            
            response = Generation.call(
                model=config.PARALEGAL_MODEL,
                messages=messages,
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"API错误: {response.code} - {response.message}"
        except Exception as e:
            return f"处理过程中发生异常: {str(e)}"