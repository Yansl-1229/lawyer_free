import os

# API Keys
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-536e06bba87a456791d32486c7bd3393")
DEEPSEEK_API_KEY = "sk-adaac8eb2e6d46b0a3986ef04bc469bf"

# Model Names
LEGAL_MODEL = 'qwen-max-latest'
LAWYER_MODEL = 'qwen-max-latest'
PARALEGAL_MODEL = 'qwen-max-latest'
CONTRACT_ANALYSIS_MODEL = "deepseek-chat"
CASE_ANALYSIS_MODEL = "deepseek-r1"

# Case Types Mapping
CASE_TYPES = {
    "1": {
        "name": "因确认劳动关系发生的争议",
        "file": "lawyer_prompts/lawyer01.txt"
    },
    "2": {
        "name": "因订立、履行、变更、解除和终止劳动合同发生的争议",
        "file": "lawyer_prompts/lawyer02.txt"
    },
    "3": {
        "name": "因除名、辞退、辞职、离职发生的争议",
        "file": "lawyer_prompts/lawyer03.txt"
    },
    "4": {
        "name": "因工作时间、休息休假、社会保险、福利、培训及劳动保护发生的争议",
        "file": "lawyer_prompts/lawyer04.txt"
    },
    "5": {
        "name": "因劳动报酬、工伤医疗费、经济补偿或赔偿金发生的争议",
        "file": "lawyer_prompts/lawyer05.txt"
    }
}