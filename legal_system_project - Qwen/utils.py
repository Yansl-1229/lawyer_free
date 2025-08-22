import os
import cv2
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
import easyocr

def classify_contract_type(text: str) -> str:
    categories = {
        # 按期限
        "固定期限劳动合同": ["合同期限自", "为期", "终止时间"],
        "无固定期限劳动合同": ["无固定期限", "不设终止日期", "除非双方解除"],
        "以完成工作任务为期限劳动合同": ["任务完成之日", "项目结束即终止", "工作成果"],
        # 按工作时间强度
        "全日制劳动合同": ["标准工时制", "五天八小时", "每日工作时长"],
        "非全日制用工合同": ["非全日制", "小时工", "计时工", "累计工作时间"],
        "零工/临时工合同": ["临时", "一次性任务", "单次派工", "劳务报酬"],
        # 按用工主体
        "劳务派遣合同": ["派遣公司", "派遣单位", "劳务派遣", "第三方用工"],
        "劳务外包合同": ["承揽", "外包", "服务费"],
        # 按身份资历
        "实习/见习协议": ["实习", "见习", "学校三方协议", "实习期间"],
        "试用期协议": ["试用期", "不符合录用条件", "提前通知解除"],
        "顾问/专家聘用合同": ["顾问", "专家", "咨询服务", "顾问费"],
        "兼职协议": ["兼职", "兼任", "工作时间安排"],
        # 专项条款（独立或附属）
        "保密协议（NDA）": ["保密协议", "保密信息", "违约责任", "保密期限"],
        "竞业限制协议": ["竞业限制", "竞业限制期", "经济补偿", "地域范围"],
        "员工持股/股权激励协议": ["股权激励", "员工持股", "认购价格", "归属期限"],
        "劳动争议调解/赔偿协议": ["争议解决", "和解", "一次性赔偿", "调解"],
        # 新兴用工模式
        "平台经济用工协议": ["平台", "派单", "灵活用工", "平台与劳动者"],
        "远程/异地用工合同": ["远程办公", "异地", "通信工具", "工作地点"],
        "弹性用工协议": ["弹性工时", "核心工作时段", "自主排班"],
    }

    for name, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return name
    return "无法识别/可能为非标准劳动合同"

def detect_seal_in_text(text: str) -> bool:
    """通过关键字判断 OCR 文本中是否有'公章'、'盖章'等字样。"""
    seal_keywords = ["公章", "盖章", "（章）", "企业章", "法人章"]
    return any(k in text for k in seal_keywords)

def detect_seal_in_image(file_path: str, min_area_ratio: float = 0.001) -> bool:
    """
    对单张图片做红色圆形印章的更严格检测：
    - 颜色分割后要求红色区域占比超过 min_area_ratio
    - Hough 圆检测验证圆形结构
    """
    if not os.path.exists(file_path):
        return False
    arr = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 空间范围
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    # 计算红色像素比例
    red_ratio = np.sum(mask > 0) / (mask.size + 1e-6)
    if red_ratio < min_area_ratio:
        return False
    # 模糊后找圆
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=35, minRadius=30, maxRadius=150
    )
    return circles is not None

def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    ext = file_path.lower()
    if ext.endswith('.pdf'):
        return _extract_from_pdf(file_path)
    elif ext.endswith('.docx'):
        return _extract_from_docx(file_path)
    elif ext.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        return _extract_from_image(file_path)
    else:
        raise ValueError("不支持的文件格式，请提供PDF、DOCX或图片文件")

def _extract_from_image(file_path: str, reader=None) -> str:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        img_array = np.fromfile(file_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析图像，cv2.imdecode 失败")
        reader = reader or easyocr.Reader(['ch_sim', 'en'])
        result = reader.readtext(image, detail=0)
        return "\n".join(result).strip()
    except Exception as e:
        raise RuntimeError(f"图片OCR识别失败: {str(e)}")

def _extract_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        raise RuntimeError(f"PDF文件读取失败: {str(e)}")
    return text

def _extract_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        raise RuntimeError(f"DOCX文件读取失败: {str(e)}")