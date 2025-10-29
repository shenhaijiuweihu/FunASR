import os
import subprocess
import yt_dlp
import datetime
import time
import threading
from flask import Flask, request, jsonify
from tqdm import tqdm
import torch
import json
import uuid
import requests
from typing import Dict, List, Tuple
import re # 确保 re 已导入

# --- 引入 FunASR 核心库 ---
from funasr import AutoModel

# --- 全局配置 ---
AUDIO_FILENAME = "audio.mp3"

# --- 新增：文本长度限制配置 ---
MAX_TEXT_LENGTH = 100000  # 最大文本长度限制
MAX_VIDEO_DURATION = 7200  # 最大视频时长限制（秒），7200秒 = 2小时

# --- 通义千问API配置 ---
# 请在这里填入您的API Key
QWEN_API_KEY = ""  # 请替换为您的实际API Key

# Qwen API配置
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen3-max"  # 使用qwen3-max-preview模型

# --- 模型配置 (升级版：集成专用标点模型) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 语音识别 (ASR) 模型：负责将语音转为文字
ASR_MODEL_ID = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# 2. 标点恢复 (Punctuation) 模型：您指定的高级标点模型，负责添加精确的标点
PUNC_MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
# 3. 语音活动检测 (VAD) 模型：负责检测音频中的说话片段，对于长音频处理至关重要
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

print("=" * 60)
print(f" FunASR 将在 {DEVICE} 上运行")
print(f" ASR  模型: {ASR_MODEL_ID}")
print(f" VAD  模型: {VAD_MODEL_ID}")
print(f" Punc 模型: {PUNC_MODEL_ID}")
print(f" 文本长度限制: {MAX_TEXT_LENGTH} 字符")
print(f" 视频时长限制: {MAX_VIDEO_DURATION} 秒 ({MAX_VIDEO_DURATION // 3600} 小时)")
print(f" LLM 模型: {QWEN_MODEL}")
print(f" API Key 状态: {'已配置' if QWEN_API_KEY != 'your-api-key-here' else '未配置（请修改代码中的QWEN_API_KEY）'}")
print("=" * 60)

print("正在加载 FunASR Pipeline 模型，首次运行会自动下载，请耐心等待...")
model = AutoModel(
    model=ASR_MODEL_ID,
    vad_model=VAD_MODEL_ID,
    punc_model=PUNC_MODEL_ID,
    device=DEVICE
)
print("模型加载完成，服务已就绪！")


def update_ytdlp():
    """在后台静默更新 yt-dlp 包"""
    try:
        print("[Auto-Update] 正在检查并更新 yt-dlp...")
        result = subprocess.run(
            ["pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            if "Successfully installed" in result.stdout:
                print("[Auto-Update] yt-dlp 已成功更新！")
            elif "Requirement already satisfied" in result.stdout:
                print("[Auto-Update] yt-dlp 已是最新版本。")
        else:
            print(f"[Auto-Update] yt-dlp 更新失败: {result.stderr}")
    except Exception as e:
        print(f"[Auto-Update] 执行更新时发生未知错误: {e}")

def periodic_updater():
    """周期性地执行更新任务"""
    update_ytdlp()
    while True:
        time.sleep(24 * 60 * 60)
        update_ytdlp()
# --- 自动更新模块结束 ---


# --- Flask 应用初始化 (统一使用一个应用) ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# --- 新增：获取视频信息函数 ---
def get_video_info(url: str) -> Dict:
    """
    获取视频的元信息，包括时长、标题等
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'skip_download': True,
    }
    if os.path.exists('cookies.txt'):
        ydl_opts['cookiefile'] = 'cookies.txt'
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_info = {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'duration_string': str(datetime.timedelta(seconds=int(info.get('duration', 0)))),
                'uploader': info.get('uploader', 'Unknown'),
                'upload_date': info.get('upload_date', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'description': info.get('description', '')[:500],
                'webpage_url': info.get('webpage_url', url)
            }
            return video_info
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return None


# --- 通义千问API调用函数 ---
def call_qwen_api(prompt: str, system_prompt: str = None) -> str:
    """
    调用通义千问API
    """
    if QWEN_API_KEY == "your-api-key-here":
        return "错误：请先配置QWEN_API_KEY"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 60000,
        "top_p": 0.9
    }
    try:
        print(f"正在调用通义千问API ({QWEN_MODEL})...")
        response = requests.post(
            QWEN_API_URL,
            headers=headers,
            json=data,
            timeout=3600
        )
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print("API调用成功")
            return content
        else:
            error_msg = f"API调用失败: {response.status_code} - {response.text}"
            print(error_msg)
            return f"错误：{error_msg}"
    except requests.exceptions.Timeout:
        return "错误：API调用超时，请稍后重试"
    except Exception as e:
        print(f"通义千问API调用失败: {e}")
        return f"错误：API调用失败 - {str(e)}"


# --- 视频概述生成函数 (带重试和最终保障的终极稳定版) ---
def generate_video_summary(transcript_text: str) -> Dict:
    """
    使用带重试机制和最终保障的策略，确保总有输出且格式正确。
    1. 首次尝试用Qwen进行高质量分析。
    2. 如果失败，再用Qwen重试一次。
    3. 如果两次都失败，则由代码强制生成保底结果。
    """
    # 系统提示词 (保持不变)
    system_prompt = """你是一个专业的视频内容分析助手。你的任务是：
1. 分析带时间戳的视频字幕文本
2. 生成结构化的视频概述
3. 提取关键时间段和要点
4. 保持时间戳的准确性
输出格式要求：
- 使用JSON格式
- 包含整体概述、时间段划分、关键时刻等部分
- 时间戳格式保持为 [HH:MM:SS]
- 确保JSON格式正确，可以被解析"""

    # 用户提示词 (保持不变)
    user_prompt = f"""请对以下带时间戳的视频字幕进行深度分析和结构化处理。

    字幕内容（部分）：
    {transcript_text}
    **绝对重要的规则 (Absolute & Mandatory Rules)**
    1.  **严格锚定时间**：你必须严格依据字幕中出现的原始 `[HH:MM:SS]` 时间戳来确定每个内容片段的起止时间。
    2.  **精确的起止点**：
        -   每个 `time_segments` 的 `start_time` **必须**是该段内容在字幕中**第一个句子**的原始时间戳。
        -   每个 `time_segments` 的 `end_time` **必须**是该段内容在字幕中**最后一个句子**的原始时间戳。
    3.  **禁止虚构与复用**：
        -   **禁止**虚构任何时间戳。
        -   **禁止**为不同的 `time_segments` 或 `key_moments` 复用相同或近似的时间范围。
        -   所有 `time_segments` 的时间区间必须按时间顺序排列，且**不能重叠**。
    4.  **关键时刻的时间范围**：每个 `key_moments` 内部的 `start_time` 和 `end_time` 也必须精确对应其在字幕中的实际发生时间范围。
    你的核心任务是：
    1.  **智能分段**：根据内容的逻辑结构和话题的自然转折，将视频划分为数个有意义的时间段（time_segments）。
    2.  **撰写章节速览 (summary)**：为每个时间段撰写一份详细的、按时间顺序的**内容梗概**，忠实复述该片段的主要流程和信息。
    3.  **挖掘关键时刻 (key_moments)**：在每个时间段内，提炼出真正关键的事件或讨论。**每个关键时刻都必须包含一个起始时间(start_time)和一个结束时间(end_time)**，以精确地框定出该事件的发生范围。


    请严格按照以下JSON格式输出，确保可以被正确解析。根对象应为 "analysis_result"。
    {{
        "analysis_result": {{
            "time_segments": [
                {{
                    "start_time": "[00:00:00]",
                    "end_time": "[00:02:00]",
                    "title": "第一部分标题",
                    "summary": "这里应是一段详细的、按时间顺序的文字描述，复述这个时间段内的主要内容和关键步骤。例如：'视频开始，主讲人首先介绍了今天的主题是解决厨房小白的炒菜难题。他建议新手使用不粘锅，并展示了如何利用工具快速切配。随后，他公布了核心の万能咸鲜调味汁配方，依次说明了盐、糖、生抽的比例...'。",
                    "key_moments": [
                        {{
                            "start_time": "[00:01:15]",
                            "end_time": "[00:01:45]",
                            "event": "在此时间段内发生的第一个关键事件，这个事件从1分15秒持续到了1分45秒。",
                            "importance": "high"
                        }},
                        {{
                            "start_time": "[00:01:50]",
                            "end_time": "[00:01:58]",
                            "event": "第二个关键事件，持续了8秒。",
                            "importance": "medium"
                        }}
                    ]
                }}
            ]
        }}
    }}


    **--- "章节速览"撰写核心原则 (MANDATORY PRINCIPLES) ---**
    -   **忠于流程，而非总结结果**：你的目标是让用户通过阅读 `summary` 就能大致了解该片段的**过程**，而不是只知道**结论**。
    -   **保留关键细节**：在 `summary` 中，你必须尽可能多地保留原始字幕中的关键动词、名词、数据和专有名词。
    -   **保持叙事连贯性**：`summary` 应是一段流畅、连贯的文字，能够独立阅读并理解整个片段的来龙去脉。
    -   **格式纯净**：确保最终输出的是纯JSON格式，不要包含任何额外的解释性文字。"""

    # --- 重试逻辑 ---
    max_retries = 2  # 总共尝试2次
    for attempt in range(max_retries):
        print(f"第 {attempt + 1}/{max_retries} 次尝试：使用 Qwen进行详细分析...")
        try:
            response_str = call_qwen_api(user_prompt, system_prompt)

            if response_str.startswith("错误："):
                raise ValueError(f"API调用返回错误: {response_str}")
            
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if not json_match:
                raise ValueError(f"第 {attempt + 1} 次尝试未能返回任何JSON结构。")
            
            data = json.loads(json_match.group())
            
            if data.get("analysis_result", {}).get("time_segments") and isinstance(data["analysis_result"]["time_segments"], list) and len(data["analysis_result"]["time_segments"]) > 0:
                print(f"第 {attempt + 1} 次尝试成功！返回详细结果。")
                return data
            else:
                print(f"第 {attempt + 1} 次尝试返回空结果。")
                if attempt < max_retries - 1:
                    print("...准备重试...")
                else:
                    raise ValueError("所有重试次数均返回空结果。")
        
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                print("...准备重试...")
            else:
                print("所有尝试均告失败！启动最终代码保障...")
                break # 跳出 for 循环

    # --- 最终保障：代码层面强制生成 ---
    print("启动最终保障：由代码直接生成保底摘要。")
    
    summary_prompt = f"请用一段话（大约200字）总结以下视频字幕的核心内容。字幕: {transcript_text}"
    summary_text = call_qwen_api(summary_prompt, "你是一个文本摘要助手。")

    if summary_text.startswith("错误："):
        summary_text = "Qwen模型调用失败，无法生成摘要。"

    fallback_result = {
        "analysis_result": {
            "time_segments": [
                {
                    "start_time": "[00:00:00]",
                    "end_time": "[N/A]",
                    "title": "视频整体内容摘要 (自动生成)",
                    "summary": summary_text,
                    "key_moments": [
                        {
                            "start_time": "[00:00:00]",
                            "end_time": "[N/A]",
                            "event": "由于详细分析失败，这是对视频全部内容的总体概括。",
                            "importance": "medium"
                        }
                    ]
                }
            ]
        }
    }
    print("最终保障成功！返回代码生成的保底结果。")
    return fallback_result



# --- 智能分析函数（增强版） ---
def advanced_video_analysis(transcript_text: str) -> Dict:
    """
    更高级的视频分析，包括情感分析、话题转换等
    """
    system_prompt = """你是一个专业的视频内容深度分析专家。请提供详细、结构化的分析结果。
输出必须是可解析的JSON格式。"""

    prompt = f"""请对以下视频内容进行深度分析：

{transcript_text}

请提供以下分析，以JSON格式输出：
{{
    "content_structure": {{
        "opening": {{
            "time_range": "[开始时间] - [结束时间]",
            "summary": "开头部分概要"
        }},
        "main_body": {{
            "time_range": "[开始时间] - [结束时间]",
            "summary": "主体部分概要",
            "key_points": ["要点1", "要点2", "要点3"]
        }},
        "conclusion": {{
            "time_range": "[开始时间] - [结束时间]",
            "summary": "结尾部分概要"
        }}
    }},
    "topic_transitions": [
        {{
            "timestamp": "[00:00:00]",
            "from_topic": "前一个话题",
            "to_topic": "后一个话题",
            "transition_type": "smooth/abrupt"
        }}
    ],
    "importance_scoring": [
        {{
            "time_range": "[00:00:00] - [00:05:00]",
            "score": 8,
            "reason": "为什么重要"
        }}
    ],
    "viewing_guide": {{
        "must_watch": [
            {{
                "timestamp": "[00:00:00]",
                "duration": "2分钟",
                "reason": "必看理由"
            }}
        ],
        "optional": ["[时间戳] - 可选内容"],
        "skippable": ["[时间戳] - 可跳过内容"]
    }},
    "content_type": "教育/娱乐/新闻/技术/其他",
    "target_audience": "目标观众群体",
    "key_takeaways": ["核心要点1", "核心要点2", "核心要点3"]
}}"""
    try:
        response = call_qwen_api(prompt, system_prompt)

        if response.startswith("错误："):
            return {"error": response}

        try:
            analysis_data = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())
                except:
                    analysis_data = {"analysis": response}
            else:
                analysis_data = {"analysis": response}
        return analysis_data
    except Exception as e:
        return {"error": str(e)}


# --- 文本长度限制函数 ---
def limit_text_length(text, max_length=MAX_TEXT_LENGTH):
    """
    限制文本长度，保持完整的句子结构
    """
    if len(text) <= max_length:
        return text, False
    lines = text.split('\n')
    result_lines, current_length, truncated = [], 0, False
    for line in lines:
        if current_length + len(line) + 1 <= max_length:
            result_lines.append(line); current_length += len(line) + 1
        else:
            truncated = True; break
    result_text = '\n'.join(result_lines)
    if truncated: result_text += "\n\n[注意：文本已截断...]"
    return result_text, truncated


def try_download_subtitle(url):
    subtitle_basename = "subtitle"
    ydl_opts = {'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['zh-Hans', 'zh', 'en'], 'skip_download': True, 'outtmpl': f'{subtitle_basename}.%(ext)s', 'quiet': True}
    if os.path.exists('cookies.txt'): ydl_opts['cookiefile'] = 'cookies.txt'
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        for ext in ['.srt', '.ass', '.vtt']:
            subtitle_file = f"{subtitle_basename}{ext}"
            if os.path.exists(subtitle_file): return subtitle_file
        return None
    except Exception as e:
        print(f"下载字幕时出错: {e}"); return None


def srt_to_timestamped_text(srt_file):
    output = ""
    try:
        with open(srt_file, "r", encoding="utf-8") as f: content = f.read()
        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                time_line, text = lines[1], ' '.join(lines[2:])
                h, m, s_ms = time_line.split(' --> ')[0].split(':')
                s, _ = s_ms.replace(',', '.').split('.')
                output += f"[{int(h):02d}:{int(m):02d}:{int(s):02d}] {text.strip()}\n"
    except Exception as e: print(f"清理字幕文件时出错: {e}")
    finally:
        if os.path.exists(srt_file): os.remove(srt_file)
    return output

def download_and_get_audio_path(url: str) -> str or None:
    """整合并优化的下载函数，包含yt-dlp健壮策略和lux备用方案"""
    if os.path.exists(AUDIO_FILENAME):
        try: os.remove(AUDIO_FILENAME)
        except OSError as e: print(f"清理旧音频文件失败: {e}")

    special_sites = ["iqiyi.com", "v.qq.com", "youku.com"]
    if any(site in url for site in special_sites):
        print(f"\n检测到特定网站，启用 lux 优先策略。")
        video_filename_base = f"lux_dl_{uuid.uuid4().hex[:8]}"
        try:
            subprocess.run(["lux", "-o", ".", "-O", video_filename_base, url], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            downloaded_video_path = next((f"{video_filename_base}{ext}" for ext in ['.mp4', '.flv', '.mkv', '.webm'] if os.path.exists(f"{video_filename_base}{ext}")), None)
            if not downloaded_video_path: raise RuntimeError("lux 下载后未找到视频文件。")
            subprocess.run(["ffmpeg", "-i", downloaded_video_path, "-y", "-vn", "-q:a", "0", "-map", "a", AUDIO_FILENAME], check=True, capture_output=True, text=True)
            os.remove(downloaded_video_path)
            if os.path.exists(AUDIO_FILENAME): return AUDIO_FILENAME
            raise RuntimeError("ffmpeg 提取音频失败。")
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            print(f"lux 优先策略执行失败: {getattr(e, 'stderr', str(e))}")
            return None
    else:
        print(f"\n使用标准下载策略 ({url[:30]}...)。")
        try:
            print("策略1：尝试使用 yt-dlp 下载并封装音频...")
            temp_audio_basename = f"yt_dlp_dl_{uuid.uuid4().hex[:8]}"
            ydl_opts = {'format': 'bestaudio/best', 'outtmpl': f'{temp_audio_basename}.%(ext)s', 'quiet': False, 'progress': True, 'no_warnings': True}
            if os.path.exists('cookies.txt'): ydl_opts['cookiefile'] = 'cookies.txt'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if ydl.download([url]) != 0: raise RuntimeError("yt-dlp download 方法返回了非零退出码")
            time.sleep(1)
            downloaded_file = next((f"{temp_audio_basename}.{ext}" for ext in ['m4a', 'webm', 'ogg', 'mp3'] if os.path.exists(f"{temp_audio_basename}.{ext}")), None)
            if not downloaded_file: raise RuntimeError("yt-dlp 下载后未找到预期的音频文件。")
            subprocess.run(["ffmpeg", "-i", downloaded_file, "-y", "-q:a", "0", AUDIO_FILENAME], check=True, capture_output=True, text=True)
            os.remove(downloaded_file)
            if os.path.exists(AUDIO_FILENAME): return AUDIO_FILENAME
            raise RuntimeError("ffmpeg 转换后未生成 audio.mp3。")
        except Exception as e:
            print(f"标准下载策略（策略1）失败: {e}")
            print("\n策略2：yt-dlp 失败，启用 lux 作为备用方案...")
            video_filename_base = f"lux_bkp_{uuid.uuid4().hex[:8]}"
            try:
                subprocess.run(["lux", "-o", ".", "-O", video_filename_base, url], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                downloaded_video_path = next((f"{video_filename_base}{ext}" for ext in ['.mp4', '.flv', '.mkv', '.webm'] if os.path.exists(f"{video_filename_base}{ext}")), None)
                if not downloaded_video_path: raise RuntimeError("备用 lux 下载后未找到视频文件。")
                subprocess.run(["ffmpeg", "-i", downloaded_video_path, "-y", "-vn", "-q:a", "0", "-map", "a", AUDIO_FILENAME], check=True, capture_output=True, text=True)
                os.remove(downloaded_video_path)
                if os.path.exists(AUDIO_FILENAME): return AUDIO_FILENAME
                return None
            except (subprocess.CalledProcessError, FileNotFoundError, Exception) as lux_e:
                print(f"备用方案 lux 下载失败: {getattr(lux_e, 'stderr', str(lux_e))}")
                return None


def transcribe_with_progress(audio_path: str) -> list:
    transcription_result, transcription_done = [], threading.Event()
    def transcribe_task():
        try:
            result = model.generate(input=audio_path, batch_size_s=300, vad_mode="auto", punc_mode="auto")
            transcription_result.extend(result)
        except Exception as e: print(f"FunASR转录错误: {e}")
        finally: transcription_done.set()
    thread = threading.Thread(target=transcribe_task)
    thread.start()
    with tqdm(total=None, desc="音频转文字中 (FunASR)", bar_format="{l_bar}{bar}| {elapsed}") as pbar:
        while not transcription_done.is_set(): pbar.update(1); time.sleep(0.1)
    thread.join()
    return transcription_result


# --- Web服务接口 (所有端点都在8000端口) ---
@app.route('/check_duration', methods=['POST'])
def check_duration():
    url = request.form.get("url")
    if not url: return jsonify({"error": "请提供url参数"}), 400
    video_info = get_video_info(url)
    if video_info is None: return jsonify({"error": "无法获取视频信息"}), 400
    if video_info['duration'] <= MAX_VIDEO_DURATION: return jsonify({"result": True, "title": video_info['title']})
    return jsonify({"result": False, "message": "视频长度过长"})

@app.route('/extract', methods=['POST'])
def extract():
    url = request.form.get("url")
    if not url: return jsonify({"error": "请提供url参数"}), 400
    
    video_info = get_video_info(url)
    if video_info is None: return jsonify({"error": "无法获取视频信息"}), 400
    if video_info['duration'] > MAX_VIDEO_DURATION: return jsonify({"error": "视频长度过长"}), 400

    print(f"视频标题: {video_info.get('title', 'N/A')}")
    
    subtitle_file = try_download_subtitle(url)
    if subtitle_file:
        transcript_text = srt_to_timestamped_text(subtitle_file)
        transcript_text, was_truncated = limit_text_length(transcript_text)
        return jsonify({"transcript": transcript_text, "source": "original_subtitle", "truncated": was_truncated, "video_info": video_info})

    print("未找到字幕，进入流程2：下载音频并转录...")
    audio_file_path = download_and_get_audio_path(url)
    if not audio_file_path: return jsonify({"error": "无法获取音频文件"}), 400
    
    results = transcribe_with_progress(audio_file_path)
    transcript_text = ""
    # 完整的转录结果拼接逻辑
    if (results and isinstance(results, list) and len(results) > 0 and 'text' in results[0]):
        if 'timestamp' in results[0] and results[0]['timestamp']:
            text_string, timestamps = results[0]['text'], results[0]['timestamp']
            current_sentence, sentence_start_time, ts_idx = "", -1, 0
            for i, char in enumerate(text_string):
                if char not in "。？！，、；：" and ts_idx < len(timestamps):
                    if sentence_start_time == -1: sentence_start_time = timestamps[ts_idx][0]
                    current_sentence += char; ts_idx += 1
                else: current_sentence += char
                if (char in "。？！" or i == len(text_string) - 1) and sentence_start_time != -1:
                    td = datetime.timedelta(seconds=int(sentence_start_time / 1000))
                    transcript_text += f"[{str(td).split('.')[0]}] {current_sentence.strip()}\n"
                    current_sentence, sentence_start_time = "", -1
        elif 'sentence_info' in results[0] and results[0]['sentence_info']:
            for seg in results[0]['sentence_info']:
                td = datetime.timedelta(seconds=int(seg.get('start', 0) / 1000))
                transcript_text += f"[{str(td).split('.')[0]}] {seg.get('text', '').strip()}\n"
        else: transcript_text = results[0]['text']

    if not transcript_text.strip(): transcript_text = "未能从音频中识别出任何文本。"
    if os.path.exists(audio_file_path): os.remove(audio_file_path)
    
    transcript_text, was_truncated = limit_text_length(transcript_text)
    return jsonify({"transcript": transcript_text, "source": "ai_transcription", "truncated": was_truncated, "video_info": video_info})

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """分析视频内容，仅返回AI分析结果，不包含原始字幕 (按照您的要求保持不变)"""
    if QWEN_API_KEY == "your-api-key-here":
        return jsonify({"error": "请先配置QWEN_API_KEY", "message": "请在代码中设置您的通义千问API Key"}), 400
    transcript_text = request.form.get("transcript") or request.json.get("transcript") if request.is_json else None
    analysis_type = request.form.get("type", "summary")
    if not transcript_text: return jsonify({"error": "请提供transcript文本"}), 400
    print(f"\n收到分析任务，文本长度: {len(transcript_text)} 字符, 类型: {analysis_type}")
    if analysis_type == "advanced":
        result = advanced_video_analysis(transcript_text)
    else:
        result = generate_video_summary(transcript_text)
    return jsonify({"analysis": result, "analysis_type": analysis_type, "message": "分析完成"})

@app.route('/extract_and_analyze', methods=['POST'])
def extract_and_analyze():
    """一站式服务：提取字幕并分析"""
    if QWEN_API_KEY == "your-api-key-here": return jsonify({"error": "请先配置QWEN_API_KEY"}), 400
    url = request.form.get("url")
    if not url: return jsonify({"error": "请提供url参数"}), 400
    
    with app.test_request_context(method='POST', data={'url': url}):
        extract_response = extract()
    
    if extract_response.status_code != 200: return extract_response
    
    extract_result = extract_response.get_json()
    transcript_text = extract_result.get("transcript")
    if not transcript_text: return jsonify({"error": "未能获取字幕文本"}), 500
    
    summary = generate_video_summary(transcript_text)
    return jsonify({
        "transcript": transcript_text,
        "transcript_info": {
            "source": extract_result.get("source"),
            "truncated": extract_result.get("truncated", False)
        },
        "video_info": extract_result.get("video_info"),
        "analysis": summary,
        "message": "提取和分析完成"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    api_status = "configured" if QWEN_API_KEY != "your-api-key-here" else "not_configured"
    return jsonify({"status": "healthy", "service": "video-analysis", "llm_model": QWEN_MODEL, "api_key_status": api_status})

# --- 服务器启动 ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("启动 yt-dlp 后台自动更新任务 (每24小时一次)...")
    updater_thread = threading.Thread(target=periodic_updater, daemon=True)
    updater_thread.start()
 
    print(f"\n{'=' * 60}")
    print("视频字幕提取与分析服务 (统一端口版)")
    print(f"{'=' * 60}")
    print(f"当前配置：")
    print(f"- 服务端口: 8000")
    print(f"- 最大文本长度限制: {MAX_TEXT_LENGTH} 字符")
    print(f"- 最大视频时长限制: {MAX_VIDEO_DURATION} 秒 ({MAX_VIDEO_DURATION // 60} 分钟)")
    print(f"- LLM 模型: {QWEN_MODEL}")
    print(f"- API Key 状态: {'已配置' if QWEN_API_KEY != 'your-api-key-here' else '未配置（请修改QWEN_API_KEY）'}")
    if QWEN_API_KEY == "your-api-key-here":
        print("\n" + "⚠️ " * 10 + "\n警告：请先在代码中配置 QWEN_API_KEY 才能使用AI分析功能\n" + "⚠️ " * 10)
    print("\n" + "=" * 60)
    print("API 接口说明：")
    print("=" * 60)
    print("\n所有接口都在端口 8000:")
    print("1. POST /check_duration - 检查视频时长")
    print("2. POST /extract - 提取视频字幕")
    print("3. POST /analyze - 分析字幕文本 (可指定 type='advanced')")
    print("4. POST /extract_and_analyze - 提取并进行标准分析")
    print("5. GET /health - 健康检查")
    print("=" * 60 + "\n")
    print("正在启动服务...")
    app.run(host="0.0.0.0", port=8000, threaded=True)