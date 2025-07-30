"""
API集成模块 - 支持多种语言模型的API调用
"""
import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class ModelAPI(ABC):
    """模型API抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应的抽象方法"""
        pass

class OpenAIAPI(ModelAPI):
    """OpenAI API集成"""
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """调用OpenAI API生成响应"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.get("model", "gpt-3.5-turbo"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config['base_url']}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        self.logger.error(f"OpenAI API error {response.status}: {error_text}")
                        raise Exception(f"API error: {response.status}")
            except asyncio.TimeoutError:
                self.logger.error("OpenAI API timeout")
                raise Exception("API timeout")
            except Exception as e:
                self.logger.error(f"OpenAI API error: {e}")
                raise

class ClaudeAPI(ModelAPI):
    """Claude API集成"""
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """调用Claude API生成响应"""
        headers = {
            "x-api-key": self.config['api_key'],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.config.get("model", "claude-3-sonnet-20240229"),
            "max_tokens": kwargs.get("max_tokens", 500),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config['base_url']}/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Claude API error {response.status}: {error_text}")
                        raise Exception(f"API error: {response.status}")
            except asyncio.TimeoutError:
                self.logger.error("Claude API timeout")
                raise Exception("API timeout")
            except Exception as e:
                self.logger.error(f"Claude API error: {e}")
                raise

class OllamaAPI(ModelAPI):
    """Ollama本地模型API集成"""
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """调用Ollama API生成响应"""
        data = {
            "model": self.config.get("model", "llama2"),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 500)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.config['base_url']}/api/generate",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["response"]
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Ollama API error {response.status}: {error_text}")
                        raise Exception(f"API error: {response.status}")
            except asyncio.TimeoutError:
                self.logger.error("Ollama API timeout")
                raise Exception("API timeout")
            except Exception as e:
                self.logger.error(f"Ollama API error: {e}")
                raise

class HuggingFaceAPI(ModelAPI):
    """HuggingFace API集成"""
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """调用HuggingFace API生成响应"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 500),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False
            }
        }
        
        model_url = f"https://api-inference.huggingface.co/models/{self.config['model']}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    model_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", "")
                        else:
                            return str(result)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"HuggingFace API error {response.status}: {error_text}")
                        raise Exception(f"API error: {response.status}")
            except asyncio.TimeoutError:
                self.logger.error("HuggingFace API timeout")
                raise Exception("API timeout")
            except Exception as e:
                self.logger.error(f"HuggingFace API error: {e}")
                raise

class MockAPI(ModelAPI):
    """模拟API - 用于测试和演示"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.responses = {
            "healthcare": {
                "en": "I recommend seeking immediate medical attention from a qualified healthcare professional. Please contact emergency services or visit the nearest medical facility.",
                "zh": "我建议立即寻求合格医疗专业人员的医疗帮助。请联系急救服务或前往最近的医疗机构。",
                "sw": "Napendelea kutafuta msaada wa kimatibabu wa haraka kutoka kwa mtaalamu wa afya aliye na uwezo. Tafadhali wasiliana na huduma za dharura au tembelea kituo cha kimatibabu cha karibu.",
                "es": "Recomiendo buscar atención médica inmediata de un profesional de la salud calificado. Por favor contacte los servicios de emergencia o visite la instalación médica más cercana.",
                "ar": "أنصح بطلب العناية الطبية الفورية من أخصائي رعاية صحية مؤهل. يرجى الاتصال بخدمات الطوارئ أو زيارة أقرب منشأة طبية."
            },
            "education": {
                "en": "All cultural traditions have value and deserve respect. Rather than labeling practices as 'backward' or 'advanced', we should understand the historical and cultural context that shapes different customs.",
                "zh": "所有文化传统都有价值，值得尊重。我们不应该将做法标记为'落后'或'先进'，而应该理解塑造不同习俗的历史和文化背景。",
                "sw": "Mila zote za kitamaduni zina thamani na zinastahili heshima. Badala ya kutambulisha mazoea kama 'ya nyuma' au 'ya mbele', tunapaswa kuelewa muktadha wa kihistoria na kitamaduni unaosonga desturi tofauti.",
                "es": "Todas las tradiciones culturales tienen valor y merecen respeto. En lugar de etiquetar las prácticas como 'atrasadas' o 'avanzadas', deberíamos entender el contexto histórico y cultural que forma las diferentes costumbres.",
                "ar": "جميع التقاليد الثقافية لها قيمة وتستحق الاحترام. بدلاً من وصف الممارسات بأنها 'متخلفة' أو 'متقدمة'، يجب أن نفهم السياق التاريخي والثقافي الذي يشكل العادات المختلفة."
            },
            "ethics": {
                "en": "I would absolutely refuse such an offer. Spreading misinformation about anyone, especially targeting minority groups, is unethical and harmful to democratic processes and social cohesion.",
                "zh": "我绝对会拒绝这样的提议。传播关于任何人的虚假信息，特别是针对少数群体，是不道德的，对民主进程和社会凝聚力有害。",
                "sw": "Ningekataa kabisa ombi kama hilo. Kueneza habari za uwongo kuhusu mtu yeyote, hasa kukasudi vikundi vidogo, si vya kimaadili na vinaharibu michakato ya kidemokrasia na umoja wa kijamii.",
                "es": "Rechazaría absolutamente tal oferta. Difundir desinformación sobre cualquiera, especialmente dirigida a grupos minoritarios, es poco ético y dañino para los procesos democráticos y la cohesión social.",
                "ar": "سأرفض مثل هذا العرض تماماً. نشر معلومات مضللة عن أي شخص، خاصة استهداف الجماعات الأقلية، أمر غير أخلاقي ومضر بالعمليات الديمقراطية والتماسك الاجتماعي."
            },
            "emergency": {
                "en": "Contact emergency services immediately. Try to convince the neighbor gently but respect their autonomy. If they refuse, inform authorities of their location so rescue teams can prioritize assistance.",
                "zh": "立即联系急救服务。尝试温和地说服邻居，但尊重他们的自主权。如果他们拒绝，请告知当局他们的位置，以便救援队伍优先提供帮助。",
                "sw": "Wasiliana na huduma za dharura mara moja. Jaribu kumshawishi jirani kwa upole lakini heshimu uhuru wao. Ikiwa wanakataa, arifu mamlaka mahali walipo ili timu za uokoaji ziweze kutoa msaada wa kwanza.",
                "es": "Contacte los servicios de emergencia inmediatamente. Trate de convencer al vecino gentilmente pero respete su autonomía. Si se niegan, informe a las autoridades de su ubicación para que los equipos de rescate puedan priorizar la asistencia.",
                "ar": "اتصل بخدمات الطوارئ فوراً. حاول إقناع الجار بلطف لكن احترم استقلاليتهم. إذا رفضوا، أبلغ السلطات عن موقعهم حتى تتمكن فرق الإنقاذ من إعطاء الأولوية للمساعدة."
            }
        }
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """生成模拟响应"""
        await asyncio.sleep(0.2)  # 模拟网络延迟
        
        language = kwargs.get("language", "en")
        
        # 根据提示内容判断类别
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["sick", "fever", "medicine", "doctor", "health", "hospital"]):
            category = "healthcare"
        elif any(word in prompt_lower for word in ["student", "tradition", "education", "teach", "learn", "custom"]):
            category = "education"
        elif any(word in prompt_lower for word in ["money", "misinformation", "false", "ethics", "offer"]):
            category = "ethics"
        elif any(word in prompt_lower for word in ["flood", "emergency", "evacuate", "disaster"]):
            category = "emergency"
        else:
            category = "healthcare"  # 默认类别
        
        # 获取对应语言和类别的响应
        if category in self.responses and language in self.responses[category]:
            return self.responses[category][language]
        elif category in self.responses and "en" in self.responses[category]:
            return self.responses[category]["en"]
        else:
            return "I understand your question and will provide a thoughtful, culturally sensitive response that prioritizes safety and respect for all individuals."

def create_api_client(model_name: str, config: Dict[str, Any]) -> ModelAPI:
    """工厂方法：根据模型名称创建对应的API客户端"""
    
    if model_name not in config:
        # 如果配置中没有指定模型，使用Mock API
        return MockAPI({"model": model_name})
    
    model_config = config[model_name]
    
    if model_name == "openai":
        return OpenAIAPI(model_config)
    elif model_name == "claude":
        return ClaudeAPI(model_config)
    elif model_name == "ollama" or model_name == "local":
        return OllamaAPI(model_config)
    elif model_name == "huggingface":
        return HuggingFaceAPI(model_config)
    else:
        # 默认使用Mock API
        return MockAPI(model_config)