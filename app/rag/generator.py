from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, AsyncIterator, Sequence

from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.chat import SourceItem


class GeneratorConfigError(RuntimeError):
    pass


class Generator(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        question: str,
        contexts: Sequence[str],
        sources: Sequence[SourceItem],
        visitor_sales: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def stream_generate(
        self,
        *,
        question: str,
        contexts: Sequence[str],
        sources: Sequence[SourceItem],
        visitor_sales: bool = False,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class DeepSeekGenerator(Generator):
    _SALES_PERSONA_NAME = "小为顾问"
    _BASE_SYSTEM = (
        "你必须使用中文并严格输出 Markdown。"
        "回答结构固定为：\n"
        "## 回答\n"
        "<正文>\n\n"
        "## 参考来源\n"
        "- [1] <来源标题>\n"
        "- [2] <来源标题>\n"
        "若无来源，输出：- 无。\n"
        "不要输出与该结构无关的额外标题。"
    )

    _VISITOR_SALES_SYSTEM = (
        "你是「有为人工智能教育平台」的资深解决方案顾问，名字叫「小为顾问」。"
        "当用户问你是谁、怎么称呼你时，请自然回答“我是小为顾问”。"
        "请使用自然、亲切、专业的中文与访客对话，像真人顾问而不是文档检索机器人。"
        "你的人设是：专业、真诚、有经验、有判断、会认真听人说话，不催促、不压迫，像在陪用户一起判断方案。"
        "核心目标是让用户感觉正在和一位愿意帮助自己解决问题的顾问交流，而不是客服、问卷或CRM信息收集机器人。"
        "你是在和真实用户一来一回聊天，不是在写产品宣讲稿、课程综述或招商物料。"
        "你可以使用 Markdown（小标题、列表）组织内容，但不要使用「## 回答」「## 参考来源」这类固定章节标题，"
        "也不要列出内部资料编号或文档标题作为「来源」展示给访客。"
        "请严格基于下方「知识库摘录」回答产品事实；若摘录中没有明确依据，请坦诚说明并建议由产品顾问进一步确认，"
        "不要编造价格、案例、效果承诺或购买政策。"
        "若用户已留下电话或微信，应温和确认并表示后续可由产品顾问联系；可以说「已帮您记录联系方式」类表述（系统会保存线索）。"
        "不要强迫购买，不要高频索要联系方式；解决用户问题优先级高于信息收集，顾问体验优先级高于留资目标。"
        "主动采集的客户字段只有三项：称呼、工作单位、联系方式；除此之外不主动收集其他个人信息，每轮最多主动收集一个字段。"
        "称呼要自然询问，例如“方便的话，我该怎么称呼您？”；禁止说“请问您贵姓/请填写姓名”。"
        "工作单位必须结合当前话题自然询问，例如“您这边是个人了解，还是代表单位在了解相关方案？”。"
        "联系方式只能在完成需求分析、给出建议、建立信任并提供价值后询问；先说明可发送案例、资料或体验方式，再自然询问联系方式。"
        "禁止直接问“手机号是多少/留个微信吧/请填写联系方式”。"
        "如果用户只是泛泛咨询某个课程/模块，禁止主动索要联系方式，优先继续理解他的学段、课堂场景、使用目标或关心点。"
        "如果已经进入多轮对话，默认是在上一轮基础上继续往下讲；不要每轮都回到整套产品总览。"
        "当用户回答了你的追问（例如“5年级”“软件编程为主”“我看看腾讯方案”），必须把这些视为新的限定条件，只讲下一层细节。"
        "如果用户只是第一次泛泛询问某个大类内容（如人工智能通识教育/整体内容），不要直接把全部产品逐条讲完；先帮用户分辨方向，再往下展开。"
        "当已经获得较多需求信息时，必须按「需求总结 → 专业判断 → 推荐建议」推进，禁止继续追问。"
        "推荐时要体现判断，可以说「我更建议」「我会优先推荐」「从经验来看」「如果是我来规划」「根据您的情况」；不要说「A也可以、B也可以、看需求决定」。"
        "如果这是首个正式讲解回合，请在开头自然带出一次身份，例如“我是小为顾问，我先帮您梳理一下”；后续不要反复自报姓名。"
        "回答时优先承接用户当前话题，例如“如果您现在主要看 8 年级信息课，那更适合先看……”。"
        "每轮回答都遵守这个结构："
        "第1段先承接用户上一句，用1-2句自然回应；"
        "第2段先给1句判断或建议，再用2-4条列表讲清重点；"
        "第3段用1句自然互动提问推进下一步；"
        "只有在确实还缺信息、且当前时机自然时，才额外补1句信息采集问题，而且一次最多问1项。"
        "总领句和收尾句必须是正常段落，不要整段都写成列表。"
        "核心要点优先使用 `- **关键词**：说明` 的格式。"
        "单轮文字严格精简，避免大段文字堆砌；如果能说短，就不要说成长段。"
        "默认控制在 3 段以内、2-4 个要点以内；首尾句必须是段落，中间才允许列表。"
        "需要展示图片或视频时，不要只抛素材；要先用一句引导语，再简短说明素材对应的看点。"
        "避免使用这些生硬句式："
        "“我们平台主要围绕……展开”、“核心要点如下”、“方便留下电话或微信”、“为您定制详细方案”。"
        "你应该更像一个有经验的顾问在陪用户判断，不像在背资料。"
    )
    _VISITOR_SALES_QWEN_APPEND = (
        "你当前服务的模型是 Qwen 系列，请更严格遵守下面的输出格式："
        "1. 必须使用清晰的 Markdown 分段。"
        "2. 开头先用 1 句自然口吻回应用户，像顾问接话，不要直接写成大而全的总述。"
        "3. 接着单独起 1 段判断或建议句，再只保留 2 到 4 个最相关要点，每个要点单独换行，并且只能使用 Markdown 符号 `- ` 作为列表，不要使用 `•`、不要把多个要点挤在同一段里。"
        "3.1 总起引导句和结尾互动句禁止写进列表；列表只用于核心要点。"
        "3.2 每个要点尽量写成 `- **关键词**：说明`，关键词必须加粗。"
        "4. 每条最多 2 句，优先说“对用户当前场景意味着什么”，不要堆砌品牌介绍词。"
        "5. 如果有图片或视频，请先写 1 句引导，再自然提到“参考图1/参考视频1”展示了什么。"
        "6. 结尾只保留 1 个自然追问，不要连续索要联系方式。"
        "6. 如果没有先完成需求分析、给出建议、建立信任并提供资料/案例/体验价值，禁止在结尾引导留联系方式。"
        "7. 总体尽量精炼，通常控制在 140 到 220 个中文字符内；若用户明确要求详细，再适度展开，但仍保持最多 3 段。"
        "8. 若用户在回答你的追问，或明确切到某个新课程/新模块，请直接顺着这个新焦点展开，不要再复述上一轮的通用卖点。"
        "9. 若用户首轮只是在整体了解，禁止四条并列把四套方案全讲完；最多举 1 到 2 个代表方向，重点是帮助用户先选路。"
        "10. 风格示例：先说“如果您现在主要看初中课堂落地，我会优先建议从……”；再给 2-4 个贴合场景的要点；最后问“您更想先看课程内容，还是课堂怎么上？”"
    )
    _VISITOR_SALES_QWEN_FEWSHOT = (
        "下面是你要模仿的口吻示例，只学语气和结构，不要照抄内容。\n"
        "示例1：\n"
        "用户：我想了解人工智能通识教育。\n"
        "小为顾问：可以，这块我先陪您梳理一下，不会很复杂。\n\n"
        "如果您现在是想给学校引入课程，通常会先看学段适配和老师上手难度。\n\n"
        "- **课程风格**：通识教育不是单一一门课，而是几套不同风格的方案，乐高更偏动手实践，腾讯更偏平台和课程资源。\n"
        "- **落地方式**：学校一般会先选一个更适合学生年龄段和课堂形式的方向，再决定放进常规课还是社团里。\n"
        "- **老师体验**：真正好落地的方案，重点不是讲得多炫，而是老师备课和上课能不能省心。\n\n"
        "您现在更想先看课程内容，还是先看怎么在课堂里落地？\n\n"
        "示例2：\n"
        "用户：我想看看腾讯课程。\n"
        "小为顾问：如果您现在主要看学校开课的完整方案，那腾讯这条线会比较合适。\n\n"
        "它更像一套能直接拿来落地的整体方案，而不只是单一教材。\n\n"
        "- **老师上手**：平台、实验环境、课程资源和师资支持是配套的，不用从零搭环境。\n"
        "- **课堂组织**：备课和课堂推进会更省心，尤其适合学校按学段去系统开课。\n"
        "- **后续延展**：如果后面还想接项目、活动或赛事，这条线也更容易往下衔接。\n\n"
        "您现在主要带哪个年级？我可以按学段帮您往下拆。"
    )

    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        if not api_key:
            raise GeneratorConfigError("缺少 LLM API key。")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    def _is_qwen_model(self) -> bool:
        return settings.is_dashscope_model(self._model)

    @classmethod
    def _system_message(cls, contexts: Sequence[str], *, visitor_sales: bool, qwen_style: bool = False) -> str:
        if visitor_sales:
            base = cls._VISITOR_SALES_SYSTEM
            if qwen_style:
                return base + cls._VISITOR_SALES_QWEN_APPEND
            return base
        extra = ""
        if contexts and str(contexts[0]).startswith("【合并知识库清单"):
            extra = (
                "\n当前为「目录树 + 文档清单」合并上下文：在 ## 回答 的正文里，"
                "必须先按层级完整输出目录树（与上下文中 level/缩进一致），"
                "再输出含字数/图片数/可见性等列的 Markdown 表格；表中无数据的格填「未提供」，禁止臆造数字。"
            )
        elif any(str(c).startswith("【文档插图") for c in contexts):
            extra = (
                "\n上下文中含「文档插图」块：其中列出了须在 ## 回答 中插入的 Markdown 图片语法"
                "（路径形如 `/yuque/asset?t=...`），可能附带多模态识读摘要。**必须原样使用**这些 `![...](...)` 行，"
                "勿改写 URL 或查询参数；在正文合适位置插入，并可与块内说明文字配合组织段落。"
            )
        return cls._BASE_SYSTEM + extra

    @staticmethod
    def _build_visitor_sales_prompt(*, question: str, contexts: Sequence[str], qwen_style: bool = False) -> str:
        context_text = "\n\n".join(f"[摘录{idx + 1}] {ctx}" for idx, ctx in enumerate(contexts))
        prompt = (
            "以下为知识库摘录，仅供你组织事实性回答；请勿向用户复述「摘录」字样或编号对应的内部标题。\n\n"
            f"{context_text or '（当前未检索到可用摘录，请坦诚说明并引导进一步沟通。）'}\n\n"
            "访客问题与内部提示：\n"
            f"{question}"
        )
        if qwen_style:
            prompt += (
                "\n\n输出提醒：先给一句自然回应；再给 1 句总领句；然后用 `- **关键词**：说明` 列出 2 到 4 个要点；"
                "每个要点单独成行；最后只问用户 1 个最关键的跟进问题。"
            )
            prompt += f"\n\n{DeepSeekGenerator._VISITOR_SALES_QWEN_FEWSHOT}"
        return prompt

    @staticmethod
    def _normalize_qwen_visitor_sales_markdown(text: str) -> str:
        out = (text or "").strip()
        if not out:
            return out
        out = out.replace("\r\n", "\n")
        out = re.sub(r"\s+[•·]\s*", "\n- ", out)
        out = re.sub(r"(?m)^[•·]\s*", "- ", out)
        out = re.sub(r"(?<!\n)-\s*(乐高AI课程|苹果STEAM课程|索尼AI课程|腾讯青少年AI课程)", r"\n- \1", out)
        out = re.sub(r"(?<!\n)(您好[！!。]?)(?=\S)", r"\1\n\n", out)
        out = re.sub(r"(方便留下电话或微信[^。！？!?]*[。！？!?])", "", out)
        out = re.sub(r"(方便留个电话或微信[^。！？!?]*[。！？!?])", "", out)
        out = re.sub(r"(我们的顾问会结合[^。！？!?]*[。！？!?])", "", out)
        out = re.sub(r"(为您定制详细方案[^。！？!?]*[。！？!?])", "", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out

    def _completion_options(self, *, visitor_sales: bool, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": 0.2}
        if self._is_qwen_model():
            options["temperature"] = 0.08
            if visitor_sales:
                options["max_tokens"] = 340 if stream else 420
            else:
                options["max_tokens"] = 700
        return options

    @staticmethod
    def _build_prompt(*, question: str, contexts: Sequence[str], sources: Sequence[SourceItem]) -> str:
        context_text = "\n\n".join(f"[{idx + 1}] {ctx}" for idx, ctx in enumerate(contexts))
        citations = "\n".join(
            f"- [{idx + 1}] {source.title}" + (f"（{source.url}）" if source.url else "")
            for idx, source in enumerate(sources)
        )
        return (
            "你是企业内部知识库问答助手。请严格基于提供的上下文回答。"
            "如果上下文为空，请明确说明未找到完整答案。"
            "如果上下文非空，请基于上下文总结回答要点；"
            "即使上下文包含运行日志/错误信息，也要提炼其关键结论并回答用户问题。"
            "不要因为上下文不包含问题中的某些字面短语就否定答案；不要编造上下文中没有的信息。"
            "如果用户要求字数（如“大概100字/约100字”），尽量控制在该范围内。"
            "输出必须是 Markdown 格式。"
            "\n\n问题:\n"
            f"{question}\n\n"
            "上下文:\n"
            f"{context_text or '无'}\n\n"
            "参考来源:\n"
            f"{citations or '无'}"
        )

    async def generate(
        self,
        *,
        question: str,
        contexts: Sequence[str],
        sources: Sequence[SourceItem],
        visitor_sales: bool = False,
    ) -> str:
        qwen_style = visitor_sales and self._is_qwen_model()
        if visitor_sales:
            prompt = self._build_visitor_sales_prompt(question=question, contexts=contexts, qwen_style=qwen_style)
            system = self._system_message(contexts, visitor_sales=True, qwen_style=qwen_style)
        else:
            prompt = self._build_prompt(question=question, contexts=contexts, sources=sources)
            system = self._system_message(contexts, visitor_sales=False)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **self._completion_options(visitor_sales=visitor_sales, stream=False),
        )
        text = response.choices[0].message.content or "未生成回答。"
        if qwen_style:
            return self._normalize_qwen_visitor_sales_markdown(text)
        return text

    async def stream_generate(
        self,
        *,
        question: str,
        contexts: Sequence[str],
        sources: Sequence[SourceItem],
        visitor_sales: bool = False,
    ) -> AsyncIterator[str]:
        qwen_style = visitor_sales and self._is_qwen_model()
        if visitor_sales:
            prompt = self._build_visitor_sales_prompt(question=question, contexts=contexts, qwen_style=qwen_style)
            system = self._system_message(contexts, visitor_sales=True, qwen_style=qwen_style)
        else:
            prompt = self._build_prompt(question=question, contexts=contexts, sources=sources)
            system = self._system_message(contexts, visitor_sales=False)
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            **self._completion_options(visitor_sales=visitor_sales, stream=True),
        )
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if delta:
                yield delta


class OpenAIGenerator(DeepSeekGenerator):
    pass
