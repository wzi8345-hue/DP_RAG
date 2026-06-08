"""Pipeline 步骤包。自动导入所有步骤以完成注册。"""

from .base import BaseStep, StepResult, register_step, get_step, list_steps

# 导入各步骤模块 (触发 @register_step 装饰器)
from .parse import ParseStep           # noqa: F401
from .chunk import ChunkStep           # noqa: F401
from .embed import EmbedStep           # noqa: F401
from .store import StoreStep           # noqa: F401
from .retrieve import RetrieveStep     # noqa: F401
from .generate import GenerateStep     # noqa: F401
