"""数据库层（psycopg 3 + pydantic，无 ORM/迁移框架）。

- base: 连接池 / 游标 / init_db / configured
- models: 实体（Conversation / Message / KbCollection / Document / UserSkill）
- schema: 幂等 DDL（lifespan 中初始化）

备份/迁移走 shell 全量备份（deploy/backup.sh）。
"""

from .base import close_pool, configured, connection, cursor, get_pool, init_db
from .models import (
    Conversation,
    ConversationShare,
    Document,
    GenerationRun,
    IngestTask,
    IngestTaskEvent,
    IngestTaskItem,
    KbCollection,
    Message,
    MessageEvent,
    UserSkill,
)

__all__ = [
    "configured",
    "connection",
    "cursor",
    "get_pool",
    "init_db",
    "close_pool",
    "Conversation",
    "ConversationShare",
    "Document",
    "GenerationRun",
    "IngestTask",
    "IngestTaskEvent",
    "IngestTaskItem",
    "KbCollection",
    "Message",
    "MessageEvent",
    "UserSkill",
]
