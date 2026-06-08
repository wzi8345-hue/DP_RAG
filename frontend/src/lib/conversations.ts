// 对话留存: 前端 localStorage 持久化 (零后端改动)。
// 每个会话保存完整消息 + backend session_id, 以便选中后继续执行 (后端凭 session_id 续接上下文)。

import type { Message } from "../components/ChatView";

const STORAGE_KEY = "dp-rag-conversations";
const MAX_CONVERSATIONS = 50;

export interface Conversation {
  id: string;
  title: string;
  sessionId: string | null;
  expert: boolean;
  messages: Message[];
  updatedAt: number;
}

function readAll(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr as Conversation[];
  } catch {
    return [];
  }
}

function writeAll(convs: Conversation[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(convs));
  } catch {
    /* 配额溢出等忽略 */
  }
}

/** 全部会话, 按最近更新倒序。 */
export function loadConversations(): Conversation[] {
  return readAll().sort((a, b) => b.updatedAt - a.updatedAt);
}

export function loadConversation(id: string): Conversation | null {
  return readAll().find((c) => c.id === id) ?? null;
}

/** upsert 一个会话 (按 id 覆盖), 并裁剪到上限。 */
export function saveConversation(conv: Conversation): void {
  const all = readAll().filter((c) => c.id !== conv.id);
  all.push(conv);
  all.sort((a, b) => b.updatedAt - a.updatedAt);
  writeAll(all.slice(0, MAX_CONVERSATIONS));
}

export function deleteConversation(id: string): void {
  writeAll(readAll().filter((c) => c.id !== id));
}

export function newConversationId(): string {
  return `c_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

/** 由消息列表推导标题 (取首条用户提问)。 */
export function deriveTitle(messages: Message[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  const text = (firstUser?.content || "新对话").trim().replace(/\s+/g, " ");
  return text.length > 28 ? text.slice(0, 28) + "…" : text;
}
