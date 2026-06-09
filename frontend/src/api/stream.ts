import type { StreamEvent } from './types'

/**
 * 解析 fetch 返回的 SSE 流（带 access_token 的 fetch + ReadableStream）。
 * 逐帧（空行分隔）解析 data: 行，回调每个事件。
 * 抛错 / abort 由调用方处理。
 */
export async function consumeSSE(
  res: Response,
  onEvent: (ev: StreamEvent) => void,
): Promise<void> {
  if (!res.body) throw new Error('No response body for SSE')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const dataLines = frame
        .split('\n')
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trim())
      if (dataLines.length === 0) continue
      try {
        onEvent(JSON.parse(dataLines.join('\n')) as StreamEvent)
      } catch {
        /* 跳过半帧 / keepalive 注释行 */
      }
    }
  }
}
