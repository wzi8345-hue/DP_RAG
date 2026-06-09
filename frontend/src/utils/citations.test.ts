import { describe, expect, it } from 'vitest'
import { parseCitations, stripStreamingCitations } from './citations'
import type { Hit } from '@/api/types'

describe('citations', () => {
  const hits: Hit[] = [
    { chunk_id: 'text_8bb1f28e', doc_id: 'doc-a', doc_name: '论文 A', content: '正文片段' },
  ]

  it('replaces bracket citation with a footnote marker', () => {
    const { markdown, citedHits, hasCitations } = parseCitations(
      '钢材耐蚀 [text_8bb1f28e, 2.1 试验, page 2, para 9]。',
      hits,
    )
    expect(hasCitations).toBe(true)
    expect(citedHits).toHaveLength(1)
    expect(markdown).toContain('[1](#cite-1)')
  })

  it('keeps text without resolvable citations untouched', () => {
    const { hasCitations } = parseCitations('没有引用的句子。', hits)
    expect(hasCitations).toBe(false)
  })

  it('hides raw citations during streaming', () => {
    const out = stripStreamingCitations('优良 [text_77336824, 3, page 7]')
    expect(out).not.toContain('text_77336824')
  })
})
