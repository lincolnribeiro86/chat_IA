import { useState, useCallback } from 'react'
import { conversationsApi } from '@/lib/api'
import type { Conversation } from '@/types'

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setConversations(await conversationsApi.list())
    } catch { /* ignore */ } finally {
      setLoading(false)
    }
  }, [])

  const remove = useCallback(async (id: string) => {
    await conversationsApi.delete(id)
    setConversations(prev => prev.filter(c => c.id !== id))
  }, [])

  const exportConv = useCallback(async (id: string, fmt: 'markdown' | 'json') => {
    const { data, ext, mime } = await conversationsApi.export(id, fmt)
    const blob = new Blob([data], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `conversation.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  return { conversations, loading, refresh, remove, exportConv }
}
