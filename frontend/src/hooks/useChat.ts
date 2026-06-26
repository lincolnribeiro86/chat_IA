import { useState, useRef, useCallback } from 'react'
import { streamChat } from '@/lib/api'
import type { Message, FileAttachment, ToolCall, UsageInfo } from '@/types'

function uid() {
  return crypto.randomUUID()
}

interface UseChatOptions {
  modelId: string
  temperature: number
  enableWebSearch: boolean
  forceWebSearch: boolean
  apiKeys?: Record<string, string>
}

export function useChat(opts: UseChatOptions) {
  const [messages, setMessages] = useState<Message[]>([])
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const append = useCallback(
    async (userContent: string, attachments: FileAttachment[] = []) => {
      const userMsg: Message = {
        id: uid(),
        role: 'user',
        content: userContent,
        images: attachments.filter(f => f.type === 'image').map(f => f.data_uri!),
      }

      const assistantId = uid()
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        streaming: true,
        tool_calls: [],
      }

      setMessages(prev => [...prev, userMsg, assistantMsg])
      setStreaming(true)

      const ctrl = new AbortController()
      abortRef.current = ctrl

      const historyForApi = [...messages, userMsg].map(m => ({
        role: m.role,
        content: m.content,
      }))

      await streamChat(
        {
          model_id: opts.modelId,
          messages: historyForApi,
          files: attachments,
          temperature: opts.temperature,
          enable_web_search: opts.enableWebSearch,
          force_web_search: opts.forceWebSearch,
          api_keys: opts.apiKeys || {},
        },
        {
          onToken(token) {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId ? { ...m, content: m.content + token } : m
              )
            )
          },
          onToolStart(toolId, name, args) {
            const tc: ToolCall = { id: toolId, name, args, status: 'running' }
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, tool_calls: [...(m.tool_calls || []), tc] }
                  : m
              )
            )
          },
          onToolResult(toolId, _name, result) {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? {
                      ...m,
                      tool_calls: (m.tool_calls || []).map(tc =>
                        tc.id === toolId ? { ...tc, result, status: 'done' as const } : tc
                      ),
                    }
                  : m
              )
            )
          },
          onUsage(usage) {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId ? { ...m, usage: usage as UsageInfo } : m
              )
            )
          },
          onDone() {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId ? { ...m, streaming: false } : m
              )
            )
            setStreaming(false)
          },
          onError(msg) {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content || `Erro: ${msg}`, streaming: false }
                  : m
              )
            )
            setStreaming(false)
          },
        },
        ctrl.signal
      )
    },
    [messages, opts]
  )

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setStreaming(false)
    setMessages(prev =>
      prev.map(m => (m.streaming ? { ...m, streaming: false } : m))
    )
  }, [])

  const clear = useCallback(() => {
    stop()
    setMessages([])
  }, [stop])

  const loadMessages = useCallback((msgs: Message[]) => {
    setMessages(msgs)
  }, [])

  return { messages, streaming, append, stop, clear, loadMessages }
}
