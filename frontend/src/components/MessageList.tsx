import { useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { User, Bot, Copy, Check } from 'lucide-react'
import { CostBadge } from './CostBadge'
import { ToolCallCard } from './ToolCallCard'
import { cn } from '@/lib/utils'
import type { Message } from '@/types'
import { useState } from 'react'

interface Props { messages: Message[] }

export function MessageList({ messages }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const isNearBottomRef = useRef(true)

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    isNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }, [])

  useEffect(() => {
    if (isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground gap-3">
        <Bot className="h-12 w-12 opacity-20" />
        <p className="text-sm">Selecione um modelo e comece a conversar</p>
      </div>
    )
  }

  return (
    <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [text])
  return (
    <button
      onClick={copy}
      className="absolute top-2 right-2 p-1.5 rounded bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-colors"
      title="Copiar"
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex gap-3', isUser && 'flex-row-reverse')}>
      {/* Avatar */}
      <div className={cn(
        'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
        isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Content */}
      <div className={cn('flex flex-col gap-1 max-w-[85%]', isUser && 'items-end')}>
        {/* Images */}
        {message.images && message.images.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-1">
            {message.images.map((uri, i) => (
              <img key={i} src={uri} alt="attachment" className="max-h-48 rounded-md border object-contain" />
            ))}
          </div>
        )}

        {/* Tool calls */}
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="w-full">
            {message.tool_calls.map(tc => <ToolCallCard key={tc.id} toolCall={tc} />)}
          </div>
        )}

        {/* Text */}
        {(message.content || message.streaming) && (
          <div className={cn(
            'rounded-2xl px-4 py-3 text-sm',
            isUser
              ? 'bg-primary text-primary-foreground rounded-tr-sm'
              : 'bg-muted rounded-tl-sm w-full'
          )}>
            {isUser ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <div className={cn('markdown-body', message.streaming && 'typing-cursor')}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    pre({ children, ...props }) {
                      // Extrai o texto do bloco de código para o botão de copiar
                      const codeEl = (children as React.ReactElement)
                      const codeText = codeEl?.props?.children ?? ''
                      const lang = (codeEl?.props?.className ?? '').replace('language-', '')
                      return (
                        <div className="relative group my-3 rounded-lg overflow-hidden border border-white/10">
                          {lang && (
                            <div className="flex items-center justify-between px-4 py-1.5 bg-zinc-800 text-xs text-zinc-400 font-mono border-b border-white/10">
                              <span>{lang}</span>
                            </div>
                          )}
                          <pre {...props} className="!m-0 !rounded-none overflow-x-auto">
                            {children}
                          </pre>
                          <CopyButton text={typeof codeText === 'string' ? codeText : String(codeText)} />
                        </div>
                      )
                    },
                    code({ className, children, ...props }) {
                      const isBlock = className?.startsWith('language-')
                      if (isBlock) {
                        return <code className={className} {...props}>{children}</code>
                      }
                      return (
                        <code className="bg-zinc-200 dark:bg-zinc-700 text-pink-600 dark:text-pink-400 px-1.5 py-0.5 rounded text-[0.85em] font-mono" {...props}>
                          {children}
                        </code>
                      )
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            )}
          </div>
        )}

        {/* Usage */}
        {message.usage && !message.streaming && (
          <CostBadge usage={message.usage} />
        )}
      </div>
    </div>
  )
}
