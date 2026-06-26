import { Globe, Loader2, CheckCircle2, XCircle } from 'lucide-react'
import type { ToolCall } from '@/types'

interface Props { toolCall: ToolCall }

const TOOL_ICONS: Record<string, React.ReactNode> = {
  web_search: <Globe className="h-3.5 w-3.5" />,
  web_scrape: <Globe className="h-3.5 w-3.5" />,
}

export function ToolCallCard({ toolCall }: Props) {
  const icon = TOOL_ICONS[toolCall.name] ?? <Globe className="h-3.5 w-3.5" />
  const query = (toolCall.args as Record<string, string>)?.query || (toolCall.args as Record<string, string>)?.url || ''

  return (
    <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/50 rounded-md px-3 py-2 my-1">
      <div className="mt-0.5">{icon}</div>
      <div className="flex-1 min-w-0">
        <span className="font-medium">{toolCall.name}</span>
        {query && <span className="ml-1 text-muted-foreground truncate">— {query}</span>}
      </div>
      <div className="shrink-0">
        {toolCall.status === 'running' && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
        {toolCall.status === 'done' && <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />}
        {toolCall.status === 'error' && <XCircle className="h-3.5 w-3.5 text-destructive" />}
      </div>
    </div>
  )
}
