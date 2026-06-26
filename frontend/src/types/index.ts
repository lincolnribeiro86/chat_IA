export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  images?: string[]        // base64 data URIs (display only)
  tool_calls?: ToolCall[]
  usage?: UsageInfo
  streaming?: boolean
}

export interface ToolCall {
  id: string
  name: string
  args: Record<string, unknown>
  result?: string
  status: 'running' | 'done' | 'error'
}

export interface UsageInfo {
  input_tokens: number
  output_tokens: number
  cost_usd?: number | null
}

export interface ModelInfo {
  id: string
  name: string
  provider: string
  supports_vision: boolean
  supports_tools: boolean
  context_window: number
}

export interface ProviderGroup {
  id: string
  name: string
  models: ModelInfo[]
}

export interface Conversation {
  id: string
  title: string
  model_used: string
  updated_at: string
}

export interface FileAttachment {
  name: string
  type: 'text' | 'image'
  content?: string
  data_uri?: string
  error?: string | null
}

export interface AppSettings {
  openai_api_key: boolean
  anthropic_api_key: boolean
  gemini_api_key: boolean
  groq_api_key: boolean
  openrouter_api_key: boolean
  deepseek_api_key: boolean
  ollama_base_url: string
  ollama_api_key: boolean
  tavily_api_key: boolean
  firecrawl_api_key: boolean
  claude_code_oauth_token: boolean
}
