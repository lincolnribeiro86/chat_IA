# Dicionário com modelos e seus limites de tokens (janela de contexto)
limites_modelos = {
    "deepseek-r1:8b": 8192,
    "llama3.1": 8192,
    "llama3.2": 8192,
    "qwen3:8b": 8192,
    "qwen2.5:7b": 8192,
    "gemma3:4b": 128000,
    "mistral": 32000,
    "phi4-mini-reasoning:3.8b": 128000,
    "phi4-mini:3.8b": 128000,
    "phi3.5:3.8b": 8192,
    "granite3.3:8b": 128000,
    "gpt-4.1-2025-04-14": 1047576,
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "gpt-4o-mini-2024-07-18": 128000,
    "o3-2025-04-16": 200000,
    "o4-mini-2025-04-16": 200000,
    "gpt-4o": 128000,
    "gpt-4": 32000,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
    "gemini-1.5-flash": 1000000,
    "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1000000,
    "gemini-2.0-flash-lite": 1000000,
    "gemini-2.0-flash-preview-image-generation": 1000000,
    "gemini-2.5-pro-preview-03-25": 1048576,
    "gemini-2.5-flash-preview-04-17": 1048576,
}


def obter_limite_tokens(modelo_selecionado: str) -> int:
    """
    Retorna o limite de tokens para o modelo selecionado.
    Se o modelo não existir, retorna um valor padrão (ex: 4096).
    """
    return limites_modelos.get(modelo_selecionado, 4096)


def limitar_texto_por_tokens(texto: str, limite_tokens: int, caracteres_por_token: int = 4) -> str:
    limite_caracteres = limite_tokens * caracteres_por_token
    return texto[:limite_caracteres]


def tokens_para_paginas_a4(tokens: int, tokens_por_pagina: int = 475) -> float:
    """
    Converte a quantidade de tokens em número aproximado de páginas A4.

    :param tokens: Número de tokens.
    :param tokens_por_pagina: Estimativa de tokens por página A4 (padrão: 475).
    :return: Número aproximado de páginas A4 (float).
    """
    paginas = tokens / tokens_por_pagina
    return round(paginas, 2)


# Exemplo de uso:
# modelo = "mistral"
# limite = obter_limite_tokens(modelo)
# print(f"Limite de tokens para o modelo {modelo}: {limite}")
