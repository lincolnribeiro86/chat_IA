"""
GPT-5 Handler com novos recursos: Verbosity, CFG, Free-form Function Calling e Minimal Reasoning
Baseado na documentação oficial do GPT-5
"""

import os
from typing import List, Dict, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class GPT5Handler:
    """Classe para gerenciar interações com GPT-5 usando os novos recursos"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.supported_models = [
            "gpt-5",
            "gpt-5-mini", 
            "gpt-5-nano"
        ]
    
    def create_response_with_verbosity(
        self,
        input_message: str,
        model: str = "gpt-5-mini",
        verbosity: str = "medium",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Cria resposta usando o parâmetro de verbosity
        
        Args:
            input_message: Mensagem do usuário
            model: Modelo GPT-5 a usar
            verbosity: "low", "medium", "high"
            conversation_history: Histórico da conversa
            
        Returns:
            Dict com resposta e metadados
        """
        try:
            # Prepara o input baseado no histórico
            if conversation_history:
                messages = conversation_history + [{"role": "user", "content": input_message}]
            else:
                messages = [{"role": "user", "content": input_message}]
            
            # GPT-5 não aceita parâmetro temperature customizado
            response = self.client.responses.create(
                model=model,
                input=messages,
                text={"verbosity": verbosity}
            )
            
            # Extrai o texto da resposta
            output_text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text
            
            return {
                "response": output_text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "verbosity": verbosity,
                "model": model
            }
            
        except Exception as e:
            return {"error": f"Erro ao criar resposta: {str(e)}"}
    
    def create_response_with_minimal_reasoning(
        self,
        input_message: str,
        model: str = "gpt-5",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Cria resposta usando minimal reasoning para respostas mais rápidas
        Ideal para tarefas determinísticas e leves
        """
        try:
            if conversation_history:
                messages = conversation_history + [{"role": "user", "content": input_message}]
            else:
                messages = [{"role": "user", "content": input_message}]
            
            response = self.client.responses.create(
                model=model,
                input=messages,
                reasoning={"effort": "minimal"}
            )
            
            # Extrai o texto da resposta
            output_text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text
            
            return {
                "response": output_text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "reasoning_effort": "minimal",
                "model": model
            }
            
        except Exception as e:
            return {"error": f"Erro ao criar resposta: {str(e)}"}
    
    def create_response_with_custom_tool(
        self,
        input_message: str,
        tool_name: str,
        tool_description: str,
        model: str = "gpt-5",
        grammar_definition: Optional[str] = None,
        grammar_syntax: str = "lark"
    ) -> Dict[str, Any]:
        """
        Cria resposta usando free-form function calling com CFG opcional
        
        Args:
            input_message: Mensagem do usuário
            tool_name: Nome da ferramenta personalizada
            tool_description: Descrição da ferramenta
            model: Modelo GPT-5 a usar
            grammar_definition: Definição da gramática (opcional)
            grammar_syntax: "lark" ou "regex"
        """
        try:
            # Configura a ferramenta personalizada
            tool_config = {
                "type": "custom",
                "name": tool_name,
                "description": tool_description
            }
            
            # Adiciona gramática se fornecida
            if grammar_definition:
                tool_config["format"] = {
                    "type": "grammar",
                    "syntax": grammar_syntax,
                    "definition": grammar_definition
                }
            
            response = self.client.responses.create(
                model=model,
                input=input_message,
                text={"format": {"type": "text"}},
                tools=[tool_config],
                parallel_tool_calls=False
            )
            
            # Extrai resposta e chamada de ferramenta
            result = {
                "model": model,
                "tool_name": tool_name
            }
            
            if len(response.output) > 1 and response.output[1].type == "custom_tool_call":
                tool_call = response.output[1]
                result.update({
                    "tool_call": {
                        "name": tool_call.name,
                        "input": tool_call.input,
                        "call_id": tool_call.call_id
                    }
                })
            
            # Extrai texto da resposta principal
            output_text = ""
            if response.output:
                item = response.output[0]
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text
            
            result["response"] = output_text
            result["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Erro ao criar resposta com ferramenta: {str(e)}"}
    
    def create_sql_query_with_dialect(
        self,
        query_description: str,
        dialect: str = "postgresql",
        model: str = "gpt-5"
    ) -> Dict[str, Any]:
        """
        Exemplo específico de CFG para gerar SQL em dialetos específicos
        """
        # Gramáticas para diferentes dialetos SQL
        postgres_grammar = '''
            // ---------- Punctuation & operators ----------
            SP: " "
            COMMA: ","
            GT: ">"
            EQ: "="
            SEMI: ";"

            // ---------- Start ----------
            start: "SELECT" SP select_list SP "FROM" SP table SP "WHERE" SP amount_filter SP "AND" SP date_filter SP "ORDER" SP "BY" SP sort_cols SP "LIMIT" SP NUMBER SEMI

            // ---------- Projections ----------
            select_list: column (COMMA SP column)*
            column: IDENTIFIER

            // ---------- Tables ----------
            table: IDENTIFIER

            // ---------- Filters ----------
            amount_filter: "total_amount" SP GT SP NUMBER
            date_filter: "order_date" SP GT SP DATE

            // ---------- Sorting ----------
            sort_cols: "order_date" SP "DESC"

            // ---------- Terminals ----------
            IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*/
            NUMBER: /[0-9]+/
            DATE: /'[0-9]{4}-[0-9]{2}-[0-9]{2}'/
        '''
        
        mssql_grammar = '''
            // ---------- Punctuation & operators ----------
            SP: " "
            COMMA: ","
            GT: ">"
            EQ: "="
            SEMI: ";"

            // ---------- Start ----------
            start: "SELECT" SP "TOP" SP NUMBER SP select_list SP "FROM" SP table SP "WHERE" SP amount_filter SP "AND" SP date_filter SP "ORDER" SP "BY" SP sort_cols SEMI

            // ---------- Projections ----------
            select_list: column (COMMA SP column)*
            column: IDENTIFIER

            // ---------- Tables ----------
            table: IDENTIFIER

            // ---------- Filters ----------
            amount_filter: "total_amount" SP GT SP NUMBER
            date_filter: "order_date" SP GT SP DATE

            // ---------- Sorting ----------
            sort_cols: "order_date" SP "DESC"

            // ---------- Terminals ----------
            IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*/
            NUMBER: /[0-9]+/
            DATE: /'[0-9]{4}-[0-9]{2}-[0-9]{2}'/
        '''
        
        grammar = postgres_grammar if dialect == "postgresql" else mssql_grammar
        tool_name = f"{dialect}_grammar"
        tool_description = f"Executes read-only {dialect.upper()} queries with specific syntax constraints. YOU MUST REASON HEAVILY ABOUT THE QUERY AND MAKE SURE IT OBEYS THE GRAMMAR."
        
        return self.create_response_with_custom_tool(
            input_message=f"Call the {tool_name} to {query_description}",
            tool_name=tool_name,
            tool_description=tool_description,
            model=model,
            grammar_definition=grammar,
            grammar_syntax="lark"
        )
    
    def stream_response_with_verbosity(
        self,
        input_message: str,
        model: str = "gpt-5-mini",
        verbosity: str = "medium",
        conversation_history: Optional[List[Dict]] = None
    ):
        """
        Generator que faz streaming da resposta com verbosity
        Para uso em interfaces de chat com resposta em tempo real
        """
        try:
            if conversation_history:
                messages = conversation_history + [{"role": "user", "content": input_message}]
            else:
                messages = [{"role": "user", "content": input_message}]
            
            # Note: O streaming com responses API pode não estar disponível
            # Esta implementação usa o chat completions como fallback
            # GPT-5 não suporta temperature customizado
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Erro no streaming: {str(e)}"
    
    def is_gpt5_model(self, model_name: str) -> bool:
        """Verifica se o modelo é da família GPT-5"""
        return any(gpt5_name in model_name for gpt5_name in ["gpt-5", "gpt-5-mini", "gpt-5-nano"])

# Exemplo de uso e testes
def test_gpt5_features():
    """Função para testar os recursos do GPT-5"""
    handler = GPT5Handler()
    
    # Teste 1: Verbosity
    print("=== Teste de Verbosity ===")
    question = "Explique o que é inteligência artificial"
    
    for verbosity in ["low", "medium", "high"]:
        result = handler.create_response_with_verbosity(
            input_message=question,
            verbosity=verbosity
        )
        print(f"\nVerbosity: {verbosity}")
        print(f"Tokens de saída: {result.get('usage', {}).get('output_tokens', 'N/A')}")
        print(f"Resposta: {result.get('response', 'Erro')[:200]}...")
    
    # Teste 2: Minimal Reasoning
    print("\n=== Teste de Minimal Reasoning ===")
    simple_task = "Classifique o sentimento como positivo|neutro|negativo: 'Adorei o filme!'"
    
    result = handler.create_response_with_minimal_reasoning(
        input_message=simple_task
    )
    print(f"Resposta rápida: {result.get('response', 'Erro')}")
    
    # Teste 3: Custom Tool com Regex
    print("\n=== Teste de Custom Tool ===")
    timestamp_result = handler.create_response_with_custom_tool(
        input_message="Gere um timestamp para 7 de agosto de 2025 às 10h",
        tool_name="timestamp_generator",
        tool_description="Gera timestamp no formato YYYY-MM-DD HH:MM",
        grammar_definition=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d$",
        grammar_syntax="regex"
    )
    
    if "tool_call" in timestamp_result:
        print(f"Timestamp gerado: {timestamp_result['tool_call']['input']}")

if __name__ == "__main__":
    test_gpt5_features()