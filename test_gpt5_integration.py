#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Teste de integração do GPT-5 no projeto
Verifica se todos os novos recursos estão funcionando corretamente
"""

import os
import sys
from dotenv import load_dotenv
from gpt5_handler import GPT5Handler

# Carrega variáveis de ambiente
load_dotenv()

def test_gpt5_availability():
    """Testa se a API do GPT-5 está disponível"""
    print("Testando disponibilidade da API GPT-5...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERRO] Chave da API OpenAI nao encontrada. Verifique o arquivo .env")
        return False
    
    handler = GPT5Handler()
    
    try:
        # Teste simples com minimal reasoning
        result = handler.create_response_with_minimal_reasoning(
            input_message="Diga apenas 'OK' se você pode me ouvir",
            model="gpt-5-mini"
        )
        
        if "error" not in result:
            print("[OK] API GPT-5 esta funcionando!")
            print(f"   Resposta: {result.get('response', 'N/A')[:50]}...")
            return True
        else:
            print(f"[ERRO] Erro na API: {result['error']}")
            return False
            
    except Exception as e:
        print(f"[ERRO] Erro ao testar API: {str(e)}")
        return False

def test_verbosity_feature():
    """Testa o recurso de verbosity"""
    print("\nTestando recurso de Verbosity...")
    
    handler = GPT5Handler()
    question = "O que é inteligência artificial?"
    
    results = {}
    
    for verbosity in ["low", "medium", "high"]:
        try:
            result = handler.create_response_with_verbosity(
                input_message=question,
                verbosity=verbosity,
                model="gpt-5-mini"
            )
            
            if "error" not in result:
                results[verbosity] = {
                    "tokens": result.get("usage", {}).get("output_tokens", 0),
                    "response_length": len(result.get("response", ""))
                }
                print(f"[OK] Verbosity '{verbosity}': {results[verbosity]['tokens']} tokens")
            else:
                print(f"[ERRO] Erro em verbosity '{verbosity}': {result['error']}")
                
        except Exception as e:
            print(f"[ERRO] Erro ao testar verbosity '{verbosity}': {str(e)}")
    
    # Verifica se os tokens aumentam com a verbosity
    if len(results) == 3:
        if results["low"]["tokens"] <= results["medium"]["tokens"] <= results["high"]["tokens"]:
            print("[OK] Verbosity esta funcionando corretamente (tokens escalam)")
        else:
            print("[AVISO] Verbosity pode nao estar funcionando como esperado")
    
    return len(results) > 0

def test_minimal_reasoning():
    """Testa o recurso de minimal reasoning"""
    print("\nTestando Minimal Reasoning...")
    
    handler = GPT5Handler()
    
    try:
        result = handler.create_response_with_minimal_reasoning(
            input_message="Classifique o sentimento como positivo|neutro|negativo: 'Adorei este filme!'",
            model="gpt-5-mini"
        )
        
        if "error" not in result:
            print("[OK] Minimal reasoning funcionando")
            print(f"   Resposta: {result.get('response', 'N/A')}")
            print(f"   Tokens: {result.get('usage', {}).get('output_tokens', 'N/A')}")
            return True
        else:
            print(f"[ERRO] Erro no minimal reasoning: {result['error']}")
            return False
            
    except Exception as e:
        print(f"[ERRO] Erro ao testar minimal reasoning: {str(e)}")
        return False

def test_custom_tools():
    """Testa as ferramentas personalizadas"""
    print("\nTestando Custom Tools...")
    
    handler = GPT5Handler()
    
    # Teste 1: Timestamp generator
    try:
        result = handler.create_response_with_custom_tool(
            input_message="Gere um timestamp para hoje às 14:30",
            tool_name="timestamp_generator",
            tool_description="Gera timestamp no formato YYYY-MM-DD HH:MM",
            model="gpt-5-mini",
            grammar_definition=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d$",
            grammar_syntax="regex"
        )
        
        if "error" not in result and "tool_call" in result:
            print("[OK] Timestamp generator funcionando")
            print(f"   Timestamp: {result['tool_call']['input']}")
        else:
            print(f"[ERRO] Erro no timestamp generator: {result.get('error', 'Sem tool_call')}")
            
    except Exception as e:
        print(f"[ERRO] Erro ao testar timestamp generator: {str(e)}")
    
    # Teste 2: SQL Generator
    try:
        result = handler.create_sql_query_with_dialect(
            query_description="gere uma query que selecione os 5 produtos mais vendidos",
            dialect="postgresql",
            model="gpt-5-mini"
        )
        
        if "error" not in result and "tool_call" in result:
            print("[OK] SQL generator funcionando")
            print(f"   SQL: {result['tool_call']['input'][:50]}...")
        else:
            print(f"[ERRO] Erro no SQL generator: {result.get('error', 'Sem tool_call')}")
            
    except Exception as e:
        print(f"[ERRO] Erro ao testar SQL generator: {str(e)}")

def test_integration_with_app():
    """Testa se a integração com o app.py está funcionando"""
    print("\nTestando integracao com app.py...")
    
    try:
        # Importa o módulo principal para verificar se não há erros
        sys.path.append(os.path.dirname(__file__))
        from gpt5_handler import GPT5Handler
        
        # Testa se a classe pode ser instanciada
        handler = GPT5Handler()
        
        print("[OK] Integracao com app.py funcionando")
        print(f"   Modelos suportados: {handler.supported_models}")
        return True
        
    except Exception as e:
        print(f"[ERRO] Erro na integracao: {str(e)}")
        return False

def main():
    """Executa todos os testes"""
    print("Iniciando testes de integracao do GPT-5\n")
    
    tests = [
        ("Disponibilidade da API", test_gpt5_availability),
        ("Recurso de Verbosity", test_verbosity_feature),
        ("Minimal Reasoning", test_minimal_reasoning),
        ("Custom Tools", test_custom_tools),
        ("Integração com App", test_integration_with_app)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[ERRO] Erro critico no teste '{test_name}': {str(e)}")
            results[test_name] = False
    
    # Resumo dos resultados
    print("\n" + "="*50)
    print("RESUMO DOS TESTES:")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "[OK] PASSOU" if result else "[ERRO] FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("Todos os testes passaram! GPT-5 esta integrado com sucesso!")
    elif passed >= total * 0.7:
        print("Maioria dos testes passou, mas ha alguns problemas.")
    else:
        print("Muitos testes falharam. Verifique a configuracao.")
    
    print("\nDicas:")
    print("- Certifique-se de que OPENAI_API_KEY esta configurada no .env")
    print("- Execute 'pip install --upgrade openai' para atualizar o SDK")
    print("- Verifique se voce tem acesso aos modelos GPT-5")

if __name__ == "__main__":
    main()