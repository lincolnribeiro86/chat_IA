import google.generativeai as genai
import os

print(f"Versão da biblioteca google-generativeai: {genai.__version__}")

if hasattr(genai, 'configure'):
    print("A função 'genai.configure' FOI encontrada.")
    try:
        # Tente configurar com uma chave dummy apenas para teste,
        # ou use sua chave real se preferir.
        # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        # if GOOGLE_API_KEY:
        #     genai.configure(api_key=GOOGLE_API_KEY)
        #     print("genai.configure() executado com sucesso (usando variável de ambiente).")
        # else:
        #     print("Variável de ambiente GOOGLE_API_KEY não encontrada para teste de configure.")
        print("Não vamos executar configure() neste teste para evitar problemas com a chave, mas ela existe.")

    except Exception as e:
        print(f"Erro ao tentar usar genai.configure(): {e}")
else:
    print("ERRO: A função 'genai.configure' NÃO FOI encontrada no módulo 'genai'.")

print(f"Local do módulo genai: {genai.__file__}")
