import psycopg2
from psycopg2 import sql
import json
import os
from datetime import datetime
import uuid

# Carrega variáveis de ambiente, se existirem (para credenciais do banco de dados)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Configuração do banco de dados PostgreSQL
# É altamente recomendável usar variáveis de ambiente para credenciais de produção
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


def connect_db():
    """
    Conecta ao banco de dados PostgreSQL usando as variáveis de ambiente ou padrões.
    Retorna o objeto de conexão ou None em caso de erro.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        # Em um ambiente de produção, você pode querer registrar este erro
        return None


def create_tables():
    """
    Cria a tabela 'conversations' no banco de dados se ela ainda não existir.
    Esta tabela armazenará os detalhes da conversa, incluindo mensagens,
    modelo usado, título e timestamps.
    """
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # Cria a tabela 'conversations' com um ID UUID padrão
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(255) NOT NULL,
                    model_used VARCHAR(255),
                    messages JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                -- Adiciona uma função para atualizar automaticamente a coluna 'updated_at'
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                -- Remove o trigger existente antes de criar um novo para evitar duplicatas
                DROP TRIGGER IF EXISTS update_conversations_updated_at ON conversations;
                -- Cria um trigger que executa a função acima antes de cada atualização
                CREATE TRIGGER update_conversations_updated_at
                BEFORE UPDATE ON conversations
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """)
            conn.commit()  # Confirma as alterações no banco de dados
            cur.close()
            print("Tabela 'conversations' verificada/criada com sucesso.")
        except Exception as e:
            print(f"Erro ao criar tabelas: {e}")
        finally:
            conn.close()


def save_conversation(messages, model_used, title="Nova Conversa"):
    """
    Salva uma nova conversa no banco de dados.
    Recebe uma lista de mensagens, o modelo usado e um título opcional.
    Retorna o ID da nova conversa ou None em caso de erro.
    """
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # Garante que o título não exceda o limite de 255 caracteres
            if len(title) > 255:
                title = title[:252] + "..."

            # Converte a lista de mensagens (formato Langchain/Streamlit) para JSON
            messages_json = json.dumps(messages)

            # Insere a nova conversa e retorna o ID gerado
            cur.execute(
                """
                INSERT INTO conversations (title, model_used, messages)
                VALUES (%s, %s, %s) RETURNING id;
                """,
                (title, model_used, messages_json)
            )
            # Obtém o ID da conversa recém-inserida
            thread_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            print(f"Nova conversa salva com ID: {thread_id}")
            return str(thread_id)  # Retorna o UUID como string
        except Exception as e:
            print(f"Erro ao salvar nova conversa: {e}")
            return None
        finally:
            conn.close()


def update_conversation(thread_id, messages, model_used):
    """
    Atualiza uma conversa existente no banco de dados.
    Recebe o ID da conversa, a lista atualizada de mensagens e o modelo usado.
    Retorna True se a atualização for bem-sucedida, False caso contrário.
    """
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # Converte a lista de mensagens atualizada para JSON
            messages_json = json.dumps(messages)

            # Atualiza as mensagens e o modelo usado para a conversa especificada
            cur.execute(
                """
                UPDATE conversations
                SET messages = %s, model_used = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """,
                (messages_json, model_used, thread_id)
            )
            conn.commit()
            cur.close()
            print(f"Conversa com ID {thread_id} atualizada.")
            return True
        except Exception as e:
            print(f"Erro ao atualizar conversa {thread_id}: {e}")
            return False
        finally:
            conn.close()


def load_conversation(thread_id):
    """
    Carrega uma conversa do banco de dados pelo seu ID.
    Retorna um dicionário contendo o título, modelo usado e mensagens,
    ou None se a conversa não for encontrada ou ocorrer um erro.
    """
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # Seleciona o título, modelo e mensagens da conversa
            cur.execute(
                """
                SELECT title, model_used, messages
                FROM conversations
                WHERE id = %s;
                """,
                (thread_id,)
            )
            result = cur.fetchone()  # Obtém o primeiro resultado
            cur.close()
            if result:
                title, model_used, messages = result
                # Converte o JSON de volta para lista de mensagens
                # messages = json.loads(messages_json)
                return {"title": title, "model_used": model_used, "messages": messages}
            return None
        except Exception as e:
            print(f"Erro ao carregar conversa {thread_id}: {e}")
            return None
        finally:
            conn.close()


def list_conversations():
    """
    Lista todas as conversas salvas no banco de dados, ordenadas pela data de atualização.
    Retorna uma lista de dicionários, cada um contendo o ID, título e data de atualização
    de uma conversa.
    """
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # Seleciona ID, título e data de atualização, ordenando pelas mais recentes
            cur.execute(
                """
                SELECT id, title, updated_at
                FROM conversations
                ORDER BY updated_at DESC;
                """
            )
            results = cur.fetchall()  # Obtém todos os resultados
            cur.close()
            # Formata os resultados em uma lista de dicionários
            return [{"id": str(row[0]), "title": row[1], "updated_at": row[2].strftime("%Y-%m-%d %H:%M")} for row in results]
        except Exception as e:
            print(f"Erro ao listar conversas: {e}")
            return []
        finally:
            conn.close()


def delete_conversation(thread_id):
    """
    Exclui uma conversa do banco de dados pelo seu ID.
    Retorna True se a exclusão for bem-sucedida, False caso contrário.
    """
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            # Executa a exclusão da conversa
            cur.execute(
                """
                DELETE FROM conversations
                WHERE id = %s;
                """,
                (thread_id,)
            )
            conn.commit()
            cur.close()
            print(f"Conversa com ID {thread_id} excluída.")
            return True
        except Exception as e:
            print(f"Erro ao excluir conversa {thread_id}: {e}")
            return False
        finally:
            conn.close()


# Inicializa as tabelas ao importar o módulo pela primeira vez
# Isso garante que a tabela 'conversations' exista quando o aplicativo iniciar
create_tables()
