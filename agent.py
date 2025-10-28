import dspy
import sqlite3
from dotenv import load_dotenv

from tools import execute_sql, get_schema, save_data_to_csv


# --- DSPy Agent Definition ---
class SQLAgentSignature(dspy.Signature):
    """
    Agente ReAct para consultar bases de datos SQLite usando lenguaje natural.
    
    TAREAS:
    - Inspeccionar el esquema de la base de datos (tablas y columnas disponibles).
    - Generar consultas SQL válidas a partir de preguntas en lenguaje natural.
    - Ejecutar las consultas SQL y analizar los resultados.
    - Interpretar los resultados y responder al usuario en lenguaje natural.
    - Exportar resultados tabulares a archivos CSV cuando el usuario lo solicite.
    
    HERRAMIENTAS DISPONIBLES:
    1. get_schema: Obtiene el esquema de una tabla específica o lista todas las tablas.
       - Entrada: table_name (str o None)
       - Salida: String con columnas/tipos o lista de tablas
    
    2. execute_sql: Ejecuta una consulta SQL (SELECT, INSERT, UPDATE, DELETE).
       - Entrada: query (str, consulta SQL válida)
       - Salida: String con resultados como lista de tuplas o mensaje de error
    
    3. save_data_to_csv: Guarda resultados tabulares en archivo CSV.
       - Entradas: data (lista de filas o string de execute_sql), filename (str)
       - Salida: Mensaje con ruta absoluta del archivo o descripción del error
    
    REGLAS Y LIMITACIONES:
    - NO inventar datos. Siempre consultar la base de datos real.
    - Si una consulta SQL falla, analizar el error y reintentar con una consulta corregida.
    - Usar get_schema antes de generar SQL si no conoces la estructura de las tablas.
    - Para exportar a CSV, primero ejecutar SELECT y luego pasar el resultado a save_data_to_csv.
    - Manejar errores de forma clara y explicar al usuario qué salió mal.
    - Máximo 7 iteraciones para resolver una consulta.
    """

    question = dspy.InputField(desc="La pregunta en lenguaje natural del usuario.")
    initial_schema = dspy.InputField(desc="El esquema inicial de la base de datos para guiarte.")
    answer = dspy.OutputField(
        desc="La respuesta final en lenguaje natural a la pregunta del usuario."
    )


class SQLAgent(dspy.Module):
    """The SQL Agent Module"""
    def __init__(self, tools: list[dspy.Tool]):
        super().__init__()
        # Initialize the ReAct agent.
        self.agent = dspy.ReAct(
            SQLAgentSignature,
            tools=tools,
            max_iters=7,  # Set a max number of steps
        )

    def forward(self, question: str, initial_schema: str) -> dspy.Prediction:
        """The forward pass of the module."""
        result = self.agent(question=question, initial_schema=initial_schema)
        return result


def configure_llm():
    """Configures the DSPy language model."""
    load_dotenv()
    llm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=4000)
    dspy.settings.configure(lm=llm)

    print("[Agent] DSPy configured with gpt-4o-mini model.")
    return llm


def create_agent(conn: sqlite3.Connection, query_history: list[str] | None = None) -> dspy.Module | None:
    if not configure_llm():
        return

    execute_sql_tool = dspy.Tool(
        name="execute_sql",
        desc=(
            "Ejecuta una consulta SQL en la base de datos SQLite."
            "Input: consulta (str, sentencia SQL valida como ser SELECT, INSERT, UPDATE, DELETE). "
            "Output: Representación en cadena de los resultados como una lista de tuplas (por ejemplo, '[(1, \"Alice\"), (2, \"Bob\")]') "
            "para consultas SELECT, o un mensaje de éxito/error para otras operaciones."
        ),
        # Use lambda to pass the 'conn' object
        func=lambda query: execute_sql(conn, query, query_history),
    )

    get_schema_tool = dspy.Tool(
        name="get_schema",
        desc=(
            "Devuelve el esquema de la base de datos para una tabla específica o todas las tablas. "
            "Input: table_name (str o None). Si es None, devuelve una lista de todos los nombres de las tablas. "
            "Si se proporciona un nombre de tabla, devuelve una cadena con los nombres y tipos de columnas para esa tabla "
            "(por ejemplo, '[('id', 'INTEGER'), ('name', 'TEXT')]')."
        ),
        # Use lambda to pass the 'conn' object
        func=lambda table_name: get_schema(conn, table_name),
    )

    save_csv_tool = dspy.Tool(
        name="save_data_to_csv",
        desc=(
            "Guarda los resultados de una consulta tabular en un archivo CSV. "
            "Inputs: "
            "  - data: lista de tuplas/listas O el string devuelto por execute_sql "
            "(por ejemplo, '[(1, \"Alice\"), (2, \"Bob\")]'). "
            "  - filename: nombre de archivo de salida deseado (str, .csv extension added automatically). "
            "Output: Mensaje de éxito con la ruta absoluta del archivo o descripción del error."
        ),
        func=save_data_to_csv
    )

    all_tools = [execute_sql_tool, get_schema_tool, save_csv_tool]     

    # 2. Instantiate and run the agent
    agent = SQLAgent(tools=all_tools)

    return agent
