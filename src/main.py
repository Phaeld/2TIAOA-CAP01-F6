
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from agents import Runner, RunConfig, SQLiteSession
from agents.extensions.models.litellm_model import LitellmModel


# Localiza os agentes do projeto.
ROOT_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT_DIR / "src" / "agents"
sys.path.insert(0, str(AGENTS_DIR))

from pipeline_context import PipelineContext
from orchestrator import orchestrator


load_dotenv(ROOT_DIR / ".env")


FEATURES = [
    "middle_ages",
    "system_load",
    "waiting_time",
    "average_hr",
    "average_spo2",
    "systolic_bp",
    "temperature",
    "event_density",
    "events_24h",
    "resource_availability",
]


def collect_case_data():
    """Collect the ten numeric features for a synthetic case."""
    case_data = {}

    print("\n=== CardioIA: dados do caso sintético ===")
    print("Informe um valor numérico para cada campo.\n")

    for feature in FEATURES:
        while True:
            raw_value = input(f"{feature}: ").strip().replace(",", ".")

            try:
                value = float(raw_value)

                if not (value == value and abs(value) != float("inf")):
                    raise ValueError

                case_data[feature] = value
                break

            except ValueError:
                print("Valor inválido. Digite um número válido.")

    return case_data


async def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Erro: GEMINI_API_KEY não encontrada no arquivo .env.")
        return

    # Modelo de linguagem usado pelos agentes.
    model = LitellmModel(
        model="gemini/gemini-3.5-flash-lite",
        api_key=api_key,
    )

    case_data = collect_case_data()
    context = PipelineContext(case_data=case_data)

    # Mantém o histórico das mensagens entre execuções.
    session = SQLiteSession("cardioia_session")

    print("\nExecutando o fluxo multiagente do CardioIA...\n")

    result = await Runner.run(
        starting_agent=orchestrator,
        input=(
            "Start the CardioIA workflow for the synthetic case stored "
            "in the shared context. Follow the configured handoffs, "
            "execute the required tools, and return the final structured "
            "recommendation. Do not invent data."
        ),
        context=context,
        session=session,
        run_config=RunConfig(model=model),
        max_turns=12,
    )

    print("=== RECOMENDAÇÃO FINAL ===")

    final_output = result.final_output

    if hasattr(final_output, "model_dump_json"):
        print(final_output.model_dump_json(indent=2))
    else:
        print(final_output)

    print("\n=== VERIFICAÇÃO DO FLUXO ===")
    print("Previsão de ML obtida:", bool(context.ml_result))
    print("Análise de protocolos obtida:", bool(context.protocol_result))


if __name__ == "__main__":
    asyncio.run(main())
