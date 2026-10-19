"""
CardioIA — entrada do sistema multiagente (OpenAI Agents SDK).

Fluxo:
  Orquestrador → Analista de Risco → Especialista em Protocolos → Planejador

Uso:
  python src/main.py --demo
  python src/main.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    SQLiteSession,
    set_tracing_disabled,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT_DIR / "src" / "agents"
sys.path.insert(0, str(AGENTS_DIR))

from pipeline_context import PipelineContext
from orchestrator import orchestrator


load_dotenv(ROOT_DIR / ".env")
set_tracing_disabled(True)


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

# Caso sintético de teste da atividade (Parte 2).
DEMO_CASE = {
    "middle_ages": 62,
    "system_load": 92,
    "waiting_time": 75,
    "average_hr": 118,
    "average_spo2": 89,
    "systolic_bp": 100,
    "temperature": 37.5,
    "event_density": 7,
    "events_24h": 5,
    "resource_availability": 25,
}

# Endpoint OpenAI-compatible do Gemini (Google AI Studio).
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def collect_case_data() -> dict:
    """Coleta interativa das 10 features do caso sintético."""
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


def build_model(api_key: str) -> OpenAIChatCompletionsModel:
    """Configura o modelo via API compatível com OpenAI (Gemini)."""
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
    )
    return OpenAIChatCompletionsModel(
        model=os.getenv("CARDIOIA_MODEL", DEFAULT_MODEL),
        openai_client=client,
    )


def print_structured_response(final_output, context: PipelineContext) -> None:
    """Exibe a resposta no formato esperado pela atividade."""
    recommendation = context.final_recommendation or {}

    if hasattr(final_output, "risk_probability"):
        recommendation = {
            "risk_probability": final_output.risk_probability,
            "risk_level": final_output.risk_level,
            "suggested_protocols": [
                {"action": p.action}
                for p in getattr(final_output, "suggested_protocols", [])
            ],
            "summary": getattr(final_output, "summary", ""),
            "disclaimer": getattr(final_output, "disclaimer", ""),
        }
    elif isinstance(final_output, dict) and "risk_probability" in final_output:
        recommendation = final_output

    probability = recommendation.get("risk_probability")
    risk_level = recommendation.get("risk_level")
    protocols = [
        item.get("action", str(item))
        for item in recommendation.get("suggested_protocols", [])
    ]

    # Fallback: valores reais do contexto compartilhado (modelo + regras).
    if probability is None and context.ml_result:
        probability = context.ml_result.get("risk_probability")
        risk_level = context.ml_result.get("risk_level")

    if not protocols and context.protocol_result:
        protocols = [
            item["action"]
            for item in context.protocol_result.get("suggested_protocols", [])
        ]

    print("\n" + "=" * 50)
    print(f"RISCO PREVISTO: {float(probability):.2f}")
    if risk_level:
        print(f"CLASSIFICACAO: {risk_level}")
    print()
    print("PROTOCOLOS SUGERIDOS:")
    if protocols:
        for action in protocols:
            print(f"- {action}")
    else:
        print("- Nenhum protocolo aplicavel para este caso.")
    print("=" * 50)

    summary = recommendation.get("summary")
    if summary:
        print("\nRESUMO OPERACIONAL:")
        print(summary)

    disclaimer = recommendation.get("disclaimer")
    if disclaimer:
        print("\nAVISO:")
        print(disclaimer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CardioIA — sistema multiagente de risco cardíaco sintético"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Executa o caso sintético de exemplo da atividade",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Erro: GEMINI_API_KEY não encontrada no arquivo .env.")
        return

    model = build_model(api_key)

    if args.demo:
        case_data = DEMO_CASE.copy()
        print("\n=== CardioIA: modo demonstração ===")
        print("Usando o caso sintético padrão da atividade:\n")
        for key, value in case_data.items():
            print(f"  {key}: {value}")
    else:
        case_data = collect_case_data()

    context = PipelineContext(case_data=case_data)
    session = SQLiteSession("cardioia_session")

    print("\nExecutando o fluxo multiagente do CardioIA...\n")
    print(
        "Orquestrador -> Analista de Risco -> "
        "Especialista em Protocolos -> Planejador\n"
    )

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

    print_structured_response(result.final_output, context)

    print("\n=== VERIFICAÇÃO DO FLUXO ===")
    print("Previsão de ML obtida:", bool(context.ml_result))
    print("Análise de protocolos obtida:", bool(context.protocol_result))
    if context.ml_result:
        print("Classe prevista (ML):", context.ml_result.get("predicted_class"))
        print(
            "Probabilidade (ML):",
            context.ml_result.get("risk_probability"),
        )


if __name__ == "__main__":
    asyncio.run(main())
