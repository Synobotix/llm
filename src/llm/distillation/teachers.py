from llm.distillation.openrouter import generate_response


TEACHERS = ["deepseek/deepseek-v4.1-flash"]
# Autres teachers possibles (OpenRouter accepte 3 modèles maximum) :
#   deepseek/deepseek-v4-flash-0731
#   nvidia/nemotron-3-ultra-550b-a55b:free
#   deepseek/deepseek-v4-flash
#   moonshotai/kimi-k3
#   deepseek/deepseek-v4-pro-0813
#   deepseek/deepseek-v4-pro
#   qwen/qwen3.8-flash
#   nvidia/nemotron-3.5-lightning:free
#   qwen/qwen3.7-flash
#   qwen/qwen3.8-27b
#   deepseek/deepseek-v3.2
#   nvidia/nemotron-3-super-120b-a12b:free
#   mistralai/mistral-nemo
#   qwen/qwen3.8-max-0902
#   moonshotai/kimi-k2.6
#   deepseek/deepseek-v4-flash-vision-exp
#   nvidia/nemotron-3.5-lightning
#   qwen/qwen3.6-35b-a3b
#   qwen/qwen3.7-plus
#   qwen/qwen3-235b-a22b-2507


def generate_with_teachers(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> tuple[str, str]:
    return generate_response(
        models=TEACHERS,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
