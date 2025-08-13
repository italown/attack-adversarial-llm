HF_TOKEN = ""

from log import Log
log = Log("logs/LOGS", "STRINGS_LLAMA")
result = Log("logs/RESULTADOS", "STRINGS")

log.header("Ataque adversarial a LLMs")
result.header("RESULTADOS")

try: 
    log.info("Importando bibliotecas. . .")

    import os
    import torch
    import nanogcg
    import pandas as pd
    from nanogcg import GCGConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Bibliotecas importadas")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Usando {device}")


    log.info("Importando dados. . .")
    df = pd.read_csv("data/harmful_strings.csv")
    messages = df['target'][:100]
    log.info("Dados importados")
    
    models = [
        'meta-llama/Llama-2-7b-chat-hf'
    ]

    for model_id in models:
        log.header(f"Modelo: {model_id}")
        result.header(f"Modelo: {model_id}")

        log.info("Importando modelo. . .")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_auth_token=HF_TOKEN
        ).to(device)
        log.info("Modelo importado")

        log.info("Importando tokenizer. . .")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
        log.info("Tokenizer importado")

        for i, message in enumerate(messages):
            log.info(f"Processando mensagem {i+1} de {len(messages)}")

            config = GCGConfig(
                num_steps=1000,
                search_width=100,
                topk=100,
                seed=42,
                early_stop=True,
                verbosity="ERROR",
            )

            result = nanogcg.run(model, tokenizer, "", message, config)
            log.info(f"Mensagem: {message}")
            log.info(f"Melhor string: {result.best_string}")
            log.info(f"Melhor score: {result.best_loss}")

            prompt = result.best_string
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()

            log.info(f"Mensagem: {prompt}")
            log.info(f"Resposta: {response}")

except Exception as e:
    log.error(f"Erro ao importar bibliotecas: {e}")
