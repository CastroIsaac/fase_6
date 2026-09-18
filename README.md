# SCIC — Sistema de Comunicação Interplanetária da Colônia

Protótipo acadêmico da missão **Aurora Siger**. O sistema usa dados simulados para demonstrar organização de dados, análise numérica, modelo simples de previsão, métricas, heap/fila de prioridade, trie, bases numéricas, eletricidade básica e gerenciamento inteligente da comunicação.

## Estrutura
- `src/scic_aurora_siger.py` — código principal em Python.
- `dados/dados_aurora_siger.csv` — base simulada.
- `dados/dados_aurora_siger.json` — mesma base em JSON.
- `relatorio/relatorio_tecnico.pdf` — relatório técnico.
- `docs/roteiro_video.txt` — roteiro para vídeo de até 5 minutos.
- `docs/guia_apresentacao_grupo.txt` — divisão sugerida para o grupo.
- `docs/checklist_entrega.txt` — conferência final.
- `graficos/` — gráficos gerados pelo sistema.

## Execução
Python 3.10+ recomendado.

Instale:
```bash
pip install pandas numpy matplotlib scikit-learn
```

Execute a partir da raiz:
```bash
python src/scic_aurora_siger.py
```

O programa carrega a base, calcula erro absoluto/relativo e potência, treina uma regressão linear, apresenta MAE/MSE/RMSE/R²/AIC/BIC, prioriza alertas com heap, permite busca por prefixo com trie e gera gráficos.

## Observação
Os dados são **simulados**, conforme permitido no enunciado. O protótipo não depende de sensores físicos, APIs externas ou sistemas avançados de telemetria.
