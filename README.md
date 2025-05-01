
# Root Cause Analysis – OutSystems Challenge

Este projeto resolve o desafio de análise de causa-raiz proposto pela OutSystems para a posição de Senior AI Engineer.

## Objetivos

- Identificar combinações de erros e parâmetros do sistema que levam a falhas.
- Treinar modelos para prever a causa provável de uma falha.
- Fornecer interpretabilidade para auxiliar em ações preventivas.

## Estrutura

```
root_cause_analysis/
├── data/                    # Dados fornecidos
├── notebooks/
│   └── eda_and_modeling.ipynb  # Análise exploratória e modelagem
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── README.md
└── pyproject.toml
```

## Como usar

1. Instale as dependências com Poetry:

```bash
poetry install
```

2. Execute o notebook:

```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```

## Modelos

Foram comparados:
- Árvores de Decisão (simples e interpretável)
- Random Forest
- XGBoost com explicabilidade via SHAP

## Dependências

Veja `pyproject.toml`.

## Autor

Desenvolvido como parte do processo seletivo para a vaga de Senior AI Engineer na OutSystems.
