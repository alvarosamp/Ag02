# AG2 - Classificação de canais de vendas

Projeto da atividade de Engenharias de Computação e Software do Inatel.
Classifica clientes em **HoReCa** (hotel, restaurante ou café) ou **Retail** (varejo), usando o modelo **Decision Tree (árvore de decisão)** e sete atributos do conjunto Wholesale customers.

## Executar

**Requisito:** Python 3.11 ou superior (validado com Python 3.14, pandas 3.0.3 e scikit-learn 1.8.0).

**1º** - Baixe o projeto e entre na pasta:

```powershell
git clone https://github.com/alvarosamp/Ag02.git
cd Ag02
```

(Se recebeu um ZIP, extraia e abra o terminal na pasta que contém `main.py`.)

**2º** - Crie o ambiente virtual:

```powershell
python -m venv .venv
```

**3º** - Instale as dependências:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**4º** - Execute o programa:

```powershell
.venv\Scripts\python.exe main.py
```

**5º** - Acompanhe o resultado: o programa treina o modelo e mostra as métricas.

**6º** - Informe os dados de um cliente quando o programa pedir:

| Campo | O que digitar |
|---|---|
| Region | `0` (Lisbon), `1` (Oporto), `2` (Other) ou o nome da região |
| Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicatessen | gasto anual em u.m., inteiro não negativo, sem separador de milhares (ex.: `10000`) |

**7º** - Leia o canal previsto (**HoReCa** ou **Retail**) e a confiança do modelo. Um valor inválido repete só aquela pergunta. Digite `s` para classificar outro cliente. Digite `sair` (ou `Ctrl+C`) para encerrar.

No Linux/macOS, troque `.venv\Scripts\python.exe` por `.venv/bin/python`.

### Outras formas de executar

Somente treinamento e avaliação, sem a entrada interativa:

```powershell
.venv\Scripts\python.exe main.py --sem-interacao
```

Previsão direta pela linha de comando (valores ilustrativos), na ordem Region, Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicatessen:

```powershell
.venv\Scripts\python.exe main.py --prever 2 10000 5000 8000 1000 3000 1000
```

O resultado é uma estimativa do modelo, não uma confirmação do canal real.

## Correspondência com as nove etapas

1. **Download:** `dados/wholesale.csv` é uma cópia do CSV pré-processado vinculado pelo professor no PDF, baixada em 31/10/2026. O projeto funciona sem rede depois da instalação das bibliotecas.
2. **Leitura:** `pd.read_csv` em `carregar_dados`.
3. **Conversão:** `Series.replace` e `astype('int64')`: HoReCa → 0; Retail → 1; Lisbon → 0; Oporto → 1; Other → 2. Gastos também são validados e convertidos para int64.
4. **Ordenação:** `reindex` coloca Region e os seis gastos antes de Channel. O nome original `Detergents_Paper` é normalizado para `Detergents Paper`, conforme o enunciado.
5. **Divisão:** `train_test_split(test_size=0.20, random_state=42, stratify=y)`: 352 amostras de treino e 88 de teste. Estratificação mantém aproximadamente a proporção das classes. Channel é retirado dos atributos.
6. **Modelo:** **Decision Tree (árvore de decisão)**, com `DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)`.
7. **Treinamento e previsão:** `fit` somente no treino; `predict` no teste.
8. **Avaliação:** acurácia, `classification_report`, matriz de confusão e comparação com classe majoritária.
9. **Novas entradas:** `input` por campo, com validação, e `predict`, exibindo HoReCa ou Retail. Alternativamente, use `--prever`.

## Modelo escolhido e justificativa

**Modelo: Decision Tree (árvore de decisão)**, uma das opções permitidas pela atividade.

**Por que a árvore de decisão:**

- **Interpretável:** as regras aprendidas podem ser lidas em `resultados/arvore.txt`.
- **Critério de divisão:** separa os exemplos por limiares dos atributos, escolhidos pelo critério **Gini**.
- **Sem padronização:** **não exige** normalizar os gastos, pois não depende da escala dos atributos.
- **Controle de sobreajuste:** `max_depth=4` e `min_samples_leaf=5` limitam a complexidade e ajudam a evitar **sobreajuste**.
- **Sem ajuste pelo teste:** os parâmetros foram **fixados antes da avaliação**, sem busca baseada no conjunto de teste.

## Limites

A codificação ordinal de Region segue a exigência da atividade, embora a região seja nominal. O conjunto é pequeno e desbalanceado (298 HoReCa e 142 Retail); por isso são apresentadas métricas por classe e baseline. A avaliação usa uma única divisão fixa, não validação cruzada. O desempenho em clientes de outra população ou período não foi verificado. Não se deve escolher parâmetros por repetidas consultas ao conjunto de teste.

## Resultados reproduzidos

| Classe | Precisão | Recall | F1 | Amostras de teste |
|---|---:|---:|---:|---:|
| HoReCa | 0,9500 | 0,9500 | 0,9500 | 60 |
| Retail | 0,8929 | 0,8929 | 0,8929 | 28 |

Acurácia: **93,18% (82/88)**. Baseline: **68,18% (60/88)**.
F1 macro: **0,9214**. Há três erros em cada classe.

| Real / previsto | HoReCa | Retail |
|---|---:|---:|
| HoReCa | 57 | 3 |
| Retail | 3 | 25 |

## Arquivos gerados

- `resultados/avaliacao.txt`: relatório legível das métricas.
- `resultados/metricas.json`: métricas e parâmetros em formato estruturado.
- `resultados/dados_processados.csv`: oito colunas reordenadas, todas int64 ao carregar.
- `resultados/predicoes_teste.csv`: atributos, rótulos reais, previsões e índices originais das 88 amostras.
- `resultados/divisao.json`: índices que permitem auditar a separação treino/teste.
- `resultados/arvore.txt`: regras da árvore em texto.
- `resultados/modelo.joblib`: modelo treinado persistido. Deve ser carregado com versões compatíveis das dependências. Carregue somente arquivos joblib de origem confiável.

Cada execução treina de novo e atualiza os resultados. Caminhos alternativos podem ser informados com `--dados` e `--saida`.

Para usar o modelo salvo em Python:

```python
import joblib
from main import BASE, LABELS, criar_entrada

modelo = joblib.load(BASE / 'resultados' / 'modelo.joblib')
entrada = criar_entrada([2, 10000, 5000, 8000, 1000, 3000, 1000])
print(LABELS[int(modelo.predict(entrada)[0])])
```

Verificações automatizadas:

```powershell
.venv\Scripts\python.exe -m unittest discover -s testes -v
```

## Fontes e atribuição

- CSV fornecido na atividade: https://raw.githubusercontent.com/marcelovca90-inatel/AG2/main/wholesale.csv
- Cardoso, M. Wholesale customers. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5030X
- Dataset original: https://archive.ics.uci.edu/dataset/292/wholesale+customers
- Licença dos dados originais: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . A versão usada é a adaptação categórica disponibilizada pelo professor; o projeto normaliza o cabeçalho e converte as categorias.
- API da árvore: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- Divisão: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- Métricas: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
