# Roteiro de apresentação - aproximadamente 6min30s

Divisão sugerida: integrante 1 apresenta até 3min20s; integrante 2 conclui. Ensaiem com cronômetro e expliquem com suas palavras. Limite do enunciado: 7 minutos.

## 0:00 a 0:35 - Objetivo

Apresentem os dois integrantes. Expliquem: “Nosso objetivo é prever o canal de um cliente atacadista: HoReCa, que reúne hotéis, restaurantes e cafés, ou Retail, o varejo. Usamos região e gastos anuais em seis categorias.”

Mostrem a pasta do projeto e o CSV.

## 0:35 a 1:30 - Dados e conversão

Mostrem `carregar_dados` em `main.py`. São 440 amostras e oito colunas, sendo sete atributos e um rótulo. Expliquem os mapeamentos de Channel e Region. Destaquem `replace`, `astype('int64')` e a verificação de ausências e gastos inteiros não negativos. Expliquem a normalização do nome Detergents_Paper.

## 1:30 a 2:15 - Organização e divisão

Mostrem `reindex`, a lista FEATURES e a função `treinar`. Channel fica por último no arquivo processado e é separado de X. A divisão usa 80% para treino, 20% para teste, com 352 e 88 amostras. `random_state=42` torna a divisão reproduzível; `stratify` preserva aproximadamente a proporção das classes.

## 2:15 a 3:20 - Modelo

Mostrem `DecisionTreeClassifier`, `fit` e `predict`. Expliquem que a árvore aprende regras de decisão a partir do treino. A profundidade máxima é quatro, e cada folha tem pelo menos cinco amostras. Esses limites reduzem complexidade. O teste fica separado do aprendizado e não foi usado para escolher parâmetros. Mostrem uma regra real do arquivo `resultados/arvore.txt` e expliquem o limiar observado, sem decorar números.

## 3:20 a 4:35 - Avaliação

Executem `python main.py --sem-interacao`. Mostrem que o modelo acerta 82 das 88 amostras, ou 93,18%. Prever sempre a classe majoritária acertaria 68,18%. Expliquem precisão (entre previsões de uma classe, quantas acertam), recall (entre exemplos reais da classe, quantos são encontrados) e F1 (média harmônica das duas métricas).

Leiam a matriz: 57 HoReCa e 25 Retail corretos; três de cada classe foram confundidos com a outra. Os números descrevem apenas esta divisão de teste.

## 4:35 a 5:40 - Demonstração interativa

Executem `python main.py`. Após as métricas, insiram os sete valores de um cliente hipotético: região 2 e gastos 10000, 5000, 8000, 1000, 3000 e 1000. Mostrem o canal que o programa imprimir. Esse cliente não tem rótulo real conhecido: a saída é uma previsão.

Se houver tempo, tentem região 9 para mostrar a validação e preencham novamente. Mostrem que é possível classificar outro cliente ou sair.

## 5:40 a 6:30 - Arquivos e conclusão

Mostrem o relatório, o modelo salvo e o CSV de previsões. Expliquem que a base é pequena e desbalanceada, e que clientes diferentes podem ter outro desempenho. Uma melhoria futura seria validação cruzada apenas no treino para escolher parâmetros, mantendo o teste reservado até o final. Confirmem que as nove etapas foram implementadas e citem a origem UCI e o CSV fornecido pelo professor.

Antes de enviar: gravem a tela com áudio legível, confiram a duração, incluam ambos os integrantes e verifiquem se o professor consegue acessar vídeo e ZIP.
