"""AG2: treinamento, avaliação e classificação de canais de vendas."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

BASE = Path(__file__).resolve().parent
FEATURES = ['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen',
            'Detergents Paper', 'Delicatessen']
CHANNELS = {'HoReCa': 0, 'Retail': 1}
REGIONS = {'Lisbon': 0, 'Oporto': 1, 'Other': 2}
LABELS = ['HoReCa', 'Retail']


def carregar_dados(caminho):
    """Lê o CSV fornecido, valida, converte e reordena as oito colunas."""
    df = pd.read_csv(caminho)
    # O CSV do professor usa underscore; o enunciado usa espaço.
    df = df.rename(columns={'Detergents_Paper': 'Detergents Paper'})
    if set(df.columns) != set(FEATURES + ['Channel']) or len(df.columns) != 8:
        raise ValueError('O CSV deve conter exatamente os sete atributos e Channel.')
    if df.empty or df.isna().any().any():
        raise ValueError('O CSV está vazio ou contém valores ausentes.')
    for coluna, mapa in [('Channel', CHANNELS), ('Region', REGIONS)]:
        if not df[coluna].isin(mapa).all():
            raise ValueError(f'Valores desconhecidos em {coluna}: use {list(mapa)}.')
        with pd.option_context('future.no_silent_downcasting', True):
            df[coluna] = df[coluna].replace(mapa).astype('int64')
    for coluna in FEATURES[1:]:
        valores = pd.to_numeric(df[coluna], errors='raise')
        if not (np.isfinite(valores) & (valores >= 0) & (valores % 1 == 0)
                & (valores < 2**63)).all():
            raise ValueError(f'{coluna} deve conter inteiros não negativos em int64.')
        df[coluna] = valores.astype('int64')
    return df.reindex(columns=FEATURES + ['Channel'])


def criar_entrada(valores):
    if len(valores) != 7:
        raise ValueError('Informe sete inteiros: região e seis gastos anuais.')
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in valores):
        raise ValueError('Todos os valores devem ser inteiros.')
    if valores[0] not in (0, 1, 2):
        raise ValueError('Região: 0 = Lisbon, 1 = Oporto, 2 = Other.')
    if any(v < 0 or v >= 2**63 for v in valores[1:]):
        raise ValueError('Gastos devem ser inteiros não negativos em int64.')
    return pd.DataFrame([valores], columns=FEATURES, dtype='int64')


def treinar(caminho, saida):
    df = carregar_dados(caminho)
    X, y = df[FEATURES], df['Channel']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    # Configuração fixada antes de avaliar o teste. Não há ajuste pelo teste.
    modelo = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    relatorio = classification_report(y_test, pred, labels=[0, 1],
                                     target_names=LABELS, digits=4, zero_division=0)
    matriz = confusion_matrix(y_test, pred, labels=[0, 1])
    resumo = {'amostras': len(df), 'treino': len(X_train), 'teste': len(X_test),
              'random_state': 42, 'acuracia': accuracy_score(y_test, pred),
              'baseline_majoritaria': float((y_test == y_train.mode()[0]).mean()),
              'ordem_classes': LABELS, 'matriz_confusao': matriz.tolist(),
              'parametros': modelo.get_params(),
              'classification_report': classification_report(
                  y_test, pred, labels=[0, 1], target_names=LABELS,
                  output_dict=True, zero_division=0)}
    texto = (f'Amostras: {len(df)} | Treino: {len(X_train)} | Teste: {len(X_test)}\n'
             f'Acurácia: {resumo["acuracia"]:.4%}\n'
             f'Baseline (classe majoritária): {resumo["baseline_majoritaria"]:.4%}\n\n'
             + relatorio + '\nMatriz de confusão (linhas=reais; colunas=previstas):\n'
             + 'Ordem: HoReCa, Retail\n' + str(matriz) + '\n')
    saida.mkdir(parents=True, exist_ok=True)
    (saida / 'avaliacao.txt').write_text(texto, encoding='utf-8')
    (saida / 'metricas.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    (saida / 'arvore.txt').write_text(export_text(modelo, feature_names=FEATURES), encoding='utf-8')
    df.to_csv(saida / 'dados_processados.csv', index=False)
    resultados = X_test.copy()
    resultados['real'] = y_test
    resultados['previsto'] = pred
    resultados.to_csv(saida / 'predicoes_teste.csv', index_label='indice_original')
    (saida / 'divisao.json').write_text(json.dumps({
        'treino': X_train.index.tolist(), 'teste': X_test.index.tolist()}, indent=2), encoding='utf-8')
    joblib.dump(modelo, saida / 'modelo.joblib')
    print(texto)
    return modelo


class Saida(Exception):
    """Usuário pediu para encerrar a entrada interativa."""


def perguntar(texto):
    resposta = input(texto).strip()
    if resposta.lower() in ('sair', 'q'):
        raise Saida
    return resposta


def ler_regiao():
    nomes = {nome.lower(): codigo for nome, codigo in REGIONS.items()}
    while True:
        resposta = perguntar('Region [0=Lisbon, 1=Oporto, 2=Other]: ').lower()
        if resposta in nomes:
            return nomes[resposta]
        if resposta in ('0', '1', '2'):
            return int(resposta)
        print('  Valor inválido: digite 0, 1, 2 ou o nome (Lisbon, Oporto, Other).')


def ler_gasto(coluna):
    while True:
        resposta = perguntar(f'{coluna} (gasto anual em u.m.): ')
        try:
            valor = int(resposta)
        except ValueError:
            print('  Valor inválido: digite um inteiro sem separadores, ex.: 10000.')
            continue
        if not 0 <= valor < 2**63:
            print('  Valor inválido: o gasto não pode ser negativo.')
            continue
        return valor


def coletar_cliente():
    """Pergunta cada atributo; um erro repete só a pergunta afetada."""
    return [ler_regiao()] + [ler_gasto(coluna) for coluna in FEATURES[1:]]


def mostrar_previsao(modelo, valores):
    entrada = criar_entrada(valores)
    indice = int(modelo.predict(entrada)[0])
    confianca = modelo.predict_proba(entrada)[0][list(modelo.classes_).index(indice)]
    print('\nResumo do cliente:')
    for coluna, valor in zip(FEATURES, valores):
        exibido = f'{valor} ({list(REGIONS)[valor]})' if coluna == 'Region' else valor
        print(f'  {coluna:<17} {exibido}')
    print(f'\nCanal previsto: {LABELS[indice]} (confiança do modelo: {confianca:.0%})\n')


def interagir(modelo):
    print('\n=== Classificação de novo cliente ===')
    print('Digite "sair" a qualquer momento para encerrar.\n')
    try:
        while True:
            mostrar_previsao(modelo, coletar_cliente())
            if perguntar('Classificar outro cliente? [s/N]: ').lower() not in ('s', 'sim'):
                break
            print()
    except (Saida, EOFError, KeyboardInterrupt):
        pass
    print('\nEncerrado.')


def main():
    # Evita caracteres quebrados em consoles que não usam UTF-8 (ex.: Windows).
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, 'reconfigure'):
            fluxo.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, default=BASE / 'dados' / 'wholesale.csv')
    parser.add_argument('--saida', type=Path, default=BASE / 'resultados')
    parser.add_argument('--sem-interacao', action='store_true', help='Somente treinar e avaliar.')
    parser.add_argument('--prever', type=int, nargs=7, metavar='VALOR',
                        help='Região, Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicatessen.')
    args = parser.parse_args()
    try:
        entrada = criar_entrada(args.prever) if args.prever is not None else None
        modelo = treinar(args.dados, args.saida)
        if entrada is not None:
            print(f'Canal previsto: {LABELS[int(modelo.predict(entrada)[0])]}')
        elif not args.sem_interacao:
            interagir(modelo)
    except (ValueError, OSError) as erro:
        parser.exit(1, f'Erro: {erro}\n')


if __name__ == '__main__':
    main()
