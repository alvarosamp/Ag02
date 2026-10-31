import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import BASE, FEATURES, carregar_dados, criar_entrada, treinar


class ProjetoTest(unittest.TestCase):
    def test_dados_e_mapeamento(self):
        df = carregar_dados(BASE / 'dados' / 'wholesale.csv')
        self.assertEqual(df.shape, (440, 8))
        self.assertEqual(list(df.columns), FEATURES + ['Channel'])
        self.assertTrue(all(str(t) == 'int64' for t in df.dtypes))
        self.assertEqual(df.iloc[0]['Region'], 2)
        self.assertEqual(df.iloc[0]['Channel'], 1)
        self.assertEqual(df.Channel.value_counts().to_dict(), {0: 298, 1: 142})

    def test_entradas_invalidas(self):
        for valores in ([9, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0],
                        [0, 1.5, 0, 0, 0, 0, 0], [0, 1]):
            with self.subTest(valores=valores), self.assertRaises(ValueError):
                criar_entrada(valores)

    def test_csv_invalido(self):
        original = pd.read_csv(BASE / 'dados' / 'wholesale.csv')
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            caminho = Path(tmp) / 'invalido.csv'
            for coluna, valor in [('Channel', 'Outro'), ('Region', 'Unknown'),
                                   ('Milk', -1), ('Grocery', None)]:
                df = original.copy()
                df.loc[0, coluna] = valor
                df.to_csv(caminho, index=False)
                with self.subTest(coluna=coluna), self.assertRaises(ValueError):
                    carregar_dados(caminho)

    def test_separacao_e_persistencia(self):
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            saida = Path(tmp)
            with redirect_stdout(io.StringIO()):
                modelo = treinar(BASE / 'dados' / 'wholesale.csv', saida)
            divisao = json.loads((saida / 'divisao.json').read_text())
            treino, teste = set(divisao['treino']), set(divisao['teste'])
            self.assertEqual((len(treino), len(teste)), (352, 88))
            self.assertFalse(treino & teste)
            self.assertEqual(treino | teste, set(range(440)))
            dados = carregar_dados(BASE / 'dados' / 'wholesale.csv')
            X = dados.loc[divisao['teste'], FEATURES]
            salvo = joblib.load(saida / 'modelo.joblib')
            self.assertTrue((modelo.predict(X) == salvo.predict(X)).all())
            pred = pd.read_csv(saida / 'predicoes_teste.csv')
            self.assertTrue((salvo.predict(X) == pred.previsto.to_numpy()).all())
            self.assertEqual(salvo.n_features_in_, 7)


if __name__ == '__main__':
    unittest.main()
