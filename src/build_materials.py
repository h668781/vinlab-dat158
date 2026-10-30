"""Generate a runnable notebook from measured results.

Only NumPy is needed. The notebook cells are executed here to capture their
real stdout; the resulting standard ipynb can also be run in Jupyter/Colab.
"""
import contextlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []
    namespace = {'__name__': '__notebook__'}
    count = 0

    def markdown(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})

    def code(text):
        nonlocal count
        count += 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(text, '<notebook>', 'exec'), namespace)
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(True),
                      'execution_count': count, 'outputs': [{'output_type': 'stream',
                      'name': 'stdout', 'text': output.getvalue().splitlines(True)}]})

    markdown('# Vinlab\n\nDAT158-prosjekt. Denne notebooken viser data, trening og feilanalysen. '
             'AI-bruk er oppgitt i rapporten og docs/AI_BRUK.md. Køyr alle cellene i rekkjefølgje.\n\n'
             'Lokalt: installer requirements.txt og start Jupyter frå prosjektmappa. '
             'I Colab: last opp prosjektet og set arbeidsmappa til vinlab før du køyrer. '
             'Alle modellane er implementerte i src/models.py med NumPy, utan scikit-learn.')
    code("from pathlib import Path\nimport sys, csv, json\nimport numpy as np\n"
         "ROOT = Path.cwd()\nif not (ROOT / 'data/winequality-red.csv').exists():\n"
         "    raise FileNotFoundError('Start notebooken frå vinlab-mappa.')\n"
         "sys.path.insert(0, str(ROOT / 'src'))\nfrom models import *\n"
         "print('NumPy:', np.__version__)\n")
    markdown('## 1 Problem og metrikkar\n\nMålet er å estimere smakspoeng frå elleve kjemiske målingar. '
             'Dette er regresjon. MAE er hovudmetrikken; RMSE, R² og delen innanfor eitt poeng gir utfyllande informasjon. '
             'Låg MAE er bra. Ein høg del innanfor eitt poeng kan også kome frå å alltid gjette ei vanleg vurdering.')
    code("with (ROOT / 'data/winequality-red.csv').open() as f:\n"
         "    reader = csv.reader(f, delimiter=';')\n    columns = next(reader)\n"
         "    raw = np.array(list(reader), dtype=float)\n"
         "_, kept = np.unique(raw[:, :-1], axis=0, return_index=True)\n"
         "data = raw[np.sort(kept)]\n"
         "print('Originale rader:', len(raw), 'Unike målingar:', len(data))\n"
         "print('Manglande/ikkje-endelege verdiar:', np.sum(~np.isfinite(raw)))\n"
         "print('Kvalitet  Tal')\n"
         "for value, n in zip(*np.unique(data[:, -1], return_counts=True)):\n"
         "    print(f'{int(value):8} {n:4}')\n"
         "print('Felt                         Min      Median       Maks')\n"
         "for i, field in enumerate(columns[:-1]):\n"
         "    print(f'{field:25} {data[:,i].min():8.4f} {np.median(data[:,i]):10.4f} {data[:,i].max():10.4f}')\n")
    markdown('## 2 Trening og validering\n\nIdentiske målingar blir fjerna før oppdeling. '
             'Seed 158 gir faste 60/20/20-delar. Skalering for ridge blir tilpassa berre på treningsdelen. '
             'Modellvalet skjer på validerings-MAE. Treningsmodellen blir frosen utan refit før endeleg test. '
             'Neste celle køyrer heile forsøket; han skriv også oppdaterte JSON-filer til nettsida.')
    code("from train import main\nmain()\n")
    code("report = json.loads((ROOT / 'results/metrics.json').read_text(encoding='utf-8'))\n"
         "print('Modell                 Trening MAE  Val MAE  Val RMSE  Val R2  Val innanfor 1')\n"
         "for run in report['candidates']:\n"
         "    v = run['validation']\n"
         "    print(f\"{run['name']:22} {run['train']['mae']:10.3f} {v['mae']:8.3f} {v['rmse']:9.3f} {v['r2']:7.3f} {100*v['within_one']:12.1f}%\")\n"
         "print('Vald modell:', report['selected'])\n"
         "print('Test:', json.dumps(report['test'], indent=2))\n"
         "print('Baseline test:', json.dumps(report['baseline_test'], indent=2))\n")
    markdown('## 3 Feilanalyse\n\nSjå spesielt på dei sjeldne kvalitetane. '
             'Små utval gir usikre mål. Permutasjonsanalysen under bruker berre valideringsdata. '
             'Han viser prediktiv nytte, ikkje årsakssamanhengar.')
    code("print('Testfeil fordelt på faktisk kvalitet:')\n"
         "for row in report['quality_slices']:\n    print(row)\n"
         "print('Permutasjonsviktigheit: auke i MAE på validering, fem repetisjonar')\n"
         "for row in report['permutation_importance_validation']:\n    print(row)\n"
         "print('Ti største absolutte testfeil:')\n"
         "for row in report['worst_errors']:\n    print(row)\n")
    markdown('## 4 Drift\n\nNettsida i dist/ lastar model.json og bruker same tersklar og rekkjefølgje på '
             'input-felta som Python-modellen. Køyr `python -m http.server 8000 --directory dist` og opne '
             '`http://localhost:8000`. JavaScript-paritet kan kontrollerast med `node tests/test_export.cjs`. '
             'Sjå rapporten og README for publisering, kjelder og vidare arbeid.\n\n'
             'Data: Cortez mfl. (2009), UCI Wine Quality, https://doi.org/10.24432/C56S3T (CC BY 4.0).')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.14'}},
                'nbformat': 4, 'nbformat_minor': 4}
    (ROOT / 'prosjekt.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    os.chdir(ROOT)
    build_notebook()
