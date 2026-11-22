# ДЗ 3. Обучение ранжированию

- Имя в лидерборде Kaggle — **Ravil Gareev**, логин — `raviol`.
- Публичный nDCG@10 — **0.62371** в [соревновании Kaggle](https://www.kaggle.com/competitions/learning-to-rank-homework-vk-ir-hse-fall-2026/submissions), сабмит № 58742842.
- Публичный nDCG@10 бейзлайна — **0.61621**.

Используется одна модель `CatBoostRanker` на всех 67 выданных признаках. Документы группируются по `QueryId`. Подбор и проверка гиперпараметров проводились на обучающих запросах с разбиением по `QueryId`. Итоговая модель обучена на всём `train.csv`.

| Параметр | Значение |
| --- | --- |
| `loss_function` | `YetiRank` |
| `iterations` | `900` |
| `depth` | `6` |
| `learning_rate` | `0.07` |
| `l2_leaf_reg` | `5` |
| `random_seed` | `0` |
| `thread_count` | `5` |

## Запуск

`PATH-TO-DATA` — каталог с `train.csv` и `test.csv` из соревнования. Обучение и генерация CSV выполняются на CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

./solution.py --train --model_file=model.bin PATH-TO-DATA
./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA
```
