# Week 1 / Day 1: Attention и Multi-Head Attention (PyTorch/CUDA)

Это CUDA-адаптация [первого урока оригинального курса](../book/src/week1-01-attention.md).
Урок №1 **не требует скачивания Qwen3** и **не требует Triton**: сначала мы
понимаем математику attention и реализуем её простыми операциями PyTorch.

## Что ты должен понять к концу урока

1. Как Query, Key и Value преобразуют входные векторы в контекстные.
2. Почему применяется масштаб `1/sqrt(D)` и где добавляется маска.
3. Как разбить embedding на независимые attention heads и собрать их обратно.
4. Как запустить тесты и сравнить свою реализацию с независимым oracle PyTorch.

## Перед началом

Сделай [настройку окружения](CUDA_START_HERE.md) и проверь эталон:

```bash
TINY_LLM_CUDA_SOLUTION=ref PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py
```

Самостоятельную реализацию всегда проверяем **без** `TINY_LLM_CUDA_SOLUTION=ref`.

## Task 1: Scaled Dot-Product Attention

Редактируй **только учебный файл** `src/tiny_llm_cuda/attention.py`,
функцию `scaled_dot_product_attention_simple`.

Пусть `Q`, `K`, `V` имеют форму `[..., L, D]`, где `L` означает
число токенов, `D` размерность одного attention head.
Используем следующую формулу:

```text
scores  = (Q @ transpose(K)) / sqrt(D)
scores  = scores + mask   # только если передана additive mask
weights = softmax(scores, axis=-1)
output  = weights @ V
```

Маска **добавляется к scores**, а не умножается на него. Для causal mask
обычно используются нули у разрешённых позиций и `-inf` у запрещённых.

**Подсказки по PyTorch:** `torch.matmul` или оператор `@`,
`key.transpose(-2, -1)`, `query.shape[-1]`, а `softmax` уже
импортирован из `basics.py`. Если параметр `scale` задан, используй
его вместо `1/sqrt(D)`.

Функция должна работать как с `[L,D]`, так и с `[B,H,L,D]`.
Поэтому не нужно жёстко прописывать количество batch/head измерений.

**Нельзя** решать Task 1 готовой `torch.nn.functional.scaled_dot_product_attention`:
тесты используют её как эталон, но упражнение заключается в написании формулы.

Запуск только тестов первой задачи:

```bash
PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py -k task_1
```

**Критерий готовности:** все тесты Task 1 проходят на CPU, а при доступной
CUDA также на GPU, включая FP16/BF16, broadcasting, собственный `scale`
и additive mask.

### Как работает пример на двух токенах

Представь, что есть `L=2` токена. Тогда `Q @ K.T` превращается в
матрицу `2x2`, в которой элемент `[i,j]` показывает сходство
Query токена `i` с Key токена `j`. После softmax каждая строка
содержит веса важности, а умножение на `V` даёт взвешенную сумму
value-векторов для соответствующего токена. Всё, никакой магии,
только несколько матриц и много видеопамяти на больших контекстах.

## Task 2a: Linear

Открой `src/tiny_llm_cuda/basics.py` и реализуй `linear(x,w,bias)`.

- `x` имеет форму `[..., I]`.
- `w` имеет форму `[O, I]`.
- Результат имеет форму `[..., O]`.
- `bias`, когда передан, имеет форму `[O]` и прибавляется к результату.

Вычисли `x @ w.T` и при необходимости прибавь `bias`.
Не используй `torch.nn.Linear` или `F.linear` в ученическом решении.

```bash
PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py -k task_2_linear
```

## Task 2b: SimpleMultiHeadAttention

В `src/tiny_llm_cuda/attention.py` заполни
`SimpleMultiHeadAttention.__init__` и `__call__`.

Допустим, `B=2`, `L=4`, `E=8`, `H=2`. Тогда `D=E/H=4`.
Каждая проекция Q/K/V последовательно преобразуется:

```text
[B,L,E] --linear(w)--> [B,L,E]
        --reshape----> [B,L,H,D]
        --transpose--> [B,H,L,D]
        --attention--> [B,H,L,D]
        --transpose--> [B,L,H,D]
        --reshape----> [B,L,E]
        --linear(wo)--> [B,L,E]
```

Именно эти переходы нужно написать в `__call__`, переиспользуя
свои функции `linear` и `scaled_dot_product_attention_simple`.
В конструкторе сохрани `hidden_size`, `num_heads`, `head_dim`,
`scale` и матрицы `wq/wk/wv/wo`. Размер `E` должен делиться на `H`.

```bash
PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py -k task_2
```

Потом весь день:

```bash
PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py
```

**Готово, когда:** все тесты зелёные; можешь объяснить, почему
`transpose(1,2)` необходим перед attention, и почему выходная форма снова
`[B,L,E]`.

## Куда заглянуть, если застрял

- Оригинал урока: [week1-01-attention.md](../book/src/week1-01-attention.md).
- Учебная заготовка: [src/tiny_llm_cuda/attention.py](../src/tiny_llm_cuda/attention.py).
- Тесты: [tests_cuda/test_week_1_day_1.py](../tests_cuda/test_week_1_day_1.py).
- Готовая CUDA-эталонная версия: [src/tiny_llm_cuda_ref/attention.py](../src/tiny_llm_cuda_ref/attention.py).
- [PyTorch SDPA документация](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

Лучше сначала попробовать самостоятельно и сравнить свои результаты с тестами.
Эталон можно открыть, если возникла конкретная проблема. На следующем
занятии будет **RoPE**, то есть позиционное вращение Q и K.
