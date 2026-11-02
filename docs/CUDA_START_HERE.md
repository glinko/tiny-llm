# CUDA-курс: начало работы на Ubuntu + NVIDIA RTX 3090

Здесь указан **реально подготовленный** путь прохождения Week 1 Day 1 в ветке `course/cuda`.
Оригинальная инструкция автора [book/src/setup.md](../book/src/setup.md)
относится к **Mac + MLX + Metal**, поэтому на Ubuntu не надо запускать
`pdm install` из оригинального курса: он тянет MLX-специфичные зависимости.

## 1. Получить учебную ветку

На Ubuntu-хосте с видеокартой:

```bash
git clone --branch course/cuda https://github.com/glinko/tiny-llm.git
cd tiny-llm
git branch --show-current
```

Если репозиторий уже клонирован, вместо повторного клонирования:

```bash
git fetch origin course/cuda
git switch course/cuda
```

Проверь: активная ветка `course/cuda`.

## 2. Создать отдельное Python-окружение

Рекомендуется Python 3.11 или 3.12. На примере Python 3.12:

```bash
python3.12 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
nvidia-smi
```

Если `python3.12` отсутствует, используй установленный `python3.11` и
соответствующий пакет `venv`. Не устанавливай PyTorch поверх рабочей
LLM-инфраструктуры сервера: отдельная виртуальная среда не конфликтует с ней.

Установи **CUDA-сборку PyTorch**, выбрав Linux / Pip / Python / совместимую CUDA
на [официальной странице PyTorch](https://pytorch.org/get-started/locally/).
Версию CUDA wheel выбирай под фактически установленный драйвер (`nvidia-smi`);
здесь сознательно не закреплена непроверенная версия драйвера/колеса.

После установки PyTorch:

```bash
python -m pip install -r requirements-cuda.txt
python scripts/check_cuda.py
```

Ожидаемый результат: `CUDA available: True`, GPU NVIDIA RTX 3090,
compute capability `8.6`, `GPU matmul smoke test: PASS`.

Если CUDA недоступна, **можно пока проходить Day 1 на CPU**, но перед GPU-оптимизациями
нужно исправить драйвер/колесо. Первые тесты специально запускаются и на CPU.

## 3. Проверить эталон (зелёный тест)

```bash
TINY_LLM_CUDA_SOLUTION=ref PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py
```

Если PyTorch работает корректно, тесты эталона должны пройти. Это **не** значит,
что ты выполнил задание: тест запускался по готовому `tiny_llm_cuda_ref`.

## 4. Запустить заготовку (ожидаемо красный тест)

```bash
PYTHONPATH=src python -m pytest -q tests_cuda/test_week_1_day_1.py -k task_1
```

Сейчас часть тестов должна упасть с `NotImplementedError`. Это ожидаемо:
ты ещё не заполнил учебные функции.

## 5. Прочитать и выполнить первое задание

Открой [CUDA_WEEK1_DAY1.md](CUDA_WEEK1_DAY1.md). Первое место для редактирования:

```text
src/tiny_llm_cuda/attention.py
    scaled_dot_product_attention_simple(...)
```

Там реализуй scaled dot-product attention из элементарных PyTorch операций,
**не** вызывая готовый `torch.nn.functional.scaled_dot_product_attention`.

Перезапусти тесты `-k task_1`. После их прохождения переходи к `linear`
и `SimpleMultiHeadAttention`, как описано в уроке.

## 6. Отчитаться о результате

Пришли в чат результаты `python scripts/check_cuda.py` и первого запуска
`-k task_1`, а затем твой код или текст ошибки, если потребуется разбор.

## Режим работы с Git

Все CUDA-занятия развиваются в `course/cuda`, **не** в `main`. Изменённые
файлы урока можно коммитить в эту ветку после прохождения тестов:

```bash
git status
git add src/tiny_llm_cuda/
git commit -m "learn(cuda): complete week 1 day 1"
git push origin course/cuda
```

Это только пример, коммитить до прохождения тестов не нужно. Не трогай
`src/tiny_llm/`: это оригинальные MLX-упражнения.

**Важно:** никакой серверной проверки на твоей RTX 3090 пока не выполнено.
Файлы и тесты подготовлены в GitHub, их нужно запустить на твоём Linux-хосте.
