# Announcement to JSON Extraction with Prompting

Read a short **field-trip announcement** and extract the key information as a
**JSON object**, by prompting a small LLM (**`Qwen/Qwen3-1.7B`**).

The 5 fields to extract:

| key | meaning | when missing |
|-----|---------|--------------|
| `when` | departure date/time | `null` |
| `where` | destination / place | `null` |
| `items` | list of things to bring (array) | `null` |
| `task` | activity / mission on site | `null` |
| `teacher` | name of the teacher in charge | `null` |

An announcement **may or may not** contain each piece of information. Part of the
task is to leave missing information as `null` instead of inventing it.

## Run

```bash
# full evaluation on all 100
uv run main.py

# a single strategy
uv run main.py --method naive
uv run main.py --method improved
```

## Results

| metric | naive | improved |
|---|---|---|
| JSON parse rate | 100.0% | 100.0% |
| overall field accuracy | 40.2% | 79.6% |
| items F1 | 28.7% | 87.2% |

Per-field accuracy:

| field | naive | improved |
|---|---|---|
| when | 40.0% | 76.0% |
| where | 55.0% | 74.0% |
| items | 12.0% | 81.0% |
| task | 16.0% | 69.0% |
| teacher | 78.0% | 98.0% |


## Prompting Strategy 

개선된 프롬프트에는 입력 안내문과 기대되는 JSON 출력 예시를 제공하는 one-shot prompting 방식을 사용하였다.

모델이 `when`, `where`, `items`, `task`, `teacher`의 다섯 필드만 추출하고, 유효한 JSON 객체만 출력하도록 지시하였다. 안내문에 존재하지 않는 정보는 추측하지 않고 `null`로 출력하며, `items`는 문자열 배열로 표현하도록 하였다.

또한 `task`의 정확도를 높이기 위해 활동이나 임무를 다른 표현으로 바꾸지 않고 안내문에 나온 문구 그대로 추출하도록 지시하였다.
