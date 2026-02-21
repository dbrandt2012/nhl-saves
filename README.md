# nhl-saves

Data application to track NHL goalie saves historically and project saves for the next game based on historical trends.

## Setup

```bash
uv sync
```

## Run

```bash
uv run streamlit run src/nhl_saves/main.py
```

## Development

```bash
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format
```
