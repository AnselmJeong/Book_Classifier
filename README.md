# Article Classifier

Targeted categories are defined in the `categories.toml` file.
Each article is classified into one of the categories by the LLM.

If you want to classify books instead of articles, you have to set the `--type` argument to `books`.

## How to use:
```
uv run classify_docs.py <root_path> # default: articles
uv run classify_docs.py <root_path> --type books
```
