# jcp-data-manager

Package for merging JCP session data with JCP LinkedIn members.

## What it does

- Loads LinkedIn member JSON data
- Loads session JSON data
- Normalizes and merges both datasets on `user_id`
- By default, enriches rows with image-based DeepFace analysis
- By default, enriches rows with name-based gender and ethnicity predictions

## Expected input shapes

The LinkedIn file should be a top-level JSON list of member records and must include either `wordpress_user_id` or `user_id`.

The sessions file should be a top-level JSON object with a `sessions` key whose value is a list. Each session record must include at least `user_id` and `session_id`.

## Install

```bash
pip install jcp-data-manager
```

This installs the merge pipeline and the default image and name analysis dependencies.


## Example usage

```bash
pip install jcp-data-manager
```

```bash
jcp-data-manager --sessions /content/jcpst-sessions-2026-04-07-22-48-30.json --linkedin /content/linkedin-member-data-2026-04-07-224846.json --output merged.parquet
```


## Project layout

```text
src/jcp_data_manager/
  __init__.py
  cli.py
  enrichment.py
  io.py
  merge.py
```
