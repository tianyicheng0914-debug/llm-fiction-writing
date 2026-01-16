# Story Engine

A two-stage fiction generation pipeline built with Streamlit and OpenRouter.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter your OpenRouter API key in the sidebar.

## How It Works

The pipeline has two stages:

**Stage 1: To-Do List Generator**
- Input: User Query + Additional Guidance
- Output: Structured story outline

**Stage 2: Story Decoder**
- Input: Editable To-Do List + Additional Guidance
- Output: Full story text

All inputs to the LLM are visible in the interface. No hidden prompts.

## Interface Layout

**Sidebar:**
- OpenRouter API Key
- Model selector (Claude 3.5 Sonnet, 4.5 Sonnet, 3 Opus)
- System prompt for Stage 1
- System prompt for Stage 2

**Main Area:**
- Stage 1: User Query, Additional Guidance, Generate button
- Stage 2: Editable To-Do List, Additional Guidance, Generate button
- Generated Story display
- Save Session button

## Editing the Code

### File Structure

```
app.py              # All application code
requirements.txt    # Dependencies (streamlit, openai)
outputs/            # Saved sessions (git-ignored)
```

### Key Sections in app.py

| Line | Section | Purpose |
|------|---------|---------|
| 15-18 | `AVAILABLE_MODELS` | Add/remove models |
| 20 | `DEFAULT_TODO_PROMPT` | Default Stage 1 system prompt |
| 22 | `DEFAULT_STORY_PROMPT` | Default Stage 2 system prompt |
| 38-50 | `call_llm()` | API call logic |
| 53-58 | `generate_todo_list()` | Stage 1 processing |
| 61-67 | `generate_story()` | Stage 2 processing |
| 119-168 | `render_sidebar()` | Sidebar UI |
| 171-255 | `render_main_interface()` | Main UI |

### Adding a New Stage

1. Add a new generation function:
```python
def generate_new_stage(client, model, system_prompt, input_text):
    return call_llm(client, model, system_prompt, input_text)
```

2. Add UI in `render_main_interface()`:
```python
st.header("Stage N: Your Stage")
input_text = st.text_area("Input", key="stage_n_input")
if st.button("Generate"):
    result = generate_new_stage(client, model_id, prompt, input_text)
```

3. Add system prompt in sidebar (in `render_sidebar()`).

### Changing Models

Edit `AVAILABLE_MODELS` dict:
```python
AVAILABLE_MODELS = {
    "Display Name": "openrouter/model-id",
}
```

## Save Format

Sessions are saved to `outputs/story_session_YYYYMMDD_HHMMSS.md` with:
- Original query
- System prompts
- Generated to-do list
- Final story

## Requirements

- Python 3.8+
- OpenRouter API key ([openrouter.ai](https://openrouter.ai))
