# Agent Coding Style Guidelines

This document outlines specific coding style conventions to be followed for all code generation and modification tasks.

## 1. Docstrings

All function and method docstrings should be written in the imperative mood.

-   **DO**: `"""Create a new user in the database."""`
-   **DON'T**: `"""Creates a new user in the database."""`
-   **DON'T**: `"""This function creates a new user..."""`

## 2. Function and Method Ordering

Code within a file should be organized in order of abstraction, from highest to lowest (top-down).

-   The main entry point or the highest-level orchestration function (e.g., `main`) should appear first.
-   Helper functions called by the main function should be defined below it.
-   Lower-level helpers called by other helpers should be defined even further down.

### Example Structure:

```python
def main():
    """Run the full process."""
    # High-level orchestration
    data = load_data()
    processed_data = process_data(data)
    save_data(processed_data)

# --- Helper Functions ---

def load_data():
    """Load data from the source."""
    # ... implementation ...

def process_data(data):
    """Process the raw data."""
    # ... implementation ...

def save_data(data):
    """Save the processed data."""
    # ... implementation ...
```

## 3. Trailing Commas

Use trailing commas in multiline lists, tuples, dictionaries, function definitions, and function calls where each item is on a new line. This improves readability and simplifies diffs in version control.

### Example:

```python
# List
items = [
    "item_a",
    "item_b",
    "item_c", # Trailing comma here
]

# Function definition
def my_function(
    param_a: int,
    param_b: str, # Trailing comma here
) -> bool:
    pass

# Function call
result = another_function(
    arg_a=1,
    arg_b="hello", # Trailing comma here
)
```

## 4. Configuration File Comments

In configuration files (e.g., `.yaml`), add comments to document the possible values for fields that have a limited, explicit set of options (i.e., enums).

### Example:

```yaml
training:
  optimizer: "Adam" # Options: "Adam", "SGD", "RMSprop"
  early_stopping:
    metric: "val_loss" # Options: "val_loss", "val_accuracy"
    mode: "min" # Options: "min", "max"
```
