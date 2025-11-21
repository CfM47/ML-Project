# ML-Project

Machine Learning Project for senior year Computer Science ML course.

![Pull Request Checks](https://github.com/CfM47/ML-Project/actions/workflows/pr-checks.yml/badge.svg)

## Data

### SEM Imgaes

Put the SEM Images dataset in the `data/sem_images/raw/` directory.

It should contain:

```
data/sem_images/raw/Brittle/{images.png...}
data/sem_images/raw/Ductile/{images.png...}
```

## Usage

This project can be run in two modes: interactive or via command-line arguments.

### Interactive Mode

The interactive mode provides a simple Text-based User Interface (TUI) to guide you through selecting and running an experiment. This is the recommended way for local development and exploration.

To start the interactive mode, run:
```bash
python main_interactive.py
```
You will be prompted to choose an approach, an action (train or validate), and a configuration file.

### Command-Line Mode

The command-line interface is ideal for automation, scripting, or running in non-interactive environments (e.g., on a server, in a Docker container, or on Google Colab/Kaggle).

The script `main.py` accepts three required arguments:

- `-a`, `--approach`: The name of the approach to run.
- `-x`, `--action`: The action to perform (`train` or `validate`).
- `-c`, `--config`: The name of the configuration file (without the `.yaml` extension).

**Example:**
```bash
python main.py --approach cnn_v1_approach --action train --config base
```
You can see all available options with the help flag:
```bash
python main.py --help
```

---

## Creating New Approaches

The project is designed to be extensible. You can add new experimental approaches, and the TUI and CLI will automatically discover them if you follow these conventions.

To create a new approach named `<my_new_approach>`:

1.  **Create a Directory**:
    Create a new directory inside `src/approaches/`:
    ```
    src/approaches/<my_new_approach>/
    ```

2.  **Add Action Scripts**:
    Inside your new directory, add at least one of the following files:
    - `train.py`: This script **must** contain a function `train(config_name: str)`.
    - `validate.py`: This script **must** contain a function `validate(config_name: str)`.

3.  **Add Configuration Files**:
    Create a `configs` subdirectory and add at least one YAML configuration file to it:
    ```
    src/approaches/<my_new_approach>/configs/base.yaml
    ```

If these steps are followed, `<my_new_approach>` will automatically appear as an option in the interactive menu and be available to the CLI.
