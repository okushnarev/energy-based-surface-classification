# Energy-Based Surface Classification for Mobile Robots in Known and Unexplored Terrains

This repository contains the source code and data for the paper **"Energy-Based Surface Classification for Mobile Robots in Known and Unexplored Terrains"** by Alexander Belyaev and Oleg Kushnarev.

The project presents an adaptive system for classifying terrain types using proprioceptive energy consumption data from a mobile robot's motors. An energy coefficient is introduced to quantify motion effort, and its dependency on motion direction is modeled using a Discrete Cosine Transform (DCT) for known surfaces. The system can classify known surfaces, detect previously unknown terrains using a neural network, and identify the parameters of these new surfaces to adapt to new environments.

## Key Features

-   **Energy Coefficient Calculation**: Utilizes motor current and velocity to quantify motion effort.
-   **Surface Modeling**: Employs Discrete Cosine Transform (DCT) to model the relationship between the energy coefficient and the robot's direction of motion for different surfaces.
-   **Probabilistic Surface Classification**: A probabilistic classifier with memory identifies known surfaces by comparing real-time energy data against pre-trained models.
-   **Unexplored Terrain Detection**: A neural network-based detector identifies significant deviations from known models, flagging encounters with new surfaces with 96.2% accuracy.
-   **New Surface Identification**: A least-squares method identifies the model parameters for newly detected surfaces, allowing the system to learn and adapt with high accuracy (MAPE <3%).

## Repository Structure

The project is organized to separate data, source code, and generated outputs.

```
.
├── README.md                 <- You are here.
├── data/                     <- All project data.
│   ├── raw/                  <- Raw experimental data from the robot.
│   ├── prepared/             <- Filtered and processed data ready for analysis.
│   ├── detection/            <- Datasets for training the unexplored surface detector.
│   ├── evaluation/           <- Results from classification and detection evaluations.
│   └── id_directions/        <- Data related to the search for optimal identification directions.
│
├── figures/                  <- Generated plots and visualizations.
│
├── ml_models/                <- Saved trained machine learning models (e.g., the NN detector).
│
├── python/                   <- Main source code directory.
│   ├── step_0_...py          <- Sequenced scripts to reproduce the paper's results.
│   └── utils/                <- Helper modules for filtering, DCT, models, etc.
│
├── pyproject.toml            <- Project dependencies for installation.
└── uv.lock                   <- Lock file for reproducible builds with `uv`.
```

## Methodology Workflow

The paper's methodology is implemented as a series of Python scripts designed to be run sequentially. This allows for a clear and reproducible workflow from raw data to final results.

1.  **Data Preparation ([step_0_prepare_data.py](python/step_0_prepare_data.py))**: Raw sensor data from `data/raw/` is loaded and filtered using a Kalman filter to reduce noise. The processed data is saved to `data/prepared/`.

2.  **Data Visualization ([step_1_visualize_data.py](python/step_1_visualize_data.py))**: Generates initial plots to visualize the relationship between the energy coefficient and motion direction for different surfaces.

3.  **DCT Model Generation ([step_2_DCT_models.py](python/step_2_DCT_models.py))**: Applies a Discrete Cosine Transform to the filtered data to create analytical models for each known surface (gray, green, table). The models are saved to `data/dct_models.pkl`.

4.  **Known Surface Classifier ([step_3_create_and_test_surface_classifier.py](python/step_3_create_and_test_surface_classifier.py))**: Builds and evaluates the probabilistic classifier on test trajectories (circle, square) to verify its accuracy on known surfaces.

5.  **Detector Dataset Creation ([step_4_create_detection_dataset.py](python/step_4_create_detection_dataset.py))**: Creates a balanced dataset to train a binary classifier to distinguish between known and unknown surfaces.

6.  **Detector Training ([step_5_train_detector_nn.py](python/step_5_train_detector_nn.py), [step_6_train_detector_other_models.py](python/step_6_train_detector_other_models.py))**: Trains various machine learning models, including the primary neural network, for the detection task. Trained models are saved in `ml_models/`.

7.  **Detector Evaluation ([step_7_evaluate_detectors.py](python/step_7_evaluate_detectors.py))**: Evaluates the trained detectors for accuracy and F-score on a held-out test set.

8.  **New Surface Identification ([step_8_search_identification_directions.py](python/step_8_search_identification_directions.py), [step_9_search_identification_time.py](python/step_9_search_identification_time.py))**: Runs simulations to find the optimal set of motion directions and the minimum data collection time required to accurately identify a new surface's parameters.

9.  **Final Visualization ([step_10_plot_identified_surfaces.py](python/step_10_plot_identified_surfaces.py))**: Generates plots comparing the identified models for a new surface against the ground-truth reference models.

## Getting Started

### Prerequisites

-   Python 3.13+
-   A Python package manager such as `uv`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/okushnarev/energy-based-surface-classification.git
    cd energy-based-surface-classification
    ```

2.  **Create a virtual environment and install dependencies:**

    *Using `uv` (recommended):*
    ```bash
    uv sync
    ```
    
3. **Download dataset from https://zenodo.org/records/17165944 with script [download_dataset.sh](download_dataset.sh):**
    ```bash
   bash download_dataset.sh https://zenodo.org/records/17165944
    ```

### Usage

The core logic is contained in the numbered scripts inside the `python/` directory. Run them sequentially to reproduce the entire pipeline from the paper.

1. _As a module_
    ```bash
    uv run -m python.step_0_prepare_data
    ```
2. _With PYTHONPATH variable_

    ```bash
    export PYTHONPATH=.
    ```
   
    ```bash
    uv run python/step_0_prepare_data.py
    ```

You can use virtual environment directly as usual via:

```bash
source .venv/bin/activate
python python/step_0_prepare_data.py
```

