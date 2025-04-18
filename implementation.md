# Implementation Plan: Routing Weight Optimization RL Agent

## 1. Goal

Implement an offline reinforcement learning agent (using CQL and d3rlpy) based on the strategy outlined in `initial_plan.md` to learn optimal routing weights from the provided dataset, aiming to minimize DRC violations and routing iterations.

## 2. Phase 1: Dataset Preparation (`MDPDataset` Creation)

**Priority:** Highest. This is the prerequisite for any model training.

**Objective:** Create a Python script (e.g., `src/data_processing/create_dataset.py`) that processes the raw `trainingdata_*.csv` files and generates a final `MDPDataset` object suitable for d3rlpy, saved to a file (e.g., `data/routing_dataset.h5` or `.pkl`).

**Detailed Steps:**

1.  **[TODO] Script Setup:**
    *   Import necessary libraries (`pandas`, `numpy`, `glob`, `os`, `sklearn.preprocessing`, `d3rlpy`).
    *   Define constants for input data directory (`../training_data`), output file path, known bad iterations, duplicate handling parameters, reward calculation parameters (`STUCK_WINDOW`, penalties, bonuses), normalization methods, etc.
    *   Implement argument parsing (e.g., using `argparse`) for input/output paths, processing options.

2.  **[TODO] Data Loading & Initial Cleaning:**
    *   Implement a function to load multiple CSVs. Consider memory efficiency (process file-by-file or use chunking if necessary, although file-by-file seems more manageable for aggregation).
    *   **Filter Missing Data:** Immediately filter out the 6 known iterations identified in `initial_plan.md` (those with 100% missing `drv`/`wireLength`).
    *   **Handle Duplicates:** Implement the chosen strategy (e.g., `df.drop_duplicates(subset=['uniqueID', 'iteration', 'boxID'], keep='first')`) specifically for the `trainingdata_bp_be_base.csv` file *before* aggregation. Log the action.
    *   Convert necessary columns to numeric types (`pd.to_numeric` with `errors='coerce'`). Log any coercion errors.

3.  **[TODO] Iteration Aggregation:**
    *   Group data by `uniqueID` and `iteration`.
    *   For each group, calculate and store iteration-level metrics:
        *   `drv` (verify consistency)
        *   `wireLength` (verify consistency)
        *   Weights (`drc_weight`, etc. - verify consistency)
        *   `pin_count`, `net_count` (verify consistency)
        *   `average_box_drv = mean(box_drv)`
        *   `max_box_drv = max(box_drv)`
        *   `num_violating = count(box_drv > 0)`
        *   `non_zero_drv_box_percentage`
        *   `drv_difference = total_box_drv - drv`
    *   Store these aggregated results (e.g., in a new DataFrame indexed by `uniqueID`, `iteration`).

4.  **[TODO] Feature Engineering (Lagging for Rewards/State):**
    *   Sort the aggregated DataFrame by `uniqueID`, `iteration`.
    *   Create lagged columns needed for state history and reward calculation (`drv_t-1`, etc.) using `groupby('uniqueID').shift()`.
    *   **Do NOT filter rows based on missing lagged or next values here.** All iteration rows are kept.

5.  **[TODO] Reward Calculation:**
    *   Calculate reward components based on the difference between iteration `t` and iteration `t-1` metrics.

    *   `primary_reward`
        *   `box_reward`
        *   `num_violating_reward`
        *   `stuck_penalty` (using `drv`, `drv_t-1`, `drv_t-2`)
        *   `convergence_bonus` (based on `next_drv == 0`)


    *   Handle `NaN` values arising from shifts at the start of trajectories (e.g., `drv_lag_1` will be NaN for the first state). These NaNs in *reward components* for the first step are expected and should be handled (e.g., treated as 0 or filtered out *after* state/action pairs are formed).

6.  **[TODO] Reward Normalization (Recommended):**
    *   Calculate mean/std dev for reward components across all *valid* calculated values (excluding NaNs from the first step).
    *   Apply normalization.

7.  **[TODO] Final Reward Combination:**
    *   Combine normalized components. The first reward value in each trajectory might still be NaN if components weren't defined.
    *   **Impute NaN `final_reward` values (corresponding to the first transition) with 0.0.**

8.  **[TODO] State Construction:**
    *   For each valid transition step `t`, construct the state vector `s_t` using features from iteration `t-1`.
    *   Include historical DRV values `drv_{t-1}` to `drv_{t-k}`.
    *   **If any historical `drv_lag_k` value is NaN (due to being near the start of a trajectory), replace it with -1.**
    *   Construct the next state vector `s_{t+1}` using features from iteration `t`, applying the same padding logic for its historical components.
    *   **Note:** `total_box_drv` feature removed from state.
    *   **Note:** `average_box_drv` added to state.

9.  **[TODO] State Normalization:**
    *   Collect all state vectors (`s_t`).
    *   Fit a scaler (e.g., `sklearn.preprocessing.RobustScaler` is recommended due to potential outliers/skewness) on the *entire* state dataset.
    *   Apply the fitted scaler to transform all state vectors (`s_t`, `s_{t+1}`).
    *   Save the fitted scaler object for later use during inference (`joblib.dump`).

10. **[DONE] Action & Terminal Extraction:**
    *   Extract the action vector `a_t` (the four weights used in iteration `t`).
    *   Determine the terminal flag `terminal_t` based on the *next* state's DRV (`next_drv == 0`).

11. **[TODO] Assemble & Save Dataset:**
    *   Gather the final lists/arrays of normalized states, actions, combined (and imputed) rewards, and terminals.
    *   **No filtering based on NaN rewards needed as they have been imputed.**
    *   Calculate and save action min/max statistics (`action_min_max.npz`).
    *   Create the `d3rlpy.dataset.MDPDataset` object.
    *   Save the dataset object.

## 3. Phase 2: Initial Model Training & Pipeline Validation

**Priority:** Medium (after dataset creation).

**Objective:** Verify the dataset loading and basic CQL training pipeline works.

**Detailed Steps:**

1.  **[TODO] Basic CQL Training Script (`train_cql.py`):**
    *   Load the saved `MDPDataset`.
    *   Load action min/max statistics from `action_min_max.npz`.
    *   Initialize `d3rlpy.preprocessing.MinMaxActionScaler` with loaded statistics.
    *   Initialize a `d3rlpy.algos.CQLConfig` passing the `MinMaxActionScaler` to `action_scaler`. Optionally configure `reward_scaler`.
    *   Create the `CQL` agent: `cql = config.create(...)`.
    *   Set the random seed globally using `d3rlpy.seed(seed_value)`.
    *   Run `cql.fit()` passing the loaded `ReplayBuffer` object and necessary arguments (`n_steps`, `experiment_name`, `evaluators`, etc.). Remove `logdir` argument from `fit` call.
    *   Monitor output for errors, Q-value progression, loss curves.
    *   Save the trained model parameters checkpoint using `cql.save_model()`.
    *   Export the inference policy using `cql.save_policy()` to TorchScript (`policy.pt`) and ONNX (`policy.onnx`).
2.  **[TODO] Basic Decision Transformer Training Script (`train_dt.py`):**
    *   **Copy and adapt `train_cql.py`.**
    *   Load the saved `MDPDataset` (same as CQL).
    *   Load action min/max statistics and initialize `MinMaxActionScaler`.
    *   Import `DecisionTransformerConfig`, `DecisionTransformer`, `GPTAdamWFactory`.
    *   **Determine `max_timestep`:** Analyze the processed data (before `MDPDataset` creation, e.g., the aggregated iteration DataFrame) to find the maximum `iteration` value within any `uniqueID`. Add this calculation to `create_dataset.py` or analyze offline.
    *   Initialize a `d3rlpy.algos.DecisionTransformerConfig`, passing necessary arguments:
        *   `context_size` (e.g., 20)
        *   `max_timestep` (determined above)
        *   `action_scaler`
        *   `reward_scaler` (optional)
        *   `optim_factory=GPTAdamWFactory()`
        *   Learning rate, warmup tokens, transformer architecture parameters (start with defaults).
    *   Create the `DT` agent: `dt = config.create(...)`.
    *   Run `dt.fit()` using the same dataset and similar arguments as CQL (`n_steps`, `experiment_name`, etc.). Use DT-specific evaluators if available, or common ones like `InitialStateValueEstimationEvaluator`.
    *   Monitor output.
    *   Save the trained model (`dt.save_model()`).
    *   Export the inference policy (`dt.save_policy()` for TorchScript/ONNX).

## 4. Phase 3: Hyperparameter Tuning & Experimentation

**Priority:** Medium-High (Iterative process after initial training).

**Objective:** Find optimal hyperparameters for the CQL and DT agents and reward function.

**Detailed Steps:**

1.  **[TODO] Experimentation Framework:**
    *   Choose and set up an experiment tracking tool (MLflow, W&B, or custom logging).
    *   Refine `train_cql.py` and `train_dt.py` to accept hyperparameters as arguments.
    *   Define ranges and strategies for tuning:
        *   **CQL:** `conservative_weight`, learning rates (actor, critic, temperature, alpha_lr).
        *   **DT:** `context_size`, `max_timestep` (if analysis suggests variability), learning rate, learning schedule (`warmup_tokens`, `final_tokens`), transformer architecture (`num_heads`, `num_layers`).
        *   **Common:** Reward weights (β₂, β₃, β₄), Stuck penalty parameters (`penalty_value`, `ε_stuck`, `m`), Convergence bonus value, Normalization methods, Action normalization.
2.  **[TODO] Run Tuning Experiments:** Execute training runs with different hyperparameter combinations for both CQL and DT.
    *   **Update (CQL):** Initial runs using `conservative_weight=1.0` (previously found best on filtered data) with default LRs (actor=3e-5, critic=3e-4) on the complete, correctly processed dataset resulted in training instability (exploding losses). **Priority should be given to tuning `conservative_weight` and significantly reducing learning rates for CQL.**
    *   **DT:** Start with default DT hyperparameters and tune `context_size`, learning rate schedule first.

## 5. Phase 4: Evaluation

**Priority:** Medium-High (Parallel with tuning).

**Objective:** Assess the performance of trained models (CQL and DT).

**Detailed Steps:**

1.  **[TODO] Evaluation Script (`evaluate_agent.py`):**
    *   Load a trained model checkpoint (CQL or DT) and the state scaler.
    *   **Adapt for DT:** DT evaluation ideally requires simulating interaction or using specialized offline metrics.
    *   **Offline Metrics:** Use d3rlpy metrics. `InitialStateValueEstimationEvaluator` should work for both. `TDErrorEvaluator` is less meaningful for DT. Need to investigate DT-specific offline evaluation metrics if available in d3rlpy or literature.
    *   **Simulation (If feasible):** The simulation logic needs to be adapted for DT. It must:
        *   Maintain a history of states, actions, and target returns up to `context_size`.
        *   Set a `target_return_to_go` for the DT policy at each step (this is a key design choice - e.g., use the dataset's average return, max return, or a desired target).
        *   Feed the sequences and target RTG to the loaded DT policy.
    *   Define metrics: Average iterations to convergence, final DRV, comparison to baseline and CQL.
2.  **[TODO] Run Evaluations:** Evaluate promising models from the tuning phase for both CQL and DT.

## 6. Phase 5: Inference System

**Priority:** Lower (Focus on training/evaluation first).

**Objective:** Integrate the trained model (chosen best between CQL/DT) into the actual routing tool.

**Detailed Steps:**

1.  **[TODO] Inference Code (`predict_weights.py`):**
    *   **Adapt for chosen model (CQL or DT).**
    *   Load the chosen saved model/policy and state scaler.
    *   **If DT:**
        *   Implement logic to manage the sequence history (observations, actions, returns-to-go, timesteps) up to `context_size`.
        *   Determine how to set the `target_return_to_go` for the policy (critical tuning parameter for performance).
        *   Feed the properly formatted sequence inputs to the loaded DT policy.
    *   **If CQL:** Load model, normalize state, predict.
    *   Normalize the state using the state scaler.
    *   Call the appropriate prediction function (`model.predict()` or policy call).
    *   Apply final domain constraints to the weights.
    *   Feed weights back to the simulator.

## 7. Code Structure Suggestion

```
routing_rl/
├── data/                     # Output MDPDataset, scalers etc.
├── notebooks/                # Exploratory analysis
├── analysis_plots/           # Output plots from test_dataset.py
├── src/
│   ├── data_processing/
│   │   └── create_dataset.py
│   ├── training/
│   │   └── train_cql.py
│   ├── evaluation/
│   │   └── evaluate_agent.py
│   ├── inference/
│   │   └── predict_weights.py
│   └── utils/                # Helper functions, constants
├── initial_plan.md
├── implementation.md
├── dataset_test.md
├── test_dataset.py
├── main_workflow.ipynb       # Notebook for running the pipeline
└── requirements.txt
```

## 8. Key Considerations & Risks

*   **Reward Sparsity:** Primary challenge. Relying on `num_violating_reward` and careful tuning is key. Affects both CQL and DT.
*   **Dataset Size:** ~6000 transitions might be insufficient for robust learning without denser rewards or more data.
*   **Action Bias:** The learned policy for `decay_weight` will be constrained by the biased data (affects both).
*   **Action Scaling:** Ensure `MinMaxActionScaler` is correctly configured. Ensure final constraints are applied during inference.
*   **Evaluation Complexity:** Evaluating the true impact requires either running the routing simulation or relying on potentially less accurate offline metrics. DT evaluation can be more complex due to target return setting.
*   **Hyperparameter Sensitivity:** Offline RL, especially CQL (`conservative_weight`, LRs) and DT (`context_size`, LR schedule, target), can be sensitive to hyperparameters.
*   **DT Specifics:** Determining `max_timestep`, tuning `context_size`, and setting target returns for DT require careful consideration.

## 9. [TODO] Colab/Notebook Workflow (`main_workflow.ipynb`)

**Objective:** Provide a top-level Jupyter notebook to run the core data processing and training pipeline, suitable for execution on Google Colab or similar platforms.

**Structure:**

1.  **Setup:**
    *   Clone the GitHub repository containing the code (if running on Colab).
    *   Install dependencies from `requirements.txt`.
    *   Import necessary libraries.
    *   Define configuration parameters (data paths, model settings, etc.) or load them from a config file.
2.  **Data Preparation:**
    *   Execute the `src/data_processing/create_dataset.py` script (using `!python ...` or `%run ...`) or integrate its core logic directly into notebook cells.
    *   Verify the output dataset file (`data/routing_dataset.h5`) is created.
3.  **Model Training (Option to choose CQL or DT):**
    *   Add logic or separate cells to run either `src/training/train_cql.py` or `src/training/train_dt.py`.
    *   Load the dataset created in the previous step.
    *   Train the chosen agent.
    *   Save the trained model.
4.  **Basic Results/Visualization:**
    *   Optionally load training logs/metrics and display basic plots (e.g., loss curves) if generated by the training script.

This notebook serves as the main entry point for running the standard training workflow. 

## 10. Updated save_dataset function

```python
# In save_dataset function:
# Instead of filtering out NaN rewards
print(f"Found {df['final_reward'].isna().sum()} NaN rewards (first steps)")
print("Setting NaN rewards to 0.0 to preserve all transitions")
df['final_reward'] = df['final_reward'].fillna(0.0)
# Skip the filtering step and use the full dataframe
df_filtered = df  # No filtering
``` 