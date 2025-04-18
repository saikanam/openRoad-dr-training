# Offline RL Algorithm for Optimizing Detailed Routing Weights

## Problem Statement
We need to optimize four weights (drc_weight, marker_weight, fixed_weight, decay_weight) for each iteration of detailed routing to minimize DRC violations, ultimately reducing the total number of routing iterations required.

## Data Structure
Training data is organized in CSV files stored in a training_data folder, with a consistent naming format:
- `training_data/trainingdata_designName.csv`

Each file contains columns:
```
uniqueID, designID, iteration, pin_count, net_count, drv, wireLength, drc_weight, marker_weight, 
fixed_weight, decay_weight, boxID, box_size, box_drv, L_N_box, L_N_drv, R_N_box, R_N_drv
```

## Data Quality Issues & Analysis Insights
Our analysis revealed several data quality issues and insights:

1. **Missing Values (2,795 rows):** Affect 6 specific iterations entirely across 5 files.
2. **Duplicate boxIDs (48 instances):** All in one run (`bp_be_top_run_68`, iter 1), with differing neighbor info.
3. **DRV Mismatches (4,894 instances):** `drv` != `sum(box_drv)`, varies by design/iteration. Iteration `drv` and `sum(box_drv)` represent related but distinct measures.
4. **Action Space (Weights):**
    - **Good:** Weights varied independently (low correlation).
    - **Challenge:** Severe bias for `decay_weight` towards 1.0. Very little data for low values. Agent likely won't learn to use low `decay_weight`.
    - Multi-modal distributions for other weights, adequate coverage but biased towards certain values.
5. **State Space:**
    - **Good:** Covers various design sizes (wirelength), iterations, and trajectory lengths.
    - **Challenge:** `drv` distribution heavily skewed towards zero. Less data for high-DRV states.
6. **Reward Sparsity:**
    - `primary_reward` (ΔDRV) and `box_reward` (ΔTotal Box DRV) are highly peaked at zero (sparse).
    - Experimental signals `max_box_drv_reward` (ΔMax Box DRV) and **`num_violating_reward` (Δ# Violating Boxes) are significantly denser, offering more frequent learning signals.**
7. **Dataset Size:** ~6000 valid transitions, potentially small given reward sparsity of primary signals.

**Note:** Data collection is rudimentary; more data is expected.

## RL Framework Selection
For offline reinforcement learning, we'll use Conservative Q-Learning (CQL), which is specifically designed for learning from fixed datasets without active environment interaction.

**Recommended Library**: d3rlpy
- d3rlpy is a Python library for offline deep reinforcement learning
- It provides implementations of CQL and other offline RL algorithms
- Installation: `pip install d3rlpy`

## RL Environment Design

### State Space
The state representation at the beginning of iteration `t` should include:
- Current iteration number (`t`)
- General design metrics (`pin_count`, `net_count`) - *Assumed constant for a given design.*
- Performance metrics from the *end* of iteration `t-1`:
    - Iteration-level DRV count (`drv_{t-1}`)
    - Box-level DRV metrics:
        - Wire length (`wireLength_{t-1}`)
- Aggregated box-level metrics from the *end* of iteration `t-1`:
    - Maximum `box_drv` on any single box (`max_box_drv_{t-1}`)
    - Count and Percentage of boxes with non-zero DRV (`num_violating_{t-1}`)
    - Average `box_drv` per box (`average_box_drv_{t-1}`)
- Historical DRV trajectory: Iteration-level DRV counts from the last `k` iterations (e.g., `[drv_{t-1}, drv_{t-2}, ..., drv_{t-k}]`). Use padding (e.g., initial DRV or a special value) for iterations `t < k`. Let's start with `k=3`. *Make sure this history is available for the reward calculation.*
- **Note:** We will *not* include the specific `designID` to encourage learning universal weights.
- **Note:** `total_box_drv` related features have been removed from the state.

### Action Space
- Four continuous values representing the weights *to be used* for iteration `t`:
  - drc_weight: Controls penalty for design rule violations
  - marker_weight: Controls sensitivity to marker areas 
  - fixed_weight: Controls how much to preserve fixed components
  - decay_weight: Controls routing decay behavior (Agent likely biased towards high values due to data).

### Reward Function

**Recommended Structure:** Combine primary DRV signal with the denser `num_violating_reward`.
- **Primary reward**: Focus on iteration-level DRV reduction:
  - `primary_reward = -(drv_t - drv_{t-1})`
- **Box-level reward component (Optional but Recommended)**: Secondary reward based on box-level improvements:
  - `box_reward = -(total_box_drv_t - total_box_drv_{t-1})`
  - *Definition:* Measures the reduction in the **sum** of DRVs across all boxes from iteration `t-1` to `t`.
- **Number Violating Reward (Strongly Recommended)**: Use the change in the count of violating boxes as a denser signal:
  - `num_violating_reward = -(num_violating_t - num_violating_{t-1})`
  - *Definition:* Measures the reduction in the **count** of boxes with `box_drv > 0` from iteration `t-1` to `t`.
- **Stuck Penalty**: Penalize if DRV hasn't improved recently:
  - `is_stuck = (drv_t >= drv_{t-1} - ε_stuck) and (drv_{t-1} >= drv_{t-2} - ε_stuck)` (requires history; `ε_stuck` tolerance)
  - `stuck_penalty = -penalty_value if is_stuck else 0` (tune `penalty_value`)
- **Convergence bonus**: Additional reward for reaching the goal:
  - `convergence_bonus = bonus_value if drv_t == 0 else 0` (tune `bonus_value`)

**Combination & Normalization:**
- **Normalize Components:** Normalize `primary_reward`, `num_violating_reward`, and `stuck_penalty` (e.g., divide by std dev) before combining.
- **Final reward**: `reward = norm(primary) + β₄*norm(num_violating) + β₂*norm(stuck) + β₃*convergence_bonus`
  - Tune β weights, starting with a significant weight for `norm(num_violating)` (β₄).
  - Note: `box_reward` (β₁) component has been removed.

**Experimental Alternatives:**
- **Simpler Reward**: Evaluate `norm(primary) + β₃*convergence_bonus` or `norm(num_violating) + β₃*convergence_bonus`.
- **Include Max Box DRV:** Add `max_box_drv_reward` as another weighted component if `num_violating_reward` isn't sufficient.

## Implementation Approach

### Data Processing Pipeline
This pipeline transforms the raw, box-level CSV data into the iteration-level transitions required for d3rlpy.

1.  **Load & Combine Data:** Load all `trainingdata_*.csv` files into a single pandas DataFrame.
2.  **Filter Invalid Data:** 
    - Remove the 6 iterations with missing `drv` and `wireLength` values.
3.  **Handle Duplicate BoxIDs:**
    - For the specific run `bp_be_top_run_68` (iteration 1), define a strategy before aggregation (e.g., average differing values between duplicates, or consistently select one row - perhaps the one with more complete neighbor info).
4.  **Aggregate per Iteration:** Group the DataFrame by `uniqueID` and `iteration`.
    *   For each group (representing one iteration of one run):
        *   Extract iteration-level features (like `drv`, `wireLength`, weights, `pin_count`, `net_count` - taking the value from the first row is usually sufficient as they are constant within the iteration).
        *   Calculate aggregated `box_drv` features (using the strategy defined in step 3 for the affected run):
            *   `max_box_drv` = `max(box_drv)`
            *   `average_box_drv` = `mean(box_drv)`
            *   `num_violating` = count of boxes with `box_drv > 0`
            *   `non_zero_drv_box_percentage` = percentage of boxes with `box_drv > 0`
        *   **Note:** We are explicitly *not* aggregating `L_N_drv` or `R_N_box` into the iteration state.
    *   Store these iteration-level and aggregated features in a new DataFrame, indexed by `uniqueID` and `iteration`.
5.  **Calculate Reward Components:** Based on the aggregated iteration data, calculate all required reward components (`primary_reward`, `num_violating_reward`, `stuck_penalty`, `convergence_bonus`). This requires shifting data to get values from `t`, `t-1`, `t-2`, etc. (`box_reward` removed).
6.  **Normalize Rewards (Optional but Recommended):** Calculate statistics (e.g., std dev) for the main reward components across the dataset and store them. Normalize these components.
7.  **Calculate and Save Action Statistics:** Determine the minimum and maximum values for each action column (`drc_weight`, etc.) across the entire dataset and save these statistics (e.g., to `action_min_max.npz`).
8.  **Construct Transitions:** Sort the iteration-level DataFrame by `uniqueID` and `iteration`.
    *   Iterate through the sorted data, grouping by `uniqueID`.
    *   For each `uniqueID`, create the sequence of transitions `(s_t, a_t, r_t, s_{t+1}, terminal_t)`:
        *   `s_t`: State built from iteration `t-1` data (including historical DRVs up to `t-1`). If historical DRVs (`drv_{t-k}`) are missing (due to being near the start of a trajectory, iterations `< k`), **pad these missing values with -1**.
        *   `a_t`: The four weights (`drc_weight`, `marker_weight`, `fixed_weight`, `decay_weight`) that were actually *used* during iteration `t`.
        *   `r_t`: Final reward calculated by combining the (potentially normalized) components using β weights (calculated based on the difference between state `t` and state `t+1` results). **For the first transition (t=1, using state t=0), where the reward based on change is undefined, set r_1 = 0.0.**
        *   `s_{t+1}`: State built from iteration `t` data (again, padding history if needed).
        *   `terminal_t`: `True` if `drv_t == 0` (i.e., the state *reached* after the transition is terminal).
    *   **Note:** All transitions, including the first and last ones in each trajectory, are kept.
9.  **Normalize States:** Apply normalization (e.g., `RobustScaler`) to the numerical features within the `states` portion of the generated transitions. Save the scaler.
10. **Create MDPDataset:** Convert the lists of normalized states, *original* actions, rewards, and terminals into d3rlpy's `MDPDataset` format and save it.

### Offline RL Training with CQL
1.  **Load Dataset and Action Stats:** Load the generated `MDPDataset` and the saved action min/max statistics.
2.  **Configure CQL:**
    *   Instantiate `d3rlpy.preprocessing.MinMaxActionScaler` using the loaded min/max action statistics.
    *   Configure `d3rlpy.algos.CQLConfig`, passing the `MinMaxActionScaler` instance to the `action_scaler` argument. Optionally configure a `reward_scaler`.
    *   Create the `CQL` algorithm instance (`cql = config.create(...)`).
3.  **Train Model:** Use `cql.fit(dataset, ...)` to train the model.
4.  **Save Model Parameters:** Save the trained model parameters using `cql.save_model(...)` (e.g., `model_final.d3` or `model.pt`).
5.  **Export Inference Policy:** Export the policy separately for deployment using `cql.save_policy(...)` in both TorchScript (`policy.pt`) and ONNX (`policy.onnx`) formats.

### Inference System
1.  **Option A (Using d3rlpy in Python Script):**
    *   Load the saved model parameters (`.pt` or `.d3`) using `config.create(...)`, `cql_algo.create_impl(...)`, and `cql_algo.load_model(...)`.
    *   Load the state scaler (`.pkl`).
    *   In the inference script (`predict_weights.py`):
        *   Receive raw state features.
        *   Normalize the state using the loaded state scaler.
        *   Call `model.predict(normalized_state)`. Because the model was trained with `action_scaler`, this will automatically return *denormalized* actions in their original approximate scale.
        *   Apply final domain constraints (e.g., `max(1.0, weight)`) to the denormalized weights.
        *   Output the constrained weights.
2.  **Option B (Using Exported Policy in C++):**
    *   Load the exported TorchScript (`policy.pt`) or ONNX (`policy.onnx`) policy using LibTorch or ONNX Runtime in C++.
    *   Load the state scaler parameters (center/scale from `.pkl`) and implement state normalization logic in C++.
    *   Normalize the state in C++.
    *   Feed the normalized state to the loaded policy to get actions. **Note:** Policies exported via `save_policy` *should* incorporate the denormalization internally if trained with an `action_scaler`.
    *   Apply final domain constraints in C++.

### Visualization and Analysis
1. Use libraries like matplotlib and seaborn for:
   - DRV progression charts comparing baseline vs. CQL-optimized runs
   - Plot of iteration-level `drv` vs. `total_box_drv` across iterations
   - Visualization of the DRV discrepancy patterns across different designs
   - Weight evolution visualization across iterations
   - Performance comparison across different designs

## Expected Outcome

- **CQL:** The CQL-trained agent should learn robust weight optimization strategies from offline data that significantly reduce the number of iterations required for complete routing. By incorporating denser reward signals like the change in the number of violating boxes, the agent may overcome some limitations of reward sparsity and find effective weight combinations, despite other data limitations.
- **Decision Transformer (DT):** The DT agent will model the routing process as a sequence problem. If successful, it might learn different types of strategies compared to CQL, potentially focusing on achieving a target return (e.g., zero DRV) given the history. Its performance will depend heavily on the `context_size`, the determined `max_timestep`, and how target returns-to-go are handled during training and inference.

## Additional Considerations

- **Algorithm Comparison:** Explicitly compare CQL and DT performance using consistent evaluation metrics (offline and, if possible, online simulation). d3rlpy offers other offline RL algorithms (BCQ, BEAR, TD3+BC) that could be compared against CQL.
- **Ensemble Methods:** Consider ensemble methods by training multiple models (of the same or different types) on different subsets of the data or with different initializations.
- Implement uncertainty estimation to gauge confidence in the model's weight predictions
- Experiment with different normalization strategies (`StandardScaler`, `RobustScaler`, design-specific) for both states and potentially rewards.
- Given the `decay_weight` bias, the learned policy will likely keep this value high. Evaluate if this constraint is acceptable.