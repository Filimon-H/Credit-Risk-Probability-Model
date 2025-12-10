# Credit Risk Probability Model

Project scaffold initialized with basic testing and CI structure.

- Source code in `src/`
- Notebooks in `notebooks/`
- Scripts in `scripts/`
- Unit tests in `tests/`

## Credit Scoring Business Understanding

### Basel II, risk measurement, and model interpretability

The Basel II Capital Accord requires banks to quantify, document, and regularly validate their credit risk models. In practice, this means that:

- The **inputs**, **transformations**, and **outputs** of the model must be transparent.
- The relationship between risk drivers (features) and predicted risk should be **understandable** to risk and compliance teams.
- The model must be **monitored over time** for stability and performance, with clear governance around changes.

Because of this, an interpretable and well-documented model is not only a technical preference but a regulatory requirement. In this project, we focus on:

- Explicit feature engineering (e.g., RFM metrics, aggregate transaction features).
- Clearly defined target construction (proxy for credit risk).
- A modeling process that can be explained end-to-end, from data to prediction.

### Need for a proxy target and its business risks

The dataset does not contain a direct "default" label (e.g., a flag that indicates whether a customer actually failed to repay a loan). To train a supervised model, we therefore construct a **proxy target** based on customer behavior:

- We derive **RFM (Recency, Frequency, Monetary)** metrics from transaction history.
- We segment customers into behavioral clusters and define the **least engaged / lowest value cluster** as **high-risk**.
- We then create a binary target column `is_high_risk` (1 = high-risk cluster, 0 = other clusters).

This approach enables us to train a model, but it comes with important business risks:

- **Label mismatch**: the proxy may not perfectly match true default behavior. Some customers labeled as high-risk might actually repay loans, and vice versa.
- **Business bias**: the proxy reflects current behavioral patterns and the current business process. If behavior or product design changes, the proxy may no longer represent credit risk well.
- **Decision risk**: if the model is used directly for credit decisions without recognizing these limitations, the bank may reject good customers or underprice/overprice risk.

For these reasons, the proxy-based model should be seen as an **analytical tool and prototype**, not a fully validated production scorecard. Any deployment must include additional data, expert judgment, and back-testing against real loan performance.

### Simple vs complex models in a regulated financial context

There is a trade-off between model complexity and interpretability:

- **Simple, interpretable models** (e.g., Logistic Regression with Weight of Evidence (WoE) features):
  - Pros:
    - Easy to explain to business stakeholders and regulators.
    - Feature effects are transparent and often monotonic.
    - Simpler to monitor for drift and to implement in production systems.
  - Cons:
    - May achieve lower raw predictive performance (e.g., AUC, F1) compared to more complex models.

- **Complex, high-performance models** (e.g., Gradient Boosting, Random Forests, XGBoost, LightGBM):
  - Pros:
    - Often deliver higher predictive power and capture nonlinear relationships and interactions.
  - Cons:
    - Harder to explain and justify in a regulated environment.
    - More challenging to validate, monitor, and govern over time.
    - Risk of overfitting and unintended biases if not carefully controlled.

In a regulated credit risk setting, **interpretability, stability, and governance are as important as raw accuracy**. In this project, we therefore:

- Compare simple and complex models.
- Emphasize documentation of feature engineering, target construction, and model behavior.
- Treat highly complex models as **benchmarks and exploratory tools**, while giving special attention to models and features that can be more easily explained and audited.
