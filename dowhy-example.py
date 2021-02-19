import pandas as pd
import dowhy.datasets
from datasets import *
from dowhy import CausalModel

credit_data = get_credit()

model = CausalModel(
    data=credit_data["df"],
    treatment=["YearsEmployed"],
    outcome=["Income"], ## TODO: Does not allow a binary outcome,
    graph=credit_data["dot_graph"]
)

# Saves the model as "causal_model.png"
model.view_model(layout="dot")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
# ATE = Average Treatment Effect
# ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
# ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
print(estimate)
print("Causal Estimate is " + str(estimate.value))


###################################################
# data = dowhy.datasets.linear_dataset(beta=10,
#         num_common_causes=5,
#         num_instruments = 2,
#         num_effect_modifiers=1,
#         num_samples=10000,
#         treatment_is_binary=True,
#         num_discrete_common_causes=1)
#
# df = data["df"]
#
# model=CausalModel(
#         data = df,
#         treatment=data["treatment_name"],
#         outcome=data["outcome_name"],
#         graph=data["gml_graph"]
#         )
