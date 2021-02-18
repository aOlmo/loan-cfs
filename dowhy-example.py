from dowhy import CausalModel
import dowhy.datasets
import pandas as pd
from adult import *

# Load some sample data
data = dowhy.datasets.linear_dataset(
    beta=10,
    num_common_causes=5,
    num_instruments=2,
    num_samples=10000,
    treatment_is_binary=True)

adult_df = get_adult()

model = CausalModel(
    data=adult_df,
    treatment=["age"],
    outcome=["income"],
    common_causes=["race"]
    # graph=data["gml_graph"]
)

# Saves the model as "causal_model.png"
model.view_model(layout="dot")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print("Causal Estimate is " + str(estimate.value))


###############################
# df = data["df"]
# print(df.head())
# print(data["dot_graph"])
# print("\n")
# print(data["gml_graph"])

# model = CausalModel(
#     data=df,
#     treatment=data["treatment_name"][0],
#     outcome=data["outcome_name"][0],
#     common_causes=['race']
#     # graph=data["gml_graph"]
# )