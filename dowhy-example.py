import pandas as pd
import dowhy.datasets
from datasets import *
from dowhy import CausalModel

credit_data = get_credit()

model = CausalModel(
    data=credit_data["df"],
    treatment=["YearsEmployed"],
    outcome=["Approved"],
    graph=credit_data["dot_graph"],
)

from sklearn.linear_model import LogisticRegressionCV

# Saves the model as "causal_model.png"
model.view_model(layout="dot")
identified_estimand_binary = model.identify_effect(proceed_when_unidentifiable=True)
# estimate = model.estimate_effect(identified_estimand, method_name="backdoor.econml.drlearner.LinearDRLearner")

orthoforest_estimate = model.estimate_effect(identified_estimand_binary,
                                             method_name="backdoor.econml.ortho_forest.ContinuousTreatmentOrthoForest",
                                             target_units=lambda df: df["Male"] == 1,
                                             confidence_intervals=False,
                                             method_params={"init_params": {
                                                 'n_trees': 2,  # not ideal, just as an example to speed up computation
                                             },
                                                 "fit_params": {}
                                             })
print(orthoforest_estimate)
# from sklearn.linear_model import LogisticRegressionCV
#
# drlearner_estimate = model.estimate_effect(identified_estimand_binary,
#                                            method_name="backdoor.econml.drlearner.LinearDRLearner",
#                                            target_units=lambda df: df["Male"] == 1,
#                                            confidence_intervals=False,
#                                            method_params={"init_params": {
#                                                'model_propensity': LogisticRegressionCV(cv=2, solver='lbfgs',
#                                                                                         multi_class='auto')
#                                            },
#                                                "fit_params": {}
#                                            })
# print(drlearner_estimate)

# print(estimate)
# print("Causal Estimate is " + str(estimate.value))

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
