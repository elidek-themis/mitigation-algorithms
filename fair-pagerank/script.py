from experiments import *
from collections.abc import Iterable

def perform_experiments(experiments: Experiment | Iterable[Experiment]):
    if isinstance(experiments, Experiment): # ensure experiments are an Iterable
        experiments = [experiments]
    for experiment in experiments:
        experiment.perform()

experiments = []
settings = Settings(True, True)
# for dataset in ["polbooks_no_neutral", "polblogs_weakly_undirected", "facebook_weakly_undirected"]: #["karate", "polbooks_no_neutral", "polblogs_weakly_undirected", "soc-political-retweet_weakly_undirected"]:
for dataset in synthetic_GPA_datasets["homophily"]: #synthetic_GPA_datasets["homophily"] + synthetic_GPA_datasets["red size"]:
    configurations = []
    for phi in [1.1]: #[np.round(0.5 + 0.1 * i, 1) for i in range(1, 5)]:
        for selector in ["dpr"]: #["ref", "defg", "duli", "dgli", "sefg", "suli", "sgli", "dpr", "spr", "rp"]:
            for intervention in ["nrb", "rrd"]:
                configurations.append(Configuration(.15, phi, selector, intervention))
    experiments.append(Experiment(dataset, configurations, settings))
perform_experiments(experiments)

# experiments = []
# settings = Settings(True, True)
# for i, k in product(range(7, 10), range(5)):
#     dataset = "synthetic_GPA-5-" + str(i) + "-" + str(i) + "_" + str(k)
#     configurations = []
#     for intervention in ["nrb", "rrd"]:
#         for selector in ["f7", "f8"]:
#             for phi in [.6]: # [np.round(i * .1 + .05, 2)]:
#                 configurations.append(Configuration(.15, phi, selector, intervention))
#     experiments.append(Experiment(dataset, configurations, settings))
# perform_experiments(experiments)

# experiments = []
# settings = Settings(True, True)
# for dataset in ["polbooks_no_neutral"]:#, "polblogs_weakly_undirected"]:
#     configurations = []
#     for intervention in ["nrb", "rrd"]:
#         for selector in ["defg"]:
#             for phi in [.6]: # [np.round(i * .1 + .05, 2)]:
#                 configurations.append(Configuration(.15, phi, selector, intervention))
#     experiments.append(Experiment(dataset, configurations, settings))
# perform_experiments(experiments)