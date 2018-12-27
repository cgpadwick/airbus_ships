import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

api = wandb.Api()
allruns = api.runs('cpadwick/airbus_ships')

result = []
for arun in allruns:
    if '_runtime' in arun.summary.keys():
        # Runtime for the runs of interest was between 4 and 5 hours.
        if arun.summary['_runtime'] > 28800.0 and arun.summary['_runtime'] < 32000:
            result.append({'max_fscore_fg': arun.summary['max_fscore_fg']})

df = pd.DataFrame(result)

ax = sns.violinplot(data=df)
ax.set_title('Violin Plot of F-Score of Repeated Runs')
plt.show()







