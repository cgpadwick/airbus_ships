import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

api = wandb.Api()
allruns = api.runs('cpadwick/airbus_ships')

result = []
for arun in allruns:
    if arun.state == 'finished':
        # Runtime for the runs of interest was between 4 and 5 hours.
        if arun.summary['_runtime'] > 28800.0 and arun.summary['_runtime'] < 34000:
            result.append({'max_fscore_fg': arun.summary['max_fscore_fg'],
                           'loss': arun.summary['loss']})

df = pd.DataFrame(result)

plt.figure(1)
ax = sns.violinplot(y='max_fscore_fg', data=df)
ax.set_title('Violin Plot of F-Score of Repeated Runs')
plt.show()

plt.figure(2)

ax = sns.scatterplot(x=df['loss'], y=df['max_fscore_fg'])
ax.set_title('Scatter Plot of Loss vs Fscore')
plt.show()







