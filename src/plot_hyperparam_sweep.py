import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate


def clean_values(x):
    if x == 'NaN':
        return 0.0
    else:
        return x


api = wandb.Api()
allruns = api.runs('blueriver/airbus_ships-keras_bin')

result = []
for arun in allruns:
    if arun.state == 'finished':
        result.append({'gamma': arun.config['gamma'],
                       'alpha': arun.config['alpha'],
                       'loss': arun.summary['loss'],
                       'max_fscore_fg': arun.summary['max_fscore_fg']})

df = pd.DataFrame(result)

df['loss'] = np.log10(df['loss'])

# Clean up the 'NaN' values
df['max_fscore_fg'] = df['max_fscore_fg'].apply(clean_values)

gam = np.linspace(np.min(df['gamma']),  np.max(df['gamma']), 20)
alp = np.linspace(np.min(df['alpha']),  np.max(df['alpha']), 20)
f = interpolate.Rbf(df['gamma'], df['alpha'], df['max_fscore_fg'], kind='linear')
mg = np.meshgrid(gam, alp)
pred = f(mg[0], mg[1])

# Clean up bad interpolation values
pred[pred < 0] = 0
pred[pred > np.max(df['max_fscore_fg'])] = np.max(df['max_fscore_fg'])

# Create labels that we can actually display on a plot
gam = ['{0:.3f}'.format(x) for x in gam]
alp = ['{0:.3f}'.format(x) for x in alp]

ax = sns.heatmap(pred, xticklabels=gam, yticklabels=alp)
ax.set_title('Fscore versus Gamma and Alpha for FocalLoss')
plt.show()









