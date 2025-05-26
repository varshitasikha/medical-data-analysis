import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1.Import data

df = pd.read_csv(r"C:\Users\sjvar\Downloads\medical_examination.csv")

# 2. Add overweight column (correct BMI classification)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)

# 3. Normalize cholesterol
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)

# 4. Normalize glucose
df['gluc'] = (df['gluc'] > 1).astype(int)

# 5. Draw categorical plot
def draw_cat_plot():
    # 6. Melt dataframe (fix spelling of cholesterol)
    df_cat = pd.melt(df, id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 7. Group and reformat data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']) \
                   .size().reset_index(name='total')

    # 8. Create catplot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                      kind='bar', data=df_cat).fig

    # 9. Return figure
    return fig

# 10. Draw heatmap
def draw_heat_map():
    # 11. Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Plot heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f',
                center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.45})

    # 16. Return figure
    return fig
