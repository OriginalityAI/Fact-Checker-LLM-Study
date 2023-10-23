#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv("../data/dfmelt.csv", index_col=0)
df['average_topic_score'] = df.groupby('topic')['score'].transform('mean')
df = df.sort_values(['average_topic_score', 'topic', 'score'], ascending=[True, False, False])
df = df.reset_index(drop=True)

colors = ['lightgreen', 'darkgreen', 'wheat', 'peru', 'maroon']
models = ["GPT-3.5", "GPT-4", "LLAMA-7B", "LLAMA-13B", "LLAMA-70B"]
model_color_mapping = {models[i]:colors[i] for i in range(5)}


# In[2]:


padding = 2

num_of_groups = df.topic.nunique()
group_size = df.model.nunique()
new_length = num_of_groups * (group_size + padding)

new_index = [] 
for group_no in range(num_of_groups):
    original_group = df.index.tolist()[group_no * group_size : (1 + group_no) * group_size]
    new_index.extend(original_group + ['x']*padding)

new_df = pd.DataFrame(np.nan, index=new_index, columns=df.columns)
new_df.loc[df.index] = df.values
new_df.loc['x'] = [None, None, 0, 0]

new_df.reset_index(drop=True, inplace=True)    
new_df['index'] = new_df.index * 360/len(new_df)
new_df


# In[3]:


import plotly.express as px
fig = px.bar_polar(new_df, r='score',
                  theta='index',
                  # template='plotly_light',
                  color = 'model',
                  color_discrete_map = model_color_mapping,
                  hover_name = 'topic',
                  hover_data = {
                      'index':False,
                      'average_topic_score':':.0%',
                  'score':':.0%'},
                   direction='counterclockwise',
                   start_angle = 135,
                   animation_group='topic',
                   # animation_frame='model',
                  range_r=[-0.25,1],
                  width=800,
                  height=800)
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white'  
)
fig.update_layout(polar=dict(
        radialaxis=dict(
            labelalias='topic',
            showticklabels=False,
            ticks='',),
        angularaxis=dict(
            showticklabels=False, 
            showgrid=False,
            ticks='',)
    ))

fig.update_layout(
    title_text="Originality AI Fact Checker Results",
    title_x=0,  # Title x-coordinate position (0.5 is centered)
)
fig.add_annotation(
    text="Testing LLM Claims Across 10 Topics",
    # xref="paper",  
    # yref="paper",
    x=-0.14,        
    y=1.0,      
    showarrow=False  
)
fig.show()


# In[4]:


import plotly
plotly.offline.plot(fig, filename='../images/polar.html')

