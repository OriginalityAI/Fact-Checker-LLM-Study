import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

gpt3 = pd.read_csv("../data/completed_gpt3.csv", index_col=0)
gpt4 = pd.read_csv("../data/completed_gpt4.csv", index_col=0)
llama7 = pd.read_csv("../data/completed_llama7.csv", index_col=0)
llama13 = pd.read_csv("../data/completed_llama13.csv", index_col=0)
llama70 = pd.read_csv("../data/completed_llama70.csv", index_col=0)

# making all columns consistent
llms = [gpt3, gpt4, llama7, llama13, llama70]


# sample data
# print("A sample from the GPT3 dataset:")
# with pd.option_context('display.max_colwidth', 300):
#     display(llms[0].loc[[504, 655, 765, 963]])
    
colors = ['lightgreen', 'darkgreen', 'wheat', 'peru', 'maroon']
models = ["GPT-3.5", "GPT-4", "LLAMA-7B", "LLAMA-13B", "LLAMA-70B"]
model_color_mapping = {models[i]:colors[i] for i in range(5)}
    
    
def viz_llm():
    fig, axs = plt.subplots(5, 1, figsize=(6, 20), 
                                sharex=True
                               )
    for i in range(5):
        df = llms[i]
        df.rename(columns={'category':'topic'}, inplace=True)
        cat_mean = df.groupby('topic')['label'].mean()*100
        cat_mean = cat_mean.sort_values()


        cat_mean = cat_mean.reset_index()

        sns.barplot(data=cat_mean, x='topic', y='label', palette=[colors[i]], ax=axs[i])

        axs[i].set_ylabel("Model Accuracy  (%)")
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
        axs[i].set_title(f"Accuracy of {models[i]} in 10 Topics")
    plt.tight_layout()
    plt.show()
 
    
def generate_pivot_table():
    main_df = pd.DataFrame()
    main_df['category'] = gpt3['category']
    main_df.rename(columns={'category':'topic'}, inplace=True)
    for i, col in enumerate(models):
        main_df[col] = llms[i]['label']
        main_df[col] = main_df[col].replace({True: 1, False: 0, None: 0.5}).astype(float)
        main_df[col] = main_df[col].fillna(0.5)
    # main_df = main_df * 100 # converting to percentages
    df = main_df.groupby('topic').mean() # pivot table
    display(df*100)
# #     melt the table for the polar visualization, not needed after first run
#     df_melt = df.reset_index()
#     df_melt = pd.melt(df_melt, id_vars=['topic'], value_vars=models, var_name='model', value_name='score')
#     df_melt.to_csv("../data/dfmelt.csv", index=True)
    return df, main_df
    
    
def viz_confidence_chart(df, main_df):
    conf_df = {}
    for model in models:
        tmp = main_df[model]
        conf_df[model] = (tmp[tmp!=0.5].count()*100/len(tmp))

    conf_df = pd.Series(conf_df)
    display(conf_df.to_frame().T)

    colors = [model_color_mapping.get(index, 'k') for index in conf_df.index]

    sns.barplot(x=conf_df.index, y=conf_df.values, palette=colors)

    plt.ylim(80, 105)
    # You can set other labels and titles as needed

    plt.ylabel("Confidence Score (%)")
    plt.title("Confidence of Models in Answering Prompts")

    # Display the plot
    plt.show()
    
    
def viz_accuracy_plots(df):
    """
    too busy. will not use. but keeping for archives
    """
    fig, ax = plt.subplots(figsize=(10, 6)) 

    (df*100).plot(kind='bar', ax=ax, width=0.8)

    ax.set_ylabel("Fact Checking Score (%)")
    ax.set_title("Fact Checking: LLM models accuracy per topic")

    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6)) 

    df.T.plot(kind='bar', ax=ax, width=0.8)

    ax.set_ylabel("Fact Checking Score (%)")
    ax.set_title("Fact Checking: LLMs Accuracy")

    plt.show()
    
def viz_heatmap(df):
#     plt.figure(figsize=(7,10))
    custom_colors = sns.color_palette("flare", as_cmap=True)

    sns.heatmap(df, cmap=custom_colors, annot=True, fmt=".0%", cbar=True, 
#                 cbar_kws={" = '.0%'}, 
                          vmax=1) 
    ax = plt.gca()
    for i in range(len(df)):
        for j in range(len(df)):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='white', lw=1))

    plt.title("Fact Checking: Accuracy of Each LLM in Topic Categories")
    plt.show()
    
    
def viz_performances(df):
    from matplotlib.cm import get_cmap
    
    model_averages = df.mean(axis=0).sort_values()
    model_averages = model_averages * 100
    display(model_averages.to_frame().T)

    plt.figure(figsize=(10, 6))  
    sns.barplot(x=model_averages.index, y=model_averages, palette=[model_color_mapping[model] for model in model_averages.index])

    plt.ylabel('Score (%)')
    plt.title('Fact Checking: Model Accuracy')
    plt.xticks(rotation=45)
    plt.show()

    cmap = get_cmap('flare', len(df.index))
    cmap

    topic_averages = df.mean(axis=1).sort_values()
    topic_averages = topic_averages * 100
    display(topic_averages.to_frame().T)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_averages.index, y=topic_averages, palette=[cmap(i) for i in range(len(topic_averages))])

    plt.ylabel('Fact Checking Score (%)')
    plt.title('Fact Checking: Average Models Accuracy Per Topic')
    plt.xticks(rotation=45)
    plt.show()