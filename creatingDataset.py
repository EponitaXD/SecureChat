import pandas as pd
from sklearn.model_selection import train_test_split

def createJailbreakDF():
    df = pd.DataFrame({
        'Prompt': pd.Series(dtype='str'),
        'Category': pd.Series(dtype='int'),
    })
    # first jailbreak dataset
    df_jailbreak = pd.read_csv("hf://datasets/rubend18/ChatGPT-Jailbreak-Prompts/dataset.csv")

    # remove all collumns from df jail except for prompts
    df_jailbreak.drop(columns=['Name', 'Votes','Jailbreak Score', 'GPT-4'], inplace=True)

    # add a collumn with all 1s
    df_jailbreak['Category'] = 1

    # second jailbreak dataset
    df_jailbreak2 = pd.read_parquet("hf://datasets/TrustAIRLab/in-the-wild-jailbreak-prompts/jailbreak_2023_12_25/train-00000-of-00001.parquet")

    df_jailbreak2.drop(columns=['platform', 'source','jailbreak', 'created_at', 'date', 'community','community_id', 'previous_community_id'], inplace=True)

    # add a collumn with all 1s
    df_jailbreak2['Category'] = 1

    # rename column
    df_jailbreak2.rename(columns={'prompt': 'Prompt'}, inplace=True)

    # concatenate data
    df = pd.concat([df, df_jailbreak], ignore_index=True)
    df = pd.concat([df, df_jailbreak2], ignore_index=True)

    return df


def createInjectionDF():
    df = pd.DataFrame({
        'Prompt': pd.Series(dtype='str'),
        'Category': pd.Series(dtype='int'),
    })

    splits = {'train': 'data/train-00000-of-00001-9564e8b05b4757ab.parquet', 'test': 'data/test-00000-of-00001-701d16158af87368.parquet'}
    df_injections = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["train"])
    df_injections_test = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["test"])

    # chage collumn name from text to Prompt
    # change column name from category to Category
    df_injections.rename(columns={'text': 'Prompt', 'label': 'Category'}, inplace=True)

    # Loop through rows
        # if Category == 1
            # change to 2
    df_injections.loc[df_injections['Category'] == 1, 'Category'] = 2

    # chage collumn name from text to Prompt
    # change column name from category to Category
    df_injections_test.rename(columns={'text': 'Prompt', 'label': 'Category'}, inplace=True)

    # Loop through rows
        # if Category == 1
            # change to 2

    df_injections_test.loc[df_injections_test['Category'] == 1, 'Category'] = 2

    # Login using e.g. `huggingface-cli login` to access this dataset
    df_injections3 = pd.read_csv("hf://datasets/xxz224/prompt-injection-attack-dataset/complete_dataset.csv")

    # remove non injection prompts
    df_injections3 = df_injections3[df_injections3['inject_label'] != 0]

    # remove all collumns from df jail except for prompts
    df_injections3.drop(columns=['id', 'target_text','target_label', 'target_task_type', 'inject_text', 'inject_label', 'inject_task_type', 'naive_attack', 'escape_attack', 'ignore_attack', 'fake_comp_attack'], inplace=True)

    df_injections3.rename(columns={'combine_attack': 'Prompt'}, inplace=True)

    df_injections3['Category'] = 2

    # concatenate dfs
    df = pd.concat([df, df_injections], ignore_index=True)
    df = pd.concat([df, df_injections_test], ignore_index=True)
    df = pd.concat([df, df_injections3], ignore_index=True)

    return df

def createGoodDF():
    df = pd.DataFrame({
        'Prompt': pd.Series(dtype='str'),
        'Category': pd.Series(dtype='int'),
    })

    df_good = pd.read_csv("hf://datasets/fka/prompts.chat/prompts.csv")

    # get all rows where type == TEXT
    df_good = df_good.loc[df_good['type'] == 'TEXT']

    # Remove all column exept prompt
    df_good.drop(columns=['act', 'for_devs','type', 'contributor'], inplace=True)

    # add category = 0
    df_good['Category'] = 0

    # change name of propt to Prompt 
    df_good.rename(columns={'prompt': 'Prompt'}, inplace=True)

    # concatenate good
    df = pd.concat([df, df_good], ignore_index=True)

    return df


"""
I want my dataset to have the following format
Prompt | Category

Cathegories will be 
0: Good
1: Jailbreak
2: Injection
"""
df = pd.DataFrame({
    'Prompt': pd.Series(dtype='str'),
    'Category': pd.Series(dtype='int'),
})

jail_df = createJailbreakDF()
injection_df = createInjectionDF()
good_df = createGoodDF()

df = pd.concat([df, jail_df], ignore_index=True)
df = pd.concat([df, injection_df], ignore_index=True)
df = pd.concat([df, good_df], ignore_index=True)

df.dropna()
df.drop_duplicates(subset=['Prompt'], keep='first', inplace=False)

print(df)
print(df.describe())
print(df['Category'].value_counts())

# separate into train, test, validation splits

# Assuming you have a DataFrame named 'df'
# Define your desired ratios
train_ratio = 0.60
val_ratio = 0.20
test_ratio = 0.20

# Step 1: Split data into training+validation (main) and test sets
# The test_size here is the final test_ratio (0.20)
df_main, df_test = train_test_split(
    df, 
    test_size=test_ratio, 
    random_state=42 # Set random_state for reproducibility
)

# Step 2: Split the remaining 'df_main' into training and validation sets
# The new validation size relative to the main set is val_ratio / (train_ratio + val_ratio)
relative_val_size = val_ratio / (train_ratio + val_ratio) # 0.20 / (0.60 + 0.20) = 0.25

df_train, df_val = train_test_split(
    df_main, 
    test_size=relative_val_size, 
    random_state=42 # Use the same random_state
)

# You now have your three dataframes
print(f"Train shape: {df_train.shape}")
print(f"Validation shape: {df_val.shape}")
print(f"Test shape: {df_test.shape}")

df_train.to_csv('train.csv')
df_val.to_csv('val.csv')
df_test.to_csv('test.csv')