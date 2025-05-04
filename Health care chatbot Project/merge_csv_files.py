import pandas as pd
import os

filepaths = [
    'data/Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv',
    'data/Disease_Control_and_PreventionQA.csv',
    'data/Genetic_and_Rare_DiseasesQA.csv',
    'data/growth_hormone_receptorQA.csv',
    'data/Heart_Lung_and_BloodQA.csv',
    'data/MedicalQuestionAnswering.csv',
    'data/Neurological_Disorders_and_StrokeQA.csv',
    'data/OtherQA.csv',
    'data/SeniorHealthQA.csv'
    
]


merged_df = pd.DataFrame(columns=['short_question', 'short_answer'])

for filepath in filepaths:
    df = pd.read_csv(filepath)

    
    question_col = next((col for col in df.columns if 'question' in col.lower()), None)
    answer_col = next((col for col in df.columns if 'answer' in col.lower()), None)

    if question_col and answer_col:
        df = df[[question_col, answer_col]]
        df.columns = ['short_question', 'short_answer']
        merged_df = pd.concat([merged_df, df], ignore_index=True)


merged_df.dropna(inplace=True)

# Save to a single CSV
merged_df.to_csv('data/combined_health_qa.csv', index=False)

print("✅ Combined CSV saved as data/combined_health_qa.csv")