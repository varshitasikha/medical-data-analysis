import unittest
import pandas as pd
import numpy as np
from medical_data_visualizer import draw_cat_plot, draw_heat_map  # Replace 'your_module' with your actual script name without .py

class TestMedicalExamination(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load dataset once for all tests
        cls.df = pd.read_csv(r"C:\Users\sjvar\Downloads\medical_examination.csv")
        
        # Add overweight column (correct BMI classification)
        cls.df['overweight'] = (cls.df['weight'] / ((cls.df['height'] / 100) ** 2)) > 25
        cls.df['overweight'] = cls.df['overweight'].astype(int)
        
        # Normalize cholesterol and glucose
        cls.df['cholesterol'] = (cls.df['cholesterol'] > 1).astype(int)
        cls.df['gluc'] = (cls.df['gluc'] > 1).astype(int)
        
        # Melt dataframe for categorical plot
        cls.df_cat = pd.melt(cls.df, id_vars=['cardio'],
                             value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
        cls.df_cat_grouped = cls.df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    def test_overweight_column(self):
        # Calculate BMI and overweight manually for first 5 rows
        bmi = self.df['weight'][:5] / ((self.df['height'][:5] / 100) ** 2)
        overweight_manual = (bmi > 25).astype(int)
        overweight_manual.name = 'overweight'  # Fix name attribute to match original
        pd.testing.assert_series_equal(self.df['overweight'][:5], overweight_manual)

    def test_catplot_data(self):
        # Check if melted dataframe has expected columns
        self.assertTrue(all(col in self.df_cat.columns for col in ['cardio', 'variable', 'value']))
        # Check if grouped dataframe has 'total' column
        self.assertIn('total', self.df_cat_grouped.columns)

    def test_heatmap_cleaning(self):
        # Calculate quantiles from original df (not filtered)
        height_low = self.df['height'].quantile(0.025)
        height_high = self.df['height'].quantile(0.975)
        weight_low = self.df['weight'].quantile(0.025)
        weight_high = self.df['weight'].quantile(0.975)

        df_heat = self.df[
            (self.df['ap_lo'] <= self.df['ap_hi']) &
            (self.df['height'] >= height_low) &
            (self.df['height'] <= height_high) &
            (self.df['weight'] >= weight_low) &
            (self.df['weight'] <= weight_high)
        ]

        # Assert conditions
        self.assertTrue((df_heat['ap_lo'] <= df_heat['ap_hi']).all())
        self.assertTrue(df_heat['height'].between(height_low, height_high).all())
        self.assertTrue(df_heat['weight'].between(weight_low, weight_high).all())

if __name__ == '__main__':
    unittest.main()
