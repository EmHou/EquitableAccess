#!/usr/bin/env python3
"""
Statistical Analysis Module for Health Equity Research
Contains implementations of non-parametric tests including Friedman test
for analyzing multimodal access time differences.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AccessTimeAnalyzer:
    """
    Analyzer class for multimodal access time data using non-parametric statistics.
    """
    
    def __init__(self, data_path=None, population_data_path=None):
        """
        Initialize the analyzer with optional data paths.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the travel times CSV file
        population_data_path : str, optional
            Path to the DP05 population data CSV file
        """
        self.data_path = data_path
        self.population_data_path = population_data_path
        self.travel_data = None
        self.population_data = None
        self.results = {}
        
    def load_travel_data(self, data_path=None):
        """
        Load travel time data from CSV file.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the travel times CSV file. If None, uses self.data_path
        """
        if data_path is None:
            data_path = self.data_path
            
        if data_path is None:
            # Try to find travel_times.csv in the results directory
            script_dir = Path(__file__).parent
            root_dir = script_dir.parent
            data_path = root_dir / "results" / "travel_times.csv"
            
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Travel times data not found at {data_path}")
            
        print(f" Loading travel time data from: {data_path}")
        
        # Load data with appropriate dtypes
        self.travel_data = pd.read_csv(data_path, dtype={"GEOID": str})
        
        # Check available columns
        print(f" Available columns: {list(self.travel_data.columns)}")
        
        # Identify travel mode columns
        self.mode_columns = {}
        for mode in ["car", "transit", "walking"]:
            col = f"{mode}_time_min"
            if col in self.travel_data.columns:
                self.mode_columns[mode] = col
                print(f" Found {mode} travel times: {col}")
            else:
                print(f"Warning:  {col} not found, skipping {mode}")
                
        if not self.mode_columns:
            raise ValueError("No travel time columns found in the data")
            
        print(f" Loaded data for {len(self.mode_columns)} travel modes")
        
        # Load population data if available
        if self.population_data_path is None:
            # Try to find DP05 data in the data directory
            script_dir = Path(__file__).parent
            root_dir = script_dir.parent
            self.population_data_path = root_dir / "data" / "illinois_census_tract_DP05.csv"
        
        if os.path.exists(self.population_data_path):
            print(f" Loading population data from: {self.population_data_path}")
            self.population_data = pd.read_csv(self.population_data_path, dtype=str)
            
            # Extract GEOID from GEO_ID (remove '1400000US' prefix)
            self.population_data['GEOID'] = self.population_data['GEO_ID'].str[-11:]
            
            # Convert population to numeric
            if 'DP05_0001E' in self.population_data.columns:
                self.population_data['total_population'] = pd.to_numeric(
                    self.population_data['DP05_0001E'], errors='coerce'
                )
                print(f" Population data loaded: {len(self.population_data)} tracts")
            else:
                print("Warning:  Population column 'DP05_0001E' not found")
        else:
            print("Warning:  Population data not found, population analysis will be skipped")
        
        return self.travel_data
    
    def prepare_friedman_data(self):
        """
        Prepare data for Friedman test by ensuring all modes have data for the same tracts.
        Removes duplicates and missing values.
        
        Returns:
        --------
        pandas.DataFrame
            Cleaned data with only tracts that have data for all travel modes
        """
        if self.travel_data is None:
            raise ValueError("No travel data loaded. Call load_travel_data() first.")
            
        # Get columns for all available modes
        mode_cols = list(self.mode_columns.values())
        
        # Remove rows with missing values in any travel mode
        clean_data = self.travel_data[mode_cols + ["GEOID"]].dropna()
        
        # Remove duplicate GEOIDs (keep first occurrence)
        print(f" Before deduplication: {len(clean_data)} rows")
        clean_data = clean_data.drop_duplicates(subset=["GEOID"], keep="first")
        print(f" After deduplication: {len(clean_data)} unique tracts")
        
        print(f" Travel modes: {list(self.mode_columns.keys())}")
        
        return clean_data
    
    def run_friedman_test(self, data=None):
        """
        Perform Friedman test to compare access times across different travel modes.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Data to analyze. If None, uses prepared data from prepare_friedman_data()
            
        Returns:
        --------
        dict
            Dictionary containing Friedman test results and statistics
        """
        if data is None:
            data = self.prepare_friedman_data()
            
        # Extract travel time columns
        mode_cols = list(self.mode_columns.values())
        
        # Prepare data for Friedman test (each row is a tract, each column is a mode)
        friedman_data = [data[col].values for col in mode_cols]
        
        # Run Friedman test
        try:
            statistic, p_value = friedmanchisquare(*friedman_data)
            
            # Calculate effect size (Kendall's W)
            n = len(data)  # number of tracts
            k = len(mode_cols)  # number of modes
            w = statistic / (n * (k - 1))
            
            # Determine significance level
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            # Store results
            self.results['friedman'] = {
                'statistic': statistic,
                'p_value': p_value,
                'significance': significance,
                'effect_size_w': w,
                'n_tracts': n,
                'n_modes': k,
                'mode_names': list(self.mode_columns.keys())
            }
            
            # Print results
            print(f"\n{'='*60}")
            print(f" FRIEDMAN TEST RESULTS")
            print(f"{'='*60}")
            print(f" Test Statistic (Chi²): {statistic:.4f}")
            print(f" P-value: {p_value:.6f}")
            print(f" Significance: {significance}")
            print(f" Effect Size (Kendall's W): {w:.4f}")
            print(f"  Number of tracts: {n}")
            print(f" Number of modes: {k}")
            
            if p_value < 0.05:
                print(f"\n SIGNIFICANT DIFFERENCES FOUND!")
                print(f"   Access times vary significantly between travel modes")
            else:
                print(f"\n No significant differences found")
                print(f"   Access times are similar across travel modes")
                
            return self.results['friedman']
            
        except Exception as e:
            print(f"❌ Error running Friedman test: {e}")
            return None
    
    def post_hoc_analysis(self):
        """
        Perform post-hoc analysis using Wilcoxon signed-rank tests for pairwise comparisons.
        Only runs if Friedman test was significant.
        """
        if 'friedman' not in self.results:
            print("Warning:  Run Friedman test first before post-hoc analysis")
            return None
            
        if self.results['friedman']['p_value'] >= 0.05:
            print(" Friedman test not significant, skipping post-hoc analysis")
            return None
            
        print(f"\n{'='*60}")
        print(f" POST-HOC PAIRWISE COMPARISONS")
        print(f"{'='*60}")
        
        data = self.prepare_friedman_data()
        mode_cols = list(self.mode_columns.values())
        mode_names = list(self.mode_columns.keys())
        
        post_hoc_results = {}
        
        # Perform pairwise Wilcoxon signed-rank tests
        for i in range(len(mode_cols)):
            for j in range(i + 1, len(mode_cols)):
                mode1, mode2 = mode_names[i], mode_names[j]
                col1, col2 = mode_cols[i], mode_cols[j]
                
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(data[col1], data[col2])
                
                # Calculate effect size (r = Z / sqrt(N))
                z_stat = stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0
                effect_size = z_stat / np.sqrt(len(data))
                
                # Determine significance
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = "ns"
                
                # Store results
                comparison_key = f"{mode1}_vs_{mode2}"
                post_hoc_results[comparison_key] = {
                    'mode1': mode1,
                    'mode2': mode2,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significance': significance,
                    'effect_size': effect_size
                }
                
                # Print results
                print(f"\n {mode1.upper()} vs {mode2.upper()}:")
                print(f"   Wilcoxon W: {statistic:.2f}")
                print(f"   P-value: {p_value:.6f}")
                print(f"   Significance: {significance}")
                print(f"   Effect Size (r): {effect_size:.4f}")
                
                # Interpret effect size
                if effect_size >= 0.5:
                    effect_desc = "Large"
                elif effect_size >= 0.3:
                    effect_desc = "Medium"
                elif effect_size >= 0.1:
                    effect_desc = "Small"
                else:
                    effect_desc = "Negligible"
                print(f"   Effect Size: {effect_desc}")
        
        self.results['post_hoc'] = post_hoc_results
        return post_hoc_results
    
    def descriptive_statistics(self):
        """
        Calculate descriptive statistics for each travel mode.
        """
        if self.travel_data is None:
            raise ValueError("No travel data loaded. Call load_travel_data() first.")
            
        data = self.prepare_friedman_data()
        mode_cols = list(self.mode_columns.values())
        mode_names = list(self.mode_columns.keys())
        
        print(f"\n{'='*60}")
        print(f" DESCRIPTIVE STATISTICS")
        print(f"{'='*60}")
        
        desc_stats = {}
        
        for mode, col in zip(mode_names, mode_cols):
            times = data[col].dropna()
            
            stats_dict = {
                'n': len(times),
                'mean': times.mean(),
                'median': times.median(),
                'std': times.std(),
                'min': times.min(),
                'max': times.max(),
                'q25': times.quantile(0.25),
                'q75': times.quantile(0.75)
            }
            
            desc_stats[mode] = stats_dict
            
            print(f"\n {mode.upper()}:")
            print(f"   Count: {stats_dict['n']}")
            print(f"   Mean: {stats_dict['mean']:.2f} minutes")
            print(f"   Median: {stats_dict['median']:.2f} minutes")
            print(f"   Std Dev: {stats_dict['std']:.2f} minutes")
            print(f"   Range: {stats_dict['min']:.1f} - {stats_dict['max']:.1f} minutes")
            print(f"   IQR: {stats_dict['q25']:.1f} - {stats_dict['q75']:.1f} minutes")
        
        self.results['descriptive'] = desc_stats
        return desc_stats
    
    def create_visualizations(self, save_path=None):
        """
        Create visualizations for the Friedman test analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Directory to save plots. If None, saves in current directory.
        """
        if 'friedman' not in self.results:
            print("Warning:  Run Friedman test first before creating visualizations")
            return
            
        if save_path is None:
            script_dir = Path(__file__).parent
            save_path = script_dir
            
        os.makedirs(save_path, exist_ok=True)
        
        data = self.prepare_friedman_data()
        mode_cols = list(self.mode_columns.values())
        mode_names = list(self.mode_columns.keys())
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multimodal Access Time Analysis - Friedman Test Results', 
                     fontsize=16, fontweight='bold')
        
        # 1. Box plot
        ax1 = axes[0, 0]
        box_data = [data[col] for col in mode_cols]
        bp = ax1.boxplot(box_data, labels=[mode.title() for mode in mode_names], 
                        patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax1.set_title('Access Time Distribution by Mode', fontweight='bold')
        ax1.set_ylabel('Time (minutes)')
        ax1.grid(True, alpha=0.3)
        
        # Add significance annotation
        if self.results['friedman']['p_value'] < 0.05:
            ax1.text(0.5, 0.95, f"Friedman Test: p < 0.001" if self.results['friedman']['p_value'] < 0.001 
                    else f"Friedman Test: p = {self.results['friedman']['p_value']:.3f}",
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Violin plot
        ax2 = axes[0, 1]
        violin_data = []
        violin_labels = []
        for mode, col in zip(mode_names, mode_cols):
            violin_data.extend(data[col].dropna())
            violin_labels.extend([mode.title()] * len(data[col].dropna()))
            
        violin_df = pd.DataFrame({'Mode': violin_labels, 'Time': violin_data})
        sns.violinplot(data=violin_df, x='Mode', y='Time', ax=ax2)
        ax2.set_title('Access Time Density by Mode', fontweight='bold')
        ax2.set_ylabel('Time (minutes)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Bar plot of means with error bars
        ax3 = axes[1, 0]
        means = [data[col].mean() for col in mode_cols]
        stds = [data[col].std() for col in mode_cols]
        
        bars = ax3.bar([mode.title() for mode in mode_names], means, 
                      yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax3.set_title('Mean Access Time by Mode (±1 SD)', fontweight='bold')
        ax3.set_ylabel('Time (minutes)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + stds[means.index(mean)],
                    f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Statistical summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"""Friedman Test Results:
        
Chi² = {self.results['friedman']['statistic']:.3f}
p-value = {self.results['friedman']['p_value']:.6f}
Significance: {self.results['friedman']['significance']}
Effect Size (W) = {self.results['friedman']['effect_size_w']:.4f}

Sample Size: {self.results['friedman']['n_tracts']} tracts
Travel Modes: {', '.join(mode_names)}

Interpretation:
{'Significant differences found' if self.results['friedman']['p_value'] < 0.05 else 'No significant differences'}"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(save_path, 'friedman_test_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Visualization saved to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def save_results(self, save_path=None):
        """
        Save all analysis results to CSV files.
        
        Parameters:
        -----------
        save_path : str, optional
            Directory to save results. If None, saves in current directory.
        """
        if save_path is None:
            script_dir = Path(__file__).parent
            save_path = script_dir
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save Friedman test results
        if 'friedman' in self.results:
            friedman_df = pd.DataFrame([self.results['friedman']])
            friedman_path = os.path.join(save_path, 'friedman_test_results.csv')
            friedman_df.to_csv(friedman_path, index=False)
            print(f" Friedman test results saved to: {friedman_path}")
        
        # Save post-hoc results
        if 'post_hoc' in self.results:
            post_hoc_df = pd.DataFrame(self.results['post_hoc']).T
            post_hoc_path = os.path.join(save_path, 'post_hoc_wilcoxon_results.csv')
            post_hoc_df.to_csv(post_hoc_path)
            print(f" Post-hoc results saved to: {post_hoc_path}")
        
        # Save descriptive statistics
        if 'descriptive' in self.results:
            desc_df = pd.DataFrame(self.results['descriptive']).T
            desc_path = os.path.join(save_path, 'descriptive_statistics.csv')
            desc_df.to_csv(desc_path)
            print(f" Descriptive statistics saved to: {desc_path}")
        
        return save_path
    
    def calculate_population_accessibility(self, time_thresholds=None):
        """
        Calculate what percentage of the population lives within different travel time thresholds.
        
        Parameters:
        -----------
        time_thresholds : list, optional
            List of time thresholds in minutes. Default: [10, 20, 30, 40, 50, 60]
            
        Returns:
        --------
        dict
            Dictionary containing accessibility percentages for each mode and threshold
        """
        if self.population_data is None:
            print("Warning:  Population data not loaded. Call load_travel_data() first.")
            return None
            
        if self.travel_data is None:
            print("Warning:  Travel data not loaded. Call load_travel_data() first.")
            return None
        
        if time_thresholds is None:
            time_thresholds = [10, 20, 30, 40, 50, 60]
        
        print(f"\n{'='*60}")
        print(f"  POPULATION ACCESSIBILITY ANALYSIS")
        print(f"{'='*60}")
        
        # Merge travel times with population data
        merged_data = self.travel_data.merge(
            self.population_data[['GEOID', 'total_population']], 
            on='GEOID', 
            how='inner'
        )
        
        # Remove duplicates and missing values
        merged_data = merged_data.drop_duplicates(subset=['GEOID'], keep='first')
        merged_data = merged_data.dropna(subset=['total_population'] + list(self.mode_columns.values()))
        
        print(f" Analyzing {len(merged_data)} tracts with population data")
        total_population = merged_data['total_population'].sum()
        print(f"  Total population: {total_population:,}")
        
        accessibility_results = {}
        
        for threshold in time_thresholds:
            print(f"\n⏰ {threshold} minutes:")
            threshold_results = {}
            
            for mode, col in self.mode_columns.items():
                # Count tracts within threshold
                tracts_within_threshold = (merged_data[col] <= threshold).sum()
                total_tracts = len(merged_data)
                
                # Calculate population within threshold
                population_within_threshold = merged_data[
                    merged_data[col] <= threshold
                ]['total_population'].sum()
                
                # Calculate percentages
                tract_percentage = (tracts_within_threshold / total_tracts) * 100
                population_percentage = (population_within_threshold / total_population) * 100
                
                threshold_results[mode] = {
                    'tracts_within_threshold': tracts_within_threshold,
                    'total_tracts': total_tracts,
                    'tract_percentage': tract_percentage,
                    'population_within_threshold': population_within_threshold,
                    'total_population': total_population,
                    'population_percentage': population_percentage
                }
                
                print(f"   {mode.capitalize():<10}: {tracts_within_threshold}/{total_tracts} tracts ({tract_percentage:.1f}%)")
                print(f"            Population: {population_within_threshold:,}/{total_population:,} ({population_percentage:.1f}%)")
            
            accessibility_results[threshold] = threshold_results
        
        # Store results
        self.results['population_accessibility'] = accessibility_results
        
        return accessibility_results
    
    def save_population_accessibility_results(self, save_path=None):
        """
        Save population accessibility results to CSV files.
        
        Parameters:
        -----------
        save_path : str, optional
            Directory to save results. If None, saves in current directory.
        """
        if 'population_accessibility' not in self.results:
            print("Warning:  Run population accessibility analysis first")
            return None
            
        if save_path is None:
            script_dir = Path(__file__).parent
            save_path = script_dir
            
        os.makedirs(save_path, exist_ok=True)
        
        # Create comprehensive results dataframe
        results_data = []
        
        for threshold, modes in self.results['population_accessibility'].items():
            for mode, data in modes.items():
                results_data.append({
                    'time_threshold_minutes': threshold,
                    'travel_mode': mode,
                    'tracts_within_threshold': data['tracts_within_threshold'],
                    'total_tracts': data['total_tracts'],
                    'tract_percentage': data['tract_percentage'],
                    'population_within_threshold': data['population_within_threshold'],
                    'total_population': data['total_population'],
                    'population_percentage': data['population_percentage']
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = os.path.join(save_path, 'population_accessibility_analysis.csv')
        results_df.to_csv(csv_path, index=False)
        print(f" Population accessibility results saved to: {csv_path}")
        
        # Create summary text file
        txt_path = os.path.join(save_path, 'population_accessibility_summary.txt')
        with open(txt_path, 'w') as f:
            f.write("POPULATION ACCESSIBILITY ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for threshold, modes in self.results['population_accessibility'].items():
                f.write(f"{threshold} minutes:\n")
                for mode, data in modes.items():
                    f.write(f"   {mode.capitalize()}: {data['tracts_within_threshold']}/{data['total_tracts']} tracts ({data['tract_percentage']:.1f}%)\n")
                    f.write(f"            Population: {data['population_within_threshold']:,}/{data['total_population']:,} ({data['population_percentage']:.1f}%)\n")
                f.write("\n")
        
        print(f" Population accessibility summary saved to: {txt_path}")
        
        return csv_path, txt_path
    
    def run_complete_analysis(self, data_path=None, save_results=True, create_plots=True):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to travel times data
        save_results : bool
            Whether to save results to files
        create_plots : bool
            Whether to create and save visualizations
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        print(" Starting complete multimodal access time analysis...")
        
        # Load data
        self.load_travel_data(data_path)
        
        # Run Friedman test
        friedman_results = self.run_friedman_test()
        
        # Run post-hoc analysis if significant
        if friedman_results and friedman_results['p_value'] < 0.05:
            self.post_hoc_analysis()
        
        # Calculate descriptive statistics
        self.descriptive_statistics()
        
        # Create visualizations
        if create_plots:
            self.create_visualizations()
        
        # Run population accessibility analysis if population data is available
        if self.population_data is not None:
            print(f"\n  Running population accessibility analysis...")
            self.calculate_population_accessibility()
        
        # Save results
        if save_results:
            self.save_results()
            if self.population_data is not None:
                self.save_population_accessibility_results()
        
        print(f"\n Analysis complete! Results stored in self.results")
        return self.results


def main():
    """
    Main function to run the analysis when script is executed directly.
    """
    # Initialize analyzer
    analyzer = AccessTimeAnalyzer()
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f" ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        if 'friedman' in results:
            friedman = results['friedman']
            print(f" Friedman Test: {'SIGNIFICANT' if friedman['p_value'] < 0.05 else 'Not significant'}")
            print(f"   Chi² = {friedman['statistic']:.3f}, p = {friedman['p_value']:.6f}")
            print(f"   Effect Size (W) = {friedman['effect_size_w']:.4f}")
        
        if 'post_hoc' in results:
            print(f" Post-hoc tests: {len(results['post_hoc'])} pairwise comparisons")
        
        print(f" Results saved to: {Path(__file__).parent}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
