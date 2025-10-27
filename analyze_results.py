"""
Results Analysis Script for Prompt Optimization Framework
Provides detailed analysis and additional visualizations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

class ResultsAnalyzer:
    """Analyze and visualize prompt optimization results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.data = self.load_all_results()
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup matplotlib and seaborn settings"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        
    def load_all_results(self) -> pd.DataFrame:
        """Load all results from directory"""
        all_data = []
        
        # Find all result directories
        if os.path.exists(self.results_dir):
            for run_dir in os.listdir(self.results_dir):
                run_path = os.path.join(self.results_dir, run_dir)
                if os.path.isdir(run_path):
                    csv_path = os.path.join(run_path, 'results_summary.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        df['run_id'] = run_dir
                        all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            print("No results found!")
            return pd.DataFrame()
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics"""
        if self.data.empty:
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print("="*80)
        
        # Overall statistics
        print("\n📊 OVERALL STATISTICS")
        print("-"*40)
        print(f"Total Runs: {self.data['run_id'].nunique()}")
        print(f"Total Evaluations: {len(self.data)}")
        print(f"Models Tested: {self.data['model'].nunique()}")
        print(f"Datasets Used: {self.data['dataset'].nunique()}")
        print(f"Strategies Employed: {self.data['strategy'].nunique()}")
        
        print(f"\nFitness Scores:")
        print(f"  Maximum: {self.data['fitness'].max():.4f}")
        print(f"  Average: {self.data['fitness'].mean():.4f}")
        print(f"  Median: {self.data['fitness'].median():.4f}")
        print(f"  Std Dev: {self.data['fitness'].std():.4f}")
        
        # Model performance
        print("\n🤖 MODEL PERFORMANCE")
        print("-"*40)
        model_stats = self.data.groupby('model')['fitness'].agg([
            'mean', 'max', 'min', 'std', 'count'
        ]).round(4)
        print(model_stats)
        
        # Dataset difficulty
        print("\n📚 DATASET ANALYSIS")
        print("-"*40)
        dataset_stats = self.data.groupby('dataset')['fitness'].agg([
            'mean', 'max', 'min', 'std', 'count'
        ]).round(4)
        print(dataset_stats)
        
        # Strategy effectiveness
        print("\n🎯 STRATEGY EFFECTIVENESS")
        print("-"*40)
        strategy_stats = self.data.groupby('strategy')['fitness'].agg([
            'mean', 'max', 'min', 'std', 'count'
        ]).round(4)
        print(strategy_stats)
        
        # Best combinations
        print("\n🏆 TOP 10 BEST CONFIGURATIONS")
        print("-"*40)
        top_configs = self.data.nlargest(10, 'fitness')[
            ['model', 'dataset', 'strategy', 'fitness', 'generation']
        ]
        for idx, row in top_configs.iterrows():
            print(f"{idx+1}. {row['model'].split('/')[-1]} | {row['dataset']} | "
                  f"{row['strategy']} | Fitness: {row['fitness']:.4f} | Gen: {row['generation']}")
        
        # Convergence analysis
        print("\n⚡ CONVERGENCE ANALYSIS")
        print("-"*40)
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            
            # Find average generation to reach 90% of max fitness
            max_fitness = model_data.groupby('generation')['fitness'].max()
            if len(max_fitness) > 0:
                threshold = 0.9 * max_fitness.max()
                convergence_gen = None
                
                for gen in sorted(max_fitness.index):
                    if max_fitness[gen] >= threshold:
                        convergence_gen = gen
                        break
                
                print(f"{model.split('/')[-1]}: "
                      f"Gen {convergence_gen if convergence_gen else 'N/A'} "
                      f"(Max: {max_fitness.max():.4f})")
    
    def create_advanced_visualizations(self, output_dir: str = "advanced_analysis"):
        """Create advanced analysis visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.data.empty:
            print("No data to visualize!")
            return
        
        print("\n📈 Generating advanced visualizations...")
        
        # 1. Strategy Evolution Across Generations
        self._plot_strategy_evolution(output_dir)
        
        # 2. Model Learning Curves
        self._plot_learning_curves(output_dir)
        
        # 3. Performance Distribution Analysis
        self._plot_performance_distributions(output_dir)
        
        # 4. Cross-Model Strategy Comparison
        self._plot_cross_model_strategy(output_dir)
        
        # 5. Dataset Difficulty Heatmap
        self._plot_dataset_difficulty(output_dir)
        
        # 6. Optimization Efficiency
        self._plot_optimization_efficiency(output_dir)
        
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def _plot_strategy_evolution(self, output_dir: str):
        """Plot how strategy effectiveness changes over generations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Strategy Evolution Across Generations', fontsize=16)
        
        strategies = self.data['strategy'].unique()[:6]  # Top 6 strategies
        
        for idx, strategy in enumerate(strategies):
            ax = axes[idx // 3, idx % 3]
            strategy_data = self.data[self.data['strategy'] == strategy]
            
            for model in self.data['model'].unique():
                model_strategy_data = strategy_data[strategy_data['model'] == model]
                if not model_strategy_data.empty:
                    gen_performance = model_strategy_data.groupby('generation')['fitness'].mean()
                    ax.plot(gen_performance.index, gen_performance.values, 
                           marker='o', label=model.split('/')[-1], alpha=0.7)
            
            ax.set_title(f'Strategy: {strategy}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Average Fitness')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self, output_dir: str):
        """Plot learning curves for each model"""
        n_models = self.data['model'].nunique()
        fig, axes = plt.subplots(1, min(n_models, 3), figsize=(15, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, model in enumerate(self.data['model'].unique()[:3]):
            ax = axes[idx] if n_models > 1 else axes[0]
            model_data = self.data[self.data['model'] == model]
            
            # Plot for each dataset
            for dataset in self.data['dataset'].unique():
                dataset_data = model_data[model_data['dataset'] == dataset]
                if not dataset_data.empty:
                    gen_stats = dataset_data.groupby('generation')['fitness'].agg(['mean', 'std', 'max'])
                    
                    # Plot mean with confidence interval
                    ax.plot(gen_stats.index, gen_stats['mean'], label=f'{dataset} (mean)', marker='o')
                    ax.fill_between(gen_stats.index,
                                   gen_stats['mean'] - gen_stats['std'],
                                   gen_stats['mean'] + gen_stats['std'],
                                   alpha=0.2)
                    ax.plot(gen_stats.index, gen_stats['max'], 
                           label=f'{dataset} (max)', linestyle='--', alpha=0.7)
            
            ax.set_title(f'{model.split("/")[-1]}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Learning Curves by Model and Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distributions(self, output_dir: str):
        """Plot fitness score distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall distribution
        ax = axes[0, 0]
        ax.hist(self.data['fitness'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(self.data['fitness'].mean(), color='red', linestyle='--', label='Mean')
        ax.axvline(self.data['fitness'].median(), color='green', linestyle='--', label='Median')
        ax.set_title('Overall Fitness Distribution')
        ax.set_xlabel('Fitness Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Distribution by model
        ax = axes[0, 1]
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]['fitness']
            ax.hist(model_data, bins=30, alpha=0.5, label=model.split('/')[-1])
        ax.set_title('Fitness Distribution by Model')
        ax.set_xlabel('Fitness Score')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        
        # Distribution by strategy
        ax = axes[1, 0]
        self.data.boxplot(column='fitness', by='strategy', ax=ax)
        ax.set_title('Fitness Distribution by Strategy')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Fitness Score')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        # Distribution by generation (violin plot)
        ax = axes[1, 1]
        generation_data = []
        generation_labels = []
        for gen in sorted(self.data['generation'].unique())[:10]:  # First 10 generations
            gen_data = self.data[self.data['generation'] == gen]['fitness'].values
            if len(gen_data) > 0:
                generation_data.append(gen_data)
                generation_labels.append(str(gen))
        
        if generation_data:
            parts = ax.violinplot(generation_data, positions=range(len(generation_data)), 
                                 widths=0.7, showmeans=True, showmedians=True)
            ax.set_xticks(range(len(generation_labels)))
            ax.set_xticklabels(generation_labels)
            ax.set_title('Fitness Distribution Evolution')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_model_strategy(self, output_dir: str):
        """Plot cross-model strategy effectiveness"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap of average performance
        ax = axes[0]
        pivot_avg = self.data.pivot_table(values='fitness', index='strategy', 
                                         columns='model', aggfunc='mean')
        
        # Clean model names for display
        pivot_avg.columns = [col.split('/')[-1] for col in pivot_avg.columns]
        
        sns.heatmap(pivot_avg, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, ax=ax, cbar_kws={'label': 'Average Fitness'})
        ax.set_title('Average Strategy Performance Across Models')
        
        # Heatmap of maximum performance
        ax = axes[1]
        pivot_max = self.data.pivot_table(values='fitness', index='strategy', 
                                         columns='model', aggfunc='max')
        pivot_max.columns = [col.split('/')[-1] for col in pivot_max.columns]
        
        sns.heatmap(pivot_max, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0.5, ax=ax, cbar_kws={'label': 'Maximum Fitness'})
        ax.set_title('Maximum Strategy Performance Across Models')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_model_strategy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dataset_difficulty(self, output_dir: str):
        """Analyze and visualize dataset difficulty"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Dataset performance by model
        ax = axes[0, 0]
        pivot_data = self.data.pivot_table(values='fitness', index='dataset', 
                                          columns='model', aggfunc='mean')
        pivot_data.columns = [col.split('/')[-1] for col in pivot_data.columns]
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_title('Dataset Performance by Model')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Average Fitness')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        # Dataset variance analysis
        ax = axes[0, 1]
        dataset_variance = self.data.groupby('dataset')['fitness'].var().sort_values()
        dataset_variance.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Dataset Score Variance (Difficulty Indicator)')
        ax.set_xlabel('Variance in Fitness Scores')
        ax.set_ylabel('Dataset')
        
        # Time to convergence by dataset
        ax = axes[1, 0]
        convergence_data = []
        for dataset in self.data['dataset'].unique():
            dataset_df = self.data[self.data['dataset'] == dataset]
            max_fitness_by_gen = dataset_df.groupby('generation')['fitness'].max()
            
            if len(max_fitness_by_gen) > 0:
                final_max = max_fitness_by_gen.max()
                threshold = 0.9 * final_max
                
                for gen in sorted(max_fitness_by_gen.index):
                    if max_fitness_by_gen[gen] >= threshold:
                        convergence_data.append({'dataset': dataset, 'generation': gen})
                        break
        
        if convergence_data:
            conv_df = pd.DataFrame(convergence_data)
            conv_df.plot(x='dataset', y='generation', kind='bar', ax=ax, legend=False)
            ax.set_title('Generations to 90% Max Fitness')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Generation')
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        # Score distribution comparison
        ax = axes[1, 1]
        datasets = self.data['dataset'].unique()
        positions = range(1, len(datasets) + 1)
        dataset_scores = [self.data[self.data['dataset'] == d]['fitness'].values for d in datasets]
        
        bp = ax.boxplot(dataset_scores, positions=positions, labels=datasets, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(datasets)))):
            patch.set_facecolor(color)
        
        ax.set_title('Score Distribution by Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Fitness Score')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_difficulty.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_efficiency(self, output_dir: str):
        """Analyze optimization efficiency"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Improvement rate over generations
        ax = axes[0, 0]
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            gen_improvement = model_data.groupby('generation')['fitness'].max().pct_change()
            ax.plot(gen_improvement.index[1:], gen_improvement.values[1:] * 100, 
                   marker='o', label=model.split('/')[-1])
        
        ax.set_title('Fitness Improvement Rate')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Improvement (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Population diversity (std dev) over generations
        ax = axes[0, 1]
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            gen_diversity = model_data.groupby('generation')['fitness'].std()
            ax.plot(gen_diversity.index, gen_diversity.values, 
                   marker='s', label=model.split('/')[-1])
        
        ax.set_title('Population Diversity Over Time')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Std Dev')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Strategy switching effectiveness
        ax = axes[1, 0]
        # Count strategy changes that led to improvement
        strategy_effectiveness = self.data.groupby(['strategy', 'generation'])['fitness'].mean().reset_index()
        pivot_strategy = strategy_effectiveness.pivot(index='generation', columns='strategy', values='fitness')
        
        # Calculate correlation between strategies
        strategy_corr = pivot_strategy.corr()
        sns.heatmap(strategy_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Strategy Correlation Matrix')
        
        # Efficiency summary
        ax = axes[1, 1]
        efficiency_metrics = []
        
        for model in self.data['model'].unique():
            model_data = self.data[self.data['model'] == model]
            
            # Calculate metrics
            max_fitness = model_data['fitness'].max()
            generations_to_max = model_data[model_data['fitness'] == max_fitness]['generation'].min()
            avg_fitness = model_data['fitness'].mean()
            final_gen_avg = model_data[model_data['generation'] == model_data['generation'].max()]['fitness'].mean()
            
            efficiency_metrics.append({
                'Model': model.split('/')[-1],
                'Max Fitness': max_fitness,
                'Gen to Max': generations_to_max,
                'Avg Fitness': avg_fitness,
                'Final Gen Avg': final_gen_avg,
                'Efficiency': max_fitness / (generations_to_max + 1) if generations_to_max else 0
            })
        
        eff_df = pd.DataFrame(efficiency_metrics)
        
        # Normalize metrics for radar chart
        from math import pi
        categories = ['Max Fitness', 'Efficiency', 'Avg Fitness', 'Final Gen Avg']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(224, projection='polar')
        
        for idx, row in eff_df.iterrows():
            values = [row['Max Fitness'], row['Efficiency'] * 10, 
                     row['Avg Fitness'], row['Final Gen Avg']]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_title('Optimization Efficiency Radar', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_detailed_report(self, output_file: str = "detailed_report.md"):
        """Export detailed markdown report"""
        if self.data.empty:
            print("No data to export!")
            return
        
        with open(output_file, 'w') as f:
            f.write("# Prompt Optimization Framework - Detailed Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Experiments**: {self.data['run_id'].nunique()}\n")
            f.write(f"- **Total Evaluations**: {len(self.data)}\n")
            f.write(f"- **Best Fitness Achieved**: {self.data['fitness'].max():.4f}\n")
            f.write(f"- **Average Fitness**: {self.data['fitness'].mean():.4f}\n\n")
            
            # Best Configuration
            best_row = self.data.loc[self.data['fitness'].idxmax()]
            f.write("### Best Configuration Found\n\n")
            f.write(f"- **Model**: {best_row['model']}\n")
            f.write(f"- **Dataset**: {best_row['dataset']}\n")
            f.write(f"- **Strategy**: {best_row['strategy']}\n")
            f.write(f"- **Fitness**: {best_row['fitness']:.4f}\n")
            f.write(f"- **Generation**: {best_row['generation']}\n\n")
            
            # Model Analysis
            f.write("## Model Performance Analysis\n\n")
            model_stats = self.data.groupby('model')['fitness'].agg(['mean', 'max', 'min', 'std', 'count'])
            f.write("| Model | Mean | Max | Min | Std Dev | Count |\n")
            f.write("|-------|------|-----|-----|---------|-------|\n")
            for model, row in model_stats.iterrows():
                f.write(f"| {model.split('/')[-1]} | {row['mean']:.4f} | {row['max']:.4f} | "
                       f"{row['min']:.4f} | {row['std']:.4f} | {row['count']} |\n")
            f.write("\n")
            
            # Dataset Analysis
            f.write("## Dataset Difficulty Analysis\n\n")
            dataset_stats = self.data.groupby('dataset')['fitness'].agg(['mean', 'max', 'min', 'std'])
            f.write("| Dataset | Mean | Max | Min | Std Dev | Difficulty |\n")
            f.write("|---------|------|-----|-----|---------|------------|\n")
            
            # Classify difficulty based on average score
            for dataset, row in dataset_stats.iterrows():
                if row['mean'] < 0.3:
                    difficulty = "Very Hard"
                elif row['mean'] < 0.5:
                    difficulty = "Hard"
                elif row['mean'] < 0.7:
                    difficulty = "Medium"
                else:
                    difficulty = "Easy"
                
                f.write(f"| {dataset} | {row['mean']:.4f} | {row['max']:.4f} | "
                       f"{row['min']:.4f} | {row['std']:.4f} | {difficulty} |\n")
            f.write("\n")
            
            # Strategy Analysis
            f.write("## Strategy Effectiveness\n\n")
            strategy_stats = self.data.groupby('strategy')['fitness'].agg(['mean', 'max', 'count'])
            strategy_stats = strategy_stats.sort_values('mean', ascending=False)
            
            f.write("| Rank | Strategy | Mean Score | Max Score | Usage Count |\n")
            f.write("|------|----------|------------|-----------|-------------|\n")
            for rank, (strategy, row) in enumerate(strategy_stats.iterrows(), 1):
                f.write(f"| {rank} | {strategy} | {row['mean']:.4f} | "
                       f"{row['max']:.4f} | {row['count']} |\n")
            f.write("\n")
            
            # Convergence Analysis
            f.write("## Convergence Analysis\n\n")
            f.write("### Generations to 90% of Maximum Fitness\n\n")
            
            for model in self.data['model'].unique():
                model_data = self.data[self.data['model'] == model]
                f.write(f"**{model.split('/')[-1]}**:\n")
                
                for dataset in self.data['dataset'].unique():
                    dataset_data = model_data[model_data['dataset'] == dataset]
                    if not dataset_data.empty:
                        max_by_gen = dataset_data.groupby('generation')['fitness'].max()
                        if len(max_by_gen) > 0:
                            threshold = 0.9 * max_by_gen.max()
                            conv_gen = None
                            for gen in sorted(max_by_gen.index):
                                if max_by_gen[gen] >= threshold:
                                    conv_gen = gen
                                    break
                            f.write(f"  - {dataset}: Generation {conv_gen if conv_gen else 'N/A'}\n")
                f.write("\n")
            
            # Key Insights
            f.write("## Key Insights\n\n")
            
            # Find most consistent strategy
            strategy_consistency = self.data.groupby('strategy')['fitness'].std()
            most_consistent = strategy_consistency.idxmin()
            f.write(f"1. **Most Consistent Strategy**: {most_consistent} "
                   f"(σ = {strategy_consistency[most_consistent]:.4f})\n")
            
            # Find best model-dataset pair
            pair_performance = self.data.groupby(['model', 'dataset'])['fitness'].max()
            best_pair = pair_performance.idxmax()
            f.write(f"2. **Best Model-Dataset Pair**: {best_pair[0].split('/')[-1]} on "
                   f"{best_pair[1]} ({pair_performance[best_pair]:.4f})\n")
            
            # Strategy distribution in top performers
            top_10_percent = self.data.nlargest(int(len(self.data) * 0.1), 'fitness')
            top_strategy = top_10_percent['strategy'].value_counts().index[0]
            f.write(f"3. **Most Common Strategy in Top 10%**: {top_strategy}\n")
            
            # Average improvement from first to last generation
            first_gen = self.data[self.data['generation'] == self.data['generation'].min()]['fitness'].mean()
            last_gen = self.data[self.data['generation'] == self.data['generation'].max()]['fitness'].mean()
            improvement = ((last_gen - first_gen) / first_gen * 100) if first_gen > 0 else 0
            f.write(f"4. **Average Improvement**: {improvement:.2f}% from first to last generation\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis:\n\n")
            
            # Model recommendation
            best_model = self.data.groupby('model')['fitness'].mean().idxmax()
            f.write(f"1. **Preferred Model**: {best_model.split('/')[-1]} shows the best average performance\n")
            
            # Strategy recommendation
            f.write(f"2. **Preferred Strategy**: {strategy_stats.index[0]} achieves highest average scores\n")
            
            # Dataset insights
            easiest_dataset = self.data.groupby('dataset')['fitness'].mean().idxmax()
            f.write(f"3. **Dataset Suitability**: {easiest_dataset} appears most suitable for prompt optimization\n")
            
            # Optimization settings
            optimal_gen = None
            for gen in range(1, self.data['generation'].max() + 1):
                gen_improvement = (self.data[self.data['generation'] == gen]['fitness'].mean() - 
                                 self.data[self.data['generation'] == gen-1]['fitness'].mean() 
                                 if gen > 1 else float('inf'))
                if gen_improvement < 0.01:  # Less than 1% improvement
                    optimal_gen = gen
                    break
            
            if optimal_gen:
                f.write(f"4. **Optimization Settings**: Consider stopping at generation {optimal_gen} "
                       f"(diminishing returns observed)\n")
            
            f.write("\n---\n")
            f.write(f"*Report generated by Prompt Optimization Framework*\n")
        
        print(f"✅ Detailed report exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Prompt Optimization Results")
    parser.add_argument("--results-dir", type=str, 
                       default="prompt_optimization_results",
                       help="Directory containing results")
    parser.add_argument("--output-dir", type=str,
                       default="advanced_analysis",
                       help="Output directory for analysis")
    parser.add_argument("--export-report", action="store_true",
                       help="Export detailed markdown report")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    if analyzer.data.empty:
        print("No results found to analyze!")
        return
    
    # Print summary statistics
    analyzer.print_summary_statistics()
    
    # Create visualizations
    analyzer.create_advanced_visualizations(args.output_dir)
    
    # Export report if requested
    if args.export_report:
        report_path = os.path.join(args.output_dir, "detailed_report.md")
        analyzer.export_detailed_report(report_path)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()