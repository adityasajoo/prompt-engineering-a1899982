#!/usr/bin/env python3
"""
Comprehensive Report Generator for Prompt Optimization Results
Generates detailed case-by-case analysis with hierarchical structure
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveReportGenerator:
    """Generate detailed hierarchical reports"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.data = self.load_results()
        
    def load_results(self) -> pd.DataFrame:
        """Load all results"""
        all_data = []
        
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
    
    def generate_comprehensive_report(self, output_file: str = "comprehensive_report.md"):
        """Generate the main comprehensive report"""
        
        if self.data.empty:
            print("No data available for report generation")
            return
        
        with open(output_file, 'w') as f:
            self._write_header(f)
            self._write_executive_summary(f)
            
            # Main case structure: Dataset-wise analysis
            for dataset_idx, dataset in enumerate(sorted(self.data['dataset'].unique()), 1):
                self._write_dataset_case(f, dataset, dataset_idx)
            
            self._write_cross_analysis(f)
            self._write_recommendations(f)
        
        print(f"✅ Comprehensive report generated: {output_file}")
    
    def _write_header(self, f):
        """Write report header"""
        f.write("# COMPREHENSIVE PROMPT OPTIMIZATION ANALYSIS REPORT\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiments:** {len(self.data)}\n")
        f.write(f"**Datasets:** {self.data['dataset'].nunique()}\n")
        f.write(f"**Models:** {self.data['model'].nunique()}\n")
        f.write(f"**Strategies:** {self.data['strategy'].nunique()}\n")
        f.write(f"**Generations:** {self.data['generation'].max() + 1}\n\n")
        f.write("---\n\n")
    
    def _write_executive_summary(self, f):
        """Write executive summary"""
        f.write("## EXECUTIVE SUMMARY\n\n")
        
        # Overall best result
        best_idx = self.data['fitness'].idxmax()
        best_row = self.data.loc[best_idx]
        
        f.write("### Best Overall Result\n\n")
        f.write(f"- **Fitness Score:** {best_row['fitness']:.4f}\n")
        f.write(f"- **Model:** {best_row['model']}\n")
        f.write(f"- **Dataset:** {best_row['dataset']}\n")
        f.write(f"- **Strategy:** {best_row['strategy']}\n")
        f.write(f"- **Generation:** {best_row['generation']}\n\n")
        
        # Performance statistics
        f.write("### Overall Performance Statistics\n\n")
        f.write(f"- **Mean Fitness:** {self.data['fitness'].mean():.4f}\n")
        f.write(f"- **Median Fitness:** {self.data['fitness'].median():.4f}\n")
        f.write(f"- **Std Deviation:** {self.data['fitness'].std():.4f}\n")
        f.write(f"- **Min Fitness:** {self.data['fitness'].min():.4f}\n")
        f.write(f"- **Max Fitness:** {self.data['fitness'].max():.4f}\n\n")
        
        f.write("---\n\n")
    
    def _write_dataset_case(self, f, dataset: str, case_num: int):
        """Write detailed case analysis for a dataset"""
        f.write(f"## CASE {case_num}: DATASET - {dataset.upper()}\n\n")
        
        dataset_data = self.data[self.data['dataset'] == dataset]
        
        # Dataset overview
        f.write("### Overview\n\n")
        f.write(f"- **Total Evaluations:** {len(dataset_data)}\n")
        f.write(f"- **Best Fitness:** {dataset_data['fitness'].max():.4f}\n")
        f.write(f"- **Average Fitness:** {dataset_data['fitness'].mean():.4f}\n")
        f.write(f"- **Worst Fitness:** {dataset_data['fitness'].min():.4f}\n\n")
        
        # 1. Model-wise Performance
        self._write_model_analysis(f, dataset_data, dataset)
        
        # 2. Strategy-wise Performance
        self._write_strategy_analysis(f, dataset_data, dataset)
        
        # 3. Generation Evolution Analysis
        self._write_generation_analysis(f, dataset_data, dataset)
        
        # 4. Mutation Impact Analysis
        self._write_mutation_analysis(f, dataset_data, dataset)
        
        # 5. Convergence Analysis
        self._write_convergence_analysis(f, dataset_data, dataset)
        
        f.write("\n---\n\n")
    
    def _write_model_analysis(self, f, dataset_data: pd.DataFrame, dataset: str):
        """Write model-wise analysis"""
        f.write("### 1. Model-wise Performance\n\n")
        
        model_stats = dataset_data.groupby('model')['fitness'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        f.write("#### Statistical Summary\n\n")
        f.write("| Model | Count | Mean | Std Dev | Min | Max |\n")
        f.write("|-------|-------|------|---------|-----|-----|\n")
        
        for model, row in model_stats.iterrows():
            model_name = model.split('/')[-1]
            f.write(f"| {model_name} | {int(row['count'])} | {row['mean']:.4f} | "
                   f"{row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
        
        f.write("\n")
        
        # Detailed per-model breakdown
        f.write("#### Detailed Model Results\n\n")
        for model_idx, model in enumerate(sorted(dataset_data['model'].unique()), 1):
            model_data = dataset_data[dataset_data['model'] == model]
            model_name = model.split('/')[-1]
            
            f.write(f"##### {model_idx}. {model_name}\n\n")
            f.write(f"- **Dataset:** {dataset}\n")
            f.write(f"- **Total Evaluations:** {len(model_data)}\n")
            f.write(f"- **Best Score:** {model_data['fitness'].max():.4f}\n")
            f.write(f"- **Average Score:** {model_data['fitness'].mean():.4f}\n")
            f.write(f"- **Consistency (Std):** {model_data['fitness'].std():.4f}\n")
            
            # Best configuration for this model
            best_idx = model_data['fitness'].idxmax()
            best = model_data.loc[best_idx]
            f.write(f"- **Best Strategy:** {best['strategy']}\n")
            f.write(f"- **Best Generation:** {best['generation']}\n\n")
        
        # Ranking
        f.write("#### Model Ranking (by Mean Fitness)\n\n")
        model_ranking = model_stats.sort_values('mean', ascending=False)
        for rank, (model, row) in enumerate(model_ranking.iterrows(), 1):
            model_name = model.split('/')[-1]
            f.write(f"{rank}. **{model_name}** - {row['mean']:.4f}\n")
        
        f.write("\n")
    
    def _write_strategy_analysis(self, f, dataset_data: pd.DataFrame, dataset: str):
        """Write strategy-wise analysis"""
        f.write("### 2. Strategy-wise Performance\n\n")
        
        strategy_stats = dataset_data.groupby('strategy')['fitness'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        f.write("#### Statistical Summary\n\n")
        f.write("| Strategy | Usage | Mean | Std Dev | Min | Max |\n")
        f.write("|----------|-------|------|---------|-----|-----|\n")
        
        for strategy, row in strategy_stats.iterrows():
            f.write(f"| {strategy} | {int(row['count'])} | {row['mean']:.4f} | "
                   f"{row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
        
        f.write("\n")
        
        # Strategy effectiveness by model
        f.write("#### Strategy Performance by Model\n\n")
        pivot = dataset_data.pivot_table(
            values='fitness', 
            index='strategy', 
            columns='model', 
            aggfunc='mean'
        ).round(4)
        
        f.write("| Strategy | " + " | ".join([m.split('/')[-1] for m in pivot.columns]) + " |\n")
        f.write("|----------|" + "|".join(["-----"] * len(pivot.columns)) + "|\n")
        
        for strategy, row in pivot.iterrows():
            values = " | ".join([f"{v:.4f}" if not pd.isna(v) else "N/A" for v in row])
            f.write(f"| {strategy} | {values} |\n")
        
        f.write("\n")
        
        # Best strategy
        best_strategy = strategy_stats['mean'].idxmax()
        f.write(f"**Best Strategy Overall:** {best_strategy} "
               f"({strategy_stats.loc[best_strategy, 'mean']:.4f})\n\n")
    
    def _write_generation_analysis(self, f, dataset_data: pd.DataFrame, dataset: str):
        """Write generation evolution analysis"""
        f.write("### 3. Generation Evolution\n\n")
        
        gen_stats = dataset_data.groupby('generation')['fitness'].agg([
            'mean', 'max', 'std'
        ]).round(4)
        
        f.write("#### Fitness Progress by Generation\n\n")
        f.write("| Generation | Mean | Max | Std Dev | Improvement |\n")
        f.write("|------------|------|-----|---------|-------------|\n")
        
        for gen, row in gen_stats.iterrows():
            if gen == 0:
                improvement = "Baseline"
            else:
                prev_mean = gen_stats.loc[gen-1, 'mean']
                improvement = f"{((row['mean'] - prev_mean) / prev_mean * 100):.2f}%"
            
            f.write(f"| {gen} | {row['mean']:.4f} | {row['max']:.4f} | "
                   f"{row['std']:.4f} | {improvement} |\n")
        
        f.write("\n")
        
        # Learning velocity
        if len(gen_stats) > 1:
            initial_fitness = gen_stats.iloc[0]['mean']
            final_fitness = gen_stats.iloc[-1]['mean']
            total_improvement = ((final_fitness - initial_fitness) / initial_fitness * 100)
            
            f.write("#### Learning Statistics\n\n")
            f.write(f"- **Initial Fitness (Gen 0):** {initial_fitness:.4f}\n")
            f.write(f"- **Final Fitness (Gen {len(gen_stats)-1}):** {final_fitness:.4f}\n")
            f.write(f"- **Total Improvement:** {total_improvement:.2f}%\n")
            f.write(f"- **Average Improvement per Generation:** {total_improvement/len(gen_stats):.2f}%\n\n")
    
    def _write_mutation_analysis(self, f, dataset_data: pd.DataFrame, dataset: str):
        """Analyze mutation and crossover effects"""
        f.write("### 4. Mutation and Evolution Analysis\n\n")
        
        # Track fitness improvements between generations
        improvements = []
        for gen in range(1, dataset_data['generation'].max() + 1):
            prev_gen = dataset_data[dataset_data['generation'] == gen - 1]
            curr_gen = dataset_data[dataset_data['generation'] == gen]
            
            if not prev_gen.empty and not curr_gen.empty:
                prev_max = prev_gen['fitness'].max()
                curr_max = curr_gen['fitness'].max()
                improvement = curr_max - prev_max
                improvements.append({
                    'generation': gen,
                    'improvement': improvement,
                    'prev_max': prev_max,
                    'curr_max': curr_max
                })
        
        if improvements:
            f.write("#### Generation-to-Generation Improvements\n\n")
            f.write("| Generation | Previous Best | Current Best | Improvement | Status |\n")
            f.write("|------------|---------------|--------------|-------------|--------|\n")
            
            for imp in improvements:
                status = "✓ Improved" if imp['improvement'] > 0 else ("✗ Degraded" if imp['improvement'] < 0 else "= Stable")
                f.write(f"| {imp['generation']} | {imp['prev_max']:.4f} | "
                       f"{imp['curr_max']:.4f} | {imp['improvement']:+.4f} | {status} |\n")
            
            f.write("\n")
            
            # Success rate
            positive_improvements = sum(1 for imp in improvements if imp['improvement'] > 0)
            success_rate = (positive_improvements / len(improvements) * 100)
            f.write(f"**Evolution Success Rate:** {success_rate:.1f}% "
                   f"({positive_improvements}/{len(improvements)} generations showed improvement)\n\n")
    
    def _write_convergence_analysis(self, f, dataset_data: pd.DataFrame, dataset: str):
        """Analyze convergence speed and stability"""
        f.write("### 5. Convergence Analysis\n\n")
        
        for model in dataset_data['model'].unique():
            model_data = dataset_data[dataset_data['model'] == model]
            model_name = model.split('/')[-1]
            
            # Find when 90% and 95% of max fitness achieved
            max_fitness = model_data['fitness'].max()
            gen_max_fitness = model_data.groupby('generation')['fitness'].max()
            
            convergence_90 = None
            convergence_95 = None
            
            for gen in sorted(gen_max_fitness.index):
                if gen_max_fitness[gen] >= 0.90 * max_fitness and convergence_90 is None:
                    convergence_90 = gen
                if gen_max_fitness[gen] >= 0.95 * max_fitness and convergence_95 is None:
                    convergence_95 = gen
            
            f.write(f"#### {model_name}\n\n")
            f.write(f"- **Maximum Fitness:** {max_fitness:.4f}\n")
            f.write(f"- **Reached 90% of Max:** Generation {convergence_90 if convergence_90 is not None else 'Not reached'}\n")
            f.write(f"- **Reached 95% of Max:** Generation {convergence_95 if convergence_95 is not None else 'Not reached'}\n")
            
            # Stability (variance in last 25% of generations)
            total_gens = model_data['generation'].max() + 1
            late_gens = model_data[model_data['generation'] >= total_gens * 0.75]
            if not late_gens.empty:
                late_variance = late_gens['fitness'].std()
                f.write(f"- **Late-stage Stability (Std Dev):** {late_variance:.4f}\n")
            
            f.write("\n")
    
    def _write_cross_analysis(self, f):
        """Write cross-cutting analysis"""
        f.write("## CROSS-DATASET ANALYSIS\n\n")
        
        # Model performance across datasets
        f.write("### Model Performance Across All Datasets\n\n")
        overall_model = self.data.groupby('model')['fitness'].agg([
            'count', 'mean', 'std', 'max'
        ]).round(4)
        overall_model = overall_model.sort_values('mean', ascending=False)
        
        f.write("| Rank | Model | Evaluations | Mean | Std Dev | Max |\n")
        f.write("|------|-------|-------------|------|---------|-----|\n")
        
        for rank, (model, row) in enumerate(overall_model.iterrows(), 1):
            model_name = model.split('/')[-1]
            f.write(f"| {rank} | {model_name} | {int(row['count'])} | "
                   f"{row['mean']:.4f} | {row['std']:.4f} | {row['max']:.4f} |\n")
        
        f.write("\n")
        
        # Strategy performance across datasets
        f.write("### Strategy Performance Across All Datasets\n\n")
        overall_strategy = self.data.groupby('strategy')['fitness'].agg([
            'count', 'mean', 'std', 'max'
        ]).round(4)
        overall_strategy = overall_strategy.sort_values('mean', ascending=False)
        
        f.write("| Rank | Strategy | Usage | Mean | Std Dev | Max |\n")
        f.write("|------|----------|-------|------|---------|-----|\n")
        
        for rank, (strategy, row) in enumerate(overall_strategy.iterrows(), 1):
            f.write(f"| {rank} | {strategy} | {int(row['count'])} | "
                   f"{row['mean']:.4f} | {row['std']:.4f} | {row['max']:.4f} |\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f):
        """Write actionable recommendations"""
        f.write("## RECOMMENDATIONS\n\n")
        
        # Best model
        best_model = self.data.groupby('model')['fitness'].mean().idxmax()
        f.write(f"### 1. Model Selection\n\n")
        f.write(f"**Recommended Model:** {best_model.split('/')[-1]}\n\n")
        f.write(f"This model achieved the highest average fitness across all datasets.\n\n")
        
        # Best strategy
        best_strategy = self.data.groupby('strategy')['fitness'].mean().idxmax()
        f.write(f"### 2. Strategy Selection\n\n")
        f.write(f"**Recommended Strategy:** {best_strategy}\n\n")
        f.write(f"This strategy consistently produced the best results.\n\n")
        
        # Optimal generation count
        f.write("### 3. Optimization Settings\n\n")
        
        # Find when improvements plateau
        gen_improvements = []
        for gen in range(1, self.data['generation'].max() + 1):
            prev = self.data[self.data['generation'] == gen - 1]['fitness'].mean()
            curr = self.data[self.data['generation'] == gen]['fitness'].mean()
            improvement = ((curr - prev) / prev * 100) if prev > 0 else 0
            gen_improvements.append((gen, improvement))
        
        # Find when improvement drops below 2%
        optimal_gens = self.data['generation'].max() + 1
        for gen, imp in gen_improvements:
            if imp < 2.0:
                optimal_gens = gen
                break
        
        f.write(f"**Recommended Generations:** {optimal_gens}\n\n")
        f.write(f"Improvements typically plateau after this point.\n\n")
        
        # Dataset-specific recommendations
        f.write("### 4. Dataset-Specific Recommendations\n\n")
        for dataset in self.data['dataset'].unique():
            dataset_data = self.data[self.data['dataset'] == dataset]
            best_model_for_dataset = dataset_data.groupby('model')['fitness'].mean().idxmax()
            best_strategy_for_dataset = dataset_data.groupby('strategy')['fitness'].mean().idxmax()
            
            f.write(f"#### {dataset}\n\n")
            f.write(f"- **Best Model:** {best_model_for_dataset.split('/')[-1]}\n")
            f.write(f"- **Best Strategy:** {best_strategy_for_dataset}\n\n")
        
        f.write("\n---\n\n")
        f.write(f"*Report generated by Comprehensive Analysis Tool*\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Comprehensive Report")
    parser.add_argument("--results-dir", type=str, 
                       default="prompt_optimization_results",
                       help="Directory containing results")
    parser.add_argument("--output", type=str,
                       default="comprehensive_report.md",
                       help="Output file name")
    
    args = parser.parse_args()
    
    generator = ComprehensiveReportGenerator(args.results_dir)
    generator.generate_comprehensive_report(args.output)
    
    print(f"\n✅ Report saved to: {args.output}")
    print(f"\nTo view: open {args.output}")


if __name__ == "__main__":
    main()