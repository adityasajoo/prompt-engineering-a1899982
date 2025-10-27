#!/usr/bin/env python3
"""
Generate Case-Wise Reports and Plots with Multi-Metric Evaluation
Creates separate folder for each dataset with detailed metric breakdowns
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

class CaseWiseGenerator:
    """Generate case-wise analysis with multi-metric evaluation"""
    
    def __init__(self, results_dir: str, output_base: str = "case_wise_results"):
        self.results_dir = results_dir
        self.output_base = output_base
        self.data = self.load_results()
        self.metric_data = self.load_metric_details()
        
        # Setup plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
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
            return pd.DataFrame()
    
    def load_metric_details(self) -> dict:
        """Load detailed metric breakdowns from JSON files"""
        metrics = {}
        
        if os.path.exists(self.results_dir):
            for run_dir in os.listdir(self.results_dir):
                run_path = os.path.join(self.results_dir, run_dir)
                if os.path.isdir(run_path):
                    for file in os.listdir(run_path):
                        if file.endswith('.json') and 'gen' in file:
                            json_path = os.path.join(run_path, file)
                            try:
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                                    if 'metric_breakdown' in data:
                                        key = f"{data['model']}_{data['dataset']}_{data['generation']}"
                                        metrics[key] = data['metric_breakdown']
                            except:
                                pass
        
        return metrics
    
    def generate_all_cases(self):
        """Generate folders for all dataset cases"""
        if self.data.empty:
            print("No data to process")
            return
        
        datasets = sorted(self.data['dataset'].unique())
        
        print("="*60)
        print("GENERATING CASE-WISE RESULTS WITH MULTI-METRIC EVALUATION")
        print("="*60)
        
        for idx, dataset in enumerate(datasets, 1):
            print(f"\nProcessing Case {idx}: {dataset}")
            self.generate_case(dataset, idx)
        
        print("\n" + "="*60)
        print("ALL CASES COMPLETED")
        print("="*60)
        print(f"\nResults saved to: {self.output_base}/")
    
    def generate_case(self, dataset: str, case_num: int):
        """Generate complete analysis for one dataset case"""
        
        case_folder = os.path.join(self.output_base, f"case_{case_num}_{dataset}")
        os.makedirs(case_folder, exist_ok=True)
        
        case_data = self.data[self.data['dataset'] == dataset]
        
        # Generate report with metrics
        report_path = os.path.join(case_folder, "report.md")
        self.generate_case_report(case_data, dataset, case_num, report_path)
        
        # Generate plots with metric breakdowns
        plots_folder = os.path.join(case_folder, "plots")
        os.makedirs(plots_folder, exist_ok=True)
        self.generate_case_plots(case_data, dataset, plots_folder)
        
        print(f"  ✓ Report: {report_path}")
        print(f"  ✓ Plots: {plots_folder}/")
    
    def generate_case_report(self, case_data: pd.DataFrame, dataset: str, case_num: int, output_file: str):
        """Generate markdown report with multi-metric details"""
        
        with open(output_file, 'w') as f:
            f.write(f"# CASE {case_num}: {dataset.upper()}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Evaluation Metrics:** ROUGE + BLEU + BERTScore (Combined)\n\n")
            f.write("---\n\n")
            
            # Overview
            f.write("## OVERVIEW\n\n")
            f.write(f"- **Dataset:** {dataset}\n")
            f.write(f"- **Total Evaluations:** {len(case_data)}\n")
            f.write(f"- **Models Tested:** {case_data['model'].nunique()}\n")
            f.write(f"- **Best Combined Score:** {case_data['fitness'].max():.4f}\n")
            f.write(f"- **Average Combined Score:** {case_data['fitness'].mean():.4f}\n\n")
            
            # Metric breakdown if available
            if self.metric_data:
                f.write("### Evaluation Metric Weights\n\n")
                f.write("- ROUGE: 35%\n")
                f.write("- BLEU: 25%\n")
                f.write("- BERTScore: 40%\n\n")
            
            f.write("---\n\n")
            
            # LLM Performance with metric breakdown
            f.write("## 1. LLM-WISE PERFORMANCE\n\n")
            
            model_stats = case_data.groupby('model')['fitness'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            
            f.write("### Combined Score Summary\n\n")
            f.write("| Rank | Model | Count | Mean | Std Dev | Min | Max |\n")
            f.write("|------|-------|-------|------|---------|-----|-----|\n")
            
            model_stats_sorted = model_stats.sort_values('mean', ascending=False)
            for rank, (model, row) in enumerate(model_stats_sorted.iterrows(), 1):
                model_name = model.split('/')[-1]
                f.write(f"| {rank} | {model_name} | {int(row['count'])} | "
                       f"{row['mean']:.4f} | {row['std']:.4f} | "
                       f"{row['min']:.4f} | {row['max']:.4f} |\n")
            
            f.write("\n### Individual Metric Scores by Model\n\n")
            
            # Calculate average metrics per model
            for model in sorted(case_data['model'].unique()):
                model_name = model.split('/')[-1]
                f.write(f"#### {model_name}\n\n")
                
                # Find metric breakdowns
                model_metrics = []
                for key, metrics in self.metric_data.items():
                    if model in key and dataset in key:
                        model_metrics.append(metrics)
                
                if model_metrics:
                    avg_rouge = np.mean([m['avg_rouge'] for m in model_metrics])
                    avg_bleu = np.mean([m['avg_bleu'] for m in model_metrics])
                    avg_bert = np.mean([m['avg_bertscore'] for m in model_metrics])
                    
                    f.write(f"- **ROUGE Score:** {avg_rouge:.4f}\n")
                    f.write(f"- **BLEU Score:** {avg_bleu:.4f}\n")
                    f.write(f"- **BERTScore:** {avg_bert:.4f}\n")
                    f.write(f"- **Combined Score:** {case_data[case_data['model']==model]['fitness'].mean():.4f}\n\n")
                else:
                    f.write(f"- **Combined Score:** {case_data[case_data['model']==model]['fitness'].mean():.4f}\n\n")
            
            f.write("**See Plots:**\n")
            f.write("- `plots/1_llm_comparison.png` - Overall comparison\n")
            f.write("- `plots/6_metric_breakdown.png` - Individual metric scores\n\n")
            f.write("---\n\n")
            
            # Strategy Performance
            f.write("## 2. STRATEGY-WISE PERFORMANCE\n\n")
            
            strategy_stats = case_data.groupby('strategy')['fitness'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            
            f.write("| Rank | Strategy | Usage | Mean | Std Dev | Min | Max |\n")
            f.write("|------|----------|-------|------|---------|-----|-----|\n")
            
            strategy_stats_sorted = strategy_stats.sort_values('mean', ascending=False)
            for rank, (strategy, row) in enumerate(strategy_stats_sorted.iterrows(), 1):
                f.write(f"| {rank} | {strategy} | {int(row['count'])} | "
                       f"{row['mean']:.4f} | {row['std']:.4f} | "
                       f"{row['min']:.4f} | {row['max']:.4f} |\n")
            
            f.write("\n**See:** `plots/2_strategy_analysis.png`\n\n")
            f.write("---\n\n")
            
            # Rest of the report (generation evolution, mutation, convergence)
            self._write_remaining_sections(f, case_data, dataset)
            
            f.write(f"\n---\n\n*Report generated for Case {case_num}: {dataset}*\n")
    
    def _write_remaining_sections(self, f, case_data, dataset):
        """Write generation, mutation, and convergence sections"""
        
        # Generation Evolution
        f.write("## 3. GENERATION EVOLUTION\n\n")
        gen_stats = case_data.groupby('generation')['fitness'].agg(['mean', 'max', 'std']).round(4)
        
        f.write("| Generation | Mean | Max | Std Dev | Improvement |\n")
        f.write("|------------|------|-----|---------|-------------|\n")
        
        for gen, row in gen_stats.iterrows():
            if gen == 0:
                improvement = "Baseline"
            else:
                prev_mean = gen_stats.loc[gen-1, 'mean']
                if prev_mean > 0:
                    improvement = f"{((row['mean'] - prev_mean) / prev_mean * 100):+.2f}%"
                else:
                    improvement = "N/A"
            
            f.write(f"| {gen} | {row['mean']:.4f} | {row['max']:.4f} | "
                   f"{row['std']:.4f} | {improvement} |\n")
        
        f.write("\n**See:** `plots/3_generation_evolution.png`\n\n---\n\n")
        
        # Convergence
        f.write("## 4. CONVERGENCE ANALYSIS\n\n")
        for model in case_data['model'].unique():
            model_data = case_data[case_data['model'] == model]
            model_name = model.split('/')[-1]
            
            max_fitness = model_data['fitness'].max()
            gen_max = model_data.groupby('generation')['fitness'].max()
            
            conv_90 = None
            for gen in sorted(gen_max.index):
                if gen_max[gen] >= 0.90 * max_fitness and conv_90 is None:
                    conv_90 = gen
                    break
            
            f.write(f"### {model_name}\n")
            f.write(f"- Max Score: {max_fitness:.4f}\n")
            f.write(f"- Reached 90% at: Generation {conv_90 if conv_90 is not None else 'Not reached'}\n\n")
        
        f.write("**See:** `plots/5_convergence.png`\n\n")
    
    def generate_case_plots(self, case_data: pd.DataFrame, dataset: str, plots_folder: str):
        """Generate all plots including metric breakdowns"""
        
        # Plot 1: LLM Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        model_stats = case_data.groupby('model')['fitness'].agg(['mean', 'max', 'std'])
        model_stats.index = [m.split('/')[-1] for m in model_stats.index]
        model_stats.plot(kind='bar', ax=ax)
        ax.set_title(f'Case: {dataset} - LLM Combined Score Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Combined Score (ROUGE+BLEU+BERTScore)')
        ax.legend(['Mean', 'Max', 'Std Dev'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '1_llm_comparison.png'), dpi=300)
        plt.close()
        
        # Plot 2: Strategy Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        strategy_mean = case_data.groupby('strategy')['fitness'].mean().sort_values(ascending=False)
        axes[0].bar(range(len(strategy_mean)), strategy_mean.values)
        axes[0].set_xticks(range(len(strategy_mean)))
        axes[0].set_xticklabels(strategy_mean.index, rotation=45, ha='right')
        axes[0].set_title('Average Score by Strategy')
        axes[0].set_ylabel('Mean Combined Score')
        
        strategy_count = case_data.groupby('strategy').size().sort_values(ascending=False)
        axes[1].bar(range(len(strategy_count)), strategy_count.values)
        axes[1].set_xticks(range(len(strategy_count)))
        axes[1].set_xticklabels(strategy_count.index, rotation=45, ha='right')
        axes[1].set_title('Strategy Usage')
        axes[1].set_ylabel('Count')
        
        plt.suptitle(f'Case: {dataset} - Strategy Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '2_strategy_analysis.png'), dpi=300)
        plt.close()
        
        # Plot 3: Generation Evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in case_data['model'].unique():
            model_data = case_data[case_data['model'] == model]
            gen_perf = model_data.groupby('generation')['fitness'].max()
            ax.plot(gen_perf.index, gen_perf.values, marker='o', 
                   label=model.split('/')[-1], linewidth=2)
        ax.set_title(f'Case: {dataset} - Score Evolution by Generation')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Combined Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '3_generation_evolution.png'), dpi=300)
        plt.close()
        
        # Plot 4: Mutation Impact
        fig, ax = plt.subplots(figsize=(12, 6))
        improvements = []
        for gen in range(1, case_data['generation'].max() + 1):
            prev_gen = case_data[case_data['generation'] == gen - 1]
            curr_gen = case_data[case_data['generation'] == gen]
            if not prev_gen.empty and not curr_gen.empty:
                improvements.append(curr_gen['fitness'].max() - prev_gen['fitness'].max())
        
        if improvements:
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
            ax.bar(range(1, len(improvements)+1), improvements, color=colors)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f'Case: {dataset} - Generation-to-Generation Improvement')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Score Improvement')
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '4_mutation_impact.png'), dpi=300)
        plt.close()
        
        # Plot 5: Convergence
        fig, ax = plt.subplots(figsize=(12, 6))
        for model in case_data['model'].unique():
            model_data = case_data[case_data['model'] == model]
            gen_stats = model_data.groupby('generation')['fitness'].agg(['mean', 'max'])
            model_name = model.split('/')[-1]
            ax.plot(gen_stats.index, gen_stats['max'], marker='o', 
                   label=f'{model_name} (max)', linewidth=2)
            ax.plot(gen_stats.index, gen_stats['mean'], marker='s', 
                   linestyle='--', alpha=0.6, label=f'{model_name} (mean)')
        ax.set_title(f'Case: {dataset} - Convergence Analysis')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Combined Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '5_convergence.png'), dpi=300)
        plt.close()
        
        # Plot 6: NEW - Metric Breakdown by Model
        if self.metric_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = sorted(case_data['model'].unique())
            model_names = [m.split('/')[-1] for m in models]
            
            rouge_scores = []
            bleu_scores = []
            bert_scores = []
            
            for model in models:
                model_metrics = []
                for key, metrics in self.metric_data.items():
                    if model in key and dataset in key:
                        model_metrics.append(metrics)
                
                if model_metrics:
                    rouge_scores.append(np.mean([m['avg_rouge'] for m in model_metrics]))
                    bleu_scores.append(np.mean([m['avg_bleu'] for m in model_metrics]))
                    bert_scores.append(np.mean([m['avg_bertscore'] for m in model_metrics]))
                else:
                    rouge_scores.append(0)
                    bleu_scores.append(0)
                    bert_scores.append(0)
            
            x = np.arange(len(model_names))
            width = 0.25
            
            ax.bar(x - width, rouge_scores, width, label='ROUGE', alpha=0.8)
            ax.bar(x, bleu_scores, width, label='BLEU', alpha=0.8)
            ax.bar(x + width, bert_scores, width, label='BERTScore', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title(f'Case: {dataset} - Individual Metric Scores by Model')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_folder, '6_metric_breakdown.png'), dpi=300)
            plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Case-Wise Reports with Multi-Metric Evaluation")
    parser.add_argument("--results-dir", type=str, 
                       default="prompt_optimization_results",
                       help="Directory containing results")
    parser.add_argument("--output-dir", type=str,
                       default="case_wise_results",
                       help="Output directory for case folders")
    
    args = parser.parse_args()
    
    generator = CaseWiseGenerator(args.results_dir, args.output_dir)
    generator.generate_all_cases()


if __name__ == "__main__":
    main()