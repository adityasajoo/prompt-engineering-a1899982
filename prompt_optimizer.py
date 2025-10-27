"""
Prompt Optimization Framework
Main script for optimizing prompts using genetic algorithms and multiple LLMs
"""

import argparse
import json
import logging
import os
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings('ignore')
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM)

# ========================= Configuration =========================

@dataclass
class Config:
    """Configuration for prompt optimization"""
    # Model configurations
    models: List[str] = None
    model_configs: Dict[str, Dict] = None
    
    # Dataset configurations
    datasets: List[str] = None
    dataset_configs: Dict[str, Dict] = None
    
    # Optimization parameters
    population_size: int = 20
    num_generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    
    # Prompt strategies
    strategies: List[str] = None
    
    # Evaluation
    max_samples_per_dataset: int = 100
    batch_size: int = 8
    max_new_tokens: int = 100
    
    # Paths
    output_dir: str = "prompt_optimization_results"
    log_dir: str = "logs"
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                "facebook/bart-large-cnn",        # Meta - trained on CNN/DailyMail
                "google/pegasus-cnn_dailymail",   # Google - specialized for news
                "google/flan-t5-large"            # Google - instruction-tuned
            ]

        if self.model_configs is None:
            self.model_configs = {
                "facebook/bart-large-cnn": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                },
                "google/pegasus-cnn_dailymail": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                },
                "google/flan-t5-large": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                }
            }
            
        if self.datasets is None:
            self.datasets = [
                "cnn_dailymail",
                "samsum", 
                "xsum"
            ]
    
        if self.dataset_configs is None:
            self.dataset_configs = {
                "cnn_dailymail": {
                    "name": "abisee/cnn_dailymail",
                    "config": "3.0.0", 
                    "split": "test"
                },
                "samsum": {
                    "name": "knkarthick/samsum",
                    "config": None, 
                    "split": "test"
                },
                "billsum": {
                    "name": "billsum",
                    "config": None,
                    "split": "test"
         }
            }
        if self.strategies is None:
            self.strategies = [
                "zero_shot",
                "few_shot",
                "chain_of_thought",
                "self_consistency",
                "tree_of_thought"
            ]
    
    # Rest of the method remains the same...
# ========================= Logging Setup =========================

class CustomLogger:
    """Custom logger for the optimization framework"""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"optimization_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("PromptOptimizer")
        self.logger.info("="*80)
        self.logger.info("PROMPT OPTIMIZATION FRAMEWORK INITIALIZED")
        self.logger.info("="*80)
        
        # Create a serializable version of config for logging
        config_dict = asdict(self.config)
        # Convert torch dtypes to strings
        for model_name, model_config in config_dict.get('model_configs', {}).items():
            if 'torch_dtype' in model_config:
                model_config['torch_dtype'] = str(model_config['torch_dtype'])
        
        self.logger.info(f"Configuration: {json.dumps(config_dict, indent=2)}")
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        getattr(self.logger, level.lower())(message)

# ========================= Prompt Engineering Strategies =========================

class PromptStrategy:
    """Base class for prompt strategies"""
    
    def __init__(self, name: str):
        self.name = name
        
    def apply(self, base_prompt: str, task: str, examples: List[Dict] = None) -> str:
        """Apply the strategy to generate a prompt"""
        raise NotImplementedError

class ZeroShotStrategy(PromptStrategy):
    """Zero-shot prompting strategy"""
    
    def __init__(self):
        super().__init__("zero_shot")
        
    def apply(self, base_prompt: str, task: str, examples: List[Dict] = None) -> str:
        return f"""Task: {task}

{base_prompt}

Answer:"""

class FewShotStrategy(PromptStrategy):
    """Few-shot prompting strategy"""
    
    def __init__(self):
        super().__init__("few_shot")
        
    def apply(self, base_prompt: str, task: str, examples: List[Dict] = None) -> str:
        prompt = f"Task: {task}\n\n"
        
        if examples:
            prompt += "Examples:\n"
            for i, ex in enumerate(examples[:3], 1):
                prompt += f"\nExample {i}:\n"
                prompt += f"Question: {ex.get('question', ex.get('input', ''))}\n"
                prompt += f"Answer: {ex.get('answer', ex.get('output', ''))}\n"
        
        prompt += f"\nNow solve:\n{base_prompt}\n\nAnswer:"
        return prompt

class ChainOfThoughtStrategy(PromptStrategy):
    """Chain-of-thought prompting strategy"""
    
    def __init__(self):
        super().__init__("chain_of_thought")
        
    def apply(self, base_prompt: str, task: str, examples: List[Dict] = None) -> str:
        return f"""Task: {task}

{base_prompt}

Let's think step by step:
1. First, let me understand what is being asked.
2. Then, I'll break down the problem.
3. Next, I'll work through each part.
4. Finally, I'll provide the answer.

Reasoning:"""

class SelfConsistencyStrategy(PromptStrategy):
    """Self-consistency prompting strategy"""
    
    def __init__(self):
        super().__init__("self_consistency")
        
    def apply(self, base_prompt: str, task: str, examples: List[Dict] = None) -> str:
        return f"""Task: {task}

{base_prompt}

Generate multiple solutions and select the most consistent one:

Solution 1:"""

class TreeOfThoughtStrategy(PromptStrategy):
    """Tree-of-thought prompting strategy"""
    
    def __init__(self):
        super().__init__("tree_of_thought")
        
    def apply(self, base_prompt: str, task: str, examples: List[Dict] = None) -> str:
        return f"""Task: {task}

{base_prompt}

Let's explore different reasoning paths:

Path 1: Start with the most obvious approach
Path 2: Consider alternative methods
Path 3: Check for edge cases

Exploring Path 1:"""

# ========================= Genetic Algorithm Components =========================

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm"""
    prompt_template: str
    strategy: str
    fitness: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class GeneticOptimizer:
    """Genetic algorithm for prompt optimization"""
    
    def __init__(self, config: Config, logger: CustomLogger):
        self.config = config
        self.logger = logger
        self.strategies = self._initialize_strategies()
        self.mutation_operations = [
            self._mutate_add_instruction,
            self._mutate_remove_instruction,
            self._mutate_rephrase,
            self._mutate_reorder,
            self._mutate_emphasis
        ]
        
    def _initialize_strategies(self) -> Dict[str, PromptStrategy]:
        """Initialize all prompt strategies"""
        return {
            "zero_shot": ZeroShotStrategy(),
            "few_shot": FewShotStrategy(),
            "chain_of_thought": ChainOfThoughtStrategy(),
            "self_consistency": SelfConsistencyStrategy(),
            "tree_of_thought": TreeOfThoughtStrategy()
        }
    
    def create_initial_population(self) -> List[Individual]:
        """Create initial population of prompts"""
        self.logger.log("Creating initial population...")
        
        base_templates = [
                "Summarize the following in one sentence: {input}",
                "Provide a brief summary: {input}",
                "What is the main point of this text? {input}",
                "Condense this into key points: {input}",
                "TL;DR: {input}",
        ]
        
        population = []
        for _ in range(self.config.population_size):
            template = random.choice(base_templates)
            strategy = random.choice(self.config.strategies)
            individual = Individual(
                prompt_template=template,
                strategy=strategy,
                metadata={"generation": 0}
            )
            population.append(individual)
            
        self.logger.log(f"Created initial population of {len(population)} individuals")
        return population
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual"""
        if random.random() < self.config.mutation_rate:
            mutation_op = random.choice(self.mutation_operations)
            mutated_template = mutation_op(individual.prompt_template)
            
            # Possibly change strategy
            new_strategy = individual.strategy
            if random.random() < 0.3:  # 30% chance to change strategy
                new_strategy = random.choice(self.config.strategies)
            
            return Individual(
                prompt_template=mutated_template,
                strategy=new_strategy,
                metadata=individual.metadata.copy()
            )
        return individual
    
    def _mutate_add_instruction(self, template: str) -> str:
        """Add an instruction to the template"""
        additions = [
            "\nBe concise and clear.",
            "\nExplain your reasoning.",
            "\nDouble-check your answer.",
            "\nConsider all possibilities.",
            "\nBe systematic in your approach."
        ]
        return template + random.choice(additions)
    
    def _mutate_remove_instruction(self, template: str) -> str:
        """Remove or simplify part of the template"""
        lines = template.split('\n')
        if len(lines) > 1:
            lines.pop(random.randint(0, len(lines)-1))
        return '\n'.join(lines)
    
    def _mutate_rephrase(self, template: str) -> str:
        """Rephrase parts of the template"""
        replacements = {
            "Please solve": ["Solve", "Address", "Work on"],
            "provide": ["give", "offer", "present"],
            "answer": ["solution", "response", "result"],
            "Think carefully": ["Consider", "Analyze", "Examine"]
        }
        
        for original, alternatives in replacements.items():
            if original in template:
                template = template.replace(original, random.choice(alternatives))
                break
        return template
    
    def _mutate_reorder(self, template: str) -> str:
        """Reorder sentences in the template"""
        sentences = template.split('. ')
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '. '.join(sentences)
        return template
    
    def _mutate_emphasis(self, template: str) -> str:
        """Add emphasis to important parts"""
        emphasis_words = ["IMPORTANT:", "NOTE:", "KEY:", "FOCUS:"]
        if not any(word in template for word in emphasis_words):
            return random.choice(emphasis_words) + " " + template
        return template
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents"""
        if random.random() < self.config.crossover_rate:
            # Template crossover
            p1_parts = parent1.prompt_template.split('\n')
            p2_parts = parent2.prompt_template.split('\n')
            
            if len(p1_parts) > 1 and len(p2_parts) > 1:
                crossover_point = random.randint(1, min(len(p1_parts), len(p2_parts)) - 1)
                child1_template = '\n'.join(p1_parts[:crossover_point] + p2_parts[crossover_point:])
                child2_template = '\n'.join(p2_parts[:crossover_point] + p1_parts[crossover_point:])
            else:
                child1_template = parent1.prompt_template
                child2_template = parent2.prompt_template
            
            # Strategy inheritance
            child1_strategy = parent1.strategy if random.random() < 0.5 else parent2.strategy
            child2_strategy = parent2.strategy if random.random() < 0.5 else parent1.strategy
            
            child1 = Individual(prompt_template=child1_template, strategy=child1_strategy)
            child2 = Individual(prompt_template=child2_template, strategy=child2_strategy)
            
            return child1, child2
        
        return parent1, parent2
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select an individual using tournament selection"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

# ========================= Model and Evaluation =========================

class ModelEvaluator:
    """Handles model loading and evaluation"""
    
    def __init__(self, config: Config, logger: CustomLogger):
        self.config = config
        self.logger = logger
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.current_tokenizer = None
        self.is_seq2seq = False
        self.detailed_scores = [] 
        
    def load_model(self, model_name: str):
        """Load a specific model"""
        self.logger.log(f"Loading model: {model_name}")
        
        try:
            # Clear previous model from memory
            if self.current_model is not None:
                del self.current_model
                del self.current_tokenizer
                torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            # Get model config
            model_config = self.config.model_configs.get(model_name, {})
            
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Determine model type and load accordingly
            
            config = AutoConfig.from_pretrained(model_name)
            model_type = config.model_type
            
            # Use appropriate model class based on architecture
            if model_type in ['t5', 'mt5', 'bart', 'mbart', 'pegasus', 'marian']:
                # Sequence-to-sequence models
                self.current_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    **model_config
                )
                self.is_seq2seq = True
            else:
                # Causal language models
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_config
                )
                self.is_seq2seq = False
            
            # Set padding token if not set
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            self.current_model_name = model_name
            self.logger.log(f"Successfully loaded model: {model_name} (type: {model_type})")
            
        except Exception as e:
            self.logger.log(f"Error loading model {model_name}: {str(e)}", "ERROR")
            raise

    def evaluate_prompt(self, prompt: str, target_answer: str) -> float:
        """Evaluate a single prompt"""
        try:
            # Tokenize input
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move to device
            device = next(self.current_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response based on model type
            with torch.no_grad():
                if self.is_seq2seq:
                    # For seq2seq models like T5
                    outputs = self.current_model.generate(
                        **inputs,
                        max_length=self.config.max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        early_stopping=True
                    )
                    response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # For causal LM models
                    outputs = self.current_model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=0.7,
                        do_sample=False,
                        pad_token_id=self.current_tokenizer.pad_token_id
                    )
                    response = self.current_tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
            
            # Calculate score
            combined_score, details = self._calculate_score_with_details(response, target_answer)
            self.detailed_scores.append({
            'response': response[:100],  # First 100 chars
            'target': target_answer[:100],
            'rouge': details.get('rouge', 0.0),
            'bleu': details.get('bleu', 0.0),
            'bertscore': details.get('bertscore', 0.0),
            'combined': combined_score
            })
            
            return combined_score
            
        except Exception as e:
            self.logger.log(f"Error evaluating prompt: {str(e)}", "WARNING")
            return 0.0

    def _calculate_score(self, response: str, target: str) -> float:
        """Calculate combined score using ROUGE, BLEU, and BERTScore"""
     
        
        response = response.strip()
        target = target.strip()
        
        if not response or not target:
            return 0.0
        
        scores = {}
        
        # 1. ROUGE Score (F1 average of ROUGE-1, ROUGE-2, ROUGE-L)
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(target, response)
            scores['rouge'] = (
                rouge_scores['rouge1'].fmeasure + 
                rouge_scores['rouge2'].fmeasure + 
                rouge_scores['rougeL'].fmeasure
            ) / 3.0
        except:
            scores['rouge'] = 0.0
        
        # 2. BLEU Score
        try:
            bleu = sacrebleu.sentence_bleu(response, [target])
            scores['bleu'] = bleu.score / 100.0  # Normalize to 0-1
        except:
            scores['bleu'] = 0.0
        
        # 3. BERTScore (F1)
        try:
            P, R, F1 = bert_score([response], [target], lang='en', verbose=False)
            scores['bertscore'] = F1.item()
        except:
            scores['bertscore'] = 0.0
        
        # Combined score (weighted average)
        # ROUGE: 35%, BLEU: 25%, BERTScore: 40%
        combined_score = (
            0.35 * scores['rouge'] +
            0.25 * scores['bleu'] +
            0.40 * scores['bertscore']
        )
        
        # Store individual scores in metadata for analysis
        if hasattr(self, 'last_eval_details'):
            self.last_eval_details = scores
        
        return combined_score
    
    def _calculate_score_with_details(self, response: str, target: str) -> tuple:
        """Calculate combined score and return details"""
        
        response = response.strip()
        target = target.strip()
        
        if not response or not target:
            return 0.0, {'rouge': 0.0, 'bleu': 0.0, 'bertscore': 0.0}
        
        scores = {}
        
        # ROUGE
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(target, response)
            scores['rouge'] = (
                rouge_scores['rouge1'].fmeasure + 
                rouge_scores['rouge2'].fmeasure + 
                rouge_scores['rougeL'].fmeasure
            ) / 3.0
        except:
            scores['rouge'] = 0.0
        
        # BLEU
        try:
            bleu = sacrebleu.sentence_bleu(response, [target])
            scores['bleu'] = bleu.score / 100.0
        except:
            scores['bleu'] = 0.0
        
        # BERTScore
        try:
            P, R, F1 = bert_score([response], [target], lang='en', verbose=False)
            scores['bertscore'] = F1.item()
        except:
            scores['bertscore'] = 0.0
        
        # Combined
        combined = (
            0.35 * scores['rouge'] +
            0.25 * scores['bleu'] +
            0.40 * scores['bertscore']
        )
        
        return combined, scores
   
   # ========================= Data Management =========================

class DataManager:
    """Manages dataset loading and sampling"""
    
    def __init__(self, config: Config, logger: CustomLogger):
        self.config = config
        self.logger = logger
        self.datasets = {}
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.is_seq2seq = False  # Add this line

    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load a specific dataset"""
        self.logger.log(f"Loading dataset: {dataset_name}")
        
        try:
            dataset_config = self.config.dataset_configs[dataset_name]
            
            # Load with trust_remote_code if specified
            trust_remote = dataset_config.get("trust_remote_code", False)
            
            dataset = load_dataset(
                dataset_config["name"],
                dataset_config.get("config"),
                split=dataset_config["split"],
                trust_remote_code=trust_remote  # Add this
            )
            
            # Convert to list and sample
            data_list = list(dataset)
            if len(data_list) > self.config.max_samples_per_dataset:
                data_list = random.sample(data_list, self.config.max_samples_per_dataset)
            
            # Normalize format
            normalized_data = self._normalize_dataset(data_list, dataset_name)
            
            self.datasets[dataset_name] = normalized_data
            self.logger.log(f"Loaded {len(normalized_data)} samples from {dataset_name}")
            
            return normalized_data
            
        except Exception as e:
            self.logger.log(f"Error loading dataset {dataset_name}: {str(e)}", "ERROR")
            return []
        
    def _normalize_dataset(self, data: List[Dict], dataset_name: str) -> List[Dict]:
        """Normalize dataset format"""
        normalized = []
        
        for item in data:
            if dataset_name == "cnn_dailymail":
                normalized.append({
                    "input": item.get("article", ""),
                    "output": item.get("highlights", ""),
                    "full_answer": item.get("highlights", "")
                })
            elif dataset_name == "samsum":
                normalized.append({
                    "input": item.get("dialogue", ""),
                    "output": item.get("summary", ""),
                    "full_answer": item.get("summary", "")
                })
            elif dataset_name == "billsum":
                normalized.append({
                    "input": item.get("text", ""),
                    "output": item.get("summary", ""),
                    "full_answer": item.get("summary", "")
                })
            elif dataset_name == "multi_news":
                normalized.append({
                    "input": item.get("document", ""),
                    "output": item.get("summary", ""),
                    "full_answer": item.get("summary", "")
                })
            else:
                # Generic normalization
                normalized.append({
                    "input": str(item.get("input", item.get("question", item.get("text", item.get("document", item.get("article", "")))))),
                    "output": str(item.get("output", item.get("answer", item.get("summary", item.get("highlights", ""))))),
                    "full_answer": str(item.get("output", item.get("answer", item.get("summary", ""))))
                })
        
        return normalized
# ========================= Results Management =========================

class ResultsManager:
    """Manages saving and loading results"""
    
    def __init__(self, config: Config, logger: CustomLogger):
        self.config = config
        self.logger = logger
        self.results = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(self.config.output_dir, self.run_id)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def save_generation_result(self, generation: int, population: List[Individual], 
                          model_name: str, dataset_name: str, 
                          detailed_scores: List[Dict] = None):
        """Save results including detailed metric scores"""
        result = {
            "generation": generation,
            "model": model_name,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "population": [
                {
                    "prompt_template": ind.prompt_template,
                    "strategy": ind.strategy,
                    "fitness": ind.fitness,
                    "metadata": ind.metadata
                }
                for ind in population
            ],
            "best_fitness": max(ind.fitness for ind in population),
            "avg_fitness": np.mean([ind.fitness for ind in population]),
            "std_fitness": np.std([ind.fitness for ind in population])
        }
        
        # Add metric breakdown if available
        if detailed_scores:
            result["metric_breakdown"] = {
                "avg_rouge": np.mean([s['rouge'] for s in detailed_scores]),
                "avg_bleu": np.mean([s['bleu'] for s in detailed_scores]),
                "avg_bertscore": np.mean([s['bertscore'] for s in detailed_scores])
            }
        
        self.results.append(result)
        
        # Save to file
        filename = f"{model_name.replace('/', '_')}_{dataset_name}_gen{generation}.json"
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

    def save_final_results(self):
        """Save all results to a master file"""
        master_file = os.path.join(self.results_dir, "all_results.json")
        with open(master_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also save as CSV for easy analysis
        df_data = []
        for result in self.results:
            for individual in result["population"]:
                df_data.append({
                    "generation": result["generation"],
                    "model": result["model"],
                    "dataset": result["dataset"],
                    "strategy": individual["strategy"],
                    "fitness": individual["fitness"],
                    "prompt_length": len(individual["prompt_template"]),
                    "timestamp": result["timestamp"]
                })
        
        df = pd.DataFrame(df_data)
        csv_file = os.path.join(self.results_dir, "results_summary.csv")
        df.to_csv(csv_file, index=False)
        
        self.logger.log(f"Saved final results to {master_file} and {csv_file}")
        return df

# ========================= Visualization =========================

class Visualizer:
    """Creates visualizations of optimization results"""
    
    def __init__(self, config: Config, logger: CustomLogger, results_dir: str):
        self.config = config
        self.logger = logger
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_all_plots(self, df: pd.DataFrame):
        """Create all visualization plots"""
        self.logger.log("Creating visualization plots...")
        
        if df.empty:
            self.logger.log("No data to visualize - skipping plot generation", "WARNING")
            return
        
        for model in df['model'].unique():
            self.create_model_plots(df, model)
        
        self.create_comparative_plots(df)
        
        self.logger.log(f"All plots saved to {self.plots_dir}")
    
    def create_model_plots(self, df: pd.DataFrame, model_name: str):
        """Create plots for a specific model"""
        model_df = df[df['model'] == model_name]
        model_name_safe = model_name.replace('/', '_')
        
        # 1. Performance by dataset
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Model: {model_name}', fontsize=16)
        
        for idx, dataset in enumerate(self.config.datasets):
            if idx < 3:
                dataset_df = model_df[model_df['dataset'] == dataset]
                if not dataset_df.empty:
                    ax = axes[idx]
                    
                    # Generation-wise performance
                    gen_performance = dataset_df.groupby('generation')['fitness'].agg(['mean', 'max', 'std'])
                    ax.plot(gen_performance.index, gen_performance['mean'], label='Mean', marker='o')
                    ax.plot(gen_performance.index, gen_performance['max'], label='Max', marker='^')
                    ax.fill_between(gen_performance.index, 
                                   gen_performance['mean'] - gen_performance['std'],
                                   gen_performance['mean'] + gen_performance['std'],
                                   alpha=0.3)
                    
                    ax.set_title(f'Dataset: {dataset}')
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name_safe}_datasets.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Strategy performance
        fig, ax = plt.subplots(figsize=(10, 6))
        strategy_performance = model_df.groupby('strategy')['fitness'].agg(['mean', 'std', 'max'])
        strategy_performance.plot(kind='bar', ax=ax)
        ax.set_title(f'Strategy Performance - {model_name}')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Fitness')
        ax.legend(['Mean', 'Std Dev', 'Max'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name_safe}_strategies.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Generation evolution heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_data = model_df.pivot_table(values='fitness', index='generation', columns='strategy', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'Fitness Evolution by Strategy - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name_safe}_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Convergence analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Convergence Analysis - {model_name}', fontsize=16)
        
        for idx, dataset in enumerate(self.config.datasets[:4]):
            ax = axes[idx // 2, idx % 2]
            dataset_df = model_df[model_df['dataset'] == dataset]
            
            if not dataset_df.empty:
                # Plot fitness distribution over generations
                generations = sorted(dataset_df['generation'].unique())
                positions = []
                data_to_plot = []
                
                for gen in generations:
                    gen_data = dataset_df[dataset_df['generation'] == gen]['fitness'].values
                    if len(gen_data) > 0:
                        positions.append(gen)
                        data_to_plot.append(gen_data)
                
                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6)
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness')
                    ax.set_title(f'Dataset: {dataset}')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name_safe}_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparative_plots(self, df: pd.DataFrame):
        """Create comparative plots across all models"""
        
        # 1. Model comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        model_performance = df.groupby('model')['fitness'].agg(['mean', 'std', 'max'])
        model_performance.plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_xlabel('Model')
        ax.set_ylabel('Fitness')
        ax.legend(['Mean', 'Std Dev', 'Max'])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Dataset difficulty
        fig, ax = plt.subplots(figsize=(10, 6))
        dataset_performance = df.groupby('dataset')['fitness'].agg(['mean', 'std', 'max'])
        dataset_performance.plot(kind='bar', ax=ax)
        ax.set_title('Dataset Difficulty Analysis')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Fitness')
        ax.legend(['Mean', 'Std Dev', 'Max'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Strategy effectiveness across models
        fig, ax = plt.subplots(figsize=(14, 8))
        strategy_model = df.pivot_table(values='fitness', index='strategy', columns='model', aggfunc='mean')
        sns.heatmap(strategy_model, annot=True, fmt='.3f', cmap='coolwarm', center=0.5, ax=ax)
        ax.set_title('Strategy Effectiveness Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'strategy_model_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Best performers summary
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Optimization Summary', fontsize=16)
        
        # Best fitness over time
        ax = axes[0, 0]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_by_gen = model_df.groupby('generation')['fitness'].max()
            ax.plot(best_by_gen.index, best_by_gen.values, label=model.split('/')[-1], marker='o')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Best Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Average fitness over time
        ax = axes[0, 1]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            avg_by_gen = model_df.groupby('generation')['fitness'].mean()
            ax.plot(avg_by_gen.index, avg_by_gen.values, label=model.split('/')[-1], marker='s')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Fitness')
        ax.set_title('Average Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Strategy distribution
        ax = axes[1, 0]
        strategy_counts = df.groupby('strategy')['fitness'].count()
        ax.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
        ax.set_title('Strategy Usage Distribution')
        
        # Convergence speed
        ax = axes[1, 1]
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            max_fitness = model_df.groupby('generation')['fitness'].max()
            
            # Find generation where 90% of max fitness is reached
            final_max = max_fitness.max()
            threshold = 0.9 * final_max
            
            convergence_gen = None
            for gen in sorted(max_fitness.index):
                if max_fitness[gen] >= threshold:
                    convergence_gen = gen
                    break
            
            if convergence_gen is not None:
                ax.bar(model.split('/')[-1], convergence_gen)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Generation to 90% Max Fitness')
        ax.set_title('Convergence Speed')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'optimization_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Final best prompts report
        self._save_best_prompts_report(df)
    
    def _save_best_prompts_report(self, df: pd.DataFrame):
        """Save a report of the best prompts found"""
        report_path = os.path.join(self.plots_dir, 'best_prompts_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BEST PROMPTS REPORT\n")
            f.write("="*80 + "\n\n")
            
            for model in df['model'].unique():
                f.write(f"\nModel: {model}\n")
                f.write("-"*40 + "\n")
                
                model_df = df[df['model'] == model]
                
                for dataset in df['dataset'].unique():
                    dataset_df = model_df[model_df['dataset'] == dataset]
                    if not dataset_df.empty:
                        best_idx = dataset_df['fitness'].idxmax()
                        best_row = dataset_df.loc[best_idx]
                        
                        f.write(f"\nDataset: {dataset}\n")
                        f.write(f"Best Fitness: {best_row['fitness']:.4f}\n")
                        f.write(f"Strategy: {best_row['strategy']}\n")
                        f.write(f"Generation: {best_row['generation']}\n")
                        
                        # Calculate statistics
                        max_gen_to_best = dataset_df[dataset_df['fitness'] == dataset_df['fitness'].max()]['generation'].min()
                        f.write(f"First Generation to Reach Best: {max_gen_to_best}\n")
                        f.write("\n")
            
            # Overall statistics
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Evaluations: {len(df)}\n")
            f.write(f"Best Overall Fitness: {df['fitness'].max():.4f}\n")
            f.write(f"Average Fitness: {df['fitness'].mean():.4f}\n")
            f.write(f"Fitness Std Dev: {df['fitness'].std():.4f}\n")
            
            # Best strategy overall
            strategy_avg = df.groupby('strategy')['fitness'].mean()
            best_strategy = strategy_avg.idxmax()
            f.write(f"Best Strategy (Average): {best_strategy} ({strategy_avg[best_strategy]:.4f})\n")
            
            # Best model overall
            model_avg = df.groupby('model')['fitness'].mean()
            best_model = model_avg.idxmax()
            f.write(f"Best Model (Average): {best_model} ({model_avg[best_model]:.4f})\n")
        
        self.logger.log(f"Best prompts report saved to {report_path}")

# ========================= Main Optimization Loop =========================

class PromptOptimizer:
    """Main class orchestrating the optimization process"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = CustomLogger(config)
        self.genetic_optimizer = GeneticOptimizer(config, self.logger)
        self.model_evaluator = ModelEvaluator(config, self.logger)
        self.data_manager = DataManager(config, self.logger)
        self.results_manager = ResultsManager(config, self.logger)
        
    def run_optimization(self, model_name: str, dataset_name: str):
        """Run optimization for a specific model and dataset"""
        self.logger.log("="*60)
        self.logger.log(f"STARTING OPTIMIZATION")
        self.logger.log(f"Model: {model_name}")
        self.logger.log(f"Dataset: {dataset_name}")
        self.logger.log("="*60)
        
        # Load model and dataset
        self.model_evaluator.load_model(model_name)
        dataset = self.data_manager.load_dataset(dataset_name)
        
        if not dataset:
            self.logger.log(f"Failed to load dataset {dataset_name}", "ERROR")
            return None
        
        # Create initial population
        population = self.genetic_optimizer.create_initial_population()
        
        # Evolution loop
        for generation in range(self.config.num_generations):
            self.logger.log(f"\nGeneration {generation + 1}/{self.config.num_generations}")
            
            # Evaluate population
            for individual in population:
                if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                    # Apply strategy to create actual prompt
                    strategy_obj = self.genetic_optimizer.strategies[individual.strategy]
                    
                    # Evaluate on multiple samples
                    scores = []
                    num_eval_samples = min(25, len(dataset))  # Evaluate on subset for efficiency
                    eval_samples = random.sample(dataset, num_eval_samples)
                    
                    for sample in eval_samples:
                        prompt = strategy_obj.apply(
                            individual.prompt_template.format(input=sample['input']),
                            f"Answer questions about {dataset_name}",
                            dataset[:3]  # Use first 3 as examples for few-shot
                        )
                        
                        score = self.model_evaluator.evaluate_prompt(prompt, sample['output'])
                        scores.append(score)
                    
                    individual.fitness = np.mean(scores)
                    individual.metadata['generation'] = generation
                    
                    self.logger.log(f"Individual evaluated - Strategy: {individual.strategy}, "
                                  f"Fitness: {individual.fitness:.4f}")
            
            # Save generation results
            self.results_manager.save_generation_result(generation, population, model_name, dataset_name,  self.model_evaluator.detailed_scores)
            self.model_evaluator.detailed_scores = []

            # Report statistics
            best_individual = max(population, key=lambda x: x.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population])
            
            self.logger.log(f"Generation {generation + 1} Stats:")
            self.logger.log(f"  Best Fitness: {best_individual.fitness:.4f}")
            self.logger.log(f"  Average Fitness: {avg_fitness:.4f}")
            self.logger.log(f"  Best Strategy: {best_individual.strategy}")
            
            # Create next generation
            if generation < self.config.num_generations - 1:
                new_population = []
                
                # Elitism - keep best individuals
                population_sorted = sorted(population, key=lambda x: x.fitness, reverse=True)
                new_population.extend(population_sorted[:self.config.elite_size])
                
                # Generate rest of population
                while len(new_population) < self.config.population_size:
                    # Selection
                    parent1 = self.genetic_optimizer.tournament_selection(population)
                    parent2 = self.genetic_optimizer.tournament_selection(population)
                    
                    # Crossover
                    child1, child2 = self.genetic_optimizer.crossover(parent1, parent2)
                    
                    # Mutation
                    child1 = self.genetic_optimizer.mutate(child1)
                    child2 = self.genetic_optimizer.mutate(child2)
                    
                    new_population.extend([child1, child2])
                
                # Trim to exact population size
                population = new_population[:self.config.population_size]
        
        # Return best individual
        best_individual = max(population, key=lambda x: x.fitness)
        self.logger.log(f"\nOptimization Complete!")
        self.logger.log(f"Best Fitness: {best_individual.fitness:.4f}")
        self.logger.log(f"Best Strategy: {best_individual.strategy}")
        self.logger.log(f"Best Prompt Template: {best_individual.prompt_template}")
        
        return best_individual
    
    def run_all_experiments(self):
        """Run optimization for all model-dataset combinations"""
        self.logger.log("\n" + "="*80)
        self.logger.log("STARTING FULL EXPERIMENTAL RUN")
        self.logger.log("="*80)
        
        best_results = {}
        
        for model_name in self.config.models:
            best_results[model_name] = {}
            
            for dataset_name in self.config.datasets:
                try:
                    best_individual = self.run_optimization(model_name, dataset_name)
                    if best_individual:
                        best_results[model_name][dataset_name] = {
                            "fitness": best_individual.fitness,
                            "strategy": best_individual.strategy,
                            "prompt": best_individual.prompt_template
                        }
                except Exception as e:
                    self.logger.log(f"Error in optimization for {model_name} on {dataset_name}: {str(e)}", "ERROR")
                    continue
        
        # Save final results
        df = self.results_manager.save_final_results()
        
        # Create visualizations
        visualizer = Visualizer(self.config, self.logger, self.results_manager.results_dir)
        visualizer.create_all_plots(df)
        
        self.logger.log("\n" + "="*80)
        self.logger.log("ALL EXPERIMENTS COMPLETE")
        self.logger.log(f"Results saved to: {self.results_manager.results_dir}")
        self.logger.log("="*80)
        
        return best_results

# ========================= CLI Interface =========================

def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization Framework")
    
    parser.add_argument("--mode", type=str, default="all", 
                      choices=["all", "single"],
                      help="Run mode: 'all' for all experiments, 'single' for specific model")
    
    parser.add_argument("--model", type=str, default=None,
                      help="Model name for single mode")
    
    parser.add_argument("--dataset", type=str, default=None,
                      help="Dataset name for single mode")
    
    parser.add_argument("--population-size", type=int, default=20,
                      help="Population size for genetic algorithm")
    
    parser.add_argument("--num-generations", type=int, default=10,
                      help="Number of generations")
    
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                      help="Mutation rate")
    
    parser.add_argument("--output-dir", type=str, default="prompt_optimization_results",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        population_size=args.population_size,
        num_generations=args.num_generations,
        mutation_rate=args.mutation_rate,
        output_dir=args.output_dir
    )
    
    # Create optimizer
    optimizer = PromptOptimizer(config)
    
    # Run optimization
    if args.mode == "single":
        if not args.model or not args.dataset:
            print("Error: --model and --dataset required for single mode")
            return
        
        optimizer.run_optimization(args.model, args.dataset)
        
        # Still create plots for single run
        df = optimizer.results_manager.save_final_results()
        visualizer = Visualizer(config, optimizer.logger, optimizer.results_manager.results_dir)
        visualizer.create_all_plots(df)
        
    else:
        optimizer.run_all_experiments()

if __name__ == "__main__":
    main()