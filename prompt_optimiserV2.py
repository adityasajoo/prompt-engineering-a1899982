"""
Prompt Optimization Framework v2.0
Optimized for instruction-tuned LLMs (GPT, Claude, Llama)
M4 MacBook Air compatible with MPS acceleration
"""

import argparse
import json
import logging
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Evaluation metrics
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# LLM providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic not available. Install with: pip install anthropic")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers")

# ========================= Configuration =========================

@dataclass
class Config:
    """Configuration for prompt optimization"""
    
    # LLM API Keys (set via environment variables)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model configurations - Using instruction-tuned LLMs
    models: List[str] = field(default_factory=lambda: [
        "gpt-3.5-turbo",           # OpenAI - fast and cheap
        "claude-3-haiku-20240307", # Anthropic - fast
        "meta-llama/Llama-3.2-3B-Instruct"  # Local - runs on M4
    ])
    
    # Dataset configurations
    datasets: List[str] = field(default_factory=lambda: [
        "cnn_dailymail",
        "samsum",
        "billsum"
    ])
    
    dataset_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "cnn_dailymail": {
            "name": "cnn_dailymail",
            "config": "3.0.0",
            "split": "test",
            "input_key": "article",
            "output_key": "highlights"
        },
        "samsum": {
            "name": "samsum",
            "config": None,
            "split": "test",
            "input_key": "dialogue",
            "output_key": "summary"
        },
        "billsum": {
            "name": "billsum",
            "config": None,
            "split": "test",
            "input_key": "text",
            "output_key": "summary"
        }
    })
    
    # Genetic Algorithm parameters
    population_size: int = 20
    num_generations: int = 15
    mutation_rate: float = 0.3  # Increased for more exploration
    crossover_rate: float = 0.7
    elite_size: int = 3  # Keep top 3
    tournament_size: int = 3
    
    # Evaluation parameters
    max_samples_per_dataset: int = 100
    samples_per_eval: int = 20  # Evaluate each prompt on 20 samples
    max_tokens: int = 150  # Max summary length
    temperature: float = 0.3  # Lower for consistency
    
    # Diversity parameters
    diversity_pressure: float = 0.15  # Weight for diversity in selection
    
    # Convergence detection
    convergence_threshold: float = 0.01  # Stop if improvement < 1%
    convergence_generations: int = 3  # Check over last 3 generations
    
    # Paths
    output_dir: str = "results"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Device configuration
    device: str = "mps"  # Use Metal Performance Shaders on M4
    
    def __post_init__(self):
        # Get API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Create directories
        for dir_path in [self.output_dir, self.log_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

# ========================= Logging Setup =========================

def setup_logger(config: Config) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.log_dir, f"optimization_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("PromptOptimizer")
    logger.info("="*80)
    logger.info("PROMPT OPTIMIZATION FRAMEWORK v2.0")
    logger.info(f"Device: {config.device}")
    logger.info("="*80)
    
    return logger

# ========================= LLM Wrapper =========================

class LLMWrapper:
    """Unified wrapper for different LLM providers"""
    
    def __init__(self, model_name: str, config: Config, logger: logging.Logger):
        self.model_name = model_name
        self.config = config
        self.logger = logger
        self.provider = self._detect_provider()
        
        # Initialize client
        if self.provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=config.openai_api_key)
            
        elif self.provider == "anthropic":
            if not config.anthropic_api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            self.client = Anthropic(api_key=config.anthropic_api_key)
            
        elif self.provider == "local":
            self._load_local_model()
        
        self.logger.info(f"Loaded model: {model_name} (provider: {self.provider})")
    
    def _detect_provider(self) -> str:
        """Detect which provider this model uses"""
        if "gpt" in self.model_name.lower():
            return "openai"
        elif "claude" in self.model_name.lower():
            return "anthropic"
        else:
            return "local"
    
    def _load_local_model(self):
        """Load local Llama model optimized for M4"""
        self.logger.info(f"Loading local model: {self.model_name}")
        
        # Use 4-bit quantization to fit in 16GB RAM
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.logger.info(f"Model loaded on device: {next(self.model.parameters()).device}")
    
    def generate(self, prompt: str) -> str:
        """Generate response from the model"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            
            elif self.provider == "local":
                # Format for Llama chat template
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                return response.strip()
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return ""
    
    def cleanup(self):
        """Clean up resources"""
        if self.provider == "local" and hasattr(self, 'model'):
            del self.model
            del self.tokenizer
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

# ========================= Evaluation Metrics =========================

class MetricsCalculator:
    """Calculate evaluation metrics for summaries"""
    
    @staticmethod
    def calculate_all_metrics(prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE, BLEU, and BERTScore"""
        
        if not prediction or not reference:
            return {
                'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
                'bleu': 0.0, 'bertscore': 0.0, 'combined': 0.0
            }
        
        metrics = {}
        
        # ROUGE scores
        try:
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            rouge_scores = scorer.score(reference, prediction)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        except:
            metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
        
        # BLEU score
        try:
            bleu = sacrebleu.sentence_bleu(prediction, [reference])
            metrics['bleu'] = bleu.score / 100.0
        except:
            metrics['bleu'] = 0.0
        
        # BERTScore
        try:
            P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
            metrics['bertscore'] = F1.item()
        except:
            metrics['bertscore'] = 0.0
        
        # Combined score (weighted average)
        metrics['combined'] = (
            0.35 * ((metrics['rouge1'] + metrics['rouge2'] + metrics['rougeL']) / 3) +
            0.25 * metrics['bleu'] +
            0.40 * metrics['bertscore']
        )
        
        return metrics

# ========================= Prompt Individual =========================

@dataclass
class PromptIndividual:
    """Represents a prompt in the population"""
    template: str
    strategy: str
    fitness: float = 0.0
    generation: int = 0
    evaluation_count: int = 0
    metric_details: Dict[str, float] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.template, self.strategy))

# ========================= Mutation Operations =========================

class MutationEngine:
    """Handles all mutation operations"""
    
    # Mutation components
    INSTRUCTION_STYLES = [
        "Summarize the following text:",
        "Provide a concise summary of:",
        "Create a brief overview of:",
        "Condense the main points from:",
        "Extract key information from:",
        "Generate a summary for:",
        "What are the main points in:",
        "TL;DR of:"
    ]
    
    LENGTH_CONSTRAINTS = [
        "in 1-2 sentences",
        "in one sentence",
        "in 2-3 sentences",
        "in under 50 words",
        "briefly",
        "concisely",
        "in a few sentences"
    ]
    
    CONTEXT_CUES = [
        "Focus on the most important information.",
        "Highlight key facts and outcomes.",
        "Include main events and conclusions.",
        "Capture the essential details.",
        "Emphasize critical points.",
        "Extract core information."
    ]
    
    FORMAT_SPECS = [
        "Use clear, simple language.",
        "Write in a professional tone.",
        "Be objective and factual.",
        "Use complete sentences.",
        "Maintain neutral perspective."
    ]
    
    REASONING_CHAINS = [
        "Think step by step about the main ideas.",
        "First identify key points, then summarize.",
        "Consider: what would someone need to know?",
        "Break down the content logically."
    ]
    
    @classmethod
    def mutate_instruction_style(cls, template: str) -> str:
        """Change the instruction phrasing"""
        lines = template.split('\n')
        lines[0] = random.choice(cls.INSTRUCTION_STYLES) + " {input}"
        return '\n'.join(lines)
    
    @classmethod
    def mutate_add_length_constraint(cls, template: str) -> str:
        """Add or change length constraints"""
        constraint = random.choice(cls.LENGTH_CONSTRAINTS)
        if "in " in template or "under " in template:
            # Replace existing constraint
            import re
            template = re.sub(r'in \d+-?\d* \w+|under \d+ \w+|briefly|concisely', 
                            constraint, template)
        else:
            # Add constraint
            lines = template.split('\n')
            lines[0] = lines[0].rstrip() + f" {constraint}."
            template = '\n'.join(lines)
        return template
    
    @classmethod
    def mutate_add_context_cue(cls, template: str) -> str:
        """Add context-focusing instructions"""
        cue = random.choice(cls.CONTEXT_CUES)
        if not any(c in template for c in cls.CONTEXT_CUES):
            return template + f"\n{cue}"
        return template
    
    @classmethod
    def mutate_format_specification(cls, template: str) -> str:
        """Add format specifications"""
        spec = random.choice(cls.FORMAT_SPECS)
        if not any(s in template for s in cls.FORMAT_SPECS):
            return template + f"\n{spec}"
        return template
    
    @classmethod
    def mutate_reasoning_chain(cls, template: str) -> str:
        """Add reasoning instructions"""
        chain = random.choice(cls.REASONING_CHAINS)
        if "step" not in template.lower() and "consider" not in template.lower():
            return template + f"\n{chain}"
        return template
    
    @classmethod
    def crossover(cls, parent1: PromptIndividual, parent2: PromptIndividual) -> Tuple[str, str]:
        """Combine elements from two prompts"""
        lines1 = [l for l in parent1.template.split('\n') if l.strip()]
        lines2 = [l for l in parent2.template.split('\n') if l.strip()]
        
        # Ensure we keep the input placeholder
        base1 = [l for l in lines1 if '{input}' in l][0] if any('{input}' in l for l in lines1) else lines1[0]
        base2 = [l for l in lines2 if '{input}' in l][0] if any('{input}' in l for l in lines2) else lines2[0]
        
        # Get additional instructions
        extra1 = [l for l in lines1 if '{input}' not in l]
        extra2 = [l for l in lines2 if '{input}' not in l]
        
        # Mix them
        child1_parts = [base1] + random.sample(extra1 + extra2, 
                                              min(2, len(extra1 + extra2)))
        child2_parts = [base2] + random.sample(extra1 + extra2,
                                              min(2, len(extra1 + extra2)))
        
        return '\n'.join(child1_parts), '\n'.join(child2_parts)

# ========================= Genetic Algorithm =========================

class GeneticOptimizer:
    """Improved genetic algorithm with diversity maintenance"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.mutation_ops = [
            MutationEngine.mutate_instruction_style,
            MutationEngine.mutate_add_length_constraint,
            MutationEngine.mutate_add_context_cue,
            MutationEngine.mutate_format_specification,
            MutationEngine.mutate_reasoning_chain
        ]
    
    def create_initial_population(self) -> List[PromptIndividual]:
        """Create diverse initial population"""
        population = []
        
        # Seed templates with varying complexity
        seed_templates = [
            "Summarize: {input}",
            "Provide a brief summary: {input}",
            "What are the key points? {input}",
            "Create a concise overview: {input}\nFocus on main ideas.",
            "Summarize the following in 2-3 sentences: {input}",
            "Extract the essential information: {input}\nBe clear and factual.",
            "TL;DR: {input}\nHighlight the most important points.",
            "Condense this text: {input}\nThink step by step about what matters most."
        ]
        
        # Create population with different strategies
        strategies = ["direct", "structured", "reasoning", "detailed", "minimal"]
        
        for i in range(self.config.population_size):
            template = seed_templates[i % len(seed_templates)]
            strategy = strategies[i % len(strategies)]
            
            population.append(PromptIndividual(
                template=template,
                strategy=strategy,
                generation=0
            ))
        
        self.logger.info(f"Created initial population: {len(population)} individuals")
        return population
    
    def mutate(self, individual: PromptIndividual, generation: int) -> PromptIndividual:
        """Apply adaptive mutation"""
        # Adaptive mutation rate (decrease over time)
        adaptive_rate = self.config.mutation_rate * (1 - generation / (self.config.num_generations * 2))
        
        if random.random() < adaptive_rate:
            # Select mutation operation
            mutation_op = random.choice(self.mutation_ops)
            new_template = mutation_op(individual.template)
            
            # Possibly change strategy
            new_strategy = individual.strategy
            if random.random() < 0.2:
                strategies = ["direct", "structured", "reasoning", "detailed", "minimal"]
                new_strategy = random.choice([s for s in strategies if s != individual.strategy])
            
            return PromptIndividual(
                template=new_template,
                strategy=new_strategy,
                generation=generation
            )
        
        return individual
    
    def tournament_selection(self, population: List[PromptIndividual]) -> PromptIndividual:
        """Tournament selection with diversity consideration"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        
        # 85% chance to select by fitness, 15% for diversity
        if random.random() > self.config.diversity_pressure:
            return max(tournament, key=lambda x: x.fitness)
        else:
            # Random selection for diversity
            return random.choice(tournament)
    
    def create_next_generation(self, population: List[PromptIndividual], 
                             generation: int) -> List[PromptIndividual]:
        """Create next generation with elitism and diversity"""
        new_population = []
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Elitism - keep top performers
        new_population.extend(sorted_pop[:self.config.elite_size])
        
        # Random survivors for diversity (2 individuals)
        diverse_survivors = random.sample(sorted_pop[self.config.elite_size:], 
                                         min(2, len(sorted_pop) - self.config.elite_size))
        new_population.extend(diverse_survivors)
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1_template, child2_template = MutationEngine.crossover(parent1, parent2)
                child1 = PromptIndividual(
                    template=child1_template,
                    strategy=parent1.strategy,
                    generation=generation
                )
                child2 = PromptIndividual(
                    template=child2_template,
                    strategy=parent2.strategy,
                    generation=generation
                )
            else:
                child1 = PromptIndividual(
                    template=parent1.template,
                    strategy=parent1.strategy,
                    generation=generation
                )
                child2 = PromptIndividual(
                    template=parent2.template,
                    strategy=parent2.strategy,
                    generation=generation
                )
            
            # Mutation
            child1 = self.mutate(child1, generation)
            child2 = self.mutate(child2, generation)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.config.population_size]

# ========================= Main Optimizer =========================

class PromptOptimizer:
    """Main optimization orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.genetic_optimizer = GeneticOptimizer(config, self.logger)
        self.metrics_calculator = MetricsCalculator()
        self.results = []
        
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load and prepare dataset"""
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        dataset_config = self.config.dataset_configs[dataset_name]
        
        dataset = load_dataset(
            dataset_config["name"],
            dataset_config.get("config"),
            split=dataset_config["split"]
        )
        
        # Sample and normalize
        data_list = list(dataset)
        if len(data_list) > self.config.max_samples_per_dataset:
            data_list = random.sample(data_list, self.config.max_samples_per_dataset)
        
        normalized = []
        for item in data_list:
            normalized.append({
                "input": item[dataset_config["input_key"]],
                "reference": item[dataset_config["output_key"]]
            })
        
        self.logger.info(f"Loaded {len(normalized)} samples")
        return normalized
    
    def evaluate_individual(self, individual: PromptIndividual, 
                          model: LLMWrapper, dataset: List[Dict]) -> float:
        """Evaluate a prompt on a subset of the dataset"""
        
        # Sample evaluation set
        eval_samples = random.sample(dataset, 
                                    min(self.config.samples_per_eval, len(dataset)))
        
        all_metrics = []
        
        for sample in eval_samples:
            # Create prompt
            prompt = individual.template.format(input=sample["input"])
            
            # Generate summary
            prediction = model.generate(prompt)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                prediction, sample["reference"]
            )
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        # Store details
        individual.metric_details = avg_metrics
        individual.evaluation_count += 1
        
        return avg_metrics['combined']
    
    def check_convergence(self, fitness_history: List[float]) -> bool:
        """Check if optimization has converged"""
        if len(fitness_history) < self.config.convergence_generations + 1:
            return False
        
        recent = fitness_history[-self.config.convergence_generations:]
        improvement = (max(recent) - min(recent)) / max(recent) if max(recent) > 0 else 0
        
        return improvement < self.config.convergence_threshold
    
    def optimize(self, model_name: str, dataset_name: str) -> Dict:
        """Run optimization for a model-dataset pair"""
        
        self.logger.info("="*80)
        self.logger.info(f"Optimizing: {model_name} on {dataset_name}")
        self.logger.info("="*80)
        
        # Load model and dataset
        model = LLMWrapper(model_name, self.config, self.logger)
        dataset = self.load_dataset(dataset_name)
        
        # Create initial population
        population = self.genetic_optimizer.create_initial_population()
        
        # Track best fitness per generation
        fitness_history = []
        best_overall = None
        
        # Evolution loop
        for generation in range(self.config.num_generations):
            self.logger.info(f"\nGeneration {generation + 1}/{self.config.num_generations}")
            
            # Evaluate population
            for individual in tqdm(population, desc="Evaluating"):
                if individual.fitness == 0.0:
                    individual.fitness = self.evaluate_individual(individual, model, dataset)
            
            # Get statistics
            best = max(population, key=lambda x: x.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population])
            
            fitness_history.append(best.fitness)
            
            if best_overall is None or best.fitness > best_overall.fitness:
                best_overall = best
            
            # Log progress
            self.logger.info(f"Best: {best.fitness:.4f} | Avg: {avg_fitness:.4f} | Strategy: {best.strategy}")
            self.logger.info(f"Metrics - R: {best.metric_details.get('rouge1', 0):.3f}, "
                           f"B: {best.metric_details.get('bleu', 0):.3f}, "
                           f"BS: {best.metric_details.get('bertscore', 0):.3f}")
            
            # Save generation results
            self.save_generation_results(generation, population, model_name, dataset_name)
            
            # Check convergence
            if generation >= 5 and self.check_convergence(fitness_history):
                self.logger.info(f"Converged at generation {generation + 1}")
                break
            
            # Create next generation
            if generation < self.config.num_generations - 1:
                population = self.genetic_optimizer.create_next_generation(
                    population, generation + 1
                )
        
        # Cleanup
        model.cleanup()
        
        self.logger.info(f"\nOptimization Complete!")
        self.logger.info(f"Best Fitness: {best_overall.fitness:.4f}")
        self.logger.info(f"Best Prompt:\n{best_overall.template}")
        
        return {
            "model": model_name,
            "dataset": dataset_name,
            "best_fitness": best_overall.fitness,
            "best_prompt": best_overall.template,
            "best_strategy": best_overall.strategy,
            "metrics": best_overall.metric_details,
            "generations": generation + 1,
            "fitness_history": fitness_history
        }
    
    def save_generation_results(self, generation: int, population: List[PromptIndividual],
                               model_name: str, dataset_name: str):
        """Save results for analysis"""
        for individual in population:
            self.results.append({
                "generation": generation,
                "model": model_name,
                "dataset": dataset_name,
                "strategy": individual.strategy,
                "fitness": individual.fitness,
                "template": individual.template,
                "rouge": individual.metric_details.get('rouge1', 0),
                "bleu": individual.metric_details.get('bleu', 0),
                "bertscore": individual.metric_details.get('bertscore', 0)
            })
    
    def run_all_experiments(self):
        """Run optimization for all combinations"""
        all_results = []
        
        for model_name in self.config.models:
            for dataset_name in self.config.datasets:
                try:
                    result = self.optimize(model_name, dataset_name)
                    all_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error: {model_name} on {dataset_name}: {str(e)}")
                    continue
        
        # Save results
        self.save_final_results(all_results)
        
        return all_results
    
    def save_final_results(self, all_results: List[Dict]):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_path = os.path.join(self.config.output_dir, f"results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.config.output_dir, f"results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Create summary
        summary = self.create_summary(all_results)
        summary_path = os.path.join(self.config.output_dir, f"summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        self.logger.info(f"Results saved to {self.config.output_dir}")
    
    def create_summary(self, results: List[Dict]) -> str:
        """Create readable summary"""
        summary = ["="*80, "OPTIMIZATION SUMMARY", "="*80, ""]
        
        for result in results:
            summary.append(f"\nModel: {result['model']}")
            summary.append(f"Dataset: {result['dataset']}")
            summary.append(f"Best Fitness: {result['best_fitness']:.4f}")
            summary.append(f"Generations: {result['generations']}")
            summary.append(f"Strategy: {result['best_strategy']}")
            summary.append(f"\nMetrics:")
            for metric, value in result['metrics'].items():
                summary.append(f"  {metric}: {value:.4f}")
            summary.append(f"\nBest Prompt:\n{result['best_prompt']}")
            summary.append("-"*80)
        
        return "\n".join(summary)

# ========================= CLI =========================

def main():
    parser = argparse.ArgumentParser(description="Prompt Optimization Framework v2.0")
    
    parser.add_argument("--models", nargs="+", default=None,
                      help="Models to use (default: gpt-3.5-turbo, claude-3-haiku, llama)")
    parser.add_argument("--datasets", nargs="+", default=None,
                      help="Datasets to use (default: all)")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--samples-per-eval", type=int, default=20,
                      help="Number of samples to evaluate each prompt on")
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        population_size=args.population_size,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        samples_per_eval=args.samples_per_eval,
        output_dir=args.output_dir
    )
    
    if args.models:
        config.models = args.models
    if args.datasets:
        config.datasets = args.datasets
    
    # Run optimization
    optimizer = PromptOptimizer(config)
    optimizer.run_all_experiments()

if __name__ == "__main__":
    main()