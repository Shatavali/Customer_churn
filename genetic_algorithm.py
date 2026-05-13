# genetic_algorithm.py
"""
Genetic Algorithm for Hyperparameter Optimization
Customer Churn Prediction Project

Uses GA to evolve optimal hyperparameters for the XGBoost classifier.
Each 'chromosome' encodes a set of hyperparameters; fitness = cross-val ROC-AUC.
"""

import numpy as np
import pandas as pd
import random
import json
import time
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Hyperparameter search space (gene bounds)
# ─────────────────────────────────────────────
PARAM_SPACE = {
    "n_estimators":       {"type": "int",   "low": 50,    "high": 500},
    "max_depth":          {"type": "int",   "low": 2,     "high": 12},
    "learning_rate":      {"type": "float", "low": 0.005, "high": 0.3},
    "subsample":          {"type": "float", "low": 0.5,   "high": 1.0},
    "colsample_bytree":   {"type": "float", "low": 0.5,   "high": 1.0},
    "min_child_weight":   {"type": "int",   "low": 1,     "high": 10},
    "gamma":              {"type": "float", "low": 0.0,   "high": 5.0},
    "reg_alpha":          {"type": "float", "low": 0.0,   "high": 1.0},
    "reg_lambda":         {"type": "float", "low": 0.5,   "high": 5.0},
}

PARAM_NAMES = list(PARAM_SPACE.keys())


# ─────────────────────────────────────────────
#  Chromosome helpers
# ─────────────────────────────────────────────

def random_chromosome():
    """Generate a random chromosome (dict of hyperparameters)."""
    chrom = {}
    for name, spec in PARAM_SPACE.items():
        if spec["type"] == "int":
            chrom[name] = random.randint(spec["low"], spec["high"])
        else:
            chrom[name] = round(random.uniform(spec["low"], spec["high"]), 5)
    return chrom


def clamp(value, spec):
    """Clamp a value within its allowed range."""
    if spec["type"] == "int":
        return int(round(max(spec["low"], min(spec["high"], value))))
    else:
        return round(max(spec["low"], min(spec["high"], value)), 5)


# ─────────────────────────────────────────────
#  Genetic operators
# ─────────────────────────────────────────────

def crossover(parent1, parent2, crossover_rate=0.7):
    """
    Uniform crossover: each gene independently taken from either parent.
    """
    if random.random() > crossover_rate:
        return deepcopy(parent1), deepcopy(parent2)

    child1, child2 = {}, {}
    for name in PARAM_NAMES:
        if random.random() < 0.5:
            child1[name], child2[name] = parent1[name], parent2[name]
        else:
            child1[name], child2[name] = parent2[name], parent1[name]
    return child1, child2


def mutate(chromosome, mutation_rate=0.15, mutation_scale=0.2):
    """
    Gaussian mutation scaled by the parameter range.
    Each gene mutated independently with probability `mutation_rate`.
    """
    mutated = deepcopy(chromosome)
    for name, spec in PARAM_SPACE.items():
        if random.random() < mutation_rate:
            current = mutated[name]
            param_range = spec["high"] - spec["low"]
            delta = np.random.normal(0, mutation_scale * param_range)
            mutated[name] = clamp(current + delta, spec)
    return mutated


def tournament_select(population, fitnesses, k=3):
    """
    Tournament selection: pick k individuals at random, return the best.
    """
    indices = random.sample(range(len(population)), k)
    best = max(indices, key=lambda i: fitnesses[i])
    return deepcopy(population[best])


# ─────────────────────────────────────────────
#  Fitness evaluation
# ─────────────────────────────────────────────

def build_pipeline(preprocessor, chromosome):
    """Build an imblearn pipeline with the given chromosome's XGBoost params."""
    clf = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        **chromosome
    )
    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", clf),
    ])


def evaluate_fitness(chromosome, preprocessor, X_train, y_train, cv=3):
    """
    Fitness = mean cross-validated ROC-AUC.
    Returns (fitness_score, std).
    """
    try:
        pipeline = build_pipeline(preprocessor, chromosome)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean()), float(scores.std())
    except Exception:
        return 0.0, 0.0


# ─────────────────────────────────────────────
#  Main GA runner
# ─────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for XGBoost hyperparameters.

    Parameters
    ----------
    population_size : int
        Number of chromosomes per generation.
    n_generations   : int
        Number of generations to evolve.
    crossover_rate  : float
        Probability that two parents exchange genes.
    mutation_rate   : float
        Per-gene probability of mutation.
    elitism         : int
        Number of top individuals carried unchanged to next generation.
    cv_folds        : int
        Cross-validation folds for fitness evaluation.
    """

    def __init__(
        self,
        population_size=20,
        n_generations=15,
        crossover_rate=0.7,
        mutation_rate=0.15,
        elitism=2,
        cv_folds=3,
        random_seed=42,
    ):
        self.population_size = population_size
        self.n_generations   = n_generations
        self.crossover_rate  = crossover_rate
        self.mutation_rate   = mutation_rate
        self.elitism         = elitism
        self.cv_folds        = cv_folds
        self.random_seed     = random_seed

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Track evolution history
        self.history = []           # list of dicts per generation
        self.best_chromosome = None
        self.best_fitness    = -np.inf
        self.best_pipeline   = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, preprocessor, progress_callback=None):
        """
        Run the genetic algorithm.

        Parameters
        ----------
        X_train, y_train   : training data
        preprocessor       : sklearn ColumnTransformer (already configured)
        progress_callback  : optional callable(gen, best_fitness, log_str)
        """
        start_time = time.time()

        # ── Initialise population ──────────────────────────────────────
        population = [random_chromosome() for _ in range(self.population_size)]

        for gen in range(self.n_generations):
            gen_start = time.time()

            # ── Evaluate fitness ──────────────────────────────────────
            fitnesses = []
            stds      = []
            for chrom in population:
                f, s = evaluate_fitness(chrom, preprocessor,
                                        X_train, y_train, self.cv_folds)
                fitnesses.append(f)
                stds.append(s)

            # ── Find generation best ──────────────────────────────────
            gen_best_idx     = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            gen_best_chrom   = deepcopy(population[gen_best_idx])

            if gen_best_fitness > self.best_fitness:
                self.best_fitness    = gen_best_fitness
                self.best_chromosome = deepcopy(gen_best_chrom)

            # ── Record history ────────────────────────────────────────
            gen_record = {
                "generation":    gen + 1,
                "best_fitness":  round(gen_best_fitness, 6),
                "mean_fitness":  round(float(np.mean(fitnesses)), 6),
                "std_fitness":   round(float(np.std(fitnesses)), 6),
                "best_params":   deepcopy(gen_best_chrom),
                "elapsed_sec":   round(time.time() - gen_start, 2),
            }
            self.history.append(gen_record)

            log_str = (
                f"Gen {gen+1:3d}/{self.n_generations} | "
                f"Best ROC-AUC: {gen_best_fitness:.4f} | "
                f"Mean: {float(np.mean(fitnesses)):.4f} | "
                f"Global Best: {self.best_fitness:.4f} | "
                f"Time: {gen_record['elapsed_sec']}s"
            )
            print(log_str)

            if progress_callback:
                progress_callback(gen + 1, self.best_fitness, log_str)

            # ── Build next generation ─────────────────────────────────
            # Sort by fitness descending
            sorted_idx = np.argsort(fitnesses)[::-1]
            sorted_pop = [population[i] for i in sorted_idx]

            next_gen = []

            # Elitism: carry top individuals unchanged
            for i in range(self.elitism):
                next_gen.append(deepcopy(sorted_pop[i]))

            # Fill rest with crossover + mutation
            while len(next_gen) < self.population_size:
                parent1 = tournament_select(population, fitnesses)
                parent2 = tournament_select(population, fitnesses)
                child1, child2 = crossover(parent1, parent2, self.crossover_rate)
                child1 = mutate(child1, self.mutation_rate)
                child2 = mutate(child2, self.mutation_rate)
                next_gen.append(child1)
                if len(next_gen) < self.population_size:
                    next_gen.append(child2)

            population = next_gen

        # ── Build and return best pipeline ────────────────────────────
        self.best_pipeline = build_pipeline(preprocessor, self.best_chromosome)
        self.best_pipeline.fit(X_train, y_train)

        total_time = round(time.time() - start_time, 1)
        print(f"\n🧬 GA complete in {total_time}s | Best ROC-AUC (CV): {self.best_fitness:.4f}")
        print(f"   Best params: {self.best_chromosome}")

        return self

    # ------------------------------------------------------------------
    def get_history_df(self):
        """Return evolution history as a DataFrame."""
        rows = []
        for h in self.history:
            row = {
                "generation":   h["generation"],
                "best_fitness": h["best_fitness"],
                "mean_fitness": h["mean_fitness"],
                "std_fitness":  h["std_fitness"],
                "elapsed_sec":  h["elapsed_sec"],
            }
            row.update({f"param_{k}": v for k, v in h["best_params"].items()})
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def save_history(self, path="ga_history.json"):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ GA history saved to {path}")


# ─────────────────────────────────────────────
#  Standalone runner (called from model.py or CLI)
# ─────────────────────────────────────────────

def run_ga_optimization(X_train, y_train, preprocessor,
                         population_size=20, n_generations=15,
                         save_history=True):
    """
    Convenience wrapper to run the GA and return the best pipeline + history.

    Returns
    -------
    ga          : fitted GeneticAlgorithm instance
    best_params : dict of best hyperparameters
    """
    ga = GeneticAlgorithm(
        population_size=population_size,
        n_generations=n_generations,
        crossover_rate=0.7,
        mutation_rate=0.15,
        elitism=2,
        cv_folds=3,
        random_seed=42,
    )
    ga.fit(X_train, y_train, preprocessor)

    if save_history:
        ga.save_history("ga_history.json")
        ga.get_history_df().to_csv("ga_evolution_log.csv", index=False)
        print("✅ Evolution log saved to ga_evolution_log.csv")

    return ga, ga.best_chromosome
