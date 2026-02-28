from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict

class SemanticAnalyzer:
    """
    Handles semantic analysis of voice commands using Doc2Vec.
    
    This component helps in understanding the semantic distance between different
    voice commands. In the context of traffic fingerprinting, similar commands 
    might produce similar traffic patterns.
    """

    def __init__(self, vector_size: int = 20, min_count: int = 1, epochs: int = 100):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
        self.command_corpus = []

    def prepare_corpus(self, commands: List[str]):
        """
        Prepares the list of commands for Doc2Vec training.
        Each command is tokenized and tagged.
        """
        self.command_corpus = [
            TaggedDocument(words=cmd.lower().split(), tags=[cmd]) 
            for cmd in commands
        ]

    def train(self, commands: List[str]):
        """
        Trains the Doc2Vec model on the provided list of commands.
        """
        if not commands:
            raise ValueError("Command list is empty.")

        print(f"Training Doc2Vec on {len(commands)} commands...")
        self.prepare_corpus(commands)
        
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=42
        )
        
        self.model.build_vocab(self.command_corpus)
        self.model.train(
            self.command_corpus, 
            total_examples=self.model.corpus_count, 
            epochs=self.model.epochs
        )
        print("Doc2Vec training complete.")

    def get_similarity(self, cmd1: str, cmd2: str) -> float:
        """
        Computes the cosine similarity between two commands.
        """
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Infer vectors for the commands (even if they were in the training set, inference is robust)
        # Note: For strict reproducibility, we seed the inference if possible, but gensim's infer_vector is stochastic.
        # We can use the trained vectors of the tags if exact retrieval is needed.
        
        # Using the tag vector lookup for trained commands is more consistent for this use case
        if cmd1 in self.model.dv and cmd2 in self.model.dv:
            return self.model.dv.similarity(cmd1, cmd2)
        
        # Fallback to inference for unseen commands
        v1 = self.model.infer_vector(cmd1.lower().split())
        v2 = self.model.infer_vector(cmd2.lower().split())
        return float(cosine_similarity([v1], [v2])[0][0])

    def get_top_k_similar(self, target_cmd: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns the top K most semantically similar commands to the target.
        """
        if not self.model:
            raise RuntimeError("Model not trained.")
            
        # Use most_similar from gensim which uses the tags
        try:
            return self.model.dv.most_similar(target_cmd, topn=k)
        except KeyError:
            # If target_cmd not in vocab, we can't use most_similar on tags directly without a vector
            # But the requirement implies we are comparing against the known set.
            # Let's infer the vector and find similar tags.
            inferred_vector = self.model.infer_vector(target_cmd.lower().split())
            return self.model.dv.most_similar([inferred_vector], topn=k)

    def evaluate_semantic_accuracy(self, y_true: List[str], y_pred_probs: np.ndarray, 
                                 classes: List[str], k: int = 3) -> float:
        """
        Calculates Top-K Accuracy where a prediction is considered correct if
        the true label is within the top K predictions, weighted by semantic similarity.
        
        Actually, standard Top-K accuracy just checks if true label is in top K probabilities.
        
        The requirement "Integrate semantic similarity into evaluation" suggests:
        Did the model predict a command that is semantically close to the true command?
        
        Metric: Semantic Precision @ 1
        If predicted != true, but similarity(predicted, true) > threshold, we count it as a "semantic match".
        
        OR
        
        A continuous score: Average Semantic Similarity of the predicted command vs true command.
        """
        
        total_score = 0
        n = len(y_true)
        
        if len(y_true) == 0:
            return 0.0

        print(f"Evaluating Semantic Similarity on {n} samples...")

        # Let's implement Average Semantic Similarity of the top prediction
        # Get the index of the max probability
        top_pred_indices = np.argmax(y_pred_probs, axis=1)
        
        similarities = []
        
        for i, true_label in enumerate(y_true):
            pred_label = classes[top_pred_indices[i]]
            
            if true_label == pred_label:
                sim = 1.0
            else:
                sim = self.get_similarity(true_label, pred_label)
            
            similarities.append(sim)
            
        avg_sim = np.mean(similarities)
        print(f"Average Semantic Similarity of Predictions: {avg_sim:.4f}")
        
        return avg_sim

if __name__ == "__main__":
    # Test stub
    analyzer = SemanticAnalyzer()
    
    commands = [
        "turn on the lights",
        "switch on the lights",
        "play music",
        "start music",
        "what is the weather",
        "tell me the weather"
    ]
    
    analyzer.train(commands)
    
    sim = analyzer.get_similarity("turn on the lights", "switch on the lights")
    print(f"Similarity ('turn on the lights', 'switch on the lights'): {sim:.4f}")
    
    sim_diff = analyzer.get_similarity("turn on the lights", "play music")
    print(f"Similarity ('turn on the lights', 'play music'): {sim_diff:.4f}")
