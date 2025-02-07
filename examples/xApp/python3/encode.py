import torch
import torch.nn as nn

class Encoder:
    def __init__(self, action_space, embedding_dim=10):
        """
        Encoder for actions and contexts.

        :param action_space: List of all possible actions ([MCS, PRB]).
        :param embedding_dim: The desired embedding dimension (default: 10).
        """
        self.embedding_dim = embedding_dim
        
        # Action embeddings: Precompute embeddings for all actions
        self.action_space = torch.tensor(action_space, dtype=torch.long)
        self.num_actions = len(action_space)
        self.action_embedding = nn.Embedding(self.num_actions, embedding_dim)
        self.action_embeddings = self.action_embedding(
            torch.arange(self.num_actions)
        )  # Precompute embeddings for actions
        
        # Context embeddings: Dynamically encode contexts
        self.context_embedding = nn.EmbeddingBag(
            num_embeddings=1000,  # Initial size for unique contexts (expandable)
            embedding_dim=embedding_dim,
            mode="mean"
        )
        self.context_index = {}  # Store unique context indices
        self.context_counter = 0  # To assign new indices dynamically

    def encode_action(self, action):
        """
        Encode a given action ([MCS, PRB]).
        :param action: [MCS, PRB] pair as a list or tensor.
        :return: Encoded 10-dimensional action embedding.
        """
        action = torch.tensor(action, dtype=torch.long)
        action_idx = (self.action_space == action).all(dim=1).nonzero(as_tuple=True)[0]
        return self.action_embeddings[action_idx]

    def encode_context(self, context):
        """
        Dynamically encode a given context ([SNR, Demand]).
        :param context: [SNR, Demand] pair as a list or tensor.
        :return: Encoded 10-dimensional context embedding.
        """
        context = tuple(context)  # Convert to tuple for dictionary key
        if context not in self.context_index:
            # Add new context dynamically
            self.context_index[context] = self.context_counter
            self.context_counter += 1
            
            # Expand embedding bag if needed
            if self.context_counter > self.context_embedding.num_embeddings:
                self.expand_context_embedding()

        # Get the index of the context and return its embedding
        context_idx = self.context_index[context]
        context_idx_tensor = torch.tensor([context_idx], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        return self.context_embedding(context_idx_tensor, offsets)

    def expand_context_embedding(self):
        """
        Expand the context embedding layer dynamically when the number of contexts exceeds its current capacity.
        """
        new_size = self.context_counter + 100  # Add buffer for new contexts
        new_context_embedding = nn.EmbeddingBag(
            num_embeddings=new_size,
            embedding_dim=self.embedding_dim,
            mode="mean",
        )
        # Copy weights from the old embedding
        new_context_embedding.weight.data[: self.context_embedding.num_embeddings] = (
            self.context_embedding.weight.data
        )
        self.context_embedding = new_context_embedding

# Example Usage
# Define possible actions ([MCS, PRB] pairs)
actions = [[9, 50], [9, 55], [10, 50], [10, 55]]  # Example actions

# Initialize Encoder
encoder = Encoder(actions)

# Encode actions
action = [9, 50]  # Example action
encoded_action = encoder.encode_action(action)
print("Encoded Action:", encoded_action)

# Encode contexts dynamically
context = [5, 20]  # Example context [SNR, Demand]
encoded_context = encoder.encode_context(context)
print("Encoded Context:", encoded_context)

# Encode another context
new_context = [8, 40]
new_encoded_context = encoder.encode_context(new_context)
print("New Encoded Context:", new_encoded_context)

# Check stored contexts
print("Stored Context Indices:", encoder.context_index)
