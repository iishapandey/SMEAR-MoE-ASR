import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k routing and load balancing loss.
    
    Args:
        experts: List of 
        Encoder projectors 
        input_dim: Input dimension
        output_dim: Output dimension
        num_experts: Total number of experts
        k: Number of experts to route to for each token
        capacity_factor: Multiplicative factor for expert capacity
        load_balancing_weight: Weight of load balancing loss
    """
    def __init__(
        self,
        experts: nn.Module,
        input_dim=768,
        output_dim=768,
        num_experts=8,
        k=2,
        capacity_factor=1.0,
        load_balancing_weight=0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.load_balancing_weight = load_balancing_weight
        
        # Router network: takes input and produces expert routing logits
        self.router = nn.Linear(input_dim, num_experts)
        
        # Module list of encoder_projector-conv1d 
        self.experts = experts
        
        # Calculate expert capacity
        # Each token can be routed to k experts, so we need to ensure experts can handle the load
        self.capacity = int(capacity_factor * k * input_dim / num_experts)
        
    def forward(self, x):
        """Forward pass with load balancing loss calculation."""
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Get router logits and probabilities
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Calculate routing and load balancing losses
        # 1. Load balance loss: encourages uniform expert assignment
        # Calculate the fraction of tokens routed to each expert
        router_prob_mean = router_probs.mean(dim=0)
        # Ideal routing probability would be uniform across experts
        ideal_prob = torch.ones_like(router_prob_mean) / self.num_experts
        # Use mean squared error between actual and ideal probabilities
        load_balance_loss = self.load_balancing_weight * F.mse_loss(router_prob_mean, ideal_prob)
        
        # Get top-k experts and their probabilities for each token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.k, dim=-1)
        # Normalize the top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Create a tensor to collect outputs from all experts
        expert_outputs = torch.zeros(batch_size * seq_len, self.output_dim, device=x_flat.device)
        
        # Process inputs through each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens should be routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue  # Skip if no tokens are routed to this expert
                
            # Get the positions where this expert is selected
            expert_positions = torch.where(expert_mask)[0]
            # Limit to expert capacity if needed
            if len(expert_positions) > self.capacity:
                # If too many tokens, randomly select up to capacity
                perm = torch.randperm(len(expert_positions), device=expert_positions.device)
                expert_positions = expert_positions[perm[:self.capacity]]
                
            # Get corresponding inputs and probabilities for this expert
            expert_inputs = x_flat[expert_positions]
            # Find position of this expert in the top-k indices for each selected token
            position_in_topk = torch.where(top_k_indices[expert_positions] == expert_idx)[1]
            # Get corresponding probabilities
            expert_probs = top_k_probs[expert_positions, position_in_topk].unsqueeze(-1)
            
            # Process inputs through the expert and scale by probability
            expert_output = self.experts[expert_idx](expert_inputs) * expert_probs
            # Accumulate the outputs
            expert_outputs[expert_positions] += expert_output
            
        # Reshape output back to original shape
        y = expert_outputs.reshape(batch_size, seq_len, self.output_dim)
        
        print("y::", y.shape)
        return y, load_balance_loss
