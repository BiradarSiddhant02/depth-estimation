from inferencing.model import Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Example usage
model = Model()  # Replace YourModel with your actual model class
num_params = count_parameters(model)
print(f"Number of parameters in the model: {num_params}")
