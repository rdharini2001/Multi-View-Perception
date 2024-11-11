import torch
import numpy as np

def mc_dropout_inference(model, input_image, dropout_prob=0.5, num_iterations=100):
    model.eval()  # Set model to evaluation mode
    predictions = []

    # Apply dropout during inference
    with torch.no_grad():
        for _ in range(num_iterations):
            # Enable dropout during inference
            model.train()  
            prediction = model(input_image)  # Forward pass through the model
            predictions.append(prediction)

    # Convert predictions to numpy for easy handling
    predictions = torch.stack(predictions).cpu().numpy()

    # Compute mean and uncertainty (standard deviation) of the predictions
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    
    return mean_prediction, uncertainty
