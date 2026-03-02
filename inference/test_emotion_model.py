"""
Test Script for Emotion Detection Model

This script tests if the pretrained emotion detection model can be loaded
and used for inference. It verifies that the model auto-downloads correctly
from Hugging Face.

Model: bhadresh-savani/distilbert-base-uncased-emotion
This model can detect 6 emotions: sadness, joy, love, anger, fear, surprise
"""

from transformers import pipeline
import torch


def test_emotion_model():
    """
    Test function to load the emotion detection model and make a prediction.
    
    This function:
    1. Loads the pretrained model from Hugging Face (auto-downloads if needed)
    2. Uses a hardcoded test sentence
    3. Predicts the emotion
    4. Prints the results with confidence score
    """
    
    print("=" * 60)
    print("Testing Emotion Detection Model")
    print("=" * 60)
    
    # Step 1: Load the emotion detection model
    # The pipeline automatically downloads the model if it's not already cached
    # This may take a few minutes the first time you run it
    print("\n[Step 1] Loading emotion detection model...")
    print("Model: bhadresh-savani/distilbert-base-uncased-emotion")
    print("Note: Model will auto-download if not already cached. This may take a few minutes...")
    
    try:
        # Create a text classification pipeline with the emotion model
        # This automatically handles tokenization and model inference
        emotion_classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True  # Get all emotion scores, not just the top one
        )
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed transformers: pip install transformers")
        print("2. Check your internet connection (needed for first download)")
        print("3. Ensure you have enough disk space (model is ~250MB)")
        return
    
    # Step 2: Define a test sentence
    # This is a hardcoded example to test the model
    print("\n[Step 2] Preparing test sentence...")
    test_sentence = "I am feeling really happy and excited about this new project!"
    print(f"Test sentence: '{test_sentence}'")
    
    # Step 3: Make a prediction
    # The model will analyze the sentence and predict the emotion
    print("\n[Step 3] Running emotion detection...")
    
    try:
        # Get predictions for all emotions
        # The result is a list of dictionaries, one for each emotion class
        predictions = emotion_classifier(test_sentence)
        
        # Extract the first item (since we're passing a single sentence)
        # predictions[0] contains a list of all emotion scores
        emotion_scores = predictions[0]
        
        print("✓ Prediction completed!")
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return
    
    # Step 4: Display results
    # Show all emotion predictions with their confidence scores
    print("\n[Step 4] Results:")
    print("-" * 60)
    
    # Sort emotions by confidence score (highest first)
    sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
    
    print("\nAll emotion predictions (sorted by confidence):")
    for i, emotion_result in enumerate(sorted_emotions, 1):
        emotion_label = emotion_result['label']
        confidence = emotion_result['score']
        # Convert confidence to percentage for easier reading
        confidence_percent = confidence * 100
        
        # Highlight the top prediction
        if i == 1:
            print(f"  🏆 {i}. {emotion_label.upper()}: {confidence_percent:.2f}% (CONFIDENCE: {confidence:.4f})")
        else:
            print(f"     {i}. {emotion_label}: {confidence_percent:.2f}% (confidence: {confidence:.4f})")
    
    # Extract the top prediction
    top_emotion = sorted_emotions[0]
    predicted_emotion = top_emotion['label']
    predicted_confidence = top_emotion['score']
    
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"  Predicted Emotion: {predicted_emotion.upper()}")
    print(f"  Confidence: {predicted_confidence:.4f} ({predicted_confidence * 100:.2f}%)")
    print("-" * 60)
    
    # Step 5: Verify model is working correctly
    print("\n[Step 5] Model verification:")
    if predicted_confidence > 0.5:  # Reasonable confidence threshold
        print("✓ Model is working correctly!")
        print(f"  The model confidently predicted '{predicted_emotion}' emotion.")
    else:
        print("⚠ Model prediction has low confidence.")
        print("  This might indicate the sentence is ambiguous or unclear.")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    """
    This block runs when the script is executed directly.
    It's a common Python pattern to allow the script to be run standalone
    or imported as a module in other scripts.
    """
    # Check if PyTorch is available
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Note: Could not check PyTorch info: {e}")
    
    # Run the test
    test_emotion_model()

