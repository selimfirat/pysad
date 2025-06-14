#!/usr/bin/env python3

import warnings
import numpy as np

print("Testing NumPy deprecation warning fixes...")

# Catch deprecation warnings
warnings.filterwarnings('error', category=DeprecationWarning, message='.*Conversion of an array with ndim > 0 to a scalar.*')

try:
    # Test RelativeEntropy model
    print("Testing RelativeEntropy...")
    from pysad.models.relative_entropy import RelativeEntropy
    
    model = RelativeEntropy(min_val=0, max_val=100)
    
    # Create test data
    X = np.random.rand(100) * 100
    
    for i, x in enumerate(X):
        model.fit_partial(x)
        score = model.score_partial(x)
        if i % 20 == 0:
            print(f"  Processed {i+1} samples, last score: {score}")
    
    print("✓ RelativeEntropy: No deprecation warnings detected!")
    
except DeprecationWarning as e:
    print(f"✗ RelativeEntropy: Still has deprecation warning: {e}")
except Exception as e:
    print(f"✗ RelativeEntropy: Other error: {e}")

try:
    # Test BaseModel score method
    print("\nTesting BaseModel score method...")
    from pysad.models.standard_absolute_deviation import StandardAbsoluteDeviation
    
    model = StandardAbsoluteDeviation()
    
    # Create test data as 2D array
    X = np.random.rand(50, 1)
    
    # Fit the model first
    for x in X:
        model.fit_partial(x)
    
    # Test the score method that processes multiple instances
    scores = model.score(X[:10])
    print(f"  Scored {len(scores)} instances successfully")
    print("✓ BaseModel score method: No deprecation warnings detected!")
    
except DeprecationWarning as e:
    print(f"✗ BaseModel score method: Still has deprecation warning: {e}")
except Exception as e:
    print(f"✗ BaseModel score method: Other error: {e}")

try:
    # Test BasePostprocessor
    print("\nTesting BasePostprocessor...")
    from pysad.transform.postprocessing.postprocessors import ZScorePostprocessor
    
    postprocessor = ZScorePostprocessor()
    
    # Create test scores
    scores = np.random.rand(50)
    
    # Test fit_transform method
    processed_scores = postprocessor.fit_transform(scores)
    print(f"  Processed {len(processed_scores)} scores successfully")
    print("✓ BasePostprocessor: No deprecation warnings detected!")
    
except DeprecationWarning as e:
    print(f"✗ BasePostprocessor: Still has deprecation warning: {e}")
except Exception as e:
    print(f"✗ BasePostprocessor: Other error: {e}")

print("\nAll tests completed!")
