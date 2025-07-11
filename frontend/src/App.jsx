import { useState } from "react";

function App() {
  const [features, setFeatures] = useState(Array(13).fill(""));
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState([]);

  // Define feature ranges
  const featureRanges = [
    { label: "Age", min: 29, max: 77 },
    { label: "Sex (0 = Female, 1 = Male)", min: 0, max: 1 },
    { label: "Chest Pain Type (0-3)", min: 0, max: 3 },
    { label: "Resting Blood Pressure", min: 94, max: 200 },
    { label: "Cholesterol", min: 126, max: 564 },
    { label: "Fasting Blood Sugar (>120 = 1)", min: 0, max: 1 },
    { label: "Resting ECG Results (0-2)", min: 0, max: 2 },
    { label: "Max Heart Rate", min: 71, max: 202 },
    { label: "Exercise Induced Angina (0 or 1)", min: 0, max: 1 },
    { label: "ST Depression", min: 0, max: 6.2 },
    { label: "Slope of Peak Exercise ST (0-2)", min: 0, max: 2 },
    { label: "Number of Major Vessels (0-3)", min: 0, max: 3 },
    { label: "Thalassemia (1-3)", min: 1, max: 3 }
  ];

  const handleInputChange = (index, value) => {
    const updatedFeatures = [...features];
    updatedFeatures[index] = value;
    setFeatures(updatedFeatures);
  };

  const handlePredict = async () => {
    // Convert input values to numbers and check for errors
    const numericFeatures = features.map(Number);
    for (let i = 0; i < numericFeatures.length; i++) {
      if (numericFeatures[i] < featureRanges[i].min || numericFeatures[i] > featureRanges[i].max) {
        setError(`Incorrect information: ${featureRanges[i].label} must be between ${featureRanges[i].min} and ${featureRanges[i].max}`);
        return;
      }
    }
    setError(null);

    const response = await fetch("http://127.0.0.1:8080/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: numericFeatures }),
    });

    const data = await response.json();
    setPrediction(data.prediction);
    setProbability(data.probability);
    
    // Clear inputs after submission
    setFeatures(Array(13).fill(""));
  };

  return (
    <div>
      <h1>Heart Disease Prediction</h1>
      <div>
        {featureRanges.map((feature, index) => (
          <div key={index} style={{ marginBottom: "10px" }}>
            <label>{feature.label} ({feature.min}-{feature.max}): </label>
            <input
              type="number"
              value={features[index]}
              onChange={(e) => handleInputChange(index, e.target.value)}
            />
          </div>
        ))}
      </div>
      <button onClick={handlePredict}>Predict</button>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {prediction !== null && (
        <div>
          <h2>Prediction: {prediction === 1 ? "Disease Detected" : "No Disease"}</h2>
          <h3>Confidence Scores: {probability.map((p, i) => <p key={i}>{p.toFixed(3)}</p>)}</h3>
        </div>
      )}
    </div>
  );
}

export default App;
