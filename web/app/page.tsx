"use client";

/**
 * Home page: client-side form that POSTs clinical features to /api/predict.
 */
import { useState } from "react";

type PredictResponse = {
  outcome: number;
  probability_no_diabetes: number;
  probability_diabetes: number;
};

/** Renders the prediction form and displays outcome or API errors. */
export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    pregnancies: 0,
    glucose: 120,
    blood_pressure: 70,
    skin_thickness: 20,
    insulin: 80,
    bmi: 28,
    diabetes_pedigree_function: 0.5,
    age: 30,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || data.message || "Request failed");
        return;
      }
      setResult(data as PredictResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <h1>Diabetes Prediction</h1>
      <p>Trainer</p>
      <p>Enter patient information to predict diabetes risk.</p>
      <form onSubmit={handleSubmit}>
        <div className="form-row">
          <div>
            <label htmlFor="pregnancies">Pregnancies</label>
            <input
              id="pregnancies"
              type="number"
              min={0}
              max={20}
              value={form.pregnancies}
              onChange={(e) =>
                setForm((f) => ({ ...f, pregnancies: Number(e.target.value) }))
              }
            />
          </div>
          <div>
            <label htmlFor="glucose">Glucose (mg/dL)</label>
            <input
              id="glucose"
              type="number"
              min={0}
              max={199}
              value={form.glucose}
              onChange={(e) =>
                setForm((f) => ({ ...f, glucose: Number(e.target.value) }))
              }
            />
          </div>
        </div>
        <div className="form-row">
          <div>
            <label htmlFor="blood_pressure">Blood Pressure (mmHg)</label>
            <input
              id="blood_pressure"
              type="number"
              min={0}
              max={122}
              value={form.blood_pressure}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  blood_pressure: Number(e.target.value),
                }))
              }
            />
          </div>
          <div>
            <label htmlFor="skin_thickness">Skin Thickness (mm)</label>
            <input
              id="skin_thickness"
              type="number"
              min={0}
              max={99}
              value={form.skin_thickness}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  skin_thickness: Number(e.target.value),
                }))
              }
            />
          </div>
        </div>
        <div className="form-row">
          <div>
            <label htmlFor="insulin">Insulin (µU/mL)</label>
            <input
              id="insulin"
              type="number"
              min={0}
              max={846}
              value={form.insulin}
              onChange={(e) =>
                setForm((f) => ({ ...f, insulin: Number(e.target.value) }))
              }
            />
          </div>
          <div>
            <label htmlFor="bmi">BMI</label>
            <input
              id="bmi"
              type="number"
              min={0}
              max={67}
              step={0.1}
              value={form.bmi}
              onChange={(e) =>
                setForm((f) => ({ ...f, bmi: Number(e.target.value) }))
              }
            />
          </div>
        </div>
        <div className="form-row">
          <div>
            <label htmlFor="diabetes_pedigree_function">
              Diabetes Pedigree Function
            </label>
            <input
              id="diabetes_pedigree_function"
              type="number"
              min={0.078}
              max={2.42}
              step={0.01}
              value={form.diabetes_pedigree_function}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  diabetes_pedigree_function: Number(e.target.value),
                }))
              }
            />
          </div>
          <div>
            <label htmlFor="age">Age (years)</label>
            <input
              id="age"
              type="number"
              min={21}
              max={81}
              value={form.age}
              onChange={(e) =>
                setForm((f) => ({ ...f, age: Number(e.target.value) }))
              }
            />
          </div>
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Predicting…" : "Predict"}
        </button>
      </form>
      {error && (
        <div className={`result result-error`}>
          {error}
        </div>
      )}
      {result && (
        <div
          className={`result ${
            result.outcome === 1 ? "high-risk" : "low-risk"
          }`}
        >
          <p>
            <strong>
              {result.outcome === 1 ? "High risk of diabetes" : "Low risk of diabetes"}
            </strong>
          </p>
          <p>No diabetes: {(result.probability_no_diabetes * 100).toFixed(1)}%</p>
          <p>Diabetes: {(result.probability_diabetes * 100).toFixed(1)}%</p>
        </div>
      )}
    </main>
  );
}
