const API = "http://127.0.0.1:5000/api";

const metricNames = [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1_score", "F1 Score"],
    ["roc_auc", "ROC-AUC"],
    ["cross_validation_mean", "5-Fold CV"]
];

async function loadEvaluation() {
    const container = document.getElementById("models");
    try {
        const response = await fetch(`${API}/evaluation`);
        const data = await response.json();

        container.innerHTML = Object.entries(data).map(([name, m]) => `
            <article class="model">
                <h3>${name}</h3>
                <div class="metrics">
                    ${metricNames.map(([key, label]) => `
                        <div class="metric">
                            <small>${label}</small>
                            <strong>${(m[key] * 100).toFixed(1)}%</strong>
                        </div>
                    `).join("")}
                </div>
                <div class="matrix">
                    <b>Confusion Matrix</b><br>
                    [[${m.confusion_matrix[0].join(", ")}],
                     [${m.confusion_matrix[1].join(", ")}]]
                    <br>CV Std: ${(m.cross_validation_std * 100).toFixed(2)}%
                </div>
            </article>
        `).join("");
    } catch (error) {
        container.innerHTML = "<p>Could not connect to backend. Start Flask first.</p>";
    }
}

document.getElementById("refreshBtn").addEventListener("click", loadEvaluation);

document.getElementById("loanForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.target);
    const data = Object.fromEntries(form.entries());

    ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
        .forEach(key => data[key] = Number(data[key]));

    const output = document.getElementById("prediction");
    output.textContent = "Predicting...";

    try {
        const response = await fetch(`${API}/predict`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });
        const result = await response.json();

        if (result.error) throw new Error(result.error);

        output.textContent =
            `Result: ${result.prediction} • Approval probability: ${result.probability}%`;
    } catch (error) {
        output.textContent = "Prediction failed. Make sure the backend is running.";
    }
});

loadEvaluation();
