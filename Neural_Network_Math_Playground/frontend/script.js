async function post(url, data) {
    const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error("Server request failed");
    return response.json();
}

function parseVector(value) {
    return value.split(",").map(Number);
}

function parseMatrix(value) {
    return value.split(";").map(row => row.trim().split(",").map(Number));
}

function format(obj) {
    return JSON.stringify(obj, null, 2);
}

async function calculateVectors() {
    try {
        const result = await post("/api/vectors", {
            a: parseVector(document.getElementById("va").value),
            b: parseVector(document.getElementById("vb").value)
        });
        document.getElementById("vectorResult").textContent = format(result);
    } catch (e) {
        document.getElementById("vectorResult").textContent = "Error: " + e.message;
    }
}

async function calculateMatrices() {
    try {
        const result = await post("/api/matrices", {
            A: parseMatrix(document.getElementById("mA").value),
            B: parseMatrix(document.getElementById("mB").value),
            vector: parseVector(document.getElementById("mv").value)
        });
        document.getElementById("matrixResult").textContent = format(result);
    } catch (e) {
        document.getElementById("matrixResult").textContent = "Error: " + e.message;
    }
}

async function calculateDerivative() {
    try {
        const result = await post("/api/derivative", {
            x: document.getElementById("dx").value
        });
        document.getElementById("derivativeResult").textContent = format(result);
    } catch (e) {
        document.getElementById("derivativeResult").textContent = "Error: " + e.message;
    }
}

async function calculateGradient() {
    try {
        const result = await post("/api/gradient", {
            x: document.getElementById("gx").value,
            y: document.getElementById("gy").value
        });
        document.getElementById("gradientResult").textContent = format(result);
    } catch (e) {
        document.getElementById("gradientResult").textContent = "Error: " + e.message;
    }
}

async function runBackprop() {
    try {
        const result = await post("/api/backprop", {
            x: document.getElementById("bx").value,
            w1: document.getElementById("bw1").value,
            w2: document.getElementById("bw2").value,
            target: document.getElementById("bt").value,
            learning_rate: document.getElementById("blr").value
        });
        document.getElementById("backpropResult").textContent = format(result);
    } catch (e) {
        document.getElementById("backpropResult").textContent = "Error: " + e.message;
    }
}
