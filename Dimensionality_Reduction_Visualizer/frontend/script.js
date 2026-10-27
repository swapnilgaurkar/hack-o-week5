const API_BASE = "http://127.0.0.1:5000";

let datasetId = null;
let numericColumns = [];
let categoricalColumns = [];

const $ = (id) => document.getElementById(id);

function showMessage(text, type = "error") {
    const box = $("message");
    box.textContent = text;
    box.className = `message ${type === "success" ? "success" : ""}`;
    box.classList.remove("hidden");
}

function hideMessage() {
    $("message").classList.add("hidden");
}

function setBusy(button, busy, busyText) {
    button.disabled = busy;
    if (busy) {
        button.dataset.original = button.textContent;
        button.textContent = busyText;
    } else if (button.dataset.original) {
        button.textContent = button.dataset.original;
    }
}

$("csvFile").addEventListener("change", () => {
    const file = $("csvFile").files[0];
    $("fileLabel").textContent = file ? file.name : "Choose CSV file";
});

$("uploadBtn").addEventListener("click", uploadDataset);

async function uploadDataset() {
    hideMessage();
    const file = $("csvFile").files[0];

    if (!file) {
        showMessage("Please choose a CSV file first.");
        return;
    }

    const button = $("uploadBtn");
    setBusy(button, true, "Analyzing...");

    try {
        const form = new FormData();
        form.append("file", file);

        const response = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: form
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Upload failed.");

        datasetId = data.dataset_id;
        numericColumns = data.numeric_columns || [];
        categoricalColumns = data.categorical_columns || [];

        $("filename").textContent = data.filename;
        $("rows").textContent = data.rows;
        $("columns").textContent = data.columns;
        $("numericFeatures").textContent = data.numeric_features;
        $("missingValues").textContent = data.missing_values;

        populateClassColumn(categoricalColumns);
        renderPreview(data.preview || []);

        $("datasetSection").classList.remove("hidden");
        $("pcaSection").classList.remove("hidden");
        $("tsneSection").classList.remove("hidden");
        $("comparisonSection").classList.remove("hidden");

        clearCharts();
        showMessage("Dataset loaded successfully. Choose a class column and run PCA or t-SNE.", "success");
        $("datasetSection").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showMessage(error.message);
    } finally {
        setBusy(button, false);
    }
}

function populateClassColumn(columns) {
    const select = $("classColumn");
    select.innerHTML = `<option value="">No class coloring</option>`;
    columns.forEach(column => {
        const option = document.createElement("option");
        option.value = column;
        option.textContent = column;
        select.appendChild(option);
    });

    // For the supplied sample dataset, automatically choose the meaningful class.
    const performance = columns.find(c => c.toLowerCase() === "performance_level");
    if (performance) select.value = performance;
}

function renderPreview(rows) {
    const table = $("previewTable");
    table.innerHTML = "";

    if (!rows.length) return;

    const columns = Object.keys(rows[0]);
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = columns.map(c => `<td>${escapeHtml(row[c])}</td>`).join("");
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function requestPayload() {
    return {
        dataset_id: datasetId,
        class_column: $("classColumn").value || null
    };
}

$("pcaBtn").addEventListener("click", runPCA);

async function runPCA() {
    if (!datasetId) return showMessage("Upload a dataset first.");

    const button = $("pcaBtn");
    setBusy(button, true, "Running PCA...");

    try {
        const response = await fetch(`${API_BASE}/api/pca`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestPayload())
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "PCA failed.");

        $("pcaVariance").classList.remove("hidden");
        $("pc1Value").textContent = `${data.explained_variance[0]}%`;
        $("pc2Value").textContent = `${data.explained_variance[1]}%`;
        $("totalValue").textContent = `${(
            data.explained_variance[0] + data.explained_variance[1]
        ).toFixed(2)}%`;

        drawScatter("pcaChart", data, "PC1", "PC2", "PCA");
        drawVariance(data);
    } catch (error) {
        showMessage(error.message);
    } finally {
        setBusy(button, false);
    }
}

function drawScatter(elementId, data, xTitle, yTitle, chartTitle) {
    if (typeof Plotly === "undefined") {
        throw new Error("Plotly.js could not be loaded. Check your internet connection.");
    }

    const groups = groupPoints(data);

    const traces = Object.entries(groups).map(([group, indices]) => ({
        x: indices.map(i => data.x[i]),
        y: indices.map(i => data.y[i]),
        mode: "markers",
        type: "scatter",
        name: group === "__all__" ? "Data points" : group,
        text: indices.map(i => `Row ${i + 1}`),
        hovertemplate: `${xTitle}: %{x:.3f}<br>${yTitle}: %{y:.3f}<br>%{text}<extra></extra>`,
        marker: { size: 9, opacity: .78 }
    }));

    Plotly.newPlot(elementId, traces, {
        title: chartTitle,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 48, r: 25, b: 55, l: 55 },
        xaxis: { title: xTitle, zeroline: false, gridcolor: "#e8eaf0" },
        yaxis: { title: yTitle, zeroline: false, gridcolor: "#e8eaf0" },
        legend: { orientation: "h", y: -0.18 },
        font: { family: "Inter, Arial, sans-serif", color: "#182033" }
    }, { responsive: true, displaylogo: false });
}

function groupPoints(data) {
    if (!data.class_values || data.class_values.length !== data.x.length) {
        return { "__all__": data.x.map((_, i) => i) };
    }

    return data.class_values.reduce((groups, value, index) => {
        const key = String(value);
        if (!groups[key]) groups[key] = [];
        groups[key].push(index);
        return groups;
    }, {});
}

function drawVariance(data) {
    const x = ["PC1", "PC2"];
    const y = data.explained_variance;

    Plotly.newPlot("varianceChart", [{
        x, y, type: "bar",
        text: y.map(v => `${v}%`),
        textposition: "auto"
    }], {
        title: "Explained Variance",
        yaxis: { title: "Variance (%)", range: [0, Math.max(100, Math.max(...y) + 15)] },
        margin: { t: 48, r: 20, b: 45, l: 50 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Inter, Arial, sans-serif", color: "#182033" }
    }, { responsive: true, displaylogo: false });
}

$("tsneBtn").addEventListener("click", runTSNE);

async function runTSNE() {
    if (!datasetId) return showMessage("Upload a dataset first.");

    const button = $("tsneBtn");
    setBusy(button, true, "Running t-SNE...");

    try {
        const payload = requestPayload();
        payload.perplexity = Number($("perplexity").value);

        const response = await fetch(`${API_BASE}/api/tsne`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "t-SNE failed.");

        $("actualPerplexity").textContent = data.perplexity;
        drawScatter("tsneChart", data, "Dimension 1", "Dimension 2", "t-SNE");
    } catch (error) {
        showMessage(error.message);
    } finally {
        setBusy(button, false);
    }
}

function clearCharts() {
    ["pcaChart", "varianceChart", "tsneChart"].forEach(id => {
        const el = $(id);
        if (el && typeof Plotly !== "undefined" && el.data) {
            Plotly.purge(el);
        }
    });

    $("pcaChart").textContent = "Run PCA to display the visualization.";
    $("tsneChart").textContent = "Run t-SNE to display the visualization.";
    $("pcaVariance").classList.add("hidden");
    $("actualPerplexity").textContent = "—";
}
