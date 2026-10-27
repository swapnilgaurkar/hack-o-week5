/* Moneta frontend: vanilla JS. Talks to the Flask API under /api. */
"use strict";
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const COLORS = ["#6f8cff", "#9b7bff", "#4ade9a", "#fbbf4a", "#ff6b7f", "#38bdf8", "#f472b6", "#a3e635", "#94a3b8"];
let categories = [], charts = {}, chartLib = null, dash = null;

/* ---------- helpers ---------- */
async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: {"Content-Type": "application/json"}, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined});
  let data = {};
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const e = new Error(data.error || "Request failed");
    e.details = data.details || [];
    throw e;
  }
  return data;
}
let toastTimer;
function toast(msg, kind = "ok") {
  const t = $("toast");
  t.textContent = msg; t.className = "toast show " + kind;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = "toast"), 3800);
}
const errMsg = (e) => e.details && e.details.length ? e.details.join("; ") : e.message;
function loadScript(src) {
  return new Promise((ok, no) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = no; document.head.appendChild(s); });
}
async function loadChartLib() {
  for (const src of ["vendor/chart.umd.min.js", "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"]) {
    try { await loadScript(src); if (window.Chart) { chartLib = window.Chart; return; } } catch (_) { /* try next */ }
  }
}
function drawChart(id, fbId, config, fallbackText) {
  if (charts[id]) charts[id].destroy();
  const fb = fbId && $(fbId);
  if (!chartLib) { if (fb) fb.textContent = "Charts unavailable (Chart.js could not load — see README). " + (fallbackText || ""); return; }
  if (fb) fb.textContent = "";
  chartLib.defaults.color = "#9aa4c7";
  chartLib.defaults.borderColor = "rgba(255,255,255,.08)";
  config.options = Object.assign({responsive: true, maintainAspectRatio: false}, config.options || {});
  charts[id] = new chartLib($(id), config);
}

/* ---------- routing ---------- */
function route() {
  const page = (location.hash || "#overview").slice(1);
  const valid = ["overview", "expenses", "insights", "predictor", "lab"].includes(page) ? page : "overview";
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + valid));
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === valid));
  if (valid === "overview" || valid === "insights") refreshDashboard();
  if (valid === "expenses") loadExpenses();
  if (valid === "lab") refreshModelStatus();
}

/* ---------- dashboard ---------- */
async function refreshDashboard() {
  try { dash = await api("/dashboard"); } catch (e) { toast(errMsg(e), "error"); return; }
  renderOverview(); renderInsights();
}
function renderOverview() {
  const d = dash;
  $("s-total").textContent = fmt(d.total_spent);
  $("s-budget").textContent = d.budget > 0 ? fmt(d.budget) : "Not set";
  $("s-remaining").textContent = d.budget > 0 ? fmt(d.remaining) : "—";
  $("s-remaining").style.color = d.budget > 0 && d.remaining < 0 ? "var(--bad)" : "";
  $("s-count").textContent = d.transaction_count;
  const u = d.utilization_percent;
  $("s-util").textContent = u === null ? "Set a budget to track usage" : u + "%";
  $("s-bar").style.width = Math.min(u || 0, 100) + "%";
  $("s-bar").classList.toggle("over", u !== null && u > 100);
  if (document.activeElement !== $("budget-input")) $("budget-input").value = d.budget > 0 ? d.budget : "";
  $("recent").innerHTML = d.recent.length ? "<table><tbody>" + d.recent.map((r) =>
    `<tr><td>${esc(r.date)}</td><td><span class="chip">${esc(r.category)}</span></td><td>${esc(r.description || "")}</td><td class="r">${fmt(r.amount)}</td></tr>`).join("") + "</tbody></table>"
    : '<div class="empty">No transactions yet. <a href="#expenses">Add an expense</a> to get started.</div>';
  drawChart("ch-daily", "fb-daily", {type: "line", data: {labels: d.daily.map((x) => x.date.slice(8)),
    datasets: [{label: "Daily spending", data: d.daily.map((x) => x.total), borderColor: "#6f8cff", backgroundColor: "rgba(111,140,255,.2)", fill: true, tension: .35, pointRadius: 2}]},
    options: {plugins: {legend: {display: false}}}}, `Total this month: ${fmt(d.total_spent)}`);
  drawChart("ch-cat", "fb-cat", {type: "doughnut", data: {labels: d.category_totals.map((c) => c.category),
    datasets: [{data: d.category_totals.map((c) => c.total), backgroundColor: COLORS, borderWidth: 0}]},
    options: {cutout: "68%", plugins: {legend: {position: "bottom", labels: {boxWidth: 10}}}}},
    d.category_totals.map((c) => `${c.category}: ${fmt(c.total)}`).join(", "));
}
function renderInsights() {
  const d = dash, has = d.transaction_count > 0;
  $("ins-empty").hidden = has; $("ins-body").hidden = !has;
  if (!has) return;
  $("i-top").textContent = d.top_category ? `${d.top_category.category} (${d.top_category.percent}%)` : "—";
  $("i-actual").textContent = fmt(d.total_spent);
  $("i-budget").textContent = d.budget > 0 ? fmt(d.budget) : "Not set";
  $("i-util").textContent = d.utilization_percent === null ? "—" : d.utilization_percent + "%";
  $("cat-list").innerHTML = d.category_totals.map((c, i) =>
    `<div class="cat-row"><div class="row"><span>${esc(c.category)}</span><b>${fmt(c.total)} · ${c.percent}%</b></div>
     <div class="progress"><div style="width:${c.percent}%;background:${COLORS[i % COLORS.length]}"></div></div></div>`).join("");
  drawChart("ch-donut", null, {type: "doughnut", data: {labels: d.category_totals.map((c) => c.category),
    datasets: [{data: d.category_totals.map((c) => c.total), backgroundColor: COLORS, borderWidth: 0}]},
    options: {cutout: "65%", plugins: {legend: {position: "bottom", labels: {boxWidth: 10}}}}});
  const datasets = [{label: "Weekly spending", data: d.weekly.map((w) => w.total), backgroundColor: "#9b7bff", borderRadius: 8}];
  if (d.budget > 0) datasets.push({type: "line", label: "Weekly budget share", data: d.weekly.map(() => +(d.budget / 4.345).toFixed(2)), borderColor: "#fbbf4a", borderDash: [6, 4], pointRadius: 0});
  drawChart("ch-weekly", "fb-weekly", {type: "bar", data: {labels: d.weekly.map((w) => "Wk " + w.week_start.slice(5)), datasets}},
    d.weekly.map((w) => `${w.week_start}: ${fmt(w.total)}`).join(", "));
}
$("budget-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const v = parseFloat($("budget-input").value);
  if (!Number.isFinite(v) || v <= 0) return toast("Budget must be a positive number", "error");
  try { await api("/budget", {method: "POST", body: {monthly_budget: v}}); toast("Budget updated"); refreshDashboard(); }
  catch (e) { toast(errMsg(e), "error"); }
});

/* ---------- expenses ---------- */
async function loadCategories() {
  categories = (await api("/categories")).categories;
  $("e-category").innerHTML = '<option value="">Select category…</option>' + categories.map((c) => `<option>${esc(c)}</option>`).join("");
  $("f-cat").innerHTML = '<option value="">All categories</option>' + categories.map((c) => `<option>${esc(c)}</option>`).join("");
}
async function loadExpenses() {
  const [sort, order] = $("f-sort").value.split(":");
  const qs = new URLSearchParams({q: $("f-q").value, category: $("f-cat").value, sort, order});
  try {
    const {expenses} = await api("/expenses?" + qs);
    $("exp-body").innerHTML = expenses.map((r) =>
      `<tr><td>${esc(r.date)}</td><td><span class="chip">${esc(r.category)}</span></td><td>${esc(r.description || "")}</td>
       <td class="r">${fmt(r.amount)}</td><td class="r"><button class="btn ghost sm" data-edit="${r.id}">Edit</button>
       <button class="btn danger sm" data-del="${r.id}">Delete</button></td></tr>`).join("");
    $("exp-empty").hidden = expenses.length > 0;
    $("exp-empty").textContent = ($("f-q").value || $("f-cat").value) ? "No expenses match your filters." : "No expenses yet. Add your first one above.";
    $("exp-body").dataset.rows = JSON.stringify(expenses);
  } catch (e) { toast(errMsg(e), "error"); }
}
function resetForm() {
  $("expense-form").reset(); $("e-id").value = ""; $("e-date").value = new Date().toISOString().slice(0, 10);
  $("form-title").textContent = "Add expense"; $("e-save").textContent = "Add expense"; $("e-cancel").hidden = true;
}
$("expense-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const amount = parseFloat($("e-amount").value), body = {date: $("e-date").value, amount, category: $("e-category").value, description: $("e-desc").value};
  document.querySelectorAll("#expense-form input,#expense-form select").forEach((i) => i.classList.remove("invalid"));
  if (!Number.isFinite(amount) || amount <= 0) { $("e-amount").classList.add("invalid"); return toast("Amount must be a positive number", "error"); }
  if (!body.date) { $("e-date").classList.add("invalid"); return toast("Please pick a date", "error"); }
  if (!body.category) { $("e-category").classList.add("invalid"); return toast("Please choose a category", "error"); }
  const id = $("e-id").value;
  try {
    await api(id ? "/expenses/" + id : "/expenses", {method: id ? "PUT" : "POST", body});
    toast(id ? "Expense updated" : "Expense added"); resetForm(); loadExpenses();
  } catch (e) { toast(errMsg(e), "error"); }
});
$("e-cancel").addEventListener("click", resetForm);
$("exp-body").addEventListener("click", async (ev) => {
  const edit = ev.target.dataset.edit, del = ev.target.dataset.del;
  const rows = JSON.parse($("exp-body").dataset.rows || "[]");
  if (edit) {
    const r = rows.find((x) => String(x.id) === edit); if (!r) return;
    $("e-id").value = r.id; $("e-date").value = r.date; $("e-amount").value = r.amount; $("e-category").value = r.category; $("e-desc").value = r.description;
    $("form-title").textContent = "Edit expense"; $("e-save").textContent = "Save changes"; $("e-cancel").hidden = false;
    window.scrollTo({top: 0, behavior: "smooth"});
  }
  if (del && confirm("Delete this expense?")) {
    try { await api("/expenses/" + del, {method: "DELETE"}); toast("Expense deleted"); loadExpenses(); }
    catch (e) { toast(errMsg(e), "error"); }
  }
});
["f-q", "f-cat", "f-sort"].forEach((id) => $(id).addEventListener(id === "f-q" ? "input" : "change", loadExpenses));

/* ---------- predictor ---------- */
const PF = ["monthly_budget", "spent_so_far", "days_remaining", "avg_daily_spending", "previous_period_spending"];
const PF_LABEL = {monthly_budget: "Monthly budget", spent_so_far: "Spent so far", days_remaining: "Days remaining", avg_daily_spending: "Average daily spending", previous_period_spending: "Previous month's spending"};
$("p-fill").addEventListener("click", async () => {
  try { await refreshDashboard(); PF.forEach((k) => ($("p-" + k).value = dash.predictor_defaults[k])); toast("Filled from your tracked expenses"); }
  catch (e) { toast(errMsg(e), "error"); }
});
$("pred-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = {};
  for (const k of PF) {
    const el = $("p-" + k), v = parseFloat(el.value); el.classList.remove("invalid");
    if (!Number.isFinite(v) || v < 0 || (k === "monthly_budget" && v <= 0) || (k === "days_remaining" && v > 31)) { el.classList.add("invalid"); return toast(`Check “${PF_LABEL[k]}”`, "error"); }
    body[k] = v;
  }
  try {
    const r = await api("/predict", {method: "POST", body});
    $("pred-empty").hidden = true; $("pred-out").hidden = false;
    $("pr-pct").textContent = (r.probability * 100).toFixed(1) + "%";
    $("pr-badge").textContent = r.risk + " risk"; $("pr-badge").className = "badge " + r.risk;
    $("pr-explain").innerHTML = "Estimated chance of exceeding the budget, based on: " +
      PF.map((k) => `${PF_LABEL[k]} <b>${fmt(r.inputs[k])}</b>`).join(", ") +
      `. A simple straight-line projection (spent + average daily × days left) would be <b>${fmt(r.naive_projection)}</b>.`;
  } catch (e) { toast(errMsg(e), "error"); }
});

/* ---------- model lab ---------- */
function renderModel(m) {
  const trained = !!(m && m.metadata);
  $("m-none").hidden = trained; $("lab-metrics").hidden = !trained;
  if (!trained) { if (charts["ch-loss"]) charts["ch-loss"].destroy(); return; }
  const md = m.metadata, k = md.metrics;
  $("st-train").textContent = `Ready — ${md.optimizer.toUpperCase()}, ${md.activation}, lr ${md.learning_rate}, ${md.epochs} epochs`;
  $("m-init").textContent = md.initial_loss.toFixed(4); $("m-final").textContent = md.final_loss.toFixed(4);
  $("m-acc").textContent = (k.accuracy * 100).toFixed(1) + "%"; $("m-prec").textContent = (k.precision * 100).toFixed(1) + "%";
  $("m-rec").textContent = (k.recall * 100).toFixed(1) + "%"; $("m-f1").textContent = (k.f1 * 100).toFixed(1) + "%";
  $("m-note").textContent = `Measured on a ${md.metrics_dataset} (${k.n_samples} synthetic examples). Not evidence of real-world accuracy.`;
  const labels = md.loss_history.map((_, i) => i + 1);
  drawChart("ch-loss", "fb-loss", {type: "line", data: {labels, datasets: [
    {label: "Training loss", data: md.loss_history, borderColor: "#6f8cff", pointRadius: 0, tension: .25},
    {label: "Validation loss", data: md.val_loss_history, borderColor: "#fbbf4a", pointRadius: 0, tension: .25}]},
    options: {scales: {x: {title: {display: true, text: "Epoch"}}}}}, `Initial ${md.initial_loss.toFixed(4)} → final ${md.final_loss.toFixed(4)}.`);
}
async function refreshModelStatus() {
  try {
    const {model} = await api("/status");
    renderModel(model);
    $("st-save").textContent = model.model_file_exists ? (model.saved_in_memory_matches_disk ? "Saved on disk and loaded" : (model.trained ? "A saved file exists; current model is not saved" : "A saved model file exists — press “Load saved model”")) : "No saved model file yet";
    if (!model.trained) $("st-train").textContent = "Not trained — train or load a model";
  } catch (e) { toast(errMsg(e), "error"); }
}
$("train-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const lr = parseFloat($("t-lr").value), ep = parseInt($("t-ep").value, 10);
  if (!(lr >= 0.00001 && lr <= 1)) return toast("Learning rate must be between 0.00001 and 1", "error");
  if (!(ep >= 1 && ep <= 1000)) return toast("Epochs must be between 1 and 1000", "error");
  const btn = $("t-train"); btn.disabled = true; btn.textContent = "Training…"; $("st-train").textContent = "Training in progress…";
  try {
    await api("/train", {method: "POST", body: {optimizer: $("t-opt").value, activation: $("t-act").value, learning_rate: lr, epochs: ep}});
    toast("Training finished"); await refreshModelStatus();
  } catch (e) { toast(errMsg(e), "error"); $("st-train").textContent = "Training failed"; }
  finally { btn.disabled = false; btn.textContent = "Train model"; }
});
$("t-save").addEventListener("click", async () => {
  try { const r = await api("/model/save", {method: "POST"}); toast(r.message); refreshModelStatus(); } catch (e) { toast(errMsg(e), "error"); }
});
$("t-load").addEventListener("click", async () => {
  try { const r = await api("/model/load", {method: "POST"}); toast(r.message); refreshModelStatus(); } catch (e) { toast(errMsg(e), "error"); }
});

/* ---------- boot ---------- */
(async function init() {
  resetForm();
  window.addEventListener("hashchange", route);
  try { await loadCategories(); } catch (e) { toast("Could not reach the server: " + e.message, "error"); }
  await loadChartLib();
  route();
})();
