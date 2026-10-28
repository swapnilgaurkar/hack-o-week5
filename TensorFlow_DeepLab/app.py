import copy
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

st.set_page_config(page_title="Hack Week DL Lab", page_icon="🧠", layout="wide")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.markdown("""<style>
.hero{padding:1.2rem 1.6rem;border-radius:16px;background:linear-gradient(120deg,#6366f1,#8b5cf6,#ec4899);margin-bottom:1.2rem}
.hero h1{margin:0;color:#fff;font-size:2rem}.hero p{margin:.3rem 0 0;color:#fff;opacity:.9}
</style>""", unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# ---------------- shared helpers ----------------
def run_epoch(model, dl, opt, bn_eval=False):
    model.train()
    if bn_eval:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    tot = n = 0
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / n


@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    tot = c = n = 0
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        o = model(x)
        tot += F.cross_entropy(o, y, reduction="sum").item()
        c += (o.argmax(1) == y).sum().item()
        n += len(y)
    return tot / n, c / n


@st.cache_resource(show_spinner="Loading FashionMNIST (first time downloads)...")
def fmnist():
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    return (datasets.FashionMNIST("data", train=True, download=True, transform=t),
            datasets.FashionMNIST("data", train=False, download=True, transform=t))


def make_cnn(reg=False):
    bn = (lambda c: nn.BatchNorm2d(c)) if reg else (lambda c: nn.Identity())
    dr = nn.Dropout(0.5) if reg else nn.Identity()
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), bn(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), bn(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), dr, nn.Linear(128, 10)).to(DEV)


# ---------------- pages ----------------
def page_home():
    hero("🧠 Deep Learning Lab", "Hack Week: 7 hands-on steps, all running live in your browser")
    c1, c2, c3 = st.columns(3)
    c1.metric("Device", "GPU" if DEV.type == "cuda" else "CPU")
    c2.metric("PyTorch", torch.__version__.split("+")[0])
    c3.metric("Steps", "7")
    st.markdown("""
**Pick a step from the sidebar:**
1. Tensors, autograd & training loop
2. CNN image classification (conv + pooling)
3. RNN / LSTM / GRU sequence modeling
4. Regularization: dropout, batch norm, early stopping
5. Transfer learning with pretrained ResNet18
6. Hugging Face Transformers (sentiment, NER, fill-mask)
7. Open LLM: generation, summarization, Q&A (no API key)
""")
    st.info("Steps 2, 4, 5, 6, 7 download datasets/models the first time. Keep internet on.")


def page_basics():
    hero("1️⃣ Tensors, Autograd & Training Loop", "The building blocks of PyTorch")
    st.subheader("Tensors")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.randn(2, 2)
    c = st.columns(4)
    for col, (name, t) in zip(c, [("a", a), ("b = randn", b), ("a + b", a + b), ("a @ b", a @ b)]):
        col.caption(name)
        col.dataframe(pd.DataFrame(t.numpy()), hide_index=True)

    st.subheader("Autograd: y = x² + 2x, dy/dx = 2x + 2")
    xv = st.slider("x", -5.0, 5.0, 3.0, 0.5)
    x = torch.tensor(xv, requires_grad=True)
    y = x ** 2 + 2 * x
    y.backward()
    m = st.columns(3)
    m[0].metric("y", f"{y.item():.2f}")
    m[1].metric("autograd dy/dx", f"{x.grad.item():.2f}")
    m[2].metric("formula 2x+2", f"{2 * xv + 2:.2f}")

    st.subheader("Training loop: learn y = 3x + 2")
    c1, c2 = st.columns(2)
    lr = c1.select_slider("Learning rate", [0.001, 0.01, 0.05, 0.1, 0.5], 0.1)
    ep = c2.slider("Epochs", 10, 300, 100, 10)
    if st.button("▶ Train", type="primary"):
        torch.manual_seed(0)
        X = torch.linspace(-1, 1, 100).unsqueeze(1)
        Y = 3 * X + 2 + 0.1 * torch.randn_like(X)
        model = nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        for _ in range(ep):
            opt.zero_grad()                     # 1 clear grads
            loss = F.mse_loss(model(X), Y)      # 2 forward + loss
            loss.backward()                     # 3 backprop
            opt.step()                          # 4 update
            losses.append(loss.item())
        st.line_chart(pd.DataFrame({"loss": losses}))
        m = st.columns(3)
        m[0].metric("learned w (true 3)", f"{model.weight.item():.2f}")
        m[1].metric("learned b (true 2)", f"{model.bias.item():.2f}")
        m[2].metric("final loss", f"{losses[-1]:.4f}")


def page_cnn():
    hero("2️⃣ CNN Image Classification", "Conv → ReLU → MaxPool ×2 → Flatten → Linear → Linear on FashionMNIST")
    tr, te = fmnist()
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Training images", 1000, 20000, 6000, 1000)
    ep = c2.slider("Epochs", 1, 8, 3)
    lr = c3.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], 1e-3)
    if st.button("🚀 Train CNN", type="primary"):
        dl = DataLoader(Subset(tr, range(n)), batch_size=128, shuffle=True)
        tdl = DataLoader(Subset(te, range(2000)), batch_size=500)
        model = make_cnn()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bar = st.progress(0.0)
        col1, col2 = st.columns(2)
        ch1, ch2 = col1.empty(), col2.empty()
        loss_h, acc_h = [], []
        for e in range(ep):
            loss_h.append(run_epoch(model, dl, opt))
            acc_h.append(evaluate(model, tdl)[1])
            ch1.line_chart(pd.DataFrame({"train loss": loss_h}))
            ch2.line_chart(pd.DataFrame({"test accuracy": acc_h}))
            bar.progress((e + 1) / ep)
        st.metric("Final test accuracy", f"{acc_h[-1]:.1%}")
        x = torch.stack([te[i][0] for i in range(10)]).to(DEV)
        with torch.no_grad():
            pred = model(x).argmax(1).cpu()
        imgs = [(x[i, 0].cpu().numpy() * 0.353 + 0.286).clip(0, 1) for i in range(10)]
        caps = []
        for i in range(10):
            true = te[i][1]
            caps.append(f"✅ {te.classes[pred[i]]}" if pred[i] == true else f"❌ {te.classes[pred[i]]} (is {te.classes[true]})")
        st.image(imgs, caption=caps, width=110)


class Seq(nn.Module):
    def __init__(self, cell, hidden):
        super().__init__()
        self.rnn = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell](1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


def page_rnn():
    hero("3️⃣ RNN / LSTM / GRU", "Predict the next value of a noisy sine wave")
    c1, c2, c3 = st.columns(3)
    cells = c1.multiselect("Models", ["RNN", "LSTM", "GRU"], ["RNN", "LSTM", "GRU"])
    ep = c2.slider("Epochs", 5, 40, 15)
    win = c3.slider("Window length", 10, 80, 30, 10)
    if st.button("▶ Train & compare", type="primary") and cells:
        torch.manual_seed(0)
        t = torch.arange(0, 100, 0.1)
        s = torch.sin(t) + 0.05 * torch.randn_like(t)
        X = torch.stack([s[i:i + win] for i in range(len(s) - win)]).unsqueeze(-1)
        Y = s[win:].unsqueeze(-1)
        sp = int(0.8 * len(X))
        Xtr, Ytr, Xte, Yte = X[:sp].to(DEV), Y[:sp].to(DEV), X[sp:].to(DEV), Y[sp:].to(DEV)
        mse, preds = {}, {"Actual": Yte.cpu().squeeze().numpy()}
        bar = st.progress(0.0)
        for k, cell in enumerate(cells):
            model = Seq(cell, 32).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(ep):
                model.train()
                perm = torch.randperm(len(Xtr), device=DEV)
                for i in range(0, len(Xtr), 64):
                    idx = perm[i:i + 64]
                    opt.zero_grad()
                    F.mse_loss(model(Xtr[idx]), Ytr[idx]).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()
            with torch.no_grad():
                p = model(Xte)
            mse[cell] = F.mse_loss(p, Yte).item()
            preds[cell] = p.cpu().squeeze().numpy()
            bar.progress((k + 1) / len(cells))
        c1, c2 = st.columns(2)
        c1.subheader("Test MSE (lower is better)")
        c1.bar_chart(pd.Series(mse))
        c2.subheader("Predictions vs actual")
        c2.line_chart(pd.DataFrame(preds))


def page_reg():
    hero("4️⃣ Regularization", "Dropout + BatchNorm + weight decay + early stopping vs a plain CNN")
    tr, te = fmnist()
    c1, c2, c3 = st.columns(3)
    max_ep = c1.slider("Max epochs", 5, 30, 20)
    pat = c2.slider("Early-stop patience", 1, 8, 3)
    wd = c3.select_slider("Weight decay (L2)", [0.0, 1e-4, 1e-3], 1e-4)
    st.caption("Small 5000-image training set on purpose, so the baseline overfits (val loss goes back up).")
    if st.button("▶ Compare", type="primary"):
        trn = DataLoader(Subset(tr, range(5000)), batch_size=64, shuffle=True)
        val = DataLoader(Subset(tr, range(5000, 6000)), batch_size=500)
        tst = DataLoader(Subset(te, range(3000)), batch_size=500)
        cols = st.columns(2)
        with st.spinner("Training both models..."):
            for col, (name, reg) in zip(cols, [("Baseline (no regularization)", False), ("Regularized + early stopping", True)]):
                model = make_cnn(reg)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd if reg else 0.0)
                hist, best, bad, best_state, stopped = [], float("inf"), 0, None, None
                for e in range(max_ep):
                    tl = run_epoch(model, trn, opt)
                    vl, _ = evaluate(model, val)
                    hist.append({"train loss": tl, "val loss": vl})
                    if reg:
                        if vl < best - 1e-4:
                            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
                        else:
                            bad += 1
                        if bad >= pat:
                            stopped = e + 1
                            break
                if reg and best_state:
                    model.load_state_dict(best_state)
                acc = evaluate(model, tst)[1]
                col.subheader(name)
                col.line_chart(pd.DataFrame(hist))
                col.metric("Test accuracy", f"{acc:.1%}")
                if stopped:
                    col.success(f"Early stopped at epoch {stopped}; best weights restored.")


@st.cache_resource(show_spinner="Loading ResNet18...")
def resnet_pre():
    from torchvision import models
    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().to(DEV)


def page_transfer():
    hero("5️⃣ Transfer Learning", "Pretrained ResNet18 (ImageNet) reused for new tasks")
    tab1, tab2 = st.tabs(["🔍 Classify your image", "🎓 Fine-tune on CIFAR-10"])
    from torchvision import models
    w = models.ResNet18_Weights.DEFAULT
    with tab1:
        up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        if up:
            from PIL import Image
            img = Image.open(up).convert("RGB")
            net = resnet_pre()
            with torch.no_grad():
                p = F.softmax(net(w.transforms()(img).unsqueeze(0).to(DEV)), 1)[0].cpu()
            v, i = p.topk(5)
            c1, c2 = st.columns([1, 2])
            c1.image(img, use_container_width=True)
            c2.bar_chart(pd.DataFrame({"probability": v.numpy()}, index=[w.meta["categories"][j] for j in i]))
    with tab2:
        c1, c2 = st.columns(2)
        n = c1.slider("Training images", 500, 4000, 1500, 500)
        ep = c2.slider("Epochs", 1, 4, 2)
        st.caption("Backbone frozen; only the new final layer (fc) trains.")
        if st.button("🎓 Train head", type="primary"):
            from torchvision import datasets, transforms
            t = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                                    transforms.Normalize(w.transforms().mean, w.transforms().std)])
            with st.spinner("Loading CIFAR-10 (first time downloads ~170MB)..."):
                tr = datasets.CIFAR10("data", train=True, download=True, transform=t)
                te = datasets.CIFAR10("data", train=False, download=True, transform=t)
            dl = DataLoader(Subset(tr, range(n)), batch_size=64, shuffle=True)
            tdl = DataLoader(Subset(te, range(500)), batch_size=100)
            model = models.resnet18(weights=w)
            for p in model.parameters():
                p.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, 10)
            model = model.to(DEV)
            opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
            accs, bar = [], st.progress(0.0)
            chart = st.empty()
            for e in range(ep):
                run_epoch(model, dl, opt, bn_eval=True)
                accs.append(evaluate(model, tdl)[1])
                chart.line_chart(pd.DataFrame({"test accuracy": accs}))
                bar.progress((e + 1) / ep)
            st.metric("Test accuracy", f"{accs[-1]:.1%}")


@st.cache_resource(show_spinner="Loading model (first time downloads)...")
def hf(task, model, **kw):
    from transformers import pipeline
    return pipeline(task, model=model, device=0 if torch.cuda.is_available() else -1, **kw)


def page_hf():
    hero("6️⃣ Hugging Face Transformers", "Pretrained transformer pipelines in one line")
    t1, t2, t3 = st.tabs(["😊 Sentiment", "🏷️ Named entities", "🧩 Fill the mask"])
    with t1:
        txt = st.text_area("Text", "I loved this hackathon, the team was amazing!", key="s")
        if st.button("Analyse", key="bs"):
            r = hf("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")(txt)[0]
            st.metric(r["label"], f"{r['score']:.1%} confident")
    with t2:
        txt = st.text_area("Text", "Hugging Face is based in New York and Paris.", key="n")
        if st.button("Find entities", key="bn"):
            r = hf("ner", "dslim/bert-base-NER", aggregation_strategy="simple")(txt)
            st.dataframe(pd.DataFrame(r)[["entity_group", "word", "score"]], use_container_width=True)
    with t3:
        txt = st.text_input("Sentence with <mask>", "Deep learning is <mask> than classical ML for images.")
        if st.button("Predict word", key="bm"):
            r = hf("fill-mask", "distilroberta-base")(txt)
            st.dataframe(pd.DataFrame(r)[["token_str", "score"]], use_container_width=True)


def page_llm():
    hero("7️⃣ Open LLM (no API key)", "Qwen2.5 instruct model running locally via Hugging Face")
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"])
    task = c2.radio("Task", ["Text generation", "Summarization", "Question answering"], horizontal=True)
    maxt = st.slider("Max new tokens", 50, 400, 150, 10)
    if task == "Text generation":
        prompt = st.text_area("Prompt", "Write a short, upbeat 3-line poem about debugging neural networks.")
    elif task == "Summarization":
        text = st.text_area("Text to summarize", "Transformers process sequences with self-attention instead of recurrence, which lets them be trained in parallel and capture long-range dependencies. Since 2017 they have become the basis of most large language models.", height=150)
        prompt = f"Summarize in 2 sentences:\n\n{text}"
    else:
        ctx = st.text_area("Context", "Transformers use self-attention instead of recurrence. They were introduced in the 2017 paper 'Attention Is All You Need'.", height=120)
        q = st.text_input("Question", "What replaced recurrence in transformers?")
        prompt = f"Answer using only the context.\n\nContext:\n{ctx}\n\nQuestion: {q}"
    if st.button("✨ Generate", type="primary"):
        gen = hf("text-generation", model)
        with st.spinner("Thinking..."):
            out = gen([{"role": "user", "content": prompt}], max_new_tokens=maxt, do_sample=False)
        st.success(out[0]["generated_text"][-1]["content"])


PAGES = {"🏠 Home": page_home, "1️⃣ Tensors & Autograd": page_basics, "2️⃣ CNN": page_cnn,
         "3️⃣ RNN / LSTM / GRU": page_rnn, "4️⃣ Regularization": page_reg,
         "5️⃣ Transfer Learning": page_transfer, "6️⃣ HF Transformers": page_hf, "7️⃣ Open LLM": page_llm}

st.sidebar.title("🧠 DL Lab")
choice = st.sidebar.radio("Steps", list(PAGES))
st.sidebar.caption(f"Running on: {DEV.type.upper()}")
PAGES[choice]()
