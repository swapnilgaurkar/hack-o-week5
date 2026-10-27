def status_for(r):
    gap=r["gap"]
    if r["train_accuracy"]<.70 and r["test_accuracy"]<.70: return "Possible Underfitting","Both training and testing performance are relatively low."
    if gap>=.10: return "Possible Overfitting","Training accuracy is noticeably higher than testing accuracy."
    return "Good Fit","Training and testing performance are reasonably close."

def analyze_results(results):
    return {k:{"status":status_for(v)[0],"explanation":status_for(v)[1]} for k,v in results.items()}

def bias_variance_payload(results):
    analysis=analyze_results(results)
    return {"models":[{"model":k,"train":v["train_accuracy"],"test":v["test_accuracy"],"gap":v["gap"],
                       "status":analysis[k]["status"],"explanation":analysis[k]["explanation"]} for k,v in results.items()],
            "note":"Empirical heuristic based on training/testing scores and generalization gap; not an exact mathematical measurement of bias and variance."}
