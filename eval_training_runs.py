from trainer import plot_all_single_training_run_evaluations, filter_model_history_by_training_run
import matplotlib.pyplot as plt
import os
import json

#get paths of all json in the models folder
json_paths = []
for root, dirs, files in os.walk("models"):
    for file in files:
        if file.endswith(".json"):
            json_paths.append(os.path.join(root, file))

#create a dictionary of all the jsons
jsons = {}
for path in json_paths:
    with open(path, "r") as f:
        jsons[path] = json.load(f)

#plot all the jsons
for path, history_dict in jsons.items():
    #set title remove everything after the first . and before the last /
    title_infos = path.split("\\")[-1].split(".")[0].split("_")
    lr = history_dict["lr"]
    batch_size = history_dict["batch_size"]
    title = f"Training Date: {title_infos[0]} Time:{title_infos[1]} Model Name: {title_infos[2]} Batch Size: {batch_size} Learning Rate: {lr}"
    history = filter_model_history_by_training_run(history_dict, 0)
    plot_all_single_training_run_evaluations(history, title=title)
    plt.savefig(f"model_evaluations/{title_infos[0]}_{title_infos[1]}_{title_infos[2]}_eval.png")