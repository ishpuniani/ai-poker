import json
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt


def read_from_json(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    # num_keys = len(json_data)
    # data = np.zeros(shape=(num_keys, num_keys))
    agent_names = []
    arr = []
    for agent_name in json_data:
        agent_names.append(agent_name)
        # row = np.array(json_data[agent_name].values())
        # np_data = np.vstack([np_data, row])
        row = list(json_data[agent_name].values())
        arr.append(row)
    np_data = np.array(arr)
    return agent_names, np_data


if __name__ == '__main__':
    agent_names, data = read_from_json('eval.json')
    # df = pd.read_json(r'eval.json')
    sb.set(font_scale=0.90)
    heat_map = sb.heatmap(data, cmap="YlGnBu", annot=True)
    heat_map.set_xticklabels(labels=agent_names, rotation=45, horizontalalignment='right')
    heat_map.set_yticklabels(labels=agent_names, rotation=0, horizontalalignment='right')
    heat_map.set(xlabel="Player 2", ylabel="Player 1")
    # heat_map = sb.heatmap(df, cmap="YlGnBu")
    plt.tight_layout()
    plt.savefig('res.png')
    plt.show()