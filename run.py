from search import exp_pack


def f(setting):
    return setting['model/layer_slice'] <= setting['model/n_layers']


if __name__ == '__main__':
    config_file = "config-Full-LOO-SKG/full-HistoryAll-Avg-ABM.yaml"
    exp_pack(config_file)
