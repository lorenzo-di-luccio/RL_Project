import ibp
import plotly
import plotly.express

if __name__ == "__main__":
    df_train = ibp.read_list_graph_data([
        "logs/LunarLander/LunarLander_imag0_log_train.csv",
        "logs/LunarLander/LunarLander_imag1_log_train.csv",
        "logs/LunarLander/LunarLander_imag2_log_train.csv"
    ])
    df_eval = ibp.read_list_graph_data([
        "logs/LunarLander/LunarLander_imag0_log_eval.csv",
        "logs/LunarLander/LunarLander_imag1_log_eval.csv",
        "logs/LunarLander/LunarLander_imag2_log_eval.csv"
    ])
    fig_train = ibp.plot_list_graphs(
        df_train,
        "LunarLander - Training",
        ["0_AvgScore", "1_AvgScore", "2_AvgScore"]
    )
    fig_train.write_image("tmp/train_graph.png")
    fig_eval = ibp.plot_list_graphs(
        df_eval,
        "LunarLander - Evaluation",
        ["0_AvgScore", "1_AvgScore", "2_AvgScore"]
    )
    fig_eval.write_image("tmp/eval_graph.png")
