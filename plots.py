import ibp

if __name__ == "__main__":
    df = ibp.read_graph_data("tmp/CartPole_imag0_log_eval.csv")
    fig_score = ibp.plot_graph(df, "CartPole", "Score")
    fig_score.write_image("tmp/score.png")
    fig_avg_score = ibp.plot_graph(df, "CartPole", "AvgScore")
    fig_avg_score.write_image("tmp/avg_score.png")
