import ibp

if __name__ == "__main__":
    df = ibp.read_graph_data("log_eval.csv")
    fig = ibp.plot_graph()
