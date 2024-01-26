import plotly
import plotly.express
import plotly.graph_objects
import polars
from typing import List

def read_graph_data(
        filename: str
) -> polars.DataFrame:
    df = polars.scan_csv(
        filename, has_header=True, separator=","
    )
    return df.collect()

def read_list_graph_data(
        filenames: List[str]
) -> polars.DataFrame:
    dfs = [polars.scan_csv(
            filename, has_header=True, separator=","
        ).rename({
            "Episode": f"{imag_budget}_Episode",
            "Score": f"{imag_budget}_Score",
            "AvgScore": f"{imag_budget}_AvgScore"
        }) for imag_budget, filename in enumerate(filenames)
    ]
    df = polars.concat(dfs, how="horizontal")
    return df.collect()

def plot_graph(
        df: polars.DataFrame,
        title: str,
        stat: str
) -> plotly.graph_objects.Figure:
    fig = plotly.express.line(
        data_frame=df.to_pandas(),
        title=title,
        x="Episode",
        y=stat,
        markers=False
    )
    return fig

def plot_list_graphs(
        df: polars.DataFrame,
        title: str,
        stats: List[str]
) -> plotly.graph_objects.Figure:
    fig = plotly.express.line(
        data_frame=df.to_pandas(),
        title=title,
        x="0_Episode",
        y=stats,
        labels={"0_Episode": "Episode", "value": "AvgScore"},
        markers=False
    )
    new_trace_names = {
        "0_AvgScore": "imag_budget=0",
        "1_AvgScore": "imag_budget=1",
        "2_AvgScore": "imag_budget=2"
    }
    fig.for_each_trace(lambda trace: trace.update(name=new_trace_names[trace.name]))
    fig_params = fig.to_dict()
    fig_params["layout"]["legend"]["title"] = None
    fig_params["layout"]["autosize"] = False
    fig.update_layout(**fig_params["layout"])
    return fig
