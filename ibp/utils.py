import plotly
import plotly.express
import plotly.graph_objects
import polars

def read_graph_data(
        filename: str
) -> polars.DataFrame:
    df = polars.scan_csv(
        filename, has_header=True, separator=","
    )
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
        markers=True
    )
    return fig