from functools import partialmethod
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import sqlite3
import click
import json
import pkg_resources
from itertools import combinations

from q2_mlab.db.schema import RegressionScore
from q2_mlab.plotting.components import (
    Mediator,
    ComponentMixin,
    Plottable,
    ButtonComponent,
    ScatterComponent,
    SegmentComponent,
    DataSourceComponent,
    SelectComponent,
)
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import (
    ColumnDataSource,
    CheckboxButtonGroup,
    TextInput,
    Legend,
    LegendItem,
)
from bokeh.models.widgets import (
    Div,
)
from bokeh.palettes import (
    Category20,
    Set3,
)
from bokeh.layouts import column, row
from bokeh.server.server import Server

groups = ['parameters_id', 'dataset', 'target', 'level', 'algorithm']
drop_cols = ['artifact_uuid', 'datetime', 'CV_IDX', 'id']

target_map = {
    'age_v2': 'age',
    'BL_AGE': 'age',
    'age': 'age',
    'bmi_v2': 'bmi',
    'BMI': 'bmi',
    'bmi': 'bmi'
}

with pkg_resources.resource_stream(
    __name__, "standard_deviations.json"
) as f:
    TARGET_SD = json.load(f)


def _get_standardized_mae(df_row, norm_dict):
    """
    """
    mae = df_row['MAE']
    target = df_row['target']
    dataset = df_row['dataset']
    cv_fold = df_row['CV_IDX']
    level = df_row['level']
    key = f"({dataset}, {target}, {level}, {cv_fold})"
    sd = norm_dict.get(key, 1)
    standardized_mae = mae / sd
    return standardized_mae


def process_db_df(df):
    # remap values for consistency
    df['level'] = df['level'].replace('none', 'MG')
    df['target'] = df['target'].map(target_map)
    df['standardized_MAE'] = df.apply(_get_standardized_mae, axis=1,
                                      args=(TARGET_SD,))

    group_stats = df.drop(
        drop_cols, axis=1
    ).groupby(
        groups
    ).agg(
        ['var', 'mean']
    )

    group_stats.columns = agg_columns = ['_'.join(col).strip() for
                                         col in group_stats.columns.values]

    group_stats.reset_index(inplace=True)

    min_by = ['dataset', 'target']
    group_mins = group_stats[agg_columns + min_by].groupby(min_by).min()
    indices = group_stats[['dataset', 'target']].to_records(
        index=False).tolist()
    expanded_group_mins = group_mins.loc[indices]
    expanded_group_mins.index = group_stats.index
    relative_group_stats = (group_stats / expanded_group_mins)[agg_columns]
    relative_group_stats.columns = ['relative_' + col for
                                    col in relative_group_stats]
    group_stats = group_stats.join(relative_group_stats)

    return group_stats


def find_segments(group_stats, across, groupby):
    """
    TODO makes some assumptions about the guarantees on pairs when there are
    more than 2 categories
    """
    seg_cols = groupby.copy()
    seg_cols.remove(across)

    group_counts = group_stats[seg_cols + [across]].groupby(seg_cols).count()

    max_n_pairs = group_counts[across].max()
    category_values = group_stats[across].unique()

    where = (group_counts[across] == max_n_pairs)
    keep_repeats = group_stats.set_index(seg_cols).loc[where]

    keep_repeats_parts = []

    for i, sub_group in enumerate(category_values):
        where = keep_repeats[across] == sub_group
        keep_repeats_parts.append(keep_repeats.loc[where])
        keep_repeats_parts[i].columns = [col + '_' + sub_group for
                                         col in keep_repeats_parts[i].columns]

    segment_df = pd.concat(keep_repeats_parts,
                           axis=1
                           )
    return segment_df


class TextInputComponent(ComponentMixin):
    def __init__(self, text_input_kwargs):
        super().__init__()
        self.text_input = TextInput(**text_input_kwargs)
        self.layout = self.text_input
        self.input_text_callback = None

    def set_mediator(self, mediator):
        super().set_mediator(mediator)
        event_name = 'text-change'
        text_change = self.make_attr_old_new_callback(event_name)
        self.input_text_callback = text_change
        self.text_input.on_change('value', self.input_text_callback)


class CheckboxButtonGroupComponent(ComponentMixin):
    def __init__(self, checkbox_kwargs):
        super().__init__()
        self.checkbox = CheckboxButtonGroup(**checkbox_kwargs)
        self.checkbox_change = None
        self.layout = self.checkbox

    def set_mediator(self, mediator):
        super().set_mediator(mediator)
        event_name = 'checkbox-change'
        self.checkbox_change = self.make_attr_old_new_callback(event_name)
        self.checkbox.on_change('active', self.checkbox_change)


class SegmentComponentExt(SegmentComponent):
    def redraw(self, x, y, seg_0, seg_1, data):
        self.data_source.data = data
        self.segment.glyph.x0 = '_'.join([x, seg_0])
        self.segment.glyph.x1 = '_'.join([x, seg_1])
        self.segment.glyph.y0 = '_'.join([y, seg_0])
        self.segment.glyph.y1 = '_'.join([y, seg_1])


palettes = {
    'Category20': Category20,
    'Set3': Set3,
}


DEFAULTS = {
    'segment_variable': 'dataset',
    'x': 'MAE_mean',
    'y': 'MAE_var',
    'x_axis_type': 'log',
    'y_axis_type': 'log',
    'cmap': 'Category20'
}


class AlgorithmScatter(Mediator, Plottable):

    def __init__(self, x, y, engine, cmap=None):
        super().__init__()
        self.x = x
        self.y = y
        self.engine = engine
        self.data = None
        self.scatter = None
        if cmap is None:
            self.cmap = Category20
        else:
            self.cmap = cmap
        self.line_segment_variable = DEFAULTS['segment_variable']
        self.data_raw = None
        self.data_static = None
        self.data = None
        self.seg_0, self.seg_1 = None, None
        self.scatter_source = None
        self.x_axis_type = DEFAULTS['x_axis_type']
        self.y_axis_type = DEFAULTS['y_axis_type']
        self.axis_types = ['linear', 'log']
        self.line_segment_pairs = {
            'dataset': ['finrisk', 'sol'],
            'level': ['16S', 'MG'],
            'target': ['age', 'bmi'],
        }
        self.scatter_tools = 'pan,wheel_zoom,box_select,lasso_select,'\
                             'reset,box_zoom,save'
        self.segment = None
        self.segment_source = None
        self.segment_button = None
        self.segment_variable_select = None
        self.x_var_select = None
        self.y_var_select = None
        self.dataset_bars = None
        self.dataset_bars_source = None
        self.dataset_bars_figure = None
        self.level_bars = None
        self.level_bars_source = None
        self.level_bars_figure = None
        self.target_bars = None
        self.target_bars_source = None
        self.target_bars_figure = None
        self.query_button = None
        self.query_input = None
        self.query_row = None
        self.layout = None

    def notify(self,
               component,
               event_name,
               *args, **kwargs,
               ):
        if (event_name == 'dropdown-select') and \
                (component is self.x_var_select):
            self.x = component.select.value
            self.scatter.scatter.glyph.x = self.x
            self.scatter.layout.xaxis.axis_label = self.x
            self.segment.segment.glyph.x0 = '_'.join([self.x, self.seg_0])
            self.segment.segment.glyph.x1 = '_'.join([self.x, self.seg_1])
        if (event_name == 'dropdown-select') and \
                (component is self.y_var_select):
            self.y = component.select.value
            self.scatter.scatter.glyph.y = self.y
            self.scatter.layout.yaxis.axis_label = self.y
            self.segment.segment.glyph.y0 = '_'.join([self.y, self.seg_0])
            self.segment.segment.glyph.y1 = '_'.join([self.y, self.seg_1])
        if (event_name == 'selected-indices') and \
                (component is self.scatter_source):
            selected_indices = self.scatter_source.data_source.selected.indices
            self.dataset_bars_source.data = self.get_dataset_counts(
                indices=selected_indices,
            )
            self.level_bars_source.data = self.get_level_counts(
                indices=selected_indices,
            )
            self.target_bars_source.data = self.get_target_counts(
                indices=selected_indices,
            )
        if (event_name == 'button-click') and \
                (component is self.query_button):
            df = self.handle_query(self.query_input.text_input.value)
            # need to update self.data due to how the hbars are currently
            # written
            self.data = df
            self.scatter_source.data_source.data = df.to_dict(orient='list')
            segment_source = find_segments(
                df,
                across=self.line_segment_variable,
                groupby=['parameters_id', 'algorithm', 'level', 'dataset',
                         'target'],
            )
            self.segment.segment.data_source.data = segment_source.to_dict(
                orient='list',
            )

            selected_indices = self.scatter_source.data_source.selected.indices
            self.dataset_bars_source.data = self.get_dataset_counts(
                indices=selected_indices,
            )
            self.level_bars_source.data = self.get_level_counts(
                indices=selected_indices,
            )
            self.target_bars_source.data = self.get_target_counts(
                indices=selected_indices,
            )
        if (event_name == 'checkbox-change') and \
                (component is self.segment_button):
            active = self.segment_button.checkbox.active
            if 0 in active:
                self.segment.segment.visible = True
            else:
                self.segment.segment.visible = False
        if (event_name == 'dropdown-select') and \
                (component is self.segment_variable_select):
            new_segment_variable = self.segment_variable_select.select.value
            self.line_segment_variable = new_segment_variable
            new_segment_data = find_segments(
                self.data,
                across=self.line_segment_variable,
                groupby=['parameters_id', 'algorithm', 'level', 'dataset',
                         'target']
            )
            line_segment_ends = self.line_segment_pairs[new_segment_variable]
            self.segment.redraw(
                self.x,
                self.y,
                *line_segment_ends,
                new_segment_data
            )

    def plot(self):
        self.data_raw = pd.read_sql_table(RegressionScore.__tablename__,
                                          con=self.engine,
                                          )
        # TODO this is temporary
        self.data_raw = self.data_raw.loc[
            self.data_raw['algorithm'] != 'MLPRegressor'
            ]
        self.data = df = process_db_df(self.data_raw)
        self.data_static = df
        self.seg_0, self.seg_1 = self.line_segment_pairs[
            self.line_segment_variable
        ]

        # ## Data Setup
        scatter_source = ColumnDataSource(df)
        self.scatter_source = DataSourceComponent(scatter_source)
        self.scatter_source.set_mediator(self)

        # ## General Setup
        algorithms = sorted(df['algorithm'].unique())
        levels = sorted(df['level'].unique())
        datasets = sorted(df['dataset'].unique())
        targets = sorted(df['target'].unique())
        plot_width = 600
        self.line_segment_pairs = {
            'level': ['16S', 'MG'],
            'target': ['age', 'bmi'],
        }
        dataset_combinations = combinations(["finrisk", "imsms", "sol"], r=2)
        for dataset_pair in dataset_combinations:
            d1, d2 = dataset_pair
            self.line_segment_pairs[f"{d1}-to-{d2}"] = [d1, d2]

        categorical_variables = ['parameters_id', 'target', 'algorithm',
                                 'level', 'dataset']
        plottable_variables = list(sorted(
            df.columns.drop(categorical_variables)
        ))

        color_scheme = self.cmap[len(algorithms)]
        algorithm_cmap = factor_cmap('algorithm', palette=color_scheme,
                                     factors=algorithms,
                                     )
        figure_kwargs = dict(x_axis_type=self.x_axis_type,
                             y_axis_type=self.y_axis_type,
                             plot_height=400,
                             tools=self.scatter_tools,
                             output_backend='webgl',
                             )

        # ## Segment Plot
        segment_source = ColumnDataSource(
            find_segments(self.data, across=self.line_segment_variable,
                          groupby=['parameters_id', 'algorithm', 'level',
                                   'dataset']
                          )
        )
        self.segment_source = DataSourceComponent(scatter_source)

        self.segment = SegmentComponentExt(data_source=segment_source)
        segment_kwargs = {
            'x0': self.x + '_' + self.seg_0,
            'x1': self.x + '_' + self.seg_1,
            'y0': self.y + '_' + self.seg_0,
            'y1': self.y + '_' + self.seg_1,
            'line_width': 0.1,
            'line_color': '#A9A9A9',
        }
        self.segment.plot(
            figure_kwargs=figure_kwargs,
            segment_kwargs=segment_kwargs,
        )

        # ## Segment Visible button
        self.segment_button = CheckboxButtonGroupComponent(
             checkbox_kwargs=dict(
                 labels=['Segments'],
                 active=[0],
             )
        )
        self.segment_button.set_mediator(self)
        self.segment_variable_select = SelectComponent(
            select_kwargs=dict(
                value=self.line_segment_variable,
                title='Segment Variable',
                options=list(self.line_segment_pairs.keys()),
            )
        )
        self.segment_variable_select.set_mediator(self)

        # ## Scatter plot
        self.scatter = ScatterComponent()

        scatter_kwargs = dict(x=self.x, y=self.y, source=scatter_source,
                              # legend_field='algorithm',
                              fill_color=algorithm_cmap,
                              name='scatter',
                              )

        self.scatter.plot(
            figure=self.segment.layout,
            scatter_kwargs=scatter_kwargs,
        )

        scatter = self.scatter.layout

        scatter.toolbar.logo = None
        scatter.xaxis.axis_label = self.x
        scatter.yaxis.axis_label = self.y
        self.scatter.scatter.glyph.line_color = 'white'
        self.scatter.scatter.glyph.line_width = 0.1
        self.scatter.scatter.nonselection_glyph.line_color = 'white'

        transform = algorithm_cmap['transform']
        legend_fig = figure(outline_line_alpha=0, toolbar_location=None)
        legend_items = []
        for i, (alg, color) in enumerate(zip(transform.factors,
                                             transform.palette)):
            legend_fig.circle(fill_color=color, name=f'circ{i}',
                              line_color='white',
                              )
            renderers = legend_fig.select(name=f'circ{i}')
            legend_item = LegendItem(
                label=alg,
                renderers=renderers,
            )
            legend_items.append(legend_item)
        legend = Legend(
            items=legend_items,
            location='top_left',
        )
        legend_fig.add_layout(legend)

        scatter.plot_width = plot_width
        scatter.plot_height = 500

        # ## Variable Selection
        self.x_var_select = SelectComponent(
            select_kwargs=dict(
                value=self.x,
                title='X variable',
                options=plottable_variables
            )
        )
        self.x_var_select.set_mediator(self)
        x_select = self.x_var_select.select

        self.y_var_select = SelectComponent(
            select_kwargs=dict(
                value=self.y,
                title='Y variable',
                options=plottable_variables
            )
        )
        self.y_var_select.set_mediator(self)
        y_select = self.y_var_select.select

        # ## Dataset Stacked Hbars
        data_getter = self.get_dataset_counts
        self.dataset_bars_source = ColumnDataSource(data_getter())
        self.dataset_bars_figure = figure(y_range=datasets, plot_height=100)
        self.dataset_bars = self.dataset_bars_figure.hbar_stack(
            algorithms, y='dataset',
            height=0.9,
            color=color_scheme,
            source=self.dataset_bars_source,
            )
        self.dataset_bars_figure.toolbar_location = None
        self.dataset_bars_figure.plot_width = plot_width

        # ## Level Stacked Hbars
        data_getter = self.get_level_counts
        self.level_bars_source = ColumnDataSource(data_getter())
        self.level_bars_figure = figure(y_range=levels, plot_height=100)
        self.level_bars = self.level_bars_figure.hbar_stack(
            algorithms, y='level',
            height=0.9,
            color=color_scheme,
            source=self.level_bars_source,
            )
        self.level_bars_figure.toolbar_location = None
        self.level_bars_figure.plot_width = plot_width

        # ## Target Stacked Hbars
        data_getter = self.get_target_counts
        self.target_bars_source = ColumnDataSource(data_getter())
        self.target_bars_figure = figure(y_range=targets, plot_height=100)
        self.target_bars = self.target_bars_figure.hbar_stack(
            algorithms, y='target',
            height=0.9,
            color=color_scheme,
            source=self.target_bars_source,
            )
        self.target_bars_figure.toolbar_location = None
        self.target_bars_figure.plot_width = plot_width

        # ## Text input
        button_width = 100
        self.query_input = TextInputComponent(
            text_input_kwargs=dict(
                title='Enter query',
                width=plot_width - button_width
            )
        )
        self.query_button = ButtonComponent(
             button_kwargs=dict(
                 label='Execute',
                 width=button_width,
             )
        )
        self.query_button.set_mediator(self)

        self.query_row = row(self.query_input.layout,
                             column(
                                 Div(text="", height=8),
                                 self.query_button.layout,
                             ))

        # ## Layout
        variable_selection = row(x_select, y_select,
                                 )
        segment_selection = row(
            self.segment_variable_select.layout,
            column(
                Div(text="", height=8),
                self.segment_button.layout,
            )
        )

        self.layout = row(
            column(
                self.query_row,
                variable_selection,
                segment_selection,
                row(
                    scatter,
                    column(
                        self.dataset_bars_figure,
                        self.level_bars_figure,
                        self.target_bars_figure,
                        legend_fig,
                    ),
                ),
            ),
        )

        return self

    def handle_query(self, text):
        if text != '':
            df = self.data_static.query(text).reset_index(drop=True)
        else:
            df = self.data_static
        return df

    def get_counts_by(self, category, by, indices=None):
        # TODO consider switching orientation of counts and by
        data = self.subset_selected(indices)
        counts = pd.crosstab(data[by], data[category])
        # algorithms = list(counts.index.values)
        counts_dict = counts.to_dict(orient='list')

        levels = sorted(self.data[by].unique())

        counts_dict[by] = list(filter(lambda x: x in counts.index, levels))
        return counts_dict

    def subset_selected(self, indices):
        # should handle None and empty list
        if not indices:
            # might want to grab data from the scatter plot instead
            data = self.data
        else:
            data = self.data.reindex(indices)
        return data

    get_level_counts = partialmethod(get_counts_by, 'algorithm', 'level')
    get_dataset_counts = partialmethod(get_counts_by, 'algorithm', 'dataset')
    get_target_counts = partialmethod(get_counts_by, 'algorithm', 'target')

    def app(self, doc):
        doc.add_root(self.layout)


@click.group('mlab-plotting')
def mlab_plotting():
    pass


@mlab_plotting.command()
@click.option(
    '--db',
    help='Path to SQLite database file.',
    type=click.Path(exists=False),
)
@click.option(
    '--color-scheme',
    help='Color scheme to plot with',
    type=click.Choice(
        list(palettes.keys()),
    ),
    default=DEFAULTS['cmap'],
)
def metric_scatter(db, color_scheme):
    run_app(db, color_scheme)


def run_app(db, color_scheme):
    # thanks https://github.com/sqlalchemy/sqlalchemy/issues/4863
    def connect():
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    engine = create_engine("sqlite://", creator=connect)
    bkapp = AlgorithmScatter(
        DEFAULTS['x'], DEFAULTS['y'],
        engine=engine,
        cmap=palettes.get(color_scheme),
    ).plot().app
    server = Server({'/': bkapp})
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
