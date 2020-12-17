from abc import abstractmethod
from bokeh.models import Button, Select
from bokeh.plotting import figure as bokeh_figure


NOT_IMPLEMENTED_MSG = 'TODO: Implement this.'


class Mediator:

    @abstractmethod
    def notify(self, component, event_name, *args, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


class ComponentMixin:

    def __init__(self, *args, **kwargs):
        self.mediator = None

    def set_mediator(self, mediator: Mediator):
        self.mediator = mediator

    def make_attr_old_new_callback(self, event_name):
        def wrapped_notify(attr, old, new):
            return self.mediator.notify(self, event_name,
                                        attr=attr,
                                        old=old,
                                        new=new,
                                        )

        return wrapped_notify

    def make_event_callback(self, event_name):
        def wrapped_notify(event):
            return self.mediator.notify(self, event_name,
                                        event,
                                        )

        return wrapped_notify


class Plottable:

    @abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


class DataSourceComponent(ComponentMixin):

    def __init__(self, data_source):
        super().__init__()
        self.data_source = data_source
        self.selected_indices_callback = None

    def set_mediator(self, mediator):
        super().set_mediator(mediator)
        event_name = 'selected-indices'

        wrapped_notify = self.make_attr_old_new_callback(event_name)

        self.selected_indices_callback = wrapped_notify

        self.data_source.selected.on_change(
            'indices',
            self.selected_indices_callback,
        )


class ButtonComponent(ComponentMixin):
    def __init__(self, button_kwargs):
        super().__init__()
        self.button = Button(**button_kwargs)
        self.layout = self.button

    def set_mediator(self, mediator):
        super().set_mediator(mediator)
        event_name = 'button-click'
        on_click = self.make_event_callback(event_name)
        self.button.on_click(
            on_click,
        )


class SelectComponent(ComponentMixin):
    def __init__(self, select_kwargs):
        super().__init__()
        self.select = Select(**select_kwargs)
        self.layout = self.select
        self.selected_value_callback = None

    def set_mediator(self, mediator):
        super().set_mediator(mediator)
        event_name = 'dropdown-select'
        dropdown_select = self.make_attr_old_new_callback(event_name)
        self.selected_value_callback = dropdown_select
        self.select.on_change('value', dropdown_select)


class TextPlotComponent(ComponentMixin, Plottable):
    def __init__(self):
        super().__init__()
        self.layout = None
        self.text = None

    def plot(self, figure=None, figure_kwargs=None, text_kwargs=None):
        if figure_kwargs is None:
            figure_kwargs = dict()
        if text_kwargs is None:
            text_kwargs = dict()

        if figure is None:
            figure = bokeh_figure(**figure_kwargs)
        self.layout = figure

        self.text = self.layout.text(
            **text_kwargs,
        )

    def add_text(self, x, y, text_color, text):
        """Adds text to plot

        x : list
            x coordinate to add text at
        y : list
            y coordinate to add text at
        text_color : list
            colors to add text in
        text : list
            text to add at coordinates
        """
        new_data = dict()
        text_data = self.text.data_source
        new_data['x'] = text_data.data['x'] + x
        new_data['y'] = text_data.data['y'] + y
        new_data['text_color'] = text_data.data['text_color'] + text_color
        new_data['text'] = text_data.data['text'] + text
        text_data.data = new_data


class ScatterComponent(ComponentMixin, Plottable):
    def __init__(self, data_source=None):
        super().__init__()
        self.data_source = data_source
        self.layout = None
        self.scatter = None

    def plot(self, figure=None, figure_kwargs=None, scatter_kwargs=None):
        if figure_kwargs is None:
            figure_kwargs = dict()
        if scatter_kwargs is None:
            scatter_kwargs = dict()

        if figure is None:
            figure = bokeh_figure(**figure_kwargs)
        self.layout = figure

        source = dict()
        if self.data_source is not None:
            source = {'source': self.data_source}
        self.scatter = self.layout.circle(**source,
                                          **scatter_kwargs,
                                          )


class SegmentComponent(ComponentMixin, Plottable):
    def __init__(self, data_source=None):
        super().__init__()
        self.data_source = data_source
        self.layout = None
        self.segment = None

    def plot(self, figure=None, figure_kwargs=None, segment_kwargs=None):
        if figure_kwargs is None:
            figure_kwargs = dict()
        if segment_kwargs is None:
            segment_kwargs = dict()

        if figure is None:
            figure = bokeh_figure(**figure_kwargs)
        self.layout = figure

        self.segment = self.layout.segment(source=self.data_source,
                                           **segment_kwargs,
                                           )
