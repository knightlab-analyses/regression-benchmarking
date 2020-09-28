import json


def trivial_remapper(parameter, value):
    return {parameter: value}


def string_or_number_remapper(parameter, value):
    if isinstance(value, str):
        return {parameter + '_STRING': value}
    elif isinstance(value, (int, float)):
        return {parameter + '_NUMBER': value}
    else:
        return dict()


def serialize_remapper(parameter, value):
    return {parameter: json.dumps(value)}


mappers = {
    'gamma': string_or_number_remapper,
    'hidden_layer_sizes': serialize_remapper,
    'learning_rate': string_or_number_remapper,
    'max_features': string_or_number_remapper,
}


def remap_parameters(parameters: dict, ):
    mapped = dict()
    for parameter, value in parameters.items():
        remap = mappers.get(parameter, trivial_remapper)
        mapped.update(remap(parameter, value))

    return mapped
