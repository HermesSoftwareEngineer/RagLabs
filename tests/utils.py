def act_class_to_dict_schema(act_class):
    """
    Converts a Pydantic class with a Union field and discriminator to a dictionary schema
    suitable for use with language models for structured output.

    Args:
        act_class: The Pydantic class to convert.  It should have an 'action' field
                   that is a Union of other Pydantic classes, and a 'discriminator'
                   defined on the Union.

    Returns:
        A dictionary representing the schema.
    """

    discriminator_field = act_class.__fields__['action'].field_info.discriminator
    union_types = act_class.__fields__['action'].type_.__args__
    
    properties = {
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": [t.__fields__[discriminator_field].default for t in union_types]},
            },
            "required": ["type"]
        }
    }

    # Add properties from each of the Union types
    for union_type in union_types:
        for field_name, field_info in union_type.__fields__.items():
            if field_name != discriminator_field:
                properties["action"]["properties"][field_name] = {
                    "type": field_info.type_.__name__,  # Basic type conversion
                    "description": field_info.description
                }
                properties["action"]["required"] = properties["action"].get("required", []) + [field_name]


    schema = {
        "name": "Act",
        "description": "Escolher qual a melhor forma para responder",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["action"]
        }
    }
    return schema